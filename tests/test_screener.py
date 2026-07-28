"""The 200-week value screen (MB-15 / MB-18).

The score is deliberately SEPARATE from the 0-100 momentum score, and it measures "how dislocated
below its own long-term trend", NOT "how good a buy" — the touch study on this codebase found
below-the-line dips underperformed SPY on a 12/24-month horizon. These tests pin that meaning.
"""
import asyncio

import pytest

from app import screener as sc


def trend(pct=None, z=None, rsi=None, direction=None, below=None, years=10.0):
    t = {"history_years": years}
    if pct is not None:
        t["price_vs_200w_sma_pct"] = pct
        t["below_line"] = below if below is not None else pct < 0
    if z is not None:
        t["drawdown_z"] = z
    if rsi is not None:
        t["rsi_14w"] = rsi
    if direction is not None:
        t["direction"] = direction
    return t


# ---------------------------------------------------------------- scoring

def test_too_little_history_scores_nothing_rather_than_zero():
    # Under ~4 years the 200-week fields are ABSENT, not zero. Scoring it 0 would rank a young
    # company as "not dislocated" when the truth is we cannot say.
    assert sc.value_score(trend(years=2.0)) is None
    assert sc.value_score(None) is None
    assert sc.value_score({}) is None


def test_a_name_above_its_line_earns_no_dislocation_credit():
    # Being "less above" the line is not partial value.
    s = sc.value_score(trend(pct=30.0))
    assert s["components"]["dislocation"] == 0.0
    s2 = sc.value_score(trend(pct=5.0))
    assert s2["components"]["dislocation"] == 0.0


def test_dislocation_scales_with_depth_below_the_line():
    shallow = sc.value_score(trend(pct=-10.0))["components"]["dislocation"]
    deep = sc.value_score(trend(pct=-40.0))["components"]["dislocation"]
    assert 0 < shallow < deep <= 1.0


def test_no_single_component_can_run_away():
    # An unbounded z would let one freak name outrank everything on its own.
    extreme = sc.value_score(trend(pct=-99.0, z=-50.0, rsi=1.0, direction="recovering"))
    assert extreme["value_score"] <= 100.0
    for v in extreme["components"].values():
        assert 0.0 <= v <= 1.0


def test_still_falling_scores_below_turning_up():
    """mungbeans' key nuance: 'deepening' is a knife, not a discount."""
    same = dict(pct=-30.0, z=-1.5, rsi=25.0)
    knife = sc.value_score(trend(**same, direction="deepening"))["value_score"]
    turning = sc.value_score(trend(**same, direction="recovering"))["value_score"]
    assert turning > knife, "a recovering name must outrank an identically-dislocated falling one"


def test_the_components_are_returned_so_a_rank_can_be_interrogated():
    s = sc.value_score(trend(pct=-25.0, z=-1.0, rsi=28.0, direction="recovering"))
    assert set(s["components"]) == {"dislocation", "drawdown_z", "oversold", "recovering"}
    assert s["rsi_14w"] == 28.0 and s["direction"] == "recovering"


# ---------------------------------------------------------------- ranking

def _row(sym, score, pct, below=True):
    return {"symbol": sym, "value_score": score, "price_vs_200w_sma_pct": pct, "below_line": below}


def test_ranking_is_highest_score_first_and_deterministic():
    rows = [_row("B", 40.0, -10.0), _row("A", 40.0, -30.0), _row("C", 70.0, -50.0)]
    out = sc.rank(rows, limit=10, below_line_only=True)
    assert [r["symbol"] for r in out] == ["C", "A", "B"]   # ties break on deeper dislocation


def test_below_line_only_excludes_names_above_their_line():
    rows = [_row("UP", 90.0, 20.0, below=False), _row("DOWN", 10.0, -5.0)]
    assert [r["symbol"] for r in sc.rank(rows, limit=10, below_line_only=True)] == ["DOWN"]
    assert len(sc.rank(rows, limit=10, below_line_only=False)) == 2


def test_limit_is_honoured():
    rows = [_row(f"S{i}", float(i), -float(i)) for i in range(30)]
    assert len(sc.rank(rows, limit=5, below_line_only=True)) == 5


# ---------------------------------------------------------------- orchestration

def test_one_bad_symbol_does_not_sink_the_screen():
    async def trend_of(client, sym):
        if sym == "BOOM":
            raise RuntimeError("upstream died")
        return trend(pct=-20.0, z=-1.0, rsi=25.0, direction="recovering")

    ranked, skipped = asyncio.run(sc.screen(None, trend_of, ["OK1", "BOOM", "OK2"], limit=10))
    assert {r["symbol"] for r in ranked} == {"OK1", "OK2"}
    assert skipped == ["BOOM"]


def test_symbols_without_enough_history_are_reported_not_silently_dropped():
    async def trend_of(client, sym):
        return trend(years=1.0) if sym == "NEW" else trend(pct=-15.0)

    ranked, skipped = asyncio.run(sc.screen(None, trend_of, ["NEW", "OLD"], limit=10))
    assert [r["symbol"] for r in ranked] == ["OLD"]
    assert skipped == ["NEW"], "a symbol we could not score must be named, not vanish"
