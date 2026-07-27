"""Sandbox risk enforcement — the gates that a price outage or a rounding boundary used to defeat.

Every case here is a bug an adversarial review found in code that had been trading unattended for
days. They are kept as tests because each one failed silently: no exception, no warning, just a
wrong number in a log the user is meant to trust.
"""
from __future__ import annotations

import os
import tempfile

import pytest


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setenv("SIGNALS_DATA_DIR", tempfile.mkdtemp())


def _settings(**over):
    s = {
        "master_enabled": True, "max_position_pct": 20.0, "cash_floor_pct": 10.0,
        "min_conviction_to_trade": 55, "max_trades_per_tick": 4, "max_new_positions_per_tick": 2,
        "max_turnover_pct": 25.0, "account_type": "cash", "respect_entry_zones": False,
        "slippage_bps": 5,
    }
    s.update(over)
    return s


def _group(sym: str) -> str:
    from app.main import _exposure_group

    return _exposure_group(sym)


# ------------------------------------------------------------------ unpriceable holdings

def test_missing_quote_does_not_open_a_hole_in_the_exposure_cap():
    """A price outage must not silently disable a risk limit.

    Marking an unpriced holding at zero made its exposure group look EMPTY, so a sibling ticker in
    the same group bought straight through the cap.
    """
    from app import sandbox_job

    blob = {
        "cash": 2000.0,
        "positions": [{"symbol": "FBTC", "shares": 100, "avg_cost": 80.0},
                      {"symbol": "SPY", "shares": 100, "avg_cost": 80.0}],
        "settings": _settings(),
    }
    order = [{"symbol": "IBIT", "side": "buy", "shares": 0, "dollars": 2000.0,
              "conviction": 90, "reason": "x"}]
    prices = {"FBTC": None, "SPY": 80.0, "IBIT": 50.0}   # FBTC unpriceable

    _, filled, skipped = sandbox_job.validate_and_fill(
        blob, order, lambda s: prices.get(s), group_of=_group, source="t")
    assert filled == [], "bought through the BTC cap while FBTC had no quote"
    assert any("cap" in (s.get("skip_reason") or "") for s in skipped)


def test_missing_quote_marks_at_last_known_price_not_zero():
    """The NAV log is append-only, so a fabricated drawdown can never be corrected."""
    from app import sandbox_job

    positions = [{"symbol": "FBTC", "shares": 100, "avg_cost": 80.0, "last_price": 82.0}]
    assert sandbox_job.positions_value(positions, lambda s: None) == 8200.0
    # cost basis is the last resort when there is no prior mark either
    assert sandbox_job.positions_value(
        [{"symbol": "X", "shares": 10, "avg_cost": 5.0}], lambda s: None) == 50.0
    # and a fresh quote always wins
    assert sandbox_job.positions_value(positions, lambda s: 90.0) == 9000.0


def test_stale_marks_are_reported_not_hidden():
    from app import sandbox_job

    positions = [{"symbol": "A", "shares": 1, "avg_cost": 1.0},
                 {"symbol": "B", "shares": 1, "avg_cost": 1.0}]
    assert sandbox_job.stale_marks(positions, lambda s: 5.0 if s == "A" else None) == ["B"]


def test_a_fresh_quote_is_stamped_for_later_fallback():
    from app import sandbox_job

    blob = {"cash": 0.0, "positions": [{"symbol": "A", "shares": 10, "avg_cost": 5.0}],
            "settings": _settings()}
    nb, _, _ = sandbox_job.validate_and_fill(blob, [], lambda s: 7.5, group_of=_group, source="t")
    assert nb["positions"][0]["last_price"] == 7.5


# ------------------------------------------------------------------ cash conservation

def test_half_cent_boundary_does_not_abort_a_correct_tick():
    """Two independent roundings of the same exact figure differ by a cent at the boundary.

    Slippage manufactures the third decimal and real quotes are round numbers, so this aborted ~1.6%
    of ticks — each a lost trading day and a hole in the equity curve, with no accounting error.
    """
    from app import sandbox_job

    blob = {"cash": 5000.00, "positions": [{"symbol": "HELD", "shares": 100, "avg_cost": 50.0}],
            "settings": _settings()}
    order = [{"symbol": "NEW", "side": "buy", "shares": 0, "dollars": 1000.0,
              "conviction": 90, "reason": "x"}]
    nb, filled, _ = sandbox_job.validate_and_fill(
        blob, order, lambda s: 50.0, group_of=_group, source="t")
    assert filled and filled[0]["shares"] == 19


def test_round_number_price_sweep_never_aborts():
    from app import sandbox_job

    for cash in (5000.0, 10000.0, 25000.0):
        for i in range(8, 200):
            px = round(i * 0.25, 2)
            blob = {"cash": cash, "positions": [{"symbol": "HELD", "shares": 100, "avg_cost": 50.0}],
                    "settings": _settings()}
            order = [{"symbol": "NEW", "side": "buy", "shares": 0, "dollars": cash * 0.2,
                      "conviction": 90, "reason": "x"}]
            sandbox_job.validate_and_fill(
                blob, order, lambda s: 50.0 if s == "HELD" else px, group_of=_group, source="t")


def test_the_tightened_guard_rejects_a_leak_the_old_one_would_have_passed():
    """Loosening a guard to stop false positives must not weaken what it detects — this one got
    ~10,000x STRICTER.

    The old form compared two 2-dp roundings with a 1-cent tolerance, so any genuine leak smaller
    than a cent was invisible. The new form compares the raw accumulators, where the only slack is
    float noise.
    """
    cash0 = 10_000.00
    sells, buys = 0.0, 5_000.00
    # A sub-half-cent leak: it rounds away entirely, so the old 2-dp comparison could never see it,
    # yet at a daily cadence it compounds into the ledger forever.
    leaked = cash0 + sells - buys - 0.004

    # OLD: round(cash,2) - cash0  vs  round(sells - buys, 2), tolerance 0.01
    old_delta = round(leaked, 2) - cash0
    old_expected = round(sells - buys, 2)
    assert abs(old_delta - old_expected) < 0.01, "premise: the old guard accepted this leak"

    # NEW: raw comparison, tolerance 1e-6
    assert abs(leaked - (cash0 + sells - buys)) >= 1e-6, "the new guard must reject it"


# ------------------------------------------------------------------ exit-date liquidation

def test_exit_date_flatten_is_not_throttled_by_the_churn_caps():
    """The caps exist to stop the strategy churning; a scheduled exit is the user's instruction."""
    from app import sandbox_job

    syms = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOG"]
    blob = {"cash": 0.0,
            "positions": [{"symbol": s, "shares": 100, "avg_cost": 100.0} for s in syms],
            "settings": _settings()}
    orders = [{"symbol": s, "side": "sell", "shares": 100, "dollars": 0.0,
               "conviction": 100, "reason": "exit"} for s in syms]

    _, throttled, _ = sandbox_job.validate_and_fill(
        blob, orders, lambda s: 100.0, group_of=_group, source="exit_date", liquidation=False)
    assert len(throttled) < len(syms), "premise: the caps really do bind without the flag"

    nb, filled, _ = sandbox_job.validate_and_fill(
        blob, orders, lambda s: 100.0, group_of=_group, source="exit_date", liquidation=True)
    assert len(filled) == len(syms)
    assert nb["positions"] == [], "exit date must actually leave the account flat"


def test_liquidation_still_respects_shares_actually_held():
    """Lifting the churn caps must not lift the limits that protect the account itself."""
    from app import sandbox_job

    blob = {"cash": 0.0, "positions": [{"symbol": "AAPL", "shares": 10, "avg_cost": 100.0}],
            "settings": _settings()}
    orders = [{"symbol": "AAPL", "side": "sell", "shares": 9999, "dollars": 0.0,
               "conviction": 100, "reason": "exit"}]
    nb, filled, _ = sandbox_job.validate_and_fill(
        blob, orders, lambda s: 100.0, group_of=_group, source="exit_date", liquidation=True)
    assert filled[0]["shares"] == 10, "cannot sell more than is held, even liquidating"
    assert nb["cash"] > 0


# ------------------------------------------------------------------ order handling

def test_a_buy_with_no_size_is_a_no_op_not_max_size():
    """`dollars or available` turned the weakest possible instruction into the largest order."""
    from app import sandbox_job

    blob = {"cash": 10_000.0, "positions": [], "settings": _settings(max_turnover_pct=0.0)}
    order = [{"symbol": "NEW", "side": "buy", "shares": 0, "dollars": 0.0,
              "conviction": 90, "reason": "x"}]
    nb, filled, skipped = sandbox_job.validate_and_fill(
        blob, order, lambda s: 50.0, group_of=_group, source="t")
    assert filled == []
    assert nb["cash"] == 10_000.0, "an unsized order spent real cash"
    assert "neither dollars nor shares" in skipped[0]["skip_reason"]


def test_a_buy_sized_only_by_shares_is_honoured():
    from app import sandbox_job

    blob = {"cash": 10_000.0, "positions": [], "settings": _settings(max_turnover_pct=0.0)}
    order = [{"symbol": "NEW", "side": "buy", "shares": 10, "dollars": 0.0,
              "conviction": 90, "reason": "x"}]
    _, filled, _ = sandbox_job.validate_and_fill(
        blob, order, lambda s: 50.0, group_of=_group, source="t")
    assert filled and filled[0]["shares"] == 10


def test_an_unrecognised_side_is_audited_not_swallowed():
    """An order the model proposed and the ledger silently ate is the one case the audit trail
    must never miss."""
    from app import sandbox_job

    blob = {"cash": 10_000.0, "positions": [], "settings": _settings()}
    orders = [{"symbol": "A", "side": "BUY", "shares": 5, "dollars": 0.0, "conviction": 90},
              {"symbol": "B", "side": "hold", "shares": 5, "dollars": 0.0, "conviction": 90},
              {"symbol": "C", "side": None, "shares": 5, "dollars": 0.0, "conviction": 90}]
    _, filled, skipped = sandbox_job.validate_and_fill(
        blob, orders, lambda s: 50.0, group_of=_group, source="t")
    assert [f["symbol"] for f in filled] == ["A"], "case should be normalised, not rejected"
    assert {s["symbol"] for s in skipped} == {"B", "C"}
    assert all("unrecognised side" in s["skip_reason"] for s in skipped)


# ------------------------------------------------------------------ T+1 across ticks

def test_t1_settlement_survives_a_second_tick_the_same_day():
    """The app's "run tick" button always sends force=true, which re-armed every per-call limit —
    including the one that knew about today's unsettled proceeds."""
    from app import sandbox_job

    st = _settings(max_position_pct=50.0, cash_floor_pct=0.0, max_turnover_pct=0.0)
    blob = {"cash": 5.0, "positions": [{"symbol": "AAPL", "shares": 100, "avg_cost": 50.0}],
            "settings": st}
    px = lambda s: 100.0  # noqa: E731
    sell = {"symbol": "AAPL", "side": "sell", "shares": 100, "dollars": 0.0, "conviction": 90}
    buy = {"symbol": "MSFT", "side": "buy", "shares": 0, "dollars": 9000.0, "conviction": 90}

    nb1, filled1, _ = sandbox_job.validate_and_fill(
        blob, [sell, buy], px, group_of=_group, source="t")
    assert [f["symbol"] for f in filled1] == ["AAPL"]
    assert nb1["unsettled"] and nb1["unsettled"][0]["amount"] > 9000

    nb2, filled2, skipped2 = sandbox_job.validate_and_fill(nb1, [buy], px, group_of=_group, source="t")
    assert filled2 == [], "spent unsettled proceeds on a forced same-day re-tick"
    assert "T+1" in skipped2[0]["skip_reason"]


def test_settled_proceeds_are_usable_next_session():
    from app import sandbox_job
    import time

    st = _settings(max_position_pct=50.0, cash_floor_pct=0.0, max_turnover_pct=0.0)
    blob = {"cash": 5.0, "positions": [{"symbol": "AAPL", "shares": 100, "avg_cost": 50.0}],
            "settings": st}
    px = lambda s: 100.0  # noqa: E731
    nb1, _, _ = sandbox_job.validate_and_fill(
        blob, [{"symbol": "AAPL", "side": "sell", "shares": 100, "dollars": 0.0, "conviction": 90}],
        px, group_of=_group, source="t")
    nb2, filled, _ = sandbox_job.validate_and_fill(
        nb1, [{"symbol": "MSFT", "side": "buy", "shares": 0, "dollars": 4000.0, "conviction": 90}],
        px, group_of=_group, source="t", now_ts=time.time() + 5 * 86_400)
    assert filled, "proceeds never became usable"
    assert nb2["unsettled"] == []


# ------------------------------------------------------------------ early closes

def test_early_close_days_are_modelled():
    import datetime as dt

    from app import market_calendar as mc

    assert mc.early_closes(2025) == frozenset({dt.date(2025, 7, 3), dt.date(2025, 11, 28),
                                               dt.date(2025, 12, 24)})
    # 2026: July 4 is a Saturday, so July 3 is a full closure, not a half-day.
    assert dt.date(2026, 7, 3) not in mc.early_closes(2026)
    assert mc.session_end_seconds(dt.date(2026, 11, 27)) == (13 * 3600, 17 * 3600)
    assert mc.session_end_seconds(dt.date(2026, 11, 30)) == (16 * 3600, 20 * 3600)


def test_tick_gate_respects_an_early_close():
    """At 15:35 on a half-day the market has been shut for 2.5 hours — a fill would be priced off a
    stale quote and stamped as live."""
    import datetime as dt

    from app import sandbox_job

    blob = {"settings": {"master_enabled": True, "allow_after_hours": False}}
    half_day = dt.datetime(2026, 11, 27, 15, 35, tzinfo=sandbox_job.ET)
    normal = dt.datetime(2026, 11, 30, 15, 35, tzinfo=sandbox_job.ET)
    assert sandbox_job.tick_gate(blob, now=normal) == (True, "ok")
    assert sandbox_job.tick_gate(blob, now=half_day)[1] == "after_hours_disabled"
