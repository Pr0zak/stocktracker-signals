"""Memory layer: the recall loop, and the shapes that silently broke it.

The headline test here is `test_real_verdict_object_reaches_the_scorecard`. A production bug shipped
because every existing test passed signals as string literals ("buy"), while the real callers pass
`verdict.model_dump()` — which hands back the Enum MEMBER, whose `str()` is "Signal.buy". Nothing
matched the buy filters, so the scorecard would have read "not enough data yet" forever while looking
perfectly healthy. Tests that construct their own convenient inputs cannot catch that; these use the
same objects the service does.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile

import pytest


@pytest.fixture()
def memory(monkeypatch):
    """A fresh, isolated memory DB per test."""
    d = tempfile.mkdtemp()
    monkeypatch.setenv("SIGNALS_DATA_DIR", d)
    import importlib

    from app import memory as mem

    importlib.reload(mem)
    yield mem
    if mem._conn is not None:
        mem._conn.close()
        mem._conn = None


def _summary(rsi: float = 32.0, price: float = 100.0, date: str = "20260101") -> dict:
    """A snapshot with enough dimensions to be recordable and recallable."""
    return {
        "symbol": "AAPL", "price": price, "as_of_date": date, "rsi14": rsi,
        "pct_vs_sma20": -4.0, "pct_vs_sma50": -8.0, "macd_hist": 0.4,
        "bollinger_pct_b": 0.2, "stochastic_k": rsi, "pct_off_52w_high": -12.0,
        "rel_strength_3mo_vs_benchmark": 0.98,
    }


def _score_all(mem, symbol: str = "AAPL") -> int:
    """Grade every pending row against a synthetic series where the symbol beats the benchmark."""
    dates = [f"2026010{i + 1}" if i < 9 else f"202601{i + 1}" for i in range(40)]
    closes = [100.0 * (1.003 ** i) for i in range(40)]
    bench = [100.0 * (1.001 ** i) for i in range(40)]
    return mem.score_symbol(symbol, dates, closes, bench_dates=dates, bench_closes=bench)


def test_real_verdict_object_reaches_the_scorecard(memory):
    """Record what the SERVICE records — a real `Verdict.model_dump()` — not a hand-written dict.

    This is the regression guard for the enum bug: `model_dump()` returns `Signal.buy`, and storing
    `str()` of that gave "Signal.buy", which matched no buy filter anywhere.
    """
    from app.analyst import Signal, Verdict

    verdict = Verdict(
        signal=Signal.buy, conviction=72, thesis="t", rationale=[], key_risks=[],
        invalidation="x", horizon="2w", catalysts=[],
    )
    dumped = verdict.model_dump()
    # Guard the premise: if pydantic ever starts returning a plain string here, this test still
    # passes but is no longer exercising the risky shape — so assert the shape explicitly.
    assert dumped["signal"] is Signal.buy, "expected model_dump() to yield the Enum member"

    for i in range(8):
        assert memory.record_verdict(
            symbol="AAPL", summary=_summary(rsi=30 + i), verdict=dumped, origin="model",
        )
    _score_all(memory)

    stats = memory.stats()
    assert "buy_calls" in stats, "a real Verdict's buy calls never reached the scorecard"
    assert stats["buy_calls"]["n"] == 8
    assert stats["buy_calls"]["beat_rate_20d"] is not None


@pytest.mark.parametrize(
    "raw,expected",
    [("buy", "buy"), ("Signal.buy", "buy"), ("STRONG_BUY", "strong_buy"), ("", None), (None, None)],
)
def test_signal_normalisation(memory, raw, expected):
    assert memory._signal_str(raw) == expected


def test_enum_member_normalises(memory):
    from app.analyst import Signal

    assert memory._signal_str(Signal.buy) == "buy"


def test_legacy_enum_rows_are_repaired(memory):
    """Rows written before normalisation must be fixed on open, not left invisible forever."""
    from app.analyst import Signal, Verdict

    v = Verdict(signal=Signal.buy, conviction=70, thesis="t", rationale=[], key_risks=[],
                invalidation="x", horizon="2w", catalysts=[]).model_dump()
    for i in range(8):
        memory.record_verdict(symbol="AAPL", summary=_summary(rsi=30 + i), verdict=v, origin="model")
    _score_all(memory)

    db = sqlite3.connect(memory._FILE)
    db.execute("UPDATE verdicts SET signal = 'Signal.buy' WHERE origin = 'model'")
    db.commit()
    db.close()
    memory._conn = None  # force a reopen so _migrate runs

    assert memory.stats()["buy_calls"]["n"] == 8


def test_backfill_rows_never_count_as_model_calls(memory):
    """A replayed setup has no opinion attached; counting it would relabel drift as calibration."""
    from app.analyst import Signal, Verdict

    v = Verdict(signal=Signal.buy, conviction=70, thesis="t", rationale=[], key_risks=[],
                invalidation="x", horizon="2w", catalysts=[]).model_dump()
    for i in range(10):
        memory.record_verdict(symbol="AAPL", summary=_summary(rsi=30 + i), verdict={},
                              origin="backfill")
    _score_all(memory)
    assert "buy_calls" not in memory.stats(), "backfill leaked into the model's scorecard"

    for i in range(6):
        memory.record_verdict(symbol="AAPL", summary=_summary(rsi=30 + i), verdict=v, origin="model")
    _score_all(memory)
    stats = memory.stats()
    assert stats["buy_calls"]["n"] == 6, "model scorecard must count ONLY real verdicts"
    assert stats["by_origin"]["backfill"] == 10


def test_sandbox_fills_scored_separately_from_analyst_calls(memory):
    from app.analyst import Signal, Verdict

    v = Verdict(signal=Signal.buy, conviction=70, thesis="t", rationale=[], key_risks=[],
                invalidation="x", horizon="2w", catalysts=[]).model_dump()
    for i in range(6):
        memory.record_verdict(symbol="AAPL", summary=_summary(rsi=30 + i), verdict=v, origin="model")
        memory.record_verdict(symbol="AAPL", summary=_summary(rsi=30 + i), verdict=v, origin="sandbox")
    _score_all(memory)
    stats = memory.stats()
    assert stats["buy_calls"]["n"] == 6
    assert stats["sandbox_buys"]["n"] == 6


def test_opposite_setups_are_not_neighbours(memory):
    """RSI 29 vs RSI 79 is about as different as two setups get; agreeing dimensions must not
    average that away (they did, scoring 0.94 against a 1.15 threshold, before the per-dimension veto)."""
    for i in range(10):
        memory.record_verdict(symbol="AAPL", summary=_summary(rsi=28 + i * 0.3), verdict={},
                              origin="backfill")
    _score_all(memory)
    assert memory.similar_setups("AAPL", _summary(rsi=29)) is not None
    overbought = {**_summary(rsi=79), "pct_vs_sma20": 5.0, "pct_vs_sma50": 11.0,
                  "pct_off_52w_high": -0.5}
    assert memory.similar_setups("AAPL", overbought) is None


def test_thin_samples_report_nothing(memory):
    """Below the minimum, emit no track record at all rather than a number that reads as evidence."""
    for i in range(3):
        memory.record_verdict(symbol="AAPL", summary=_summary(rsi=30 + i), verdict={},
                              origin="backfill")
    _score_all(memory)
    assert memory.similar_setups("AAPL", _summary()) is None


def test_unscored_rows_teach_nothing(memory):
    """A verdict with no realized outcome yet must not contribute to a track record."""
    for i in range(20):
        memory.record_verdict(symbol="AAPL", summary=_summary(rsi=30 + i * 0.2), verdict={},
                              origin="backfill")
    assert memory.similar_setups("AAPL", _summary()) is None  # nothing scored yet


def test_partial_horizon_is_not_scored(memory):
    """A 20-day return must never be computed from fewer than 20 bars."""
    memory.record_verdict(symbol="AAPL", summary=_summary(date="20260101"), verdict={},
                          origin="backfill")
    dates = [f"2026010{i + 1}" for i in range(9)]          # only 9 bars exist
    closes = [100.0 * (1.003 ** i) for i in range(9)]
    assert memory.score_symbol("AAPL", dates, closes) == 0
    assert memory.stats()["scored"] == 0


def test_recording_never_raises_on_junk(memory):
    """Memory is best-effort by contract — it must not be able to break a verdict."""
    assert memory.record_verdict(symbol="AAPL", summary={}, verdict={}) is not None
    assert memory.record_verdict(symbol="", summary={"price": "not a number"}, verdict={}) is not None
    assert memory.similar_setups("AAPL", {}) is None


def test_blocked_summary_counts_which_rules_bind(memory):
    """The aggregate is the whole reason the blocked log exists."""
    assert memory.blocked_summary() is None          # nothing logged yet
    for _ in range(3):
        memory.add_note("blocked", "buy NVDA blocked", symbol="NVDA",
                        meta={"skip_reason": "position cap"})
    memory.add_note("blocked", "buy TSLA blocked", symbol="TSLA",
                    meta={"skip_reason": "excluded ticker"})
    out = memory.blocked_summary()
    assert out["total"] == 4
    assert out["by_reason"]["position cap"] == 3
    assert out["by_reason"]["excluded ticker"] == 1


def test_recent_notes_filters_and_orders(memory):
    memory.add_note("strategy", "week one")
    memory.add_note("strategy", "week two")
    memory.add_note("research", "a finding")
    strat = memory.recent_notes(kind="strategy")
    assert [n["body"] for n in strat] == ["week two", "week one"]   # newest first
    assert len(memory.recent_notes()) == 3


def test_fts_index_is_gone(memory):
    """The unused full-text index must be dropped, not left as a maintenance surface."""
    memory.add_note("research", "some prose")        # would have fired the old triggers
    db = sqlite3.connect(memory._FILE)
    names = {r[0] for r in db.execute("SELECT name FROM sqlite_master")}
    db.close()
    assert "notes_fts" not in names
    assert "notes_ai" not in names and "notes_ad" not in names


def test_seeding_targets_only_symbols_without_a_baseline(memory):
    for i in range(25):
        memory.record_verdict(symbol="AAPL", summary=_summary(rsi=30 + i * 0.2), verdict={},
                              origin="backfill")
    memory.record_verdict(symbol="MSFT", summary=_summary(), verdict={}, origin="backfill")
    missing = memory.symbols_missing_baseline(["AAPL", "MSFT", "NVDA"])
    assert missing == ["MSFT", "NVDA"]               # AAPL has enough; MSFT has only 1 row
