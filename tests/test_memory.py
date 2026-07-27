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


_DAYS = [f"2026010{i + 1}" if i < 9 else f"202601{i + 1}" for i in range(40)]


def _day(i: int) -> str:
    """A distinct scorable bar date. Only the first 20 bars leave a full 20-session horizon."""
    return _DAYS[i % 20]


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
            symbol="AAPL", summary=_summary(rsi=30 + i, date=_day(i)), verdict=dumped, origin="model",
        )
    _score_all(memory)

    stats = memory.stats()
    assert "buy_calls" in stats, "a real Verdict's buy calls never reached the scorecard"
    assert stats["buy_calls"]["n"] == 8
    assert stats["buy_calls"]["correct_rate_20d"] is not None


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
        memory.record_verdict(symbol="AAPL", summary=_summary(rsi=30 + i, date=_day(i)), verdict=v, origin="model")
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
        memory.record_verdict(symbol="AAPL", summary=_summary(rsi=30 + i, date=_day(i)), verdict={},
                              origin="backfill")
    _score_all(memory)
    assert "buy_calls" not in memory.stats(), "backfill leaked into the model's scorecard"

    for i in range(6):
        memory.record_verdict(symbol="AAPL", summary=_summary(rsi=30 + i, date=_day(i)), verdict=v, origin="model")
    _score_all(memory)
    stats = memory.stats()
    assert stats["buy_calls"]["n"] == 6, "model scorecard must count ONLY real verdicts"
    assert stats["by_origin"]["backfill"] == 10


def test_sandbox_fills_scored_separately_from_analyst_calls(memory):
    from app.analyst import Signal, Verdict

    v = Verdict(signal=Signal.buy, conviction=70, thesis="t", rationale=[], key_risks=[],
                invalidation="x", horizon="2w", catalysts=[]).model_dump()
    for i in range(6):
        memory.record_verdict(symbol="AAPL", summary=_summary(rsi=30 + i, date=_day(i)), verdict=v, origin="model")
        memory.record_verdict(symbol="AAPL", summary=_summary(rsi=30 + i, date=_day(i)), verdict=v, origin="sandbox")
    _score_all(memory)
    stats = memory.stats()
    assert stats["buy_calls"]["n"] == 6
    assert stats["sandbox_buys"]["n"] == 6


def test_opposite_setups_are_not_neighbours(memory):
    """RSI 29 vs RSI 79 is about as different as two setups get; agreeing dimensions must not
    average that away (they did, scoring 0.94 against a 1.15 threshold, before the per-dimension veto)."""
    for i in range(10):
        memory.record_verdict(symbol="AAPL", summary=_summary(rsi=28 + i * 0.3, date=_day(i)), verdict={},
                              origin="backfill")
    _score_all(memory)
    assert memory.similar_setups("AAPL", _summary(rsi=29)) is not None
    overbought = {**_summary(rsi=79), "pct_vs_sma20": 5.0, "pct_vs_sma50": 11.0,
                  "pct_off_52w_high": -0.5}
    assert memory.similar_setups("AAPL", overbought) is None


def test_thin_samples_report_nothing(memory):
    """Below the minimum, emit no track record at all rather than a number that reads as evidence."""
    for i in range(3):
        memory.record_verdict(symbol="AAPL", summary=_summary(rsi=30 + i, date=_day(i)), verdict={},
                              origin="backfill")
    _score_all(memory)
    assert memory.similar_setups("AAPL", _summary()) is None


def test_unscored_rows_teach_nothing(memory):
    """A verdict with no realized outcome yet must not contribute to a track record."""
    for i in range(20):
        memory.record_verdict(symbol="AAPL", summary=_summary(rsi=30 + i * 0.2, date=_day(i)), verdict={},
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
        memory.record_verdict(symbol="AAPL", summary=_summary(rsi=30 + i * 0.2, date=_day(i)), verdict={},
                              origin="backfill")
    memory.record_verdict(symbol="MSFT", summary=_summary(), verdict={}, origin="backfill")
    missing = memory.symbols_missing_baseline(["AAPL", "MSFT", "NVDA"])
    assert missing == ["MSFT", "NVDA"]               # AAPL has enough; MSFT has only 1 row


def test_sell_calls_are_scored_with_an_inverted_convention(memory):
    """A sell is RIGHT when the name underperforms the index — the alternative to holding it.

    Scored on raw forward return, every sell in a rising market would look wrong regardless of
    judgement, which is exactly backwards.
    """
    from app.analyst import Signal, Verdict

    def dump(sig):
        return Verdict(signal=sig, conviction=70, thesis="t", rationale=[], key_risks=[],
                       invalidation="x", horizon="2w", catalysts=[]).model_dump()

    for i in range(8):
        memory.record_verdict(symbol="AAPL", summary=_summary(rsi=30 + i, date=_day(i)), verdict=dump(Signal.sell),
                              origin="model")
    # Symbol rises 0.3%/bar while the benchmark rises 0.1%/bar: the name OUTperformed, so these
    # sells were WRONG even though the raw return is positive.
    _score_all(memory)
    stats = memory.stats()
    assert "sell_calls" in stats
    assert stats["sell_calls"]["avg_fwd_20d_pct"] > 0          # price went up...
    assert stats["sell_calls"]["avg_avoided_20d_pct"] < 0      # ...so the sell avoided nothing
    assert stats["sell_calls"]["correct_rate_20d"] == 0.0      # and was never right


def test_sell_calls_score_well_when_the_name_underperforms(memory):
    from app.analyst import Signal, Verdict

    v = Verdict(signal=Signal.sell, conviction=70, thesis="t", rationale=[], key_risks=[],
                invalidation="x", horizon="2w", catalysts=[]).model_dump()
    for i in range(8):
        memory.record_verdict(symbol="LOSER", summary={**_summary(rsi=30 + i, date=_day(i)), "symbol": "LOSER"},
                              verdict=v, origin="model")
    dates = [f"2026010{i + 1}" if i < 9 else f"202601{i + 1}" for i in range(40)]
    falling = [100.0 * (0.998 ** i) for i in range(40)]        # name falls
    bench = [100.0 * (1.001 ** i) for i in range(40)]          # index rises
    memory.score_symbol("LOSER", dates, falling, bench_dates=dates, bench_closes=bench)

    card = memory.stats()["sell_calls"]
    assert card["avg_avoided_20d_pct"] > 0, "selling a name that fell should score positively"
    assert card["correct_rate_20d"] == 1.0


def test_hold_is_scored_as_neither_side(memory):
    """'hold' is the absence of a call; scoring it would reward doing nothing in a rising market."""
    from app.analyst import Signal, Verdict

    v = Verdict(signal=Signal.hold, conviction=50, thesis="t", rationale=[], key_risks=[],
                invalidation="x", horizon="2w", catalysts=[]).model_dump()
    for i in range(10):
        memory.record_verdict(symbol="AAPL", summary=_summary(rsi=30 + i, date=_day(i)), verdict=v, origin="model")
    _score_all(memory)
    stats = memory.stats()
    assert "buy_calls" not in stats and "sell_calls" not in stats


def test_backfill_does_not_crowd_out_the_model_scorecard(memory):
    """Backfill outnumbers live verdicts ~140:1, so truncating to the k nearest BEFORE filtering by
    origin made when_model_said_buy structurally unreachable — and worse every night as seeding ran.
    """
    from app.analyst import Signal, Verdict

    v = Verdict(signal=Signal.buy, conviction=70, thesis="t", rationale=[], key_risks=[],
                invalidation="x", horizon="2w", catalysts=[]).model_dump()
    # 200 near-identical backfill rows, then 6 real buy calls slightly further away.
    for i in range(200):
        memory.record_verdict(symbol="AAPL", summary=_summary(rsi=32 + i * 0.001, date=_day(i)), verdict={},
                              origin="backfill")
    for i in range(6):
        memory.record_verdict(symbol="AAPL", summary=_summary(rsi=34 + i * 0.1, date=_day(i)), verdict=v,
                              origin="model")
    _score_all(memory)

    track = memory.similar_setups("AAPL", _summary(rsi=32))
    assert track and "this_symbol" in track
    assert "when_model_said_buy" in track["this_symbol"], \
        "the model's own calls were crowded out by replayed setups"
    assert track["this_symbol"]["when_model_said_buy"]["n"] == 6


def test_rel_strength_disagreement_vetoes_a_match(memory):
    """rel_strength is named in the docstring as the point of the exercise, yet its 0.8 weight
    silently exempted it from the veto — so opposite relative strength still matched."""
    def snap(rs):
        return {**_summary(rsi=45.0), "rel_strength_3mo_vs_benchmark": rs}

    for i in range(12):
        memory.record_verdict(symbol="A", summary={**snap(0.28 + i * 0.005), "as_of_date": _day(i)}, verdict={},
                              origin="backfill")
    _score_all(memory, "A")
    assert memory.similar_setups("A", snap(0.30)) is not None      # same regime matches
    assert memory.similar_setups("A", snap(-0.30)) is None         # opposite must not


def test_the_same_bar_is_recorded_once_per_origin(memory):
    """A refresh=true verdict, or a manual /scan/run alongside the timer, used to insert ANOTHER row
    for the identical bar — inflating `n` and double-weighting that bar in every median."""
    from app.analyst import Signal, Verdict

    v = Verdict(signal=Signal.buy, conviction=70, thesis="t", rationale=[], key_risks=[],
                invalidation="x", horizon="2w", catalysts=[]).model_dump()
    for _ in range(5):                                   # same bar, recorded five times
        memory.record_verdict(symbol="AAPL", summary=_summary(date=_day(0)), verdict=v,
                              origin="model")
    assert memory.stats()["by_origin"]["model"] == 1

    # a different origin for the same bar is a genuinely different observation
    memory.record_verdict(symbol="AAPL", summary=_summary(date=_day(0)), verdict=v, origin="sandbox")
    assert memory.stats()["by_origin"]["sandbox"] == 1
    # and a different bar is its own row
    memory.record_verdict(symbol="AAPL", summary=_summary(date=_day(1)), verdict=v, origin="model")
    assert memory.stats()["by_origin"]["model"] == 2


def test_the_newest_read_of_a_bar_wins(memory):
    from app.analyst import Signal, Verdict

    def dump(sig, conv):
        return Verdict(signal=sig, conviction=conv, thesis="t", rationale=[], key_risks=[],
                       invalidation="x", horizon="2w", catalysts=[]).model_dump()

    memory.record_verdict(symbol="AAPL", summary=_summary(date=_day(0)), verdict=dump(Signal.hold, 40),
                          origin="model")
    memory.record_verdict(symbol="AAPL", summary=_summary(date=_day(0)), verdict=dump(Signal.buy, 80),
                          origin="model")
    import sqlite3
    db = sqlite3.connect(memory._FILE)
    row = db.execute("SELECT signal, conviction FROM verdicts").fetchall()
    db.close()
    assert row == [("buy", 80)], "a re-read of the same bar should supersede, not duplicate"


def test_migration_succeeds_on_a_database_that_already_has_duplicates(memory, monkeypatch):
    """The exact production failure: the unique index was created in _SCHEMA, which runs BEFORE the
    dedupe, so opening a real database full of duplicate bars raised IntegrityError — and because
    _db() had already published the half-built connection, every later call skipped the migration
    silently and the index was never created at all.
    """
    import sqlite3

    from app.analyst import Signal, Verdict

    v = Verdict(signal=Signal.buy, conviction=70, thesis="t", rationale=[], key_risks=[],
                invalidation="x", horizon="2w", catalysts=[]).model_dump()
    memory.record_verdict(symbol="AAPL", summary=_summary(date=_day(0)), verdict=v, origin="model")

    # Forge duplicates the way the old code could, bypassing the index.
    db = sqlite3.connect(memory._FILE)
    db.execute("DROP INDEX IF EXISTS ux_verdicts_bar")
    for _ in range(4):
        db.execute(
            "INSERT INTO verdicts (symbol, ts, asof_date, signal, origin) VALUES (?,?,?,?,?)",
            ("AAPL", 1.0, _day(0), "buy", "model"),
        )
    db.commit()
    assert db.execute("SELECT COUNT(*) FROM verdicts").fetchone()[0] == 5
    db.close()

    memory._conn.close()
    memory._conn = None          # reopen: schema + migration must both survive this

    stats = memory.stats()
    assert stats["by_origin"]["model"] == 1, "duplicates were not collapsed"

    db = sqlite3.connect(memory._FILE)
    names = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    db.close()
    assert "ux_verdicts_bar" in names, "the unique index was never created"


def test_a_failed_open_does_not_poison_later_calls(memory, monkeypatch):
    """A half-initialised connection must never be published — it made one startup error look like
    permanent health while silently skipping every migration."""
    calls = {"n": 0}
    real_migrate = memory._migrate

    def flaky(db):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return real_migrate(db)

    memory._conn = None
    monkeypatch.setattr(memory, "_migrate", flaky)
    with pytest.raises(RuntimeError):
        memory._db()
    assert memory._conn is None, "a failed open left a live connection behind"
    # the next attempt must fully initialise
    assert memory._db() is not None
    assert calls["n"] == 2
