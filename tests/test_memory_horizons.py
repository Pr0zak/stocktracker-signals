"""The long horizons (63 and 252 bars) and the two clocks that stopped them from ever existing.

Until 2026-09-04 memory.py graded every verdict at 5 and 20 sessions and nothing else, for a sandbox
whose objective is stated in years. Two things beyond the schema made a one-year mark impossible even
once the columns existed, and both are pinned here:

1. `scan_job.score_memory` fetched the default 1y range. A row old enough to carry a 252-bar mark has
   its anchor at index 0 of a 252-bar series or just outside it, so the mark could never be computed
   from what was fetched. The range must follow the oldest pending row.
2. `_RETAIN_DAYS` was 730. The oldest backfill rows (2024-09-24) would have been deleted on
   2026-09-24, three weeks before the first of them could have been graded at a year.

Every test that scores builds its own multi-year series; the 40-bar fixtures in test_memory.py are
deliberately too short to reach the new horizons, which is itself the first case below.
"""
from __future__ import annotations

import sqlite3
import time

import pytest

from tests.test_memory import _summary, memory  # noqa: F401 — the fixture


def _dates(n: int) -> list[str]:
    """`n` distinct, sortable 8-digit bar dates (the shape `_asof_date` requires)."""
    return [f"{20200000 + i}" for i in range(n)]


def _series(n: int, *, step: float = 1.001, bench_step: float = 1.0005):
    dates = _dates(n)
    closes = [100.0 * (step ** i) for i in range(n)]
    bench = [100.0 * (bench_step ** i) for i in range(n)]
    return dates, closes, bench


def _record(mem, symbol: str, bar: int, *, ts: float | None = None, origin: str = "backfill",
            signal: str | None = None, rsi: float = 32.0) -> int:
    v = {"signal": signal} if signal else {}
    rid = mem.record_verdict(
        symbol=symbol, summary=_summary(rsi=rsi, date=_dates(400)[bar]), verdict=v,
        origin=origin, ts=ts,
    )
    assert rid
    return rid


def _row(mem, rid: int) -> sqlite3.Row:
    return mem._db().execute("SELECT * FROM verdicts WHERE id = ?", (rid,)).fetchone()


# --- the marks fill in one horizon at a time ---------------------------------------------------


def test_a_short_series_writes_the_twenty_day_marks_and_leaves_the_long_ones_null(memory):
    rid = _record(memory, "AAPL", 0)
    dates, closes, bench = _series(40)
    assert memory.score_symbol("AAPL", dates, closes, bench_dates=dates, bench_closes=bench) == 1
    r = _row(memory, rid)
    assert r["fwd_20d"] is not None and r["scored_at"] is not None
    assert r["fwd_63d"] is None and r["fwd_252d"] is None
    assert r["bench_fwd_63d"] is None and r["bench_fwd_252d"] is None


def test_the_long_marks_fill_in_later_without_touching_the_twenty_day_grade(memory):
    rid = _record(memory, "AAPL", 0)
    dates, closes, bench = _series(40)
    memory.score_symbol("AAPL", dates, closes, bench_dates=dates, bench_closes=bench)
    first = _row(memory, rid)

    # A quarter later the series reaches 63 bars past the anchor but not 252.
    dates, closes, bench = _series(120)
    assert memory.score_symbol("AAPL", dates, closes, bench_dates=dates, bench_closes=bench) == 1
    mid = _row(memory, rid)
    assert mid["fwd_63d"] == pytest.approx((closes[63] - closes[0]) / closes[0] * 100, abs=1e-3)
    assert mid["bench_fwd_63d"] == pytest.approx((bench[63] - bench[0]) / bench[0] * 100, abs=1e-3)
    assert mid["fwd_252d"] is None
    # The grade it was given at 20 days is the grade it keeps.
    assert mid["fwd_20d"] == first["fwd_20d"]
    assert mid["scored_at"] == first["scored_at"]

    # A year on: the one-year mark lands, nothing else moves, and the row is then finished.
    dates, closes, bench = _series(300)
    assert memory.score_symbol("AAPL", dates, closes, bench_dates=dates, bench_closes=bench) == 1
    last = _row(memory, rid)
    assert last["fwd_252d"] == pytest.approx((closes[252] - closes[0]) / closes[0] * 100, abs=1e-3)
    assert last["fwd_63d"] == mid["fwd_63d"]
    assert memory.score_symbol("AAPL", dates, closes, bench_dates=dates, bench_closes=bench) == 0


def test_a_never_graded_row_gets_every_available_horizon_in_one_visit(memory):
    """The backfill path: rows written from a 2y replay are old enough for all four marks at once."""
    rid = _record(memory, "AAPL", 10)
    dates, closes, bench = _series(300)
    assert memory.score_symbol("AAPL", dates, closes, bench_dates=dates, bench_closes=bench) == 1
    r = _row(memory, rid)
    for h in memory.HORIZONS:
        assert r[f"fwd_{h}d"] is not None, f"missing fwd_{h}d"
        assert r[f"bench_fwd_{h}d"] is not None, f"missing bench_fwd_{h}d"


def test_a_partial_long_window_is_not_scored_early(memory):
    """A 252-bar return is never computed from 251 bars and called a year."""
    rid = _record(memory, "AAPL", 0)
    dates, closes, bench = _series(252)  # bars 0..251: the 252nd forward bar does not exist
    memory.score_symbol("AAPL", dates, closes, bench_dates=dates, bench_closes=bench)
    assert _row(memory, rid)["fwd_252d"] is None
    dates, closes, bench = _series(253)
    memory.score_symbol("AAPL", dates, closes, bench_dates=dates, bench_closes=bench)
    assert _row(memory, rid)["fwd_252d"] is not None


def test_a_row_without_a_twenty_day_outcome_is_still_left_entirely_unscored(memory):
    """The existing all-or-nothing rule for the first grade survives the per-horizon rewrite."""
    rid = _record(memory, "AAPL", 30)
    dates, closes, bench = _series(40)
    assert memory.score_symbol("AAPL", dates, closes, bench_dates=dates, bench_closes=bench) == 0
    r = _row(memory, rid)
    assert r["fwd_5d"] is None and r["scored_at"] is None


# --- what is pending, and how far back the fetch must reach ---------------------------------


def test_pending_follows_each_horizons_own_age_gate(memory):
    now = time.time()
    day = 86_400
    # Graded at 20d a month ago: nothing to fetch for until the quarter mark is possible.
    young = _record(memory, "AAPL", 0, ts=now - 30 * day)
    dates, closes, bench = _series(40)
    memory.score_symbol("AAPL", dates, closes, bench_dates=dates, bench_closes=bench)
    assert _row(memory, young)["scored_at"] is not None
    assert "AAPL" not in memory.pending_work()

    # The same row four months old: pending again, for its 63-bar mark.
    memory._db().execute("UPDATE verdicts SET ts = ? WHERE id = ?", (now - 120 * day, young))
    memory._db().commit()
    work = memory.pending_work()
    assert work["AAPL"] == pytest.approx(now - 120 * day, abs=1.0)

    # A second symbol whose oldest pending row is older reports THAT row's age, not the newest.
    _record(memory, "MSFT", 0, ts=now - 400 * day)
    _record(memory, "MSFT", 1, ts=now - 20 * day)
    work = memory.pending_work()
    assert work["MSFT"] == pytest.approx(now - 400 * day, abs=1.0)
    assert set(memory.pending_symbols()) == {"AAPL", "MSFT"}


def test_a_fully_graded_row_is_never_pending_again(memory):
    rid = _record(memory, "AAPL", 0, ts=time.time() - 800 * 86_400)
    dates, closes, bench = _series(300)
    memory.score_symbol("AAPL", dates, closes, bench_dates=dates, bench_closes=bench)
    r = _row(memory, rid)
    assert r["fwd_252d"] is not None
    assert memory.pending_work() == {}


def test_scoring_range_reaches_past_the_oldest_pending_row():
    from app.scan_job import scoring_range

    assert scoring_range(0) == "1y"
    assert scoring_range(200) == "1y"
    # A row waiting on its one-year mark is at least ~355 days old; 1y would put its anchor at the
    # very edge of the series or outside it, which is exactly the silent bound this replaces.
    assert scoring_range(360) == "2y"
    assert scoring_range(700) == "5y"
    assert scoring_range(1800) == "10y"


def test_score_memory_fetches_the_range_the_oldest_pending_row_needs(memory, monkeypatch):
    """The regression: with the default range, a 252-bar mark was unreachable however old the row."""
    import asyncio

    from app import scan_job
    from app.market import Series

    now = time.time()
    _record(memory, "AAPL", 0, ts=now - 400 * 86_400)
    _record(memory, "BTC-USD", 0, ts=now - 100 * 86_400)
    seen: dict[str, str] = {}
    dates, closes, _ = _series(300)

    async def fake_fetch(client, symbol, rng="1y", **kw):
        seen[symbol] = rng
        return Series(symbol=symbol, closes=closes, opens=[None] * 300, volumes=[None] * 300,
                      dates=dates, fifty_two_high=max(closes), fifty_two_low=min(closes),
                      currency="USD", highs=[], lows=[])

    monkeypatch.setattr(scan_job, "fetch_series", fake_fetch)
    monkeypatch.setattr(scan_job.memory, "prune", lambda: 0)
    scored = asyncio.run(scan_job.score_memory())
    assert scored == 2
    assert seen["AAPL"] == "2y", "a 400-day-old row cannot be inside a 1y series"
    assert seen["BTC-USD"] == "1y", "a 100-day-old row needs no more than the default"
    assert seen["^GSPC"] == "2y", "the benchmark must cover the widest symbol window"


# --- retention: the rows that can carry a one-year mark must survive to carry it ----------------


def test_retention_outlives_the_one_year_horizon_by_years(memory):
    """730 days would have deleted the 2024-09-24 backfill rows on 2026-09-24."""
    assert memory.retention_days() >= 4 * 365
    now = time.time()
    keep = _record(memory, "AAPL", 0, ts=now - 800 * 86_400)
    drop = _record(memory, "AAPL", 1, ts=now - (memory.retention_days() + 5) * 86_400)
    memory.add_note("strategy", "old note")
    memory._db().execute("UPDATE notes SET ts = ?", (now - 800 * 86_400,))
    memory._db().commit()
    removed = memory.prune()
    assert removed == 2, "one verdict past retention and one note past the notes window"
    assert _row(memory, keep) is not None
    assert _row(memory, drop) is None
    assert memory.recent_notes() == []


# --- migration: a database from before the columns existed ------------------------------------


def test_migration_adds_the_long_horizon_columns_to_an_older_database(memory):
    memory._db()  # create the file with today's schema, then strip it back to last week's
    memory._conn.close()
    memory._conn = None
    db = sqlite3.connect(memory._FILE)
    for col in ("fwd_63d", "fwd_252d", "bench_fwd_63d", "bench_fwd_252d"):
        db.execute(f"ALTER TABLE verdicts DROP COLUMN {col}")
    db.commit()
    have = {r[1] for r in db.execute("PRAGMA table_info(verdicts)")}
    assert "fwd_252d" not in have
    db.close()

    # reopen: schema + migration must both run against the older layout
    have = {r["name"] for r in memory._db().execute("PRAGMA table_info(verdicts)")}
    assert {"fwd_63d", "fwd_252d", "bench_fwd_63d", "bench_fwd_252d"} <= have
    # ...and the new columns are NULL on the old rows rather than anything that reads as a grade.
    rid = _record(memory, "AAPL", 0)
    assert _row(memory, rid)["fwd_252d"] is None


# --- what the readers say ---------------------------------------------------------------------


def _seed_cohort(memory, *, n_symbols: int, bars_per_symbol: int, series_len: int, rsi: float = 32.0):
    """A neighbourhood of `n_symbols × bars_per_symbol` rows, graded against `series_len` bars."""
    for s in range(n_symbols):
        sym = f"S{s:02d}"
        for b in range(bars_per_symbol):
            _record(memory, sym, b, rsi=rsi + b * 0.1)
        dates, closes, bench = _series(series_len)
        memory.score_symbol(sym, dates, closes, bench_dates=dates, bench_closes=bench)


def test_similar_setups_reports_each_long_horizon_only_once_it_has_a_sample(memory):
    _seed_cohort(memory, n_symbols=3, bars_per_symbol=3, series_len=120)  # 63d yes, 252d no
    track = memory.similar_setups("ZZZ", _summary(rsi=32.0))
    near = track["analogues"]
    assert "vs_benchmark" in near
    assert "vs_benchmark_63d" in near
    assert "vs_benchmark_252d" not in near, "no row carries a one-year mark yet"
    q = near["vs_benchmark_63d"]
    assert q["n"] == 9 and q["n_symbols"] == 3
    assert set(q) == {"n", "n_symbols", "median_excess_63d_pct", "beat_rate_63d"}


def test_similar_setups_carries_the_symbol_count_beside_every_n(memory):
    """Forty bars from two names is nearer two observations than forty at a one-year horizon."""
    _seed_cohort(memory, n_symbols=2, bars_per_symbol=20, series_len=300)
    near = memory.similar_setups("ZZZ", _summary(rsi=32.0))["analogues"]
    assert near["n"] == 40 and near["n_symbols"] == 2
    assert near["vs_benchmark_252d"]["n"] == 40
    assert near["vs_benchmark_252d"]["n_symbols"] == 2


def test_the_long_horizon_cohort_is_drawn_from_rows_that_carry_the_mark(memory):
    """The nearest rows are usually the newest — the ones without a long outcome. Taking the
    20-day window's rows for the year block would report nothing for a year."""
    # 40 fresh rows nearest to the query, graded at 20d only...
    for b in range(40):
        _record(memory, "NEW", b, rsi=32.0 + b * 0.01)
    dates, closes, bench = _series(70)
    memory.score_symbol("NEW", dates, closes, bench_dates=dates, bench_closes=bench)
    # ...and 6 older rows a little further away that DO carry the one-year mark.
    for b in range(6):
        _record(memory, "OLD", b, rsi=34.0 + b * 0.01)
    dates, closes, bench = _series(300)
    memory.score_symbol("OLD", dates, closes, bench_dates=dates, bench_closes=bench)

    near = memory.similar_setups("ZZZ", _summary(rsi=32.0))["analogues"]
    assert near["n"] == 40 and near["n_symbols"] == 1
    assert near["vs_benchmark_252d"]["n"] == 6
    assert near["vs_benchmark_252d"]["n_symbols"] == 1


def test_the_sql_prefilter_removes_only_what_the_python_veto_would(memory):
    """The prefilter exists for speed; it must not change who counts as a neighbour."""
    feats = memory.features_from_summary(_summary(rsi=32.0))
    where, params = memory._veto_sql(feats)
    assert where.count("AND (") == 5, "one predicate per veto axis the query has"
    # An axis the query lacks contributes nothing — the Python check skips it too.
    feats["rsi14"] = None
    where2, params2 = memory._veto_sql(feats)
    assert where2.count("AND (") == 4 and len(params2) == 8
    # A row NULL on an axis is kept, matching `_distance`'s skip.
    assert "IS NULL OR" in where


def test_stats_keeps_every_twenty_day_key_and_adds_the_horizons(memory):
    """The app's "Is it any good?" card reads the 20-day keys by name; those cannot move."""
    from app.analyst import Signal, Verdict

    v = Verdict(signal=Signal.buy, conviction=70, thesis="t", rationale=[], key_risks=[],
                invalidation="x", horizon="2w", catalysts=[]).model_dump()
    for b in range(8):
        memory.record_verdict(symbol="AAPL", summary=_summary(date=_dates(400)[b]), verdict=v,
                              origin="model")
    dates, closes, bench = _series(300)
    memory.score_symbol("AAPL", dates, closes, bench_dates=dates, bench_closes=bench)

    st = memory.stats()
    card = st["buy_calls"]
    assert {"n", "avg_fwd_20d_pct", "avg_excess_20d_pct", "correct_rate_20d"} <= set(card)
    assert card["n"] == 8 and card["n_symbols"] == 1
    assert set(card["at_252d"]) == {"n", "n_symbols", "avg_fwd_252d_pct", "avg_excess_252d_pct",
                                    "correct_rate_252d"}
    assert st["horizons"]["252d"]["scored"] == 8
    assert st["horizons"]["252d"]["oldest_pending_age_days"] is None
    assert st["retention_days"] == memory.retention_days()


def test_stats_gives_the_sandbox_cards_no_long_horizon_rates(memory):
    """Own-decision outcomes are dollars in the ledger cost block, never a rate at any horizon."""
    for b in range(6):
        _record(memory, "AAPL", b, origin="sandbox", signal="buy")
    dates, closes, bench = _series(300)
    memory.score_symbol("AAPL", dates, closes, bench_dates=dates, bench_closes=bench)
    card = memory.stats()["sandbox_buys"]
    assert card["n"] == 6
    assert "correct_rate_20d" in card, "the existing 20-day card is unchanged"
    assert not any(k.startswith("at_") for k in card)


def test_stats_reports_a_row_nothing_can_grade_as_pending_not_as_zero(memory):
    """A row whose bar is not in any series stays pending; its age is shown, not hidden."""
    _record(memory, "GONE", 0, ts=time.time() - 900 * 86_400)
    st = memory.stats()
    assert st["horizons"]["20d"]["scored"] == 0
    assert st["horizons"]["20d"]["oldest_pending_age_days"] == pytest.approx(900, abs=1)
    assert "buy_calls" not in st
