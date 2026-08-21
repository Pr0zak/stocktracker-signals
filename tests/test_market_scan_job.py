"""The nightly market scan — the job that must survive 3,200 chances to die.

Three incidents shape every test here.

The first is the merged error counter. /screener/value and /smart_money both learned the hard way
that "we looked at 3,200 names and 2,900 came back" is a different statement from "2,900 came back
and 300 were too young to measure": the first is a data-source problem and the second is a fact
about the market. A single `errors` number collapses them, so `scanned`, `fetch_failed` and
`too_short` are kept apart and are asserted here to PARTITION the symbols attempted exactly — no
double counting, no symbol falling through a crack, on any path including a mid-run exception.

The second is `memory.prune()`, which is called only from the tail of `score_memory()` — a function
that returns early at `if not syms: return 0`. The nightly prune therefore does not run on any night
with nothing to score, and nobody noticed because the failure is invisible until the disk fills.
`test_prune_runs_even_when_the_run_stored_nothing` and its refusal twin pin the fix: retention runs
on EVERY exit path, including the ones that did no work at all.

The third is the concurrency ceiling. The only measured burst against Yahoo's chart endpoint from
this box is 400 symbols at concurrency 6 with zero non-200s; this job is ~8x that scale against an
unofficial endpoint. The gate therefore starts BELOW the measured point, earns each extra permit
over ~250 clean completions, and halves on a cluster of throttle signals — all of which is asserted
directly rather than assumed, because a gate that silently stopped backing off would look exactly
like a gate that never needed to.

Isolation is `SIGNALS_DATA_DIR` plus `importlib.reload()`: both `scan_store` and `market_scan_job`
bind their data directory at import time, so setting the env var without reloading writes into the
previous test's database.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import importlib
import json
import time

import pytest


@pytest.fixture()
def job(tmp_path, monkeypatch):
    """The job module and its store, rooted at a throwaway data dir."""
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    from app import scan_store as ss

    importlib.reload(ss)
    from app import market_scan_job as mj

    importlib.reload(mj)
    yield mj
    if ss._conn is not None:
        ss._conn.close()
        ss._conn = None


def _series(symbol: str, n: int = 300):
    """A clean rising ramp with full OHLC — enough bars for every metric in the contract."""
    from app.market import Series

    closes = [100.0 + 0.25 * i for i in range(n)]
    base = dt.date(2025, 1, 1)
    return Series(
        symbol=symbol,
        closes=closes,
        opens=list(closes),
        volumes=[1_000_000.0] * n,
        dates=[(base + dt.timedelta(days=i)).strftime("%Y%m%d") for i in range(n)],
        fifty_two_high=max(closes),
        fifty_two_low=min(closes),
        currency="USD",
        highs=[c * 1.01 for c in closes],
        lows=[c * 0.99 for c in closes],
    )


def _universe(mp, mj, symbols: list[str] | None, *, built_at: float | None = None) -> None:
    """Stand in for a published universe blob. `None` means nothing has ever been built.

    Patched through monkeypatch, never by assignment: `app.universe` is NOT reloaded by the fixture,
    so a bare `universe.load = ...` would outlive this test and silently redefine the universe for
    every module that runs after it.
    """
    blob = None if symbols is None else {
        "built_at": time.time() if built_at is None else built_at, "symbols": list(symbols),
    }
    mp.setattr(mj.universe, "load", lambda: blob)


def _fetch(mp, mj, *, bad: set[str] = frozenset(), short: dict[str, int] | None = None, calls=None):
    """Install a fake fetch_series. `bad` raises, `short` returns a stub with too few bars."""
    short = short or {}

    async def fake(client, symbol, rng="1y", *, fallback=True):
        if calls is not None:
            calls.append({"symbol": symbol, "rng": rng, "fallback": fallback})
        if symbol in bad:
            raise RuntimeError(f"Yahoo chart fetch failed for {symbol}: 404")
        return _series(symbol, n=short.get(symbol, 300))

    mp.setattr(mj, "fetch_series", fake)
    return fake


def _run(mj, **kw) -> dict:
    return asyncio.run(mj.run_market_scan(**kw))


# ---- the three counters

def test_a_symbol_whose_fetch_raises_is_counted_and_does_not_sink_the_run(job, monkeypatch):
    """One dead ticker out of thousands must cost exactly one row, never the night."""
    _universe(monkeypatch, job, ["AAA", "BBB", "CCC"])
    _fetch(monkeypatch, job, bad={"BBB"})

    out = _run(job)

    assert out["status"] == "ok"
    assert out["scanned"] == 2
    assert out["fetch_failed"] == 1
    assert out["too_short"] == 0
    assert out["rows_written"] == 2
    assert [e["symbol"] for e in out["errors"]] == ["BBB"]
    assert out["errors"][0]["stage"] == "fetch"


def test_a_symbol_with_too_little_history_is_counted_apart_from_a_fetch_failure(job, monkeypatch):
    """A three-week-old IPO is not a broken data source, and the summary must not say it is."""
    _universe(monkeypatch, job, ["AAA", "IPO", "BAD"])
    _fetch(monkeypatch, job, bad={"BAD"}, short={"IPO": 12})

    out = _run(job)

    assert out["too_short"] == 1
    assert out["fetch_failed"] == 1
    assert out["scanned"] == 1
    stages = {e["symbol"]: e["stage"] for e in out["errors"]}
    assert stages == {"IPO": "history", "BAD": "fetch"}


def test_the_three_counters_partition_every_symbol_attempted(job, monkeypatch):
    """scanned + fetch_failed + too_short == attempted, exactly. No overlap, no gap."""
    syms = [f"S{i}" for i in range(12)]
    _universe(monkeypatch, job, syms)
    _fetch(monkeypatch, job, bad={"S1", "S4", "S9"}, short={"S2": 5, "S7": 49})

    out = _run(job)

    assert out["attempted"] == 12
    assert out["scanned"] + out["fetch_failed"] + out["too_short"] == out["attempted"]
    assert (out["scanned"], out["fetch_failed"], out["too_short"]) == (7, 3, 2)


def test_a_symbol_that_fails_while_being_measured_still_lands_in_exactly_one_counter(job, monkeypatch):
    """swing.metrics() should never raise — but if it does, the partition must still hold, and
    `stage` (not the counter) is what records that the failure was in the measurement."""
    _universe(monkeypatch, job, ["AAA", "BOOM"])
    _fetch(monkeypatch, job)

    def explode(series, *, bench_closes=None):
        if series.symbol == "BOOM":
            raise ZeroDivisionError("synthetic")
        return real_metrics(series, bench_closes=bench_closes)

    real_metrics = job.swing.metrics
    monkeypatch.setattr(job.swing, "metrics", explode)
    out = _run(job)

    assert out["scanned"] + out["fetch_failed"] + out["too_short"] == out["attempted"] == 2
    assert out["fetch_failed"] == 1
    assert [e["stage"] for e in out["errors"]] == ["measure"]


def test_a_symbol_repeated_in_the_universe_is_only_attempted_once(job, monkeypatch):
    """A duplicate would be counted twice in `scanned` and then collapse to one stored row, leaving
    two numbers that disagree with no way to explain the difference."""
    _universe(monkeypatch, job, ["AAA", "AAA", "BBB"])
    _fetch(monkeypatch, job)

    out = _run(job)

    assert out["attempted"] == 2
    assert out["scanned"] == out["rows_written"] == 2


# ---- what actually gets fetched

def test_bars_are_fetched_over_two_years_with_the_webull_fallback_disabled(job, monkeypatch):
    """`1y` leaves a 200-day average with no history behind it, and the Webull fallback burns ~57s
    per failing symbol against an unofficial endpoint — at this scale that is an outage, not a
    rescue."""
    calls: list[dict] = []
    _universe(monkeypatch, job, ["AAA", "BBB"])
    _fetch(monkeypatch, job, calls=calls)

    _run(job)

    assert calls, "fetch_series was never called"
    for c in calls:
        assert c["rng"] == "2y"
        assert c["fallback"] is False
    assert calls[0]["symbol"] == "SPY"      # the benchmark is fetched on the same terms


def test_an_absent_benchmark_leaves_relative_strength_unmeasured_rather_than_zero(job, monkeypatch):
    """No SPY means we could not measure relative strength. It must not read as 'in line with the
    market', which is what a 0.0 there would say."""
    _universe(monkeypatch, job, ["AAA"])
    _fetch(monkeypatch, job, bad={"SPY"})

    out = _run(job)

    assert out["benchmark"] is None
    assert out["scanned"] == 1               # the run continues without it
    assert out["fetch_failed"] == 0          # SPY is not one of the universe symbols
    row = job.scan_store.symbol_row("AAA")
    assert row["rel_strength_3mo"] is None
    assert "rel_strength_3mo" in row["unmeasured"]


# ---- storage

def test_rows_land_in_the_store_and_breadth_reads_them_back(job, monkeypatch):
    """The whole point of the night: a cross-section that a later reader can ask questions of."""
    _universe(monkeypatch, job, ["AAA", "BBB", "CCC"])
    _fetch(monkeypatch, job)

    out = _run(job)

    assert job.scan_store.latest_date() == out["session"]
    breadth = job.scan_store.breadth()
    assert breadth["available"] is True
    assert breadth["n"] == 3
    assert breadth["pct_above_sma50"] == 100.0     # a rising ramp is above every average it has
    assert [r["symbol"] for r in job.scan_store.cross_section()] == ["AAA", "BBB", "CCC"]


def test_breadth_is_unavailable_when_nothing_could_be_measured(job, monkeypatch):
    """Every symbol failing is not a bearish market reading; it is no reading at all."""
    _universe(monkeypatch, job, ["AAA", "BBB"])
    _fetch(monkeypatch, job, bad={"AAA", "BBB"})

    out = _run(job)

    assert out["status"] == "empty"
    assert out["rows_written"] == 0
    breadth = job.scan_store.breadth()
    assert breadth["available"] is False
    assert breadth["pct_above_sma50"] is None


# ---- re-running a night

def test_a_session_already_stored_is_skipped_unless_forced(job, monkeypatch):
    """The timer can fire twice; a re-run is a decision, not an accident."""
    _universe(monkeypatch, job, ["AAA", "BBB"])
    calls: list[dict] = []
    _fetch(monkeypatch, job, calls=calls)

    first = _run(job)
    assert first["status"] == "ok"
    after_first = len(calls)

    second = _run(job)
    assert second["status"] == "skipped"
    assert second["scanned"] is None            # not 0 — we did not measure anything, and say so
    assert len(calls) == after_first            # nothing was fetched

    third = _run(job, force=True)
    assert third["status"] == "ok"
    assert third["scanned"] == 2
    assert len(calls) > after_first
    # Re-measuring a night upserts; it must not duplicate the cross-section.
    assert job.scan_store.breadth()["n"] == 2


def test_limit_takes_the_head_of_the_universe(job, monkeypatch):
    """`--limit 400` is the only slice anyone has measured against Yahoo, and the universe is sorted
    cap-descending, so the head is the meaningful end to take."""
    _universe(monkeypatch, job, [f"S{i}" for i in range(20)])
    calls: list[dict] = []
    _fetch(monkeypatch, job, calls=calls)

    out = _run(job, limit=5)

    assert out["attempted"] == 5
    fetched = [c["symbol"] for c in calls if c["symbol"] != "SPY"]
    assert fetched == ["S0", "S1", "S2", "S3", "S4"]


# ---- refusals

def test_an_absent_universe_is_refused_rather_than_rebuilt_inline(job, monkeypatch):
    """Rebuilding here would couple a 3,200-symbol scan to a 5,600-symbol directory fetch with its
    own coverage gate; a half-fetched universe published at 2am would be scanned as the market."""
    _universe(monkeypatch, job, None)
    _fetch(monkeypatch, job)

    out = _run(job)

    assert out["status"] == "refused"
    assert "universe" in out["reason"]
    assert out["scanned"] is None and out["attempted"] is None


def test_a_stale_universe_is_refused_and_force_does_not_wave_it_through(job, monkeypatch):
    """`force` re-runs tonight; it does not authorise measuring last month's symbol list."""
    _universe(monkeypatch, job, ["AAA"], built_at=time.time() - 30 * 86_400)
    _fetch(monkeypatch, job)

    for kw in ({}, {"force": True}):
        out = _run(job, **kw)
        assert out["status"] == "refused"
        assert "stale" in out["reason"]
        assert out["universe_symbols"] == 1     # we could read it; we refused to use it


# ---- housekeeping

def test_prune_runs_even_when_the_run_stored_nothing(job, monkeypatch):
    """The memory.prune() lesson: retention must not be reachable only through the success path."""
    seen: list[int] = []
    monkeypatch.setattr(job.scan_store, "prune", lambda *a, **k: seen.append(1) or 7)
    _universe(monkeypatch, job, ["AAA", "BBB"])
    _fetch(monkeypatch, job, bad={"AAA", "BBB"})

    out = _run(job)

    assert out["rows_written"] == 0
    assert len(seen) == 1
    assert out["pruned"] == 7


def test_prune_runs_on_a_refusal_and_on_a_skip_too(job, monkeypatch):
    """A month of refusals must still age out a quarter-old cross-section."""
    seen: list[str] = []
    monkeypatch.setattr(job.scan_store, "prune", lambda *a, **k: seen.append("p") or 0)

    _universe(monkeypatch, job, None)
    _fetch(monkeypatch, job)
    assert _run(job)["status"] == "refused"

    _universe(monkeypatch, job, ["AAA"])
    monkeypatch.setattr(job.scan_store, "latest_date",
                        lambda: job.session_date().strftime("%Y%m%d"))
    assert _run(job)["status"] == "skipped"

    assert len(seen) == 2


def test_a_failing_prune_does_not_discard_a_completed_scan(job, monkeypatch):
    """The rows are already committed; a housekeeping error must not lose the summary of the run."""
    def boom(*a, **k):
        raise OSError("disk gone")

    monkeypatch.setattr(job.scan_store, "prune", boom)
    _universe(monkeypatch, job, ["AAA"])
    _fetch(monkeypatch, job)

    out = _run(job)

    assert out["status"] == "ok"
    assert out["rows_written"] == 1
    assert out["pruned"] is None          # None, not 0: the prune did not report, it failed


# ---- the summary artifact

def test_the_summary_is_published_atomically_and_parses(job, monkeypatch):
    _universe(monkeypatch, job, ["AAA", "BBB"])
    _fetch(monkeypatch, job)

    out = _run(job)

    assert job.LATEST.exists()
    on_disk = json.loads(job.LATEST.read_text())
    assert on_disk["session"] == out["session"]
    assert on_disk["scanned"] == 2
    # No temp files left behind by the unique-temp + os.replace publish.
    assert [p.name for p in job.LATEST.parent.glob("*.tmp")] == []


def test_a_refusal_overwrites_the_summary_so_a_dead_job_cannot_look_healthy(job, monkeypatch):
    """Leaving last week's success on disk while the job refuses nightly is how a broken job keeps
    reporting a green dashboard."""
    _universe(monkeypatch, job, ["AAA"])
    _fetch(monkeypatch, job)
    _run(job)
    assert json.loads(job.LATEST.read_text())["status"] == "ok"

    _universe(monkeypatch, job, None)
    _run(job)

    blob = json.loads(job.LATEST.read_text())
    assert blob["status"] == "refused"
    assert blob["scanned"] is None


# ---- the session the night is filed under

def test_the_night_is_the_last_completed_session_not_the_calendar_date(job):
    """The timer fires at 06:30 CT, before the open — 'today' has no bars yet."""
    tue_premarket = dt.datetime(2026, 8, 18, 7, 30, tzinfo=dt.timezone.utc)   # 03:30 ET Tuesday
    assert job.session_date(tue_premarket) == dt.date(2026, 8, 17)

    tue_evening = dt.datetime(2026, 8, 18, 22, 30, tzinfo=dt.timezone.utc)    # 18:30 ET Tuesday
    assert job.session_date(tue_evening) == dt.date(2026, 8, 18)


def test_a_holiday_reports_the_previous_session_rather_than_a_night_of_its_own(job):
    """Independence Day 2026 falls on a Saturday and is observed on Friday July 3. A scan run that
    weekend describes Thursday's session — the same one the previous run stored, which is what makes
    the 'already stored' gate the holiday rule."""
    sat = dt.datetime(2026, 7, 4, 20, 0, tzinfo=dt.timezone.utc)
    assert job.market_calendar.is_market_holiday(dt.date(2026, 7, 3))
    assert job.session_date(sat) == dt.date(2026, 7, 2)


# ---- the ramping gate

def _gate(job, **kw):
    return job._AdaptiveGate(**kw)


def test_the_gate_starts_below_the_only_concurrency_ever_measured(job):
    """6 was measured at 400 symbols. This job is ~8x that, so it starts at 4 and earns the rest."""
    assert job._CONCURRENCY_START == 4
    assert job._CONCURRENCY_START < 6
    assert _gate(job).limit == 4


def test_the_gate_widens_one_permit_at_a_time_and_stops_at_the_ceiling(job):
    async def drive():
        g = _gate(job, start=4, ceiling=6, widen_every=10)
        for _ in range(200):
            await g.record(throttled=False)
        return g

    g = asyncio.run(drive())
    assert g.limit == 6            # ceiling, not 24
    assert g.widenings == 2


def test_a_cluster_of_throttle_signals_halves_the_gate(job):
    async def drive():
        g = _gate(job, start=8, floor=2, cluster=3, window=60.0)
        now = 1000.0
        for i in range(3):
            await g.record(throttled=True, now=now + i)
        return g

    g = asyncio.run(drive())
    assert g.limit == 4
    assert g.backoffs == 1


def test_the_gate_never_falls_below_its_floor(job):
    async def drive():
        g = _gate(job, start=8, floor=2, cluster=2, window=60.0)
        for i in range(40):
            await g.record(throttled=True, now=1000.0 + i)
        return g

    g = asyncio.run(drive())
    assert g.limit == 2


def test_isolated_failures_spread_over_time_do_not_back_the_gate_off(job):
    """A handful of delisted tickers across a whole run is not the endpoint pushing back."""
    async def drive():
        g = _gate(job, start=6, cluster=3, window=60.0)
        for i in range(10):
            await g.record(throttled=True, now=1000.0 + i * 120)   # two minutes apart
        return g

    g = asyncio.run(drive())
    assert g.limit == 6
    assert g.backoffs == 0
    assert g.throttled == 10       # counted and reported, just not acted on


def test_a_backoff_resets_the_widen_clock(job):
    """Widening straight after a backoff would walk back into the rate limiter it just escaped."""
    async def drive():
        g = _gate(job, start=8, floor=2, widen_every=10, cluster=2, window=60.0)
        for i in range(2):
            await g.record(throttled=True, now=1000.0 + i)
        assert g.limit == 4
        for _ in range(9):
            await g.record(throttled=False)
        return g

    g = asyncio.run(drive())
    assert g.limit == 4            # 9 clean completions since the backoff is not yet 10


def test_the_gate_never_lets_more_than_its_limit_run_at_once(job):
    """The bound is the point. An unbounded gather against an unofficial endpoint is the hazard."""
    async def drive():
        g = _gate(job, start=3, ceiling=3)
        peak = 0
        live = 0

        async def one():
            nonlocal peak, live
            async with g:
                live += 1
                peak = max(peak, live)
                await asyncio.sleep(0)
                live -= 1
        await asyncio.gather(*[one() for _ in range(50)])
        return peak

    assert asyncio.run(drive()) == 3


def test_a_shrunken_gate_still_drains_its_waiters(job):
    """Halving the limit while tasks are queued must not strand them — the run would hang forever
    and a hung nightly job is indistinguishable from a slow one."""
    async def drive():
        g = _gate(job, start=8, floor=2, cluster=1, window=60.0)
        done = 0

        async def one(i: int):
            nonlocal done
            async with g:
                await asyncio.sleep(0)
                await g.record(throttled=(i < 1))
                done += 1
        await asyncio.wait_for(asyncio.gather(*[one(i) for i in range(30)]), timeout=5)
        return done, g.limit

    done, limit = asyncio.run(drive())
    assert done == 30
    assert limit == 4


# ---- throttle classification

def test_a_slow_failure_is_read_as_a_timeout_whatever_its_message_says(job):
    """market._fetch_chart raises OUTSIDE its except block, so there is no cause chain to walk and
    str(httpx.ReadTimeout()) is routinely empty. The wall clock is the reliable witness."""
    assert job._is_throttle(RuntimeError("Yahoo chart fetch failed for X: "), 31.0) is True
    assert job._is_throttle(RuntimeError("Yahoo chart fetch failed for X: 404"), 0.2) is False


def test_a_429_shrinks_the_gate_but_a_dead_ticker_does_not(job):
    assert job._is_throttle(RuntimeError("... 429 Too Many Requests"), 0.1) is True
    assert job._is_throttle(ValueError("No data found, symbol may be delisted"), 0.1) is False


# ---- the printed line

def test_the_printed_summary_omits_counters_that_were_never_measured(job, monkeypatch):
    """`scanned None / failed None / short None of None` is not honesty, it is noise. A path that
    attempted nothing says so by leaving the clause out; the JSON artifact keeps the nulls."""
    _universe(monkeypatch, job, None)
    _fetch(monkeypatch, job)

    line = job.summary_line(_run(job))

    assert line.startswith("market scan refused")
    assert "None" not in line
    assert "pruned 0" in line


def test_the_printed_summary_reports_all_four_counters_on_a_real_run(job, monkeypatch):
    """The one line that reaches the journal has to carry the distinction the counters exist for.

    `suspect` joined the line when the integrity gate was added: a night where Yahoo starts serving
    mixed pre/post-split bars across a whole exchange would otherwise look, from the journal, like a
    night where those names simply were not in the universe.
    """
    _universe(monkeypatch, job, ["AAA", "BBB", "IPO", "BAD"])
    _fetch(monkeypatch, job, bad={"BAD"}, short={"IPO": 9})

    line = job.summary_line(_run(job))

    assert "scanned 2 / failed 1 / short 1 / suspect 0 of 4" in line
    assert "stored 2" in line
    assert "concurrency 4" in line


def test_a_corrupt_universe_blob_is_a_stated_refusal_not_a_traceback(job, monkeypatch):
    """universe.load() hands back whatever JSON is on disk, so a mangled `built_at` reaches
    is_stale() as a string and raises there. An unattended job that dies on that writes no summary
    at all, which reads from the dashboard exactly like a job that never ran."""
    monkeypatch.setattr(job.universe, "load",
                        lambda: {"built_at": "not-a-number", "symbols": ["AAA"]})
    _fetch(monkeypatch, job)

    out = _run(job)

    assert out["status"] == "refused"
    assert "unreadable" in out["reason"]
    assert job.LATEST.exists()
