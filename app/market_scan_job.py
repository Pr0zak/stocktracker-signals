"""
SWT-1 step 5 — the nightly, LLM-FREE cross-section of the whole liquid market.

This is the job that turns the previous four pieces into one artifact a night: `universe.load()`
supplies the liquidity-gated symbol list, `market.fetch_series` supplies the bars,
`swing.metrics()` measures each name, and `scan_store.insert_night()` stores the result. There is
no model call anywhere in this file and there must never be one: ~3,200 symbols x one analyst call
is a three-figure nightly bill for numbers that are a deterministic function of the bars. The LLM's
turn comes later, over the handful of names this scan ranks to the top.

What this module actually protects
----------------------------------
1. **It never rebuilds the universe inline.** A stale or absent universe is a REFUSAL with a reason,
   not a silent rebuild. Rebuilding here would couple a 3,200-symbol scan to a 5,600-symbol
   directory fetch that has its own coverage gate (`universe.publish`), and a half-fetched universe
   published at 2am would then be scanned as though it were the market. `scan_job.py` owns the
   rebuild hook; this job consumes what that hook published and says so when there is nothing good
   to consume.

2. **Four counters that are never merged.** `scanned` / `fetch_failed` / `too_short` /
   `suspect_series` partition every symbol attempted. "We measured 2,900 names" and "we measured
   2,900 and 300 more never answered" are different statements about the market, and a single
   `errors` count collapses them. This is the same house rule /screener/value and /smart_money
   already follow. `suspect_series` is the fourth because it is a fourth distinct fact: the fetch
   worked and the history was long enough, and the data is STILL unusable because Yahoo served a
   mixed pre/post-split basis. Folding that into `fetch_failed` would file a data-quality
   regression under "network trouble", where nobody would look for it.

3. **Housekeeping that is not coupled to work.** `scan_store.prune()` runs at the tail of EVERY
   exit path, including the refusals and the "already scanned tonight" skip. `memory.prune()` is the
   cautionary tale: it is called only from the tail of `score_memory()`, which returns early at
   `if not syms: return 0`, so the prune that was supposed to run nightly does not run on any night
   with nothing to score. Retention must not depend on the run having found something to do.

4. **A concurrency ceiling that is measured, not assumed.** See `_AdaptiveGate`.

Run it standalone: `python -m app.market_scan_job --limit 400`. Nothing here is wired into main.py
yet — routes are a later step of SWT-1.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import os
import threading
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

from . import market_calendar, scan_store, swing, universe
from .market import fetch_series

log = logging.getLogger("signals.market_scan")

_DATA_DIR = Path(os.environ.get("SIGNALS_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))

# The run summary, for app/observability.py to read later (it is not wired up yet — this step adds
# no routes). Rewritten on EVERY exit path, including refusals: leaving last week's success in place
# while the job refuses nightly is precisely how a dead job looks healthy from the dashboard.
LATEST = _DATA_DIR / "market_scan_latest.json"

_ET = ZoneInfo("America/New_York")

# 2y of daily bars, NOT market.fetch_series's "1y" default. `sma(closes, 200)` over ~250 trading
# days leaves 50 bars of headroom — not enough for a 200-day slope, not enough for any cross history
# ("how long has it been above the 200?"), and one holiday-thinned year away from not seeding at all.
# The cost is real and stated: it roughly DOUBLES the nightly wire bytes (~250 -> ~500 bars per
# symbol, i.e. very roughly 60MB -> 120MB across a 3,200-name universe). Worth it — the alternative
# is a 200-day average that exists but has no history behind it.
_RANGE = "2y"

# Below this, a row is not worth storing. 50 closes is the shallowest STRUCTURAL read the contract
# has: it is exactly what `above_sma50` needs, and a row that cannot answer even that contributes
# nothing to the cross-section while still counting toward breadth's coverage denominator (see
# scan_store._MIN_COVERAGE). A recent IPO is therefore counted in `too_short` and named, never
# stored as a row full of Nones and never quietly dropped.
_MIN_BARS = 50

# The benchmark for rel_strength_3mo. SPY rather than ^GSPC: this scan ranks tradeable names against
# a tradeable benchmark, and SPY's bars come back on the same adjusted basis as everything else in
# the run. If it fails the whole run still proceeds — rel_strength_3mo lands in `unmeasured` for
# every row, and the summary says the benchmark was absent rather than reporting zeros.
_BENCHMARK = "SPY"

# How many failing symbols are written into the summary artifact. The counters are exact; this is
# the sample a human reads.
_MAX_ERRORS_REPORTED = 50


# ===================================================================================== the session

def session_date(now: dt.datetime | None = None) -> dt.date:
    """The most recent session whose bars are FINAL — the night this scan describes.

    Not `date.today()`. The nightly timer fires at 06:30 America/Chicago, i.e. BEFORE the open, so
    "today" has no bars at all and Yahoo's last bar is yesterday's. Stamping the cross-section with
    today's date would file every night's data under a session it did not measure, and
    `scan_store.insert_night` explicitly refuses to let a row disagree with its batch's date — so
    the batch's date had better be right.

    Rolling back through weekends and holidays is also what makes the market-holiday rule work
    without a separate check: on a closed day this returns the SAME session as the previous run, and
    the "already scanned" gate below then skips it. A holiday is not a night with no data; it is a
    night whose data is the previous session's, already stored.
    """
    et = (now or dt.datetime.now(dt.timezone.utc)).astimezone(_ET)
    d = et.date()
    if market_calendar.is_trading_day(d):
        close_s, _after = market_calendar.session_end_seconds(d)   # honours the 1pm half-days
        if et.hour * 3600 + et.minute * 60 + et.second >= close_s:
            return d
    for _ in range(10):        # a 10-day walk clears any weekend + holiday cluster
        d -= dt.timedelta(days=1)
        if market_calendar.is_trading_day(d):
            return d
    return d


# ============================================================== bounded, ramping, backing-off gate

# The only concurrency figure anyone has actually MEASURED against Yahoo's chart endpoint from this
# box is 400 symbols at concurrency 6 with zero non-200 responses. This job is ~8x that scale
# against an endpoint that is unofficial, undocumented and free to start returning 429s at any
# volume it likes. Two conclusions follow, and both are the reason this is not a plain
# `asyncio.Semaphore(6)`:
#
#   * START BELOW THE MEASURED POINT. 6 was measured for 400 symbols; nothing was measured for
#     3,200. 4 is a deliberate step back from the only evidence we have, because the failure mode of
#     guessing high is a rate-limit ban that costs the whole night, while the cost of guessing low is
#     a few extra minutes of wall clock on an unattended job.
#   * EARN THE WIDENING, DO NOT ASSUME IT. +1 permit per ~250 clean completions, ceiling 10. 250 is
#     ~8% of the universe: long enough that a rate limiter with a per-minute window has had time to
#     react, short enough that the run still reaches the ceiling around the halfway mark.
#
# And back off HARD, not gently: 3 throttle signals inside 60s HALVES the limit (floor 2). Halving
# rather than -1 because a rate limiter is a cliff, not a slope — once it engages, stepping down one
# permit at a time just collects more 429s on the way down. The widen clock resets on a backoff, so
# a full 250 clean completions must pass before we try widening again.
_CONCURRENCY_START = 4
_CONCURRENCY_MAX = 10
_CONCURRENCY_MIN = 2
_WIDEN_EVERY = 250
_BACKOFF_CLUSTER = 3
_BACKOFF_WINDOW_S = 60.0


class _AdaptiveGate:
    """A semaphore whose limit moves. Bounded at both ends, never unbounded.

    `asyncio.Semaphore` cannot be resized after construction, which is the whole requirement here,
    so this is a Condition with an explicit active-count. Single event loop, so the counters need no
    lock of their own; the Condition's lock exists to make `wait_for` correct, not to protect them.
    """

    def __init__(self, *, start: int = _CONCURRENCY_START, ceiling: int = _CONCURRENCY_MAX,
                 floor: int = _CONCURRENCY_MIN, widen_every: int = _WIDEN_EVERY,
                 cluster: int = _BACKOFF_CLUSTER, window: float = _BACKOFF_WINDOW_S) -> None:
        self.limit = max(1, int(start))
        self.ceiling = max(self.limit, int(ceiling))
        self.floor = max(1, min(int(floor), self.limit))
        self.widen_every = max(1, int(widen_every))
        self.cluster = max(1, int(cluster))
        self.window = float(window)
        self.completed = 0
        self.backoffs = 0
        self.widenings = 0
        self.throttled = 0
        self.high_water = self.limit
        self._active = 0
        self._last_widen_at = 0
        self._events: list[float] = []
        self._cond = asyncio.Condition()

    async def __aenter__(self) -> "_AdaptiveGate":
        async with self._cond:
            await self._cond.wait_for(lambda: self._active < self.limit)
            self._active += 1
        return self

    async def __aexit__(self, *_exc: object) -> None:
        async with self._cond:
            self._active -= 1
            self._cond.notify()      # exactly one slot came free
        return None

    async def record(self, *, throttled: bool, now: float | None = None) -> None:
        """One symbol finished. Adjust the limit, then wake anyone the adjustment freed."""
        t = time.monotonic() if now is None else now
        widened = False
        async with self._cond:
            self.completed += 1
            if throttled:
                self.throttled += 1
                self._events.append(t)
                self._events = [x for x in self._events if t - x <= self.window]
                if len(self._events) >= self.cluster:
                    # Clear the window on a backoff. Without it the same cluster keeps re-firing on
                    # every subsequent failure and walks the limit straight down to the floor on
                    # what was one incident.
                    self._events.clear()
                    new = max(self.floor, self.limit // 2)
                    if new < self.limit:
                        log.warning("market scan: throttled — concurrency %d -> %d", self.limit, new)
                        self.limit = new
                        self.backoffs += 1
                    self._last_widen_at = self.completed
                    return
            if self.completed - self._last_widen_at >= self.widen_every and self.limit < self.ceiling:
                self.limit += 1
                self.widenings += 1
                self.high_water = max(self.high_water, self.limit)
                self._last_widen_at = self.completed
                widened = True
            if widened:
                self._cond.notify_all()

    def snapshot(self) -> dict:
        return {
            "start": _CONCURRENCY_START, "final": self.limit, "high_water": self.high_water,
            "ceiling": self.ceiling, "floor": self.floor,
            "widenings": self.widenings, "backoffs": self.backoffs,
            "throttle_signals": self.throttled,
        }


# Two hosts x a 15s httpx timeout inside market._fetch_chart, so a genuine double timeout takes ~30s and
# a single-host one ~15s. Anything that failed after this long failed on the clock.
_TIMEOUT_HINT_S = 14.0
_THROTTLE_MARKERS = ("429", "too many requests", "rate limit", "timeout", "timed out", "503")


def _is_throttle(exc: BaseException, elapsed: float) -> bool:
    """Did this failure look like the endpoint pushing back, rather than the symbol being bad?

    A delisted ticker 404s instantly and must NOT shrink the concurrency; a 429 or a stalled read
    must. Classification is deliberately belt-and-braces because the evidence is thin:
    `market._fetch_chart` raises its RuntimeError OUTSIDE the except block, so there is no
    `__cause__` chain to walk — only the formatted `{last_err}`, and `str(httpx.ReadTimeout())` is
    frequently the empty string. Hence the wall-clock test: a failure that took ~15s or ~30s hit the
    httpx timeout whatever its message says.
    """
    if elapsed >= _TIMEOUT_HINT_S:
        return True
    e: BaseException | None = exc
    parts: list[str] = []
    for _ in range(6):          # bounded walk — a cyclic __context__ must not hang the scan
        if e is None:
            break
        if isinstance(e, httpx.TimeoutException):
            return True
        if isinstance(e, httpx.HTTPStatusError) and e.response.status_code in (429, 503):
            return True
        parts.append(f"{type(e).__name__} {e}")
        e = e.__cause__ or e.__context__
    blob = " ".join(parts).lower()
    return any(m in blob for m in _THROTTLE_MARKERS)


# ============================================================================================ run

async def _fetch_benchmark(client: httpx.AsyncClient) -> list[float] | None:
    """SPY's closes, or None. None means rel_strength_3mo is UNMEASURED for the night — not zero."""
    try:
        s = await fetch_series(client, _BENCHMARK, rng=_RANGE, fallback=False)
    except Exception:  # noqa: BLE001 — one absent benchmark must not cost 3,200 measured rows; the
        # summary reports benchmark=None and every row's rel_strength_3mo lands in `unmeasured`.
        log.warning("market scan: benchmark %s unavailable — relative strength will be unmeasured",
                    _BENCHMARK, exc_info=True)
        return None
    return s.closes or None


def _universe_symbols(blob: dict, limit: int | None) -> list[str]:
    """The symbol list to scan, de-duplicated, order preserved.

    De-duplication is not cosmetic: a repeated symbol would be fetched twice, counted twice in
    `scanned`, and then collapse to ONE row by `ON CONFLICT(d, symbol)` — so `scanned` would exceed
    the rows actually stored and the two numbers would disagree with nobody able to say why.

    `limit` takes the HEAD of the list, and the head is meaningful: universe.build sorts
    cap-descending, so `--limit 400` is the 400 largest names — the same shape of slice as the only
    burst anyone has measured against Yahoo from this box.
    """
    seen: set[str] = set()
    out: list[str] = []
    for s in blob.get("symbols") or []:
        sym = str(s or "").strip().upper()
        if sym and sym not in seen:
            seen.add(sym)
            out.append(sym)
    if limit is not None and limit > 0:
        out = out[:limit]
    return out


async def run_market_scan(*, force: bool = False, limit: int | None = None) -> dict:
    """One nightly market-wide measurement pass. Never raises; always returns a summary dict.

    `force` re-runs a session that has already been stored (the rows are upserted, not duplicated).
    It deliberately does NOT override the universe gate: "the symbol list is a week stale" is not a
    condition a re-run flag should be able to wave through, because the resulting cross-section
    would be of last week's market and nothing in the stored row would say so.
    """
    t0 = time.monotonic()
    night = session_date()
    d = night.strftime("%Y%m%d")
    summary = _blank_summary(d, night)

    blob = universe.load()
    if not blob or not (blob.get("symbols") or []):
        return _finish(summary, status="refused", reason=(
            "no universe on disk — POST /universe/build (or the nightly scan_job hook) must publish "
            "one first; this job does not rebuild inline"))
    summary["universe_built_at"] = blob.get("built_at")
    summary["universe_symbols"] = len(blob.get("symbols") or [])
    try:
        stale = universe.is_stale(blob)
        age_d = (time.time() - float(blob.get("built_at") or 0)) / 86_400.0
    except Exception as e:  # noqa: BLE001 — universe.load() returns whatever JSON is on disk, so a
        # corrupt `built_at` reaches is_stale as a string and raises there. That must be a refusal
        # with a stated reason, not an unhandled traceback out of an unattended job that then writes
        # no summary at all and looks like it never ran.
        return _finish(summary, status="refused",
                       reason=f"universe blob is unreadable ({type(e).__name__}: {e}) — rebuild it")
    if stale:
        return _finish(summary, status="refused", reason=(
            f"universe is stale ({age_d:.1f} days old, limit {universe.STALE_AFTER_S / 86_400:.0f}) — "
            "refusing to scan last week's symbol list; rebuild it, then re-run"))

    stored = scan_store.latest_date()
    if stored == d and not force:
        return _finish(summary, status="skipped", reason=(
            f"session {d} is already stored — pass force=True to re-measure it"))

    symbols = _universe_symbols(blob, limit)
    if not symbols:
        return _finish(summary, status="refused", reason="universe contained no usable symbols")

    rows: list[dict] = []
    errors: list[dict] = []
    counts = {"scanned": 0, "fetch_failed": 0, "too_short": 0, "suspect_series": 0}
    unmeasured_tally: dict[str, int] = {}

    async with httpx.AsyncClient() as client:
        bench = await _fetch_benchmark(client)
        summary["benchmark"] = {"symbol": _BENCHMARK, "bars": len(bench)} if bench else None

        gate = _AdaptiveGate()

        async def one(sym: str) -> None:
            # Counter bookkeeping needs no lock: this is one event loop, and every mutation below
            # happens between awaits.
            async with gate:
                t = time.monotonic()
                try:
                    series = await fetch_series(client, sym, rng=_RANGE, fallback=False)
                except Exception as e:  # noqa: BLE001 — one dead ticker out of 3,200 must never
                    # sink the night. It is counted in `fetch_failed`, named in `errors`, and the
                    # run continues; a raise here would lose every row measured so far.
                    throttled = _is_throttle(e, time.monotonic() - t)
                    counts["fetch_failed"] += 1
                    errors.append({"symbol": sym, "stage": "fetch",
                                   "throttled": throttled, "error": f"{type(e).__name__}: {e}"[:200]})
                    await gate.record(throttled=throttled)
                    return
                await gate.record(throttled=False)

                if len(series.closes) < _MIN_BARS:
                    counts["too_short"] += 1
                    errors.append({"symbol": sym, "stage": "history", "throttled": False,
                                   "error": f"{len(series.closes)} bars, need {_MIN_BARS}"})
                    return
                # Reject a series before measuring it, not after. Yahoo serves some reverse-split
                # names as a mixture of pre- and post-split bars with no adjustment applied, and
                # metrics computed over that are not merely noisy — they are enormous, so they sort
                # to the top of every momentum ranking and skew every cross-sectional percentile
                # built on the night. See swing._MAX_BAR_RATIO for the measured threshold.
                #
                # This is a FOURTH counter rather than a bucket folded into fetch_failed, because it
                # is a fourth distinct fact: the fetch worked, the history was long enough, and the
                # data is still unusable. Merging it would hide a Yahoo data regression inside a
                # number everyone reads as "network trouble".
                jump = swing.implausible_jump(series.closes)
                if jump is not None:
                    counts["suspect_series"] += 1
                    errors.append({"symbol": sym, "stage": "integrity", "throttled": False,
                                   "error": f"{jump:.1f}x single-bar move — mixed split basis"})
                    return
                try:
                    m = swing.metrics(series, bench_closes=bench)
                except Exception as e:  # noqa: BLE001 — metrics() is pure and defensive, so this is
                    # a should-never-happen; if it ever does, it is one symbol's row, not the run.
                    # It counts as fetch_failed to keep the three counters a strict partition of the
                    # symbols attempted — `stage` in `errors` is what says it failed while MEASURING.
                    counts["fetch_failed"] += 1
                    errors.append({"symbol": sym, "stage": "measure", "throttled": False,
                                   "error": f"{type(e).__name__}: {e}"[:200]})
                    return
                if int(m.get("bars") or 0) < _MIN_BARS:
                    # Fetched enough bars, but enough of them were null/non-finite that the usable
                    # count fell under the floor. Same fact about the symbol, same counter.
                    counts["too_short"] += 1
                    errors.append({"symbol": sym, "stage": "history", "throttled": False,
                                   "error": f"{m.get('bars')} usable bars of {len(series.closes)}"})
                    return
                for k in m.get("unmeasured") or ():
                    unmeasured_tally[k] = unmeasured_tally.get(k, 0) + 1
                rows.append({"symbol": sym, **m})
                counts["scanned"] += 1

        await asyncio.gather(*[one(s) for s in symbols])

    summary.update(counts)
    summary["attempted"] = len(symbols)
    summary["concurrency"] = gate.snapshot()
    # The most-often-unmeasured keys, biggest first. When ATR or ADX comes back None across half the
    # universe this is the line that shows it, instead of it hiding inside per-row nulls nobody reads.
    summary["unmeasured_top"] = dict(sorted(unmeasured_tally.items(), key=lambda kv: -kv[1])[:8])
    summary["errors_total"] = len(errors)
    summary["errors"] = errors[:_MAX_ERRORS_REPORTED]

    # One batch, one transaction. insert_night swallows its own failures and returns 0, so
    # `rows_written` disagreeing with `scanned` is the signal that the store refused the night.
    summary["rows_written"] = scan_store.insert_night(rows, d=d) if rows else 0

    # Remove rows this night still holds that this run no longer produces — but ONLY when the run
    # covered the whole universe. insert_night upserts by design (a partial writer must not null the
    # columns it knows nothing about), and the cost of that is that "stopped being measurable" has no
    # way to mean "remove". It bit immediately: the integrity gate began rejecting twelve
    # mixed-split-basis names, a forced re-run stored 3,101 rows instead of 3,113, and all twelve
    # corrupt rows stayed exactly where they were — still at the top of every momentum ranking.
    #
    # `limit` is the guard and it is not optional. A `--limit 400` run retiring "everything it did
    # not produce" would delete the other 2,700 rows and leave a night's cross-section that is a
    # sample of itself, with breadth and percentiles computed off it as though it were the market.
    summary["retired"] = (
        scan_store.retire_absent(d, [r["symbol"] for r in rows])
        if (rows and limit is None) else 0
    )
    summary["duration_s"] = round(time.monotonic() - t0, 1)
    status = "ok" if rows else "empty"
    reason = None if rows else "every symbol failed or was too short — nothing was stored"
    return _finish(summary, status=status, reason=reason)


def _blank_summary(d: str, night: dt.date) -> dict:
    """The summary skeleton. Every counter starts as None, NOT 0.

    A refusal that reports `scanned: 0` reads off a dashboard as "we scanned the market and found
    nothing", which is a market claim. `None` says we did not measure, which is the truth on every
    path that exits before the fetch loop; `run_market_scan` overwrites them with real integers only
    once symbols have actually been attempted.
    """
    return {
        "job": "market_scan",
        "status": None,
        "reason": None,
        "generated_at": time.time(),
        "as_of": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "session": d,
        "session_is_holiday": market_calendar.is_market_holiday(night),
        "universe_built_at": None,
        "universe_symbols": None,
        "attempted": None,
        "scanned": None,
        "fetch_failed": None,
        "too_short": None,
        "suspect_series": None,
        "rows_written": None,
        "retired": None,
        "benchmark": None,
        "concurrency": None,
        "unmeasured_top": None,
        "errors_total": None,
        "errors": None,
        "duration_s": None,
        "pruned": None,
    }


def _finish(summary: dict, *, status: str, reason: str | None) -> dict:
    """Prune, publish the summary, return it. EVERY exit path goes through here.

    Retention is deliberately not conditional on the scan having done anything — see the module
    docstring's memory.prune() note. A month of refusals must still age out a quarter-old
    cross-section, or the one thing that keeps scan.db bounded stops running exactly when the job
    stops working.
    """
    summary["status"] = status
    summary["reason"] = reason
    try:
        summary["pruned"] = scan_store.prune()
    except Exception:  # noqa: BLE001 — prune() already swallows its own errors and returns 0; this
        # is the belt for the braces, because a housekeeping failure must not lose the summary of a
        # run that otherwise succeeded. None (not 0) says the prune did not report.
        log.warning("market scan: prune failed", exc_info=True)
        summary["pruned"] = None
    _write_summary(summary)
    if status in ("refused", "empty"):
        log.warning("market scan %s: %s", status, reason)
    return summary


def _write_summary(summary: dict) -> None:
    """Atomic publish: per-writer-unique temp + os.replace (the app/universe.py:190-196 pattern).

    NOT `LATEST.write_text(...)` — that is app/scan_job.py:437, a known non-atomic bug: a reader
    that opens the file mid-write gets a truncated document, and a crash mid-write leaves one
    permanently.
    """
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = LATEST.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(summary, indent=2))
        os.replace(tmp, LATEST)
    except Exception:  # noqa: BLE001 — the summary is an artifact for the ops page; failing to
        # write it must not discard a completed scan whose rows are already committed.
        log.warning("market scan: could not write %s", LATEST, exc_info=True)


def summary_line(out: dict) -> str:
    """The one line a human wants out of a run. See the print() note in __main__.

    Built here rather than inline so the counters clause can be OMITTED on a path that never
    attempted a symbol: "scanned None / failed None / short None of None" is not more honest than
    saying nothing, it is just unreadable, and the JSON artifact carries the nulls properly.
    """
    parts = [f"market scan {out.get('status')} \u00b7 session {out.get('session')}"]
    if out.get("attempted") is not None:
        conc = out.get("concurrency") or {}
        parts.append(
            f"scanned {out['scanned']} / failed {out['fetch_failed']} / short {out['too_short']}"
            f" / suspect {out.get('suspect_series')} of {out['attempted']}"
        )
        parts.append(f"stored {out.get('rows_written')}")
        if out.get("retired"):
            parts.append(f"retired {out['retired']}")
        parts.append(
            f"concurrency {conc.get('final')} (peak {conc.get('high_water')},"
            f" backoffs {conc.get('backoffs')}, throttles {conc.get('throttle_signals')})"
        )
        parts.append(f"{out.get('duration_s')}s")
    parts.append(f"pruned {out.get('pruned')}")
    if out.get("reason"):
        parts.append(str(out["reason"]))
    return " \u00b7 ".join(parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nightly LLM-free market scan (SWT-1).")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="scan only the first N symbols of the universe (cap-descending). 400 is the only "
             "concurrency-tested slice \u2014 prove the run there before pointing it at the full list.",
    )
    parser.add_argument("--force", action="store_true",
                        help="re-measure a session that is already stored")
    args = parser.parse_args()
    out = asyncio.run(run_market_scan(force=args.force, limit=args.limit))
    # THE print() IS LOAD-BEARING, not decoration. Nothing in this repo ever calls
    # logging.basicConfig(), so the root logger has no handler and no level configured: every
    # log.info in this file is dropped on the floor when the job runs under systemd (only
    # logging.lastResort rescues WARNING and above, and only to stderr). Verified in production
    # against the existing nightly unit: `journalctl -u signals-scan --since -30days |
    # grep -ci universe` returns 0, despite a universe rebuild demonstrably having happened in that
    # window. stdout is the only channel that reliably reaches the journal, so the one line a human
    # would want is printed rather than logged.
    print(summary_line(out))
