"""
Optional news + earnings context for the analyst (Tier 3), via Finnhub's free tier. Adds recent
headlines and the next earnings date to a stock's snapshot so Claude can ground the `catalysts` and
`key_risks` fields in real events. No-op (empty) when no Finnhub key is configured, or for crypto.

NEWS-7 — the free tier allows 60 requests a minute, and this module is where every fan-out over the
watchlist lands, so it is where the budget has to be kept. Two mechanisms, and they do different
jobs:

  * `RATE`, a process-wide sliding-window limiter every request here passes through. It PACES rather
    than drops: a caller waits for a slot up to its own deadline, and only a caller that would wait
    past that deadline is refused. Background work (the nightly scan) can afford to wait a minute;
    an HTTP handler answering a phone cannot, so it passes a short `wait` and takes an honest
    "unknown" for the remainder.

  * `earnings_on()`, which answers "who reports on this date" for a whole watchlist in ONE request
    instead of one per symbol. That is the actual fix for the morning brief, which was spending 100
    requests — two per symbol, of which it read one — inside a single second and getting about 40 of
    them refused with HTTP 429 every trading day since at least 2026-08-25.

Every failure path here reports absence honestly. A refused lookup and a symbol with genuinely no
earnings scheduled are the same empty answer to a caller that only looks at the value, and a brief
that says "no catalysts today" because it was rate-limited is indistinguishable from one that
checked. So the lookups return an explicit ok/not-ok alongside the value, and callers must carry it.
"""
from __future__ import annotations

import asyncio
import collections
import datetime as dt
import logging
import time
from typing import NamedTuple
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")

import httpx

from . import settings_store
from .redact import http_error

log = logging.getLogger("signals.news")

_BASE = "https://finnhub.io/api/v1"

# Finnhub's free tier is 60 calls/minute against the KEY, not the client, so the budget is shared
# with anything else holding it — the app's own quote fetches included. 55 leaves that headroom.
_WINDOW_SECONDS = 60.0
_CALLS_PER_WINDOW = 55
# What a caller with no opinion is willing to wait for a slot. Sized for the nightly scan, which
# fans out over the whole watchlist at once and would rather take a minute than lose its context.
# Interactive handlers pass something far shorter.
_DEFAULT_WAIT = 75.0


class _RateLimiter:
    """A sliding-window limiter: at most `per_window` acquisitions in any `window` seconds.

    Not a token bucket. A bucket refilling at 55/60s would still let a 55-request burst through the
    instant the process starts, and then drip — which is the shape that produced the 429s. A sliding
    window holds the guarantee that matters to the upstream: no more than N in any 60-second span.
    """

    def __init__(self, per_window: int = _CALLS_PER_WINDOW, window: float = _WINDOW_SECONDS) -> None:
        self.per_window = per_window
        self.window = window
        self._hits: collections.deque[float] = collections.deque()
        self._lock = asyncio.Lock()

    async def acquire(self, wait: float) -> bool:
        """Claim a slot, waiting at most `wait` seconds. False means the caller must not send.

        The lock is released across the sleep so a caller with a shorter deadline is not held behind
        one with a longer deadline; the re-check under the lock after waking is what keeps the
        window honest despite the resulting race.

        Timestamps come from `time.monotonic()`, not `loop.time()`. The two are the same clock under
        asyncio's default loop, but loop.time()'s ORIGIN is per-loop, and this object is module-level
        state that outlives any single `asyncio.run()` — a deque of timestamps from a previous loop
        would be compared against a fresh loop's clock and could hold the window open or closed for
        an arbitrary interval.
        """
        deadline = time.monotonic() + max(0.0, wait)
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._hits and now - self._hits[0] >= self.window:
                    self._hits.popleft()
                if len(self._hits) < self.per_window:
                    self._hits.append(now)
                    return True
                sleep_for = max(self.window - (now - self._hits[0]), 0.01)
            if time.monotonic() + sleep_for > deadline:
                return False
            await asyncio.sleep(sleep_for)

    def reset(self) -> None:
        """Forget the window, and the lock with it. For tests only — never call this to 'make room'.

        The lock is replaced because an `asyncio.Lock` binds to the loop that first CONTENDS it, and
        a lock carrying a binding to a finished loop raises on the next contention.
        """
        self._hits.clear()
        self._lock = asyncio.Lock()


RATE = _RateLimiter()


class EarningsLookup(NamedTuple):
    """`date` is the next scheduled earnings date, or None.

    `ok` is the half that must not be dropped: False means the lookup FAILED and the date is
    UNKNOWN, which is a different claim from a successful lookup that found nothing scheduled.
    """

    date: str | None
    ok: bool


async def _get(
    client: httpx.AsyncClient,
    path: str,
    params: dict,
    *,
    what: str,
    timeout: float = 12.0,
    wait: float = _DEFAULT_WAIT,
):
    """One rate-limited Finnhub GET. Returns the decoded JSON, or None if anything went wrong.

    `what` names the lookup for the log line. Failures are logged through `redact.http_error`, never
    with the raw exception: httpx puts the whole request URL in its message and the API key is a
    query parameter (SEC-1).
    """
    key = settings_store.get().get("finnhub_api_key", "")
    if not key:
        return None
    if not await RATE.acquire(wait):
        # Not sending is deliberate. Sending anyway would earn a 429 and spend the caller's latency
        # to learn what is already known.
        log.warning("news: %s skipped — Finnhub rate budget exhausted (%d/%ds), waited %.0fs",
                    what, RATE.per_window, int(RATE.window), wait)
        return None
    try:
        r = await client.get(f"{_BASE}{path}", params={**params, "token": key}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:  # noqa: BLE001 — decorative context; never fail the caller on it
        # Still logged. A verdict silently written without the news that explains the move looks
        # identical to one written on a genuinely quiet tape.
        log.warning("news: %s failed (%s)", what, http_error(e))
        return None


# Finnhub truncates /calendar/earnings at this many rows and says nothing about having done so.
# Measured against the live key on 2026-09-01: a 90-day market-wide window returned exactly 1500
# rows spanning ONLY 2026-11-09..11-30, while the same window fetched in 15-day chunks returned
# 3,903 distinct rows — every September and October date silently dropped, and the surviving tail
# looking for all the world like a complete answer. So a market-wide query is only trustworthy over
# a window small enough to stay under the cap, and a result that reaches it must be treated as
# unknown rather than read.
_CALENDAR_ROW_CAP = 1500


def _match_keys(symbol: object) -> set[str]:
    """The forms a ticker may be spelled in, for matching a calendar row against a watchlist entry.

    Necessary because the two sides disagree, and the disagreement is silent. Verified against the
    live API on 2026-09-01: the market-wide calendar for 2026-10-30 carries exactly one Berkshire
    row and spells it `BRK.A`, while this user's watchlist holds `BRK-A` and `BRK-B`. An exact-match
    intersection reports BOTH as not reporting on their own earnings day — the same "absent rendered
    as nothing to report" defect the caller exists to remove, reintroduced one layer down. The
    per-symbol lookup never hit this: it asked BY symbol and read the date off whatever row came
    back, without ever comparing spellings.

    Two keys per symbol. The punctuation-normalised whole (`BRK-A` and `BRK.A` both give `BRKA`),
    and — only for a symbol that carries a class or series suffix — its root (`BRK`), because an
    issuer reports ONCE and the feed files that report under a single share class. `GME.WS` matching
    a `GME` row is the same case and the same right answer. A symbol with no separator contributes
    no root key, so plain tickers can never collide on a prefix.
    """
    s = str(symbol or "").strip().upper()
    if not s:
        return set()
    keys = {s.replace(".", "").replace("-", "")}
    root = s.split(".")[0].split("-")[0]
    if root != s and root:
        keys.add(root)
    return keys


async def earnings_on(
    client: httpx.AsyncClient, day: str, symbols: list[str] | set[str], *, wait: float = _DEFAULT_WAIT,
) -> tuple[set[str], bool]:
    """Which of `symbols` report on `day` (an ET YYYY-MM-DD), in ONE request for the whole list.

    Returns (symbols reporting, ok). `ok` False means the market-wide calendar could not be read and
    the answer is UNKNOWN — an empty set with ok False is not "nobody reports today".

    This exists because the per-symbol form costs one request each and the morning brief asks it for
    every equity on the watchlist at once. One day of the market-wide calendar is ~35 rows, three
    orders of magnitude under the truncation cap, so the whole question fits in a single call.
    """
    want = [str(s).strip().upper() for s in symbols if str(s).strip()]
    if not want:
        return set(), True
    body = await _get(client, "/calendar/earnings", {"from": day, "to": day},
                      what=f"earnings calendar for {day}", wait=wait)
    if not isinstance(body, dict):
        return set(), False
    cal = body.get("earningsCalendar")
    if not isinstance(cal, list):
        return set(), False
    if len(cal) >= _CALENDAR_ROW_CAP:
        log.warning("news: earnings calendar for %s returned %d rows — at the truncation cap, so "
                    "the day is unknown rather than empty", day, len(cal))
        return set(), False
    reported: set[str] = set()
    for r in cal:
        if isinstance(r, dict):
            reported |= _match_keys(r.get("symbol"))
    return {s for s in want if _match_keys(s) & reported}, True


async def fetch_next_earnings(
    client: httpx.AsyncClient, symbol: str, *, wait: float = _DEFAULT_WAIT,
) -> EarningsLookup:
    """The symbol's next scheduled earnings date within ~90 days, and whether the lookup succeeded.

    Split out of `fetch_context` for NEWS-7: callers that want only the date were paying for a
    headline fetch they then threw away, which doubled the demand on a 60-a-minute budget.
    """
    today = dt.date.today()
    body = await _get(
        client, "/calendar/earnings",
        {"from": today.isoformat(), "to": (today + dt.timedelta(days=90)).isoformat(), "symbol": symbol},
        what=f"{symbol} earnings-calendar lookup", wait=wait,
    )
    if not isinstance(body, dict):
        return EarningsLookup(None, False)
    cal = body.get("earningsCalendar")
    if cal is None:
        cal = []
    if not isinstance(cal, list):
        # Malformed is UNKNOWN, not "nothing scheduled". Collapsing the two here is the same
        # substitution this whole task exists to remove, one layer further in.
        log.warning("news: %s earnings-calendar returned a non-list payload (%s)",
                    symbol, type(cal).__name__)
        return EarningsLookup(None, False)
    dates = sorted(d for d in (str(r.get("date") or "") for r in cal if isinstance(r, dict)) if d)
    if not cal:
        return EarningsLookup(None, True)
    if not dates:
        log.warning("news: %s earnings-calendar returned %d rows with no usable date", symbol, len(cal))
        return EarningsLookup(None, False)
    return EarningsLookup(dates[0], True)


async def fetch_headlines(
    client: httpx.AsyncClient, symbol: str, *, wait: float = _DEFAULT_WAIT,
) -> list[str] | None:
    """A few recent headlines for the analyst snapshot. None means the lookup failed."""
    today = dt.date.today()
    news = await _get(
        client, "/company-news",
        {"symbol": symbol, "from": (today - dt.timedelta(days=10)).isoformat(), "to": today.isoformat()},
        what=f"{symbol} headline context", wait=wait,
    )
    if not isinstance(news, list):
        return None
    return [h for h in ((n.get("headline") or "").strip() for n in news) if h][:5]


async def fetch_context(client: httpx.AsyncClient, symbol: str, *, wait: float = _DEFAULT_WAIT) -> dict:
    """Headlines plus the next earnings date, as snapshot keys. Absent keys mean absent context.

    Kept as the one call the analyst paths make, so nothing there has to know about the split. A
    caller that needs only one half should call that half directly and spend one request, not two.
    """
    out: dict = {}
    heads = await fetch_headlines(client, symbol, wait=wait)
    if heads:
        out["recent_news"] = heads
    earnings = await fetch_next_earnings(client, symbol, wait=wait)
    if earnings.date:
        out["next_earnings"] = earnings.date
    return out


async def fetch_dated_news(
    client: httpx.AsyncClient, symbol: str, days: int = 16, *, wait: float = _DEFAULT_WAIT,
) -> list[dict] | None:
    """Company news over the last [days], each carrying its ET date so the analyst can line headlines
    up against specific price moves (AIE-4). Returns [{date: YYYY-MM-DD, headline, summary, source, url}]
    newest-first.

    Returns **None when the LOOKUP FAILED** (no key, HTTP error, rate limit, timeout) and **[] only
    when the fetch succeeded and the symbol genuinely has no coverage**. Callers must not conflate
    them.

    That distinction is the whole point of this signature. It used to return [] on every failure,
    silently and without a log line — so on 2026-08-03 GME fell 10.4% on a $1.4B convertible-notes
    dilution, the app asked why, one lookup came back empty, and /news_moves rendered "No headlines
    available for this day" and CACHED that confident claim for an hour. Finnhub had 26 articles
    including "Why GameStop Stock Just Slumped". Because nothing was logged, it is still not
    recoverable whether that lookup was rate-limited by the ~10-call burst the app fires when a
    symbol is opened, or whether Finnhub had simply not indexed yet.
    """
    if not settings_store.get().get("finnhub_api_key", ""):
        log.warning("news: no Finnhub key configured — %s lookup skipped", symbol)
        return None
    today = dt.date.today()
    # _get logs its own failure, redacted. LOUD on purpose there: a rate limit and "this company
    # made no news" are the same empty list to every caller downstream; the log is the only place
    # they stay distinguishable.
    raw = await _get(
        client, "/company-news",
        {"symbol": symbol, "from": (today - dt.timedelta(days=days)).isoformat(),
         "to": today.isoformat()},
        what=f"{symbol} dated-news lookup (this is not 'no news')", wait=wait,
    )
    if not isinstance(raw, list):
        return None
    out: list[dict] = []
    for n in raw:
        head = (n.get("headline") or "").strip()
        ts = n.get("datetime")
        if not head or not ts:
            continue
        try:
            # ET, as the docstring promises and as the analyst assumes when lining headlines up
            # against price moves. utcfromtimestamp pushed anything published after 20:00 ET onto the
            # FOLLOWING calendar day, so an evening headline appeared to precede the next session's
            # move rather than follow that day's.
            date = dt.datetime.fromtimestamp(int(ts), _ET).date().isoformat()
        except Exception:  # noqa: BLE001
            continue
        out.append({
            "date": date,
            "headline": head,
            "summary": (n.get("summary") or "").strip()[:280],
            "source": (n.get("source") or "").strip(),
            "url": (n.get("url") or "").strip(),
        })
    out.sort(key=lambda x: x["date"], reverse=True)
    return out
