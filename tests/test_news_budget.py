"""NEWS-7 — the morning brief must not spend the whole Finnhub minute, and a partial answer must
say it is partial.

Two defects, one call site. `/daily_brief` built `catalysts_today` by calling `fetch_context()` once
per watchlist equity; that helper makes TWO requests and the brief read ONE of them, so a 50-name
watchlist fired 100 requests in about a second against a 60-a-minute free tier. Measured in the
journal on the deployed container, one burst per trading day at the brief's timestamp: 2026-08-25
61 x 429, 08-26 59, 08-27 60, 08-28 60, 08-31 16, 09-01 58.

The second defect is the one that reached the user. `_reports_today()` swallowed the failure and
returned None, so a name whose lookup was refused looked exactly like a name that is not reporting,
and the brief narrated "no catalysts today" over an answer drawn from roughly 40% of the watchlist.

Async tests follow this repo's convention — `asyncio.run()` inside a sync test, no plugin.
"""

import asyncio
import time

import httpx
import pytest

from app import news
from app.news import EarningsLookup, _RateLimiter, earnings_on, fetch_context, fetch_next_earnings

WATCHLIST = [f"SYM{i}" for i in range(50)]


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _keyed_and_unpaced(monkeypatch):
    """A configured key (so nothing short-circuits) and an empty rate window per test."""
    monkeypatch.setattr(news.settings_store, "get", lambda: {"finnhub_api_key": "test-key"})
    news.RATE.reset()
    yield
    news.RATE.reset()


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _calendar(rows):
    return httpx.Response(200, json={"earningsCalendar": rows})


# --- the request budget -------------------------------------------------------------------------


def test_a_whole_watchlist_costs_one_request_not_a_hundred():
    calls: list[httpx.Request] = []

    def handler(req):
        calls.append(req)
        return _calendar([{"symbol": "SYM7", "date": "2026-09-01"},
                          {"symbol": "NOTMINE", "date": "2026-09-01"}])

    async def body():
        async with _client(handler) as c:
            return await earnings_on(c, "2026-09-01", WATCHLIST)

    reporting, ok = run(body())
    assert len(calls) == 1, "the brief's fan-out is the whole defect"
    assert calls[0].url.path == "/api/v1/calendar/earnings"
    assert reporting == {"SYM7"} and ok is True


def test_the_market_wide_call_asks_for_one_day_only():
    """Not a range. Finnhub truncates this endpoint at 1500 rows without saying so, and a 90-day
    market-wide window measured 1500 rows covering only the LAST three weeks of it."""
    seen: dict = {}

    def handler(req):
        seen.update(dict(req.url.params))
        return _calendar([])

    async def body():
        async with _client(handler) as c:
            await earnings_on(c, "2026-09-01", ["AAPL"])

    run(body())
    assert seen["from"] == seen["to"] == "2026-09-01"


def test_asking_about_nobody_sends_nothing():
    def handler(req):  # pragma: no cover — reaching this is the failure
        raise AssertionError("no request should be made for an empty symbol list")

    async def body():
        async with _client(handler) as c:
            return await earnings_on(c, "2026-09-01", [])

    assert run(body()) == (set(), True)


def test_the_earnings_only_lookup_makes_one_request_not_two():
    """The halving that keeps /calendar under the limit: fetch_context() would also fetch headlines
    that caller discards."""
    paths: list[str] = []

    def handler(req):
        paths.append(req.url.path)
        return _calendar([{"symbol": "AAPL", "date": "2026-11-01"}])

    async def body():
        async with _client(handler) as c:
            return await fetch_next_earnings(c, "AAPL")

    got = run(body())
    assert paths == ["/api/v1/calendar/earnings"]
    assert got == EarningsLookup("2026-11-01", True)


def test_fetch_context_still_returns_both_halves_for_the_analyst_paths():
    def handler(req):
        if req.url.path.endswith("company-news"):
            return httpx.Response(200, json=[{"headline": "A thing happened"}])
        return _calendar([{"symbol": "AAPL", "date": "2026-11-01"}])

    async def body():
        async with _client(handler) as c:
            return await fetch_context(c, "AAPL")

    assert run(body()) == {"recent_news": ["A thing happened"], "next_earnings": "2026-11-01"}


def test_the_earliest_scheduled_date_wins_not_the_first_row():
    def handler(req):
        return _calendar([{"symbol": "AAPL", "date": "2026-12-01"},
                          {"symbol": "AAPL", "date": "2026-10-30"}])

    async def body():
        async with _client(handler) as c:
            return await fetch_next_earnings(c, "AAPL")

    assert run(body()).date == "2026-10-30"


# --- absence, not substitution ------------------------------------------------------------------


def test_a_rate_limited_calendar_is_unknown_not_empty():
    """The defect the user actually saw: 429 -> [] -> 'no catalysts today', stated confidently."""

    async def body():
        async with _client(lambda req: httpx.Response(429)) as c:
            return await earnings_on(c, "2026-09-01", WATCHLIST)

    reporting, ok = run(body())
    assert reporting == set()
    assert ok is False, "an unread calendar must not read as a checked one"


def test_a_successfully_read_empty_day_is_complete():
    """The other half of the claim: ok True with an empty set genuinely means nobody reports."""

    async def body():
        async with _client(lambda req: _calendar([])) as c:
            return await earnings_on(c, "2026-09-01", WATCHLIST)

    assert run(body()) == (set(), True)


def test_a_truncated_calendar_is_refused_rather_than_read():
    """1500 rows is Finnhub's silent cap. Reading it would answer from whichever slice survived."""
    rows = [{"symbol": f"S{i}", "date": "2026-09-01"} for i in range(news._CALENDAR_ROW_CAP)]
    rows[0]["symbol"] = "SYM7"

    async def body():
        async with _client(lambda req: _calendar(rows)) as c:
            return await earnings_on(c, "2026-09-01", WATCHLIST)

    reporting, ok = run(body())
    assert ok is False
    assert reporting == set(), "a name found in a truncated page is not a checked answer"


def test_a_failed_per_symbol_lookup_is_not_absent_earnings():
    async def body():
        async with _client(lambda req: httpx.Response(503)) as c:
            return await fetch_next_earnings(c, "AAPL")

    assert run(body()) == EarningsLookup(None, False)


def test_a_symbol_with_nothing_scheduled_reports_ok():
    async def body():
        async with _client(lambda req: _calendar([])) as c:
            return await fetch_next_earnings(c, "AAPL")

    assert run(body()) == EarningsLookup(None, True)


def test_a_malformed_payload_is_a_failed_lookup():
    async def body():
        async with _client(lambda req: httpx.Response(200, json={"earningsCalendar": "nope"})) as c:
            return await fetch_next_earnings(c, "AAPL"), await earnings_on(c, "2026-09-01", ["AAPL"])

    one, many = run(body())
    assert one.ok is False
    assert many == (set(), False)


def test_no_configured_key_is_a_failed_lookup_not_an_empty_day(monkeypatch):
    monkeypatch.setattr(news.settings_store, "get", lambda: {"finnhub_api_key": ""})

    def handler(req):  # pragma: no cover
        raise AssertionError("must not call Finnhub without a key")

    async def body():
        async with _client(handler) as c:
            return await earnings_on(c, "2026-09-01", ["AAPL"]), await fetch_next_earnings(c, "AAPL")

    many, one = run(body())
    assert many == (set(), False)
    assert one.ok is False


# --- the limiter ---------------------------------------------------------------------------------


def test_the_window_admits_its_quota_immediately_and_then_holds():
    async def body():
        r = _RateLimiter(per_window=3, window=30.0)
        got = [await r.acquire(0.0) for _ in range(3)]
        return got, await r.acquire(0.0)

    got, fourth = run(body())
    assert got == [True] * 3
    assert fourth is False, "the fourth must wait, not send"


def test_a_caller_willing_to_wait_gets_its_slot_when_the_window_rolls():
    async def body():
        r = _RateLimiter(per_window=2, window=0.15)
        assert await r.acquire(0.0) and await r.acquire(0.0)
        refused = await r.acquire(0.0)
        return refused, await r.acquire(1.0)

    refused, paced = run(body())
    assert refused is False
    assert paced is True, "pacing, not dropping"


def test_a_refused_slot_sends_nothing_at_all():
    """Sending anyway would earn the 429 that the limiter exists to avoid."""
    sent: list[httpx.Request] = []

    def handler(req):
        sent.append(req)
        return _calendar([])

    async def body():
        for _ in range(news.RATE.per_window):
            assert await news.RATE.acquire(0.0)
        async with _client(handler) as c:
            return await fetch_next_earnings(c, "AAPL", wait=0.0)

    got = run(body())
    assert sent == []
    assert got == EarningsLookup(None, False), "budget exhausted is unknown, not 'nothing scheduled'"


def test_a_fifty_name_watchlist_fits_inside_one_minute_of_budget():
    """The /calendar path is still per-symbol. One request each, against the window, has to fit —
    otherwise the fix has only moved the burst rather than removed it."""

    async def body():
        r = _RateLimiter()
        return r.per_window, [await r.acquire(0.0) for _ in range(50)]

    per_window, got = run(body())
    assert per_window >= 50
    assert all(got)


def test_the_limiter_is_shared_so_two_fan_outs_cannot_each_spend_the_minute():
    """news.RATE is module-level on purpose: the quota belongs to the KEY, not to a call site."""

    def handler(req):  # pragma: no cover — must not be reached
        raise AssertionError("the shared budget was already spent")

    async def body():
        for _ in range(news.RATE.per_window):
            assert await news.RATE.acquire(0.0)
        async with _client(handler) as c:
            return await earnings_on(c, "2026-09-01", ["AAPL"], wait=0.0)

    assert run(body()) == (set(), False)


def test_the_default_wait_lets_background_work_pace_rather_than_lose_its_context():
    """The nightly scan fans out over the whole watchlist and would rather take a minute than write
    verdicts with no news in them; an HTTP handler passes something far shorter."""
    assert news._DEFAULT_WAIT > news._WINDOW_SECONDS


def test_dated_rows_with_no_usable_date_are_a_failed_lookup_not_an_empty_calendar():
    """A non-empty calendar whose rows carry no date is malformed. Returning ok True there would
    put "nothing scheduled" on a symbol whose schedule was never actually read."""

    async def body():
        async with _client(lambda req: _calendar([{"symbol": "AAPL"}, {"symbol": "AAPL", "date": ""}])) as c:
            return await fetch_next_earnings(c, "AAPL")

    assert run(body()) == EarningsLookup(None, False)


# --- symbol spelling ------------------------------------------------------------------------------
#
# The market-wide calendar and the watchlist do not agree on how to spell a class share, and the
# disagreement is silent. Verified against the live API on 2026-09-01: the one-day calendar for
# 2026-10-30 carries exactly one Berkshire row, spelled BRK.A, while the watchlist holds BRK-A and
# BRK-B. An exact-match intersection reports both as not reporting on their own earnings day.


def test_a_class_share_matches_the_row_its_issuer_actually_files_under():
    def handler(req):
        return _calendar([{"symbol": "BRK.A", "date": "2026-10-30"}])

    async def body():
        async with _client(handler) as c:
            return await earnings_on(c, "2026-10-30", ["BRK-A", "BRK-B", "AAPL"])

    reporting, ok = run(body())
    assert ok is True
    assert reporting == {"BRK-A", "BRK-B"}, "both share classes report when the issuer reports"


def test_punctuation_alone_does_not_hide_a_match():
    async def body():
        async with _client(lambda req: _calendar([{"symbol": "BRK-B", "date": "2026-10-30"}])) as c:
            return await earnings_on(c, "2026-10-30", ["BRK.B"])

    assert run(body())[0] == {"BRK.B"}


def test_the_watchlist_spelling_is_what_comes_back_not_the_feed_s():
    """The caller matches these against its own watchlist and shows them to a user."""

    async def body():
        async with _client(lambda req: _calendar([{"symbol": "GME", "date": "2026-10-30"}])) as c:
            return await earnings_on(c, "2026-10-30", ["GME.WS"])

    assert run(body())[0] == {"GME.WS"}


def test_a_plain_ticker_never_matches_on_a_shared_prefix():
    """The root key exists only for symbols carrying a separator, so AAPL cannot match an AAP row."""

    async def body():
        async with _client(lambda req: _calendar([{"symbol": "AAP", "date": "2026-10-30"}])) as c:
            return await earnings_on(c, "2026-10-30", ["AAPL"])

    assert run(body())[0] == set()


def test_a_non_dict_row_does_not_take_the_lookup_down():
    async def body():
        async with _client(lambda req: _calendar(["garbage", {"symbol": "AAPL", "date": "2026-10-30"}])) as c:
            return await earnings_on(c, "2026-10-30", ["AAPL"])

    assert run(body()) == ({"AAPL"}, True)


# --- the limiter's invariant under real concurrency ------------------------------------------------


def test_the_window_holds_under_many_concurrent_callers():
    """The lock is released across the sleep, so the re-check after waking is the only thing keeping
    the window honest. Drive it hard enough that a racing re-check would show."""
    r = _RateLimiter(per_window=10, window=0.20)
    stamps: list[float] = []

    async def body():
        async def one():
            if await r.acquire(2.0):
                stamps.append(time.monotonic())
        await asyncio.gather(*[one() for _ in range(60)])

    run(body())
    assert len(stamps) == 60, "everyone willing to wait should eventually get through"
    stamps.sort()
    # No window-length span may contain more than per_window acquisitions.
    for i, start in enumerate(stamps):
        inside = sum(1 for t in stamps[i:] if t - start < r.window)
        assert inside <= r.per_window, f"{inside} acquisitions inside one {r.window}s window"


def test_a_cancelled_waiter_does_not_hold_the_lock():
    async def body():
        r = _RateLimiter(per_window=1, window=5.0)
        assert await r.acquire(0.0)
        task = asyncio.create_task(r.acquire(4.0))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # The limiter must still be usable — and still refuse, since the window has not rolled.
        return await asyncio.wait_for(r.acquire(0.0), timeout=1.0)

    assert run(body()) is False


def test_the_limiter_survives_being_used_across_separate_event_loops():
    """news.RATE is module-level and tests call asyncio.run() repeatedly. A lock bound to a finished
    loop, or a timestamp from that loop's clock, must not poison the next run."""
    r = _RateLimiter(per_window=2, window=0.05)

    async def contend():
        await asyncio.gather(*[r.acquire(1.0) for _ in range(4)])

    run(contend())
    run(contend())
    assert run(contend()) is None or True  # reaching here without RuntimeError is the assertion


def test_the_nightly_scan_s_whole_watchlist_still_gets_its_news():
    """The limiter's blast radius. scan_job.run_scan() gathers _score() for every watchlist stock
    with NO semaphore, and each one calls fetch_context() — two Finnhub requests, issued in
    sequence. For 50 equities that is 100 requests arriving at once against a 55-per-60s window.

    The limiter PACES rather than drops, so the question is whether the default deadline outlasts
    the queue. It must: a scan that quietly wrote 45 verdicts with no news in them, having
    previously had news, would be a regression dressed as a rate-limit fix.
    """
    r = _RateLimiter(per_window=55, window=0.20)   # the real ratio, time-compressed
    scaled_wait = 0.20 * (news._DEFAULT_WAIT / news._WINDOW_SECONDS)
    got: list[bool] = []

    async def body():
        async def one_symbol():
            # fetch_context's two requests, in the order it issues them.
            got.append(await r.acquire(scaled_wait))
            got.append(await r.acquire(scaled_wait))
        await asyncio.gather(*[one_symbol() for _ in range(50)])

    run(body())
    assert len(got) == 100
    assert all(got), f"{got.count(False)} of 100 scan lookups would have been dropped"
