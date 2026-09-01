"""CAL-1 — a catalyst calendar that stops must say that it stopped.

`shorts.calendar()` used to end at `events[:30]`, a row cap applied with no marker. Measured against
the real watchlist on 2026-09-01: the last earnings event shown was GOOGL on 2026-10-27, while AAPL
(10-28), XOM (10-29) and Berkshire (10-30) were all successfully looked up — the payload's own
`earnings_unchecked` was empty — and then dropped by the slice. A user asking "when does Apple
report" got nothing, and no reason.

Lower severity than the NEWS-7 defect it was found beside, because the missing rows are the furthest
away rather than an arbitrary 40% of the watchlist. Same class, though: a partial answer presented as
a whole one.
"""
import asyncio
import importlib
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from app import shorts


def _future(n: int) -> str:
    return (date.today() + timedelta(days=n)).isoformat()


# --- the data layer no longer decides how many rows a screen shows -------------------------------


def test_the_gatherer_returns_everything_inside_its_horizon():
    """Forty earnings dates must all come back; the caller caps and reports, not this."""
    syms = [f"S{i}" for i in range(40)]
    earnings = {s: _future(10 + i) for i, s in enumerate(syms)}

    async def no_ftd(client, sym):
        return None

    real = shorts.ftd
    shorts.ftd = no_ftd
    try:
        events = asyncio.run(shorts.calendar(None, syms, earnings))
    finally:
        shorts.ftd = real

    got = [e for e in events if e["kind"] == "earnings"]
    assert len(got) == 40, "the 30-row slice used to eat ten of these silently"


def test_the_horizon_is_a_date_bound_not_a_row_count():
    """A far-future date must not ride in just because the list happens to be short."""
    earnings = {"NEAR": _future(30), "FAR": _future(400)}

    async def no_ftd(client, sym):
        return None

    real = shorts.ftd
    shorts.ftd = no_ftd
    try:
        events = asyncio.run(shorts.calendar(None, ["NEAR", "FAR"], earnings, horizon_days=120))
    finally:
        shorts.ftd = real

    syms = {e["symbol"] for e in events if e["kind"] == "earnings"}
    assert syms == {"NEAR"}


def test_events_come_back_soonest_first():
    earnings = {"LATE": _future(90), "SOON": _future(3), "MID": _future(40)}

    async def no_ftd(client, sym):
        return None

    real = shorts.ftd
    shorts.ftd = no_ftd
    try:
        events = asyncio.run(shorts.calendar(None, list(earnings), earnings))
    finally:
        shorts.ftd = real

    dates = [e["date"] for e in events]
    assert dates == sorted(dates)


# --- the route caps, and says so ------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    import app.sandbox_store as ss
    import app.settings_store as st
    importlib.reload(st)
    importlib.reload(ss)
    import app.main as m
    importlib.reload(m)
    monkeypatch.setattr(m.settings_store, "get", lambda: {
        "watchlist": ["AAPL"], "crypto_watchlist": [], "finnhub_api_key": "k",
        "verdict_ttl_seconds": 3600,
    })
    from app.news import EarningsLookup

    async def fake_next_earnings(http, sym, *, wait=0.0):
        return EarningsLookup(None, True)
    monkeypatch.setattr(m, "fetch_next_earnings", fake_next_earnings)
    return m, monkeypatch


def _serve(m, monkeypatch, n_events: int):
    async def fake_calendar(http, syms, earnings=None, **kw):
        return [{"date": _future(i + 1), "symbol": f"S{i}", "label": "Earnings", "kind": "earnings"}
                for i in range(n_events)]
    monkeypatch.setattr(m.shorts, "calendar", fake_calendar)
    with TestClient(m.app) as c:
        return c.get("/calendar").json()


def test_a_truncated_calendar_reports_the_total_and_where_it_stopped(client):
    m, monkeypatch = client
    body = _serve(m, monkeypatch, m._CALENDAR_MAX_EVENTS + 17)

    assert len(body["events"]) == m._CALENDAR_MAX_EVENTS
    assert body["events_total"] == m._CALENDAR_MAX_EVENTS + 17
    assert body["truncated_after"] == body["events"][-1]["date"], \
        "the client has to be able to say 'through <date>', not just 'there is more'"


def test_a_complete_calendar_says_so_rather_than_leaving_it_to_be_inferred(client):
    m, monkeypatch = client
    body = _serve(m, monkeypatch, 5)

    assert len(body["events"]) == 5
    assert body["events_total"] == 5
    assert body["truncated_after"] is None, "null is the whole signal — a short list is not a cut one"


def test_the_total_is_present_even_when_there_is_nothing(client):
    m, monkeypatch = client
    body = _serve(m, monkeypatch, 0)
    assert body["events"] == [] and body["events_total"] == 0 and body["truncated_after"] is None


def test_exactly_at_the_cap_is_not_reported_as_truncated(client):
    """The off-by-one that would tell a user rows are missing when none are."""
    m, monkeypatch = client
    body = _serve(m, monkeypatch, m._CALENDAR_MAX_EVENTS)
    assert body["events_total"] == m._CALENDAR_MAX_EVENTS
    assert body["truncated_after"] is None
