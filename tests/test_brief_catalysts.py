"""NEWS-7 at the route boundary: what /daily_brief and /calendar actually publish when the earnings
calendar could not be read.

The unit tests in test_news_budget.py pin the lookups. These pin the two places the answer is turned
into something a person reads — the JSON the phone decodes, and the snapshot handed to the analyst
that writes the push notification. Both are places where "we could not check" was silently becoming
"there is nothing to report".
"""
import importlib

import pytest
from fastapi.testclient import TestClient

WATCHLIST = ["AAPL", "MSFT", "NVDA"]


@pytest.fixture
def app_mod(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    import app.sandbox_store as ss
    import app.settings_store as st
    importlib.reload(st)
    importlib.reload(ss)
    import app.main as m
    importlib.reload(m)

    monkeypatch.setattr(m.settings_store, "get", lambda: {
        "watchlist": WATCHLIST, "crypto_watchlist": ["BTC"], "finnhub_api_key": "k",
    })

    async def fake_snapshot(http, watchlist):
        return {"session": "PRE", "indices": {}, "vix": None}
    monkeypatch.setattr(m.market_now, "build_snapshot", fake_snapshot)

    async def fake_movers(side, count):
        return []
    monkeypatch.setattr(m, "_movers_side", fake_movers)
    monkeypatch.setattr(m.usage_store, "record", lambda *a, **k: None)
    return m


def _capture_brief(m, monkeypatch):
    """Swap the analyst for a recorder, so the test can read the snapshot it was handed."""
    seen: dict = {}

    async def fake_daily_brief(snapshot, *, deep=False):
        seen.update(snapshot)
        return (
            type("B", (), {"title": "t", "body": "b", "tone": "mixed"})(),
            {"model": "test", "input_tokens": 0, "output_tokens": 0},
        )

    monkeypatch.setattr(m, "daily_brief", fake_daily_brief)
    return seen


# --- /daily_brief ---------------------------------------------------------------------------------


def test_an_unreadable_calendar_publishes_incomplete_rather_than_empty(app_mod, monkeypatch):
    """The measured defect: 429 on the earnings calendar rendered as "no catalysts today"."""
    m = app_mod
    seen = _capture_brief(m, monkeypatch)

    async def fake_earnings_on(http, day, symbols, *, wait=0.0):
        return set(), False
    monkeypatch.setattr(m, "earnings_on", fake_earnings_on)

    with TestClient(m.app) as c:
        body = c.get("/daily_brief").json()

    assert body["catalysts_today"] == []
    assert body["catalysts_complete"] is False, "the client must be able to tell unknown from none"
    # And the analyst must be told, or it narrates the empty list as fact.
    assert seen["catalysts_complete"] is False
    assert "catalysts_note" in seen


def test_a_read_calendar_publishes_complete_and_names_who_reports(app_mod, monkeypatch):
    m = app_mod
    seen = _capture_brief(m, monkeypatch)

    async def fake_earnings_on(http, day, symbols, *, wait=0.0):
        return {"NVDA"}, True
    monkeypatch.setattr(m, "earnings_on", fake_earnings_on)

    with TestClient(m.app) as c:
        body = c.get("/daily_brief").json()

    assert body["catalysts_today"] == ["NVDA"]
    assert body["catalysts_complete"] is True
    assert "catalysts_note" not in seen, "nothing to explain when the lookup succeeded"


def test_a_genuinely_quiet_day_is_complete_and_empty(app_mod, monkeypatch):
    m = app_mod
    _capture_brief(m, monkeypatch)

    async def fake_earnings_on(http, day, symbols, *, wait=0.0):
        return set(), True
    monkeypatch.setattr(m, "earnings_on", fake_earnings_on)

    with TestClient(m.app) as c:
        body = c.get("/daily_brief").json()
    assert body["catalysts_today"] == [] and body["catalysts_complete"] is True


def test_the_brief_asks_about_equities_only_and_in_one_call(app_mod, monkeypatch):
    """Crypto has no earnings calendar, and the whole point of the fix is one request."""
    m = app_mod
    _capture_brief(m, monkeypatch)
    calls: list[list] = []

    async def fake_earnings_on(http, day, symbols, *, wait=0.0):
        calls.append(sorted(symbols))
        return set(), True
    monkeypatch.setattr(m, "earnings_on", fake_earnings_on)

    with TestClient(m.app) as c:
        c.get("/daily_brief")

    assert calls == [sorted(WATCHLIST)]


def test_a_raising_lookup_does_not_take_the_brief_down(app_mod, monkeypatch):
    m = app_mod
    _capture_brief(m, monkeypatch)

    async def boom(http, day, symbols, *, wait=0.0):
        raise RuntimeError("network gone")
    monkeypatch.setattr(m, "earnings_on", boom)

    with TestClient(m.app) as c:
        r = c.get("/daily_brief")
    assert r.status_code == 200
    assert r.json()["catalysts_complete"] is False


# --- /calendar ------------------------------------------------------------------------------------


def test_the_calendar_names_the_symbols_whose_earnings_lookup_failed(app_mod, monkeypatch):
    """Otherwise a refused lookup reads on the calendar exactly like a company with nothing due."""
    m = app_mod
    from app.news import EarningsLookup

    async def fake_next_earnings(http, sym, *, wait=0.0):
        if sym == "MSFT":
            return EarningsLookup(None, False)
        return EarningsLookup("2026-11-01", True)

    monkeypatch.setattr(m, "fetch_next_earnings", fake_next_earnings)

    async def no_shorts(http, syms, earnings):
        return [{"date": d, "symbol": s, "label": "Earnings", "kind": "earnings"}
                for s, d in earnings.items()]
    monkeypatch.setattr(m.shorts, "calendar", no_shorts)

    with TestClient(m.app) as c:
        body = c.get("/calendar").json()

    assert body["earnings_unchecked"] == ["MSFT"]
    # The BTC halving rides on the same list; only the earnings events are the claim under test.
    earnings_events = {e["symbol"] for e in body["events"] if e["kind"] == "earnings"}
    assert earnings_events == {"AAPL", "NVDA"}


def test_a_fully_read_calendar_reports_nothing_unchecked(app_mod, monkeypatch):
    m = app_mod
    from app.news import EarningsLookup

    async def fake_next_earnings(http, sym, *, wait=0.0):
        return EarningsLookup("2026-11-01", True)
    monkeypatch.setattr(m, "fetch_next_earnings", fake_next_earnings)

    async def no_shorts(http, syms, earnings):
        return []
    monkeypatch.setattr(m.shorts, "calendar", no_shorts)

    with TestClient(m.app) as c:
        assert c.get("/calendar").json()["earnings_unchecked"] == []


def test_an_incomplete_brief_is_not_frozen_for_the_whole_morning(app_mod, monkeypatch):
    """A complete brief is cached ~30 min; one whose catalysts were unreadable must expire far
    sooner, or a single rate-limit blip decides the morning and every retry is served the same
    unknown without ever asking again. Same reasoning as _NEWS_FAIL_TTL_OFFSET."""
    m = app_mod
    _capture_brief(m, monkeypatch)
    complete = {"v": True}

    async def fake_earnings_on(http, day, symbols, *, wait=0.0):
        return (set(), complete["v"])
    monkeypatch.setattr(m, "earnings_on", fake_earnings_on)

    with TestClient(m.app) as c:
        c.get("/daily_brief")
        stamp_complete = next(v[0] for k, v in m._cache.items() if k[0] == "daily_brief")

        m._cache.clear()
        complete["v"] = False
        c.get("/daily_brief")
        stamp_partial = next(v[0] for k, v in m._cache.items() if k[0] == "daily_brief")

    # The partial one is back-dated, so it falls out of the cache sooner.
    assert stamp_partial < stamp_complete
    assert (stamp_complete - stamp_partial) == pytest.approx(
        m._DAILY_BRIEF_TTL - m._BRIEF_INCOMPLETE_TTL, abs=2.0)
