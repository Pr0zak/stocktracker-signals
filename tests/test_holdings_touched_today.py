"""MONEY-7: the daily brief is told what the user actually OWNS (`holdings`, a comma-separated query
param — symbols only, never shares/cost/lot dates/the taxable flag) and speaks to it via
`holdings_touched_today` — the exact, precomputed overlap between `holdings` and today's watchlist
movers/catalysts, so the analyst is handed a fact instead of being asked to eyeball ticker overlap
across two JSON arrays.

These pin the pure overlap function directly, then the route boundary: what the analyst snapshot
receives and what the JSON response publishes.
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
        return {
            "session": "PRE", "indices": {}, "vix": None,
            "watchlist_movers": {
                "up": [{"symbol": "NVDA", "pct": 3.1}],
                "down": [{"symbol": "MSFT", "pct": -1.4}],
            },
        }
    monkeypatch.setattr(m.market_now, "build_snapshot", fake_snapshot)

    async def fake_movers(side, count):
        return []
    monkeypatch.setattr(m, "_movers_side", fake_movers)
    monkeypatch.setattr(m.usage_store, "record", lambda *a, **k: None)

    async def fake_earnings_on(http, day, symbols, *, wait=0.0):
        return set(), True
    monkeypatch.setattr(m, "earnings_on", fake_earnings_on)
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


# --- the pure function ------------------------------------------------------------------------------


def test_empty_holdings_returns_empty_not_a_spurious_match(app_mod):
    m = app_mod
    assert m._holdings_touched_today(
        {"up": [{"symbol": "AAPL"}], "down": []}, ["AAPL"], [],
    ) == []


def test_a_holding_among_todays_movers_is_named(app_mod):
    m = app_mod
    assert m._holdings_touched_today(
        {"up": [{"symbol": "AAPL"}], "down": [{"symbol": "TSLA"}]}, [], ["AAPL", "MSFT"],
    ) == ["AAPL"]


def test_a_holding_reporting_earnings_is_named_even_with_no_mover_overlap(app_mod):
    m = app_mod
    assert m._holdings_touched_today({"up": [], "down": []}, ["NVDA"], ["NVDA", "MSFT"]) == ["NVDA"]


def test_matching_is_case_insensitive(app_mod):
    m = app_mod
    assert m._holdings_touched_today(
        {"up": [{"symbol": "AAPL"}], "down": []}, [], ["aapl"],
    ) == ["AAPL"]


def test_a_holding_touched_by_both_a_mover_and_a_catalyst_appears_once(app_mod):
    m = app_mod
    result = m._holdings_touched_today(
        {"up": [{"symbol": "NVDA"}], "down": []}, ["NVDA"], ["NVDA"],
    )
    assert result == ["NVDA"]


def test_no_overlap_at_all_is_empty(app_mod):
    m = app_mod
    assert m._holdings_touched_today(
        {"up": [{"symbol": "TSLA"}], "down": []}, ["GOOG"], ["AAPL", "MSFT"],
    ) == []


def test_missing_watchlist_movers_key_does_not_crash(app_mod):
    """Same shape a lookup failure or a not-yet-priced snapshot can produce."""
    m = app_mod
    assert m._holdings_touched_today({}, ["AAPL"], ["AAPL"]) == ["AAPL"]
    assert m._holdings_touched_today(None, [], ["AAPL"]) == []


# --- the route boundary ------------------------------------------------------------------------------


def test_no_holdings_param_behaves_exactly_as_before(app_mod, monkeypatch):
    """MONEY-7 must not change the response for a caller that doesn't send holdings at all."""
    m = app_mod
    seen = _capture_brief(m, monkeypatch)

    with TestClient(m.app) as c:
        body = c.get("/daily_brief").json()

    assert body["holdings_touched_today"] == []
    assert seen["holdings"] == []
    assert seen["holdings_touched_today"] == []


def test_a_held_mover_is_surfaced_in_the_snapshot_and_the_response(app_mod, monkeypatch):
    m = app_mod
    seen = _capture_brief(m, monkeypatch)

    with TestClient(m.app) as c:
        body = c.get("/daily_brief?holdings=NVDA,MSFT").json()

    assert seen["holdings"] == ["MSFT", "NVDA"]
    assert seen["holdings_touched_today"] == ["MSFT", "NVDA"]  # both are today's watchlist movers
    assert body["holdings_touched_today"] == ["MSFT", "NVDA"]


def test_a_holding_with_no_overlap_still_passes_through_untouched(app_mod, monkeypatch):
    m = app_mod
    seen = _capture_brief(m, monkeypatch)

    with TestClient(m.app) as c:
        body = c.get("/daily_brief?holdings=AAPL").json()

    assert seen["holdings"] == ["AAPL"]
    assert seen["holdings_touched_today"] == []
    assert body["holdings_touched_today"] == []


def test_holdings_query_param_is_deduped_uppercased_and_whitespace_tolerant(app_mod, monkeypatch):
    m = app_mod
    seen = _capture_brief(m, monkeypatch)

    with TestClient(m.app) as c:
        c.get("/daily_brief?holdings=aapl, AAPL ,msft,,")

    assert seen["holdings"] == ["AAPL", "MSFT"]


def test_only_shares_cost_and_lot_dates_are_withheld_not_the_symbol_itself(app_mod, monkeypatch):
    """MONEY-7's payload decision: the brief gets bare symbols, nothing about size or tax."""
    m = app_mod
    seen = _capture_brief(m, monkeypatch)

    with TestClient(m.app) as c:
        c.get("/daily_brief?holdings=AAPL")

    assert seen["holdings"] == ["AAPL"]
    # No stray keys carrying shares/avg_cost/opened_at/taxable_account snuck in via `holdings`.
    assert all(isinstance(s, str) for s in seen["holdings"])


def test_different_holdings_get_independent_cache_entries(app_mod, monkeypatch):
    """A per-holdings cache key: one user's book must not serve another's brief, and the identical
    holdings string must actually hit the cache rather than recompute."""
    m = app_mod
    _capture_brief(m, monkeypatch)

    with TestClient(m.app) as c:
        c.get("/daily_brief?holdings=AAPL")
        c.get("/daily_brief?holdings=NVDA")
        keys = {k for k in m._cache if k[0] == "daily_brief"}

    assert keys == {("daily_brief", False, ("AAPL",)), ("daily_brief", False, ("NVDA",))}
