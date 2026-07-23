"""Offline tests for the market-now pulse (app/market_now.py) + analyst.market_overview routing."""
from __future__ import annotations

import asyncio
import datetime as dt
from zoneinfo import ZoneInfo

from app import analyst, llm_cli, market_now as mn, settings_store

ET = ZoneInfo("America/New_York")


def _at(y, mo, d, h, mi):
    return dt.datetime(y, mo, d, h, mi, tzinfo=ET)


def test_session_phases():
    # 2026-07-23 is a Thursday (weekday)
    assert _at(2026, 7, 23, 10, 0).weekday() < 5
    assert mn.session_phase(_at(2026, 7, 23, 5, 0)) == "PRE"
    assert mn.session_phase(_at(2026, 7, 23, 9, 29)) == "PRE"
    assert mn.session_phase(_at(2026, 7, 23, 9, 30)) == "REGULAR"
    assert mn.session_phase(_at(2026, 7, 23, 15, 59)) == "REGULAR"
    assert mn.session_phase(_at(2026, 7, 23, 16, 0)) == "AFTER"
    assert mn.session_phase(_at(2026, 7, 23, 19, 59)) == "AFTER"
    assert mn.session_phase(_at(2026, 7, 23, 20, 0)) == "CLOSED"
    assert mn.session_phase(_at(2026, 7, 23, 2, 0)) == "CLOSED"
    # 2026-07-25 is a Saturday
    assert _at(2026, 7, 25, 10, 0).weekday() == 5
    assert mn.session_phase(_at(2026, 7, 25, 10, 0)) == "CLOSED"


def test_assemble_ranks_sectors_and_watchlist_movers():
    quotes = {
        "^GSPC": {"pct": 0.5, "price": 5000}, "^IXIC": {"pct": 0.8}, "^DJI": {"pct": 0.2},
        "^RUT": {"pct": -0.3}, "^VIX": {"pct": -4.0, "price": 14.2},
        "XLK": {"pct": 1.2}, "XLC": {"pct": 0.9}, "XLY": {"pct": 0.6}, "XLI": {"pct": 0.4},
        "XLF": {"pct": 0.3}, "XLV": {"pct": 0.1}, "XLB": {"pct": 0.0}, "XLRE": {"pct": -0.1},
        "XLP": {"pct": -0.2}, "XLU": {"pct": -0.5}, "XLE": {"pct": -0.9},
        "AAPL": {"pct": 2.1}, "TSLA": {"pct": -3.4}, "NVDA": {"pct": 1.0},
    }
    snap = mn.assemble(quotes, ["AAPL", "TSLA", "NVDA"], "REGULAR", top=3)
    assert snap["session"] == "REGULAR"
    assert snap["vix"] == {"level": 14.2, "pct": -4.0}
    assert snap["sector_leaders"][0] == {"name": "Technology", "symbol": "XLK", "pct": 1.2}
    assert snap["sector_laggards"][0]["symbol"] == "XLE"          # most negative first
    assert snap["watchlist_movers"]["up"][0]["symbol"] == "AAPL"
    assert snap["watchlist_movers"]["down"][0]["symbol"] == "TSLA"
    # a negative name never shows in "up" and vice-versa
    assert all(m["pct"] > 0 for m in snap["watchlist_movers"]["up"])
    assert all(m["pct"] < 0 for m in snap["watchlist_movers"]["down"])


def test_assemble_uses_session_pct_after_hours():
    quotes = {"AAPL": {"pct": 0.1, "post_pct": 3.0}, "MSFT": {"pct": -0.2, "post_pct": -1.5}}
    snap = mn.assemble(quotes, ["AAPL", "MSFT"], "AFTER", top=3)
    ups = {m["symbol"]: m["pct"] for m in snap["watchlist_movers"]["up"]}
    assert ups.get("AAPL") == 3.0     # AFTER → post-market % is what moves it, not the 0.1 regular
    downs = {m["symbol"]: m["pct"] for m in snap["watchlist_movers"]["down"]}
    assert downs.get("MSFT") == -1.5


def test_market_overview_routes_to_cli_scan_tier(monkeypatch):
    monkeypatch.setattr(settings_store, "get",
                        lambda: {"llm_provider": "cli", "deep_model": "claude-opus-4-8", "scan_model": "claude-haiku-4-5"})
    seen = {}

    async def fake_text(system, prompt, *, model, max_tokens=2048, thinking=False):
        seen["model"] = model
        seen["thinking"] = thinking
        return "Risk-on tape; tech leads. Not investment advice.", {"provider": "cli", "model": model}
    monkeypatch.setattr(llm_cli, "text", fake_text)

    txt, u = asyncio.run(analyst.market_overview({"session": "REGULAR"}, deep=False))
    assert u["provider"] == "cli"
    assert seen["model"] == "claude-haiku-4-5"   # scan tier for deep=False
    assert seen["thinking"] is False             # haiku → no thinking
    assert "Not investment advice" in txt


def test_market_overview_deep_uses_opus_with_thinking(monkeypatch):
    monkeypatch.setattr(settings_store, "get",
                        lambda: {"llm_provider": "cli", "deep_model": "claude-opus-4-8", "scan_model": "claude-haiku-4-5"})
    seen = {}

    async def fake_text(system, prompt, *, model, max_tokens=2048, thinking=False):
        seen["model"] = model
        seen["thinking"] = thinking
        return "x. Not investment advice.", {"provider": "cli", "model": model}
    monkeypatch.setattr(llm_cli, "text", fake_text)

    asyncio.run(analyst.market_overview({"session": "REGULAR"}, deep=True))
    assert seen["model"] == "claude-opus-4-8" and seen["thinking"] is True
