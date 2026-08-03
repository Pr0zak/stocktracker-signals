"""News lookups: "the source failed" must never render as "nothing happened".

Written after GME fell 10.4% on a $1.4B convertible-notes dilution (2026-08-03), the app asked why,
and it answered "No headlines available for this day" — then cached that for an hour. Finnhub had 26
articles, including one titled "Why GameStop Stock Just Slumped". The bug was a bare
`except Exception: return []`, which made an outage and a quiet tape the same value, with no log line
to tell them apart after the fact.

The contract these tests pin down: **None = the lookup failed. [] = it succeeded and there is
genuinely nothing.** Only the second is a claim this service is entitled to make.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from app import news


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setattr(news.settings_store, "get", lambda: {"finnhub_api_key": "test-key"})


def _fetch(handler, symbol: str = "GME"):
    """Run fetch_dated_news against a mocked transport (the repo tests async with asyncio.run rather
    than taking a pytest-asyncio dependency)."""
    async def _run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await news.fetch_dated_news(c, symbol)
    return asyncio.run(_run())


# --------------------------------------------------------------------- failures must be None

def test_a_rate_limit_returns_none_not_empty():
    """THE case. 429 is the likeliest real failure — the app fires ~10 calls when a symbol opens."""
    assert _fetch(lambda r: httpx.Response(429, json={"error": "API limit reached"})) is None


def test_a_server_error_returns_none():
    assert _fetch(lambda r: httpx.Response(502, text="bad gateway")) is None


def test_a_timeout_returns_none():
    def boom(request):
        raise httpx.ReadTimeout("too slow", request=request)
    assert _fetch(boom) is None


def test_a_non_list_payload_returns_none():
    assert _fetch(lambda r: httpx.Response(200, json={"error": "nope"})) is None


def test_a_missing_key_returns_none_not_empty(monkeypatch):
    """No key configured is "we couldn't look", not "there is no news"."""
    monkeypatch.setattr(news.settings_store, "get", lambda: {"finnhub_api_key": ""})
    assert _fetch(lambda r: httpx.Response(200, json=[])) is None


# --------------------------------------------------------------------- success must stay distinct

def test_a_genuinely_quiet_symbol_returns_empty_list():
    """The other side of the contract: a SUCCESSFUL fetch with no articles is a real claim, and has
    to stay distinguishable from a failure so the endpoint can keep making it."""
    assert _fetch(lambda r: httpx.Response(200, json=[]), symbol="QUIETCO") == []


def test_real_articles_are_parsed_newest_first():
    payload = [
        {"headline": "Older", "datetime": 1785600000, "summary": "s", "source": "Reuters", "url": "u"},
        {"headline": "Why GameStop Stock Just Slumped", "datetime": 1785760000,
         "summary": "dilution", "source": "Motley Fool", "url": "u2"},
    ]
    rows = _fetch(lambda r: httpx.Response(200, json=payload))
    assert rows is not None and len(rows) == 2
    assert rows[0]["headline"] == "Why GameStop Stock Just Slumped"


def test_undated_or_headless_articles_are_skipped_without_failing_the_batch():
    payload = [
        {"headline": "", "datetime": 1785760000},
        {"headline": "No timestamp"},
        {"headline": "Good one", "datetime": 1785760000, "summary": "", "source": "", "url": ""},
    ]
    rows = _fetch(lambda r: httpx.Response(200, json=payload))
    assert rows is not None and [r["headline"] for r in rows] == ["Good one"]


# --------------------------------------------------------------------- and it must be diagnosable

def test_failure_is_logged_so_it_stays_diagnosable(caplog):
    """The original incident was unrecoverable after the fact precisely because nothing was written
    down — there is still no way to know whether GME's 08:59 lookup was rate-limited or just early."""
    with caplog.at_level("WARNING", logger="signals.news"):
        _fetch(lambda r: httpx.Response(429, json={}))
    assert any("GME" in r.getMessage() for r in caplog.records), caplog.text
    assert any("not 'no news'" in r.getMessage() for r in caplog.records), caplog.text
