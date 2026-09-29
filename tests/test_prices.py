"""PX-1 — the app's price source (app/prices.py): the Yahoo pass-through and the crypto rows."""
from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

from app import prices


@pytest.fixture(autouse=True)
def _fresh():
    prices._cache.clear()
    prices._inflight.clear()
    prices._crypto_cache.clear()
    prices._cg_blocked_until = 0.0
    prices._cg_blocked_key = ""
    yield


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_only_chart_and_search_are_forwarded():
    assert prices.allowed("v8/finance/chart/AAPL")
    assert prices.allowed("v8/finance/chart/BRK-B")
    assert prices.allowed("v8/finance/chart/^VIX")
    assert prices.allowed("v1/finance/search")
    assert not prices.allowed("v7/finance/quote")
    assert not prices.allowed("v8/finance/chart/../../v1/test/getcrumb")
    assert not prices.allowed("v8/finance/chart/AAPL/extra")
    assert not prices.allowed("http://evil/")


def test_now_relative_periods_share_a_minute_bucket():
    a = prices.cache_key("v8/finance/chart/BTC-USD", {"period1": "1790600041", "period2": "1790686441", "interval": "5m"})
    b = prices.cache_key("v8/finance/chart/BTC-USD", {"period1": "1790600059", "period2": "1790686459", "interval": "5m"})
    assert a == b


def test_passthrough_caches_and_passes_a_404_through():
    calls = []

    def handler(req: httpx.Request):
        calls.append(str(req.url))
        if "NOPE" in req.url.path:
            return httpx.Response(404, json={"chart": {"result": None, "error": {"code": "Not Found"}}})
        return httpx.Response(200, json={"chart": {"result": [{"meta": {"symbol": "AAPL"}}]}})

    async def run():
        async with _client(handler) as c:
            s1, b1, hit1 = await prices.chart_passthrough(c, "v8/finance/chart/AAPL", "range=1d&interval=1m")
            s2, b2, hit2 = await prices.chart_passthrough(c, "v8/finance/chart/AAPL", "range=1d&interval=1m")
            s3, b3, _ = await prices.chart_passthrough(c, "v8/finance/chart/NOPE", "range=1d&interval=1m")
            return (s1, hit1, s2, hit2, b1 == b2, s3, json.loads(b3))

    s1, hit1, s2, hit2, same, s3, b3 = asyncio.run(run())
    assert (s1, hit1, s2, hit2, same) == (200, False, 200, True, True)
    assert s3 == 404 and b3["chart"]["error"]["code"] == "Not Found"
    assert len([u for u in calls if "AAPL" in u]) == 1


def test_a_rate_limit_is_not_retried_on_the_other_host():
    calls = []

    def handler(req: httpx.Request):
        calls.append(req.url.host)
        return httpx.Response(429, text="Too Many Requests")

    async def run():
        async with _client(handler) as c:
            await prices.chart_passthrough(c, "v8/finance/chart/AAPL", "range=1d&interval=1m")

    with pytest.raises(prices.UpstreamError):
        asyncio.run(run())
    assert calls == ["query1.finance.yahoo.com"]


def test_a_server_error_fails_over_to_query2():
    def handler(req: httpx.Request):
        if req.url.host.startswith("query1"):
            return httpx.Response(503)
        return httpx.Response(200, json={"chart": {"result": []}})

    async def run():
        async with _client(handler) as c:
            return await prices.chart_passthrough(c, "v8/finance/chart/AAPL", "range=1y&interval=1d")

    status, _, _ = asyncio.run(run())
    assert status == 200


def test_concurrent_identical_requests_share_one_fetch():
    calls = []

    async def handler(req: httpx.Request):
        calls.append(1)
        await asyncio.sleep(0.05)
        return httpx.Response(200, json={"chart": {"result": []}})

    async def run():
        async with _client(handler) as c:
            return await asyncio.gather(*(
                prices.chart_passthrough(c, "v8/finance/chart/SPY", "range=1d&interval=1m") for _ in range(5)))

    out = asyncio.run(run())
    assert len(calls) == 1 and all(o[0] == 200 for o in out)


def test_row_from_bars_measures_a_trailing_day_and_refuses_an_old_tape():
    now = 1_790_700_000.0
    row = prices.row_from_bars("bitcoin", "btc", [(int(now) - 86_400, 80_000.0), (int(now) - 300, 82_000.0)], now)
    assert row["symbol"] == "BTC" and row["price"] == 82_000.0
    assert row["change"] == pytest.approx(2_000.0) and row["change_percent"] == pytest.approx(2.5)
    assert row["source"] == "yahoo" and row["as_of"] == now
    assert prices.row_from_bars("bitcoin", "BTC", [(int(now) - 90_000, 1.0), (int(now) - 7200, 2.0)], now) is None
    assert prices.row_from_bars("bitcoin", "BTC", [(int(now) - 60, 1.0)], now) is None


def test_crypto_falls_back_to_yahoo_when_coingecko_refuses_and_then_backs_off():
    gecko_calls = []

    def handler(req: httpx.Request):
        if "coingecko" in req.url.host:
            gecko_calls.append(1)
            return httpx.Response(429, json={"status": {"error_code": 429}})
        now = int(time.time())
        return httpx.Response(200, json={"chart": {"result": [{
            "timestamp": [now - 86_000, now - 120],
            "indicators": {"quote": [{"close": [100.0, 110.0]}]},
        }]}})

    async def run():
        async with _client(handler) as c:
            first = await prices.crypto_markets(c, [("bitcoin", "BTC"), ("ethereum", "ETH")])
            prices._crypto_cache.clear()
            await prices.crypto_markets(c, [("bitcoin", "BTC")])
            return first

    rows = asyncio.run(run())
    assert [r["id"] for r in rows] == ["bitcoin", "ethereum"]
    assert all(r["source"] == "yahoo" and r["change_percent"] == pytest.approx(10.0) for r in rows)
    assert len(gecko_calls) == 1  # the second call stayed inside the backoff
    assert "refused" in prices.coingecko_status()


def test_coingecko_rows_derive_the_dollar_change_from_the_percentage():
    def handler(req: httpx.Request):
        return httpx.Response(200, json=[{
            "id": "ethereum", "symbol": "eth", "current_price": 2694.89,
            "price_change_24h": -0.44, "price_change_percentage_24h": 0.05,
            "sparkline_in_7d": {"price": [1.0, 2.0]},
        }])

    async def run():
        async with _client(handler) as c:
            return await prices.crypto_markets(c, [("ethereum", "ETH")])

    (row,) = asyncio.run(run())
    assert row["source"] == "coingecko" and row["change"] > 0
    assert row["change_percent"] == pytest.approx(0.05)


def test_a_coin_nobody_can_price_is_absent_not_zero():
    def handler(req: httpx.Request):
        if "coingecko" in req.url.host:
            return httpx.Response(200, json=[])
        return httpx.Response(404, json={"chart": {"result": None, "error": {"code": "Not Found"}}})

    async def run():
        async with _client(handler) as c:
            return await prices.crypto_markets(c, [("nocoin", "NOPE")])

    assert asyncio.run(run()) == []


def test_routes_refuse_other_paths_and_serve_cached_bytes(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app

    async def fake(client, path, query):
        assert path == "v8/finance/chart/AAPL" and "interval=1m" in query
        return 200, b'{"chart":{"result":[]}}', True

    async def fake_crypto(client, pairs):
        assert pairs == [("bitcoin", "BTC")]
        return []

    monkeypatch.setattr(prices, "chart_passthrough", fake)
    monkeypatch.setattr(prices, "crypto_markets", fake_crypto)
    with TestClient(app) as c:
        assert c.get("/prices/yahoo/v7/finance/quote?symbols=AAPL").status_code == 404
        r = c.get("/prices/yahoo/v8/finance/chart/AAPL?range=1d&interval=1m")
        assert r.status_code == 200 and r.headers["X-Price-Cache"] == "hit"
        assert r.json() == {"chart": {"result": []}}
        assert c.get("/prices/crypto?coins=bitcoin:btc").json()["rows"] == []
        assert c.get("/prices/crypto?coins=../etc:BTC").status_code == 422


def test_the_demo_key_is_sent_and_a_new_key_skips_the_old_backoff(monkeypatch):
    from app import settings_store
    key = {"v": ""}
    monkeypatch.setattr(settings_store, "get", lambda: {"coingecko_api_key": key["v"]})
    monkeypatch.delenv("COINGECKO_API_KEY", raising=False)
    seen = []

    def handler(req: httpx.Request):
        if "coingecko" in req.url.host:
            k = req.headers.get("x-cg-demo-api-key")
            seen.append(k)
            if not k:
                return httpx.Response(429, json={})
            return httpx.Response(200, json=[{"id": "bitcoin", "symbol": "btc", "current_price": 100.0,
                                              "price_change_percentage_24h": 1.0}])
        return httpx.Response(503)

    async def run():
        async with _client(handler) as c:
            first = await prices.crypto_markets(c, [("bitcoin", "BTC")])
            key["v"] = "CG-demo-abcd"
            prices._crypto_cache.clear()
            second = await prices.crypto_markets(c, [("bitcoin", "BTC")])
            return first, second

    first, second = asyncio.run(run())
    assert first == []
    assert seen == [None, "CG-demo-abcd"]
    assert second[0]["source"] == "coingecko"
    assert prices.coingecko_status() == "ok (Demo key)"


def test_settings_report_the_key_without_ever_returning_it(monkeypatch):
    from fastapi.testclient import TestClient
    from app import settings_store
    from app.main import app

    state = {"coingecko_api_key": ""}
    real_get = settings_store.get
    monkeypatch.setattr(settings_store, "get", lambda: {**real_get(), **state})

    def fake_update(patch, **kw):
        if patch.get("coingecko_api_key"):
            state["coingecko_api_key"] = patch["coingecko_api_key"]
        if patch.get("clear_coingecko_api_key"):
            state["coingecko_api_key"] = ""
        return {}

    monkeypatch.setattr(settings_store, "update", fake_update)
    with TestClient(app) as c:
        r = c.post("/api/settings", json={"coingecko_api_key": "CG-secretvalue9z"})
        assert r.status_code == 200
        body = r.json()
        assert body["coingecko_api_key_set"] is True and body["coingecko_api_key_hint"] == "…ue9z"
        assert "CG-secretvalue9z" not in r.text
        body = c.post("/api/settings", json={"clear_coingecko_api_key": True}).json()
        assert body["coingecko_api_key_set"] is False


def test_store_saves_ignores_blank_and_clears_the_key(monkeypatch):
    from app import settings_store as ss
    monkeypatch.setattr(ss, "_current", {"coingecko_api_key": ""})
    monkeypatch.setattr(ss, "_atomic_write_json", lambda *a, **k: None)
    ss.update({"coingecko_api_key": " CG-abc123 "})
    assert ss.get()["coingecko_api_key"] == "CG-abc123"
    ss.update({"coingecko_api_key": ""})
    assert ss.get()["coingecko_api_key"] == "CG-abc123"
    ss.update({"clear_coingecko_api_key": True})
    assert ss.get()["coingecko_api_key"] == ""
