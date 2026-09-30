"""ABOUT-1 — company profile (app/profile.py) and GET /profile/{symbol}."""
from __future__ import annotations

import pytest

from app import profile as pf

VICR = {
    "assetProfile": {
        "sector": "Technology", "industry": "Electronic Components", "fullTimeEmployees": 1092,
        "city": "Andover", "state": "MA", "country": "United States", "website": "https://www.vicorpower.com",
        "longBusinessSummary": "Vicor Corporation designs power components. Vicor Corporation was "
                               "incorporated in 1981 and is headquartered in Andover, Massachusetts.",
    },
    "price": {"longName": "Vicor Corporation", "quoteType": "EQUITY", "exchangeName": "NasdaqGS",
              "marketCap": {"raw": 13_320_000_000}},
    "summaryDetail": {"trailingPE": {"raw": 92.91}, "forwardPE": {"raw": 48.62}},
    "defaultKeyStatistics": {"shortPercentOfFloat": {"raw": 0.0976}},
    "financialData": {"revenueGrowth": {"raw": 0.493}, "profitMargins": {"raw": 0.3065},
                      "targetMeanPrice": {"raw": 386.25}, "numberOfAnalystOpinions": {"raw": 4}},
}


def test_parse_reads_the_profile_and_the_plain_figures():
    p = pf.parse("vicr", VICR)
    assert p["symbol"] == "VICR" and p["name"] == "Vicor Corporation"
    assert (p["sector"], p["industry"], p["exchange"]) == ("Technology", "Electronic Components", "NasdaqGS")
    assert p["founded"] == 1981 and p["employees"] == 1092
    assert p["market_cap"] == 13_320_000_000
    assert p["revenue_growth_pct"] == 49.3 and p["profit_margin_pct"] == 30.65
    assert p["short_pct_float"] == 9.76 and p["n_analysts"] == 4


def test_what_yahoo_lacks_is_none_not_zero():
    p = pf.parse("XYZ", {"price": {"shortName": "XYZ Fund", "quoteType": "ETF"},
                         "fundProfile": {"categoryName": "Large Blend"}})
    assert p["category"] == "Large Blend"
    assert p["sector"] is None and p["pe"] is None and p["dividend_yield_pct"] is None
    assert p["founded"] is None and p["n_analysts"] is None


def test_plain_line_is_kept_until_the_description_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "_PLAIN_FILE", tmp_path / "plain.json")
    monkeypatch.setattr(pf, "_plain", None)
    pf.store_plain("VICR", "desc one", "Makes power modules", "Aircraft makers")
    assert pf.cached_plain("VICR", "desc one")["what_it_does"] == "Makes power modules"
    assert pf.cached_plain("VICR", "desc two") is None


def test_endpoint_writes_the_plain_line_once_and_only_when_asked(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from app import analyst
    from app.main import app

    monkeypatch.setattr(pf, "_PLAIN_FILE", tmp_path / "plain.json")
    monkeypatch.setattr(pf, "_plain", None)

    async def fake_facts(client, sym):
        return pf.parse(sym, VICR) if sym == "VICR" else None

    calls = []

    async def fake_plain(name, sector, industry, summary):
        calls.append(name)
        return analyst.PlainProfile(what_it_does="Makes power modules", customers="Aircraft makers"), \
            {"model": "m", "input_tokens": 1, "output_tokens": 1}

    monkeypatch.setattr(pf, "facts", fake_facts)
    monkeypatch.setattr(analyst, "plain_profile", fake_plain)
    monkeypatch.setattr("app.main.usage_store.record", lambda *a, **k: None)
    with TestClient(app) as c:
        assert c.get("/profile/VICR").json()["what_it_does"] is None, "no AI unless asked"
        body = c.get("/profile/VICR?plain=true").json()
        assert body["what_it_does"] == "Makes power modules" and body["customers"] == "Aircraft makers"
        c.get("/profile/VICR?plain=true")
        assert calls == ["Vicor Corporation"], "written once, then served from the cache"
        assert c.get("/profile/NOPE").status_code == 404
        assert c.get("/profile/..%2Fetc").status_code in (404, 422)


def test_a_failed_plain_write_still_returns_the_facts(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from app import analyst
    from app.main import app

    monkeypatch.setattr(pf, "_PLAIN_FILE", tmp_path / "plain.json")
    monkeypatch.setattr(pf, "_plain", None)

    async def fake_facts(client, sym):
        return pf.parse(sym, VICR)

    async def boom(*a, **k):
        raise RuntimeError("model down")

    monkeypatch.setattr(pf, "facts", fake_facts)
    monkeypatch.setattr(analyst, "plain_profile", boom)
    with TestClient(app) as c:
        body = c.get("/profile/VICR?plain=true").json()
    assert body["sector"] == "Technology" and body["what_it_does"] is None
    assert "model down" in body["plain_error"]
