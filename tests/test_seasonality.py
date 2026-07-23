"""Offline tests for seasonality (app/seasonality.py). Yahoo chart fetch is mocked with synthetic bars."""
from __future__ import annotations

import asyncio
import datetime as dt

from app import seasonality as sea


def _chart(monthly):
    """Build a minimal Yahoo chart response from [(year, month, close), ...]."""
    ts = [int(dt.datetime(y, m, 1, tzinfo=dt.timezone.utc).timestamp()) for (y, m, _) in monthly]
    closes = [c for (_, _, c) in monthly]
    return {"chart": {"result": [{"timestamp": ts, "indicators": {"quote": [{"close": closes}]}}]}}


def test_seasonality_detects_a_january_effect(monkeypatch):
    # 4 years where January is always +10% (Dec->Jan), every other month flat.
    bars, price = [], 100.0
    for y in range(2021, 2025):
        for m in range(1, 13):
            if m == 1:
                price *= 1.10
            bars.append((y, m, round(price, 4)))

    async def fake_chart(client, symbol, rng="10y", interval="1mo"):
        return _chart(bars)
    monkeypatch.setattr(sea, "_fetch_chart", fake_chart)

    d = asyncio.run(sea.seasonality(None, "TEST"))
    assert d is not None
    jan = next(m for m in d["months"] if m["name"] == "Jan")
    assert jan["avg_pct"] == 10.0 and jan["hit_rate"] == 100 and jan["n"] == 3   # 2022/23/24 (2021 has no prior)
    assert d["best_month"]["name"] == "Jan"


def test_seasonality_none_when_too_little_history(monkeypatch):
    bars = [(2024, m, 100.0) for m in range(1, 6)]   # only 5 months

    async def fake_chart(client, symbol, rng="10y", interval="1mo"):
        return _chart(bars)
    monkeypatch.setattr(sea, "_fetch_chart", fake_chart)
    assert asyncio.run(sea.seasonality(None, "TEST")) is None


def test_compact_slims_and_drops_empty():
    assert sea.compact(None) is None
    assert sea.compact({"years": 10}) is None     # no current_month → nothing to show
    full = {
        "years": 11,
        "current_month": {"name": "Jul", "avg_pct": 6.8, "hit_rate": 91, "n": 11},
        "best_month": {"name": "Aug", "avg_pct": 7.0, "hit_rate": 67},
        "worst_month": {"name": "Sep", "avg_pct": -2.0, "hit_rate": 40},
        "months": [{"name": "Jan"}],   # verbose field dropped by compact
    }
    c = sea.compact(full)
    assert c["current_month"]["name"] == "Jul" and c["years"] == 11
    assert c["best_month"]["name"] == "Aug" and "months" not in c
