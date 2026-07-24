"""
Optional news + earnings context for the analyst (Tier 3), via Finnhub's free tier. Adds recent
headlines and the next earnings date to a stock's snapshot so Claude can ground the `catalysts` and
`key_risks` fields in real events. No-op (empty) when no Finnhub key is configured, or for crypto.
"""
from __future__ import annotations

import datetime as dt

import httpx

from . import settings_store

_BASE = "https://finnhub.io/api/v1"


async def fetch_context(client: httpx.AsyncClient, symbol: str) -> dict:
    key = settings_store.get().get("finnhub_api_key", "")
    if not key:
        return {}
    out: dict = {}
    today = dt.date.today()

    try:  # recent company news — a few latest headlines
        r = await client.get(
            f"{_BASE}/company-news",
            params={"symbol": symbol, "from": (today - dt.timedelta(days=10)).isoformat(),
                    "to": today.isoformat(), "token": key},
            timeout=12,
        )
        r.raise_for_status()
        news = r.json()
        if isinstance(news, list):
            heads = [n.get("headline", "").strip() for n in news if n.get("headline")]
            if heads:
                out["recent_news"] = heads[:5]
    except Exception:  # noqa: BLE001 — decorative context; never fail the verdict on it
        pass

    try:  # next scheduled earnings date within ~90 days
        r = await client.get(
            f"{_BASE}/calendar/earnings",
            params={"from": today.isoformat(), "to": (today + dt.timedelta(days=90)).isoformat(),
                    "symbol": symbol, "token": key},
            timeout=12,
        )
        r.raise_for_status()
        cal = r.json().get("earningsCalendar") or []
        if cal:
            out["next_earnings"] = min(cal, key=lambda e: e.get("date", "9999")).get("date")
    except Exception:  # noqa: BLE001
        pass

    return out


async def fetch_dated_news(client: httpx.AsyncClient, symbol: str, days: int = 16) -> list[dict]:
    """Company news over the last [days], each carrying its ET date so the analyst can line headlines
    up against specific price moves (AIE-4). Returns [{date: YYYY-MM-DD, headline, summary, source, url}]
    newest-first. Empty when no Finnhub key, on any failure, or for a symbol with no coverage."""
    key = settings_store.get().get("finnhub_api_key", "")
    if not key:
        return []
    today = dt.date.today()
    try:
        r = await client.get(
            f"{_BASE}/company-news",
            params={"symbol": symbol, "from": (today - dt.timedelta(days=days)).isoformat(),
                    "to": today.isoformat(), "token": key},
            timeout=12,
        )
        r.raise_for_status()
        raw = r.json()
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for n in raw:
        head = (n.get("headline") or "").strip()
        ts = n.get("datetime")
        if not head or not ts:
            continue
        try:
            date = dt.datetime.utcfromtimestamp(int(ts)).date().isoformat()
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
