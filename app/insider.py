"""
Insider buying (SEC Form 4, via Finnhub's free tier) — the BULLISH informed-money mirror of the
short-pressure block. Surfaces OPEN-MARKET purchases (transaction code 'P') over the trailing 12
months with conviction + cluster flags. Free; no-op without a Finnhub key or for crypto. Cached
per symbol (12h), so a watchlist sweep costs one call per symbol per half-day.

Finnhub contract (verified against the live free tier): /stock/insider-transactions returns
`data: [{name, share, change, transactionCode, transactionPrice, transactionDate, filingDate, ...}]`.
`change` is the SHARES TRANSACTED (delta); `share` is the resulting holding — so the dollar value of
a buy is `abs(change) * transactionPrice`, NOT `share * price`.
"""
from __future__ import annotations

import datetime as dt
import time
from datetime import date

import httpx

from . import settings_store

_BASE = "https://finnhub.io/api/v1"
_cache: dict[str, tuple[float, dict | None]] = {}
_TTL = 12 * 3600
_CONVICTION_USD = 500_000    # mungbeans' open-market conviction floor
_ALWAYS_CONVICTION_USD = 2_000_000


def _has_cluster(buys: list[dict]) -> bool:
    """3+ DISTINCT insiders buying within any rolling 30-day window."""
    dated = sorted((date.fromisoformat(b["date"]), b["name"]) for b in buys)
    for i, (d0, _) in enumerate(dated):
        names = {n for (d, n) in dated[i:] if (d - d0).days <= 30}
        if len(names) >= 3:
            return True
    return False


async def insider_buying(client: httpx.AsyncClient, symbol: str) -> dict | None:
    key = settings_store.get().get("finnhub_api_key", "")
    if not key:
        return None
    sym = symbol.upper()
    hit = _cache.get(sym)
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]

    out: dict | None = None
    try:
        r = await client.get(
            f"{_BASE}/stock/insider-transactions",
            params={"symbol": sym, "token": key}, timeout=15,
        )
        r.raise_for_status()
        body = r.json()
        data = body.get("data", []) if isinstance(body, dict) else []
        cutoff = (dt.date.today() - dt.timedelta(days=365)).isoformat()
        buys: list[dict] = []
        for t in data:
            if t.get("transactionCode") != "P":  # open-market PURCHASE only
                continue
            d = str(t.get("transactionDate") or "")
            if not d or d < cutoff:  # ISO dates compare lexicographically
                continue
            qty = abs(t.get("change") or 0)
            price = t.get("transactionPrice") or 0
            value = int(qty * price)
            if value <= 0:
                continue
            buys.append({"name": t.get("name", ""), "date": d, "shares": int(qty), "value": value})
        if buys:
            buys.sort(key=lambda b: b["date"], reverse=True)
            total = sum(b["value"] for b in buys)
            largest = max(b["value"] for b in buys)
            out = {
                "buy_count_12m": len(buys),
                "buy_total_12m": total,
                "largest_buy_value": largest,
                "has_conviction_buy": largest >= _CONVICTION_USD or total >= _ALWAYS_CONVICTION_USD,
                "has_cluster_buy": _has_cluster(buys),
                "latest_buys": buys[:3],
            }
        else:
            out = {"buy_count_12m": 0}
    except Exception:  # noqa: BLE001 — smart-money context is enrichment, never a blocker
        out = None
    _cache[sym] = (time.time(), out)
    return out


def compact(data: dict | None) -> dict | None:
    """Slim block for the analyst snapshot — only when there were actual open-market buys."""
    if not data or data.get("buy_count_12m", 0) == 0:
        return None
    return {
        "buy_count_12m": data["buy_count_12m"],
        "buy_total_12m": data["buy_total_12m"],
        "largest_buy_value": data["largest_buy_value"],
        "has_conviction_buy": data["has_conviction_buy"],
        "has_cluster_buy": data["has_cluster_buy"],
    }
