"""
Market-wide candidate discovery for /recommendations scope="market".

Pulls live candidates from Yahoo's predefined screeners — four deliberately different angles
(actives, day gainers, growth tech, undervalued large caps) — round-robins across them for
diversity, filters obvious junk (sub-$5 price, sub-$2B equities, non-EQUITY/ETF quote types),
dedupes against the user's watchlist, and caps the pool. Falls back to a small curated universe of
mega-caps + core ETFs when the screener API is unreachable, so market mode always works.
"""
from __future__ import annotations

import asyncio
from itertools import zip_longest

import httpx

_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
_SCREENS = ("most_actives", "day_gainers", "growth_technology_stocks", "undervalued_large_caps")

# Deterministic fallback when the screeners are unreachable: mega-caps + core sector/index ETFs.
FALLBACK = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "TSLA", "AMD", "CRM",
    "JPM", "V", "UNH", "XOM", "COST", "LLY", "HD", "KO", "PEP", "MRK",
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLE", "XLF", "XLV", "SMH", "GLD",
]


async def _screen(client: httpx.AsyncClient, scr: str, count: int = 15) -> list[dict]:
    for host in ("query1", "query2"):
        try:
            r = await client.get(
                f"https://{host}.finance.yahoo.com/v1/finance/screener/predefined/saved",
                params={"scrIds": scr, "count": count},
                headers=_UA,
                timeout=10,
            )
            r.raise_for_status()
            return r.json()["finance"]["result"][0]["quotes"]
        except Exception:  # noqa: BLE001 — try the next host / return empty
            continue
    return []


def _raw(v) -> float:
    """Screener fields are usually raw numbers but occasionally {raw: ..., fmt: ...}."""
    if isinstance(v, dict):
        v = v.get("raw", 0)
    return float(v or 0)


async def discover(client: httpx.AsyncClient, exclude: set[str], cap: int = 15) -> list[str]:
    """Candidate symbols beyond the watchlist, most interesting first. Never raises."""
    try:
        quote_lists = await asyncio.gather(*[_screen(client, s) for s in _SCREENS])
    except Exception:  # noqa: BLE001
        quote_lists = []

    seen: set[str] = set()
    out: list[str] = []
    # Round-robin across the four screens so no single angle dominates the capped pool.
    for group in zip_longest(*quote_lists):
        for q in group:
            if q is None:
                continue
            sym = str(q.get("symbol", "")).upper()
            if not sym or sym in seen or sym in exclude:
                continue
            qt = q.get("quoteType", "EQUITY")
            if qt not in ("EQUITY", "ETF"):
                continue
            price = _raw(q.get("regularMarketPrice"))
            mktcap = _raw(q.get("marketCap"))
            if price and price < 5:  # skip penny-ish names
                continue
            if qt == "EQUITY" and mktcap and mktcap < 2_000_000_000:  # skip micro caps
                continue
            seen.add(sym)
            out.append(sym)
            if len(out) >= cap:
                return out
    if len(out) < 5:  # screeners down/blocked — use the curated universe
        out = [s for s in FALLBACK if s not in exclude]
    return out[:cap]
