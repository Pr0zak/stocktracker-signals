"""
Webull unofficial market-data fallback for symbols Yahoo doesn't carry (warrants, OTC, some ADRs).

No auth — uses Webull's public app quote endpoints. Resolves a symbol to Webull's tickerId via
search, then pulls daily bars. FALLBACK ONLY: Yahoo stays primary everywhere; these are
reverse-engineered app endpoints (fragile, personal-use), so results are cached hard and this is
only reached for symbols Yahoo already failed on.

Bar CSV format from the charts endpoint: `ts,open,close,high,low,preClose,volume,vwap`, newest-first.
"""
from __future__ import annotations

import re
import time

import httpx

_HDRS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept": "application/json",
    "did": "stocktracker0000000000000000000",  # any stable device id works for market data
    "App": "global",
}
_SEARCH = "https://quotes-gw.webullfintech.com/api/search/pc/tickers"
_CHARTS = "https://quotes-gw.webullfintech.com/api/quote/charts/query"

_id_cache: dict[str, tuple[float, int | None]] = {}
_bars_cache: dict[str, tuple[float, list]] = {}
_ID_TTL = 7 * 86400   # tickerId is stable
_BARS_TTL = 3600      # daily bars: 1h


def _norm(s: str) -> str:
    """Loose symbol key: strip everything but A-Z0-9 so 'GME.WS' == 'GME WS' == 'GME-WS'."""
    return re.sub(r"[^A-Z0-9]", "", s.upper())


async def resolve_ticker_id(client: httpx.AsyncClient, symbol: str) -> int | None:
    key = _norm(symbol)
    hit = _id_cache.get(key)
    if hit and time.time() - hit[0] < _ID_TTL:
        return hit[1]
    tid: int | None = None
    try:
        kw = re.split(r"[.\- ]", symbol)[0]  # search the base symbol to surface warrant/class variants
        r = await client.get(
            _SEARCH,
            params={"keyword": kw, "pageIndex": 1, "pageSize": 20, "regionId": 6},
            headers=_HDRS, timeout=12,
        )
        r.raise_for_status()
        for t in r.json().get("data", []):
            if _norm(str(t.get("symbol", ""))) == key and t.get("tickerId"):
                tid = int(t["tickerId"])
                break
    except Exception:  # noqa: BLE001
        return hit[1] if hit else None
    _id_cache[key] = (time.time(), tid)
    return tid


async def history(client: httpx.AsyncClient, symbol: str, count: int = 800) -> list[dict] | None:
    """Daily bars oldest-first as [{t: epoch_ms, o, h, l, c, v}], or None if unavailable."""
    key = _norm(symbol)
    hit = _bars_cache.get(key)
    if hit and time.time() - hit[0] < _BARS_TTL:
        return hit[1]
    tid = await resolve_ticker_id(client, symbol)
    if tid is None:
        return None
    try:
        r = await client.get(
            _CHARTS, params={"tickerIds": tid, "type": "d1", "count": count}, headers=_HDRS, timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        rows = (data[0] if isinstance(data, list) and data else {}).get("data", [])
    except Exception:  # noqa: BLE001
        return hit[1] if hit else None
    bars: list[dict] = []
    for row in rows:
        p = row.split(",")
        if len(p) < 7:
            continue
        try:
            bars.append({
                "t": int(p[0]) * 1000,
                "o": float(p[1]), "c": float(p[2]), "h": float(p[3]), "l": float(p[4]),
                "v": float(p[6]),
            })
        except ValueError:
            continue
    bars.sort(key=lambda b: b["t"])  # oldest-first for charting
    if not bars:
        return None
    _bars_cache[key] = (time.time(), bars)
    return bars
