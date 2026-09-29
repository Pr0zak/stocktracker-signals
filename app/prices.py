"""
PX-1 — the app's price source.

The phone used to fetch every price itself: Yahoo for stocks, ETFs, charts and the VIX, CoinGecko
for crypto. On 2026-09-29 CoinGecko refused every keyless call from the home network for hours, and
because each phone and each widget asked on its own, nothing could share a good answer or back off
together — four watchlist rows sat 16 hours old under "4 of 59 out of date".

This module makes the service the first place the app asks:

* `chart_passthrough` serves Yahoo's own v8 chart and v1 search responses, byte for byte, from a
  short cache. The app already parses that format in one place, so it reads a served response with
  the same code as a direct one and falls back to Yahoo itself when this service cannot be reached.
  Only those two paths are forwarded; this is not a general proxy.
* `crypto_markets` answers the watchlist's crypto rows: CoinGecko when it will talk to us (with a
  Demo key if `COINGECKO_API_KEY` is set), and a trailing-24-hour row built from Yahoo's
  `<SYM>-USD` chart for any coin it did not answer for.

Every crypto row carries `as_of` and `source`, so the app never has to stamp "now" on a number it
did not just see.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections import OrderedDict
from urllib.parse import parse_qsl, urlencode

import httpx

from . import options

_ALLOWED_PATH = re.compile(r"^(v8/finance/chart/[A-Za-z0-9.\-^=%]{1,32}|v1/finance/search)$")

# Cache lifetimes, by how fast the thing asked for can change. An intraday bar interval means a
# live price is in the answer, so it is kept about as long as the app's own quote memo (15 s).
_INTRADAY_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m"}
_TTL_INTRADAY = 15.0
_TTL_HOURLY = 120.0
_TTL_DAILY = 900.0
_TTL_SEARCH = 3600.0
_CACHE_MAX = 400  # a 1-day/1-minute chart is ~40 KB, so this caps the memo near 16 MB

_cache: "OrderedDict[str, tuple[float, float, int, bytes]]" = OrderedDict()  # key -> (at, ttl, status, body)
_inflight: dict[str, asyncio.Future] = {}
_gate = asyncio.Semaphore(6)


class UpstreamError(Exception):
    """Every Yahoo host failed. The caller answers 502 so the app falls back to asking directly."""


def allowed(path: str) -> bool:
    return bool(_ALLOWED_PATH.match(path))


def _ttl(path: str, params: dict[str, str]) -> float:
    if path.startswith("v1/finance/search"):
        return _TTL_SEARCH
    interval = params.get("interval", "")
    if interval in _INTRADAY_INTERVALS:
        return _TTL_INTRADAY
    if interval == "1h":
        return _TTL_HOURLY
    return _TTL_DAILY


def cache_key(path: str, params: dict[str, str]) -> str:
    """The memo key. `period1`/`period2` are "now"-relative seconds on every crypto and 3-year chart,
    so they are bucketed to the minute here; otherwise no two requests would ever share an entry."""
    norm = {}
    for k, v in sorted(params.items()):
        if k in ("period1", "period2") and v.isdigit():
            v = str(int(v) // 60 * 60)
        norm[k] = v
    return f"{path}?{urlencode(norm)}"


async def _fetch_yahoo(client: httpx.AsyncClient, path: str, params: dict[str, str]) -> tuple[int, bytes]:
    last: Exception | None = None
    async with _gate:
        for host in options._HOSTS:
            try:
                r = await client.get(f"https://{host}/{path}", params=params, headers=options._headers(), timeout=12)
            except Exception as e:  # noqa: BLE001 — try the other host
                last = e
                continue
            # A 404 is Yahoo saying the symbol does not exist. That is an answer, and its body is the
            # JSON error the app already reads as "no data" — pass it through rather than retry.
            if r.status_code in (200, 404):
                return r.status_code, r.content
            last = RuntimeError(f"Yahoo HTTP {r.status_code}")
            # A rate limit is about us, not about query1; asking query2 would only double the load.
            if r.status_code == 429:
                break
    raise UpstreamError(str(last))


async def chart_passthrough(client: httpx.AsyncClient, path: str, query: str) -> tuple[int, bytes, bool]:
    """(status, body, cached) for one allowed Yahoo path. Concurrent identical requests share one
    upstream fetch. Raises UpstreamError when Yahoo could not be reached."""
    params = dict(parse_qsl(query, keep_blank_values=True))
    key = cache_key(path, params)
    now = time.monotonic()
    hit = _cache.get(key)
    if hit and now - hit[0] < hit[1]:
        _cache.move_to_end(key)
        return hit[2], hit[3], True
    fut = _inflight.get(key)
    if fut is not None:
        status, body = await asyncio.shield(fut)
        return status, body, True
    fut = asyncio.get_running_loop().create_future()
    _inflight[key] = fut
    try:
        status, body = await _fetch_yahoo(client, path, params)
        fut.set_result((status, body))
    except BaseException as e:
        fut.set_exception(e)
        fut.exception()  # mark retrieved: nobody else may be waiting
        raise
    finally:
        _inflight.pop(key, None)
    if status == 200:
        _cache[key] = (time.monotonic(), _ttl(path, params), status, body)
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return status, body, False


# ---- crypto -----------------------------------------------------------------------------------

_CG_BASE = "https://api.coingecko.com/api/v3"
_CG_BACKOFF_S = 600.0  # after a 429, leave CoinGecko alone this long rather than keep being refused
_CG_TTL = 30.0
_MAX_BAR_AGE_S = 30 * 60

_cg_blocked_until = 0.0
_cg_blocked_key = ""
_crypto_cache: dict[str, tuple[float, list[dict]]] = {}


def coingecko_key() -> str:
    """The key set from the app (settings.json) wins; the env var is the fallback for a fresh CT."""
    from . import settings_store
    return (settings_store.get().get("coingecko_api_key") or os.environ.get("COINGECKO_API_KEY", "")).strip()


def _blocked() -> bool:
    """In the back-off — but only for the key that was refused. Saving a new key retries at once."""
    return time.time() < _cg_blocked_until and _cg_blocked_key == coingecko_key()


def row_from_bars(coin_id: str, symbol: str, bars: list[tuple[int, float]], now_s: float) -> dict | None:
    """A trailing-24h row from (epoch_s, close) bars: the last close is the price, the first bar (about
    a day back) the base. None when there is too little to measure, or when the newest bar is over 30
    minutes old — an old tape is not a current price."""
    if len(bars) < 2:
        return None
    last_t, last = bars[-1]
    base = bars[0][1]
    if now_s - last_t > _MAX_BAR_AGE_S or base <= 0 or last <= 0:
        return None
    change = last - base
    return {
        "id": coin_id, "symbol": symbol.upper(), "price": last, "change": change,
        "change_percent": change / base * 100.0, "sparkline": [p for _, p in bars],
        # When the source was READ, like the CoinGecko rows and the app's own quotes — not the bar's
        # start, which is up to five minutes behind by construction and would mark every row stale.
        "as_of": float(now_s), "source": "yahoo",
    }


async def _yahoo_row(client: httpx.AsyncClient, coin_id: str, symbol: str) -> dict | None:
    now = int(time.time())
    path = f"v8/finance/chart/{symbol.upper()}-USD"
    try:
        status, body, _ = await chart_passthrough(
            client, path, urlencode({"period1": now - 86_400, "period2": now, "interval": "5m"}))
    except UpstreamError:
        return None
    if status != 200:
        return None
    try:
        res = (json.loads(body).get("chart") or {}).get("result") or []
        r0 = res[0]
        ts = r0.get("timestamp") or []
        closes = ((r0.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
    except (ValueError, IndexError, AttributeError):
        return None
    bars = [(t, c) for t, c in zip(ts, closes) if c is not None]
    return row_from_bars(coin_id, symbol, bars, time.time())


async def _coingecko_rows(client: httpx.AsyncClient, ids: list[str]) -> dict[str, dict]:
    global _cg_blocked_until, _cg_blocked_key
    if _blocked():
        return {}
    headers = {"Accept": "application/json"}
    if coingecko_key():
        headers["x-cg-demo-api-key"] = coingecko_key()
    try:
        r = await client.get(
            f"{_CG_BASE}/coins/markets",
            params={"vs_currency": "usd", "ids": ",".join(ids), "sparkline": "true",
                    "price_change_percentage": "24h"},
            headers=headers, timeout=12,
        )
    except Exception:  # noqa: BLE001 — Yahoo covers it
        return {}
    if r.status_code == 429 or r.status_code == 403:
        _cg_blocked_until = time.time() + _CG_BACKOFF_S
        _cg_blocked_key = coingecko_key()
        return {}
    if r.status_code != 200:
        return {}
    out: dict[str, dict] = {}
    now = time.time()
    for c in r.json() or []:
        price = c.get("current_price")
        if not isinstance(price, (int, float)) or price <= 0:
            continue
        pct = c.get("price_change_percentage_24h")
        chg = c.get("price_change_24h")
        # The percentage is the source of truth; the dollar change is derived from it, exactly as the
        # app's own CoinGecko path does (the two fields are not always from the same snapshot).
        if isinstance(pct, (int, float)) and pct > -100:
            chg = price - price / (1 + pct / 100.0)
        elif isinstance(chg, (int, float)) and price - chg > 0:
            pct = chg / (price - chg) * 100.0
        else:
            chg, pct = 0.0, 0.0
        out[c.get("id", "")] = {
            "id": c.get("id", ""), "symbol": (c.get("symbol") or "").upper(), "price": float(price),
            "change": float(chg), "change_percent": float(pct),
            "sparkline": ((c.get("sparkline_in_7d") or {}).get("price") or [])[-168:],
            "as_of": now, "source": "coingecko",
        }
    return out


async def crypto_markets(client: httpx.AsyncClient, pairs: list[tuple[str, str]]) -> list[dict]:
    """Rows for (coingecko_id, symbol) pairs. A coin neither source can price is left out, and the
    app treats a missing row as "could not refresh" — never as a zero."""
    key = ",".join(sorted(f"{i}:{s}" for i, s in pairs))
    hit = _crypto_cache.get(key)
    if hit and time.monotonic() - hit[0] < _CG_TTL:
        return hit[1]
    got = await _coingecko_rows(client, [i for i, _ in pairs])
    missing = [(i, s) for i, s in pairs if i not in got]
    if missing:
        rows = await asyncio.gather(*(_yahoo_row(client, i, s) for i, s in missing))
        for (i, _), row in zip(missing, rows):
            if row:
                got[i] = row
    out = [got[i] for i, _ in pairs if i in got]
    if out:
        _crypto_cache[key] = (time.monotonic(), out)
    return out


def coingecko_status() -> str:
    """For the ops page: whether CoinGecko is in its backoff, and whether a key is set."""
    keyed = "Demo key" if coingecko_key() else "no key"
    left = _cg_blocked_until - time.time()
    return f"refused, retrying in {int(left // 60) + 1}m ({keyed})" if _blocked() else f"ok ({keyed})"
