"""Sector / industry classification, cached on disk.

The heat map was a FLAT treemap: 80 tiles in market-cap order with no structure, so JNJ sat between
ASML and INTC and there was no way to read "tech is red, energy is green" at a glance. Sector blocks
are the defining feature of the finviz-style map the user is comparing against, and the classification
simply was not in the pipeline — universe rows carry symbol/name/is_etf/market_cap and nothing else.

Source is Yahoo `quoteSummary?modules=assetProfile`, which returns the same taxonomy finviz uses
(Technology / Semiconductors, Energy / Oil & Gas Integrated, Healthcare / Drug Manufacturers). It
needs the cookie+crumb dance, which `options._ensure_auth` already owns.

Cached to disk with a LONG ttl: a company's sector changes approximately never, so this is a
write-once-per-symbol cost rather than a per-request one. A lookup that fails is cached as a
NEGATIVE result for a short while only — so an outage doesn't permanently pin a symbol to "unknown",
which on the map means being dumped in an "Other" block forever.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path

import httpx

from . import options

log = logging.getLogger("signals.sectors")

_DATA_DIR = Path(os.environ.get("SIGNALS_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
_FILE = _DATA_DIR / "sectors.json"

# Sector membership is effectively static. Re-checking weekly would be pure waste.
TTL_SECONDS = 90 * 24 * 3600
# A failed lookup is remembered only briefly — long enough to stop hammering a broken endpoint,
# short enough that a transient failure doesn't strand the symbol in "Other" for months.
MISS_TTL_SECONDS = 6 * 3600

# Concurrency against Yahoo. quoteSummary is one call per symbol and the map asks for up to 200, so
# this is the difference between a slow first paint and a rate-limit.
_CONCURRENCY = 6

_cache: dict[str, dict] | None = None


def _load() -> dict[str, dict]:
    global _cache
    if _cache is not None:
        return _cache
    try:
        _cache = json.loads(_FILE.read_text()) if _FILE.exists() else {}
    except Exception:  # noqa: BLE001 — a corrupt cache is just a cold one
        log.warning("sectors: cache unreadable, starting cold", exc_info=True)
        _cache = {}
    return _cache


def _save() -> None:
    try:
        _FILE.parent.mkdir(parents=True, exist_ok=True)
        _FILE.write_text(json.dumps(_cache or {}))
    except Exception:  # noqa: BLE001 — the cache is an optimisation, never a blocker
        log.warning("sectors: cache write failed", exc_info=True)


def _fresh(row: dict, now: float) -> bool:
    ttl = TTL_SECONDS if row.get("sector") else MISS_TTL_SECONDS
    return (now - float(row.get("ts") or 0)) < ttl


async def _fetch_one(client: httpx.AsyncClient, sym: str, crumb: str) -> dict:
    try:
        r = await client.get(
            f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{sym}",
            params={"modules": "assetProfile", "crumb": crumb},
            headers=options._headers(), timeout=20,
        )
        r.raise_for_status()
        ap = (r.json()["quoteSummary"]["result"] or [{}])[0].get("assetProfile") or {}
        return {"sector": (ap.get("sector") or "").strip() or None,
                "industry": (ap.get("industry") or "").strip() or None,
                "ts": time.time()}
    except Exception:  # noqa: BLE001 — one unknown symbol must not fail the batch
        return {"sector": None, "industry": None, "ts": time.time()}


async def lookup(client: httpx.AsyncClient, symbols: list[str]) -> dict[str, dict]:
    """{SYMBOL: {sector, industry}} for every requested symbol, fetching only cache misses.

    Never raises: a symbol that cannot be classified comes back with sector None, which the caller
    renders as "Other" rather than dropping. Being unclassified is a fact about our data, not a
    reason to make a stock disappear from the market map.
    """
    cache = _load()
    now = time.time()
    want = [s.upper() for s in symbols]
    missing = [s for s in want if not _fresh(cache.get(s) or {}, now)]

    if missing:
        try:
            crumb = await options._ensure_auth(client)
            sem = asyncio.Semaphore(_CONCURRENCY)

            async def one(sym: str):
                async with sem:
                    cache[sym] = await _fetch_one(client, sym, crumb)

            await asyncio.gather(*[one(s) for s in missing])
            _save()
            hit = sum(1 for s in missing if (cache.get(s) or {}).get("sector"))
            log.info("sectors: fetched %d, classified %d", len(missing), hit)
        except Exception:  # noqa: BLE001 — fall through to whatever is cached
            log.warning("sectors: batch lookup failed", exc_info=True)

    return {s: {"sector": (cache.get(s) or {}).get("sector"),
                "industry": (cache.get(s) or {}).get("industry")} for s in want}
