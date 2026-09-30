"""
ABOUT-1 — what a company is, for the detail screen's About tab.

A ticker like VICR says nothing to someone who has not met it. The app showed a price and a chart
and went straight to alerts and signal cards; nowhere did it say "Vicor makes power modules". This
module answers that from Yahoo's company profile (`assetProfile`, the same source the sector map
uses) plus the handful of key statistics a person weighs first, and, when asked, a plain-English
line written once per company by the analyst model.

Three kinds of data, three lifetimes:

* The profile facts (sector, industry, description, staff, headquarters) change approximately never;
* The statistics (size, growth, margin, price-per-profit, short interest, analyst target) move daily;
* The plain-English line is written from the description, so it is kept until the description
  changes, and is never regenerated per view.

Every figure Yahoo does not have is None, and the app draws "—" or leaves the tile out. A fund
without a company profile is still answered: its category and description stand in.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path

import httpx

from . import options

log = logging.getLogger("signals.profile")

_DATA_DIR = Path(os.environ.get("SIGNALS_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
_FACTS_FILE = _DATA_DIR / "profiles.json"
_PLAIN_FILE = _DATA_DIR / "profile_plain.json"

FACTS_TTL = 12 * 3600      # statistics move daily; half a day keeps them current enough
MISS_TTL = 30 * 60         # a failed lookup is retried soon, not stranded

_MODULES = "assetProfile,price,summaryDetail,defaultKeyStatistics,financialData,fundProfile"

_facts: dict[str, dict] | None = None
_plain: dict[str, dict] | None = None


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001 — missing or corrupt cache starts empty
        return {}


def _save(path: Path, data: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(path)
    except Exception:  # noqa: BLE001 — the cache is an optimisation, never a blocker
        log.warning("profile: cache write failed (%s)", path.name, exc_info=True)


def _raw(d: dict, key: str) -> float | None:
    v = d.get(key)
    if isinstance(v, dict):
        v = v.get("raw")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _pct(v: float | None) -> float | None:
    return round(v * 100.0, 2) if v is not None else None


_FOUNDED = re.compile(r"\b(?:incorporated|founded|established)\s+in\s+(1[6-9]\d\d|20\d\d)\b", re.I)


def parse(symbol: str, result: dict) -> dict:
    """Pure: one quoteSummary `result` object -> the profile the app reads."""
    ap = result.get("assetProfile") or {}
    pr = result.get("price") or {}
    sd = result.get("summaryDetail") or {}
    ks = result.get("defaultKeyStatistics") or {}
    fd = result.get("financialData") or {}
    fp = result.get("fundProfile") or {}
    summary = (ap.get("longBusinessSummary") or "").strip() or None
    m = _FOUNDED.search(summary or "")
    return {
        "symbol": symbol.upper(),
        "name": pr.get("longName") or pr.get("shortName") or symbol.upper(),
        "quote_type": pr.get("quoteType"),
        "exchange": pr.get("exchangeName"),
        "sector": (ap.get("sector") or "").strip() or None,
        "industry": (ap.get("industry") or "").strip() or None,
        # Funds have no sector; their category ("Large Blend", "Technology") is the equivalent.
        "category": (fp.get("categoryName") or "").strip() or None,
        "summary": summary,
        "employees": ap.get("fullTimeEmployees"),
        "city": ap.get("city"),
        "state": ap.get("state"),
        "country": ap.get("country"),
        "website": ap.get("website"),
        "founded": int(m.group(1)) if m else None,
        "market_cap": _raw(pr, "marketCap") or _raw(sd, "marketCap"),
        "revenue_growth_pct": _pct(_raw(fd, "revenueGrowth")),
        "profit_margin_pct": _pct(_raw(fd, "profitMargins")),
        "pe": _raw(sd, "trailingPE"),
        "forward_pe": _raw(sd, "forwardPE"),
        "short_pct_float": _pct(_raw(ks, "shortPercentOfFloat")),
        "target_mean": _raw(fd, "targetMeanPrice"),
        "n_analysts": int(_raw(fd, "numberOfAnalystOpinions") or 0) or None,
        "dividend_yield_pct": _pct(_raw(sd, "dividendYield")),
    }


async def _fetch(client: httpx.AsyncClient, symbol: str) -> dict | None:
    crumb = await options._ensure_auth(client)
    last: Exception | None = None
    for host in options._HOSTS:
        try:
            r = await client.get(f"https://{host}/v10/finance/quoteSummary/{symbol}",
                                 params={"modules": _MODULES, "crumb": crumb},
                                 headers=options._headers(), timeout=20)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            res = (r.json().get("quoteSummary") or {}).get("result") or []
            return parse(symbol, res[0]) if res else None
        except Exception as e:  # noqa: BLE001 — try the other host
            last = e
    raise RuntimeError(f"profile fetch failed: {last}")


async def facts(client: httpx.AsyncClient, symbol: str) -> dict | None:
    """The cached profile for `symbol`, refreshed when stale. None = Yahoo has no such symbol.
    Raises only when Yahoo could not be reached and nothing is cached."""
    global _facts
    if _facts is None:
        _facts = _load(_FACTS_FILE)
    sym = symbol.upper()
    row = _facts.get(sym)
    now = time.time()
    ttl = FACTS_TTL if row and row.get("profile") else MISS_TTL
    if row and now - float(row.get("ts") or 0) < ttl:
        return row.get("profile")
    try:
        prof = await _fetch(client, sym)
    except Exception:
        if row and row.get("profile"):
            return row["profile"]   # stale beats nothing; the response carries its as_of
        raise
    _facts[sym] = {"ts": now, "profile": prof}
    if prof:
        prof["as_of"] = now
    _save(_FACTS_FILE, _facts)
    return prof


def _desc_key(summary: str | None) -> str:
    return hashlib.sha1((summary or "").encode()).hexdigest()[:12]


def cached_plain(symbol: str, summary: str | None) -> dict | None:
    """The stored plain-English line, if it was written from THIS description."""
    global _plain
    if _plain is None:
        _plain = _load(_PLAIN_FILE)
    row = _plain.get(symbol.upper())
    if row and row.get("desc") == _desc_key(summary):
        return row
    return None


def store_plain(symbol: str, summary: str | None, what: str, customers: str) -> dict:
    global _plain
    if _plain is None:
        _plain = _load(_PLAIN_FILE)
    row = {"desc": _desc_key(summary), "what_it_does": what.strip(), "customers": customers.strip(),
           "ts": time.time()}
    _plain[symbol.upper()] = row
    _save(_PLAIN_FILE, _plain)
    return row
