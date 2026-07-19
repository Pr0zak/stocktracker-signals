"""
SEC XBRL company facts — free, keyless. Two durability signals the Finnhub /stock/metric feed can't
give us:

  • Free-cash-flow trend (MB-13): operating cash flow − capex, per fiscal year.
  • Share-count trend (MB-14): shares outstanding over time — falling = buybacks, rising = dilution.

Computed from annual (10-K / FY) us-gaap + dei concepts. Best-effort: returns None (or omits a leg)
when a company's tags don't line up. Only hit on-demand (the detail screen's /quality), so SEC's
rate limit is a non-issue. SEC requires an identifying User-Agent on every request.
"""
from __future__ import annotations

import time

import httpx

_UA = {"User-Agent": "stocktracker-signals/1.0 (+https://github.com/Pr0zak/stocktracker)"}
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_CONCEPT = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/{tax}/{tag}.json"

_OCF_TAGS = (
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
)
_CAPEX_TAGS = (
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
)
_SHARE_TAGS = (  # (taxonomy, tag)
    ("dei", "EntityCommonStockSharesOutstanding"),
    ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
)

_cik_map: dict[str, int] | None = None
_cik_ts = 0.0
_cache: dict[str, tuple[float, dict | None]] = {}
_TTL = 24 * 3600


async def _ciks(client: httpx.AsyncClient) -> dict[str, int]:
    """Ticker→CIK map from SEC's directory (cached a week; ~1 MB, fetched at most once)."""
    global _cik_map, _cik_ts
    if _cik_map is not None and time.time() - _cik_ts < 7 * 24 * 3600:
        return _cik_map
    r = await client.get(_TICKERS_URL, headers=_UA, timeout=30)
    r.raise_for_status()
    _cik_map = {row["ticker"].upper(): int(row["cik_str"]) for row in (r.json() or {}).values()}
    _cik_ts = time.time()
    return _cik_map


async def _annual(client: httpx.AsyncClient, cik: int, tag: str, tax: str = "us-gaap") -> list[tuple[str, float]]:
    """[(fiscal-year-end, value)] for annual (FY / 10-K) filings — one per fiscal year, keeping the
    latest-filed value, oldest→newest. Empty on any miss (404, no annual rows, network error)."""
    try:
        r = await client.get(_CONCEPT.format(cik=cik, tax=tax, tag=tag), headers=_UA, timeout=20)
        if r.status_code != 200:
            return []
        units = (r.json() or {}).get("units", {})
        rows = units.get("USD") or units.get("shares") or (next(iter(units.values()), []) if units else [])
        by_fy: dict[int, tuple[str, float, str]] = {}  # fy -> (end, val, filed)
        for it in rows:
            if it.get("form") not in ("10-K", "10-K/A") or it.get("fp") != "FY":
                continue
            fy, val, end, filed = it.get("fy"), it.get("val"), it.get("end", ""), it.get("filed", "")
            if fy is None or val is None:
                continue
            prev = by_fy.get(fy)
            if prev is None or filed > prev[2]:
                by_fy[fy] = (end, float(val), filed)
        return [(v[0], v[1]) for _, v in sorted(by_fy.items())]
    except Exception:  # noqa: BLE001
        return []


async def _first_nonempty(client, cik, tags, tax="us-gaap") -> dict[str, float]:
    for tag in tags:
        rows = await _annual(client, cik, tag, tax=tax)
        if rows:
            return dict(rows)  # keyed by fiscal-year-end
    return {}


async def fetch_fundamentals(client: httpx.AsyncClient, symbol: str) -> dict | None:
    """FCF-trend + share-count-trend block for a US filer, or None. Cached 24h."""
    sym = symbol.upper()
    hit = _cache.get(sym)
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]

    out: dict = {}
    try:
        cik = (await _ciks(client)).get(sym)
        if cik is not None:
            # --- Free cash flow: OCF − capex, aligned by fiscal-year-end ---
            ocf = await _first_nonempty(client, cik, _OCF_TAGS)
            capex = await _first_nonempty(client, cik, _CAPEX_TAGS)
            fcf = [(end, ocf[end] - capex[end]) for end in sorted(ocf) if end in capex][-5:]
            if len(fcf) >= 2:
                latest = fcf[-1][1]
                earlier = [v for _, v in fcf[:-1]]
                mean_earlier = sum(earlier) / len(earlier)
                trend = (
                    "rising" if latest > mean_earlier * 1.05
                    else "falling" if latest < mean_earlier * 0.95
                    else "flat"
                )
                out["fcf_latest"] = round(latest)
                out["fcf_trend"] = trend
                out["fcf_positive_years"] = sum(1 for _, v in fcf if v > 0)
                out["fcf_years"] = len(fcf)

            # --- Share count: dilution (+) vs buybacks (−) over the window ---
            shares: list[tuple[str, float]] = []
            for tax, tag in _SHARE_TAGS:
                shares = await _annual(client, cik, tag, tax=tax)
                if len(shares) >= 2:
                    break
            shares = shares[-5:]
            if len(shares) >= 2 and shares[0][1] > 0:
                out["shares_change_pct"] = round((shares[-1][1] / shares[0][1] - 1) * 100, 1)
                out["shares_years"] = len(shares)
    except Exception:  # noqa: BLE001 — fundamentals are enrichment, never a blocker
        out = {}
    result = out or None
    _cache[sym] = (time.time(), result)
    return result
