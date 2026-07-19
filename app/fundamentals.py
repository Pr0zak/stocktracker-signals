"""
Quality tags (Finnhub basic-financials, /stock/metric) — durable / profitable / low-leverage
descriptors, surfaced as stance-NEUTRAL context (never a buy call). Free tier; no-op without a key
or for crypto; cached 12h per symbol.

Finnhub units, verified against the live free tier: ROE and margins are PERCENT (roeTTM = 8.37 means
8.37%), while totalDebt/totalEquity is a RATIO (~0.7, NOT a percent) — so low_debt = D/E < 0.5, not
< 50. Free-cash-flow is NOT in /stock/metric, so buffett_quality approximates the "positive FCF" leg
with a positive net margin (positive earnings); it's a descriptor, not a screen.
"""
from __future__ import annotations

import logging
import time

import httpx

from . import settings_store

log = logging.getLogger("uvicorn.error")

_BASE = "https://finnhub.io/api/v1"
_cache: dict[str, tuple[float, dict | None]] = {}
_fin_cache: dict[str, tuple[float, dict | None]] = {}
_TTL = 12 * 3600

# S&P 500 Dividend Aristocrats (25+ years of consecutive dividend increases). Static reference data;
# drifts at the annual January reconstitution — refresh then. (~65 members.)
DIVIDEND_ARISTOCRATS = frozenset({
    "MMM", "ABBV", "ABT", "AFL", "APD", "ALB", "AMCR", "ADM", "AOS", "ATO", "ADP", "BDX", "BF.B",
    "BEN", "CAH", "CAT", "CB", "CHRW", "CINF", "CTAS", "CLX", "KO", "CL", "ED", "DOV", "ECL", "EMR",
    "ESS", "EXPD", "XOM", "FAST", "FRT", "GD", "GPC", "HRL", "ITW", "IBM", "JNJ", "KVUE", "KMB",
    "LEG", "LIN", "LOW", "MKC", "MCD", "MDT", "NEE", "NDSN", "NUE", "O", "PNR", "PEP", "PPG", "PG",
    "ROP", "SPGI", "SHW", "SWK", "SYY", "TROW", "TGT", "GWW", "WMT", "WST",
})


async def fetch_quality(client: httpx.AsyncClient, symbol: str) -> dict | None:
    sym = symbol.upper()
    hit = _cache.get(sym)
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]

    aristocrat = sym in DIVIDEND_ARISTOCRATS
    key = settings_store.get().get("finnhub_api_key", "")
    if not key:
        # Aristocrat membership needs no feed — still worth returning so the chip works keyless.
        out = {"dividend_aristocrat": True} if aristocrat else None
        _cache[sym] = (time.time(), out)
        return out

    out: dict | None = None
    try:
        r = await client.get(
            f"{_BASE}/stock/metric",
            params={"symbol": sym, "metric": "all", "token": key}, timeout=15,
        )
        r.raise_for_status()
        m = (r.json() or {}).get("metric", {}) or {}

        def num(k: str) -> float | None:
            v = m.get(k)
            return float(v) if isinstance(v, (int, float)) else None

        roe = num("roeTTM")                                  # percent
        gross = num("grossMarginTTM")                        # percent
        net = num("netProfitMarginTTM")                      # percent
        de = num("totalDebt/totalEquityQuarterly") or num("totalDebt/totalEquityAnnual")  # RATIO

        high_roe = roe is not None and roe > 15
        low_debt = de is not None and de < 0.5
        wide_moat = bool(high_roe and gross is not None and gross > 40)
        buffett = bool(high_roe and low_debt and net is not None and net > 0)

        block = {
            "roe": roe, "gross_margin": gross, "net_margin": net, "debt_to_equity": de,
            "high_roe": high_roe, "low_debt": low_debt, "wide_moat": wide_moat,
            "buffett_quality": buffett, "dividend_aristocrat": aristocrat,
        }
        # Only return something if the feed gave at least one metric (or it's an aristocrat).
        if any(v is not None for v in (roe, gross, net, de)) or aristocrat:
            out = block
    except Exception:  # noqa: BLE001 — quality tags are enrichment, never a blocker
        out = {"dividend_aristocrat": True} if aristocrat else None
    _cache[sym] = (time.time(), out)
    return out


def compact(data: dict | None) -> dict | None:
    """Slim block for the analyst snapshot — drop null metrics, keep the flags."""
    if not data:
        return None
    keep = ("roe", "gross_margin", "net_margin", "debt_to_equity",
            "high_roe", "low_debt", "wide_moat", "buffett_quality", "dividend_aristocrat",
            "fcf_trend", "fcf_positive_years", "fcf_years", "shares_change_pct")
    slim = {k: data[k] for k in keep if data.get(k) is not None}
    return slim or None


def _pick(items: list, needle: str) -> float | None:
    """First numeric value whose XBRL concept ends with `needle` (Finnhub prefixes vary)."""
    for it in items or []:
        c = str(it.get("concept", ""))
        if c == needle or c.endswith(needle):
            v = it.get("value")
            if isinstance(v, (int, float)):
                return float(v)
    return None


async def fetch_financials(client: httpx.AsyncClient, symbol: str) -> dict | None:
    """FCF-trend (operating cash flow − capex) + share-count-trend from Finnhub's as-reported SEC
    financials (`/stock/financials-reported`, annual 10-Ks). Best-effort; None without a key, for
    crypto, or when the statements don't line up. Cached 12h. (MB-13 / MB-14.)"""
    sym = symbol.upper()
    hit = _fin_cache.get(sym)
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]
    key = settings_store.get().get("finnhub_api_key", "")
    out: dict | None = None
    if key:
        try:
            r = await client.get(
                f"{_BASE}/stock/financials-reported",
                params={"symbol": sym, "freq": "annual", "token": key}, timeout=20,
            )
            reports = (r.json() or {}).get("data", []) or []
            if reports:  # one-shot diagnostic: exact OCF/capex/share concepts + values
                cf0 = (reports[0].get("report") or {}).get("cf") or []
                ic0 = (reports[0].get("report") or {}).get("ic") or []
                rel = [(str(x.get("concept")), x.get("value")) for x in cf0
                       if any(k in str(x.get("concept")) for k in
                              ("OperatingActivities", "PropertyPlantAndEquipment", "CapitalExpend", "PurchaseOfProperty"))]
                sh = [(str(x.get("concept")), x.get("value")) for x in ic0
                      if "SharesOutstanding" in str(x.get("concept"))]
                log.info("financials %s http=%s reports=%s year0=%s cf_rel=%s sh=%s",
                         sym, r.status_code, len(reports), reports[0].get("year"), rel[:6], sh[:4])
            r.raise_for_status()
            by_year: dict[int, tuple[float | None, float | None, float | None]] = {}
            for rep in reports:
                if rep.get("form") not in ("10-K", "10-K/A"):
                    continue
                yr = rep.get("year")
                rpt = rep.get("report") or {}
                cf, ic = rpt.get("cf") or [], rpt.get("ic") or []
                ocf = _pick(cf, "NetCashProvidedByUsedInOperatingActivities")
                capex = _pick(cf, "PaymentsToAcquirePropertyPlantAndEquipment")
                sh = (_pick(ic, "WeightedAverageNumberOfDilutedSharesOutstanding")
                      or _pick(ic, "WeightedAverageNumberOfSharesOutstandingBasic"))
                if yr is not None:
                    by_year[yr] = (ocf, capex, sh)
            years = sorted(by_year)
            fcf = [(y, by_year[y][0] - by_year[y][1]) for y in years
                   if by_year[y][0] is not None and by_year[y][1] is not None][-5:]
            shares = [(y, by_year[y][2]) for y in years if by_year[y][2]][-5:]
            block: dict = {}
            if len(fcf) >= 2:
                latest = fcf[-1][1]
                earlier = [v for _, v in fcf[:-1]]
                mean_earlier = sum(earlier) / len(earlier)
                block["fcf_latest"] = round(latest)
                block["fcf_trend"] = (
                    "rising" if latest > mean_earlier * 1.05
                    else "falling" if latest < mean_earlier * 0.95 else "flat"
                )
                block["fcf_positive_years"] = sum(1 for _, v in fcf if v > 0)
                block["fcf_years"] = len(fcf)
            if len(shares) >= 2 and shares[0][1] > 0:
                block["shares_change_pct"] = round((shares[-1][1] / shares[0][1] - 1) * 100, 1)
                block["shares_years"] = len(shares)
            out = block or None
        except Exception:  # noqa: BLE001 — fundamentals are enrichment, never a blocker
            out = None
    _fin_cache[sym] = (time.time(), out)
    return out
