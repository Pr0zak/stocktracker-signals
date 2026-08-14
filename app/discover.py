"""
Market-wide candidate discovery for /recommendations scope="market".

Pulls live candidates from Yahoo's predefined screeners, round-robins across them for diversity,
filters obvious junk (sub-$5 price, sub-$2B equities, non-EQUITY/ETF quote types), dedupes against
whatever the caller already has, and caps the pool. Falls back to a small curated universe of
mega-caps + core ETFs when the screener API is unreachable, so market mode always works.

Two screen sets: the default four are momentum/value angles ("what moved today"), while
[WIDE_SCREENS] adds two contrarian angles and all eleven Morningstar sector screens ("what exists").
The sandbox needs the second because it is building a diversified book against a standing plan, not
answering a question about today.
"""
from __future__ import annotations

import asyncio
from itertools import zip_longest

import httpx

_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
_SCREENS = ("most_actives", "day_gainers", "growth_technology_stocks", "undervalued_large_caps")

# Contrarian angles. The four above are three parts momentum to one part value, so a pool drawn from
# them alone can only ever propose things that have already moved.
_ANGLE_SCREENS = ("day_losers", "undervalued_growth_stocks")

# Morningstar's per-sector screens: the sector leaders, eleven sectors, all verified live 2026-08-14.
#
# These exist because the momentum screens have a structural blind spot — they return what is moving,
# which on any given day is a handful of sectors. Measured that day, a 45-name momentum-only pool
# contained no MSFT, COST or JNJ while the standing strategy targeted all three at 8/7/6%, so 21% of
# the plan was unfillable for want of a candidate. Sector screens fix that at the source rather than
# by hand-listing companies: which COMPANIES to consider stays the model's decision, and this only
# guarantees every sector gets a hearing.
_SECTOR_SCREENS = (
    "ms_basic_materials", "ms_communication_services", "ms_consumer_cyclical",
    "ms_consumer_defensive", "ms_energy", "ms_financial_services", "ms_healthcare",
    "ms_industrials", "ms_real_estate", "ms_technology", "ms_utilities",
)

# The wide set, for callers that want breadth rather than today's movers.
#
# Two working screens are deliberately left OUT. `most_shorted_stocks` returns sub-$1 microcaps
# (NUWE, KUST, VIVK) that the junk filters would drop anyway, and `portfolio_anchors` returns MUTUAL
# FUNDS (VFIAX, FXAIX) — share classes with minimums and no intraday fill, which nothing downstream
# can price or trade. `aggressive_small_caps` and `small_cap_gainers` also work but skew to the
# speculative end (SPCE, PSQH, BW); they are a defensible future addition, not a silent one.
WIDE_SCREENS = _SCREENS + _ANGLE_SCREENS + _SECTOR_SCREENS

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


async def discover(client: httpx.AsyncClient, exclude: set[str], cap: int = 15,
                   screens: tuple[str, ...] = _SCREENS, allow_etf: bool = True,
                   min_market_cap: float = 2_000_000_000.0) -> list[str]:
    """Candidate symbols beyond the caller's own list, most interesting first. Never raises.

    `screens` defaults to the four momentum/value angles. Pass [WIDE_SCREENS] for a pool that also
    covers every sector — the sandbox does, because a momentum-only pool cannot fill a diversified
    plan. It stays a parameter rather than a new default so /recommendations, which answers "where
    should this cash go TODAY", keeps the movers-first pool it was tuned for.
    """
    try:
        # Fetch at least as many per screen as the caller wants in total. The junk filters below
        # (quote type, sub-$5, sub-$2B) and the cross-screen dedupe both cut into the raw lists, so a
        # fixed 15 per screen silently capped the result far under a larger `cap`.
        per_screen = min(100, max(15, cap // max(1, len(screens)) + 5))
        quote_lists = await asyncio.gather(*[_screen(client, s, per_screen) for s in screens])
    except Exception:  # noqa: BLE001
        quote_lists = []

    seen: set[str] = set()
    out: list[str] = []
    # Round-robin across the screens so no single angle dominates the capped pool. With the wide
    # set this is what guarantees sector breadth: the cap lands evenly, not on whichever screen sorted
    # first.
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
            # Applied here because `quoteType` is only available inside this loop -- the function
            # returns bare symbols, so a caller cannot filter on something it never sees.
            if not allow_etf and qt == "ETF":
                continue
            price = _raw(q.get("regularMarketPrice"))
            mktcap = _raw(q.get("marketCap"))
            if price and price < 5:  # skip penny-ish names
                continue
            # Skip anything below the caller's size floor. `mktcap and` keeps a name whose cap the
            # screener did not report: absent is not the same as small, and a missing field is a fact
            # about Yahoo's response rather than about the company. That does mean a raised floor
            # leaks the occasional unreported name — the alternative silently drops real companies on
            # a data gap, which is the worse trade for a universe this is meant to widen.
            if qt == "EQUITY" and mktcap and mktcap < min_market_cap:
                continue
            seen.add(sym)
            out.append(sym)
            if len(out) >= cap:
                return out
    # Fall back ONLY when the screeners returned nothing at all. The trigger used to be "fewer than
    # five names survived", which conflates two unrelated situations: the API being unreachable, and
    # the caller's own filters being strict. Raise min_market_cap high enough and the second one
    # fires — silently replacing a deliberately narrow universe with a hardcoded list that the floor
    # was never applied to. A configuration doing exactly what it was told is not a failure, and the
    # user gets no signal that their setting stopped being honoured.
    if not any(quote_lists):
        out = [s for s in FALLBACK if s not in exclude]
    return out[:cap]
