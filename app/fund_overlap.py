"""What a fund holds, how much funds overlap, and how they have done (FUND-1..6).

The Funds screen answers the questions a fund buyer actually has once fees are on the table (see
fund_cost.py): what does this fund cover, do two of my funds hold the same thing, and how have they
done against each other.

OVERLAP IS LED BY PRICES, NOT BY SECTORS. On 2026-09-26 VOO and VXUS came out 72% alike by sector
weight while sharing none of their top holdings — one is American companies, the other everyone
else's, and a sector table cannot see a border. Their returns correlated at 0.73. SPY and VOO, by
contrast, correlated at 1.000. So the verdict on a pair ("same fund", "move together") comes from
two years of returns, which works for anything with a price, bitcoin funds included; sectors and
top holdings are shown as what each fund covers, and as supporting detail.

Yahoo lists only each fund's TEN largest holdings. Shared top holdings are therefore a floor, never
the whole overlap: VOO holds all 500 S&P companies, but only its ten biggest can be matched. The
payload says "top ten" everywhere it uses them.

Returns are measured the way fund_peers.py measures look-alikes: weekly, or four-weekly when a
mutual fund is involved, because a fund priced once a day at a fair-valued NAV carries timing noise
that washes out over a month. Total returns are dividends-in (Yahoo's adjusted closes).
"""
from __future__ import annotations

import asyncio
import bisect
import datetime as dt
import logging
import math
import time
from typing import Awaitable, Callable

import httpx

from . import fund_cost, market_now, options
from .market import Series, fetch_series
from .redact import redact

log = logging.getLogger("signals.fund_overlap")

PROFILE_TTL_SECONDS = 24 * 3600      # top holdings move monthly at most; sectors slower still
SERIES_TTL_SECONDS = 6 * 3600
FAILED_TTL_SECONDS = 10 * 60         # a failed read is retried soon, not stuck for a day
MAX_SYMBOLS = 80                     # what one overlap request may name (stocks are sorted out)
MAX_FUNDS = 40                       # funds actually measured: pairs grow with the square (780 at 40)
MAX_PERFORMANCE_SYMBOLS = 12
_CONCURRENCY = 6

# Funds that correlate this closely rise and fall as one: counted as ONE bet. Complete linkage, so a
# fund joins a set only when it clears the bar against every member — SPMO (0.87 with VOO, 0.92 with
# QQQM on 2026-09-26) stays its own bet instead of being chained in through QQQM.
SAME_BET_CORR = 0.90

SECTOR_LABELS = {
    "technology": "Tech",
    "financial_services": "Finance",
    "healthcare": "Health care",
    "consumer_cyclical": "Shopping & leisure",
    "consumer_defensive": "Everyday goods",
    "communication_services": "Communication",
    "industrials": "Industrials",
    "energy": "Energy",
    "basic_materials": "Materials",
    "utilities": "Utilities",
    "realestate": "Real estate",
}

# Share classes of one company, so "shares its top holdings" counts Alphabet once, not twice.
_SAME_COMPANY = {"GOOG": "GOOGL", "BRK-A": "BRK-B", "BRK.A": "BRK-B", "BRK.B": "BRK-B"}

REGION_LABELS = {
    "us": "US stocks",
    "international": "Stocks outside the US",
    "world": "Stocks worldwide",
    "bonds": "Bonds",
    "mixed": "Stocks and bonds",
    "commodities": "Gold & commodities",
    "crypto": "Crypto",
    "leveraged": "Leveraged or inverse",
}

_profiles: dict[str, tuple[float, dict | None]] = {}
_series: dict[str, tuple[float, Series | None]] = {}

ProfileFetch = Callable[[httpx.AsyncClient, str], Awaitable[dict | None]]
HistoryFetch = Callable[[httpx.AsyncClient, str], Awaitable[Series]]


# --- what a fund covers -----------------------------------------------------------------------

def region(category: str | None, stock: float | None, bond: float | None) -> str | None:
    """A plain region code from Yahoo's (Morningstar) category and the stock/bond split."""
    c = (category or "").lower()
    if "digital" in c:
        return "crypto"
    if "trading" in c:                   # "Trading--Leveraged Equity", "Trading--Inverse ..."
        return "leveraged"
    if "commodit" in c or "precious" in c:
        return "commodities"
    if "allocation" in c:
        return "mixed"
    if any(k in c for k in ("bond", "muni", "treasury", "government", "corporate", "inflation",
                            "ultrashort", "securitized", "bank loan", "credit", "money market")):
        return "bonds"
    if "world" in c or "global" in c:
        return "world"
    if any(k in c for k in ("foreign", "emerging", "europe", "japan", "china", "india", "pacific",
                            "latin", "asia", "international")):
        return "international"
    if stock is not None and stock >= 0.5:
        return "us"
    if bond is not None and bond >= 0.5:
        return "bonds"
    return None


def parse_profile(result: dict) -> dict:
    """The fields the app shows, from one quoteSummary result (topHoldings + fundProfile)."""
    th = result.get("topHoldings") or {}
    fp = result.get("fundProfile") or {}

    def raw(d, k):
        v = (d.get(k) or {}).get("raw") if isinstance(d.get(k), dict) else None
        return float(v) if isinstance(v, (int, float)) else None

    weights: dict[str, float] = {}
    for entry in th.get("sectorWeightings") or []:
        for k, v in entry.items():
            w = v.get("raw") if isinstance(v, dict) else None
            if isinstance(w, (int, float)) and w > 0:
                weights[k] = weights.get(k, 0.0) + float(w)
    total = sum(weights.values())
    sectors = [
        {"key": k, "label": SECTOR_LABELS.get(k, k.replace("_", " ").title()), "pct": round(w / total * 100, 1)}
        for k, w in sorted(weights.items(), key=lambda kv: -kv[1])
    ] if total > 0 else []

    merged: dict[str, dict] = {}
    for h in th.get("holdings") or []:
        sym = (h.get("symbol") or "").upper()
        w = (h.get("holdingPercent") or {}).get("raw")
        if not sym or not isinstance(w, (int, float)):
            continue
        key = _SAME_COMPANY.get(sym, sym)
        row = merged.setdefault(key, {"symbol": key, "name": h.get("holdingName"), "pct": 0.0})
        row["pct"] += float(w) * 100
    top = sorted(merged.values(), key=lambda r: -r["pct"])
    for r in top:
        r["pct"] = round(r["pct"], 2)

    stock, bond = raw(th, "stockPosition"), raw(th, "bondPosition")
    return {
        "category": fp.get("categoryName"),
        "family": fp.get("family"),
        "stock_pct": round(stock * 100, 1) if stock is not None else None,
        "bond_pct": round(bond * 100, 1) if bond is not None else None,
        "region": region(fp.get("categoryName"), stock, bond),
        "sectors": sectors,
        "top_holdings": top,
        "top10_pct": round(sum(r["pct"] for r in top), 1) if top else None,
    }


async def _yahoo_profile(client: httpx.AsyncClient, symbol: str) -> dict | None:
    crumb = await options._ensure_auth(client)
    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
    params = {"modules": "topHoldings,fundProfile", "crumb": crumb}
    r = await client.get(url, params=params, headers=options._headers(), timeout=20)
    if r.status_code == 401:
        params["crumb"] = await options._ensure_auth(client, force=True, stale=crumb)
        r = await client.get(url, params=params, headers=options._headers(), timeout=20)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    res = ((r.json().get("quoteSummary") or {}).get("result") or [None])[0]
    # Yahoo drops a module from the result when that module errors. A missing topHoldings is a
    # failed read, not a fund that holds nothing — parsed, it would cache "no sectors, no holdings"
    # for a day and report VOO and SPY as sharing none of their top holdings.
    if not res or "topHoldings" not in res:
        return None
    return parse_profile(res)


def _positive(s: Series) -> Series:
    """The series without bars at or below zero. Yahoo occasionally prints a 0.0 close for a thin
    fund or a mutual fund's NAV; kept, it divides by zero in every return and drawdown."""
    keep = [i for i, c in enumerate(s.closes) if c is not None and c > 0]
    if len(keep) == len(s.closes):
        return s
    closes = [s.closes[i] for i in keep]
    return Series(symbol=s.symbol, closes=closes, opens=closes, volumes=[None] * len(keep),
                  dates=[s.dates[i] for i in keep], fifty_two_high=s.fifty_two_high,
                  fifty_two_low=s.fifty_two_low, currency=s.currency, source=s.source)


async def _yahoo_history(client: httpx.AsyncClient, symbol: str) -> Series:
    return await fetch_series(client, symbol, rng="5y", fallback=False)


async def _load_history(fetch_history: "HistoryFetch", client: httpx.AsyncClient, symbol: str) -> Series:
    return _positive(await fetch_history(client, symbol))


async def _cached(cache: dict, key: str, ttl: float, now: float, load: Callable[[], Awaitable]):
    hit = cache.get(key)
    if hit is not None and now - hit[0] < (ttl if hit[1] is not None else FAILED_TTL_SECONDS):
        return hit[1]
    try:
        val = await load()
    except Exception as e:  # noqa: BLE001 — unknown is reported as unknown by the caller
        log.warning("fund_overlap: %s failed: %s", key, redact(e))
        val = None
    cache[key] = (now, val)
    return val


# --- measuring pairs --------------------------------------------------------------------------

def _dates(s: Series) -> list[dt.date]:
    return [dt.date(int(d[:4]), int(d[4:6]), int(d[6:8])) for d in s.dates]


def correlation(a: Series, b: Series, *, step: int, years: float = 2.0) -> tuple[float | None, int]:
    """Correlation of `step`-bar returns over the last [years] both have prices for, and how many
    returns it rests on. None below a minimum sample: half a year of weekly returns, or a year of
    four-weekly ones."""
    pa = dict(zip(a.dates, a.closes))
    pb = dict(zip(b.dates, b.closes))
    common = sorted(set(pa) & set(pb))
    if not common:
        return None, 0
    last = dt.date(int(common[-1][:4]), int(common[-1][4:6]), int(common[-1][6:8]))
    cutoff = (last - dt.timedelta(days=round(365.25 * years))).strftime("%Y%m%d")
    days = [d for d in common if d >= cutoff][::-1][::step][::-1]   # anchored on the latest close
    ra = [pa[days[i]] / pa[days[i - 1]] - 1 for i in range(1, len(days))]
    rb = [pb[days[i]] / pb[days[i - 1]] - 1 for i in range(1, len(days))]
    n = len(ra)
    if n < (26 if step <= 5 else 12):
        return None, n
    ma, mb = sum(ra) / n, sum(rb) / n
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((y - mb) ** 2 for y in rb)
    if va <= 0 or vb <= 0:
        return None, n
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / math.sqrt(va * vb), n


def shared_top(a: dict | None, b: dict | None) -> dict:
    """Top-ten holdings two funds have in common. A FLOOR on their overlap (see the docstring)."""
    ta = {h["symbol"]: h for h in (a or {}).get("top_holdings") or []}
    tb = {h["symbol"]: h for h in (b or {}).get("top_holdings") or []}
    names = sorted(set(ta) & set(tb), key=lambda s: -min(ta[s]["pct"], tb[s]["pct"]))
    return {
        "shared_top": names,
        "shared_top_count": len(names),
        # The weight the pair holds in common among those names: the smaller of the two each time.
        "shared_top_min_pct": round(sum(min(ta[s]["pct"], tb[s]["pct"]) for s in names), 1) if names else 0.0,
    }


def sector_alike(a: dict | None, b: dict | None) -> float | None:
    """How alike two funds' sector mixes are, 0-100. Shown as detail only: see the docstring."""
    sa = {s["key"]: s["pct"] for s in (a or {}).get("sectors") or []}
    sb = {s["key"]: s["pct"] for s in (b or {}).get("sectors") or []}
    if not sa or not sb:
        return None
    return round(sum(min(sa.get(k, 0.0), sb.get(k, 0.0)) for k in set(sa) | set(sb)), 1)


def same_bets(symbols: list[str], corr: dict[tuple[str, str], float | None]) -> list[list[str]]:
    """Group funds that rise and fall together (complete linkage at SAME_BET_CORR).

    An unknown correlation never links two funds: "we could not measure it" is not "they move
    together", and merging on it would claim an overlap nobody observed.
    """
    groups = [[s] for s in symbols]

    def link(g1: list[str], g2: list[str]) -> float | None:
        vals = [corr.get((x, y)) if (x, y) in corr else corr.get((y, x)) for x in g1 for y in g2]
        return None if any(v is None for v in vals) else min(vals)

    while True:
        best: tuple[float, int, int] | None = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                v = link(groups[i], groups[j])
                if v is not None and v >= SAME_BET_CORR and (best is None or v > best[0]):
                    best = (v, i, j)
        if best is None:
            break
        _, i, j = best
        groups[i] = groups[i] + groups[j]
        del groups[j]
    return sorted(groups, key=lambda g: (-len(g), g[0]))


# --- how a fund has done ----------------------------------------------------------------------

def returns(s: Series, *, last: dt.date | None = None) -> dict:
    """Dividends-in total return over 1, 2, 3 and 5 years, in percent. None when the fund's history
    does not reach that far back (a week of slack for where Yahoo's 5-year range happens to start).

    [last] pins where every window ends: the fund's latest bar on or before that day, with the start
    targets counted back from it. performance() passes the latest day EVERY requested fund has a
    price for, because left to their own last bars funds shown side by side are not measured over
    the same days: an ETF's Yahoo chart carries today's live bar from the opening bell, while a
    mutual fund's NAV for today is not there until the evening. On a 1.5% day that put the ETF's
    "2 years" a whole session ahead of the fund's, and the app printed the day's move as 1.5 points
    of "hidden cost" against the cheapest look-alike."""
    ds = _dates(s)
    out: dict[str, float | None] = {}
    none = {k: None for k in ("1y", "2y", "3y", "5y")}
    if len(ds) < 2:
        return none
    end = len(ds) - 1 if last is None else bisect.bisect_right(ds, last) - 1
    if end < 1:
        return none
    last, last_c = ds[end], s.closes[end]
    for label, years in (("1y", 1), ("2y", 2), ("3y", 3), ("5y", 5)):
        target = last - dt.timedelta(days=round(365.25 * years))
        if ds[0] <= target:
            i = bisect.bisect_right(ds, target) - 1
        elif (ds[0] - target).days <= 7:
            i = 0
        else:
            out[label] = None
            continue
        out[label] = round((last_c / s.closes[i] - 1) * 100, 2)
    return out


def worst_drop(s: Series) -> dict:
    """The deepest fall from a high over the fetched history (up to five years), with its dates."""
    ds = _dates(s)
    if len(ds) < 2:
        return {"worst_drop_pct": None, "worst_drop_from": None, "worst_drop_to": None}
    peak, peak_i, worst, w_from, w_to = s.closes[0], 0, 0.0, 0, 0
    for i, c in enumerate(s.closes):
        if c > peak:
            peak, peak_i = c, i
        d = c / peak - 1
        if d < worst:
            worst, w_from, w_to = d, peak_i, i
    return {
        "worst_drop_pct": round(worst * 100, 1),
        "worst_drop_from": ds[w_from].isoformat() if worst < 0 else None,
        "worst_drop_to": ds[w_to].isoformat() if worst < 0 else None,
    }


def aligned_chart(hs: dict[str, Series]) -> dict | None:
    """The comparison chart: every fund's percent change since the first day ALL of them have a
    price, sampled weekly on days they all traded, anchored on the latest such day.

    Aligned here rather than on the phone because each fund's own weekly sample drifts: a mutual
    fund's NAV history can miss a day an ETF traded, and after that every fifth bar lands on a
    different date, so two lines that should share an axis would share almost no points."""
    if not hs:
        return None
    common = set.intersection(*(set(h.dates) for h in hs.values()))
    days = sorted(common)[::-1][::5][::-1]
    if len(days) < 2:
        return None
    lines: dict[str, list[float]] = {}
    for sym, h in hs.items():
        by_day = dict(zip(h.dates, h.closes))
        base = by_day[days[0]]
        lines[sym] = [round((by_day[d] / base - 1) * 100, 2) for d in days]
    return {"dates": [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in days], "lines": lines}


# --- the endpoints' work ----------------------------------------------------------------------

# How old an in-session spread reading may get before a look during the session re-measures it.
SPREAD_REFRESH_SECONDS = 3600


async def _refresh_spreads(
    client: httpx.AsyncClient, funds: list[str], *, now: float,
    quotes: fund_cost.Fetch | None, phase: str | None,
) -> None:
    """Re-read bid/ask for ETFs whose spread is missing or old — but only while the market is open.

    The fee cache holds a Yahoo read for twelve hours, so a fund first read before the open would
    otherwise never be measured in-session that day. Outside the session a bid/ask is stale or zero
    and is not worth reading, so the last in-session figure stands, with its own time.
    """
    if (phase or market_now.session_phase()) != "REGULAR":
        return
    stale = [s for s in funds
             if s not in fund_cost.MUTUAL_FUNDS
             and (s not in fund_cost._spreads or now - fund_cost._spreads[s][0] > SPREAD_REFRESH_SECONDS)]
    if not stale:
        return
    try:
        got = await (quotes or market_now.fetch_quotes)(client, stale)
    except Exception as e:  # noqa: BLE001 — the spread just stays unknown or old
        log.warning("fund_overlap: spread read failed: %s", redact(e))
        return
    for s, q in got.items():
        fund_cost._note_spread(s, q, now)

async def overlap(
    client: httpx.AsyncClient,
    symbols: list[str],
    *,
    saved: dict[str, float],
    saved_as_of: str,
    fetch_profile: ProfileFetch = _yahoo_profile,
    fetch_history: HistoryFetch = _yahoo_history,
    quotes: fund_cost.Fetch | None = None,
    now: float | None = None,
    phase: str | None = None,
) -> dict:
    """Profiles for the funds among [symbols], every pair's overlap, and the "same bet" sets.

    Single stocks are sorted out first (by the same classification the fee card uses) and named in
    `not_funds`, so a caller can pass a whole watchlist.
    """
    now = time.time() if now is None else now
    asked = list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
    want, over = asked[:MAX_SYMBOLS], asked[MAX_SYMBOLS:]
    live = await fund_cost.refresh(client, want, now=now,
                                   **({"fetch": quotes} if quotes is not None else {}))
    kinds = {s: fund_cost.row(s, saved=saved, saved_as_of=saved_as_of)["kind"] for s in want}
    all_funds = [s for s in want if kinds[s] in ("etf", "mutual_fund")]
    funds = all_funds[:MAX_FUNDS]       # request order: callers put the symbol they ask about first
    # Only what Yahoo positively called something else is "not a fund". A symbol it did not answer
    # for is unknown, and a fund past the cap was never measured — neither is a single stock, and
    # the app used to treat both as one.
    not_funds = [s for s in want if kinds[s] == "other"]
    unknown = [s for s in want if kinds[s] == "unknown"]
    unmeasured = all_funds[MAX_FUNDS:] + over
    await _refresh_spreads(client, funds, now=now, quotes=quotes, phase=phase)
    rows = {s: fund_cost.row(s, saved=saved, saved_as_of=saved_as_of) for s in want}

    sem = asyncio.Semaphore(_CONCURRENCY)

    async def prof(s: str):
        async with sem:
            return await _cached(_profiles, s, PROFILE_TTL_SECONDS, now, lambda: fetch_profile(client, s))

    async def hist(s: str):
        async with sem:
            return await _cached(_series, s, SERIES_TTL_SECONDS, now, lambda: _load_history(fetch_history, client, s))

    profs = dict(zip(funds, await asyncio.gather(*[prof(s) for s in funds])))
    hists = dict(zip(funds, await asyncio.gather(*[hist(s) for s in funds])))

    out_funds: dict[str, dict] = {}
    for s in funds:
        p = profs.get(s)
        g = fund_cost.group_of(s)
        reg = (p or {}).get("region")
        label = REGION_LABELS.get(reg) if reg else None
        if g is not None and g.id in ("bitcoin", "ether"):
            reg, label = "crypto", g.label.capitalize()
        h = hists.get(s)
        out_funds[s] = {
            **rows[s],
            "group_id": g.id if g else None,
            # False = Yahoo's holdings lookup failed: unknown, which is not the same as "holds nothing".
            "profile_ok": p is not None,
            "category": (p or {}).get("category"),
            "region": reg,
            "region_label": label,
            "stock_pct": (p or {}).get("stock_pct"),
            "bond_pct": (p or {}).get("bond_pct"),
            "sectors": (p or {}).get("sectors") if p is not None else None,
            "top_holdings": (p or {}).get("top_holdings") if p is not None else None,
            "top10_pct": (p or {}).get("top10_pct"),
            # First date of the price history read, which reaches back at most five years: a later
            # date than that is the fund's own start (FBTC: 2024-01-11).
            "history_start": _dates(h)[0].isoformat() if h is not None and h.dates else None,
        }

    corr: dict[tuple[str, str], float | None] = {}
    pairs: list[dict] = []
    for i, a in enumerate(funds):
        for b in funds[i + 1:]:
            mf = rows[a]["kind"] == "mutual_fund" or rows[b]["kind"] == "mutual_fund"
            step = 20 if mf else 5
            c, n = (None, 0)
            if hists.get(a) is not None and hists.get(b) is not None:
                c, n = correlation(hists[a], hists[b], step=step)
            corr[(a, b)] = c
            pairs.append({
                "a": a, "b": b,
                "corr": round(c, 3) if c is not None else None,
                "corr_basis": "4-weekly" if mf else "weekly",
                "corr_points": n,
                **shared_top(profs.get(a), profs.get(b)),
                "sector_alike_pct": sector_alike(profs.get(a), profs.get(b)),
            })
    return {
        "funds": out_funds,
        "not_funds": not_funds,
        "unknown": unknown,
        "unmeasured": unmeasured,
        "pairs": pairs,
        "same_bets": same_bets(funds, corr),
        "same_bet_corr": SAME_BET_CORR,
        "live": live,
        "as_of": now,
    }


async def performance(
    client: httpx.AsyncClient,
    symbols: list[str],
    *,
    include_series: bool = False,
    fetch_history: HistoryFetch = _yahoo_history,
    now: float | None = None,
) -> dict:
    """Total returns (dividends in), the worst drop, and optionally a weekly price line per symbol.
    A symbol whose history could not be read comes back with `available: False`, never zeros."""
    now = time.time() if now is None else now
    want = list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))[:MAX_PERFORMANCE_SYMBOLS]
    sem = asyncio.Semaphore(_CONCURRENCY)

    async def hist(s: str):
        async with sem:
            return await _cached(_series, s, SERIES_TTL_SECONDS, now, lambda: _load_history(fetch_history, client, s))

    got = dict(zip(want, await asyncio.gather(*[hist(s) for s in want])))
    usable = {s: h for s, h in got.items() if h is not None and len(h.closes) >= 2}
    # Every fund's returns end on the latest day ALL of them have a price (see returns()).
    shared = set.intersection(*(set(h.dates) for h in usable.values())) if usable else set()
    end = max(shared) if shared else None
    end_date = dt.date(int(end[:4]), int(end[4:6]), int(end[6:8])) if end else None
    out: dict[str, dict] = {}
    for s in want:
        h = usable.get(s)
        if h is None:
            out[s] = {"symbol": s, "available": False}
            continue
        try:
            out[s] = {
                "symbol": s,
                "available": True,
                "history_start": _dates(h)[0].isoformat(),
                "last_date": _dates(h)[-1].isoformat(),
                "returns": returns(h, last=end_date),
                **worst_drop(h),
            }
        except Exception as e:  # noqa: BLE001 — one fund's bad data must not sink the others
            log.warning("fund_overlap: performance for %s failed: %s", s, redact(e))
            out[s] = {"symbol": s, "available": False}
    chart = None
    if include_series:
        try:
            chart = aligned_chart({s: h for s, h in usable.items() if out[s]["available"]})
        except Exception as e:  # noqa: BLE001
            log.warning("fund_overlap: comparison chart failed: %s", redact(e))
    # The day every return above is measured to; null when the histories share no day.
    return {"funds": out, "chart": chart, "aligned_to": end_date.isoformat() if end_date else None, "as_of": now}
