"""A catalogue of well-known funds to explore by return, fee and type (FUND-8).

The rest of the Funds screen starts from the user's own list. This is for the funds they do NOT own
yet: about 180 widely held ETFs, plus the Fidelity index funds that copy them (the user buys on
Fidelity), each with a plain name and a type. They are measured the way the rest of the Funds
screen measures: dividends-in returns to one shared day, and fees as fund_cost reads them.

WHAT IS IN IT. Broad US and world index funds, growth / value / size styles, dividend and
option-income funds, the US sectors (SPDR, Vanguard and Fidelity versions), some popular themes,
bonds from T-bills to long Treasuries, gold, silver and oil, and the spot bitcoin and ether funds.
No leveraged or inverse funds: a 3x fund tops or bottoms any ranking by construction and is not
built to be held for years. The list is picked by hand so every name is plain words; being on it is
not a claim that a fund is good.

CHEAPER COPIES come only from fund_cost.GROUPS, where sameness was measured. Funds that share a
plain name here but not a group are only alike: XLK holds the S&P 500's tech stocks, VGT the whole
US tech sector, and they have drifted up to 8 points apart. The app calls those "similar".

WINDOWS. Each fund's return AND its worst drop are measured over the same 1-, 3- and 5-year windows,
all ending on `aligned_to`. A worst drop over "all the history there is" would flatter a young fund
that never lived through 2022, so a fund too young for a window has no figure for it (None, shown
"—"), and a ranking by smallest drop compares like with like.

BUILDING. One five-year price history per fund (shared with the rest of the Funds screen through
fund_overlap's series cache) and one batch of quotes. Built in the background at startup and again
whenever the last build is older than EXPLORE_TTL_SECONDS. A request in between is answered from the
last finished build, with its time; only the very first request after a restart waits for one.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time
from dataclasses import dataclass

import httpx

from . import fund_cost, fund_overlap
from .market import Series
from .redact import redact

log = logging.getLogger("signals.fund_catalog")

EXPLORE_TTL_SECONDS = 6 * 3600
# A fund whose last price is this many days behind the newest one is left unmeasured rather than
# allowed to pull every other fund's end date back with it.
STALE_BAR_DAYS = 5
WINDOWS: tuple[tuple[str, int], ...] = (("1y", 1), ("3y", 3), ("5y", 5))


@dataclass(frozen=True)
class Entry:
    symbol: str
    name: str        # plain words, printed as-is
    category: str    # a key of CATEGORIES


# Order is the order of the app's chips.
CATEGORIES: tuple[tuple[str, str], ...] = (
    ("us", "US stocks"),
    ("style", "Growth & value"),
    ("size", "Small & mid"),
    ("income", "Dividends"),
    ("sector", "Sectors"),
    ("theme", "Themes"),
    ("world", "Outside the US"),
    ("bonds", "Bonds & cash"),
    ("commodities", "Gold & oil"),
    ("crypto", "Crypto"),
)


def _e(category: str, name: str, *symbols: str) -> tuple[Entry, ...]:
    return tuple(Entry(s, name, category) for s in symbols)


CATALOG: tuple[Entry, ...] = (
    *_e("us", "S&P 500", "VOO", "IVV", "SPY", "SPYM", "FXAIX", "FNILX"),
    *_e("us", "Whole US market", "VTI", "ITOT", "SCHB", "SPTM", "FSKAX", "FZROX"),
    *_e("us", "Nasdaq-100", "QQQ", "QQQM"),
    *_e("us", "Whole Nasdaq", "ONEQ", "FNCMX"),
    *_e("us", "Dow 30", "DIA"),
    *_e("us", "S&P 500, equal weight", "RSP"),

    *_e("style", "Growth stocks", "VUG", "SCHG"),
    *_e("style", "Big growth (Russell)", "IWF", "VONG", "FSPGX"),
    *_e("style", "S&P 500 growth", "IVW", "VOOG", "SPYG"),
    *_e("style", "Value stocks", "VTV", "SCHV"),
    *_e("style", "Big value (Russell)", "IWD", "VONV", "FLCOX"),
    *_e("style", "S&P 500 value", "IVE", "VOOV", "SPYV"),
    *_e("style", "Momentum stocks", "MTUM"),
    *_e("style", "S&P 500 momentum", "SPMO"),
    *_e("style", "Quality stocks", "QUAL"),
    *_e("style", "Low-swing stocks", "USMV"),

    *_e("size", "2,000 small companies", "IWM", "VTWO", "FSSNX"),
    *_e("size", "600 small companies", "IJR", "SPSM", "VIOO"),
    *_e("size", "Small companies", "VB", "SCHA"),
    *_e("size", "Small value", "AVUV"),
    *_e("size", "400 mid-size companies", "IJH", "SPMD", "IVOO", "MDY"),
    *_e("size", "Mid-size companies", "VO"),

    *_e("income", "Dividend stocks", "SCHD"),
    *_e("income", "High dividend", "VYM", "HDV"),
    *_e("income", "Dividend growers", "DGRO", "VIG"),
    *_e("income", "25+ years of raises", "NOBL"),
    *_e("income", "S&P 500 high dividend", "SPYD"),
    *_e("income", "Option income", "JEPI"),
    *_e("income", "Nasdaq option income", "JEPQ"),
    *_e("income", "Nasdaq covered calls", "QYLD"),

    *_e("sector", "Tech", "XLK", "VGT", "FTEC"),
    *_e("sector", "Chip makers", "SMH", "SOXX"),
    *_e("sector", "Software", "IGV"),
    *_e("sector", "Health care", "XLV", "VHT", "FHLC"),
    *_e("sector", "Biotech", "XBI", "IBB"),
    *_e("sector", "Banks & finance", "XLF", "VFH", "FNCL"),
    *_e("sector", "Regional banks", "KRE"),
    *_e("sector", "Energy", "XLE", "VDE", "FENY"),
    *_e("sector", "Oil & gas drillers", "XOP"),
    *_e("sector", "Industrials", "XLI", "VIS", "FIDU"),
    *_e("sector", "Defense & aerospace", "ITA"),
    *_e("sector", "Shopping & leisure", "XLY", "VCR", "FDIS"),
    *_e("sector", "Everyday goods", "XLP", "VDC", "FSTA"),
    *_e("sector", "Utilities", "XLU", "VPU", "FUTY"),
    *_e("sector", "Materials", "XLB", "VAW", "FMAT"),
    *_e("sector", "Real estate", "XLRE", "VNQ", "FREL"),
    *_e("sector", "Communication", "XLC", "VOX", "FCOM"),

    *_e("theme", "Disruptive tech", "ARKK"),
    *_e("theme", "Robotics & AI", "BOTZ"),
    *_e("theme", "Cybersecurity", "CIBR"),
    *_e("theme", "Clean energy", "ICLN"),
    *_e("theme", "Solar", "TAN"),
    *_e("theme", "Uranium", "URA"),
    *_e("theme", "US infrastructure", "PAVE"),
    *_e("theme", "Home builders", "ITB"),
    *_e("theme", "Airlines", "JETS"),
    *_e("theme", "Retail", "XRT"),

    *_e("world", "All non-US stocks", "VXUS", "IXUS", "VEU", "FTIHX", "FZILX"),
    *_e("world", "Whole world", "VT"),
    *_e("world", "Rich countries", "VEA", "SCHF", "SPDW"),
    *_e("world", "Rich countries, no Canada", "IEFA", "EFA", "FSPSX"),
    *_e("world", "Emerging markets", "VWO", "SCHE", "SPEM"),
    *_e("world", "Emerging, with Korea", "IEMG", "EEM"),
    *_e("world", "Europe", "VGK"),
    *_e("world", "Japan", "EWJ"),
    *_e("world", "India", "INDA"),
    *_e("world", "China big companies", "FXI"),
    *_e("world", "China internet", "KWEB"),
    *_e("world", "Brazil", "EWZ"),

    *_e("bonds", "US bonds", "BND", "AGG", "SCHZ", "FXNAX"),
    *_e("bonds", "T-bills, like cash", "SGOV", "BIL"),
    *_e("bonds", "Short Treasuries", "SHY", "VGSH"),
    *_e("bonds", "7-10 year Treasuries", "IEF"),
    *_e("bonds", "Long Treasuries", "TLT"),
    *_e("bonds", "Inflation-protected", "TIP"),
    *_e("bonds", "Company bonds", "LQD"),
    *_e("bonds", "High-yield bonds", "HYG"),
    *_e("bonds", "Tax-free city & state", "MUB"),
    *_e("bonds", "Non-US bonds", "BNDX"),

    *_e("commodities", "Gold", "GLD", "IAU", "GLDM", "IAUM", "SGOL", "AAAU", "OUNZ"),
    *_e("commodities", "Silver", "SLV"),
    *_e("commodities", "Gold miners", "GDX"),
    *_e("commodities", "Copper miners", "COPX"),
    *_e("commodities", "Crude oil", "USO"),

    *_e("crypto", "Bitcoin", "IBIT", "FBTC", "BITB", "ARKB", "GBTC", "HODL", "BTCO", "BRRR", "EZBC", "BTCW"),
    # Grayscale's cheaper spin-offs trade as BTC and ETH, the same letters as the coins themselves.
    *_e("crypto", "Bitcoin (mini trust)", "BTC"),
    *_e("crypto", "Ether", "ETHA", "FETH", "ETHE", "ETHW", "ETHV", "EZET", "QETH"),
    *_e("crypto", "Ether (mini trust)", "ETH"),
)

_ENTRY: dict[str, Entry] = {e.symbol: e for e in CATALOG}

_built: dict | None = None
_task: asyncio.Task | None = None


def entry(symbol: str) -> Entry | None:
    return _ENTRY.get(symbol.upper())


def _drop(closes: list[float]) -> float | None:
    """The deepest fall from a running high, in percent (0.0 when it never fell)."""
    if len(closes) < 2:
        return None
    peak, worst = closes[0], 0.0
    for c in closes:
        peak = max(peak, c)
        worst = min(worst, c / peak - 1)
    return round(worst * 100, 1) + 0.0     # + 0.0 turns a rounded -0.0 into 0.0


def window_stats(s: Series, last: dt.date) -> tuple[dict, dict]:
    """Return and worst drop over each window in WINDOWS, all ending on the bar at or before
    [last]. None where the fund's history does not reach back that far."""
    ds = fund_overlap._dates(s)
    rets: dict[str, float | None] = {k: None for k, _ in WINDOWS}
    drops: dict[str, float | None] = {k: None for k, _ in WINDOWS}
    end = fund_overlap.end_index(ds, last)
    if end is None:
        return rets, drops
    for label, years in WINDOWS:
        i = fund_overlap.window_start(ds, end, years)
        if i is None:
            continue
        rets[label] = round((s.closes[end] / s.closes[i] - 1) * 100, 2)
        drops[label] = _drop(s.closes[i:end + 1])
    return rets, drops


def cheaper_copy(symbol: str, rows: dict[str, dict]) -> dict | None:
    """The cheapest fund measured to hold the same thing, when it costs less: an ETF when one is
    cheaper, a mutual fund only when no cheaper ETF exists (the same rule the app uses)."""
    g = fund_cost.group_of(symbol)
    fee = rows.get(symbol, {}).get("expense_ratio_pct")
    if g is None or fee is None:
        return None
    peers = [rows[m] for m in g.members
             if m != symbol and m in rows and rows[m]["expense_ratio_pct"] is not None
             and rows[m]["expense_ratio_pct"] < fee - 1e-9]
    etfs = [r for r in peers if r["kind"] == "etf"]
    pick = min(etfs or peers, key=lambda r: (r["expense_ratio_pct"], r["symbol"]), default=None)
    if pick is None:
        return None
    e = entry(pick["symbol"])
    return {
        "symbol": pick["symbol"],
        "name": e.name if e else None,
        "long_name": pick["name"],
        "expense_ratio_pct": pick["expense_ratio_pct"],
        "saves_per_10k": round((fee - pick["expense_ratio_pct"]) * 100.0, 2),
        "mutual_fund": pick["kind"] == "mutual_fund",
        "fidelity_only": pick["fidelity_only"],
    }


def _end_date(hists: dict[str, Series]) -> tuple[dt.date | None, set[str]]:
    """The day every return is measured to, and the funds too far behind it to be measured.

    The end is the earliest last bar among the funds that are current: during the session an ETF's
    chart already carries today's live bar while a mutual fund's NAV for today arrives that evening,
    so ending on the newest bar would measure them over different days (see fund_overlap.returns).
    A fund whose last bar is more than STALE_BAR_DAYS behind the newest is left out instead of
    dragging everyone else's end date back with it.
    """
    last = {s: fund_overlap._dates(h)[-1] for s, h in hists.items()}
    if not last:
        return None, set()
    newest = max(last.values())
    stale = {s for s, d in last.items() if (newest - d).days > STALE_BAR_DAYS}
    current = [d for s, d in last.items() if s not in stale]
    return (min(current) if current else None), stale


async def build(
    client: httpx.AsyncClient,
    *,
    saved: dict[str, float],
    saved_as_of: str,
    fetch_history: fund_overlap.HistoryFetch = fund_overlap._yahoo_history,
    quotes: fund_cost.Fetch | None = None,
    now: float | None = None,
) -> dict:
    """Measure every catalogue fund. Never raises for a Yahoo failure: a fund it cannot read comes
    back `available: False`, and a fee it cannot find is None."""
    now = time.time() if now is None else now
    symbols = [e.symbol for e in CATALOG]
    # Every group member too, so a cheaper copy outside the catalogue (EZBC, BTCW, ...) is priced.
    wanted = list(dict.fromkeys(symbols + [m for g in fund_cost.GROUPS for m in g.members]))
    live = await fund_cost.refresh(client, wanted, now=now, **({"fetch": quotes} if quotes is not None else {}))
    rows = {s: fund_cost.row(s, saved=saved, saved_as_of=saved_as_of) for s in wanted}

    sem = asyncio.Semaphore(fund_overlap._CONCURRENCY)

    async def hist(s: str):
        async with sem:
            return await fund_overlap._cached(
                fund_overlap._series, s, fund_overlap.SERIES_TTL_SECONDS, now,
                lambda: fund_overlap._load_history(fetch_history, client, s))

    got = dict(zip(symbols, await asyncio.gather(*[hist(s) for s in symbols])))
    usable = {s: h for s, h in got.items() if h is not None and len(h.closes) >= 2}
    end, stale = _end_date(usable)

    funds: list[dict] = []
    for e in CATALOG:
        r = rows[e.symbol]
        g = fund_cost.group_of(e.symbol)
        h = usable.get(e.symbol)
        out = {
            "symbol": e.symbol,
            "name": e.name,
            "category": e.category,
            "long_name": r["name"],
            "kind": r["kind"] if r["kind"] in ("etf", "mutual_fund") else "etf",
            "expense_ratio_pct": r["expense_ratio_pct"],
            "fee_source": r["fee_source"],
            "fee_dated": r["fee_dated"],
            "listed_zero": r["listed_zero"],
            "fidelity": r["fidelity"],
            "fidelity_only": r["fidelity_only"],
            "net_assets": r["net_assets"],
            "group_id": g.id if g else None,
            "group_label": g.label if g else None,
            "cheaper": cheaper_copy(e.symbol, rows),
            "available": False,
            "history_start": None,
            "returns": {k: None for k, _ in WINDOWS},
            "drops": {k: None for k, _ in WINDOWS},
        }
        if h is not None and end is not None and e.symbol not in stale:
            try:
                rets, drops = window_stats(h, end)
                out.update(available=True, history_start=fund_overlap._dates(h)[0].isoformat(),
                           returns=rets, drops=drops)
            except Exception as ex:  # noqa: BLE001 — one fund's bad data must not sink the rest
                log.warning("fund_catalog: %s failed: %s", e.symbol, redact(ex))
        funds.append(out)

    counts = {c: 0 for c, _ in CATEGORIES}
    for f in funds:
        counts[f["category"]] += 1
    return {
        "categories": [{"id": c, "label": label, "count": counts[c]} for c, label in CATEGORIES],
        "funds": funds,
        "aligned_to": end.isoformat() if end else None,
        "measured": sum(1 for f in funds if f["available"]),
        "live": live,
        "built_at": now,
    }


async def _build_and_keep(client: httpx.AsyncClient, saved: dict[str, float], saved_as_of: str) -> dict:
    global _built
    got = await build(client, saved=saved, saved_as_of=saved_as_of)
    # A build that measured nothing (Yahoo down) does not replace one that measured something.
    if _built is None or got["measured"] > 0:
        _built = got
    return _built


def _start(client: httpx.AsyncClient, saved: dict[str, float], saved_as_of: str) -> asyncio.Task:
    global _task
    if _task is None or _task.done():
        _task = asyncio.create_task(_build_and_keep(client, saved, saved_as_of))
    return _task


async def warm(client: httpx.AsyncClient, *, saved: dict[str, float], saved_as_of: str, delay: float = 20.0) -> None:
    """Build once shortly after startup, so the first person to open Explore does not wait."""
    await asyncio.sleep(delay)
    try:
        await _start(client, saved, saved_as_of)
    except Exception as e:  # noqa: BLE001 — the next request retries
        log.warning("fund_catalog: warm-up build failed: %s", redact(e))


async def explore(
    client: httpx.AsyncClient,
    *,
    saved: dict[str, float],
    saved_as_of: str,
    now: float | None = None,
) -> dict:
    """The last finished build, refreshed in the background when it is old. `refreshing` says a
    newer one is on its way; `built_at` says how old this one is."""
    now = time.time() if now is None else now
    if _built is None:
        await asyncio.shield(_start(client, saved, saved_as_of))
    elif now - _built["built_at"] > EXPLORE_TTL_SECONDS:
        _start(client, saved, saved_as_of)
    assert _built is not None
    return {**_built, "refreshing": _task is not None and not _task.done(), "as_of": now}
