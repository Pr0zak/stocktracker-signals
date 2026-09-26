"""What a fund costs, and which other funds hold the same thing for less (FC-1).

An expense ratio is a yearly percentage of what you hold, taken out of the fund's price a sliver at a
time. Nobody ever sends a bill, which is exactly what makes it easy to miss: 0.0945% reads like
nothing, and it is $9.45 a year on every $10,000 — three times what VOO charges for the same 500
companies. The app turns the percentage into dollars; this module supplies the percentages and the
set of funds each one can honestly be compared with.

WHERE THE FEES COME FROM. Yahoo's v7 quote carries `netExpenseRatio` for ETFs and mutual funds alike,
and on 2026-09-26 it was right for exactly the funds the older quoteSummary field gets wrong: FZROX
and FZILX came back 0.00%, their real price, where `fundProfile.annualReportExpenseRatio` had said
1.760% and 0.590%. That older failure is why main.py's saved table carries no mutual funds, and why
this module never falls back to that table for one. Live figures are cached for FEE_TTL_SECONDS.
When Yahoo does not answer, the last live figure is served with its own timestamp, then the saved
table's figure with the table's date, and otherwise the fee is None — unknown, never 0. One figure
Yahoo gives is never taken at its word: 0% on an ETF (see ISSUER_FEES).

WHAT COUNTS AS "THE SAME THING". The user asked to compare a generic ETF with "pretty much the same"
fund at Fidelity, where they buy. A name is not evidence of sameness, so every group below was
MEASURED on 2026-09-26 over two years of daily closes (Yahoo, dividends reinvested): within a group,
every pair's returns correlate at 0.995 or better, and their two-year total returns differ by at most
3 percentage points once the fee gap itself is taken out. Returns are weekly, or four-weekly when one
side is a mutual fund: a fund priced once a day at a fair-valued NAV carries timing noise that washes
out over a month. FXNAX shows it plainly — its tracking error against AGG falls from 1.36 points a
year on daily returns to 0.32 on four-weekly ones, and its two-year return lands within 0.2 points.
A real difference does not wash out that way. `research/fund_peers.py` re-runs the check live.

Look-alikes that FAILED that check, and so are deliberately not offered as cheaper versions of each
other:
  - IEFA / EFA / FSPSX vs VEA: about 9 points apart. MSCI EAFE leaves out Canada and VEA's FTSE index
    does not, so they are two groups.
  - IEMG / EEM vs VWO: about 18 points apart. MSCI counts South Korea as an emerging market and FTSE
    does not.
  - IEMG vs EEM, same MSCI family: EEM came out 3.65 points ahead before fees — IEMG also holds small
    companies — with a tracking error that stays near 1.2 points a year at every interval. And FPADX
    (Fidelity Emerging Markets Index) sits 5 points from IEMG. So no MSCI emerging-markets group.
  - The Select Sector SPDRs (XLK, XLC, ...) vs the Vanguard and Fidelity sector funds: up to 8 points.
    The SPDRs hold only S&P 500 members. Vanguard's and Fidelity's sector funds follow closely related
    MSCI indexes and pass easily, so those pairs are grouped and the SPDRs are not.
  - Dividend funds (SCHD, VYM, DGRO, FDVV) correlate at only 0.72-0.89 with one another: different
    stocks. SPMO vs MTUM is 11 points apart. VUG / SCHG / SPYG vs IWF are 5-16 points apart. None of
    them is a near-copy of another.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

import httpx

from . import market_now
from .redact import redact

log = logging.getLogger("signals.fund_cost")

# Issuers cut fees a few times a decade, not a few times a day. Half a day keeps the card honest
# about "checked at" without a Yahoo call on every screen open.
FEE_TTL_SECONDS = 12 * 3600
# A symbol Yahoo did not return at all (a typo, a delisting, a renamed ticker) is asked about again
# sooner, so a transient gap in Yahoo's answer does not stick for half a day.
MISSING_TTL_SECONDS = 3600
# One v7 call per chunk. The largest group is eleven funds, so even a full request stays one or two
# calls; the chunk only keeps the URL a sane length.
_CHUNK = 60
# Most symbols one request may ask about (each also pulls in its group).
MAX_SYMBOLS = 20


@dataclass(frozen=True)
class Group:
    """Funds that hold the same thing. [label] is printed by the app as-is, so it is plain words."""
    id: str
    label: str
    members: tuple[str, ...]
    note: str | None = None


GROUPS: tuple[Group, ...] = (
    Group("sp500", "the S&P 500, 500 of the biggest US companies",
          ("VOO", "IVV", "SPYM", "SPY", "FXAIX", "FNILX")),
    Group("total_us", "the whole US stock market",
          ("VTI", "ITOT", "SCHB", "SPTM", "FSKAX", "FZROX")),
    Group("total_intl", "stocks outside the US, rich and emerging countries",
          ("VXUS", "IXUS", "VEU", "FTIHX", "FZILX")),
    Group("developed_intl", "rich countries outside the US, Canada included",
          ("VEA", "SCHF", "SPDW")),
    Group("eafe", "rich countries outside the US and Canada",
          ("IEFA", "EFA", "FSPSX")),
    Group("em_ftse", "emerging markets, not counting South Korea",
          ("VWO", "SPEM", "SCHE")),
    Group("nasdaq100", "the Nasdaq-100, the 100 biggest Nasdaq companies",
          ("QQQ", "QQQM")),
    Group("nasdaq_all", "every stock on the Nasdaq",
          ("ONEQ", "FNCMX")),
    Group("us_bonds", "high-quality US bonds",
          ("BND", "AGG", "SCHZ", "FXNAX")),
    Group("russell2000", "2,000 small US companies (Russell 2000)",
          ("IWM", "VTWO", "FSSNX")),
    Group("sp600", "600 small US companies (S&P 600)",
          ("IJR", "SPSM", "VIOO")),
    Group("sp400", "400 mid-size US companies (S&P 400)",
          ("IJH", "SPMD", "IVOO", "MDY")),
    Group("r1000_growth", "big US growth companies (Russell 1000 Growth)",
          ("IWF", "VONG", "FSPGX")),
    Group("r1000_value", "big US value companies (Russell 1000 Value)",
          ("IWD", "VONV", "FLCOX")),
    Group("sp500_growth", "the growth half of the S&P 500",
          ("IVW", "VOOG", "SPYG")),
    Group("sp500_value", "the value half of the S&P 500",
          ("IVE", "VOOV", "SPYV")),
    Group("gold", "gold bars held in a vault",
          ("GLD", "IAU", "GLDM", "IAUM", "SGOL", "OUNZ", "AAAU")),
    Group("bitcoin", "bitcoin",
          ("IBIT", "FBTC", "BITB", "ARKB", "BTCO", "HODL", "BRRR", "GBTC", "BTC", "EZBC", "BTCW")),
    # Over the two measured years ETHE trailed ETHA by 2.9 points while charging 2.25 points a year
    # more, and ETH beat ETHA by 2.0 on a 0.10-point fee gap: staking rewards, passed through. For
    # those funds the fee alone overstates what holding them costs.
    Group("ether", "ether",
          ("ETHA", "FETH", "ETHW", "ETHV", "ETH", "ETHE", "EZET", "QETH"),
          note="Some ether funds also earn staking rewards, so the fee is not the whole story."),
    Group("tech", "US tech stocks", ("VGT", "FTEC")),
    Group("health", "US health-care stocks", ("VHT", "FHLC")),
    Group("financials", "US bank and finance stocks", ("VFH", "FNCL")),
    Group("energy", "US energy stocks", ("VDE", "FENY")),
    Group("utilities", "US utility stocks", ("VPU", "FUTY")),
    Group("industrials", "US industrial stocks", ("VIS", "FIDU")),
    Group("materials", "US materials stocks", ("VAW", "FMAT")),
    Group("real_estate", "US real-estate stocks", ("VNQ", "FREL")),
    Group("communication", "US communication stocks", ("VOX", "FCOM")),
    Group("staples", "US everyday-goods stocks", ("VDC", "FSTA")),
    Group("discretionary", "US shopping and leisure stocks", ("VCR", "FDIS")),
)

_GROUP_OF: dict[str, Group] = {m: g for g in GROUPS for m in g.members}

# Fidelity's own funds. The user buys on Fidelity, which is the whole reason the comparison exists,
# so the app marks them.
FIDELITY = frozenset({
    "FXAIX", "FNILX", "FSKAX", "FZROX", "FTIHX", "FZILX", "FSPSX", "FNCMX", "ONEQ", "FXNAX", "FSSNX",
    "FSPGX", "FLCOX", "FBTC", "FETH", "FTEC", "FHLC", "FNCL", "FENY", "FUTY", "FIDU", "FMAT", "FREL",
    "FCOM", "FSTA", "FDIS",
})

# Mutual funds rather than ETFs: bought and sold once a day, at that day's closing price. Yahoo's
# quoteType says the same when it answers; this is what the payload relies on when it does not.
MUTUAL_FUNDS = frozenset({
    "FXAIX", "FNILX", "FSKAX", "FZROX", "FTIHX", "FZILX", "FSPSX", "FNCMX", "FXNAX", "FSSNX", "FSPGX",
    "FLCOX",
})

# Fidelity ZERO funds: 0% by design rather than by a waiver, and held ONLY at Fidelity. Moving to
# another broker means selling them, which in a taxable account realises every gain at once.
FIDELITY_ONLY = frozenset({"FZROX", "FZILX", "FNILX"})

# Fees read off the fund companies' own pages, for funds where Yahoo's figure is wrong or missing.
#
# Wrong: an ETF at exactly 0% is running a fee waiver, and Yahoo keeps showing a waiver after it
# ends. On 2026-09-26 every ETF it listed at 0% was charging real money, per the issuers: VanEck's
# August 2026 ETF guide prices HODL at 0.20% gross and net; CoinShares' BRRR page states a 0.25%
# sponsor fee and no waiver; Bitwise's ETHW page says its waiver ended 2025-01-22, 0.20% since. So a
# 0% ETF figure from Yahoo is never believed; these stand in for it, and any other ETF listed at 0%
# is reported as unknown. Fidelity's ZERO funds really are free, and are mutual funds, so this does
# not touch them.
#
# Missing: Yahoo has no fee at all for SPYM (SPLG's ticker since 2025-10-31) and files it as an
# EQUITY. State Street's SPYM page gives 0.02%.
#
# Only a missing or zero Yahoo figure is replaced, so a real fee change Yahoo does pick up still wins.
ISSUER_FEES: dict[str, float] = {"HODL": 0.20, "BRRR": 0.25, "ETHW": 0.20, "SPYM": 0.02}
ISSUER_FEES_CHECKED = "2026-09-26"

# SYMBOL -> (fetched_at, {"long_name", "quote_type", "fee"}), or (fetched_at, None) for a symbol
# Yahoo did not return.
_cache: dict[str, tuple[float, dict | None]] = {}

Fetch = Callable[[httpx.AsyncClient, list[str]], Awaitable[dict[str, dict]]]


def group_of(symbol: str) -> Group | None:
    return _GROUP_OF.get(symbol.upper())


def _fresh(symbol: str, now: float) -> bool:
    hit = _cache.get(symbol)
    if hit is None:
        return False
    ttl = FEE_TTL_SECONDS if hit[1] is not None else MISSING_TTL_SECONDS
    return now - hit[0] < ttl


def _kind(symbol: str, live: dict | None, saved: dict[str, float]) -> str:
    if symbol in MUTUAL_FUNDS:
        return "mutual_fund"
    # Every other grouped symbol is an ETF, whatever Yahoo says: it files SPYM (SPLG's ticker since
    # 2025-10-31) under EQUITY, with no fee at all.
    if symbol in _GROUP_OF:
        return "etf"
    qt = (live or {}).get("quote_type")
    if qt == "ETF":
        return "etf"
    if qt == "MUTUALFUND":
        return "mutual_fund"
    if qt:
        return "other"
    return "etf" if symbol in saved else "unknown"


def _row(symbol: str, saved: dict[str, float], saved_as_of: str) -> dict:
    fetched_at, live = _cache.get(symbol, (None, None))
    name = (live or {}).get("long_name")
    kind = _kind(symbol, live, saved)
    fee = (live or {}).get("fee")
    listed_zero = kind == "etf" and fee == 0.0      # see ISSUER_FEES: a waiver, possibly long over
    if listed_zero:
        fee = None
    source = "yahoo" if fee is not None else None
    checked_at = fetched_at if fee is not None else None
    dated = None
    if fee is None and symbol in ISSUER_FEES:
        fee, source, dated = ISSUER_FEES[symbol], "issuer", ISSUER_FEES_CHECKED
    # The saved table is for ETFs only — see the module docstring for what its source got wrong for
    # mutual funds — and a zero in it is the same stale waiver Yahoo shows, so neither is used.
    elif fee is None and symbol not in MUTUAL_FUNDS and saved.get(symbol):
        fee, source, dated = saved[symbol], "saved", saved_as_of
    return {
        "symbol": symbol,
        "name": name,
        "kind": kind,
        "expense_ratio_pct": fee,
        # "yahoo" = read live at `fee_checked_at` (epoch seconds, possibly older than the TTL when
        # Yahoo stopped answering). "issuer" = the fund company's own figure, checked on `fee_dated`
        # because Yahoo's was wrong. "saved" = main.py's table, dated `fee_dated`. None = unknown.
        "fee_source": source,
        "fee_checked_at": checked_at,
        "fee_dated": dated,
        # Yahoo listed this ETF at 0%, which was not believed. When the fee is still None, that is
        # the reason it is unknown.
        "listed_zero": listed_zero,
        "fidelity": symbol in FIDELITY,
        "fidelity_only": symbol in FIDELITY_ONLY,
        "staking": "staking" in (name or "").lower(),
    }


def _fee_order(row: dict) -> tuple:
    fee = row["expense_ratio_pct"]
    return (fee is None, fee if fee is not None else 0.0, row["symbol"])


async def lookup(
    client: httpx.AsyncClient,
    symbols: list[str],
    *,
    saved: dict[str, float],
    saved_as_of: str,
    fetch: Fetch = market_now.fetch_quotes,
    now: float | None = None,
) -> dict:
    """Fee rows for [symbols], each with its comparison group (itself included, cheapest first).

    `live` is False when Yahoo was asked and did not answer; the rows then say where each fee came
    from instead. Never raises for a Yahoo failure — a fee it cannot get is None, not an error, so
    one unreachable symbol cannot take the rest of a batch down with it.
    """
    now = time.time() if now is None else now
    want: list[str] = []
    for s in symbols:
        s = s.strip().upper()
        if s and s not in want:
            want.append(s)
    want = want[:MAX_SYMBOLS]

    needed: list[str] = []
    for s in want:
        g = _GROUP_OF.get(s)
        for m in (g.members if g else (s,)):
            if m not in needed:
                needed.append(m)

    stale = [s for s in needed if not _fresh(s, now)]
    live = True
    if stale:
        try:
            got: dict[str, dict] = {}
            for i in range(0, len(stale), _CHUNK):
                got.update(await fetch(client, stale[i:i + _CHUNK]))
            for s in stale:
                q = got.get(s)
                _cache[s] = (now, None if q is None else {
                    "long_name": q.get("long_name") or q.get("name"),
                    "quote_type": q.get("quote_type"),
                    "fee": q.get("expense_ratio_pct"),
                })
        except Exception as e:  # noqa: BLE001 — the rows below say what is unknown
            live = False
            log.warning("fund_cost: Yahoo quote failed for %d symbols: %s", len(stale), redact(e))

    funds: dict[str, dict] = {}
    for s in want:
        row = _row(s, saved, saved_as_of)
        g = _GROUP_OF.get(s)
        row["group"] = None if g is None else {
            "id": g.id,
            "label": g.label,
            "note": g.note,
            "funds": sorted((_row(m, saved, saved_as_of) for m in g.members), key=_fee_order),
        }
        funds[s] = row
    return {"funds": funds, "live": live, "as_of": now}
