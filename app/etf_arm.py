"""ETF-only sandbox arms: which funds they may buy, how look-alike funds share one cap, which copy a
buy lands on, and the fixed mix the no-AI arm follows.

Two arms use this, and they are each other's control. `etf` is the analyst: a weekly plan written
from each fund group's fee, return and worst drop, and a daily decision that fills it. `etf-rules`
has no model at all: a target-date style mix set by age and risk tolerance, filled mechanically.
Both buy only from the fund catalogue (app/fund_catalog.py), ETFs only, and both route every buy
through the same cheapest-copy rule. If the analyst adds anything, it shows up as the gap between
those two curves.

HOW COST AND HISTORY ARE USED, and why they are used differently.

Cost picks the FUND. Among funds measured to hold the same thing (fund_cost.GROUPS: weekly-return
correlation of 0.995 or more), the fee is almost the whole difference in what an owner keeps, so a
buy of any member is routed onto the cheapest ETF in the group. That is code, not a prompt
instruction: on 2026-08-06 the model was TOLD cheapest-first applied to new money only, and sold SPY
to buy VTI on the very next tick. Routing never touches a sell and never splits a holding: new money
joins the copy the book already owns (sandbox_job.prefer_vehicle's rules, reused unchanged).

History sets the MIX and the SIZE, never a ranking to chase. A fund's worst drop over 1, 3 and 5
years says how much of the book it can carry through a bad year; its past return, across different
kinds of funds, says very little about the next five years — Yahoo's own "top ETFs" ranking returned
gold miners, metals miners and esports on 2026-08-14. The analyst sees both, and the prompt says
which job each one does.

CAP GROUPS. The per-group cap stops one exposure dominating the book. With ETFs that only works if
funds that move as one share a cap: otherwise VUG, SCHG, IWF and VONG could each sit under 20% and
put 80% of the book in one bet (the 2026-08-05 failure, where VTI and SPY each passed a 25% cap
while together holding 46.3%). ETF_GROUPS below is complete-linkage clustering at 0.90 on two years
of daily returns, measured 2026-09-28 by research/etf_groups.py. Complete linkage, not a chain: a
fund joins a group only if it moves with EVERY member at 0.90 or more, so SPMO is not chained into
the S&P group through QQQ (the same rule fund_overlap's "move together" sets use).

The grouping is the ETF arms' own vocabulary and differs from main's `_exposure_group` in places,
because main's map was built one anchor at a time on older data: main puts QQQ/QQQM and SPMO in
US_EQUITY; measured here QQQ/QQQM sit with the tech sector funds (0.965) and SPMO with MTUM (0.968).
The ETF arms never share a book with main, so the two maps never meet in one cap.

BROAD vs NARROW caps. A total-market fund holds thousands of companies; capping it at the 20% meant
for one stock would make an ordinary index portfolio (roughly half US stocks) impossible. So the
groups in BROAD_GROUPS use `broad_position_pct` (default 60) and every other group — sectors,
themes, single countries, gold, bitcoin — keeps `max_position_pct`.
"""
from __future__ import annotations

import datetime as dt
from typing import Callable, Iterable

from . import fund_catalog, fund_cost
from .sandbox_job import (BTC_ETFS, GOLD_ETFS, effective_age, prefer_vehicle)

UNIVERSE_ALL = "all"
UNIVERSE_ETF = "etf"
UNIVERSES = (UNIVERSE_ALL, UNIVERSE_ETF)

DEFAULT_BROAD_POSITION_PCT = 60.0

# Measured 2026-09-28: complete linkage at 0.90 over two years of daily returns for every ETF in the
# catalogue (research/etf_groups.py). The comment on each line is the lowest pairwise correlation
# inside the group. Funds absent from this map are their own group (their ticker), exactly as in
# main's `_exposure_group`.
ETF_GROUPS: dict[str, tuple[str, ...]] = {
    # 0.904. S&P 500, total US market, the growth indexes, quality and the whole world (VT is ~60% US
    # and moves with the S&P at 0.965 daily).
    "US_EQUITY": ("VOO", "SPYM", "IVV", "SPY", "VTI", "ITOT", "SCHB", "SPTM", "QUAL", "VT", "ONEQ",
                  "IVW", "VOOG", "SPYG", "VUG", "IWF", "VONG", "SCHG"),
    # 0.900. Value, the Dow, equal weight, dividend growers, mid caps.
    "US_VALUE": ("DIA", "RSP", "IWD", "VONV", "IVE", "VOOV", "SPYV", "VYM", "DGRO", "VIG", "VTV",
                 "SCHV", "VO"),
    # 0.909.
    "US_SMALL_MID": ("IWM", "VTWO", "SCHA", "VB", "IJH", "MDY", "IVOO", "SPMD", "IJR", "SPSM", "VIOO",
                     "AVUV"),
    # 0.965. The Nasdaq-100 and the tech sector funds.
    "US_TECH": ("QQQ", "QQQM", "XLK", "VGT", "FTEC"),
    "US_MOMENTUM": ("MTUM", "SPMO"),                           # 0.968
    "US_DIVIDEND": ("SCHD", "SPYD"),                           # 0.910
    "COVERED_CALL": ("JEPQ", "QYLD"),                          # 0.941
    "SEMIS": ("SMH", "SOXX"),                                  # 0.982
    "BIOTECH": ("XBI", "IBB"),                                 # 0.928
    "ENERGY": ("XLE", "VDE", "FENY", "XOP"),                   # 0.933
    "INDUSTRIALS": ("XLI", "VIS", "FIDU", "PAVE"),             # 0.937
    "HEALTH": ("XLV", "VHT", "FHLC"),                          # 0.990
    "FINANCIALS": ("XLF", "VFH", "FNCL"),                      # 0.994
    "DISCRETIONARY": ("XLY", "VCR", "FDIS"),                   # 0.993
    "STAPLES": ("XLP", "VDC", "FSTA"),                         # 0.985
    "UTILITIES": ("XLU", "VPU", "FUTY"),                       # 0.997
    "MATERIALS": ("XLB", "VAW", "FMAT"),                       # 0.984
    "REAL_ESTATE": ("XLRE", "VNQ", "FREL"),                    # 0.989
    "COMMUNICATION": ("XLC", "VOX", "FCOM"),                   # 0.965
    # 0.930. All non-US, rich countries, Europe.
    "INTL": ("VXUS", "IXUS", "VEU", "VEA", "SPDW", "SCHF", "IEFA", "EFA", "VGK"),
    "EM": ("VWO", "SPEM", "SCHE", "IEMG", "EEM"),              # 0.951
    "CHINA": ("FXI", "KWEB"),                                  # 0.935
    "US_BONDS": ("BND", "AGG", "SCHZ", "IEF"),                 # 0.969
    "SHORT_TREASURY": ("SHY", "VGSH"),                         # 0.956
    "LONG_BONDS": ("TLT", "LQD"),                              # 0.901
    # T-bill funds barely move, so a daily correlation between them measures rounding, not
    # sameness (0.64). Grouped by what they hold: three-month Treasury bills.
    "T_BILLS": ("SGOV", "BIL"),
    "GOLD": ("GLD", "IAU", "OUNZ", "IAUM", "SGOL", "AAAU", "GLDM"),                          # 0.999
    "BTC": ("IBIT", "FBTC", "BITB", "GBTC", "BRRR", "ARKB", "BTCO", "EZBC", "HODL", "BTCW", "BTC"),
    "ETH": ("ETHA", "FETH", "ETHE", "ETHW", "ETH", "ETHV", "EZET", "QETH"),                   # 0.999
}

_GROUP_OF: dict[str, str] = {s: g for g, members in ETF_GROUPS.items() for s in members}

# Groups diversified enough to carry the larger cap: thousands of companies, or government bonds.
BROAD_GROUPS = frozenset({"US_EQUITY", "INTL", "US_BONDS", "SHORT_TREASURY", "T_BILLS"})

# What the no-AI arm buys for each part of its mix. The cheapest-copy router then lands the buy on
# the cheapest fund holding the same index, so these name the INDEX, not a fee choice.
REPRESENTATIVES: dict[str, tuple[str, ...]] = {
    "US_EQUITY": ("VTI", "ITOT", "SCHB", "VOO"),
    "INTL": ("VXUS", "IXUS", "VEU"),
    "US_BONDS": ("BND", "AGG", "SCHZ"),
}


def is_etf_arm(settings: dict | None) -> bool:
    return str((settings or {}).get("universe") or UNIVERSE_ALL).lower() == UNIVERSE_ETF


# Every ETF in the catalogue, in catalogue order. Mutual funds are left out: they trade once a day
# at the closing NAV, and this ledger fills at the live price near the close.
_UNIVERSE: tuple[str, ...] = tuple(
    e.symbol for e in fund_catalog.CATALOG if e.symbol not in fund_cost.MUTUAL_FUNDS)
_UNIVERSE_SET = frozenset(_UNIVERSE)


def universe() -> list[str]:
    return list(_UNIVERSE)


def universe_set() -> frozenset[str]:
    return _UNIVERSE_SET


# Plain-word names a strategist may write for a group, resolved onto the real key. The vocabulary is
# handed to it verbatim, but a plan that says BONDS for US_BONDS would otherwise name a group no fund
# belongs to: canonicalize_targets would leave it, target_gaps would report it at 0% forever, and the
# daily model would be told to keep filling a gap no buy can close. None of these is a ticker on the
# fund list, so no fund can resolve to one by accident.
_ALIASES: dict[str, str] = {
    **{k: "US_EQUITY" for k in ("US_STOCKS", "US_STOCK", "US_MARKET", "TOTAL_US", "TOTAL_MARKET",
                                "SP500", "S&P500", "S&P_500", "US_LARGE_CAP", "US_CORE")},
    **{k: "INTL" for k in ("INTERNATIONAL", "INTL_EQUITY", "EX_US", "NON_US", "DEVELOPED",
                           "DEVELOPED_MARKETS", "INTERNATIONAL_EQUITY", "FOREIGN")},
    **{k: "EM" for k in ("EMERGING", "EMERGING_MARKETS")},
    **{k: "US_BONDS" for k in ("BONDS", "US_BOND", "AGGREGATE_BONDS", "CORE_BONDS", "BOND")},
    **{k: "SHORT_TREASURY" for k in ("SHORT_TREASURIES", "TREASURIES", "SHORT_BONDS")},
    **{k: "T_BILLS" for k in ("TBILLS", "T-BILLS", "BILLS", "CASH_LIKE")},
    **{k: "US_TECH" for k in ("TECH", "TECHNOLOGY", "NASDAQ", "NASDAQ100", "NASDAQ_100")},
    **{k: "US_VALUE" for k in ("VALUE",)},
    **{k: "US_SMALL_MID" for k in ("SMALL_CAP", "SMALL_CAPS", "MID_CAP", "SMALL_MID", "US_SMALL")},
    **{k: "US_DIVIDEND" for k in ("DIVIDEND", "DIVIDENDS")},
    **{k: "HEALTH" for k in ("HEALTHCARE", "HEALTH_CARE")},
    **{k: "BTC" for k in ("BITCOIN",)},
    **{k: "ETH" for k in ("ETHER", "ETHEREUM")},
}


def group_of(symbol: str) -> str:
    """The ETF arms' cap group for a ticker (or a group name, or a plain-word alias of one); a
    ticker outside the map is its own group."""
    base = str(symbol or "").strip().upper().removesuffix("-USD")
    return _GROUP_OF.get(base) or _ALIASES.get(base) or base


def known_groups() -> frozenset[str]:
    """Every label a plan may name: the groups plus every ungrouped fund on the list."""
    return frozenset(group_of(s) for s in _UNIVERSE)


def cap_pct_of(settings: dict) -> Callable[[str], float]:
    """group -> the per-group cap (percent of equity) for an ETF arm."""
    narrow = float(settings.get("max_position_pct", 20.0))
    broad = float(settings.get("broad_position_pct") or DEFAULT_BROAD_POSITION_PCT)
    # Never tighter than the narrow cap: a broad cap below it would make the index funds the most
    # restricted thing in the book, which inverts the reason for having two tiers.
    broad = max(broad, narrow)
    return lambda g: broad if str(g or "").upper() in BROAD_GROUPS else narrow


# How far past its plan target a group may be bought. Two points: enough that the last whole share
# of a target is not refused, small enough that an oversized order cannot rewrite the plan.
PLAN_TARGET_SLACK_PCT = 2.0


def target_limit_of(plan: dict | None, settings: dict) -> Callable[[str], float] | None:
    """group -> the most of equity (percent) buys may take it to, for validate_and_fill.

    A group the plan names may go to its target plus PLAN_TARGET_SLACK_PCT. A group the plan does
    NOT name keeps the narrow cap, broad or not: the broad cap is there so an index-fund PLAN is
    reachable, not so an unplanned buy can put 60% of the book in T-bills. None (no limit beyond
    the caps) when there is no plan to hold buys to — the first tick before the weekly review has
    run, or a review that failed."""
    targets = (plan or {}).get("targets") or []
    if not targets:
        return None
    want: dict[str, float] = {}
    for t in targets:
        g = group_of(str(t.get("exposure_group") or ""))
        try:
            want[g] = want.get(g, 0.0) + float(t.get("target_pct") or 0.0)
        except (TypeError, ValueError):
            continue
    narrow = float(settings.get("max_position_pct", 20.0))
    return lambda g: (want[g] + PLAN_TARGET_SLACK_PCT) if g in want else narrow


# ------------------------------------------------------------------ cheapest copy

def fee_table(build: dict | None) -> dict[str, tuple[float | None, float | None]]:
    """symbol -> (fee %, net assets) from a fund_catalog build. Empty when there is no build."""
    out: dict[str, tuple[float | None, float | None]] = {}
    for f in (build or {}).get("funds") or []:
        out[str(f.get("symbol") or "").upper()] = (f.get("expense_ratio_pct"), f.get("net_assets"))
    return out


def cheapest_member(members: Iterable[str], *, fees: dict[str, tuple[float | None, float | None]],
                    price_of: Callable[[str], float | None],
                    exclude: Iterable[str] = ()) -> str | None:
    """The cheapest fund among `members` that can actually be bought today: an ETF on this arm's
    list, not excluded, with a KNOWN fee and a live price. Ties go to the bigger fund (tighter
    spreads), then the ticker. None when nothing qualifies — an unknown fee is never treated as a
    free one."""
    etfs = universe_set()
    banned = {str(x).upper() for x in exclude}
    best: tuple | None = None
    for m in members:
        m = m.upper()
        if m not in etfs or m in banned:
            continue
        fee, assets = fees.get(m, (None, None))
        px = price_of(m)
        if fee is None or not px or px <= 0:
            continue
        key = (round(float(fee), 4), -float(assets or 0.0), m)
        if best is None or key < best:
            best = key
    return best[2] if best else None


def route_to_cheapest(
    orders: list[dict], *, positions: list[dict], fees: dict[str, tuple[float | None, float | None]],
    price_of: Callable[[str], float | None], exclude: set[str] | None = None,
) -> tuple[list[dict], list[str]]:
    """Send each BUY onto the cheapest ETF holding the same index, or onto the copy already held.

    One `prefer_vehicle` pass per fund_cost group, with that group's cheapest buyable ETF as the
    preference. prefer_vehicle already carries every rule this needs and its tests pin them: buys
    only, consolidate onto a held copy rather than open a second one, only route onto something
    priced, never onto an excluded ticker, convert a share count to dollars across share prices,
    drop an entry zone priced against the other fund, and say so in the order's reason.

    Bitcoin and gold are skipped: those have a user-chosen vehicle (preferred_btc_etf /
    preferred_gold_etf) that validate_and_fill applies, and routing them here first would override
    the user's choice with a fee comparison."""
    notes: list[str] = []
    for g in fund_cost.GROUPS:
        members = frozenset(m for m in g.members if m in universe_set())
        if len(members) < 2 or members & (BTC_ETFS | GOLD_ETFS):
            continue
        pref = cheapest_member(members, fees=fees, price_of=price_of, exclude=exclude or ())
        if not pref:
            continue
        fee = fees.get(pref, (None, None))[0]
        routed, n = prefer_vehicle(
            orders, preferred=pref, positions=positions, price_of=price_of, family=members,
            label=f"lowest-fee copy ({fee:.2f}%/yr)", exclude=exclude)
        # prefer_vehicle returns one order per order, in order. A FEE route that lands a small buy on
        # a copy with a dearer share (ITOT $130 -> VTI $300 for $200, same 0.03%) turns an order that
        # would have filled into an "under one share" refusal — losing the buy to save nothing, or a
        # basis point. Such a route is undone. A route onto a HELD copy is kept even then: that one
        # exists to stop a second position in one index, and a refusal that says so is the honest
        # outcome.
        held = {str(p.get("symbol", "")).upper() for p in positions if (p.get("shares") or 0) > 0}
        kept_notes = list(n)
        for i, (o, r) in enumerate(zip(orders, routed)):
            src, dst = str(o.get("symbol") or "").upper(), str(r.get("symbol") or "").upper()
            if src == dst or dst in held:
                continue
            dollars = float(r.get("dollars") or 0.0)
            px_dst, px_src = price_of(dst) or 0.0, price_of(src) or 0.0
            if 0 < px_src <= dollars < px_dst:
                routed[i] = o
                kept_notes = [x for x in kept_notes if not x.startswith(f"{src}→{dst} ")]
        orders = routed
        notes.extend(kept_notes)
    return orders, notes


# ------------------------------------------------------------------ the no-AI mix

def glidepath_stock_pct(years_to_retirement: float | None, risk_tolerance: str | None) -> float:
    """Share of the invested money in stock funds, from years to retirement.

    The shape of a Vanguard-style target-date fund: 90% stocks while retirement is 25+ years off,
    falling in a straight line to 50% at retirement, then to 30% seven years after. Risk tolerance
    moves the whole line: conservative -15 points, aggressive +10. Unknown runway uses 20 years."""
    y = 20.0 if years_to_retirement is None else float(years_to_retirement)
    if y >= 25:
        pct = 90.0
    elif y >= 0:
        pct = 50.0 + 40.0 * y / 25.0
    else:
        pct = max(30.0, 50.0 + 20.0 * y / 7.0)   # y is negative after retirement
    adj = {"conservative": -15.0, "aggressive": 10.0}.get(str(risk_tolerance or "").lower(), 0.0)
    return max(20.0, min(95.0, pct + adj))


US_SHARE_OF_STOCKS = 0.60      # target-date funds from Vanguard and Fidelity hold about 60/40 US/non-US


def glidepath_plan(settings: dict, today: dt.date | None = None) -> dict:
    """The no-AI arm's standing plan, in the same shape as a weekly StrategyNote, so rules_decision,
    target_gaps and the app read it exactly as they read the analyst's.

    Three funds: the whole US market, all non-US stocks, and US bonds. Recomputed every tick from
    the arm's settings, so it moves a little each birthday and never needs a review.

    Every target is kept under its group's cap. The bond share passes the 60% broad cap for a
    retired or conservative account (63% at 7+ years past retirement), and a target the ledger will
    never fill makes the arm re-propose the same refused buy every tick while the money sits in
    cash. Anything over a cap moves to the other groups that have room; what none can take becomes
    cash, and the notes say so."""
    today = today or dt.date.today()
    age = effective_age(settings, today)
    ret = settings.get("retirement_age")
    years = (float(ret) - float(age)) if (age is not None and ret) else None
    stock = glidepath_stock_pct(years, settings.get("risk_tolerance"))
    cash = max(0.0, float(settings.get("cash_floor_pct") or 0.0))
    invest = 100.0 - cash
    want = {"US_EQUITY": invest * stock / 100.0 * US_SHARE_OF_STOCKS,
            "INTL": invest * stock / 100.0 * (1 - US_SHARE_OF_STOCKS)}
    want["US_BONDS"] = invest - want["US_EQUITY"] - want["INTL"]
    caps = cap_pct_of(settings)
    over = sum(max(0.0, v - caps(g)) for g, v in want.items())
    want = {g: min(v, caps(g)) for g, v in want.items()}
    for g in want:                       # US first, then non-US, then bonds
        if over <= 1e-9:
            break
        take = min(over, caps(g) - want[g])
        want[g] += take
        over -= take
    extra_cash = max(0.0, over)
    targets = [{"exposure_group": g, "target_pct": round(v, 1)} for g, v in want.items()]
    # Rounding each target to one decimal can leave the sum a tenth off; the last target absorbs it
    # so the plan still sums to exactly (100 - cash), which is what allocation_gap audits.
    cash_target = round(cash + extra_cash, 1)
    drift = round(100.0 - cash_target - sum(t["target_pct"] for t in targets), 1)
    if drift:
        targets[-1]["target_pct"] = round(targets[-1]["target_pct"] + drift, 1)
    runway = f"{years:.0f} years to retirement" if years is not None else "runway unknown (20 years assumed)"
    return {
        "stance": "neutral",
        "cash_target_pct": cash_target,
        "targets": targets,
        "themes": ["Whole US market", "All non-US stocks", "US bonds"],
        "avoid": ["Picking funds by past return", "Selling to switch funds"],
        "notes": (f"Fixed mix, no AI. {stock:.0f}% stocks for {runway}, "
                  f"{settings.get('risk_tolerance') or 'balanced'} risk. "
                  f"Stocks split 60/40 US and non-US. Each buy goes to the cheapest copy."
                  + (f" {extra_cash:.0f}% held as cash: the caps leave no fund room for it."
                     if extra_cash >= 0.05 else "")),
        "source": "glidepath",
    }


# ------------------------------------------------------------------ what the analyst sees

# Technicals kept on an ETF candidate row. A fund row is about what it holds, what it costs and how
# deep it has fallen; the full stock-picking set (gaps, similar-setup track records) is noise here
# and costs ~100 tokens a row across ~100 rows.
_ROW_TECH_KEYS = ("pct_vs_sma50", "rsi14", "pct_off_52w_high", "rel_strength_3mo_vs_benchmark",
                  "golden_cross")
_ROW_LONG_KEYS = ("price_vs_200w_sma_pct", "zone")


def pool_symbols(build: dict | None, *, exclude: Iterable[str] = (), allow_crypto_etf: bool = True,
                 fees: dict | None = None, prefer: Iterable[str] = ()) -> list[str]:
    """One symbol per distinct holding: the cheapest ETF of each fund_cost group (by fee, then size),
    plus every catalogue ETF that belongs to no such group. This is the list the analyst is shown,
    before prices are known. Sorting by fee here is what makes the row the analyst reads the fund
    the router would land on anyway.

    `prefer` names funds that win their group outright — the user's chosen bitcoin and gold funds,
    which validate_and_fill routes buys onto whatever the fee says, so the row shown must be that
    fund or its fee would describe a fund the arm never buys."""
    fees = fees if fees is not None else fee_table(build)
    etfs = universe_set()
    banned = {str(x).upper() for x in exclude}
    preferred = {str(x).upper() for x in prefer if x}
    grouped: set[str] = set()
    out: list[str] = []
    for g in fund_cost.GROUPS:
        members = [m for m in g.members if m in etfs and m not in banned]
        grouped.update(m for m in g.members if m in etfs)
        if not members:
            continue
        chosen = next((m for m in members if m in preferred), None)
        pick = chosen or cheapest_member(members, fees=fees, price_of=lambda _s: 1.0, exclude=banned)
        out.append(pick or members[0])
    for s in _UNIVERSE:
        if s not in grouped and s not in banned:
            out.append(s)
    if not allow_crypto_etf:
        out = [s for s in out if group_of(s) not in ("BTC", "ETH")]
    return list(dict.fromkeys(out))


def funds_by_symbol(build: dict | None) -> dict[str, dict]:
    return {str(f.get("symbol") or "").upper(): f for f in (build or {}).get("funds") or []}


def candidate_row(sym: str, *, funds: dict[str, dict], tech_row: dict | None) -> dict | None:
    """One ETF candidate row: what it holds, what it costs, how it has done and how deep it fell,
    plus a few trend readings. None when the fund has no price (the analyst cannot size it).
    `funds` is funds_by_symbol(build)."""
    fund = funds.get(sym.upper())
    if not tech_row or not tech_row.get("price"):
        return None
    fee = (fund or {}).get("expense_ratio_pct")
    g = fund_cost.group_of(sym)
    row = {
        "symbol": sym,
        "name": (fund or {}).get("name"),
        "type": dict(fund_catalog.CATEGORIES).get((fund or {}).get("category"), None),
        "exposure_group": group_of(sym),
        "price": tech_row["price"],
        # Unknown stays None — never 0, which would read as a free fund.
        "fee_pct": fee,
        "fee_per_10k_usd": round(fee * 100.0, 2) if fee is not None else None,
        "return_pct": (fund or {}).get("returns"),
        "worst_drop_pct": (fund or {}).get("drops"),
    }
    if g:
        row["same_index_as"] = [m for m in g.members if m != sym and m in universe_set()]
    tech = {k: v for k, v in (tech_row.get("technicals") or {}).items() if k in _ROW_TECH_KEYS}
    if tech:
        row["technicals"] = tech
    lt = {k: v for k, v in (tech_row.get("long_term") or {}).items() if k in _ROW_LONG_KEYS}
    if lt:
        row["long_term"] = lt
    return row


def group_summary(build: dict | None, *, settings: dict, exclude: Iterable[str] = (),
                  allow_crypto_etf: bool = True) -> dict[str, dict]:
    """Per cap group, for the weekly plan: its cap, the funds in it (cheapest first, with fees) and
    how its cheapest fund has done. The strategist plans in groups, so it needs each group's cost,
    return and worst drop without reading ~170 fund rows."""
    fees = fee_table(build)
    funds = {f["symbol"]: f for f in (build or {}).get("funds") or [] if f.get("symbol")}
    banned = {str(x).upper() for x in exclude}
    caps = cap_pct_of(settings)
    by_group: dict[str, list[str]] = {}
    for s in universe():
        if s in banned:
            continue
        g = group_of(s)
        if not allow_crypto_etf and g in ("BTC", "ETH"):
            continue
        by_group.setdefault(g, []).append(s)
    out: dict[str, dict] = {}
    for g, members in by_group.items():
        ranked = sorted(members, key=lambda m: (
            fees.get(m, (None, None))[0] is None, fees.get(m, (None, None))[0] or 0.0,
            -float(fees.get(m, (None, None))[1] or 0.0), m))
        lead = funds.get(ranked[0]) or {}
        out[g] = {
            "cap_pct": caps(g),
            "broad": g in BROAD_GROUPS,
            "names": sorted({(funds.get(m) or {}).get("name") for m in members} - {None}),
            "funds": [{"symbol": m, "fee_pct": fees.get(m, (None, None))[0]} for m in ranked[:6]],
            "measured_on": ranked[0],
            "return_pct": lead.get("returns"),
            "worst_drop_pct": lead.get("drops"),
        }
    return out


def history_as_of(build: dict | None) -> str | None:
    return (build or {}).get("aligned_to")
