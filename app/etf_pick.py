"""The ETF pick: at most one fund a trading day, beside the stock pick, on a different thesis.

The stock pick ranks by relative strength and trend, the best-evidenced factors for single names.
Applied to funds that ranking crowns whichever narrow theme ran hardest lately (Yahoo's own top-ETF
list on 2026-08-14 was gold miners, metals miners, esports). So this card states a different
thesis, the one the user chose on 2026-09-28: **a sound, low-cost fund at a better price** — a fund
worth owning that has pulled back from its high while its long trend is intact.

It is still "a better price on something worth owning", never "wait for a lower one":
entry-timing-measured showed waiting for dips loses on this universe, so the card does not tell
anyone to hold off. It only chooses, among funds that are down a little, the one most worth adding.

PURE, like daily_pick.py: no clients, no files, no model. main.py fetches the fund catalogue and the
price histories, calls `shortlist()`, and hands the finalists to the same analyst and the same
`daily_pick.reconcile()` the stock pick uses.

THE SCREEN (hard filters, each one a reject reason):
  * an ETF on the catalogue (etf_arm.universe), one row per index — the cheapest copy
  * not crypto, not T-bills or short Treasuries (those never "pull back"; bitcoin always does)
  * a known fee at or under MAX_FEE_PCT — an unknown fee is not a low one
  * at least three years of history, so the 3-year worst drop is real
  * above its 200-day average: the long trend is intact
  * MIN_PULLBACK_PCT to MAX_PULLBACK_PCT below its 52-week high: on sale, not broken

THE SCORE (stated, weights over 0-100 components, higher = more attractive):
  * discount 0.40 — how far below its high, peaking at DISCOUNT_FULL_PCT
  * health 0.25 — how comfortably above its 200-day (just above is fragile, far above is stretched)
  * calm 0.20 — its 3-year worst drop; a fund that fell 15% is easier to hold than one that fell 45%
  * cost 0.15 — its yearly fee

ABSENT IS NOT ZERO, as everywhere here: a component that could not be measured rejects the fund
rather than scoring it as the worst in the list.
"""
from __future__ import annotations

from typing import Iterable

from . import etf_arm

MIN_PULLBACK_PCT = 5.0
MAX_PULLBACK_PCT = 25.0
DISCOUNT_FULL_PCT = 15.0
MAX_FEE_PCT = 0.60
SHORTLIST_N = 8

WEIGHTS = {"discount": 0.40, "health": 0.25, "calm": 0.20, "cost": 0.15}

# Groups this card never offers. Crypto funds swing 30% as a matter of course, so "pulled back" says
# nothing about them; T-bill and short-Treasury funds barely move, so they are never on sale.
EXCLUDED_GROUPS = frozenset({"BTC", "ETH", "T_BILLS", "SHORT_TREASURY"})

REJECT_EXCLUDED = "excluded_type"
REJECT_FEE = "fee_unknown_or_high"
REJECT_HISTORY = "under_3_years"
REJECT_UNMEASURED = "unmeasured"
REJECT_BELOW_200 = "below_200_day"
REJECT_NOT_DOWN = "not_on_sale"
REJECT_BROKEN = "fell_too_far"


def _sma(xs: list[float], n: int) -> float | None:
    return sum(xs[-n:]) / n if len(xs) >= n else None


def _ret(xs: list[float], n: int) -> float | None:
    return (xs[-1] / xs[-1 - n] - 1.0) * 100.0 if len(xs) > n and xs[-1 - n] > 0 else None


def tech_row(closes: list[float], *, bench_closes: list[float] | None = None,
             rsi14: float | None = None, high_52w: float | None = None) -> dict | None:
    """A scan-row-shaped reading of one fund from its daily closes, so daily_pick.factors_for can
    describe it exactly as it describes a stock. None when there are not 200 days to read."""
    xs = [float(x) for x in closes if x is not None and x > 0]
    if len(xs) < 200:
        return None
    px = xs[-1]
    s50, s150, s200 = _sma(xs, 50), _sma(xs, 150), _sma(xs, 200)
    hi = high_52w if high_52w and high_52w > 0 else max(xs[-252:])
    row = {
        "price": px,
        "above_sma50": px > s50,
        "above_sma200": px > s200,
        "ma_stacked": px > s50 > s150 > s200,
        "pct_vs_sma50": round((px / s50 - 1.0) * 100.0, 2),
        "pct_vs_sma200": round((px / s200 - 1.0) * 100.0, 2),
        "pct_off_52w_high": round(min(0.0, (px / hi - 1.0) * 100.0), 2),
        "mom_60d": _ret(xs, 60),
        "rsi14": rsi14,
    }
    if bench_closes:
        b = [float(x) for x in bench_closes if x is not None and x > 0]
        f63, b63 = _ret(xs, 63), _ret(b, 63)
        if f63 is not None and b63 is not None:
            row["rel_strength_3mo"] = round(f63 - b63, 2)
    return row


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def reject_reason(fund: dict | None, row: dict | None) -> str | None:
    """The first hard filter a fund fails, or None when it is eligible. `fund` is its catalogue row."""
    sym = str((fund or {}).get("symbol") or "").upper()
    if not sym or etf_arm.group_of(sym) in EXCLUDED_GROUPS or (fund or {}).get("category") == "crypto":
        return REJECT_EXCLUDED
    fee = (fund or {}).get("expense_ratio_pct")
    if fee is None or fee > MAX_FEE_PCT:
        return REJECT_FEE
    drops = (fund or {}).get("drops") or {}
    if drops.get("3y") is None:
        return REJECT_HISTORY
    if not row or row.get("pct_off_52w_high") is None or row.get("pct_vs_sma200") is None:
        return REJECT_UNMEASURED
    if not row.get("above_sma200"):
        return REJECT_BELOW_200
    off = -float(row["pct_off_52w_high"])
    if off < MIN_PULLBACK_PCT:
        return REJECT_NOT_DOWN
    if off > MAX_PULLBACK_PCT:
        return REJECT_BROKEN
    return None


def score(fund: dict, row: dict) -> dict:
    """The stated score for an eligible fund, with its parts, all 0-100."""
    off = -float(row["pct_off_52w_high"])
    above = float(row["pct_vs_sma200"])
    drop3 = float((fund.get("drops") or {})["3y"])
    fee = float(fund["expense_ratio_pct"])
    parts = {
        "discount": _clamp((off - MIN_PULLBACK_PCT) / (DISCOUNT_FULL_PCT - MIN_PULLBACK_PCT) * 100.0),
        # Best from 2% to 10% above the 200-day; thin above it is fragile, far above it is stretched.
        "health": _clamp(100.0 - abs(above - 6.0) * (100.0 / 14.0)) if above >= 0 else 0.0,
        # -10% worst drop scores 100, -50% scores 0.
        "calm": _clamp((50.0 + drop3) / 40.0 * 100.0),
        "cost": _clamp((1.0 - fee / MAX_FEE_PCT) * 100.0),
    }
    total = sum(WEIGHTS[k] * v for k, v in parts.items())
    return {"score": round(total, 2), "parts": {k: round(v, 1) for k, v in parts.items()}}


def shortlist(build: dict | None, rows: dict[str, dict | None], *, pool: Iterable[str],
              limit: int = SHORTLIST_N) -> dict:
    """Filter + rank the pool. Returns {ranked, rejects, scanned, eligible}, like daily_pick.shortlist.

    `rows` maps symbol -> tech_row (None when it could not be read); `pool` is one symbol per index
    (etf_arm.pool_symbols). Ties are broken on the symbol so two runs over the same data agree."""
    funds = etf_arm.funds_by_symbol(build)
    rejects: dict[str, int] = {}
    scored: list[dict] = []
    scanned = 0
    for sym in pool:
        scanned += 1
        fund, row = funds.get(sym), rows.get(sym)
        why = reject_reason(fund, row)
        if why is not None:
            rejects[why] = rejects.get(why, 0) + 1
            continue
        s = score(fund, row)
        scored.append({"symbol": sym, **s, "row": row, "fund": fund,
                       "penalty_reasons": [], "partial": False})
    scored.sort(key=lambda c: (-c["score"], c["symbol"]))
    # One fund per group of funds that move together (etf_arm.ETF_GROUPS). The pool already holds one
    # per INDEX, but XLF and FNCL are different indexes that move at 0.994, and on the first live run
    # (2026-09-28) they took two of the eight slots — as did SCHA and SPSM. The best-scoring member
    # of each group stays; the analyst chooses between different things, not copies.
    seen: set[str] = set()
    distinct: list[dict] = []
    for c in scored:
        g = etf_arm.group_of(c["symbol"])
        if g in seen:
            continue
        seen.add(g)
        distinct.append(c)
    return {"ranked": distinct[:max(1, int(limit))], "rejects": rejects, "scanned": scanned,
            "eligible": len(scored)}


def fund_factors(fund: dict | None) -> dict[str, dict]:
    """The two factors only a fund has: what it costs and how far it has fallen before. Same shape as
    daily_pick.factors_for's rows; absent when unmeasured."""
    from .daily_pick import _factor   # the one factor shape, not a copy of it
    out: dict[str, dict] = {}
    fee = (fund or {}).get("expense_ratio_pct")
    if fee is not None:
        out["fee"] = _factor("fee", f"costs {fee:.2f}% a year (${fee * 100:.0f} per $10,000)",
                             value=float(fee), unit="pct")
    drops = (fund or {}).get("drops") or {}
    d3 = drops.get("3y")
    if d3 is not None:
        d5 = drops.get("5y")
        tail = f"; {abs(d5):.0f}% in 5 years" if d5 is not None else ""
        out["worst_drop"] = _factor("worst_drop", f"fell at most {abs(d3):.0f}% in 3 years{tail}",
                                    value=float(d3), unit="pct")
    return out


def none_reason(sl: dict) -> str:
    """Why no fund made the list, in plain words, from the reject counts."""
    r = sl.get("rejects") or {}
    if not sl.get("scanned"):
        return "The fund list could not be read this morning"
    not_down = r.get(REJECT_NOT_DOWN, 0)
    if not_down and not_down >= max(r.values()):
        return "No sound fund is 5% or more off its high today"
    return "No fund passed the screen: sound, low-cost, above its 200-day and 5-25% off its high"
