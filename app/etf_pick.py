"""The ETF pick: at most one fund a trading day, beside the stock pick, on its own thesis.

Until 2026-10-21 this card was "a sound, low-cost fund at a better price": a fund above its 200-day
that had pulled back 5-25% from its high. Measured over its first 19 live runs it never produced a
pick, and the reason was structural: the screen kept only the funds LAGGING the market (3-month
relative strength -5 to -9 points) while the analyst's base rule weights relative strength and
momentum most, so every candidate arrived with its best-evidenced factor against it. The momentum
leaders of the catalogue (health care, energy, value, the Nasdaq-100) were rejected as "not on sale".

The thesis now is the one the evidence supports at a 3-12 month horizon: **the sound, low-cost fund
with the strongest momentum** — up over the last 6 months, above its 200-day, ranked by its 6- and
12-month gain and its 6-month gain against the S&P 500, then by how calm it has been and what it
costs. A fund that has pulled back a little from its high while its longer gains stay intact is a
better entry on a leader, never a reason to wait (entry-timing-measured showed waiting for dips
loses on this universe).

PURE, like daily_pick.py: no clients, no files, no model. main.py fetches the fund catalogue and the
price histories, calls `shortlist()`, and hands the finalists to the same analyst and the same
`daily_pick.reconcile()` the stock pick uses.

THE SCREEN (hard filters, each one a reject reason):
  * an ETF on the catalogue (etf_arm.universe), one row per index — the cheapest copy
  * not crypto, not T-bills or short Treasuries (crypto swings 30% as a matter of course; cash
    funds never have momentum)
  * a known fee at or under MAX_FEE_PCT — an unknown fee is not a low one
  * at least three years of history, so the 3-year worst drop is real
  * above its 200-day average: the long trend is intact
  * up over the last 6 months: a fund that is falling has no momentum to buy

THE SCORE (stated, weights over 0-100 components, higher = more attractive):
  * gain_6m   0.40 — 6-month gain, full credit at GAIN_6M_FULL_PCT
  * gain_12m  0.20 — 12-month gain (from the catalogue's aligned returns), full at GAIN_12M_FULL_PCT
  * rel_6m    0.15 — 6-month gain minus the S&P 500's, full at +REL_6M_FULL_PP, zero at -REL_6M_FULL_PP
  * calm      0.15 — its 3-year worst drop; a fund that fell 15% is easier to hold than one that fell 45%
  * cost      0.10 — its yearly fee

There is deliberately no "health" term. The old one peaked 6% above the 200-day and scored ZERO at
20% above, which punished exactly the strong uptrend this card now looks for.

ABSENT IS NOT ZERO, as everywhere here: a component that could not be measured rejects the fund
rather than scoring it as the worst in the list. The one exception is the 12-month gain, which a
fund listed under a year ago cannot have; it scores 0 on that part and the card says so.
"""
from __future__ import annotations

from typing import Iterable

from . import etf_arm

MAX_FEE_PCT = 0.60
SHORTLIST_N = 8
GAIN_6M_FULL_PCT = 25.0     # +25% in 6 months earns full credit on gain_6m
GAIN_12M_FULL_PCT = 40.0    # +40% in 12 months earns full credit on gain_12m
REL_6M_FULL_PP = 10.0       # 10 points ahead of the S&P over 6 months is full credit; 10 behind is 0
MOM_BARS_6M = 126           # ~6 months of sessions
MOM_BARS_3M = 63
WEIGHTS = {"gain_6m": 0.40, "gain_12m": 0.20, "rel_6m": 0.15, "calm": 0.15, "cost": 0.10}

# Groups this card never offers. Crypto funds swing 30% as a matter of course; T-bill and
# short-Treasury funds are cash and never have momentum worth the name.
EXCLUDED_GROUPS = frozenset({"BTC", "ETH", "T_BILLS", "SHORT_TREASURY"})

REJECT_EXCLUDED = "excluded_type"
REJECT_FEE = "fee_unknown_or_high"
REJECT_HISTORY = "under_3_years"
REJECT_UNMEASURED = "unmeasured"
REJECT_BELOW_200 = "below_200_day"
REJECT_NO_MOMENTUM = "not_up_6_months"


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
        "mom_126d": _ret(xs, MOM_BARS_6M),
        "rsi14": rsi14,
    }
    if bench_closes:
        b = [float(x) for x in bench_closes if x is not None and x > 0]
        f63, b63 = _ret(xs, MOM_BARS_3M), _ret(b, MOM_BARS_3M)
        if f63 is not None and b63 is not None:
            row["rel_strength_3mo"] = round(f63 - b63, 2)
        f126, b126 = row["mom_126d"], _ret(b, MOM_BARS_6M)
        if f126 is not None and b126 is not None:
            row["rel_strength_6mo"] = round(f126 - b126, 2)
            row["bench_mom_126d"] = round(b126, 2)
    return row


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _num(x) -> float | None:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


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
    if not row or row.get("mom_126d") is None or row.get("pct_vs_sma200") is None:
        return REJECT_UNMEASURED
    if not row.get("above_sma200"):
        return REJECT_BELOW_200
    if float(row["mom_126d"]) <= 0.0:
        return REJECT_NO_MOMENTUM
    return None


def score(fund: dict, row: dict) -> dict:
    """The stated score for an eligible fund, with its parts, all 0-100."""
    g6 = float(row["mom_126d"])
    g12 = _num((fund.get("returns") or {}).get("1y"))
    rel6 = _num(row.get("rel_strength_6mo"))
    drop3 = float((fund.get("drops") or {})["3y"])
    fee = float(fund["expense_ratio_pct"])
    parts = {
        "gain_6m": _clamp(g6 / GAIN_6M_FULL_PCT * 100.0),
        # A fund under a year old has no 12-month gain; it scores 0 here and the factor row is absent.
        "gain_12m": _clamp(g12 / GAIN_12M_FULL_PCT * 100.0) if g12 is not None else 0.0,
        # Level with the S&P scores 50; 10 points ahead 100; 10 behind 0. Unmeasured (no benchmark
        # bars this morning) is treated as level, and the factor row is absent so the card never
        # claims a comparison it did not make.
        "rel_6m": _clamp((rel6 + REL_6M_FULL_PP) / (2.0 * REL_6M_FULL_PP) * 100.0) if rel6 is not None else 50.0,
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


def _signed(x: float) -> str:
    return f"{x:+.1f}"


def fund_factors(fund: dict | None, row: dict | None = None) -> dict[str, dict]:
    """The factors only a fund has: its 6- and 12-month gain, what it costs and how far it has fallen
    before. Same shape as daily_pick.factors_for's rows; absent when unmeasured."""
    from .daily_pick import _factor   # the one factor shape, not a copy of it
    out: dict[str, dict] = {}
    g6 = _num((row or {}).get("mom_126d"))
    if g6 is not None:
        b6 = _num((row or {}).get("bench_mom_126d"))
        tail = f", vs {_signed(b6)}% for the S&P 500" if b6 is not None else ""
        out["gain_6m"] = _factor("gain_6m", f"{_signed(g6)}% over the last 6 months{tail}",
                                 value=round(g6, 2), unit="pct")
    g12 = _num(((fund or {}).get("returns") or {}).get("1y"))
    if g12 is not None:
        out["gain_12m"] = _factor("gain_12m", f"{_signed(g12)}% over the last 12 months",
                                  value=round(g12, 2), unit="pct")
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
    top = max(r.values()) if r else 0
    if r.get(REJECT_NO_MOMENTUM, 0) and r[REJECT_NO_MOMENTUM] >= top:
        return "No sound fund is up over the last 6 months today"
    if r.get(REJECT_BELOW_200, 0) and r[REJECT_BELOW_200] >= top:
        return "No sound fund is above its 200-day average today"
    return "No fund passed the screen: sound, low-cost, above its 200-day and up over 6 months"
