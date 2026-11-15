"""Long-calls-and-puts sandbox arms: a contract ledger beside the share ledger, chain pricing, the
rule exits, sizing and fills, the no-model twin's decision, and the per-tick chain context.

WHAT THESE ARMS ARE FOR. Two arms, `options` (the analyst picks underlying, side and conviction)
and `options-rules` (no model: the directional score the calls card already computes, fed into the
same contract picker), buying long calls and long puts only. Both read ONE chain snapshot built after
main persists, so the gap between their curves is the AI and nothing else. The question they answer
is in dollars: does a dollar of premium placed on a directional read earn more than the same dollar
left in the S&P over the same window, after the spread and time decay? Loss on any contract is
capped at the premium paid. No short options, no spreads, no assignment, no real orders.

WHY A SEPARATE LEDGER. The share ledger (`blob["positions"]`, sandbox_job.validate_and_fill, the
state and arms endpoints) sizes by shares and marks by a quote per symbol. A contract is a hundred
shares of something else, priced by its own bid and ask, and it expires. Teaching the share code
about multipliers and expiries would put a special case in every reader; instead the contracts live
under `blob["options"]["contracts"]`, `positions` stays empty on these arms, and every trade row
carries its own side values (`buy_to_open`, `sell_to_close`, `expired`, `settled`), so a reader that
filters on `buy`/`sell` skips them by construction rather than by remembering to.

WHAT IS CODE AND WHAT IS THE MODEL. The model (or the directional score) chooses an underlying and a
side. Code chooses the contract (the expiry window and the delta the calls card already uses, 0.50
if it fits the per-contract cap, else 0.32), sizes it, fills it at the ASK, marks it at the mid,
closes it at the BID on the rule exits (days to expiry, stop, target), and cash-settles anything
that reaches expiry at intrinsic value against the underlying's close that day. A fill at mid would
flatter every round trip by the whole spread, which on these contracts is bigger than any edge
plausibly being measured.

UNKNOWN IS NOT CALM. No chain today means no marks, no exits and no entries, with every held
contract carried at its last mark and said so. A stock with an unknown earnings date (the lookup
failed, as opposed to found nothing) gets no new contract. IV rank is shown when the history can
support it and shown as unknown otherwise; it unlocks nothing either way, because main's pool is a
daily market screen and most names will have no rank for months.
"""
from __future__ import annotations

import asyncio
import copy
import datetime as dt
import math
import time
from typing import Awaitable, Callable, Iterable

from . import options
from .sandbox_job import today_et_str

UNIVERSE_OPTIONS = "options"
MULTIPLIER = 100
# Always offered alongside the top directional reads: the natural vehicles for an index view, and the
# only puts in the measured set that stay liquid at every strike.
INDEX_ETFS = ("SPY", "QQQ", "IWM")
MAX_UNDERLYINGS = 20
MIN_UNDERLYING_PRICE = 5.0
DELTA_FIRST = 0.50
DELTA_FALLBACK = 0.32
MIN_DTE_OPEN = 30
# A flip (call to put or back) on one name within this many days needs a second run to confirm.
FLIP_WINDOW_DAYS = 30
MARKET_OPEN_STATE = "REGULAR"

SIDE_OPEN = "buy_to_open"
SIDE_CLOSE = "sell_to_close"
SIDE_EXPIRED = "expired"       # reached expiry out of the money: proceeds 0
SIDE_SETTLED = "settled"       # reached expiry in the money: cash-settled at intrinsic value
CLOSING_SIDES = frozenset({SIDE_CLOSE, SIDE_EXPIRED, SIDE_SETTLED})
SIDES = frozenset({SIDE_OPEN, *CLOSING_SIDES})

# Settings these arms read, with their defaults. They ride in the arm's `settings` like every other
# limit, so the twins can be told apart by a diff of two settings blocks.
DEFAULTS: dict[str, float] = {
    "opt_max_premium_pct": 15.0,     # premium on one contract position, % of equity
    "opt_max_at_risk_pct": 30.0,     # all open premium together, % of equity
    "opt_max_open": 6,               # open contract positions
    "opt_max_new_per_tick": 2,
    "opt_exit_dte": 21,              # close at or under this many calendar days to expiry
    "opt_stop_pct": -50.0,           # close when the mark is this far under the premium paid
    "opt_target_pct": 100.0,         # close when the mark is this far over the premium paid
    "opt_max_spread_pct": 10.0,      # (ask - bid) / mid, percent
    "opt_min_oi": 250,
    "opt_pause_drawdown_pct": 10.0,  # a drawdown this deep pauses new entries ...
    "opt_pause_ticks": 5,            # ... for this many runs
}


def setting(settings: dict | None, key: str) -> float:
    v = (settings or {}).get(key)
    return float(DEFAULTS[key] if v is None else v)


def is_options_arm(settings: dict | None) -> bool:
    return str((settings or {}).get("universe") or "").lower() == UNIVERSE_OPTIONS


def state(blob: dict) -> dict:
    """The arm's options block, created on first use so an arm switched to this universe has one."""
    st = blob.get("options")
    if not isinstance(st, dict):
        st = {}
        blob["options"] = st
    st.setdefault("contracts", [])
    st.setdefault("signals", {})          # underlying -> {"right", "date"}: the last proposed direction
    st.setdefault("pause_ticks_left", 0)
    st.setdefault("marks_as_of", None)    # ET date of the last fresh marks
    return st


def contracts_of(blob: dict) -> list[dict]:
    return list((blob.get("options") or {}).get("contracts") or [])


# ------------------------------------------------------------------ contract identity and display

def occ_root(contract_symbol: str) -> str:
    """The underlying root of an OCC symbol: everything before the 15-character date/right/strike tail."""
    s = (contract_symbol or "").upper()
    return s[:-15] if len(s) > 15 else ""


def is_standard(contract_symbol: str, underlying: str) -> bool:
    """Standard contracts only: an adjusted contract (after a split, a spin-off, a special dividend)
    carries a different root — `SPY1` beside `SPY` — and delivers something other than 100 shares, so
    its quote is not comparable and its intrinsic value is not strike minus spot."""
    return occ_root(contract_symbol) == (underlying or "").upper()


def describe(c: dict) -> str:
    strike = c.get("strike")
    s = f"{strike:g}" if isinstance(strike, (int, float)) else str(strike)
    return f"{str(c.get('underlying') or '').upper()} ${s} {c.get('right')} exp {c.get('expiry')}"


def _days_between(a: str | None, b: str | None) -> int | None:
    try:
        return (dt.date.fromisoformat(str(b)) - dt.date.fromisoformat(str(a))).days
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ marks and valuation

def quote_from(c: options.OptionContract, *, underlying: str, spot: float | None,
               market_state: str | None, dte: float | None) -> dict:
    """A plain dict quote for one contract, from an annotated OptionContract. Everything a fill, a
    mark and the analyst's row need, and nothing that is not serialisable."""
    mid = options.mid_price(c.bid, c.ask)
    sp = options.spread_pct(c.bid, c.ask)
    return {
        "contract_symbol": c.contract_symbol, "underlying": underlying.upper(), "right": c.type,
        "strike": c.strike, "expiry_ts": int(c.expiration),
        "expiry": dt.datetime.fromtimestamp(int(c.expiration), dt.timezone.utc).date().isoformat(),
        "dte": round(dte, 1) if dte is not None else None,
        "bid": c.bid, "ask": c.ask, "mid": round(mid, 4) if mid is not None else None,
        "last_price": c.last_price, "last_trade_epoch": c.last_trade_epoch,
        "iv": c.implied_volatility, "delta": c.delta, "theta": c.theta,
        "open_interest": c.open_interest, "volume": c.volume,
        "spread_pct": round(sp * 100.0, 2) if sp is not None else None,
        "breakeven": round(c.breakeven, 2) if c.breakeven is not None else None,
        "cost_one": round(float(c.ask) * MULTIPLIER, 2) if c.ask else None,
        "spot": spot, "market_state": market_state,
    }


def fresh_mark(q: dict | None) -> float | None:
    """The mark a live two-sided quote supports: the mid. A one-sided quote (0 bid) supports none —
    half of one number is not a price, and marking there wrote invented drawdowns before."""
    if not q:
        return None
    m = q.get("mid")
    return float(m) if isinstance(m, (int, float)) and m > 0 else None


def mark_of(c: dict, marks: dict[str, dict]) -> tuple[float | None, bool]:
    """(mark per share, stale). Fresh mid first; else the last stored mark; else the premium paid.
    Never zero: a contract that cannot be priced today is an approximation, not a wipe-out."""
    m = fresh_mark(marks.get(c["contract_symbol"]))
    if m is not None:
        return m, False
    for key in ("last_mark", "avg_premium"):
        v = c.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return float(v), True
    return None, True


def apply_marks(contracts: list[dict], marks: dict[str, dict], *, today: str) -> list[str]:
    """Store today's fresh marks on the contracts. Returns the contracts valued from a stale mark."""
    stale: list[str] = []
    for c in contracts:
        m = fresh_mark(marks.get(c["contract_symbol"]))
        if m is not None:
            c["last_mark"] = round(m, 4)
            c["last_mark_date"] = today
            q = marks[c["contract_symbol"]]
            if q.get("spot"):
                c["last_spot"] = q["spot"]
        else:
            stale.append(c["contract_symbol"])
    return stale


def contracts_value(contracts: Iterable[dict], marks: dict[str, dict]) -> float:
    total = 0.0
    for c in contracts:
        m, _ = mark_of(c, marks)
        if m is not None:
            total += m * MULTIPLIER * int(c.get("count") or 0)
    return round(total, 2)


def premium_at_risk(contracts: Iterable[dict]) -> float:
    return round(sum(float(c.get("cost_total") or 0.0) for c in contracts), 2)


# ------------------------------------------------------------------ picking a contract

def quote_gate(q: dict, *, settings: dict | None, underlying: str) -> str | None:
    """Why this contract may not be bought today, or None. Liquidity and identity only; sizing is
    separate so a skip row can say which one bound."""
    if not is_standard(q.get("contract_symbol") or "", underlying):
        return "non-standard contract (adjusted root)"
    bid, ask = q.get("bid"), q.get("ask")
    if not bid or bid <= 0 or not ask or ask <= 0:
        return "no two-sided quote"
    oi = q.get("open_interest")
    if oi is None or oi < setting(settings, "opt_min_oi"):
        return f"open interest {oi if oi is not None else 'unknown'} under {int(setting(settings, 'opt_min_oi'))}"
    sp = q.get("spread_pct")
    if sp is None or sp > setting(settings, "opt_max_spread_pct"):
        return f"spread {sp if sp is not None else 'unknown'}% over {setting(settings, 'opt_max_spread_pct'):g}%"
    if q.get("dte") is not None and q["dte"] < MIN_DTE_OPEN:
        return f"{q['dte']:g} days to expiry, under {MIN_DTE_OPEN}"
    return None


def nearest_by_delta(quotes: Iterable[dict], target: float) -> dict | None:
    """The quote whose |delta| is nearest `target`, among those with a delta. Puts carry a negative
    delta, so the absolute value is what is compared."""
    with_delta = [q for q in quotes if isinstance(q.get("delta"), (int, float))]
    if not with_delta:
        return None
    return min(with_delta, key=lambda q: abs(abs(float(q["delta"])) - target))


def pick_contracts(quotes: Iterable[dict], *, settings: dict | None, underlying: str) -> dict:
    """{"atm": quote|None, "otm": quote|None}: the 0.50- and 0.32-delta contracts that pass the
    liquidity gate, from one side of one expiry. A contract that fails the gate is not offered at
    all, so neither the model nor the sizing can pick an illiquid one."""
    ok = [q for q in quotes if quote_gate(q, settings=settings, underlying=underlying) is None]
    atm = nearest_by_delta(ok, DELTA_FIRST)
    otm = nearest_by_delta(ok, DELTA_FALLBACK)
    if otm is not None and atm is not None and otm["contract_symbol"] == atm["contract_symbol"]:
        otm = None
    return {"atm": atm, "otm": otm}


# ------------------------------------------------------------------ exits (code, never the model)

def rule_exits(contracts: Iterable[dict], marks: dict[str, dict], *, now_ts: float,
               settings: dict | None) -> list[dict]:
    """Close orders the rules demand today: days to expiry, stop, target. Judged on a FRESH mark
    only — a stale mark cannot trigger a stop, but the days-to-expiry exit needs no mark at all and
    fires regardless, because expiry does not wait for a quote."""
    out: list[dict] = []
    exit_dte = setting(settings, "opt_exit_dte")
    stop, target = setting(settings, "opt_stop_pct"), setting(settings, "opt_target_pct")
    for c in contracts:
        dte = (options.expiry_epoch(int(c["expiry_ts"])) - now_ts) / 86_400.0
        base = dict(underlying=c["underlying"], symbol=c["underlying"], right=c["right"],
                    action="close", contract_symbol=c["contract_symbol"], conviction=None)
        if dte <= exit_dte:
            out.append({**base, "rule": "dte",
                        "reason": f"{max(dte, 0):.0f} days to expiry — time-decay exit at {exit_dte:g}"})
            continue
        m = fresh_mark(marks.get(c["contract_symbol"]))
        paid = float(c.get("avg_premium") or 0.0)
        if m is None or paid <= 0:
            continue
        pct = (m / paid - 1.0) * 100.0
        if pct <= stop:
            out.append({**base, "rule": "stop",
                        "reason": f"down {abs(pct):.0f}% on the premium — stop at {abs(stop):g}%"})
        elif pct >= target:
            out.append({**base, "rule": "target",
                        "reason": f"up {pct:.0f}% on the premium — target at {target:g}%"})
    return out


def expired(contracts: Iterable[dict], *, now_ts: float) -> list[dict]:
    """Contracts past their 16:00 ET expiry. Normally none: the days-to-expiry exit closes a
    contract three weeks before. This is the path for a service that was down past an expiry."""
    return [c for c in contracts if options.expiry_epoch(int(c["expiry_ts"])) <= now_ts]


def intrinsic(right: str, strike: float, close: float) -> float:
    return max(0.0, close - strike) if right == "call" else max(0.0, strike - close)


def settle_expired(blob: dict, closes: dict[str, float | None], *, now_ts: float,
                   spy_price: float | None) -> list[dict]:
    """Cash-settle every expired contract whose underlying close on the expiry date is known, at
    intrinsic value. One whose close is unknown is HELD and said so — settling at zero would invent
    a total loss on a contract that may have finished in the money. Mutates the blob; returns rows."""
    st = state(blob)
    rows: list[dict] = []
    keep: list[dict] = []
    today = today_et_str()
    for c in st["contracts"]:
        if options.expiry_epoch(int(c["expiry_ts"])) > now_ts:
            keep.append(c)
            continue
        close = closes.get(c["contract_symbol"])
        if close is None or close <= 0:
            keep.append(c)
            rows.append(_skip_row(c["underlying"], c["right"], action="close", now_ts=now_ts, today=today,
                                  reason=f"expired {c['expiry']} — awaiting the underlying's close on that day",
                                  contract=c, source="expiry"))
            continue
        value = intrinsic(c["right"], float(c["strike"]), float(close))
        proceeds = round(value * MULTIPLIER * int(c["count"]), 2)
        rows.append(_close_row(blob, c, price=round(value, 4), proceeds=proceeds,
                               side=SIDE_SETTLED if value > 0 else SIDE_EXPIRED,
                               spot_now=float(close), spy_price=spy_price, now_ts=now_ts, today=today,
                               source="expiry", rule="expiry",
                               reason=(f"expired in the money — settled at ${value:.2f} intrinsic against the "
                                       f"{c['expiry']} close of ${close:.2f}" if value > 0 else
                                       f"expired out of the money against the {c['expiry']} close of ${close:.2f}")))
    st["contracts"] = keep
    return rows


# ------------------------------------------------------------------ sizing

def size_open(cand: dict, right: str, *, equity: float, cash: float, at_risk: float,
              conviction: int | None, settings: dict | None) -> tuple[dict | None, int, str | None]:
    """(quote, count, skip_reason). The 0.50-delta contract if one fits the per-contract cap, else the
    0.32, else nothing. Count scales with conviction inside the cap and never breaches the at-risk
    cap or the cash floor; a single contract that fits the cap but not the remaining room is refused
    with the room named, because "unaffordable" and "no room left" are different problems."""
    cap = setting(settings, "opt_max_premium_pct") / 100.0 * equity
    room = setting(settings, "opt_max_at_risk_pct") / 100.0 * equity - at_risk
    floor = float((settings or {}).get("cash_floor_pct", 0.0) or 0.0) / 100.0 * equity
    spend = min(cap, room, cash - floor)
    side = cand.get("calls" if right == "call" else "puts") or {}
    if side.get("atm") is None and side.get("otm") is None:
        return None, 0, f"no liquid {right} on {cand.get('symbol')} today"
    if spend <= 0:
        if room <= 0:
            return None, 0, f"premium at risk is at its {setting(settings, 'opt_max_at_risk_pct'):g}% cap"
        return None, 0, f"cash floor of {float((settings or {}).get('cash_floor_pct', 0.0) or 0.0):g}% leaves nothing to spend"
    conv = max(0, min(100, int(conviction if conviction is not None else 100)))
    for key in ("atm", "otm"):
        q = side.get(key)
        if not q or not q.get("cost_one"):
            continue
        one = float(q["cost_one"])
        if one > cap:
            continue
        want = min(cap * conv / 100.0, spend)
        count = int(math.floor(want / one))
        if count < 1 and one <= spend:
            count = 1
        if count >= 1:
            return q, count, None
        return None, 0, (f"one {right} costs ${one:,.0f}; ${spend:,.0f} left under the at-risk cap and cash floor")
    return None, 0, (f"no {right} at 0.50 or 0.32 delta fits the per-contract cap of ${cap:,.0f}")


# ------------------------------------------------------------------ the two-run rule on a flip

def flip_block(signals: dict, underlying: str, right: str, *, today: str) -> str | None:
    """A direction that flipped since the last read on this name waits one run. Only a FLIP waits:
    the first read of a name trades, and the same direction read again trades."""
    prev = (signals or {}).get(underlying.upper())
    if not prev or prev.get("right") == right or prev.get("date") == today:
        return None
    age = _days_between(prev.get("date"), today)
    if age is None or age > FLIP_WINDOW_DAYS:
        return None
    return (f"direction flipped from {prev.get('right')} to {right} since the last read on "
            f"{underlying.upper()} ({prev.get('date')}) — needs a second run to confirm")


def record_signals(signals: dict, orders: Iterable[dict], *, today: str) -> None:
    """Every proposed OPEN is recorded, filled or not: the two-run rule is about what was read, and
    a proposal the cap refused was still a read."""
    for o in orders:
        if str(o.get("action") or "").lower() != "open":
            continue
        und = str(o.get("underlying") or o.get("symbol") or "").upper()
        if und and o.get("right") in ("call", "put"):
            signals[und] = {"right": o["right"], "date": today}


# ------------------------------------------------------------------ rows

def _base_row(c_or_q: dict, *, now_ts: float, today: str, side: str, source: str) -> dict:
    return {
        "ts": now_ts, "date": today, "symbol": str(c_or_q.get("contract_symbol") or "").upper(),
        "display": describe(c_or_q), "instrument": "option", "multiplier": MULTIPLIER,
        "underlying": str(c_or_q.get("underlying") or "").upper(), "right": c_or_q.get("right"),
        "strike": c_or_q.get("strike"), "expiry": c_or_q.get("expiry"),
        "side": side, "source": source,
    }


def _skip_row(underlying: str, right: str, *, action: str, now_ts: float, today: str, reason: str,
              contract: dict | None = None, quote: dict | None = None, source: str,
              conviction: int | None = None, thesis: str | None = None) -> dict:
    ref = contract or quote or {"underlying": underlying, "right": right}
    row = _base_row(ref, now_ts=now_ts, today=today, source=source,
                    side=SIDE_OPEN if action == "open" else SIDE_CLOSE)
    row.update({"status": "skipped", "shares": 0, "price": None, "gross": None,
                "conviction": conviction, "reason": thesis or "", "skip_reason": reason})
    if not (contract or quote):
        row["symbol"] = underlying.upper()
        row["display"] = f"{underlying.upper()} {right}"
    return row


def _close_row(blob: dict, c: dict, *, price: float, proceeds: float, side: str, spot_now: float | None,
               spy_price: float | None, now_ts: float, today: str, source: str, rule: str | None,
               reason: str, conviction: int | None = None) -> dict:
    """Book one close: cash in, realised P/L, and the scoring fields — what the same dollars would
    have done in the index over the same days, and whether the direction was right at all."""
    cost_total = float(c.get("cost_total") or 0.0)
    pl = round(proceeds - cost_total, 2)
    blob["cash"] = round(float(blob.get("cash") or 0.0) + proceeds, 2)
    blob["realized_pl_total"] = round(float(blob.get("realized_pl_total") or 0.0) + pl, 2)
    spot0 = c.get("spot_at_open")
    direction_right = None
    if isinstance(spot0, (int, float)) and spot_now:
        direction_right = (spot_now > spot0) if c["right"] == "call" else (spot_now < spot0)
    spy0 = c.get("spy_at_open")
    index_alt = None
    if isinstance(spy0, (int, float)) and spy0 > 0 and spy_price:
        index_alt = round(cost_total * (float(spy_price) / float(spy0) - 1.0), 2)
    row = _base_row(c, now_ts=now_ts, today=today, side=side, source=source)
    row.update({
        "status": "filled", "shares": int(c["count"]), "price": price, "gross": proceeds,
        "cash_after": blob["cash"], "reason": reason, "rule": rule, "conviction": conviction,
        "cost_total": round(cost_total, 2), "realized_pl": pl,
        "held_days": _days_between(c.get("opened_date"), today),
        "spot_at_open": spot0, "spot_at_close": spot_now, "direction_right": direction_right,
        "spy_at_open": spy0, "spy_at_close": spy_price, "index_alt_usd": index_alt,
        "opened_date": c.get("opened_date"),
    })
    return row


def review_skips(dropped: Iterable[dict], verdict: dict, *, now_ts: float, today: str) -> list[dict]:
    """Rows for the orders the review model refused, in this ledger's own shape. Both cases travel:
    `reason` is the analyst's, `skip_reason` the reviewer's."""
    concerns = [str(c) for c in (verdict.get("concerns") or []) if str(c).strip()]
    note = str(verdict.get("note") or "").strip()
    detail = " · ".join(x for x in ([note] + concerns) if x) or "no reason given"
    verb = "rejected the whole decision" if not verdict.get("approve", True) else "dropped this order"
    return [_skip_row(str(o.get("underlying") or o.get("symbol") or ""), str(o.get("right") or ""),
                      action=str(o.get("action") or "open"), now_ts=now_ts, today=today,
                      reason=f"review model {verb} — {detail}", source="review",
                      conviction=o.get("conviction"), thesis=o.get("reason")) for o in dropped]


# ------------------------------------------------------------------ fill

def fill(blob: dict, orders: list[dict], *, candidates: dict[str, dict], marks: dict[str, dict],
         spy_price: float | None, now_ts: float, source: str, settings: dict | None,
         exclude: set[str] | None = None, today: str | None = None) -> tuple[dict, list[dict], list[dict]]:
    """Apply close and open orders to the contract ledger under the hard limits. Returns
    (new_blob, filled_rows, skipped_rows). Closes first (they free room), at the BID; opens at the
    ASK. The blob is copied, never mutated. Raises AssertionError when cash is not conserved — the
    caller must not persist then.

    `candidates` maps an underlying to today's candidate row (its priced contracts); `marks` maps a
    contract symbol to today's quote. An open with no candidate row, or a close with no bid, is a
    skip with the reason written down, never a fill at a guessed price."""
    b = copy.deepcopy(blob)
    st = state(b)
    today = today or today_et_str()
    exclude = {str(x).upper() for x in (exclude or set())}
    cash0 = round(float(b.get("cash") or 0.0), 2)
    filled: list[dict] = []
    skipped: list[dict] = []
    spent = 0.0
    received = 0.0

    closes = [o for o in orders if str(o.get("action") or "").lower() == "close"]
    opens = [o for o in orders if str(o.get("action") or "").lower() == "open"]

    done: set[str] = set()
    for o in closes:
        und = str(o.get("underlying") or o.get("symbol") or "").upper()
        want = str(o.get("contract_symbol") or "").upper()
        c = next((x for x in st["contracts"]
                  if (want and x["contract_symbol"] == want) or (not want and x["underlying"] == und)), None)
        if c is None or c["contract_symbol"] in done:
            if c is None:
                skipped.append(_skip_row(und, str(o.get("right") or ""), action="close", now_ts=now_ts, today=today,
                                         reason=f"no open contract on {und}", source=source,
                                         thesis=o.get("reason")))
            continue
        q = marks.get(c["contract_symbol"])
        bid = q.get("bid") if q else None
        if not bid or bid <= 0:
            skipped.append(_skip_row(und, c["right"], action="close", now_ts=now_ts, today=today,
                                     reason="no bid today — held, will retry next run", contract=c,
                                     source=source, thesis=o.get("reason")))
            continue
        proceeds = round(float(bid) * MULTIPLIER * int(c["count"]), 2)
        filled.append(_close_row(b, c, price=float(bid), proceeds=proceeds, side=SIDE_CLOSE,
                                 spot_now=q.get("spot") if q else None, spy_price=spy_price, now_ts=now_ts,
                                 today=today, source=source, rule=o.get("rule"), reason=str(o.get("reason") or ""),
                                 conviction=o.get("conviction")))
        received += proceeds
        done.add(c["contract_symbol"])
    st["contracts"] = [c for c in st["contracts"] if c["contract_symbol"] not in done]

    cash = round(float(b["cash"]), 2)
    equity = round(cash + contracts_value(st["contracts"], marks), 2)
    at_risk = premium_at_risk(st["contracts"])
    paused = int(st.get("pause_ticks_left") or 0)
    floor_conv = int(float((settings or {}).get("min_conviction_to_trade", 0) or 0))
    new_count = 0
    opens.sort(key=lambda o: -int(o.get("conviction") or 0))
    for o in opens:
        und = str(o.get("underlying") or o.get("symbol") or "").upper()
        right = str(o.get("right") or "").lower()
        conv = o.get("conviction")
        thesis = str(o.get("reason") or "")

        def skip(reason: str, quote: dict | None = None) -> None:
            skipped.append(_skip_row(und, right, action="open", now_ts=now_ts, today=today, reason=reason,
                                     quote=quote, source=source, conviction=conv, thesis=thesis))

        if right not in ("call", "put"):
            skip(f"unknown side {right!r}")
            continue
        if und in exclude:
            skip("on this arm's exclusion list")
            continue
        if paused > 0:
            skip(f"new entries paused after a drawdown — {paused} run(s) left")
            continue
        if any(c["underlying"] == und for c in st["contracts"]):
            skip(f"already holds a contract on {und} — one per underlying")
            continue
        if len(st["contracts"]) >= int(setting(settings, "opt_max_open")):
            skip(f"{len(st['contracts'])} contracts open — at the {int(setting(settings, 'opt_max_open'))} cap")
            continue
        if new_count >= int(setting(settings, "opt_max_new_per_tick")):
            skip(f"{new_count} new today — at the per-run cap")
            continue
        if conv is not None and int(conv) < floor_conv:
            skip(f"conviction {int(conv)} under the {floor_conv} floor")
            continue
        cand = candidates.get(und)
        if not cand:
            skip(f"no priced contract on {und} today")
            continue
        blocked = entry_block(cand)
        if blocked:
            skip(blocked)
            continue
        why = flip_block(st["signals"], und, right, today=today)
        if why:
            skip(why)
            continue
        q, count, reason = size_open(cand, right, equity=equity, cash=cash, at_risk=at_risk,
                                     conviction=conv, settings=settings)
        if q is None:
            skip(reason or "could not size")
            continue
        cost = round(float(q["ask"]) * MULTIPLIER * count, 2)
        cash = round(cash - cost, 2)
        spent += cost
        at_risk = round(at_risk + cost, 2)
        c = {
            "contract_symbol": q["contract_symbol"], "underlying": und, "right": right,
            "strike": q["strike"], "expiry_ts": q["expiry_ts"], "expiry": q["expiry"],
            "count": count, "avg_premium": float(q["ask"]), "cost_total": cost,
            "opened_at": now_ts, "opened_date": today, "spot_at_open": q.get("spot"),
            "spy_at_open": spy_price, "delta_at_open": q.get("delta"), "iv_at_open": q.get("iv"),
            "conviction": conv, "reason": thesis, "last_mark": q.get("mid"), "last_mark_date": today,
            "last_spot": q.get("spot"), "source": source,
        }
        c["display"] = describe(c)
        st["contracts"].append(c)
        row = _base_row(c, now_ts=now_ts, today=today, side=SIDE_OPEN, source=source)
        row.update({"status": "filled", "shares": count, "price": float(q["ask"]), "gross": cost,
                    "cash_after": cash, "reason": thesis, "conviction": conv, "delta": q.get("delta"),
                    "iv": q.get("iv"), "dte": q.get("dte"), "spot": q.get("spot"), "spy": spy_price,
                    "open_interest": q.get("open_interest"), "spread_pct": q.get("spread_pct")})
        filled.append(row)
        new_count += 1
    b["cash"] = cash
    record_signals(st["signals"], opens, today=today)

    expect = round(cash0 - spent + received, 2)
    assert abs(b["cash"] - expect) < 0.011, (
        f"cash not conserved: start {cash0} - opened {spent:.2f} + closed {received:.2f} = {expect} "
        f"but ledger says {b['cash']}")
    return b, filled, skipped


# ------------------------------------------------------------------ decisions

def directional(technicals: dict | None) -> dict:
    return options.directional_read(technicals or {})


def candidate_rows(all_candidates: Iterable[dict], *, held: Iterable[str], exclusions: Iterable[str]) -> list[dict]:
    """This arm's view of the shared candidates: minus what it holds (one per underlying) and its
    own exclusions. Order preserved."""
    held_set = {str(s).upper() for s in held}
    ex = {str(s).upper() for s in exclusions}
    return [c for c in all_candidates
            if str(c.get("symbol") or "").upper() not in held_set
            and str(c.get("symbol") or "").upper() not in ex]


def entry_block(cand: dict) -> str | None:
    """Why the ledger would refuse ANY new contract on this candidate today, before sizing: quotes
    not live, or a stock whose earnings date is unknown or falls inside the expiry. One place, read
    by the fill and by the rules twin — the twin has two slots a run and must not spend them on
    names the ledger is certain to refuse (the 2026-11-15 dry run did exactly that and filled nothing)."""
    if cand.get("market_state") and cand["market_state"] != MARKET_OPEN_STATE:
        return f"quotes not live ({cand['market_state']}) — no entry outside the regular session"
    earn = cand.get("earnings") or {}
    if not cand.get("is_fund", False):
        if not earn.get("known", False):
            return "earnings date unknown (lookup failed) — no new contract on a stock without it"
        if earn.get("straddles"):
            return f"expiry {cand.get('expiry')} spans earnings on {earn.get('date')}"
    return None


def rules_decision(cands: list[dict], *, settings: dict | None) -> dict:
    """The no-model twin: the directional score reads call at or above 55 and put at or below 45,
    conviction is the score's distance in that direction, and the strongest reads go first. It
    proposes only a side that has a liquid contract today and only names the ledger will not refuse
    outright (entry_block), like the analyst is only shown such."""
    orders: list[dict] = []
    for c in cands:
        if entry_block(c):
            continue
        d = c.get("directional") or directional(c.get("technicals"))
        score = int(d.get("score", 50))
        if score >= options.BULLISH_SCORE:
            right, conv = "call", score
        elif score <= 45:
            right, conv = "put", 100 - score
        else:
            continue
        side = c.get("calls" if right == "call" else "puts") or {}
        if side.get("atm") is None and side.get("otm") is None:
            continue
        reasons = [str(r) for r in (d.get("reasons") or [])][:3]
        orders.append({
            "symbol": c["symbol"], "underlying": c["symbol"], "right": right, "action": "open",
            "conviction": conv,
            "reason": f"directional score {score}" + (f": {'; '.join(reasons)}" if reasons else ""),
        })
    orders.sort(key=lambda o: -o["conviction"])
    orders = orders[:int(setting(settings, "opt_max_new_per_tick"))]
    calls = sum(1 for o in orders if o["right"] == "call")
    puts = len(orders) - calls
    posture = (f"Rules: {len(orders)} directional read(s) cleared the band — {calls} call, {puts} put."
               if orders else "Rules: no read beyond the neutral band today — holding.")
    return {"posture": posture, "orders": orders}


def open_contract_rows(contracts: Iterable[dict], marks: dict[str, dict], *, now_ts: float) -> list[dict]:
    """The held contracts as the analyst and the state endpoint see them: mark, P/L on the premium,
    days to expiry, days held, and whether the mark is today's."""
    out: list[dict] = []
    for c in contracts:
        m, stale = mark_of(c, marks)
        paid = float(c.get("avg_premium") or 0.0)
        q = marks.get(c["contract_symbol"]) or {}
        out.append({
            "contract_symbol": c["contract_symbol"], "display": describe(c), "underlying": c["underlying"],
            "right": c["right"], "strike": c["strike"], "expiry": c["expiry"], "count": int(c["count"]),
            "premium_paid": paid, "cost_total": c.get("cost_total"),
            "mark": round(m, 4) if m is not None else None, "mark_stale": stale,
            "mark_date": None if not stale else c.get("last_mark_date"),
            "value": round(m * MULTIPLIER * int(c["count"]), 2) if m is not None else None,
            "pl_pct": round((m / paid - 1.0) * 100.0, 1) if (m is not None and paid > 0) else None,
            "pl_usd": round((m - paid) * MULTIPLIER * int(c["count"]), 2) if (m is not None) else None,
            "dte": round((options.expiry_epoch(int(c["expiry_ts"])) - now_ts) / 86_400.0, 1),
            "held_days": _days_between(c.get("opened_date"), today_et_str()),
            "spot": q.get("spot") or c.get("last_spot"), "spot_at_open": c.get("spot_at_open"),
            "delta": q.get("delta"), "iv": q.get("iv"), "bid": q.get("bid"), "ask": q.get("ask"),
            "opened_date": c.get("opened_date"), "conviction": c.get("conviction"), "reason": c.get("reason"),
        })
    return out


def _prompt_quote(q: dict | None) -> dict | None:
    if not q:
        return None
    return {k: q.get(k) for k in ("strike", "expiry", "dte", "bid", "ask", "cost_one", "delta", "iv",
                                  "open_interest", "spread_pct", "breakeven")}


def analyst_payload(blob: dict, cands: list[dict], marks: dict[str, dict], *, equity: float, cash: float,
                    spy_price: float | None, now_ts: float, settings: dict | None,
                    macro: dict | None = None, recent_activity: list[dict] | None = None,
                    scorecard: dict | None = None) -> dict:
    st = state(blob)
    at_risk = premium_at_risk(st["contracts"])
    cap = setting(settings, "opt_max_premium_pct") / 100.0 * equity
    room = setting(settings, "opt_max_at_risk_pct") / 100.0 * equity - at_risk
    payload: dict = {
        "equity": round(equity, 2), "cash": round(cash, 2),
        "cash_pct": round(cash / equity * 100.0, 1) if equity else None,
        "premium_at_risk_usd": at_risk,
        "premium_at_risk_pct": round(at_risk / equity * 100.0, 1) if equity else None,
        "limits": {
            "per_contract_usd": round(cap, 2), "at_risk_room_usd": round(max(room, 0.0), 2),
            "max_open": int(setting(settings, "opt_max_open")),
            "new_per_run": int(setting(settings, "opt_max_new_per_tick")),
            "cash_floor_pct": (settings or {}).get("cash_floor_pct"),
            "min_conviction": (settings or {}).get("min_conviction_to_trade"),
            "exits_by_code": {"days_to_expiry": int(setting(settings, "opt_exit_dte")),
                              "stop_pct": setting(settings, "opt_stop_pct"),
                              "target_pct": setting(settings, "opt_target_pct")},
        },
        "open_contracts": open_contract_rows(st["contracts"], marks, now_ts=now_ts),
        "candidates": [{
            "symbol": c["symbol"], "price": c.get("price"), "exposure_group": c.get("exposure_group"),
            "is_fund": c.get("is_fund"), "technicals": c.get("technicals"),
            "directional_read": c.get("directional"), "earnings": c.get("earnings"),
            "expiry": c.get("expiry"), "dte": c.get("dte"), "atm_iv": c.get("atm_iv"),
            "iv_rank": c.get("iv_rank"), "expected_move_pct": c.get("expected_move_pct"),
            "call": _prompt_quote((c.get("calls") or {}).get("atm")) or _prompt_quote((c.get("calls") or {}).get("otm")),
            "call_cheaper": _prompt_quote((c.get("calls") or {}).get("otm")) if (c.get("calls") or {}).get("atm") else None,
            "put": _prompt_quote((c.get("puts") or {}).get("atm")) or _prompt_quote((c.get("puts") or {}).get("otm")),
            "put_cheaper": _prompt_quote((c.get("puts") or {}).get("otm")) if (c.get("puts") or {}).get("atm") else None,
        } for c in cands],
        "spy": spy_price,
    }
    if int(st.get("pause_ticks_left") or 0) > 0:
        payload["new_entries_paused_runs_left"] = int(st["pause_ticks_left"])
    if macro:
        payload["macro"] = macro
    if recent_activity:
        payload["your_recent_activity"] = recent_activity
    if scorecard and scorecard.get("closed"):
        payload["your_closed_contracts_in_dollars"] = scorecard
    return payload


def recent_activity(trades: list[dict], *, today: str, days: int = 14, limit: int = 20) -> list[dict]:
    """This arm's own option rows from the last `days`, newest first, trimmed for the prompt."""
    out: list[dict] = []
    for t in sorted(trades, key=lambda r: -float(r.get("ts") or 0.0)):
        if t.get("instrument") != "option":
            continue
        age = _days_between(t.get("date"), today)
        if age is None or age > days:
            continue
        row = {"date": t.get("date"), "what": t.get("display") or t.get("symbol"), "side": t.get("side"),
               "status": t.get("status"), "contracts": t.get("shares"), "price": t.get("price"),
               "dollars": t.get("gross")}
        if t.get("status") == "skipped":
            row["why_not"] = t.get("skip_reason")
        else:
            row["reason"] = (t.get("reason") or "")[:110]
            if t.get("realized_pl") is not None:
                row["realized_pl"] = t.get("realized_pl")
        out.append(row)
        if len(out) >= limit:
            break
    return out


def scorecard(trades: Iterable[dict], contracts: Iterable[dict] = ()) -> dict:
    """Dollars, not rates. Every closed contract: premium deployed, what it came back as, and what
    the same dollars would have done in the index over the same days; how often the direction was
    right, and how often it was right and the contract still lost (decay and the spread)."""
    closed = premium = pl = 0.0
    index_alt = 0.0
    n_index = 0
    right = right_lost = 0
    for t in trades:
        if t.get("instrument") != "option" or t.get("status") != "filled" or t.get("side") not in CLOSING_SIDES:
            continue
        closed += 1
        premium += float(t.get("cost_total") or 0.0)
        p = float(t.get("realized_pl") or 0.0)
        pl += p
        if isinstance(t.get("index_alt_usd"), (int, float)):
            index_alt += float(t["index_alt_usd"])
            n_index += 1
        if t.get("direction_right") is True:
            right += 1
            if p < 0:
                right_lost += 1
    held = list(contracts)
    return {
        "closed": int(closed), "premium_deployed_usd": round(premium, 2), "pl_usd": round(pl, 2),
        "index_alt_usd": round(index_alt, 2) if n_index else None, "index_alt_n": n_index,
        "direction_right": right, "direction_right_but_lost": right_lost,
        "open": len(held), "premium_at_risk_usd": premium_at_risk(held),
    }


def fingerprint(cands: Iterable[dict]) -> list[dict]:
    """What the arm was shown, compact: one row per underlying with its read and the two costs."""
    out = []
    for c in cands:
        d = c.get("directional") or {}
        out.append({"symbol": c.get("symbol"), "score": d.get("score"), "signal": d.get("signal"),
                    "call_usd": ((c.get("calls") or {}).get("atm") or (c.get("calls") or {}).get("otm") or {}).get("cost_one"),
                    "put_usd": ((c.get("puts") or {}).get("atm") or (c.get("puts") or {}).get("otm") or {}).get("cost_one"),
                    "expiry": c.get("expiry")})
    return out


def update_pause(blob: dict, *, current_drawdown_pct: float | None, settings: dict | None) -> str | None:
    """Count a paused run down, or start a pause when the drawdown is deep enough. Returns a note for
    the warnings when a pause begins. A None drawdown (too little history) starts nothing."""
    st = state(blob)
    left = int(st.get("pause_ticks_left") or 0)
    if left > 0:
        st["pause_ticks_left"] = left - 1
        return None
    if current_drawdown_pct is None:
        return None
    dd = abs(float(current_drawdown_pct))
    # ONE pause per drawdown. Without this it re-armed the moment it expired, on the same drawdown,
    # and the arm ran five runs off and one on for as long as it stayed under water. It re-arms only
    # once the book is back within 1% of its peak, i.e. a new drawdown.
    if dd < 1.0:
        st["pause_spent"] = False
        return None
    if dd >= setting(settings, "opt_pause_drawdown_pct") and not st.get("pause_spent"):
        st["pause_spent"] = True
        st["pause_ticks_left"] = int(setting(settings, "opt_pause_ticks"))
        return (f"drawdown {abs(float(current_drawdown_pct)):.1f}% — new entries paused for "
                f"{int(setting(settings, 'opt_pause_ticks'))} runs (exits still run)")
    return None


# ------------------------------------------------------------------ the per-tick chain context (I/O)

def rank_underlyings(rows: Iterable[dict], *, limit: int = MAX_UNDERLYINGS) -> list[dict]:
    """The index ETFs first, then the strongest directional reads, up to `limit`. Crypto and
    sub-$5 names are dropped: no listed options on the first, junk strikes on the second."""
    by_sym: dict[str, dict] = {}
    for r in rows:
        sym = str(r.get("symbol") or "").upper()
        if not sym or sym.endswith("-USD") or sym in by_sym:
            continue
        px = r.get("price")
        if sym not in INDEX_ETFS and (not isinstance(px, (int, float)) or px < MIN_UNDERLYING_PRICE):
            continue
        by_sym[sym] = r
    out: list[dict] = [by_sym[s] for s in INDEX_ETFS if s in by_sym]
    rest = [r for s, r in by_sym.items() if s not in INDEX_ETFS]
    rest.sort(key=lambda r: -abs(int(directional(r.get("technicals")).get("score", 50)) - 50))
    for r in rest:
        if len(out) >= limit:
            break
        out.append(r)
    return out


async def build_context(
    client, *, rows: list[dict], held: list[dict], now_ts: float,
    earnings_of: Callable[[str], Awaitable[object]], is_fund: Callable[[str], bool],
    settings: dict | None = None, limit: int = MAX_UNDERLYINGS, concurrency: int = 4,
    record_iv: bool = True, log=None,
) -> dict:
    """One chain snapshot for every options arm on this tick: marks for every held contract and a
    priced candidate row per underlying. Built once, after main persists, inside the caller's try.
    Each underlying is its own try, so one bad chain costs one row, not the tick.

    `earnings_of(symbol)` returns a news.EarningsLookup (date, ok); it is called only for stocks,
    since a fund reports no earnings. `is_fund(symbol)` says which is which."""
    sem = asyncio.Semaphore(concurrency)
    marks: dict[str, dict] = {}
    cands: list[dict] = []
    errors: list[str] = []

    async def _mark(underlying: str, expiry_ts: int, wanted: list[dict]) -> None:
        async with sem:
            try:
                ch = await options.fetch_chain(client, underlying, expiry_ts=expiry_ts)
                options.annotate_expiry(ch, now_ts=now_ts)
                dte = (options.expiry_epoch(expiry_ts) - now_ts) / 86_400.0
                by_sym = {c.contract_symbol: c for c in (ch.expiry.calls + ch.expiry.puts)} if ch.expiry else {}
                for w in wanted:
                    c = by_sym.get(w["contract_symbol"])
                    if c is None:
                        errors.append(f"{w['contract_symbol']} missing from its chain")
                        continue
                    marks[w["contract_symbol"]] = quote_from(c, underlying=underlying, spot=ch.spot,
                                                              market_state=ch.market_state, dte=dte)
            except Exception as e:  # noqa: BLE001 — one chain, one row lost
                errors.append(f"marks {underlying} {expiry_ts}: {type(e).__name__}: {e}")

    async def _candidate(row: dict) -> None:
        sym = str(row.get("symbol") or "").upper()
        async with sem:
            try:
                fund = bool(is_fund(sym))
                earn_date, known = None, True
                if not fund:
                    e = await earnings_of(sym)
                    earn_date = getattr(e, "date", None) if e is not None else None
                    known = bool(getattr(e, "ok", False)) if e is not None else False
                ch = await options.fetch_chain(client, sym)
                pick, _warns = options.select_expiry(ch, now=now_ts, earnings_date=earn_date)
                if not pick:
                    errors.append(f"{sym}: no usable expiry")
                    return
                ch = await options.fetch_chain(client, sym, expiry_ts=int(pick["ts"]))
                if ch.expiry is None or not ch.spot:
                    errors.append(f"{sym}: chain had no contracts or no spot")
                    return
                options.annotate_expiry(ch, now_ts=now_ts)
                dte = (options.expiry_epoch(int(pick["ts"])) - now_ts) / 86_400.0
                qs_c = [quote_from(c, underlying=sym, spot=ch.spot, market_state=ch.market_state, dte=dte)
                        for c in ch.expiry.calls]
                qs_p = [quote_from(c, underlying=sym, spot=ch.spot, market_state=ch.market_state, dte=dte)
                        for c in ch.expiry.puts]
                atm_iv = ch.expiry.atm_iv
                rank = None
                if atm_iv:
                    hist = options.load_iv_history(sym)
                    rank = options.iv_rank(hist, atm_iv, current_dte=int(round(dte)),
                                           history_dte=options.load_iv_history_dte(sym))
                    if record_iv:
                        options.append_iv_history(sym, atm_iv, dte=int(round(dte)))
                em = ch.expiry.expected_move
                cands.append({
                    "symbol": sym, "price": ch.spot, "market_state": ch.market_state,
                    "exposure_group": row.get("exposure_group"), "is_fund": fund,
                    "technicals": row.get("technicals") or {},
                    "directional": directional(row.get("technicals")),
                    "earnings": {"date": earn_date, "known": known,
                                 "straddles": bool(earn_date and str(earn_date) <= str(pick["iso"]))},
                    "expiry": pick["iso"], "expiry_ts": int(pick["ts"]), "dte": round(dte, 1),
                    "atm_iv": round(atm_iv, 4) if atm_iv else None, "iv_rank": rank,
                    "expected_move_pct": round(em / ch.spot * 100.0, 1) if (em and ch.spot) else None,
                    "calls": pick_contracts(qs_c, settings=settings, underlying=sym),
                    "puts": pick_contracts(qs_p, settings=settings, underlying=sym),
                })
            except Exception as e:  # noqa: BLE001
                errors.append(f"{sym}: {type(e).__name__}: {e}")

    groups: dict[tuple[str, int], list[dict]] = {}
    for c in held:
        groups.setdefault((str(c["underlying"]).upper(), int(c["expiry_ts"])), []).append(c)
    jobs = [_mark(u, ts, ws) for (u, ts), ws in groups.items()]
    jobs += [_candidate(r) for r in rank_underlyings(rows, limit=limit)]
    await asyncio.gather(*jobs)
    cands.sort(key=lambda c: -abs(int((c.get("directional") or {}).get("score", 50)) - 50))
    if log is not None and errors:
        log.info("options arms: %d chain problem(s): %s", len(errors), "; ".join(errors[:6]))
    return {"marks": marks, "candidates": cands, "by_underlying": {c["symbol"]: c for c in cands},
            "errors": errors, "as_of": now_ts}
