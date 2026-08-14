"""
Pure trade-logic for the AI paper-trading sandbox — gating, deterministic order validation + fills, and
NAV computation. NO network / no LLM here: these functions take already-fetched data (quotes, an analyst
decision) and mutate an in-memory ledger blob, so they're fully unit-testable offline. The I/O
orchestration (build snapshots, call the analyst, fetch quotes, persist) lives in main.py's /sandbox/tick
endpoint, where the shared httpx client + the `_build_portfolio_snapshot`/`_snapshot` helpers already live.

The invariant that matters: the LLM only PROPOSES; this module is the sole authority on what the ledger
does. It clamps every order to the cash floor, the per-exposure cap, and the shares actually held, and
asserts cash conservation before the caller commits.
"""
from __future__ import annotations

import datetime as dt
import time
from typing import Callable
from zoneinfo import ZoneInfo

from . import market_calendar

ET = ZoneInfo("America/New_York")

# Session open in ET seconds-of-day (mirrors market_now.session_phase, kept local to avoid a cycle).
# The CLOSE is per-date via market_calendar.session_end_seconds — it moves on 1pm half-days.
_REG_OPEN = 9 * 3600 + 30 * 60   # 09:30


def now_et() -> dt.datetime:
    return dt.datetime.now(ET)


def today_et_str(now: dt.datetime | None = None) -> str:
    return (now or now_et()).date().isoformat()


def age_on(birth_date: str | None, today: dt.date | None = None) -> int | None:
    """Whole years from `birth_date` (ISO yyyy-mm-dd) to `today`. None if unset or unparseable.

    The account's age is stored as a DATE rather than a number because an age is only true for a
    year. `current_age` was written once, by hand, and then aged silently into a lie — and it is not
    a cosmetic field: the glidepath the entire strategy hangs off is (retirement_age - current_age),
    so a stale age quietly lengthens the runway by however long ago it was typed. A birth date is
    the same fact with the drift taken out.
    """
    if not birth_date:
        return None
    try:
        born = dt.date.fromisoformat(str(birth_date).strip())
    except (ValueError, TypeError):
        return None
    today = today or now_et().date()
    if born > today:
        return None
    # Take a year off when this year's birthday has not arrived. Comparing (month, day) tuples also
    # settles Feb 29 without a special case: in a common year the birthday turns over on Mar 1.
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def effective_age(settings: dict, today: dt.date | None = None) -> int | None:
    """The account's age now: derived from `birth_date` when one is set, else the stored number.

    birth_date WINS wherever both exist, because the stored number is the stale one by construction.
    The fallback is what lets an account configured before birth_date existed keep working untouched
    until someone sets a date — until then it behaves exactly as it did.
    """
    derived = age_on(settings.get("birth_date"), today)
    # Explicit None check, not `or`: age 0 is a real answer and would be swallowed by truthiness.
    return derived if derived is not None else settings.get("current_age")


def settings_for_prompt(settings: dict, today: dt.date | None = None) -> dict:
    """What the analyst sees: `current_age` refreshed from `birth_date`, and `birth_date` dropped.

    The whole settings dict is json.dumps'd into both the daily and weekly prompts, so every key
    left in it is sent to the model verbatim. The model reasons about RUNWAY, which the derived age
    already carries in full; an exact date of birth adds nothing to the decision and is the more
    personal of the two facts. So derive it here, then drop the source.
    """
    out = {k: v for k, v in settings.items() if k != "birth_date"}
    age = effective_age(settings, today)
    if age is not None:
        out["current_age"] = age
    return out


def is_crypto(symbol: str) -> bool:
    return symbol.upper().endswith("-USD")


def round_shares(symbol: str, shares: float) -> float:
    """Whole shares for stocks/ETFs, 6-dp fractional for crypto (matches _sanitize_plan)."""
    return round(shares, 6) if is_crypto(symbol) else float(int(shares))


# A whole-share buy sized at or above this fraction of one share rounds UP to one, provided a whole
# share clears every limit. Below it, the order stands as written and floors to zero.
#
# Sizing floors everywhere else, because flooring can only ever leave a limit further away. The one
# place it cannot is the last share: floor turns "just under one" into NOTHING, and an order that
# buys nothing is not a smaller version of the order — it is the absence of it. Measured 2026-08-13:
# a GOOGL buy of $346 against a $346.39 fill evaporated, on a book holding $6,032 of idle cash with
# an $862 target for that name and no limit within reach. The model was told upstream that "the
# server re-sizes to the available cash, so approximate is fine", which is true downward and false
# here — so it approximated, and the order silently ceased to exist.
#
# 0.9, not 0.5. Round-to-NEAREST would be the tidier rule, but it hands back up to a 2x overspend on
# the analyst's stated size, and this codebase already has a considered position on that: the
# 2026-08-03 skip-reason test uses a $200 order against a $374 share as its example of an order that
# is "simply smaller than one share" and ought not to fill. It should still not fill. This rescue is
# only for an order that MEANT a whole share and missed — on slippage, on intraday drift between the
# model's snapshot and the fill, or on shading the figure to a round number. A 10% band covers all
# three; past it, the fraction was deliberate. Keeping it narrow also keeps it from becoming an
# incentive: a model that learns a small number buys a whole share will write small numbers.
#
# It applies to the FRACTIONAL REMAINDER, not just to the zero case, because the same arithmetic bites
# at every size. An order sized the obvious way — N shares times the last price — divides by a fill
# that slippage has already pushed above that price, so it lands at N − 0.0005 and floors to N−1. At
# N=1 that is the GOOGL order vanishing; at N=10 it is nine shares and a silently unspent share's
# worth of cash. Same cause, same fix.
ROUND_UP_MIN_SHARE = 0.9


def _usd(x: float) -> str:
    """Dollars for a human. Cents below $100, because `:,.0f` renders a $0.40 order as "$0"."""
    return f"${x:,.2f}" if abs(x) < 100 else f"${x:,.0f}"


# Spot-bitcoin ETFs. Every one holds the same asset, so which you own is a question about custody,
# liquidity and fees — the user's call, not the model's.
BTC_ETFS = frozenset({"IBIT", "FBTC", "GBTC", "BITB", "ARKB", "BTCO", "HODL", "BRRR", "EZBC", "BTCW"})


def prefer_btc_etf(
    orders: list[dict], *, preferred: str, positions: list[dict],
    price_of: Callable[[str], float | None],
) -> tuple[list[dict], list[str]]:
    """Redirect BUYs of a spot-bitcoin ETF onto the user's chosen vehicle.

    Returns (orders, notes). Pure: `orders` is a new list, inputs are not mutated.

    Four rules, each load-bearing:

    * **BUYS ONLY.** A sell is never redirected — you can only sell what you actually hold, and
      rewriting "sell IBIT" into "sell FBTC" would turn a valid exit into a "not held" skip.
    * **Consolidate, don't fragment.** If the book already holds the ordered vehicle and NOT the
      preferred one, leave it alone. Splitting one exposure across two funds to honour a preference
      costs spread and leaves two positions where the cap logic expects one; the preference applies
      to NEW exposure, and takes effect naturally once the old vehicle is closed.
    * **Only to something fillable.** If the preferred ETF has no price this tick it cannot fill, so
      the order is left as-is rather than converted into a guaranteed skip.
    * **Say so.** Every substitution returns a note, which lands in the trade log — a silent symbol
      rewrite would make the log disagree with what the model actually proposed.
    """
    pref = (preferred or "").strip().upper()
    if not pref or pref not in BTC_ETFS:
        return list(orders), []
    held = {str(p.get("symbol", "")).upper() for p in positions if (p.get("shares") or 0) > 0}
    out: list[dict] = []
    notes: list[str] = []
    for o in orders:
        sym = str(o.get("symbol") or "").strip().upper()
        side = str(o.get("side") or "").strip().lower()
        if side != "buy" or sym not in BTC_ETFS or sym == pref:
            out.append(o)
            continue
        if sym in held and pref not in held:
            out.append(o)
            continue
        px = price_of(pref)
        if not px or px <= 0:
            out.append(o)
            continue
        # Everything below is share-price arithmetic, and it matters because the two funds hold the
        # same asset at very different share prices (IBIT ~$36 vs FBTC ~$55).
        #
        # Size in DOLLARS, which is share-price independent. An order may carry `dollars`, or only a
        # `shares` count — and a raw `shares` count carried across would buy ~1.5x the intended
        # notional. So convert via the ORIGINAL fund's price and drop `shares`; without the convert,
        # a shares-only order would arrive with neither field and be skipped outright.
        dollars = float(o.get("dollars") or 0.0)
        if dollars <= 0:
            want_shares = float(o.get("shares") or 0.0)
            src_px = price_of(sym) or 0.0
            if want_shares > 0 and src_px > 0:
                dollars = want_shares * src_px
            else:
                out.append(o)   # nothing sizeable to carry over — leave the order untouched
                continue
        # The entry zone was priced against the ORDERED fund's share price, so carrying it across
        # would gate the new symbol against a band that has nothing to do with it — and with
        # respect_entry_zones on, would block every substituted buy forever. Drop it.
        sub = {k: v for k, v in o.items() if k not in ("entry_low", "entry_high", "shares")}
        sub["symbol"] = pref
        sub["dollars"] = round(dollars, 2)
        reason = str(o.get("reason") or "").strip()
        sub["reason"] = (reason + f" [routed {sym}→{pref}: preferred bitcoin ETF]").strip()
        out.append(sub)
        notes.append(f"{sym}→{pref} (preferred bitcoin ETF)")
    return out, notes


def intra_group_swaps(orders: list[dict], *, group_of: Callable[[str], str]) -> set[int]:
    """Indices of orders that sell and re-buy the SAME exposure in one tick. Both sides are returned.

    A swap inside one exposure group changes nothing about what the account owns. It pays the spread
    twice and, in a taxable account, realises a gain for the privilege. The prompt has said "NEVER
    sell one to buy its equivalent" since the sandbox was built — and on 2026-08-06 the model did it
    anyway, selling all 3 SPY to buy VTI on the argument that 0.095% is a worse expense ratio than
    0.030%. That reasoning is correct for NEW money and wrong for an existing holding: it banked
    $85.56 of SHORT-TERM gains to save ~$1.50 a year in fees. The buy was then blocked by the group
    cap, so the proceeds did not even reach VTI — cash went from 39.9% to 58.3%.

    The prompt caveat existed and was ignored, which is the argument for enforcing it here instead.
    This module is the authority on what the ledger does; a same-group round trip is mechanically
    detectable, so it does not need to be trusted to a paragraph.

    Only pairs matter: a lone sell (trimming a group that is over its cap) is untouched, and so is a
    sell funding a buy in a DIFFERENT group — that is a real allocation change, governed by the
    prompt's sell rules rather than by this one.
    """
    by_group: dict[str, dict[str, list[int]]] = {}
    for i, o in enumerate(orders):
        side = str(o.get("side") or "").strip().lower()
        sym = str(o.get("symbol") or "").strip().upper()
        if side not in ("buy", "sell") or not sym:
            continue
        g = group_of(sym)
        by_group.setdefault(g, {"buy": [], "sell": []})[side].append(i)
    blocked: set[int] = set()
    for sides in by_group.values():
        if sides["buy"] and sides["sell"]:
            blocked.update(sides["buy"])
            blocked.update(sides["sell"])
    return blocked


def same_day_reversals(
    orders: list[dict], positions: list[dict], *, now_ts: float | None = None,
) -> set[int]:
    """Indices of SELL orders that unwind a position bought earlier the same trading day.

    Measured 2026-08-11. The 14:35 tick bought 7 VXUS toward a 22% target; a second tick 54 minutes
    later re-ran the weekly review, which rewrote that target to 12%, and the same tick sold 5 of
    those shares back. Both decisions were correct against the plan in front of them. The account
    still paid the spread twice and booked $3.84 of SHORT-TERM gain — realised against the position's
    average cost, so a round trip that netted nine cents on the shares themselves became a taxable
    event forty times that size.

    `intra_group_swaps` cannot catch this: the two legs are in different ticks, so they are never in
    one order list. What links them is the position's own `last_add_at`, which the fill path already
    stamps on every buy.

    Same TRADING DAY in exchange time, not 24 hours — a sell the next morning is a normal overnight
    decision, and measuring in hours would block it while letting a 23-hour-old add through.

    Deliberately NOT applied to a liquidation: an exit-date flatten is the user asking to be out, and
    a guard against churn must never be the reason the account is still holding stock on the date
    they named.
    """
    now_ts = now_ts or time.time()
    today = dt.datetime.fromtimestamp(now_ts, ET).date()
    bought_today = {
        str(p.get("symbol") or "").upper()
        for p in positions
        if (p.get("last_add_at") or 0)
        and dt.datetime.fromtimestamp(float(p["last_add_at"]), ET).date() == today
    }
    return {
        i for i, o in enumerate(orders)
        if str(o.get("side") or "").strip().lower() == "sell"
        and str(o.get("symbol") or "").strip().upper() in bought_today
    }


def accrue_cash_interest(blob: dict, *, now: dt.datetime | None = None) -> float:
    """Credit idle cash a money-market yield. Returns the amount credited (0.0 when nothing is due).

    The ledger paid NOTHING on cash, which quietly biased every decision that weighs holding cash.
    Real cash sits in a sweep/MMF earning roughly the T-bill rate, so a 0% model overstates the cost
    of being uninvested and pushes the strategist to deploy harder than reality justifies. Measured
    2026-08-05 on this account's own numbers: the 40%-cash position looks like it costs $193 against
    the benchmark at 0%, but nearer $177 once the cash earns its keep — and over the 18-year runway
    the gap between those two assumptions is $8.5k of projected terminal value.

    It also makes the benchmark comparison fair. The benchmark shadow is 100% invested by
    construction, so charging the sandbox 0% on cash penalised it twice for the same choice.

    Accrual is ACT/365 simple daily, cursored on `last_interest_date` so a forced re-run on the same
    day cannot double-credit. Interest is earnings, not a contribution: it raises `cash` and
    `interest_total` but NEVER `funded_total`, so `total_return_pct` keeps counting it as return.
    """
    now = now or now_et()
    today = now.date()
    apy = float((blob.get("settings") or {}).get("cash_apy_pct") or 0.0)
    cash = float(blob.get("cash") or 0.0)
    last = blob.get("last_interest_date")
    if apy <= 0.0 or cash <= 0.0:
        blob["last_interest_date"] = today.isoformat()
        return 0.0
    if not last:
        # First run: start the clock, don't back-pay a period we never modelled.
        blob["last_interest_date"] = today.isoformat()
        return 0.0
    try:
        days = (today - dt.date.fromisoformat(last)).days
    except ValueError:
        blob["last_interest_date"] = today.isoformat()
        return 0.0
    if days <= 0:
        return 0.0
    amount = round(cash * (apy / 100.0) * (days / 365.0), 2)
    if amount <= 0.0:
        blob["last_interest_date"] = today.isoformat()
        return 0.0
    blob["cash"] = round(cash + amount, 2)
    blob["interest_total"] = round(float(blob.get("interest_total") or 0.0) + amount, 2)
    blob["last_interest_date"] = today.isoformat()
    return amount


# How far the weekly targets may fall short of (100 - cash_target) before it counts as a real hole
# rather than rounding. A point or two is noise; ten is a policy decision nobody made.
ALLOCATION_SLACK_PCT = 3.0


def canonicalize_targets(
    note: dict | None, *, group_of: Callable[[str], str] | None = None,
) -> list[str]:
    """Rewrite a strategy note's targets onto today's exposure groups, in place. Returns the labels
    that were rewritten or merged away, for logging.

    A note names its targets by GROUP LABEL, and the strategist picks those labels from what it sees
    in the book — so nothing stops it naming one group twice under two spellings. On 2026-08-10 the
    weekly plan asked for `US_EQUITY` 22% AND `SP500` 18%; `SP500` has been an alias of `US_EQUITY`
    since the 2026-08-05 merge, so that is 40% of one group against a 25% cap. The plan summed to
    exactly (100 - cash_target) and read as six diversified groups, which is precisely why it slipped
    past every eye on it: the arithmetic was right, the vocabulary was not.

    `allocation_gap` already collapses aliases to CHECK the plan, but checking happens after the note
    is stored, so every downstream consumer — the daily tick's fill logic, the prompt that shows the
    plan back to the decision model, the app — still saw two entries for one exposure. Fixing the
    stored note instead means the alias can only exist for the instant between parse and store.

    Merging is safe in a way that clamping would not be: two labels for one group is a naming error
    with exactly one correct reading, whereas a merged target that lands over the cap is a real
    policy problem that must stay visible for `allocation_gap` to report and the next review to fix.
    """
    if not note:
        return []
    targets = note.get("targets") or []
    if not targets:
        return []
    _g = group_of or (lambda s: s)
    merged: dict[str, dict] = {}
    changed: list[str] = []
    for t in targets:
        raw = str(t.get("exposure_group") or "").strip()
        canon = _g(raw) if raw else raw
        if canon != raw:
            changed.append(f"{raw}->{canon}")
        prior = merged.get(canon)
        if prior is None:
            merged[canon] = {**t, "exposure_group": canon}
        else:
            changed.append(f"{canon} named twice, summed")
            prior["target_pct"] = round(
                float(prior.get("target_pct") or 0.0) + float(t.get("target_pct") or 0.0), 1)
    note["targets"] = list(merged.values())
    return changed


def allocation_gap(
    note: dict | None, *, max_position_pct: float = 25.0,
    group_of: Callable[[str], str] | None = None,
) -> dict | None:
    """Audit a weekly strategy note's targets. Returns None when the plan is sound.

    Two failure modes, both of which make the daily tick unable to execute the plan as written:

    * **Unallocated.** Targets that sum to less than (100 - cash_target_pct) leave the remainder as
      cash by default. Nobody decided that, and on a long-runway account it is the dominant cost —
      measured 2026-08-04, a 64% plan against a 22% cash target left the book at 42% cash and the
      whole S&P shortfall was that idle cash, not the picks.
    * **Unreachable.** A target above `max_position_pct` can never be filled, so the tick fires
      blocked orders against it forever.

    Returns a dict describing the problem so the caller can log it AND feed it to the next review,
    which is the only stage that can actually fix the plan.
    """
    if not note:
        return None
    targets = note.get("targets") or []
    cash_target = float(note.get("cash_target_pct") or 0.0)
    total = sum(float(t.get("target_pct") or 0.0) for t in targets)
    investable = max(0.0, 100.0 - cash_target)
    shortfall = investable - total

    # Collapse the targets onto their CURRENT exposure groups before checking the cap. Checking each
    # target on its own misses the case that matters: after VTI/SPY/VOO/QQQM were merged into
    # US_EQUITY on 2026-08-05, the standing plan's "VTI 25%" and "SP500 21%" each passed a 25% cap
    # individually while together asking for 46% of ONE group. The daily tick then fired a blocked
    # VTI buy every day and deployed nothing — exactly the "unreachable target = blocked orders
    # forever" failure this function exists to catch, hidden by the plan naming two labels for one
    # thing. A regrouping can turn a sound old plan into an impossible one without a word of it
    # changing, so the check has to be done against today's map, not the plan's vocabulary.
    _g = group_of or (lambda s: s)
    by_group: dict[str, float] = {}
    for t in targets:
        g = _g(str(t.get("exposure_group") or ""))
        by_group[g] = by_group.get(g, 0.0) + float(t.get("target_pct") or 0.0)
    over_cap = [g for g, pct in by_group.items() if pct > max_position_pct + 0.01 and g]
    # Fewest groups that could cover the invested share without breaching the per-group cap.
    need = int(-(-investable // max_position_pct)) if max_position_pct > 0 else 0
    if shortfall <= ALLOCATION_SLACK_PCT and not over_cap and len(targets) >= need:
        return None
    return {
        "targets_sum_pct": round(total, 1),
        "cash_target_pct": round(cash_target, 1),
        "investable_pct": round(investable, 1),
        "unallocated_pct": round(max(0.0, shortfall), 1),
        "groups": len(targets),
        "groups_needed": need,
        "targets_over_cap": [g for g in over_cap if g],
    }


# US long-term capital-gains treatment starts at MORE THAN one year of holding.
_LONG_TERM_DAYS = 366


def annotate_holding_period(
    book_positions: list[dict], ledger_positions: list[dict], *, now_ts: float | None = None,
) -> None:
    """Add holding period + capital-gains status to the book rows the analyst sees, in place.

    `_build_portfolio_snapshot` prices a `Holding(symbol, shares, avg_cost)`, which drops `opened_at`
    — so the decision model could see a 19% gain on a position and had no way to know it was four days
    old. Selling that is an ordinary-income short-term gain; the same sale a year later is taxed at
    the long-term rate. That difference is large enough to be worth weighing, and it was simply not
    in front of the model.

    Deliberately informational, not a gate: `sandbox_job` blocks things that endanger the ACCOUNT
    (cash conservation, caps, shares held). Tax efficiency is a preference to weigh against the
    reason for selling, so it belongs in the prompt, not in the validator.
    """
    now_ts = now_ts or time.time()
    by_sym = {str(p.get("symbol", "")).upper(): p for p in ledger_positions}
    for row in book_positions:
        led = by_sym.get(str(row.get("symbol", "")).upper())
        # `last_add_at`, not `opened_at`: adding to a position starts a fresh holding period for the
        # NEW shares (each tax lot is clocked separately). Using the older date would overstate how
        # much of the position already qualifies for long-term treatment.
        started = (led or {}).get("last_add_at") or (led or {}).get("opened_at")
        if not isinstance(started, (int, float)):
            continue
        days = int((now_ts - float(started)) // 86_400)
        row["holding_days"] = days
        row["capital_gains"] = "long_term" if days >= _LONG_TERM_DAYS else "short_term"
        if days < _LONG_TERM_DAYS:
            row["days_to_long_term"] = _LONG_TERM_DAYS - days


def tick_gate(blob: dict, *, now: dt.datetime | None = None, force: bool = False) -> tuple[bool, str]:
    """Whether the tick should place trades. Returns (proceed, status). `force` (a manual "run now")
    relaxes the intraday-phase check but still requires a real trading day and honours the day cursor
    only when not forced."""
    now = now or now_et()
    today = now.date()
    if not (blob.get("settings") or {}).get("master_enabled", False):
        return False, "disabled"
    if not market_calendar.is_trading_day(today):
        return False, "market_closed"
    sod = now.hour * 3600 + now.minute * 60 + now.second
    allow_after = bool((blob.get("settings") or {}).get("allow_after_hours", False))
    # The day's REAL close, not a hard-coded 16:00. On the three 1pm-ET half-days a year the market
    # has been shut for hours by the time the daily tick runs, so a fill would be priced off a stale
    # quote and stamped as a live trade.
    reg_end, after_end = market_calendar.session_end_seconds(today)
    in_regular = _REG_OPEN <= sod < reg_end
    in_after = reg_end <= sod < after_end
    if not force:
        if in_after and not allow_after:
            return False, "after_hours_disabled"
        if not (in_regular or in_after):
            return False, "outside_session"
    if blob.get("last_tick_date") == today.isoformat() and not force:
        return False, "already_ran"
    return True, "ok"


def mark_price(p: dict, price_of: Callable[[str], float | None]) -> float | None:
    """Best available mark for a position: fresh quote, else last known mark, else cost basis.

    Marking an unpriceable holding at ZERO (the old behaviour) was wrong in two directions at once.
    It wrote an invented drawdown into the append-only NAV log, which nothing can later correct; and
    because the exposure-cap check used the same marks, the position's group looked EMPTY, so a
    sibling ticker could buy straight through the cap — a price outage silently disabled a risk
    limit. A stale mark is an approximation; zero is a fabrication.
    """
    px = price_of(p["symbol"])
    if px and px > 0:
        return float(px)
    for key in ("last_price", "avg_cost"):
        v = p.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return None


def stale_marks(positions: list[dict], price_of: Callable[[str], float | None]) -> list[str]:
    """Symbols currently valued from a stale mark, so the caller can surface it rather than hide it."""
    out: list[str] = []
    for p in positions:
        px = price_of(p["symbol"])
        if not (px and px > 0) and float(p.get("shares") or 0) > 0:
            out.append(p["symbol"])
    return out


def positions_value(positions: list[dict], price_of: Callable[[str], float | None]) -> float:
    """Mark-to-market value of the book, falling back to the last known mark (see `mark_price`)."""
    total = 0.0
    for p in positions:
        px = mark_price(p, price_of)
        if px:
            total += p["shares"] * px
    return round(total, 2)


def _find(positions: list[dict], symbol: str) -> dict | None:
    s = symbol.upper()
    for p in positions:
        if p["symbol"].upper() == s:
            return p
    return None


def exit_date_flatten_orders(blob: dict, price_of: Callable[[str], float | None]) -> list[dict] | None:
    """If today (ET) is on/after the configured exit_date, return sell-everything orders (bypasses the
    LLM). None means "not past exit date"."""
    ed = (blob.get("settings") or {}).get("exit_date")
    if not ed:
        return None
    try:
        if now_et().date() < dt.date.fromisoformat(ed):
            return None
    except ValueError:
        return None
    orders = []
    for p in blob.get("positions", []):
        if price_of(p["symbol"]):
            orders.append({"symbol": p["symbol"], "side": "sell", "shares": p["shares"],
                           "dollars": 0.0, "conviction": 100, "reason": "Exit date reached — flattening to cash."})
    return orders


# Which ticker expresses a GROUP target, for groups that span more than one vehicle. Ordered
# cheapest-first by expense ratio, because within a group the exposure is by definition the same and
# cost is the only durable difference left. Every other group IS its own ticker (VXUS, SCHD, AMZN…),
# which the identity fallback handles without needing an entry here.
GROUP_REPRESENTATIVE: dict[str, tuple[str, ...]] = {
    "US_EQUITY": ("SPLG", "VTI", "VOO", "IVV", "ITOT", "SCHB", "SPY", "QQQM", "QQQ", "SPMO"),
    "ETH": ("FETH", "ETHA"),
    "GOLD": ("GLDM", "IAU", "GLD"),
}


def group_representative(
    group: str, *, positions: list[dict], price_of: Callable[[str], float | None],
    group_of: Callable[[str], str], preferred_btc_etf: str = "FBTC",
) -> str | None:
    """The ticker to buy in order to express a target stated as an exposure GROUP.

    Strategy notes name targets by group, not by ticker, so anything acting on a plan mechanically has
    to invert that map. Held-first matters most: buying SPLG to top up a US_EQUITY target already
    expressed as VTI would split one exposure across two vehicles, which is the fragmentation the
    grouping exists to prevent. Returns None when nothing in the group can be priced — the caller
    must skip rather than guess."""
    held = [p for p in positions
            if group_of(p["symbol"]) == group and float(p.get("shares") or 0) > 0]
    if held:
        return max(held, key=lambda p: float(p["shares"]) * (price_of(p["symbol"]) or 0.0)
                   )["symbol"].upper()
    if group == "BTC":
        # Same rule prefer_btc_etf enforces on the LLM's buys: which bitcoin vehicle to own is the
        # user's call, so the mechanical path must not quietly pick a different one.
        c = (preferred_btc_etf or "").strip().upper()
        if c and price_of(c):
            return c
    for c in GROUP_REPRESENTATIVE.get(group, ()):
        if price_of(c):
            return c
    return group if price_of(group) else None


def rules_decision(
    blob: dict, *, plan: dict | None, group_of: Callable[[str], str],
    price_of: Callable[[str], float | None],
) -> dict:
    """Fill toward the standing plan's targets. No model, no market opinion, no judgement.

    This is the control arm. It exists to answer a question the live account cannot answer about
    itself: does the daily analyst add anything over mechanically executing the plan it was already
    given? On 2026-08-04 the measured shortfall against the S&P was entirely idle cash while stock
    selection was net positive, which is the shape of a result where the answer might be "no" — but
    with a single ledger there is no way to tell, because the analyst's picks and the analyst's
    deployment are the same experiment.

    Deliberately BUY-ONLY. Two reasons, and neither is laziness. Selling to reach a target would make
    this a rebalancer rather than a deployment baseline, and it would inherit exactly the behaviour
    the account already bans for the LLM ("selling to fund another purchase is banned"). An overweight
    group is left alone; the cap in validate_and_fill is what stops one running away.

    Pure: reads the blob, returns a decision, touches nothing. Same {posture, orders} shape the
    analyst returns, so validate_and_fill remains the single authority over both."""
    settings = blob.get("settings") or {}
    positions = blob.get("positions") or []
    cash = float(blob.get("cash") or 0.0)
    targets = (plan or {}).get("targets") or []
    if not targets:
        # An unpriced plan is not "everything is on target" — say which it is.
        return {"posture": "No standing plan — holding.", "orders": []}

    equity = cash + positions_value(positions, price_of)
    if equity <= 0:
        return {"posture": "Unfunded — nothing to allocate.", "orders": []}

    # Reserve whichever is larger: the plan's cash target or the hard floor. The floor is a risk
    # limit and the target is an intention; deploying through either would be the arm overriding its
    # own instructions, which is the one thing a mechanical baseline must never do.
    cash_target_pct = float((plan or {}).get("cash_target_pct") or 0.0)
    reserve_pct = max(cash_target_pct, float(settings.get("cash_floor_pct") or 0.0))
    budget = cash - reserve_pct / 100.0 * equity
    if budget <= 0:
        return {"posture": f"At the {reserve_pct:.0f}% cash reserve — nothing to deploy.", "orders": []}

    group_value: dict[str, float] = {}
    for p in positions:
        px = mark_price(p, price_of)
        if px:
            g = group_of(p["symbol"])
            group_value[g] = group_value.get(g, 0.0) + float(p["shares"]) * px

    # Collapse targets onto their current exposure groups first — a plan written before a regrouping
    # names labels that no longer exist on their own, and two of them can be the same group today.
    want: dict[str, float] = {}
    for t in targets:
        g = group_of(str(t.get("exposure_group") or ""))
        want[g] = want.get(g, 0.0) + float(t.get("target_pct") or 0.0)

    gaps = sorted(((g, pct / 100.0 * equity - group_value.get(g, 0.0)) for g, pct in want.items()),
                  key=lambda kv: -kv[1])
    orders: list[dict] = []
    max_orders = max(1, int(settings.get("max_trades_per_tick") or 4))
    for g, gap in gaps:
        if gap <= 0 or budget <= 0 or len(orders) >= max_orders:
            break
        sym = group_representative(
            g, positions=positions, price_of=price_of, group_of=group_of,
            preferred_btc_etf=str(settings.get("preferred_btc_etf") or "FBTC"))
        if not sym:
            continue
        px = price_of(sym) or 0.0
        dollars = min(gap, budget)
        # Under one share is not worth an order. validate_and_fill would skip it and log a blocked
        # row, and a group within a share of its target every single day would fill the log with
        # noise that reads like a problem.
        if px <= 0 or (dollars < px and not is_crypto(sym)):
            continue
        orders.append({
            "symbol": sym, "side": "buy", "dollars": round(dollars, 2),
            # There is no model here to be confident or otherwise. 100 states plainly that this is
            # mechanical, and keeps it clear of a conviction floor that only means something for a
            # judgement this arm never makes.
            "conviction": 100,
            "reason": (f"{g} is {gap / equity * 100:.1f}pp under its {want[g]:.0f}% target "
                       f"(${gap:,.0f}); mechanical fill, no view taken."),
        })
        budget -= dollars

    if not orders:
        return {"posture": "Every group at or above its target — holding.", "orders": []}
    return {"posture": (f"Mechanically filling {len(orders)} underweight "
                        f"group{'s' if len(orders) != 1 else ''} toward the standing plan."),
            "orders": orders}


def validate_and_fill(
    blob: dict,
    orders: list[dict],
    price_of: Callable[[str], float | None],
    *,
    group_of: Callable[[str], str],
    now_ts: float | None = None,
    source: str = "haiku_tick",
    exclude: set[str] | None = None,
    liquidation: bool = False,
) -> tuple[dict, list[dict], list[dict]]:
    """Apply an analyst order list to the ledger under hard risk limits. Returns (new_blob, filled_rows,
    skipped_rows). Sells run before buys (free cash / cut exposure first). The blob is copied, not mutated
    in place. Raises AssertionError on a cash-conservation violation — the caller must NOT persist then.

    `liquidation=True` lifts the anti-CHURN limits (per-tick trade count and the turnover cap) for an
    exit-date flatten. Those caps exist to stop the strategy trading itself into the ground; a
    liquidation the user scheduled is not churn, and throttling it left the account still holding most
    of the book on the very date they asked to be out — while the response said "flattening to cash".
    Every risk limit that protects the ACCOUNT (cash conservation, shares actually held) still applies."""
    now_ts = now_ts or time.time()
    b = {**blob, "positions": [dict(p) for p in blob.get("positions", [])]}
    s = {**b.get("settings", {})}
    cash0 = round(float(b.get("cash", 0.0)), 2)
    cash = cash0
    positions = b["positions"]

    # Stamp a fresh mark on every position we can price, so `mark_price`'s fallback is yesterday's
    # close rather than a cost basis from months ago if a quote later goes missing.
    for p in positions:
        _px = price_of(p["symbol"])
        if _px and _px > 0:
            p["last_price"] = round(float(_px), 6)

    max_pos_pct = float(s.get("max_position_pct", 20.0))
    cash_floor_pct = float(s.get("cash_floor_pct", 10.0))
    slip = float(s.get("slippage_bps", 5)) / 10_000.0
    min_conv = int(s.get("min_conviction_to_trade", 55))
    max_trades = len(orders) + 1 if liquidation else int(s.get("max_trades_per_tick", 4))
    max_new = int(s.get("max_new_positions_per_tick", 2))
    allow_crypto = bool(s.get("allow_crypto", False))          # direct spot (BTC-USD)
    allow_crypto_etf = bool(s.get("allow_crypto_etf", True))   # spot-crypto ETFs (IBIT/FBTC/FETH)
    allow_etf = bool(s.get("allow_etf", True))  # ETF filtering is best-effort (source-tagged upstream)
    respect_zones = bool(s.get("respect_entry_zones", True))
    # Brokerage realism. CASH accounts settle T+1 (US moved from T+2 on 2024-05-28), so proceeds from
    # today's sales cannot fund today's buys — reusing them is a good-faith violation that a real broker
    # would block. MARGIN accounts may reuse proceeds immediately but are subject to FINRA's pattern-day
    # -trader rule under $25k equity. (The once-a-day cadence means a same-day round trip is structurally
    # impossible, so PDT is currently moot — the guard exists so a faster cadence stays compliant.)
    account_type = str(s.get("account_type", "cash")).lower()
    is_cash_account = account_type != "margin"
    avoid_wash = bool(s.get("avoid_wash_sales", True))
    recent_loss_sales: dict[str, float] = {k.upper(): v for k, v in (blob.get("recent_loss_sales") or {}).items()}
    # Churn control: the max notional (buys+sells) allowed to change hands this tick, as a % of the
    # starting equity. 0 disables the cap. Computed off the pre-trade book so it's a stable budget.
    turnover_pct = 0.0 if liquidation else float(s.get("max_turnover_pct", 0.0) or 0.0)
    start_equity = cash0 + positions_value(positions, price_of)
    turnover_budget = (turnover_pct / 100.0 * start_equity) if turnover_pct > 0 else float("inf")
    traded_notional = 0.0

    filled: list[dict] = []
    skipped: list[dict] = []
    buy_notional = 0.0
    sell_notional = 0.0
    new_positions = 0

    def _skip(o: dict, why: str) -> None:
        skipped.append({"ts": now_ts, "date": today_et_str(), "symbol": o.get("symbol", "").upper(),
                        "side": o.get("side"), "status": "skipped", "shares": 0.0, "price": None,
                        "conviction": o.get("conviction"), "source": source, "reason": o.get("reason", ""),
                        "entry_low": o.get("entry_low"), "entry_high": o.get("entry_high"),
                        "skip_reason": why})

    def _fill(o: dict, side: str, shares: float, price: float, realized: float, cash_after: float,
              pos_after: dict | None, size_note: str | None = None) -> None:
        # `size_note` records any case where the executed size is not the size the analyst asked for.
        # A fill row that shows $346.39 against a $346 order has to say why on its face, or the trade
        # log is quietly answering a question nobody can see it was asked.
        filled.append({
            "ts": now_ts, "date": today_et_str(), "symbol": o["symbol"].upper(), "side": side,
            "status": "filled", "shares": round(shares, 6), "price": round(price, 4),
            "gross": round(shares * price, 2), "cash_after": round(cash_after, 2),
            "avg_cost_after": round(pos_after["avg_cost"], 4) if pos_after else None,
            "realized_pl": round(realized, 2), "exposure_group": group_of(o["symbol"]),
            "conviction": o.get("conviction"), "source": source, "reason": o.get("reason", ""),
            "entry_low": o.get("entry_low"), "entry_high": o.get("entry_high"),
            "size_note": size_note,
        })

    def _side(o: dict) -> str:
        return str(o.get("side") or "").strip().lower()

    # Route bitcoin-ETF buys onto the user's chosen vehicle before any gate runs, so the caps,
    # conviction floor and turnover budget all see the symbol that will actually be bought. Skipped
    # for a liquidation, which is only ever selling.
    if not liquidation:
        orders, _routed = prefer_btc_etf(
            orders, preferred=str(s.get("preferred_btc_etf") or ""),
            positions=positions, price_of=price_of,
        )

    # A sell-and-rebuy inside one exposure group is a wash: same exposure afterwards, two spreads
    # paid, and a realised gain in a taxable account. Dropped BEFORE any gate so neither leg can
    # execute — and recorded as skips, because an order the ledger silently ate is the one thing the
    # audit trail must never miss. Exempt during a liquidation, which is only ever selling.
    if not liquidation:
        swapped = intra_group_swaps(orders, group_of=group_of)
        for i in sorted(swapped):
            o = orders[i]
            _skip(o, f"same-exposure swap within '{group_of(str(o.get('symbol') or ''))}' "
                     f"— sells and re-buys the same thing, paying spread and tax for no change")
        orders = [o for i, o in enumerate(orders) if i not in swapped]

        # ...and the cross-tick version of the same mistake: selling back what an earlier tick bought
        # today. `intra_group_swaps` only sees one order list, so it cannot link two ticks; the
        # position's own `last_add_at` can. Same rationale — spread paid twice, a short-term gain
        # realised against the average cost rather than against this morning's price.
        reversed_today = same_day_reversals(orders, positions, now_ts=now_ts)
        for i in sorted(reversed_today):
            o = orders[i]
            _skip(o, f"{str(o.get('symbol') or '').upper()} was bought earlier today — a same-day "
                     f"reversal pays the spread twice and realises a short-term gain for no change "
                     f"in what the account holds")
        orders = [o for i, o in enumerate(orders) if i not in reversed_today]

    sells = [o for o in orders if _side(o) == "sell"]
    buys = [o for o in orders if _side(o) == "buy"]
    # Anything else was previously dropped before any gate ran: no fill, no skip row, nothing in the
    # trade log or the blocked notes, and `orders_skipped` in the response didn't mention it either.
    # An order the model proposed and the ledger silently ate is the one case the audit trail must
    # never miss, so unrecognised sides are recorded as skips.
    for o in orders:
        if _side(o) not in ("buy", "sell"):
            _skip(o, f"unrecognised side {o.get('side')!r}")

    # ---- SELLS first ----
    for o in sells:
        if len(filled) >= max_trades:
            _skip(o, "max_trades_per_tick reached"); continue
        sym = o["symbol"].upper()
        px = price_of(sym)
        if not px or px <= 0:   # a negative quote would book a sale as a cash outflow
            _skip(o, "no fresh price"); continue
        pos = _find(positions, sym)
        if not pos or pos["shares"] <= 0:
            _skip(o, "not held"); continue
        room = turnover_budget - traded_notional
        if room <= 0:
            _skip(o, f"turnover cap ({turnover_pct:.0f}% of equity) reached"); continue
        want = float(o.get("shares") or 0) or (float(o.get("dollars") or 0) / px)
        want = min(want, room / px)   # clamp the trade to the remaining churn budget
        shares = round_shares(sym, min(want, pos["shares"]))
        if shares <= 0:
            _skip(o, "nothing to sell (or turnover cap left no room)"); continue
        fill = px * (1 - slip)
        traded_notional += shares * fill
        proceeds = shares * fill
        realized = shares * (fill - pos["avg_cost"])
        cash += proceeds
        sell_notional += proceeds
        pos["shares"] = round(pos["shares"] - shares, 8)
        b["realized_pl_total"] = round(float(b.get("realized_pl_total", 0.0)) + realized, 2)
        if realized < 0:              # a loss sale starts the 30-day wash-sale clock for this name
            recent_loss_sales[sym] = now_ts
        _fill(o, "sell", shares, fill, realized, cash, pos)
        if pos["shares"] <= 1e-9:
            positions.remove(pos)

    # ---- BUYS by conviction ----
    # Equity + per-group values are pinned AFTER sells so caps use one stable denominator for the tick.
    pv_after_sells = positions_value(positions, price_of)
    equity = round(cash + pv_after_sells, 2)
    floor = cash_floor_pct / 100.0 * equity
    # T+1: in a cash account today's sale proceeds are UNSETTLED and can't fund today's buys.
    # T+1 settlement, tracked ON THE LEDGER rather than scoped to this call.
    #
    # It used to be `cash - sell_notional`, which only knew about sells made in THIS invocation — so
    # a second tick the same afternoon (the app's "run tick" button always sends force=true) saw
    # those proceeds as fully settled and spent them. That is a good-faith violation a real cash
    # account would block, and the only thing preventing it was that nobody usually ticked twice.
    today = dt.datetime.fromtimestamp(now_ts, ET).date()
    unsettled = [
        u for u in (b.get("unsettled") or [])
        if str(u.get("settles_on") or "") > today.isoformat()
    ]
    if is_cash_account and sell_notional > 0:
        unsettled.append({
            "amount": round(sell_notional, 2),
            "settles_on": market_calendar.next_trading_day(today).isoformat(),
        })
    b["unsettled"] = unsettled
    unsettled_total = sum(float(u.get("amount") or 0.0) for u in unsettled)
    buying_power = (cash - unsettled_total) if is_cash_account else cash
    # Same marks as the equity figure above — if these two ever diverge, the cap is measured against
    # a different book than the one it is capping.
    group_value: dict[str, float] = {}
    for p in positions:
        px = mark_price(p, price_of)
        if px:
            g = group_of(p["symbol"])
            group_value[g] = group_value.get(g, 0.0) + p["shares"] * px

    for o in sorted(buys, key=lambda x: -int(x.get("conviction") or 0)):
        if len(filled) >= max_trades:
            _skip(o, "max_trades_per_tick reached"); continue
        sym = o["symbol"].upper()
        if exclude and (sym in exclude or sym.removesuffix("-USD") in exclude):
            _skip(o, "excluded ticker"); continue
        if is_crypto(sym) and not allow_crypto:
            _skip(o, "direct spot crypto disabled (use the ETF)"); continue
        # A non "-USD" symbol whose exposure group is BTC/ETH is a spot-crypto ETF (IBIT/FBTC/FETH…).
        if not is_crypto(sym) and group_of(sym) in ("BTC", "ETH") and not allow_crypto_etf:
            _skip(o, "crypto ETFs disabled"); continue
        if int(o.get("conviction") or 0) < min_conv:
            _skip(o, f"below conviction floor ({min_conv})"); continue
        px = price_of(sym)
        # `not px` already rejects None and 0.0; the `<= 0` is for a NEGATIVE quote, which a bad feed
        # can produce and which every downstream comparison here mis-handles — a negative `fill` makes
        # the round-up guard trivially true and turns `cash -= cost` into a deposit.
        if not px or px <= 0:
            _skip(o, "no fresh price"); continue
        # Entry-zone discipline: don't chase. If the analyst named a zone and the market is above its
        # top, defer the buy — it stays a candidate on later ticks instead of filling at any price.
        # (Below the zone is fine: cheaper than it wanted to pay.)
        if respect_zones:
            hi = o.get("entry_high")
            try:
                hi = float(hi) if hi is not None else None
            except (TypeError, ValueError):
                hi = None
            if hi and hi > 0 and px > hi:
                _skip(o, f"price {px:.2f} above entry zone (≤{hi:.2f}) — waiting"); continue
        # Wash-sale guard (IRS §1091): rebuying a name sold at a LOSS within 30 days disallows the loss.
        if avoid_wash and sym in recent_loss_sales:
            days = (now_ts - float(recent_loss_sales[sym])) / 86_400.0
            if days < 30:
                _skip(o, f"wash-sale window ({30 - int(days)}d left since the loss sale)"); continue
        available = min(cash, buying_power) - floor
        if available <= 0:
            # Same rule as the ladder below: blame T+1 only when settling would actually fix it.
            # Unsettled proceeds being present is not enough — if the cash floor swallows the balance
            # even fully settled, next session is identical and "funds free next session" is a
            # forecast that never comes true, repeated daily.
            settles_into_a_share = buying_power < cash - 0.01 and cash - floor >= px * (1 + slip)
            _skip(o, "unsettled proceeds (T+1) — funds free next session"
                  if settles_into_a_share else "at cash floor")
            continue
        g = group_of(sym)
        cap_room = max_pos_pct / 100.0 * equity - group_value.get(g, 0.0)
        if cap_room <= 0:
            _skip(o, f"exposure '{g}' at {max_pos_pct:.0f}% cap"); continue
        is_new = _find(positions, sym) is None
        if is_new and new_positions >= max_new:
            _skip(o, "max_new_positions_per_tick reached"); continue
        room = turnover_budget - traded_notional
        if room <= 0:
            _skip(o, f"turnover cap ({turnover_pct:.0f}% of equity) reached"); continue
        # Size from `dollars`, else from an explicit `shares` count, else skip. The old
        # `float(dollars or 0) or available` fallback silently reinterpreted a zero/absent/null
        # `dollars` as "spend everything not behind the cash floor" — the largest possible order
        # from the weakest possible instruction, and `shares` was never consulted on a buy at all.
        fill = px * (1 + slip)
        want_dollars = float(o.get("dollars") or 0.0)
        if want_dollars <= 0:
            want_shares = float(o.get("shares") or 0.0)
            if want_shares > 0:
                # Budget at the FILL price, not the raw quote — costing 10 shares at `px` leaves
                # slippage unfunded, so `spend / fill` lands just under 10 and floors to 9.
                want_dollars = want_shares * fill
            else:
                _skip(o, "buy order specified neither dollars nor shares"); continue
        headroom = min(available, cap_room, room)
        spend = min(want_dollars, headroom)
        size_note: str | None = None
        if is_crypto(sym):
            shares = round_shares(sym, spend / fill)
        else:
            # Rescue the last share — see ROUND_UP_MIN_SHARE. Rounding up is only ever safe when the
            # UPSIZED cost still clears every limit, which is what `(shares + 1) * fill <= headroom`
            # asserts; because `spend` collapses to `headroom` whenever a limit is the binding term,
            # that test also fails automatically for any order a limit already cut. The rescue can
            # therefore only fire when the analyst's own number was the constraint — the one case
            # where honouring it cannot breach anything.
            exact = spend / fill
            shares = float(int(exact))
            if exact - shares >= ROUND_UP_MIN_SHARE and (shares + 1) * fill <= headroom:
                shares += 1
                size_note = (f"analyst asked for {_usd(want_dollars)} ({exact:.3f} shares); "
                             f"rounded up to {int(shares)} at ${fill:,.2f}")
        if shares <= 0:
            # Name the constraint that actually bound. "cash/cap/turnover" covered three unrelated
            # causes, so the blocked-trade log could not answer the one question it exists for —
            # and a T+1 hold looked identical to simply being out of money.
            #
            # Two rules keep this honest. First, `fill <= headroom` decides IF a limit was involved
            # at all: a whole share fitting inside every limit means nothing but the order's own size
            # stopped it. Testing the order size directly instead gets this wrong in both directions
            # — a too-small order matched no branch and the old `else` blamed the cash floor
            # regardless (measured 2026-08-03: a VTI buy logged as "cash floor left room for less
            # than one share" while cash held $5,000 against a $534 floor, thirteen shares' worth),
            # and a full-size order under a tight cap read as "no limit was binding" when the cap was
            # the whole story. Second, WHICH limit gets named is decided by equality against
            # `headroom`, so it is the TIGHTEST one — the constraint that would still bind if the
            # others were lifted — rather than whichever branch happens to be tested first.
            #
            # Blocked reasons feed the weekly strategy review, so a wrong one teaches the strategist
            # that the wrong constraint is binding and it re-plans around a limit that was never in
            # the way. They are also GROUPed BY in memory.blocked_summary(), so the cap and turnover
            # strings stay free of per-event dollar figures — a varying number in there splits one
            # recurring problem into a column of counts of 1.
            if fill <= headroom:
                why = (f"order was for {_usd(want_dollars)}, under one share at ${fill:,.2f} "
                       f"— no limit was binding")
            # T+1 is only the answer if settling WOULD fix it. `available` is already net of the cash
            # floor, so an account short of a share on the floor alone is short of one regardless of
            # settlement — and "frees up next session" is then a promise the next session cannot keep.
            elif (available == headroom and is_cash_account and unsettled_total > 0.01
                  and cash - floor >= fill):
                why = "unsettled proceeds (T+1) — frees up next session"
            elif available == headroom:
                why = (f"cash floor ({cash_floor_pct:.0f}% of equity) left "
                       f"under one share at ${fill:,.2f}")
            elif cap_room == headroom:
                why = f"exposure '{g}' cap left room for less than one share"
            else:
                why = f"turnover cap ({turnover_pct:.0f}% of equity) left room for less than one share"
            _skip(o, why); continue
        cost = shares * fill
        traded_notional += cost
        cash -= cost
        buying_power -= cost
        buy_notional += cost
        group_value[g] = group_value.get(g, 0.0) + cost
        pos = _find(positions, sym)
        if pos:
            tot = pos["shares"] + shares
            pos["avg_cost"] = (pos["shares"] * pos["avg_cost"] + shares * fill) / tot
            pos["shares"] = round(tot, 8)
            pos["last_add_at"] = now_ts
        else:
            pos = {"symbol": sym, "shares": round(shares, 8), "avg_cost": fill,
                   "exposure_group": g, "opened_at": now_ts, "last_add_at": now_ts}
            positions.append(pos)
            new_positions += 1
        _fill(o, "buy", shares, fill, 0.0, cash, pos, size_note)

    # Keep the wash-sale clock on the ledger, pruned to the 30-day window it governs.
    b["recent_loss_sales"] = {k: v for k, v in recent_loss_sales.items() if now_ts - float(v) < 31 * 86_400}

    # Cash-conservation invariant (fail-closed): the caller aborts + does not persist on violation.
    #
    # Checked on the RAW accumulators, not on two independently-rounded values. The previous form
    # compared round(cash,2) - cash0 against round(sell_notional - buy_notional, 2); when the exact
    # figure sat on a half-cent boundary those were both legal roundings of the SAME number that
    # differ by exactly 0.01, and the strict `< 0.01` then aborted a perfectly correct tick.
    # Slippage manufactures that third decimal and real quotes are round numbers, so the boundary is
    # hit often — measured at 1.6% of ticks, each one a lost trading day, a hole in the equity curve,
    # and once a month an orphaned deposit row already appended to the append-only log. `cash` and
    # the notionals accumulate the same floats, so the true invariant is exact to float noise.
    drift = abs(cash - (cash0 + sell_notional - buy_notional))
    assert drift < 1e-6, (
        f"cash not conserved: cash={cash!r} vs cash0+sells-buys="
        f"{cash0 + sell_notional - buy_notional!r} (drift={drift!r})"
    )
    b["cash"] = round(cash, 2)
    return b, filled, skipped


def nav_row(blob: dict, *, positions_val: float, spy_price: float | None, now_ts: float | None = None) -> dict:
    """One equity-curve point: total equity (cash + marked positions) + the benchmark's shadow value."""
    now_ts = now_ts or time.time()
    cash = round(float(blob.get("cash", 0.0)), 2)
    equity = round(cash + positions_val, 2)
    bench = blob.get("benchmark") or {}
    bench_val = round(float(bench.get("shares", 0.0)) * spy_price, 2) if spy_price else None
    return {
        "ts": now_ts, "date": today_et_str(), "equity": equity, "cash": cash,
        "positions_value": round(positions_val, 2), "funded_total": round(float(blob.get("funded_total", 0.0)), 2),
        "benchmark_symbol": bench.get("symbol", "^GSPC"), "benchmark_value": bench_val,
        "num_positions": len(blob.get("positions", [])),
    }
