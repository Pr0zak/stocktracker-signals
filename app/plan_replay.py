"""SWT-8 — replay a recorded plan against the bars that actually came next.

Three things in this service already score a decision, and none of them scores the user's. memory.py
scores what the ANALYST said, against a fixed 20-day forward return with no stop and no target. The
sandbox scores what the PAPER TRADER did, in its own ledger with its own money. The options tracker
scores real option positions and nothing else. So "did following these plans actually work" has no
data behind it, and the honest answer today is that nobody knows.

This module is the MECHANICAL half of the answer: given a plan exactly as it stood on a date — an
entry zone, a stop, a target — walk the daily bars that followed and say what the plan did. The
journal's other half records what the USER did with the same plan. Two curves, and the gap between
them is the part no other number in this app can show.

Pure: no I/O, no httpx, no network, no numpy/pandas — the same register as app/swing.py, app/chase.py
and app/percentiles.py, so every branch below is testable from a handful of synthetic bars.

TWO RULES GOVERN EVERY DECISION HERE, and both exist because a backtest's natural failure mode is to
flatter itself:

  1. THE INTRABAR AMBIGUITY RESOLVES AGAINST THE TRADE. A daily bar whose low reached the stop AND
     whose high reached the target cannot say which came first. We assume the STOP, and we set
     `ambiguous: true` so the assumption is visible on the row it was load-bearing for. A backtest
     that silently resolves its own ambiguities in its favour is the single most common way one
     flatters itself, and the flattery is invisible afterwards: the curve just looks good.

  2. ABSENT IS NEVER ZERO. A plan with no usable stop has no risk denominator, so its R is None —
     not 0. 0R is a real claim ("it made exactly what it risked"), and a plan that never filled, or
     is still running, has made no claim at all. Same for a still-open plan: no exit price, no
     fabricated mark in the exit field.

Long-only. Every plan this service produces is a buy (app/analyst.py's PlanAction is
buy_now / buy_on_pullback / wait / avoid), and the stop-below / target-above arithmetic below is
written for that direction. A short plan handed to `replay` is refused rather than silently
replayed upside-down.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Sequence

log = logging.getLogger("signals.replay")

# How many sessions a filled plan is given to reach one of its own levels.
#
# The reference this came from closes its replays at ten trading days, and ten days is the wrong
# instrument for THIS app. The plan prompt (app/analyst.py:233) asks for a stop below the zone and a
# target above it at a risk:reward of ~1.5, which in practice is the ~5%-down / ~8%-up pair
# app/chase.py works through. A liquid large-cap moving ~1.5% a day does not travel 8% in ten
# sessions except on a catalyst, so a ten-day clock would resolve most plans as `time` — and a curve
# where nearly every trade times out is measuring the CLOCK, not the plan.
#
# 20 was the other candidate, for comparability with memory.py's fwd_20d. Rejected: fwd_20d is a
# fixed-window RETURN with no stop and no target, where this horizon is a DEADLINE for two levels to
# be reached. Borrowing the number would suggest the two are comparable when they measure different
# things.
#
# 40 sessions is about two calendar months. Long enough that a 5/8 pair of levels genuinely can be
# reached by ordinary drift rather than only by news; short enough that `time` still means "this
# plan went nowhere" rather than "we got bored"; and under the ~63 sessions of an earnings cycle, so
# a replay window cannot straddle two reports and quietly become a study of earnings. It is a
# PARAMETER, not a constant of nature — the whole point of naming it is that the answer is not baked
# in and a caller can ask the same plan a different question.
DEFAULT_HORIZON_DAYS = 40

# How long the entry order is left working before the plan is declared never-taken.
#
# A plan is not just its levels, it is its levels ON A DATE: the entry zone was drawn against that
# day's RSI, that day's moving averages, that week's market. Leaving the limit order open for the
# full holding horizon would let a zone touched seven weeks later count as "the plan filled", when
# by then it is a different trade wearing the plan's clothes — and it would flatter the curve, since
# the plans that eventually get filled that way are disproportionately the ones that fell far enough
# to come back. One trading week is the compromise: long enough for a buy_on_pullback plan to
# actually get its pullback, short enough that the setup the plan was written against still exists.
DEFAULT_FILL_WINDOW_DAYS = 5

# Outcomes. `open` and `never_filled` are NOT failures and NOT flat trades — they are the two ways a
# plan can have produced no result yet, and they are deliberately distinct from a 0R stop-out.
TARGET = "target"
STOP = "stop"
TIME = "time"
OPEN = "open"
NEVER_FILLED = "never_filled"


def norm_date(d: Any) -> str:
    """`d` as the bare YYYYMMDD that Series.dates uses. Raises on anything that is not a date.

    Tolerates "2026-08-21" and datetime.date because both turn up at call sites. Deliberately NOT
    reused from scan_store._norm_d, which is otherwise identical: importing that module here would
    drag sqlite3 and a database path into a module whose whole value is that it has neither.
    """
    digits = "".join(c for c in str(d or "") if c.isdigit())
    if len(digits) < 8:
        raise ValueError(f"not a YYYYMMDD date: {d!r}")
    return digits[:8]


def _num(v: Any) -> float | None:
    """A usable finite float, or None. NaN and +/-inf are ABSENT, not values — a NaN compares False
    against every threshold below and would hand back an outcome derived from garbage."""
    if isinstance(v, bool):  # True would arithmetic as 1.0 and become a $1 level
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _level(v: Any) -> float | None:
    """A price level, where zero and negatives mean ABSENT rather than "a level at zero".

    This is the chase.py incident in another costume: the app's EntryPlan once modelled these as
    non-nullable Doubles, so a null decoded to 0.0 and shipped as a confident "Stop $0 - target $0"
    onto a money decision. A 0.0 arriving here is overwhelmingly that null, not a plan to stop out at
    zero, and treating it as a level would make every trade look like it ran forever without risk.
    """
    f = _num(v)
    return f if (f is not None and f > 0) else None


def _zone(entry: Any) -> tuple[float | None, float | None]:
    """The entry as a (low, high) zone. A scalar is a zone of zero width — a single limit price.

    Accepts what the callers actually hold: a bare number, a 2-sequence, or the plan dict itself
    (entry_low/entry_high, or low/high). Half a zone is usable — the TOP is the limit price that
    decides whether the plan ever filled, and a missing bottom only costs the sanity check on the
    stop.
    """
    if isinstance(entry, dict):
        lo = _level(entry.get("entry_low", entry.get("low")))
        hi = _level(entry.get("entry_high", entry.get("high")))
    elif isinstance(entry, (list, tuple)):
        if len(entry) != 2:
            return (None, None)
        lo, hi = _level(entry[0]), _level(entry[1])
    else:
        px = _level(entry)
        lo = hi = px
    if lo is not None and hi is None:
        hi = lo          # only a floor named: treat it as the single price the plan would pay
    if hi is not None and lo is None:
        lo = hi          # only a ceiling named: the widest honest reading is a limit at the top
    return (lo, hi)


def _bar(b: Any) -> tuple[str | None, float, float, float, float] | None:
    """One (date, open, high, low, close) tuple, or None if the bar cannot be walked.

    A bar missing any of its four prices is not repaired and not guessed at: it is dropped and
    COUNTED, and `bars_skipped` on the result says how many were. Yahoo nulls the odd session, and
    silently walking past those holes would let a replay claim it saw a window it only partly saw.
    A high below its own low is corrupt in the same way and goes out the same door.
    """
    if isinstance(b, dict):
        d = b.get("date", b.get("d"))
        o, h, l, c = (b.get("open"), b.get("high"), b.get("low"), b.get("close"))
    else:
        d = getattr(b, "date", None)
        o, h, l, c = (getattr(b, "open", None), getattr(b, "high", None),
                      getattr(b, "low", None), getattr(b, "close", None))
    o, h, l, c = _num(o), _num(h), _num(l), _num(c)
    if None in (o, h, l, c):
        return None
    if h < l or h <= 0 or l <= 0:
        return None
    return (str(d) if d is not None else None, o, h, l, c)


def bars_from_series(series: Any, plan_date: Any) -> list[dict]:
    """The daily bars a plan recorded on `plan_date` could actually have traded.

    STRICTLY AFTER the plan date, and that is the whole point of the helper. A plan is written from
    a session's closing snapshot — market.summarize() reports the LAST bar — so letting the replay
    trade that same session's range is lookahead: the plan would be filling inside a day whose high,
    low and close were already known to the analyst that drew the levels. One free bar of hindsight
    per trade is more than enough to invent an edge.

    Every field is read through getattr, the way swing.py reads a Series, so a truncated or stubbed
    series degrades to a shorter bar list instead of raising an AttributeError from inside a route.
    `Series.highs`/`.lows` are on the SPLIT-ADJUSTED basis (see the rescaling comment in
    market.fetch_series), which is what makes it legal to compare them against a plan's levels at all.
    """
    d0 = norm_date(plan_date)
    dates = list(getattr(series, "dates", None) or [])
    opens = list(getattr(series, "opens", None) or [])
    highs = list(getattr(series, "highs", None) or [])
    lows = list(getattr(series, "lows", None) or [])
    closes = list(getattr(series, "closes", None) or [])

    def _at(seq: Sequence, i: int) -> Any:
        return seq[i] if i < len(seq) else None

    out: list[dict] = []
    for i, d in enumerate(dates):
        if str(d) <= d0:
            continue
        out.append({"date": str(d), "open": _at(opens, i), "high": _at(highs, i),
                    "low": _at(lows, i), "close": _at(closes, i)})
    return out


def _blank(horizon_days: int, fill_window_days: int) -> dict:
    """Every key present on every path, None where unknown.

    Present-and-null beats missing for the same reason chase.annotate attaches all four of its keys:
    a consumer that sees `r: null` knows the R could not be computed, where a missing `r` is
    ambiguous with an older server that never computed one.
    """
    return {
        "outcome": None,
        "reason": None,
        "refused": False,      # True = the PLAN was unusable (a caller error), not the market
        "ambiguous": False,
        "entry_price": None,
        "entry_date": None,
        "exit_price": None,
        "exit_date": None,
        "bars_held": None,
        "r": None,
        "return_pct": None,
        "mark_price": None,    # last observed close, for OPEN plans only — NOT an exit
        "mark_date": None,
        "horizon_days": horizon_days,
        "fill_window_days": fill_window_days,
        "bars_seen": 0,
        "bars_skipped": 0,
    }


def replay(
    bars: Sequence[Any],
    *,
    entry: Any,
    stop: Any = None,
    target: Any = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    fill_window_days: int = DEFAULT_FILL_WINDOW_DAYS,
) -> dict:
    """What a plan would have done, walked bar by bar over the sessions that followed it.

    `bars` must begin with the first session the plan could TRADE — the one after it was recorded.
    `bars_from_series` does that slicing; see the lookahead note there.

    Returns a dict, never raises, and never invents a price. `outcome` is one of:

      * `"target"` / `"stop"` — a level was reached. `exit_price` is the level, or the OPEN when the
        session gapped straight through it, because a stop is a market order once it triggers and
        pretending it filled at the stop price on a gap-down is exactly the kind of small,
        systematic optimism that adds up to a fictional curve.
      * `"time"`  — the horizon expired with the plan still in; exit at that session's close.
      * `"open"`  — the horizon has NOT expired yet. No exit price. `mark_price` carries the last
        close so a reader can see where it stands, under a name that cannot be mistaken for a fill.
      * `"never_filled"` — price never traded into the entry zone inside the fill window (SWT-3's
        chase problem, seen after the fact), or it opened through the stop before the plan could be
        entered. No entry, no R, and emphatically not a 0R trade.
      * `None` with `refused: true` — the plan itself was unusable; `reason` says which part.
      * `None` with `refused: false` — nothing has been DECIDED yet: no bars have traded since the
        plan date, or the entry has not filled and its fill window has not closed. Neither is an
        outcome, and in particular neither is `never_filled`, which is a claim about days that in
        this case have not happened.

    `r` is (exit - entry) / (entry - stop): the outcome in units of the risk the plan defined. It is
    None whenever no usable stop was named, because no risk defined means no R — not 0R.
    """
    hd = _int(horizon_days)
    fw = _int(fill_window_days)
    out = _blank(hd if hd is not None else horizon_days, fw if fw is not None else fill_window_days)
    if hd is None or hd < 1:
        return {**out, "refused": True,
                "reason": f"horizon_days must be a whole number of sessions >= 1, got {horizon_days!r}"}
    if fw is None or fw < 1:
        return {**out, "refused": True,
                "reason": f"fill_window_days must be a whole number of sessions >= 1, got {fill_window_days!r}"}

    lo, hi = _zone(entry)
    st, tg = _level(stop), _level(target)

    if hi is None:
        return {**out, "refused": True,
                "reason": "the plan names no entry level — there is nothing to have been filled at"}
    if lo is not None and lo > hi:
        # chase.py refuses an inverted zone for the same reason: the model swapped its own two
        # numbers, so neither one means what it says, and every level test below would still
        # "work" while measuring against the wrong end.
        return {**out, "refused": True,
                "reason": f"entry zone is inverted (low {lo:g} above high {hi:g})"}
    if st is not None and st >= lo:
        # A stop at or inside the zone is not a stop: a fill at the bottom of the zone would already
        # be under it, and R's denominator would come out zero or negative.
        return {**out, "refused": True,
                "reason": f"stop {st:g} is not below the entry zone ({lo:g}-{hi:g}) — it is not a stop"}
    if tg is not None and tg <= hi:
        # Long-only (see the module docstring): a target at or under the entry is either a short
        # plan or a typo, and replaying it as a long would report a guaranteed instant winner.
        return {**out, "refused": True,
                "reason": f"target {tg:g} is not above the entry zone ({lo:g}-{hi:g}) — this replay is long-only"}

    walk: list[tuple[str | None, float, float, float, float]] = []
    skipped = 0
    for b in bars or ():
        nb = _bar(b)
        if nb is None:
            skipped += 1
            continue
        walk.append(nb)
    out["bars_seen"] = len(walk)
    out["bars_skipped"] = skipped
    if skipped:
        # Reported in the payload AND logged: a replay walked over holes is still a real answer, but
        # a symbol that quietly drops sessions every time is a data problem, not a trading one.
        log.debug("replay: dropped %d unusable bar(s) of %d", skipped, skipped + len(walk))
    if not walk:
        # Not a refusal — the plan is fine, the tape just has not happened yet (a plan recorded
        # today) or every bar was unusable. Either way there is no outcome, and no outcome is
        # reported rather than an "open" that would imply a position exists.
        return {**out, "reason": ("no usable bars to replay against"
                                  + (f" ({skipped} unusable)" if skipped else ""))}

    # ---- entry -------------------------------------------------------------------------------
    # A buy limit resting at the TOP of the zone. It fills on the first session that trades down to
    # it; the fill is the open when the session gapped below the limit, since a resting order fills
    # at the better price, and the limit itself otherwise.
    fill_i: int | None = None
    fill_px = 0.0
    for i, (d, o, h, l, c) in enumerate(walk[:fw]):
        if l > hi:
            continue  # never traded down into the zone this session
        if st is not None and o <= st:
            # The session opened at or through the plan's own stop. Filling here would report a
            # trade that was entered and stopped in the same instant — a ~0R that looks like a
            # controlled loss when what actually happened is that the setup was void before the
            # bell. Say the plan was never taken, and say why.
            # entry_date is deliberately left None here: a date in that field beside a null
            # entry_price reads as a half-recorded fill. The session is named in the reason instead.
            return {**out, "outcome": NEVER_FILLED,
                    "reason": (f"session {d} opened at {o:g}, at or below the plan's stop {st:g}, before "
                               "the entry could fill — the plan was void, not entered")}
        fill_i, fill_px = i, min(o, hi)
        break
    if fill_i is None:
        if len(walk) < fw:
            # The order is still working. `never_filled` is a CLAIM — that the market never offered
            # this plan — and it cannot be made from a window that has not closed yet: the zone may
            # well be touched tomorrow. Same shape as the no-bars case: no outcome, and a reason.
            return {**out, "reason": (f"the entry zone ({lo:g}-{hi:g}) has not been touched in "
                                      f"{len(walk)} of {fw} session(s) — the fill window is still open")}
        return {**out, "outcome": NEVER_FILLED,
                "reason": (f"price never traded into the entry zone ({lo:g}-{hi:g}) within "
                           f"{fw} session(s) — the plan was never tradeable")}

    d_in = walk[fill_i][0]
    out.update(entry_price=round(fill_px, 4), entry_date=d_in)
    risk = (fill_px - st) if st is not None else None

    # ---- management --------------------------------------------------------------------------
    end = min(len(walk), fill_i + hd)
    for j in range(fill_i, end):
        d, o, h, l, c = walk[j]
        first = j == fill_i
        # On the fill bar the open is BEHIND US — it happened before the limit filled — so the
        # gap rules below must not be applied to it. Everywhere else the open is the first price of
        # the session and settles the ordering that the bar's high and low cannot.
        if not first and st is not None and o <= st:
            return _exit(out, STOP, o, d, fill_i, j, fill_px, risk, ambiguous=False)
        if not first and tg is not None and o >= tg:
            return _exit(out, TARGET, o, d, fill_i, j, fill_px, risk, ambiguous=False)
        hit_stop = st is not None and l <= st
        hit_target = tg is not None and h >= tg
        if hit_stop and hit_target:
            # THE CRUX. One daily bar, both levels touched, and no way to know the order — the
            # session may have run to the target and then reversed, or stopped out and rallied. We
            # take the STOP, the pessimistic reading, and flag the row so that a curve built on
            # these can be inspected for how much of it rests on the assumption. The alternative
            # (take the target, say nothing) is the mistake this whole module is written against.
            return _exit(out, STOP, st, d, fill_i, j, fill_px, risk, ambiguous=True)
        if hit_stop:
            return _exit(out, STOP, st, d, fill_i, j, fill_px, risk, ambiguous=False)
        if hit_target:
            return _exit(out, TARGET, tg, d, fill_i, j, fill_px, risk, ambiguous=False)

    last_d, _o, _h, _l, last_c = walk[end - 1]
    if end == fill_i + hd:
        # The clock ran out with the plan still in. This is a real, measurable outcome — the plan
        # went nowhere — so it exits at the last in-horizon close.
        return _exit(out, TIME, last_c, last_d, fill_i, end - 1, fill_px, risk, ambiguous=False)
    # Still running: fewer bars exist than the horizon allows. No exit price, because there was no
    # exit; `mark_price` says where it stands without pretending to be a fill, and `r` stays None
    # because an unrealized R is not an R.
    return {**out, "outcome": OPEN, "bars_held": end - fill_i,
            "mark_price": round(last_c, 4), "mark_date": last_d,
            "reason": (f"still open — {end - fill_i} of {hd} session(s) elapsed and neither level "
                       "has been reached")}


def _exit(out: dict, outcome: str, price: float, date: str | None,
          fill_i: int, exit_i: int, fill_px: float, risk: float | None, *, ambiguous: bool) -> dict:
    """One resolved trade. `bars_held` counts the sessions the position EXISTED IN, so a plan that
    filled and stopped inside the same session held 1 bar, not 0 — zero would read as never having
    been on."""
    r = None
    if risk is not None and risk > 0:
        r = round((price - fill_px) / risk, 3)
    return {
        **out,
        "outcome": outcome,
        "ambiguous": ambiguous,
        "exit_price": round(price, 4),
        "exit_date": date,
        "bars_held": exit_i - fill_i + 1,
        "r": r,
        "return_pct": round((price / fill_px - 1.0) * 100.0, 3) if fill_px else None,
    }


def _int(v: Any) -> int | None:
    """A whole number, or None. Floats that are not whole are refused rather than truncated: a
    horizon of 20.7 sessions is a caller bug, and silently walking 20 hides it."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    f = _num(v)
    if f is None or f != int(f):
        return None
    return int(f)
