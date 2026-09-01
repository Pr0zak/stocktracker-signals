"""SWT-3 — say out loud when the user is ABOUT TO CHASE an entry.

The plan already carries an entry zone and the detail screen already shows the live price, and that
was exactly the problem: "entry $68.67-$70.73" printed a few millimetres from "$74.02" is a warning
only to a reader who does the division. Nobody does the division on a phone, in a hurry, with the
buy button on the same screen. This module does it server-side and NAMES the answer, so the screen
can say "chasing" instead of leaving two numbers lying next to each other.

Derived, never asked of the model. The analyst commits to the zone — that is a judgement. Dividing a
live quote by it is not, and a model asked to do the arithmetic will occasionally get it wrong in the
same confident tone it uses for the parts it does know.

ABSENT IS NEVER ZERO. Every input here is allowed to be missing: the analyst may decline to name a
zone, the quote feed drops symbols, and a zone can come back inverted. Each of those makes the read
UNKNOWABLE, and an unknowable chase read is None — never 0.0, never "ok". The app models these
levels as nullable Doubles for precisely this reason: they were once non-nullable and defaulted to
0.0, and the app shipped a confident "Stop $0 · target $0" onto a money decision.
"""
from __future__ import annotations

import math
from typing import NamedTuple

# How far above the TOP of the entry zone still reads as "ok" rather than a chase, in percent.
#
# This band is not permission to pay up — it is the width of the measurement error. The quote the app
# renders is delayed, the fill is a later print, and an ordinary liquid name wobbles around a percent
# through a session; a boundary much under 1% would fire on that noise alone and teach the reader to
# swipe past the one warning that mattered.
#
# It cannot be much wider either, because overshooting the zone damages the plan from both ends at
# once: every percent paid above the top is a percent of reward given up AND a percent of risk taken
# on. For a plan built to the ~1.5 risk:reward the plan prompt demands — stop ~5% under the zone top,
# target ~8% over it — paying x% over the top leaves (8-x)/(5+x): 1.17 at x=1, 0.86 at x=2, 0.63 at
# x=3. The trade the user believed they were taking has stopped existing somewhere around 2%.
#
# 1.5 sits between those two costs. Being wrong on the low side costs a needless caution on a trade
# that was fine and the user pays it no mind; being wrong on the high side costs SILENCE on a trade
# whose entire edge has already been paid away at the open. The second is much the more expensive
# error, so the band is set at the tight end of the noise range rather than the middle of it.
#
# The sandbox has no band at all — it defers any buy priced above `entry_high` (see sandbox_job.py).
# It can afford that: it is a machine, and waiting a day costs it nothing. A person looking at a
# screen needs the difference between "a few cents over" and "you are chasing this".
CHASE_OK_PCT = 1.5

BELOW_ZONE = "below_zone"
IN_ZONE = "in_zone"
OK = "ok"
TOO_DEEP = "chase_too_deep"


class ChaseRead(NamedTuple):
    """`pct` above the zone top (negative = under it), a status, and a warning if the zone is broken.

    pct and status are independently nullable and are BOTH None whenever the read cannot be taken.
    `warning` is only for a zone that is malformed — a plain absence (no zone, no price) is not a
    defect worth telling the user about, it just means there is nothing to show.
    """

    pct: float | None
    status: str | None
    warning: str | None


_ABSENT = ChaseRead(None, None, None)


def _num(v) -> float | None:
    """A usable finite float, or None. NaN and ±inf are ABSENT, not values — they compare in ways
    that would quietly hand back a status derived from garbage."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def read(price, entry_low, entry_high) -> ChaseRead:
    """Where a live price sits relative to the analyst's own entry zone.

    Inputs are deliberately untyped and coerced here: they arrive from a model's JSON, from a quote
    feed, or from a cached payload, and any of them can be None, a string, or a NaN.
    """
    px = _num(price)
    hi = _num(entry_high)
    lo = _num(entry_low)

    # No price = no read. Not "in zone", not 0% — we simply do not know where it is trading.
    if px is None or px <= 0:
        return _ABSENT
    # No zone top = no read. This is the case the whole task exists for: the analyst declining to
    # commit to a level must NOT come out the other side as a confident "ok, not chasing".
    # A non-positive top is a broken level, not a zone anything can be above or below.
    if hi is None or hi <= 0:
        return _ABSENT
    # An inverted zone means the model swapped its own two numbers, so neither one means what it
    # says. Percentages computed off it look perfectly reasonable and are nonsense, so say so
    # instead. (The /plan route sanitizes inversions before this runs; other callers may not.)
    if lo is not None and lo > hi:
        return ChaseRead(
            None, None,
            f"entry zone is inverted (low {lo:g} above high {hi:g}) — no chase read from it",
        )

    # Rounded BEFORE the comparison so the status can never contradict the number shown beside it:
    # a card reading "1.5% above the zone" labelled chase_too_deep is its own little lie.
    pct = round((px / hi - 1.0) * 100.0, 2)
    if px > hi:
        return ChaseRead(pct, OK if pct <= CHASE_OK_PCT else TOO_DEEP, None)
    if lo is None:
        # Top known, bottom not: we can still say how far under the top it is, but "in the zone" and
        # "cheaper than the zone" are indistinguishable without the floor. Neither is a warning, so
        # the honest answer is a number with no label rather than a guessed label.
        return ChaseRead(pct, None, None)
    return ChaseRead(pct, BELOW_ZONE if px < lo else IN_ZONE, None)


def annotate(payload: dict, price) -> dict:
    """A copy of a /plan payload with the chase read attached, read off `payload["plan"]`.

    Always attaches all four keys, None included: a client that sees `chase_status` present and null
    knows the read was attempted and could not be taken, where a missing key is ambiguous with an
    older server. `chase_price` is the quote the read was taken at, so a screen can show its own
    arithmetic instead of asking the reader to trust a bare percentage.
    """
    plan = payload.get("plan") or {}
    r = read(price, plan.get("entry_low"), plan.get("entry_high"))
    return {
        **payload,
        "chase_pct": r.pct,
        "chase_status": r.status,
        "chase_warning": r.warning,
        "chase_price": _num(price),
    }


def annotate_pick(pick: dict, price) -> dict:
    """A copy of one /recommendations pick with the chase read attached, in place on the pick.

    Same four keys and the same always-present contract as `annotate`, differing only in where the
    entry zone is read from: a pick carries `entry_low`/`entry_high` on itself, where a /plan payload
    nests them under `plan`.
    """
    r = read(price, pick.get("entry_low"), pick.get("entry_high"))
    pick["chase_pct"] = r.pct
    pick["chase_status"] = r.status
    pick["chase_warning"] = r.warning
    pick["chase_price"] = _num(price)
    return pick


def annotate_picks(picks: list[dict], prices: dict) -> list[dict]:
    """Annotate each pick with ITS OWN symbol's live price (SWT-15).

    One /recommendations call ranks many symbols, so the price lookup is per-pick. A symbol missing
    from `prices` is annotated with None — which the app renders as no chase line at all, rather than
    as a comfortable 0%. `prices` is keyed by UPPERCASE symbol.
    """
    for p in picks:
        annotate_pick(p, prices.get(str(p.get("symbol") or "").strip().upper()))
    return picks
