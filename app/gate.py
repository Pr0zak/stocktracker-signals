"""The five-leg market regime gate — the one place that answers "is this a market to be buying in?".

WHAT DECISION THIS PROTECTS
---------------------------
Every buy-side consumer in this service (the sandbox trader, the swing scan's candidate list, the
app's suggestion cards) is a stock-picker: it ranks names against each other and hands back the best
of whatever it was given. None of them look up. A ranker fed a tape that is bleeding still returns a
top five, and it returns it with exactly the same confidence it had in April. This module is the
lookup: five cheap, mechanical, well-understood readings of the market as a whole, evaluated
together, so a consumer can stand aside for a reason it can name.

The five legs are deliberately boring — two trend legs (SPY, QQQ over their 50-day EMAs), one
participation leg (breadth), one volatility leg (VIX), one momentum leg (SPY's 20-day return). They
are not tuned, not backtested, and not weighted by a model. They are the conditions under which the
mechanical strategies downstream were designed to work, written down where they can be checked.

THE INVARIANT
-------------
An unmeasurable leg is NEVER a silent pass and NEVER a silent fail. Its `ok` is None and its name
goes in `unmeasured`; `passed` is True only when all five are explicitly True, and None — not False
— when a leg is unknown and none of the measurable ones failed.

Why this is the whole point of the module: "we stood aside because VIX was 24" and "we stood aside
because the nightly scan had not run" are different facts about the world. A gate that reports
`passed: false` because it could not read breadth is indistinguishable from a bearish market, and it
teaches its consumers to distrust a real closure. A gate that reports `passed: true` for the same
reason is worse — it is an all-clear invented out of a missing table. `scan_store.breadth()` already
guards exactly this defect with its `available` flag, and the app-ui-honesty-invariants note tracks
the class; this module is the same discipline applied to a decision rather than a display.

An explicit False DOES outrank an unknown: if VIX is 26 we know enough to stand aside, and the
reason we report is real, whether or not breadth was readable alongside it.

Purity
------
`legs_from()` is the whole judgement and takes no I/O — already-fetched bars, an already-read VIX
level, an already-read breadth dict. `evaluate()` does nothing but fetch, call it, and stamp the
result. That split is what makes the invariant above testable without a network.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Callable
from typing import Any

import httpx

from . import market, market_now, scan_store, swing

log = logging.getLogger("signals.gate")

# --- thresholds ---------------------------------------------------------------------------------
# Conventions, not tuned parameters — the same round numbers a discretionary trader would quote, so
# that a consumer reading "breadth 54%, gate closed" can check the claim against any chart package.
# Changing one of these changes what every downstream strategy considers a tradeable market, so they
# live here as named constants rather than inline in the leg that uses them.
_EMA_PERIOD = 50
_BREADTH_PCT = 55.0
_VIX_MAX = 20.0
_MOM_BARS = 20

# --- scoring spans ------------------------------------------------------------------------------
# How far past its threshold a leg has to sit to earn FULL margin credit (see market_score). Each is
# a judgement call about what "comfortably clear" looks like in that leg's own units, and each is
# roughly one ordinary swing in that reading — not a statistical quantity.
_SPAN_EMA_PCT = 3.0        # SPY/QQQ 3% above the 50-EMA
_SPAN_BREADTH_PCT = 20.0   # breadth at 75% rather than 55%
_SPAN_VIX = 6.0            # VIX at 14 rather than 20
_SPAN_MOM_PCT = 5.0        # +5% over 20 sessions

# Per-leg margin in that leg's own units, signed so POSITIVE always means "further past the
# threshold". The direction is not derivable from (value, threshold) alone — VIX passes by being
# BELOW its line and the two MA legs are proportional rather than absolute — so it is written per
# leg here instead of being guessed generically at the call site.
_MARGIN: dict[str, Callable[[float, float], float | None]] = {
    "spy_above_ema50": lambda v, t: ((v / t - 1.0) * 100.0) if t > 0 else None,
    "qqq_above_ema50": lambda v, t: ((v / t - 1.0) * 100.0) if t > 0 else None,
    "breadth_55": lambda v, t: v - t,
    "vix_under_20": lambda v, t: t - v,
    "spy_mom_20d": lambda v, t: v - t,
}
_SPAN: dict[str, float] = {
    "spy_above_ema50": _SPAN_EMA_PCT,
    "qqq_above_ema50": _SPAN_EMA_PCT,
    "breadth_55": _SPAN_BREADTH_PCT,
    "vix_under_20": _SPAN_VIX,
    "spy_mom_20d": _SPAN_MOM_PCT,
}

# Three symbols, three sockets, one round trip of wall clock. The bound exists so the fan-out is a
# named number that cannot grow by accident when a sixth leg is added: Yahoo tolerates far more than
# three concurrent chart requests, and anything lower would serialise a fetch that sits in a request
# path. It is NOT a rate limit — market_scan_job owns that problem for the bulk universe.
_MAX_CONCURRENT_FETCHES = 3


# --- the pure part ------------------------------------------------------------------------------

def legs_from(spy: Any, qqq: Any, vix: float | None, breadth: dict | None) -> list[dict]:
    """The five legs, in contract order, from already-fetched inputs. Pure: no I/O, never raises.

    `spy`/`qqq` are duck-typed on `.closes` exactly as `swing.metrics()` is, so a stub object or a
    None (the fetch failed) degrades to an unmeasurable leg instead of an AttributeError. `vix` is a
    level, `breadth` is a `scan_store.breadth()` dict.

    Every leg's `ok` is True, False, or None, and None ONLY ever means "could not measure". No leg
    ever falls back to a default, a previous value, or the other side's reading.
    """
    return [
        _ma_leg("SPY > 50-EMA", "spy_above_ema50", spy, "SPY"),
        _ma_leg("QQQ > 50-EMA", "qqq_above_ema50", qqq, "QQQ"),
        _breadth_leg(breadth),
        _vix_leg(vix),
        _mom_leg(spy),
    ]


def _ma_leg(name: str, key: str, series: Any, label: str) -> dict:
    """Last close vs the 50-day EMA. Unmeasurable when the fetch failed or there are <50 bars."""
    closes = _closes(series)
    if not closes:
        return _leg(name, key, None, None, None, f"no {label} bars — fetch failed or empty")
    price = closes[-1]
    ema50 = swing.ema(closes, _EMA_PERIOD)
    if ema50 is None or not _finite(ema50) or ema50 <= 0:
        # Fewer than 50 bars is not "below the average": a series with 30 bars has no opinion about
        # its 50-day EMA, and inventing one either way is a trend claim with nothing behind it.
        return _leg(name, key, None, price, None,
                    f"{label} has {len(closes)} bars — a {_EMA_PERIOD}-bar EMA is not defined yet")
    pct = (price / ema50 - 1.0) * 100.0
    return _leg(name, key, price > ema50, round(price, 2), round(ema50, 2),
                f"{label} {price:,.2f} is {pct:+.2f}% vs its {_EMA_PERIOD}-EMA")


def _breadth_leg(breadth: dict | None) -> dict:
    """Share of the scanned universe above its own 50-SMA, from the nightly cross-section.

    This is the leg most likely to be unmeasurable in normal operation — the scan is a nightly
    systemd oneshot, so a fresh install, a failed run, or a query before the first night all land
    here. `scan_store.breadth()` already reports that honestly with `available: False` and None
    readings; this leg's only job is not to throw that away.
    """
    name, key = f"Breadth > {_BREADTH_PCT:.0f}%", "breadth_55"
    b = breadth or {}
    if not b.get("available"):
        return _leg(name, key, None, None, _BREADTH_PCT, "no scan stored — breadth cannot be measured")
    pct = _num(b.get("pct_above_sma50"))
    if pct is None:
        # A scan exists but too few of its rows carried a measurable 50-SMA (scan_store's coverage
        # floor). A percentage over a minority of the universe is a sample presented as a census.
        return _leg(name, key, None, None, _BREADTH_PCT,
                    "the stored scan could not measure enough names to call breadth")
    n = b.get("n")
    age = b.get("age_hours")
    note = f"{pct:.1f}% of {n} scanned names are above their 50-SMA"
    if age is not None:
        note += f" (scan is {age:.0f}h old)"
    return _leg(name, key, pct > _BREADTH_PCT, round(pct, 1), _BREADTH_PCT, note)


def _vix_leg(vix: float | None) -> dict:
    name, key = f"VIX < {_VIX_MAX:.0f}", "vix_under_20"
    v = _num(vix)
    if v is None:
        return _leg(name, key, None, None, _VIX_MAX, "no VIX quote — fetch failed")
    return _leg(name, key, v < _VIX_MAX, round(v, 2), _VIX_MAX, f"VIX is {v:.2f}")


def _mom_leg(series: Any) -> dict:
    """SPY's 20-session return. Positive is the pass; measured on closes, so it needs 21 bars."""
    name, key = f"SPY {_MOM_BARS}-day momentum > 0", "spy_mom_20d"
    closes = _closes(series)
    if not closes:
        return _leg(name, key, None, None, 0.0, "no SPY bars — fetch failed or empty")
    # Reused rather than re-derived: momentum_pct spans the gaps BETWEEN bars, so it needs 21 closes
    # for a 20-day read and returns None instead of a partial one. Rewriting that here is how the
    # two halves of a codebase end up disagreeing about what "20-day" means.
    mom = swing.momentum_pct(closes, _MOM_BARS)
    if mom is None:
        return _leg(name, key, None, None, 0.0,
                    f"SPY has {len(closes)} bars — a {_MOM_BARS}-bar return is not defined yet")
    return _leg(name, key, mom > 0.0, round(mom, 2), 0.0,
                f"SPY is {mom:+.2f}% over {_MOM_BARS} sessions")


def market_score(legs: list[dict]) -> float | None:
    """0-100, how the market is clearing its lines. None when ANY leg is unmeasurable.

    THE FORMULA. Each leg earns up to 1.0:

        0.5 for being explicitly True, plus
        0.5 * clamp(margin / span, 0, 1) for how far past its threshold it sits

    averaged over the five legs and multiplied by 100. So five legs scraping over their lines score
    ~50, five clearing them comfortably score ~100, and a failing leg contributes 0 (its margin is
    negative and clamps away). `margin` and `span` are per-leg and in the leg's own units — see
    _MARGIN and _SPAN, which encode that VIX passes by going DOWN.

    Why not "legs passing / 5 * 100": that is a five-valued number wearing a percentage's clothes.
    It cannot tell a market where SPY sits 0.1% over its EMA from one where it sits 8% over, and
    those are not the same market — the first is a coin flip on tomorrow's open, the second is a
    trend. Both print 100 under the naive version.

    WHAT THIS IS NOT. It is not a probability of anything. It is not a forecast of a return, a
    direction, or a duration. It is not a position size and must never be multiplied by one — a
    score of 100 does not mean "size up", it means "all five conditions are comfortably met right
    now, and they can stop being met tomorrow". It is a compression of the five legs into one
    sortable number so a history can be plotted; the legs themselves are the decision.

    Returns None — never a partial score — when a leg is unmeasurable, because an average over four
    legs presented on the same 0-100 scale as an average over five is a different statistic wearing
    the first one's name.
    """
    if not legs:
        return None
    total = 0.0
    for leg in legs:
        if leg.get("ok") is None:
            return None
        value, threshold = _num(leg.get("value")), _num(leg.get("threshold"))
        fn = _MARGIN.get(str(leg.get("key")))
        if value is None or threshold is None or fn is None:
            # A measured leg with no usable margin cannot be scored, and scoring the rest as if the
            # set were whole is the same lie the None-on-unmeasurable rule exists to prevent.
            return None
        margin = fn(value, threshold)
        if margin is None:
            return None
        frac = max(0.0, min(1.0, margin / _SPAN[leg["key"]]))
        total += (0.5 if leg["ok"] else 0.0) + 0.5 * frac
    return round(total / len(legs) * 100.0, 1)


def verdict(legs: list[dict]) -> tuple[bool | None, list[str], list[str]]:
    """(passed, failing, unmeasured) — the three-valued rule, in one place so it cannot drift.

    An explicit False outranks an unknown: we know enough to stand aside and the reason is real.
    Only an all-True set passes, and an otherwise-clean set with an unknown in it is None.
    """
    failing = [str(l.get("name")) for l in legs if l.get("ok") is False]
    unmeasured = [str(l.get("name")) for l in legs if l.get("ok") is None]
    if failing:
        passed: bool | None = False
    elif unmeasured or not legs:
        passed = None
    else:
        passed = True
    return passed, failing, unmeasured


def _summary_note(passed: bool | None, failing: list[str], unmeasured: list[str], available: bool) -> str:
    """One human sentence that never conflates "closed" with "unknown"."""
    if not available:
        return "Market gate unavailable: none of the five legs could be measured."
    if passed is True:
        return "Market gate open: all five legs pass."
    if passed is False:
        tail = f", and {_join(unmeasured)} could not be measured" if unmeasured else ""
        return f"Market gate closed: {_join(failing)} failed{tail}."
    return (f"Market gate undecided: {_join(unmeasured)} could not be measured, "
            "so we cannot say the market is clear.")


# --- the I/O part -------------------------------------------------------------------------------

async def evaluate(client: httpx.AsyncClient, *, d: str | None = None) -> dict:
    """Fetch the inputs, run `legs_from`, stamp and record the result. Never raises.

    `d` names the night whose breadth to read (and the day the row is filed under); None means the
    latest scan we hold. The price legs are always LIVE — SPY/QQQ/VIX come off the current tape
    regardless of `d`, because there is no stored history of them here and pretending otherwise
    would file a live reading under a historical date without saying so.
    """
    try:
        # Bounded, and the bound is the whole task list — see _MAX_CONCURRENT_FETCHES. Each fetch
        # swallows its own failure into None so one dead symbol nulls ONE leg instead of the gate.
        sem = asyncio.Semaphore(_MAX_CONCURRENT_FETCHES)
        spy, qqq, vix, breadth = await asyncio.gather(
            _series(client, "SPY", sem),
            _series(client, "QQQ", sem),
            _vix_level(client, sem),
            # scan_store is synchronous SQLite; every caller in this codebase wraps it.
            asyncio.to_thread(scan_store.breadth, d),
        )
        legs = legs_from(spy, qqq, vix, breadth)
        passed, failing, unmeasured = verdict(legs)
        # "Could not evaluate at all" is every leg unknown — not merely one of them. A gate with
        # four measured legs and one hole is an evaluation with a hole in it, and its `passed: None`
        # plus `unmeasured` already say so precisely.
        available = any(l["ok"] is not None for l in legs)
        out = {
            "passed": passed,
            "available": available,
            "as_of": _as_of(d, spy, breadth),
            "evaluated_at": time.time(),
            "market_score": market_score(legs),
            "legs": legs,
            "failing": failing,
            "unmeasured": unmeasured,
            "note": _summary_note(passed, failing, unmeasured, available),
        }
        if available:
            # An evaluation that measured NOTHING is not filed. The row is keyed on the day and
            # upserted, so writing it would let a 30-second network outage overwrite a real morning
            # reading with five nulls — losing the day's actual gate history to a transient.
            await asyncio.to_thread(scan_store.record_gate, out, d=out["as_of"])
        return out
    except Exception:  # noqa: BLE001 — the gate is consulted from request paths and from the
        # sandbox tick; an unexpected failure here must degrade to "we do not know", never take the
        # caller down. Unavailable-with-no-legs is the one shape that cannot be misread as a market
        # reading, because `passed` is None and every leg name lands in `unmeasured`.
        log.warning("gate: evaluate failed", exc_info=True)
        return unavailable(d)


def unavailable(d: str | None = None, *, note: str | None = None) -> dict:
    """The contract shape for "we could not evaluate", with all five legs present and null.

    The legs are still listed. A consumer that iterates them to render a checklist gets five rows of
    "—" rather than an empty panel that looks like a gate nobody asked about.
    """
    legs = legs_from(None, None, None, None)
    passed, failing, unmeasured = verdict(legs)
    return {
        "passed": passed,
        "available": False,
        "as_of": _day(d),
        "evaluated_at": time.time(),
        "market_score": None,
        "legs": legs,
        "failing": failing,
        "unmeasured": unmeasured,
        "note": note or _summary_note(passed, failing, unmeasured, False),
    }


async def _series(client: httpx.AsyncClient, symbol: str, sem: asyncio.Semaphore) -> Any:
    """Daily bars, or None if the fetch failed. A failure NULLS a leg — it never fails one."""
    async with sem:
        try:
            # fallback=False: SPY and QQQ are the two most liquid ETFs listed. If Yahoo has no bars
            # for them the problem is Yahoo, and the Webull rescue (a search + a chart against an
            # unofficial endpoint, ~27s) would buy nothing but latency in a request path.
            return await market.fetch_series(client, symbol, fallback=False)
        except Exception:  # noqa: BLE001 — an upstream outage makes this leg UNKNOWN, and the
            # invariant is that unknown is never rendered as a pass or a fail. Raising here would
            # take the other four legs with it.
            log.warning("gate: %s fetch failed", symbol, exc_info=True)
            return None


async def _vix_level(client: httpx.AsyncClient, sem: asyncio.Semaphore) -> float | None:
    """The current VIX level, or None. Reuses market_now's batched v7 quote — the ONE ^VIX path.

    Not a chart fetch: market_now already fetches ^VIX by symbol for the market pulse, cookie+crumb
    handshake and all, and a second way to read the same number is a second way for two screens to
    disagree about what the VIX is.
    """
    async with sem:
        try:
            quotes = await market_now.fetch_quotes(client, [market_now.VIX])
        except Exception:  # noqa: BLE001 — same rule as _series: no quote means an UNKNOWN leg.
            log.warning("gate: VIX quote failed", exc_info=True)
            return None
    row = quotes.get(market_now.VIX) or {}
    # session_price, not `price`: outside regular hours regularMarketPrice is frozen at the last
    # close, and this reuses the one helper that already knows which print is the live one.
    return _num(market_now.session_price(row))


# --- helpers ------------------------------------------------------------------------------------

def _leg(name: str, key: str, ok: bool | None, value: float | None,
         threshold: float | None, note: str) -> dict:
    """One leg in the published shape. `ok=None` is the only way a leg says "I do not know"."""
    return {"name": name, "key": key, "ok": ok, "value": value, "threshold": threshold, "note": note}


def _closes(series: Any) -> list[float]:
    """Finite closes off a duck-typed series. [] for None, a stub, or an all-null series."""
    raw = getattr(series, "closes", None) or []
    try:
        return [float(x) for x in raw if _finite(x)]
    except TypeError:
        return []


def _as_of(d: str | None, spy: Any, breadth: dict | None) -> str:
    """The day this gate describes, as YYYYMMDD.

    Preference order is a claim about which reading dates the gate: an explicit `d` wins, then SPY's
    last bar (four of the five legs are live prices), then the scan's night, then today in UTC.
    """
    dates = getattr(spy, "dates", None) or []
    return (
        _day(d)
        or (str(dates[-1]) if dates else None)
        or ((breadth or {}).get("as_of") if (breadth or {}).get("available") else None)
        or time.strftime("%Y%m%d", time.gmtime())
    )


def _day(d: Any) -> str | None:
    """`d` as a bare YYYYMMDD, or None. Never raises — a bad date must not sink an evaluation."""
    digits = "".join(c for c in str(d or "") if c.isdigit())
    return digits[:8] if len(digits) >= 8 else None


def _join(names: list[str]) -> str:
    """Leg names as an English list, so the note reads as a sentence rather than a repr."""
    if not names:
        return "nothing"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _num(v: Any) -> float | None:
    """A finite float or None. bool is refused: True would arithmetic as 1.0 and score a leg."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))
