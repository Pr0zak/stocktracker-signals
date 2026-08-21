"""
SWT-1 — the mechanical swing-trade indicator set, deliberately NOT folded into market.summarize().

Swing selection needs a vocabulary the analyst snapshot does not carry: volatility (ATR), trend
strength (ADX), where the bar closed inside its own range (CLV), participation (relative volume,
dollar volume) and multi-timeframe moving-average structure. All of it is derived from bars that
market.py already fetches, so the temptation is to bolt it onto summarize(). Three concrete reasons
not to, each of which has already cost this repo a debugging session in one form or another:

  1. summarize() is called with objects that are NOT full Series. tests/test_portfolio_snapshot.py:21
     monkeypatches fetch_series with a stub exposing only `closes`; the first time summarize() reads
     `.highs` that test dies with an AttributeError — and it would die inside a portfolio-pricing
     test that has nothing to do with swing scanning. This module therefore reads every field except
     `closes` through getattr() and reports what it could not see in `unmeasured`, so a bar-poor or
     stub Series degrades to "not measurable" instead of raising.

  2. summarize()'s dict is the LLM-FACING payload. It is described to the model at app/analyst.py:226
     and shipped on EVERY analyst call — the nightly scan, /signal, every sandbox tick. Ten more keys
     is ten more keys of prompt on every one of those calls, in service of a mechanical scan that
     makes no model call at all.

  3. memory.py persists the summarize() dict per verdict, and `_SANDBOX_TECH_KEYS` (app/main.py:2164)
     pins the technical key set so historical rows stay comparable. Widening summarize() splits the
     recorded population into before/after and quietly weakens every match against it.

Pure module: no I/O, no httpx, no network, no numpy/pandas. `metrics()` is a deterministic function
of the bars handed to it, which is what makes the scan cheap over hundreds of symbols and
exhaustively testable. Every helper returns None — never 0.0, never a partial — when it does not
have enough bars to answer, and `metrics()` names each of those keys in `unmeasured` so a consumer
can tell "measured, and it is zero" from "could not measure".
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

# Reused, not reimplemented: _avg_volume averages the `lookback` bars ENDING BEFORE `end`, i.e. it
# excludes the current bar. A naive mean over the last 20 bars lets a 10x volume day dilute its own
# average and understates rel_volume by roughly a third on exactly the bars that matter.
from .gaps import _avg_volume

if TYPE_CHECKING:  # typing only — importing market at runtime would drag httpx into a pure module
    from .market import Series

log = logging.getLogger("signals.swing")

# --- periods ---------------------------------------------------------------------------------
# Wilder's originals. They are conventions, not tuned parameters: the point of a mechanical scan is
# that its numbers match what every charting package shows the user, so these do not get "optimized".
_ATR_PERIOD = 14
_ADX_PERIOD = 14
_VOL_LOOKBACK = 20

# ADX(14) is a smoothing of a smoothing: 14 bars to seed +DM/-DM/TR, then 14 DX readings to seed the
# ADX itself, so it is not mathematically defined below 28 bars. But the FIRST value at 28 bars is a
# plain mean of 14 DX values with no Wilder history behind it — it moves wildly on the next bar and
# reads as a confident trend-strength number when it is really a seed. 40 bars gives the ADX a dozen
# smoothing steps before we are willing to quote it. Below that we return None rather than a number
# we would not act on.
_ADX_MIN_BARS = 40

# A volume average needs enough bars to mean anything. Yahoo nulls the odd session's volume and
# half-days are legitimately thin, so demanding all 20 would drop symbols for a cosmetic reason;
# but an "average" built from three bars turns rel_volume into noise, and rel_volume is what decides
# whether a move had participation behind it. Three quarters of the window is the compromise.
_MIN_VOL_BARS = 15

# Trading days in a year — the 52-week-high window.
_YEAR_BARS = 252


def _finite(x: object) -> bool:
    """True for a real, usable number. bool is excluded on purpose: True would otherwise arithmetic
    as 1.0 and a stray flag in a price list would silently become a $1 bar."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _align(*seqs: list) -> list[list]:
    """Trim a set of bar-aligned lists to a common length from the TAIL.

    The newest bar is the one that must line up. Aligning from the head instead — which is what a
    plain `seq[:n]` does — would pair today's high with a close from a week ago the moment one list
    is longer than another, and nothing downstream would look wrong."""
    n = min((len(s) for s in seqs), default=0)
    return [list(s[len(s) - n:]) for s in seqs]


def _true_range(high: float, low: float, prev_close: float) -> float:
    """Wilder's true range: the day's own range, or the gap-inclusive range if the bar opened away
    from yesterday's close. Using only (high - low) understates volatility exactly on gap days."""
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def _wilder(values: list[float], period: int) -> list[float | None]:
    """Wilder's smoothing (RMA) as a running AVERAGE, aligned to `values`.

    Seed with a simple mean of the first `period` values, then avg = (avg*(period-1) + x)/period.
    This is NOT a rolling simple mean: old bars decay geometrically and never drop off a window
    edge. An ATR computed as mean(last 14 true ranges) diverges from every charting package's
    number after the first spike, which is the classic way a hand-rolled ATR ends up subtly wrong.

    Returned as an average rather than Wilder's running SUM form. For ATR the average IS the answer;
    for the DI ratio the choice cancels, because +DM, -DM and TR are all smoothed over the same
    period and the division removes the common factor.
    """
    out: list[float | None] = [None] * len(values)
    if len(values) < period or period < 1:
        return out
    avg = sum(values[:period]) / period
    out[period - 1] = avg
    for i in range(period, len(values)):
        avg = (avg * (period - 1) + values[i]) / period
        out[i] = avg
    return out


# --- close-only indicators -------------------------------------------------------------------
# Deliberate duplication of market._sma / _ema_series / _rsi. Importing market would pull httpx and
# the whole fetch stack into a module whose entire value is being pure and import-cheap, and those
# helpers are private to market by intent. Same formulas (both are ports of ChartMath.kt) — if one
# side is ever corrected, correct the other.

def sma(values: list[float], period: int) -> float | None:
    if len(values) < period or period < 1:
        return None
    return sum(values[-period:]) / period


def ema_series(values: list[float], period: int) -> list[float | None]:
    """EMA at every bar (None until seeded), so callers can measure its SLOPE, not just its level."""
    out: list[float | None] = [None] * len(values)
    if len(values) < period or period < 1:
        return out
    k = 2.0 / (period + 1)
    ema = sum(values[:period]) / period
    out[period - 1] = ema
    for i in range(period, len(values)):
        ema = values[i] * k + ema * (1 - k)
        out[i] = ema
    return out


def ema(values: list[float], period: int) -> float | None:
    s = ema_series(values, period)
    return s[-1] if s else None


def rsi(values: list[float], period: int = 14) -> float | None:
    if len(values) <= period or period < 1:
        return None
    gain = loss = 0.0
    for i in range(1, period + 1):
        ch = values[i] - values[i - 1]
        gain += max(ch, 0.0)
        loss += max(-ch, 0.0)
    avg_gain, avg_loss = gain / period, loss / period
    for i in range(period + 1, len(values)):
        ch = values[i] - values[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(ch, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-ch, 0.0)) / period
    if avg_loss == 0.0:
        # No down bars in the whole window. 100 is the standard reading; a flat line (no up bars
        # either) is not overbought, it is directionless, so it reads 50.
        return 50.0 if avg_gain == 0.0 else 100.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def momentum_pct(values: list[float], period: int) -> float | None:
    """Simple `period`-bar return in percent. Needs period+1 bars: the return spans the gaps
    BETWEEN bars, so a 20-day momentum reads closes[-21] against closes[-1]."""
    if len(values) < period + 1 or period < 1:
        return None
    prev = values[-1 - period]
    if not _finite(prev) or prev <= 0:
        return None
    return (values[-1] / prev - 1.0) * 100.0


def ema_slope_pct(values: list[float], period: int = 20, lookback: int = 5) -> float | None:
    """How much the EMA itself has moved over the last `lookback` bars, in percent.

    The level of an EMA says where price sits; the slope says whether the trend is still being fed.
    Measured on the EMA's own defined values (the leading Nones are dropped first), so a series that
    just barely seeded its EMA returns None rather than comparing against a None."""
    defined = [x for x in ema_series(values, period) if x is not None]
    if len(defined) < lookback + 1:
        return None
    prev = defined[-1 - lookback]
    if prev <= 0:
        return None
    return (defined[-1] / prev - 1.0) * 100.0


def rel_strength_pct(closes: list[float], bench: list[float] | None, period: int = 63) -> float | None:
    """This name's `period`-bar return MINUS the benchmark's, in percentage points.

    Note this is a different quantity from market.relative_strength(), which returns the slope of
    the price/benchmark RATIO as a fraction. Both are legitimate; this one is the additive form the
    swing scan ranks on ("beat SPY by 8 points over three months") and the units are pct POINTS.
    Aligned tail-to-tail because the two series can differ in length (a benchmark rarely halts)."""
    if not bench:
        return None
    n = min(len(closes), len(bench))
    if n <= period or period < 1:
        return None
    c, b = closes[-n:], bench[-n:]
    c_prev, b_prev = c[-1 - period], b[-1 - period]
    if not (_finite(c_prev) and _finite(b_prev)) or c_prev <= 0 or b_prev <= 0:
        return None
    return ((c[-1] / c_prev - 1.0) - (b[-1] / b_prev - 1.0)) * 100.0


# --- OHLC indicators -------------------------------------------------------------------------

def adr_pct(highs: list[float], lows: list[float], closes: list[float], period: int = 20) -> float | None:
    """Average daily RANGE as a percent of close — how much room a typical bar gives a swing.

    Distinct from ATR: no gap component and no Wilder smoothing, just the mean of (high-low)/close.
    That is the number position sizing wants ("this thing moves 3% a day"), where ATR is the number
    a stop wants (it includes the overnight gap that would have taken the stop out)."""
    highs, lows, closes = _align(highs, lows, closes)
    n = len(closes)
    if n < period or period < 1:
        return None
    vals = []
    for h, lo, c in zip(highs[-period:], lows[-period:], closes[-period:]):
        if not (_finite(h) and _finite(lo) and _finite(c)) or c <= 0 or h < lo:
            return None      # a corrupt bar in the window makes the average a lie; refuse instead
        vals.append((h - lo) / c)
    return sum(vals) / len(vals) * 100.0


def atr(highs: list[float], lows: list[float], closes: list[float],
        period: int = _ATR_PERIOD) -> float | None:
    """Wilder's ATR(`period`).

    Needs period+1 bars: each true range compares a bar against the PREVIOUS close, so 14 true
    ranges take 15 bars. Seeded with the mean of the first `period` true ranges and then Wilder-
    smoothed (see _wilder) — not a rolling mean, which is the usual way this comes out wrong.
    """
    highs, lows, closes = _align(highs, lows, closes)
    n = len(closes)
    if n < period + 1 or period < 1:
        return None
    trs: list[float] = []
    for i in range(1, n):
        h, lo, prev_c = highs[i], lows[i], closes[i - 1]
        if not (_finite(h) and _finite(lo) and _finite(prev_c)):
            return None
        trs.append(_true_range(h, lo, prev_c))
    smoothed = _wilder(trs, period)
    return smoothed[-1]


def adx(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = _ADX_PERIOD,
    min_bars: int = _ADX_MIN_BARS,
) -> float | None:
    """Wilder's ADX(`period`) — trend STRENGTH, direction-agnostic (0-100, >25 is a real trend).

    Written out step by step on purpose. This is the most error-prone function in the module and a
    subtly wrong ADX is worse than no ADX: it looks authoritative, it is compared against a
    threshold, and nothing downstream can tell it is wrong.

    The chain, exactly as Wilder defines it:
      1. Per bar i: up = high[i] - high[i-1], down = low[i-1] - low[i].
         +DM = up   when up > down and up > 0,   else 0
         -DM = down when down > up and down > 0, else 0
         Only the larger side counts, and only when positive — an inside bar (both <= 0) scores zero
         on BOTH sides. That zero is measured, not absent: the bar genuinely had no directional
         movement. True range for the same bar is the gap-inclusive range (_true_range).
      2. Wilder-smooth +DM, -DM and TR over `period`.
      3. +DI = 100 * smoothed(+DM) / smoothed(TR),  -DI likewise.
         DX  = 100 * |+DI - -DI| / (+DI + -DI)  — how one-sided the movement is, 0..100.
      4. ADX = Wilder smoothing of the DX series. So a first ADX needs `period` DX values, each of
         which needed `period` bars of DM/TR: 2*period bars before it exists at all, and a good deal
         more before it means anything (see _ADX_MIN_BARS).
    """
    highs, lows, closes = _align(highs, lows, closes)
    n = len(closes)
    # 2*period is the mathematical floor, not a taste call: n bars give n-1 DM/TR values, the first
    # smoothed one lands at index period-1, leaving n-period DX values, and the ADX seed needs
    # `period` of them. n >= 2*period, i.e. 28 bars for ADX(14). min_bars is the stricter policy
    # gate on top of that (see _ADX_MIN_BARS).
    if n < min_bars or n < 2 * period or period < 1:
        return None

    # ---- 1. per-bar directional movement and true range
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    trs: list[float] = []
    for i in range(1, n):
        h, lo, ph, pl, pc = highs[i], lows[i], highs[i - 1], lows[i - 1], closes[i - 1]
        if not (_finite(h) and _finite(lo) and _finite(ph) and _finite(pl) and _finite(pc)):
            return None
        up = h - ph          # how far today's high extended above yesterday's
        down = pl - lo       # how far today's low extended below yesterday's
        plus_dm.append(up if (up > down and up > 0) else 0.0)
        minus_dm.append(down if (down > up and down > 0) else 0.0)
        trs.append(_true_range(h, lo, pc))

    # ---- 2. Wilder smoothing of all three, on the same period so the DI division is well-posed
    sm_plus = _wilder(plus_dm, period)
    sm_minus = _wilder(minus_dm, period)
    sm_tr = _wilder(trs, period)

    # ---- 3. +DI / -DI / DX per bar, over the bars where all three are seeded
    dx: list[float] = []
    for p, m, t in zip(sm_plus, sm_minus, sm_tr):
        if p is None or m is None or t is None:
            continue
        if t <= 0:
            # A smoothed true range of zero means every bar in the window had zero range AND no gap
            # — a halted or synthetic series. +DI/-DI are undefined there. Refuse the whole reading
            # rather than skip the bar: skipping would splice non-adjacent bars into the ADX
            # smoothing below and quietly report a number computed off a broken chain.
            return None
        plus_di = 100.0 * p / t
        minus_di = 100.0 * m / t
        di_sum = plus_di + minus_di
        if di_sum == 0.0:
            # Both DIs zero: the window contained no directional movement at all. DX = 0 is the
            # MEASURED answer here ("no trend"), not an absent one, so it stays in the series.
            dx.append(0.0)
        else:
            dx.append(100.0 * abs(plus_di - minus_di) / di_sum)

    # ---- 4. ADX = Wilder smoothing of DX
    if len(dx) < period:
        return None
    return _wilder(dx, period)[-1]


def clv(highs: list[float], lows: list[float], closes: list[float]) -> float | None:
    """Close Location Value of the LAST bar: ((c-l) - (h-c)) / (h-l), in -1..+1.

    +1 = closed on the high (buyers held the bar), -1 = closed on the low, 0 = dead middle. On a
    doji-flat bar (high == low) the denominator is zero and the value is UNDEFINED — it must come
    back None. Returning 0.0 there would read as "closed mid-range", i.e. an actual measurement,
    which is precisely the absent-rendered-as-a-number failure this codebase keeps fixing."""
    highs, lows, closes = _align(highs, lows, closes)
    if not (highs and lows and closes):
        return None
    h, lo, c = highs[-1], lows[-1], closes[-1]
    if not (_finite(h) and _finite(lo) and _finite(c)):
        return None
    if h <= lo:
        return None
    return ((c - lo) - (h - c)) / (h - lo)


# --- volume ------------------------------------------------------------------------------------

def rel_volume(volumes: list[float | None], lookback: int = _VOL_LOOKBACK) -> float | None:
    """Last bar's volume as a multiple of the average of the `lookback` bars BEFORE it.

    The exclusion is the whole point and the reason this reuses gaps._avg_volume rather than taking
    a mean of the last 20 bars: a 10x volume day included in its own denominator reports itself as
    ~7x. rel_volume gates "did this move have participation", so understating it by a third on the
    loudest bars defeats the metric."""
    end = len(volumes) - 1
    if end < lookback:
        # Not a full prior window. A ratio against a 6-bar average is not a relative volume, it is
        # noise with a decimal point on it.
        return None
    last = volumes[end]
    if not _finite(last) or last <= 0:
        return None
    window = volumes[end - lookback:end]
    usable = [v for v in window if _finite(v) and v > 0]
    if len(usable) < _MIN_VOL_BARS:
        return None
    avg = _avg_volume(volumes, end, lookback=lookback)   # EXCLUDES bar `end` — see the import note
    if not avg:
        return None
    return float(last) / avg


def dollar_volume(closes: list[float], volumes: list[float | None],
                  period: int = _VOL_LOOKBACK) -> float | None:
    """Mean close*volume over the last `period` bars — the liquidity floor for a swing position.

    Known inexactness, stated rather than papered over: `closes` here is the SPLIT-ADJUSTED close
    while `volumes` is the raw reported share count, so across a split the product is off by the
    split factor for the pre-split bars in the window (a 10:1 split makes those bars look 10x
    smaller in dollars). For a liquidity screen with an order-of-magnitude threshold that is
    acceptable; it would not be acceptable as an input to anything precise."""
    closes, volumes = _align(closes, volumes)
    if len(closes) < period or period < 1:
        return None
    vals = []
    for c, v in zip(closes[-period:], volumes[-period:]):
        if _finite(c) and _finite(v) and c > 0 and v > 0:
            vals.append(float(c) * float(v))
    if len(vals) < _MIN_VOL_BARS:
        return None
    return sum(vals) / len(vals)


# --- assembly ----------------------------------------------------------------------------------

def _complete_ohlc_tail(
    highs: list, lows: list, closes: list,
) -> tuple[list[float], list[float], list[float]]:
    """The longest RECENT run of bars where high, low and close are all present and coherent.

    Yahoo nulls the occasional bar. Dropping such a bar from the middle would splice two
    non-adjacent sessions into one true range / one directional-movement pair and overstate both, so
    instead we truncate history at the hole and work with the clean tail.

    If the LAST bar is the incomplete one the result is empty, deliberately: an ATR or a CLV
    computed off the previous session would be returned alongside today's `price` as though the two
    described the same bar."""
    h, lo, c = _align(highs, lows, closes)
    n = len(c)
    if n == 0:
        return [], [], []
    start = n
    for i in range(n - 1, -1, -1):
        if not (_finite(h[i]) and _finite(lo[i]) and _finite(c[i])) or h[i] < lo[i]:
            break
        start = i
    return (
        [float(x) for x in h[start:n]],
        [float(x) for x in lo[start:n]],
        [float(x) for x in c[start:n]],
    )


def _round(v: float | None, ndigits: int = 2) -> float | None:
    return round(v, ndigits) if v is not None else None


# --- series integrity --------------------------------------------------------------------------

# A single-session price ratio beyond which the series is not describing a price move.
#
# Yahoo's chart endpoint serves some reverse-split names as a MIXTURE of pre- and post-split bars in
# one array, and its `adjclose` does not correct them — measured on 2026-08-21, BYND's two-year
# series oscillated 0.59 -> 17.85 -> 0.56 -> 16.98 -> 0.61 -> 15.84 while the adjusted/raw ratio sat
# at exactly 1.0000 on every bar. WETO carried a 300x bar, DFNS 101x, TNON 46x. Every metric derived
# from such a series is fiction, and because the fiction is enormous it sorts straight to the top of
# any momentum ranking and dominates any percentile computed across the universe.
#
# 10x is chosen from the measured separation, not from theory. On the same night the largest
# single-bar move among genuine movers was 6.17x (FCUV) with the biotech readouts that drove the
# real >100% twenty-day gains sitting at 1.31-1.64x, while the smallest corrupt jump was 30.5x.
# Anything in the 6x-30x gap would work; 10x sits in it with room on both sides. The cost of being
# wrong is bounded and known: a real 10x single session would be dropped from one night's scan and
# named in `suspect_series`, not silently swallowed.
_MAX_BAR_RATIO = 10.0


def worst_bar_ratio(closes: list[float] | None) -> float | None:
    """The most extreme one-session price ratio in the series, expressed as >= 1.0.

    Direction-agnostic: a 30x jump and a 0.033x collapse are the same corruption seen from either
    end, and a mixed-basis series always contains both. None when there is nothing to compare.
    """
    c = [float(x) for x in (closes or []) if _finite(x) and float(x) > 0]
    if len(c) < 2:
        return None
    worst = 1.0
    for i in range(1, len(c)):
        r = c[i] / c[i - 1]
        worst = max(worst, r, 1.0 / r)
    return worst


def implausible_jump(closes: list[float] | None) -> float | None:
    """The offending ratio when the series contains a break too large to be a price move, else None.

    Returns the ratio rather than a bool so the caller can NAME the number it rejected — "SYM: 30.5x
    single-bar jump" is auditable, "SYM: rejected" is not.
    """
    worst = worst_bar_ratio(closes)
    return worst if (worst is not None and worst >= _MAX_BAR_RATIO) else None


def metrics(series: Series, *, bench_closes: list[float] | None = None) -> dict:
    """The full mechanical read on one name. Pure: no fetching, no model call, no side effects.

    Every numeric is float|None and every flag bool|None, where None ALWAYS means "not measurable"
    and NEVER 0/False. `unmeasured` lists, in key order, every field that came back None — so a
    consumer can distinguish a name whose momentum really is zero from a name that has 30 bars of
    history. A field is never filled from a previous value or a default.

    `series` is duck-typed on purpose: only `closes` is required. highs/lows/volumes are read via
    getattr so a stub Series (tests/test_portfolio_snapshot.py:21) or a Webull-sourced Series that
    never carried highs degrades to unmeasured fields instead of an AttributeError.
    """
    raw_closes = list(getattr(series, "closes", None) or [])
    # Two views of the closes, on purpose:
    #   `closes`      — non-finite bars filtered out, for the 1-D indicators where only the sequence
    #                   of prices matters and a hole is better dropped than propagated.
    #   `raw_closes`  — untouched, because it is INDEX-ALIGNED with volumes/highs/lows and any
    #                   filtering here would silently pair a close with someone else's volume.
    closes = [float(x) for x in raw_closes if _finite(x)]
    highs = list(getattr(series, "highs", None) or [])
    lows = list(getattr(series, "lows", None) or [])
    volumes = list(getattr(series, "volumes", None) or [])
    hh, ll, cc = _complete_ohlc_tail(highs, lows, raw_closes)
    if highs and len(cc) < len(raw_closes):
        # The one thing the returned dict cannot say for itself. When ATR/ADX come back unmeasured
        # across a swathe of the universe, this is the line that says whether the cause was short
        # history or a hole in the OHLC arrays that truncated it.
        log.debug("%s: usable OHLC tail is %d of %d bars",
                  getattr(series, "symbol", "?"), len(cc), len(raw_closes))

    price = closes[-1] if closes else None
    sma20, sma50 = sma(closes, 20), sma(closes, 50)
    sma150, sma200 = sma(closes, 150), sma(closes, 200)
    atr14 = atr(hh, ll, cc)

    # 52-week high from intraday highs when the window is COMPLETE, else from closes. A close-based
    # high understates the real high, which makes pct_off_52w_high slightly less negative — so the
    # fallback is flagged by being consistent, not by being silently mixed: we never take the max of
    # a partially-populated highs list, which would be a half-intraday, half-nothing number.
    high_window = highs[len(highs) - min(len(highs), _YEAR_BARS):]
    close_window = closes[-_YEAR_BARS:]
    if high_window and len(high_window) >= len(close_window) and all(_finite(x) for x in high_window):
        ref_high: float | None = max(float(x) for x in high_window)
    elif close_window:
        ref_high = max(close_window)
    else:
        ref_high = None
    # Cannot be positive by construction (the max includes the last bar); clamped so float noise
    # never prints a +0.0001% "above its own 52-week high".
    _high_ok = bool(price and ref_high and ref_high > 0)
    pct_off_high = min((price / ref_high - 1.0) * 100.0, 0.0) if _high_ok else None

    out: dict = {
        "price": _round(price, 4),
        "bars": len(closes),
        "sma20": _round(sma20, 4),
        "sma50": _round(sma50, 4),
        "sma150": _round(sma150, 4),
        "sma200": _round(sma200, 4),
        "ema20": _round(ema(closes, 20), 4),
        "ema50": _round(ema(closes, 50), 4),
        "above_sma50": (price > sma50) if (price is not None and sma50 is not None) else None,
        "above_sma200": (price > sma200) if (price is not None and sma200 is not None) else None,
        # The classic stage-2 stack. All three or nothing: with sma150 missing, "sma50 > sma200"
        # is a DIFFERENT claim and reporting it under this key would misdescribe the structure.
        "ma_stacked": (sma50 > sma150 > sma200) if None not in (sma50, sma150, sma200) else None,
        "adr20_pct": _round(adr_pct(hh, ll, cc)),
        "atr14": _round(atr14, 4),
        "atr14_pct": _round(atr14 / price * 100.0) if (atr14 is not None and price) else None,
        "adx14": _round(adx(hh, ll, cc)),
        "clv": _round(clv(hh, ll, cc), 3),
        "rel_volume": _round(rel_volume(volumes)),
        "dollar_volume_20d": _round(dollar_volume(raw_closes, volumes), 0),
        "mom_20d": _round(momentum_pct(closes, 20)),
        "mom_60d": _round(momentum_pct(closes, 60)),
        "rsi14": _round(rsi(closes)),
        "pct_off_52w_high": _round(pct_off_high),
        "pct_vs_sma50": _round((price / sma50 - 1.0) * 100.0) if (price and sma50) else None,
        "pct_vs_sma200": _round((price / sma200 - 1.0) * 100.0) if (price and sma200) else None,
        "rel_strength_3mo": _round(rel_strength_pct(closes, bench_closes)),
        "ema20_slope_pct": _round(ema_slope_pct(closes)),
    }
    # Built from the dict itself rather than hand-maintained, so a key added above can never be
    # forgotten here and quietly count as "measured". `bars` is an int and never None.
    out["unmeasured"] = [k for k, v in out.items() if v is None]
    return out
