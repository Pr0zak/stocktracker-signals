"""SWT-1 — the mechanical swing indicator set (app/swing.py).

Two classes of failure are pinned here, and both have precedent in this repo.

THE FIRST is absent-rendered-as-a-number. Every helper in swing.py must return None — never 0.0,
never a partial average, never False — when it does not have the bars to answer. A 0.0 ADX on a
30-bar listing reads as "no trend" to a screen that then ranks it against names with 300 bars of
history; a 0.0 CLV on a doji reads as "closed mid-range" when the truth is that the bar had no range
to be located in. The insufficient-data assertions below are the invariant most likely to rot as
someone "tidies up" a guard, so each function is asserted individually rather than in a loop over a
convenience wrapper.

THE SECOND is a plausible-looking indicator that is quietly wrong. ATR and ADX are Wilder-smoothed,
not rolling means, and the difference only shows up after a volatility spike — a hand-rolled ATR
that averages the last 14 true ranges tracks a real ATR closely for months and then diverges exactly
when a stop is being placed. Both are checked against series whose arithmetic is worked out by hand
in the comments, so a future reader can re-verify the numbers rather than trusting a recorded
output that may itself have been produced by a bug.

Pure module, so: no fixtures, no network, no monkeypatching, no SIGNALS_DATA_DIR. The bars are built
in the test. The Series stand-in is duck-typed on purpose — swing.metrics() reads everything except
`closes` through getattr precisely so that a stub Series (the shape at
tests/test_portfolio_snapshot.py:21) degrades to unmeasured fields instead of an AttributeError, and
importing market.Series here would both couple these tests to the httpx-carrying fetch layer and
hide that behaviour.
"""
from dataclasses import dataclass

import pytest

from app import swing


@dataclass
class Bars:
    """A Series-shaped bag of bars. Only `closes` is required, same as swing.metrics() requires."""
    closes: list
    highs: list | None = None
    lows: list | None = None
    volumes: list | None = None


class ClosesOnly:
    """The stub shape from tests/test_portfolio_snapshot.py — no highs, no lows, no volumes."""
    def __init__(self, closes):
        self.closes = closes


def ramp(n=40, step=1.0, rng=1.0, base=100.0):
    """`n` bars marching up `step` per bar, each `rng` wide, closing at mid-range.

    Returns (highs, lows, closes). With the defaults every bar is:
        low = 100+i, high = 101+i, close = 100.5+i
    which makes the Wilder math checkable by hand — see the ATR/ADX tests.
    """
    highs, lows, closes = [], [], []
    for i in range(n):
        low = base + i * step
        highs.append(low + rng)
        lows.append(low)
        closes.append(low + rng / 2.0)
    return highs, lows, closes


def flat_series(n, price=100.0, rng=2.0):
    """`n` identical bars: close `price`, high/low `rng`/2 either side of it."""
    return ([price + rng / 2.0] * n, [price - rng / 2.0] * n, [price] * n)


# ---- the insufficient-data invariant: None, never zero -----------------------------------------

def test_the_close_only_indicators_return_none_rather_than_zero_without_enough_bars():
    # `is None` rather than `== None` on purpose: 0.0 == None is False, but a caller that wrote
    # `if not value` would treat a 0.0 and a None identically, which is the bug this guards.
    assert swing.sma([1.0] * 19, 20) is None
    assert swing.ema([1.0] * 19, 20) is None
    assert swing.ema_series([1.0] * 19, 20) == [None] * 19
    assert swing.rsi([1.0] * 14, 14) is None, "RSI(14) needs 15 closes — 14 gives 13 changes"
    assert swing.momentum_pct([1.0] * 20, 20) is None, "a 20-bar return spans 21 bars"
    assert swing.ema_slope_pct([1.0] * 24, 20, 5) is None, "needs 6 defined EMA values, i.e. 25 bars"
    assert swing.rel_strength_pct([1.0] * 64, None) is None, "no benchmark is not a zero spread"
    assert swing.rel_strength_pct([1.0] * 63, [1.0] * 63, 63) is None


def test_the_ohlc_indicators_return_none_rather_than_zero_without_enough_bars():
    assert swing.adr_pct(*ramp(19), period=20) is None
    assert swing.atr(*ramp(14)) is None, "ATR(14) needs 15 bars: 14 true ranges take 15 closes"
    assert swing.adx(*ramp(39)) is None
    assert swing.clv([], [], []) is None


def test_the_volume_indicators_return_none_rather_than_zero_without_enough_bars():
    assert swing.rel_volume([1000.0] * 20) is None, "20 bars leaves only 19 in the prior window"
    assert swing.dollar_volume([10.0] * 19, [100.0] * 19) is None
    # A window that is mostly nulls is not an average, whatever its arithmetic mean says.
    thin = [None] * 14 + [1000.0] * 6 + [5000.0]
    assert swing.rel_volume(thin) is None
    assert swing.dollar_volume([10.0] * 21, thin) is None


def test_a_series_with_no_bars_at_all_measures_nothing_and_still_returns_the_contract():
    m = swing.metrics(Bars(closes=[]))
    assert m["bars"] == 0
    assert m["price"] is None
    assert "price" in m["unmeasured"]
    assert all(m[k] is None for k in m["unmeasured"])


# ---- Wilder's ATR ------------------------------------------------------------------------------

def test_atr14_on_a_constant_range_series_is_that_true_range():
    # Hand check, step by step, on ramp(40) — bar i is low=100+i, high=101+i, close=100.5+i:
    #   true range of bar i = max(high-low, |high - prev_close|, |low - prev_close|)
    #                       = max(1.0,      |101+i - (100.5+i-1)| = 1.5, |100+i - (100.5+i-1)| = 0.5)
    #                       = 1.5   for every bar from 1 onwards
    #   seed  = mean of the first 14 true ranges = 1.5
    #   step  = (1.5*13 + 1.5)/14 = 1.5, and it stays 1.5 forever
    # So ATR(14) is exactly 1.5. Note this bar's own high-low is only 1.0: the extra 0.5 is the
    # overnight gap, which is the whole reason ATR is not just an average daily range.
    assert swing.atr(*ramp(40)) == pytest.approx(1.5)


def test_atr14_uses_wilder_smoothing_and_not_a_rolling_mean():
    # Same ramp(40), but bar 34's low is dropped 10 points (a spike low, close unchanged):
    #   TR[34] = max(high-low = 11.0, |high - prev_close| = 1.5, |low - prev_close| = 9.5) = 11.0
    #   TR[35] is back to 1.5 — it keys off close[34], which we did not touch.
    # Wilder from there, with the ATR sitting at exactly 1.5 before the spike:
    #   ATR[34] = (1.5*13 + 11.0)/14 = 1.5 + 9.5/14
    #   each later quiet bar: ATR = (ATR*13 + 1.5)/14 = 1.5 + (previous excess)*(13/14)
    #   bar 34 is 5 bars from the end, so ATR[39] = 1.5 + (9.5/14)*(13/14)**5 = 1.9684596...
    # A rolling mean of the last 14 true ranges would instead give (1.5*13 + 11.0)/14 = 2.1785714,
    # because it weights a five-day-old spike exactly as heavily as yesterday. The 0.21 gap between
    # the two is the entire point of this test.
    highs, lows, closes = ramp(40)
    lows[34] -= 10.0
    expected = 1.5 + (9.5 / 14) * (13 / 14) ** 5
    rolling_mean = (1.5 * 13 + 11.0) / 14

    got = swing.atr(highs, lows, closes)
    assert got == pytest.approx(expected, abs=1e-9)
    assert got != pytest.approx(rolling_mean, abs=1e-3), "this is a rolling mean, not Wilder smoothing"


def test_atr_is_reported_as_a_percentage_of_price_too():
    m = swing.metrics(Bars(*_bars_kwargs(ramp(60))))
    assert m["atr14"] == pytest.approx(1.5)
    # price is the last close (159.5 on ramp(60)); 1.5/159.5*100 = 0.94%
    assert m["atr14_pct"] == pytest.approx(m["atr14"] / m["price"] * 100, abs=0.01)


# ---- Wilder's ADX -----------------------------------------------------------------------------

def test_adx14_on_a_perfectly_trending_series_is_exactly_one_hundred():
    # Hand check on ramp(40), bar i = (low 100+i, high 101+i, close 100.5+i):
    #   up   = high[i] - high[i-1] = +1.0
    #   down = low[i-1] - low[i]   = -1.0
    #   up > down and up > 0  ->  +DM = 1.0, -DM = 0.0   for every bar
    #   TR = 1.5 for every bar (derived in the ATR test above)
    #   Wilder smoothing of constants leaves them constant: +DM14 = 1.0, -DM14 = 0.0, TR14 = 1.5
    #   +DI = 100 * 1.0/1.5 = 66.667      -DI = 100 * 0.0/1.5 = 0
    #   DX  = 100 * |66.667 - 0| / (66.667 + 0) = 100 for every bar
    #   ADX = Wilder smoothing of a constant 100 = 100
    # A one-sided market is what 100 means; nothing real gets there, which is exactly why it is a
    # good arithmetic check.
    assert swing.adx(*ramp(40)) == pytest.approx(100.0)


def test_adx_reads_high_on_a_clean_trend_and_low_on_a_directionless_one():
    trending = ramp(60, step=0.8)
    assert swing.adx(*trending) > 25, "a series that only goes up is a trend by any definition"

    # A sawtooth: every bar reverses the last one, so +DM and -DM alternate one for one, +DI and -DI
    # end up near-equal, DX collapses toward zero and so does the ADX. This is the case a naive
    # implementation gets wrong by taking |up| and |down| without the "larger side only" rule — it
    # scores both directions on every bar and reports a strong trend in a flat market.
    highs, lows, closes = [], [], []
    for i in range(60):
        base = 100.0 + (1.0 if i % 2 else 0.0)
        highs.append(base + 1.0)
        lows.append(base)
        closes.append(base + 0.5)
    assert swing.adx(highs, lows, closes) < 20


def test_adx_refuses_to_quote_a_number_before_it_has_smoothing_history():
    # ADX(14) is not mathematically defined below 28 bars, and the value AT 28 bars is a plain mean
    # of 14 DX readings with no Wilder history behind it — it swings hard on the next bar while
    # looking like a settled trend-strength number. The module requires 40.
    assert swing.adx(*ramp(39)) is None
    assert swing.adx(*ramp(40)) is not None
    assert swing.adx(*ramp(30), min_bars=28) is not None, "the 40-bar floor is policy, not arithmetic"
    assert swing.adx(*ramp(27), min_bars=1) is None, "below 2*period there is no ADX to smooth"


def test_a_frozen_series_has_no_measurable_trend_strength():
    # Every bar identical: no range, no gap, so the smoothed true range is zero and +DI/-DI are
    # undefined. That has to come back None — a 0.0 here would be read as "measured, no trend".
    highs, lows, closes = flat_series(60, rng=0.0)
    assert swing.adx(highs, lows, closes) is None
    assert swing.atr(highs, lows, closes) == pytest.approx(0.0), "a zero ATR IS measurable — it is 0"


# ---- close location value ----------------------------------------------------------------------

def test_clv_is_plus_one_at_the_high_and_minus_one_at_the_low():
    assert swing.clv([10.0], [8.0], [10.0]) == pytest.approx(1.0)
    assert swing.clv([10.0], [8.0], [8.0]) == pytest.approx(-1.0)
    assert swing.clv([10.0], [8.0], [9.0]) == pytest.approx(0.0)
    assert swing.clv([10.0], [8.0], [9.5]) == pytest.approx(0.5)


def test_clv_on_a_zero_range_bar_is_unmeasurable_and_not_zero():
    got = swing.clv([9.0], [9.0], [9.0])
    assert got is None, "high == low divides by zero; 0.0 would claim the bar closed mid-range"
    m = swing.metrics(Bars(closes=[9.0] * 30, highs=[9.0] * 30, lows=[9.0] * 30))
    assert m["clv"] is None and "clv" in m["unmeasured"]


def test_clv_reads_the_last_bar_and_only_the_last_bar():
    highs, lows, closes = ramp(30)
    closes[-1] = highs[-1]
    assert swing.clv(highs, lows, closes) == pytest.approx(1.0)


# ---- volume ------------------------------------------------------------------------------------

def test_rel_volume_excludes_the_current_bar_from_its_own_average():
    # 20 quiet bars then a 10x bar. The average must be built from the 20 BEFORE it, giving 10.0.
    # Including the current bar (the naive "mean of the last 20") gives 6.90 — a third of the signal
    # thrown away on exactly the bars where relative volume is the point.
    volumes = [1000.0] * 20 + [10000.0]
    assert swing.rel_volume(volumes) == pytest.approx(10.0)

    naive = 10000.0 / (sum(volumes[-20:]) / 20)
    assert swing.rel_volume(volumes) != pytest.approx(naive, abs=0.5)


def test_rel_volume_tolerates_a_few_null_bars_but_not_a_hollow_window():
    volumes = [1000.0] * 12 + [None] * 3 + [1000.0] * 5 + [4000.0]
    assert swing.rel_volume(volumes) == pytest.approx(4.0), "3 null sessions is normal Yahoo data"
    volumes = [1000.0] * 4 + [None] * 16 + [4000.0]
    assert swing.rel_volume(volumes) is None, "an average of 4 bars is not a 20-day average"


def test_dollar_volume_is_the_mean_of_close_times_volume():
    assert swing.dollar_volume([10.0] * 20, [100.0] * 20) == pytest.approx(1000.0)
    # Only the last 20 bars count, so old history cannot prop up a name that has gone quiet.
    closes = [10.0] * 40
    volumes = [1_000_000.0] * 20 + [100.0] * 20
    assert swing.dollar_volume(closes, volumes) == pytest.approx(1000.0)


# ---- average daily range ------------------------------------------------------------------------

def test_adr20_pct_on_a_constant_two_percent_range_is_two_percent():
    # close 100, high 101, low 99 -> (101-99)/100 = 2% every bar, so the 20-bar mean is 2.0.
    highs, lows, closes = [101.0] * 25, [99.0] * 25, [100.0] * 25
    assert swing.adr_pct(highs, lows, closes) == pytest.approx(2.0)


def test_adr_is_the_bars_own_range_and_ignores_the_overnight_gap():
    # ramp(40) bars are 1.0 wide on a ~120 close, but their true range is 1.5 because of the gap.
    # ADR must report the former: it sizes a position, where ATR sizes a stop.
    highs, lows, closes = ramp(40)
    assert swing.adr_pct(highs, lows, closes) == pytest.approx(
        sum(1.0 / c for c in closes[-20:]) / 20 * 100, abs=1e-9
    )


# ---- the metrics() contract ---------------------------------------------------------------------

def _bars_kwargs(hlc):
    """ramp()/flat_series() return (highs, lows, closes); Bars takes (closes, highs, lows)."""
    highs, lows, closes = hlc
    return closes, highs, lows


def test_metrics_on_a_ten_bar_series_names_everything_it_could_not_measure():
    m = swing.metrics(Bars(*_bars_kwargs(ramp(10))))
    assert m["bars"] == 10
    assert m["price"] == pytest.approx(109.5)

    # What ten bars CAN answer: the last bar's location in its own range, and the distance from the
    # highest high seen so far. Everything else needs more history than exists.
    assert m["clv"] is not None
    assert m["pct_off_52w_high"] is not None

    expected_unmeasured = {
        "sma20", "sma50", "sma150", "sma200", "ema20", "ema50",
        "above_sma50", "above_sma200", "ma_stacked",
        "adr20_pct", "atr14", "atr14_pct", "adx14",
        "rel_volume", "dollar_volume_20d", "mom_20d", "mom_60d", "rsi14",
        "pct_vs_sma50", "pct_vs_sma200", "rel_strength_3mo", "ema20_slope_pct",
    }
    assert set(m["unmeasured"]) == expected_unmeasured
    for key in expected_unmeasured:
        assert m[key] is None, f"{key} is listed as unmeasured but carries a value"
    assert all(m[k] is None for k in m["unmeasured"]), "unmeasured must mirror the Nones exactly"


def test_metrics_returns_the_whole_contract_and_leaves_nothing_unmeasured_on_a_full_year():
    highs, lows, closes = ramp(260, step=0.4)
    volumes = [1_000_000.0] * 260
    bench = [100.0 + i * 0.1 for i in range(260)]
    m = swing.metrics(Bars(closes, highs, lows, volumes), bench_closes=bench)

    assert set(m) == {
        "price", "bars", "sma20", "sma50", "sma150", "sma200", "ema20", "ema50",
        "above_sma50", "above_sma200", "ma_stacked", "adr20_pct", "atr14", "atr14_pct",
        "adx14", "clv", "rel_volume", "dollar_volume_20d", "mom_20d", "mom_60d", "rsi14",
        "pct_off_52w_high", "pct_vs_sma50", "pct_vs_sma200", "rel_strength_3mo",
        "ema20_slope_pct", "unmeasured",
    }
    assert m["unmeasured"] == [], f"a clean 260-bar series measured nothing at: {m['unmeasured']}"
    assert m["above_sma50"] is True and m["above_sma200"] is True
    assert m["ma_stacked"] is True, "a monotonic ramp stacks 50 > 150 > 200 by construction"
    assert m["adx14"] > 25
    # The distinction the whole `unmeasured` field exists for: every ramp bar closes at mid-range,
    # so the CLV really is 0.0. A measured zero stays a zero and must NOT appear in `unmeasured`.
    assert m["clv"] == 0.0 and "clv" not in m["unmeasured"]
    assert m["rel_volume"] == pytest.approx(1.0)
    assert m["mom_20d"] > 0 and m["mom_60d"] > m["mom_20d"]
    assert m["rel_strength_3mo"] == pytest.approx(
        (closes[-1] / closes[-64] - 1) * 100 - (bench[-1] / bench[-64] - 1) * 100, abs=0.01
    )
    assert m["ema20_slope_pct"] > 0


def test_metrics_never_reports_a_price_above_its_own_52_week_high():
    # The max includes the last bar, so this can only ever be <= 0. Pinned because a +0.0001% here
    # would be read downstream as a breakout.
    for n in (10, 40, 300):
        m = swing.metrics(Bars(*_bars_kwargs(ramp(n))))
        assert m["pct_off_52w_high"] <= 0.0


def test_metrics_survives_a_series_that_carries_no_highs_or_lows():
    # This is the tests/test_portfolio_snapshot.py:21 stub shape, and the reason the swing set lives
    # outside market.summarize(): reading `.highs` off it must not raise. The OHLC metrics come back
    # unmeasured, and pct_off_52w_high falls back to the closes.
    m = swing.metrics(ClosesOnly([100.0 + i for i in range(300)]))
    assert m["bars"] == 300
    assert m["sma200"] is not None, "the close-only half of the set still works"
    for key in ("atr14", "atr14_pct", "adx14", "clv", "adr20_pct", "rel_volume", "dollar_volume_20d"):
        assert m[key] is None and key in m["unmeasured"]
    assert m["pct_off_52w_high"] == pytest.approx(0.0), "the last close IS the highest close here"


def test_metrics_truncates_history_at_a_hole_rather_than_splicing_across_it():
    # A null high in the middle of the series must not let bar N-1 and bar N+1 be treated as
    # consecutive: their combined move would be scored as one enormous true range. The clean tail
    # after the hole is 45 bars, which is still enough for ATR and ADX.
    highs, lows, closes = ramp(120)
    highs[74] = None
    m = swing.metrics(Bars(closes, highs, lows))
    assert m["atr14"] == pytest.approx(1.5), "the spliced-bar bug would inflate this well past 1.5"
    assert m["adx14"] == pytest.approx(100.0)

    # Truncate to fewer than 40 clean bars and ADX goes unmeasured rather than being computed on a
    # short tail and reported as if it had full history.
    highs2, lows2, closes2 = ramp(120)
    lows2[90] = None
    m2 = swing.metrics(Bars(closes2, highs2, lows2))
    assert m2["adx14"] is None and "adx14" in m2["unmeasured"]
    assert m2["atr14"] == pytest.approx(1.5), "29 clean bars is still enough for ATR(14)"


def test_metrics_reports_booleans_as_none_when_the_average_behind_them_is_missing():
    # above_sma200 with no 200-bar average is not False. A screen filtering on `not above_sma200`
    # would otherwise sweep in every newly listed name in the universe.
    m = swing.metrics(Bars(*_bars_kwargs(ramp(60))))
    assert m["above_sma50"] is True
    assert m["above_sma200"] is None and m["ma_stacked"] is None
    assert "above_sma200" in m["unmeasured"] and "ma_stacked" in m["unmeasured"]


# ---- series integrity: rejecting a mixed split basis before it is measured

def test_a_mixed_split_basis_is_detected_by_its_single_bar_jump():
    """Yahoo serves some reverse-split names as pre- and post-split bars INTERLEAVED in one array,
    with no adjustment applied. Measured against the live endpoint on 2026-08-21, BYND's two-year
    series oscillated 0.59 -> 17.85 -> 0.56 -> 16.98 -> 0.61 -> 15.84 while the adjclose/raw ratio
    sat at exactly 1.0000 on every bar, so the adjusted-close path could not save it. WETO carried a
    300x bar, DFNS 101x, TNON 46x.

    Metrics over such a series are not noisy, they are enormous — WETO measured mom_20d of 25,652%
    — so they sort straight to the top of any momentum ranking and dominate any percentile computed
    across the night. The series has to be rejected before it is measured, not filtered afterwards.
    """
    bynd = [0.59, 17.85, 0.56, 16.98, 0.61, 15.84] * 10
    assert swing.implausible_jump(bynd) is not None
    # The worst break is the DOWN leg 17.85 -> 0.56 (31.9x), not the 30.2x up-jump that opens the
    # series — which is the point of measuring the ratio in both directions.
    assert swing.worst_bar_ratio(bynd) == pytest.approx(17.85 / 0.56, rel=0.01)


def test_the_threshold_clears_real_single_session_movers():
    """The separation is measured, not theoretical. On the same night the largest single-bar move
    among genuine movers was 6.17x (FCUV), with the biotech readouts behind the real >100% 20-day
    gains at 1.31-1.64x; the smallest CORRUPT jump was 30.5x. A 10x line sits in that gap with room
    on both sides, so a real mover is never discarded as a data error.
    """
    steady = [10.0 * (1.01 ** i) for i in range(120)]
    assert swing.implausible_jump(steady) is None

    readout = [10.0] * 60 + [61.7] * 60            # 6.17x in one session — the measured real max
    assert swing.implausible_jump(readout) is None, "a genuine 6x session must survive"

    corrupt = [10.0] * 60 + [305.0] * 60           # 30.5x — the measured corrupt minimum
    assert swing.implausible_jump(corrupt) == pytest.approx(30.5)


def test_the_collapse_direction_is_caught_too():
    # A mixed-basis series always contains both directions; catching only the up-jump would let the
    # same corruption through whenever the series happens to end on the low basis.
    assert swing.implausible_jump([100.0, 100.0, 2.0, 2.0]) == pytest.approx(50.0)


def test_integrity_helpers_return_none_rather_than_a_verdict_when_there_is_nothing_to_compare():
    # None means "not measurable" here exactly as it does everywhere else in this module: a
    # single-bar series is not a clean series, and must not be reported as one.
    assert swing.worst_bar_ratio([]) is None
    assert swing.worst_bar_ratio([10.0]) is None
    assert swing.implausible_jump(None) is None
    assert swing.worst_bar_ratio([0.0, 0.0]) is None, "non-positive closes cannot form a ratio"


def test_the_offending_ratio_is_returned_so_a_rejection_can_be_named():
    # "SYM: 30.5x single-bar move" is auditable; "SYM: rejected" is not. The job writes this number
    # into its errors list.
    assert swing.implausible_jump([1.0, 40.0]) == pytest.approx(40.0)
