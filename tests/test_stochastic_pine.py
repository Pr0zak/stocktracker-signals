"""Golden values for _stochastic_k against Pine Script's ta.stoch semantics.

    ta.stoch(close, high, low, n) = 100 * (close - lowest(low, n)) / (highest(high, n) - lowest(low, n))

The window's extremes come from the BAR HIGHS AND LOWS, not from the closes. Taking them from the
closes makes %K read exactly 0 whenever the last close is the window's lowest close and exactly 100
whenever it is the highest — measured over 1-2 years of daily bars on six symbols, that is roughly
30% of bars against 0-1% for a true stochastic. The value is then shipped to the analyst in
summarize() as `stochastic_k` and persisted per verdict by memory.py, so a saturating input has been
feeding both the model and the track record.

The fixture is the same five bars as the app-side StochasticPineTest, so the two implementations are
pinned to one set of numbers rather than to each other.
"""

import math

import pytest

from app.market import Series, _stochastic_k, summarize

#   i | close | high | low
#   0 |  10   |  12  |  8
#   1 |  11   |  13  |  9
#   2 |  12   |  14  |  7
#   3 |  11   |  15  | 10
#   4 |  13   |  16  | 11
CLOSES = [10.0, 11.0, 12.0, 11.0, 13.0]
HIGHS = [12.0, 13.0, 14.0, 15.0, 16.0]
LOWS = [8.0, 9.0, 7.0, 10.0, 11.0]


def test_percent_k_matches_the_hand_computed_pine_value():
    # bars 2..4 → lowest low 7, highest high 16, close 13 → 100 * 6/9
    assert _stochastic_k(CLOSES, HIGHS, LOWS, period=3) == pytest.approx(100.0 * 6.0 / 9.0)


def test_the_close_basis_answer_saturates_where_the_true_one_does_not():
    """The regression these golden values exist to catch."""
    window = CLOSES[-3:]
    lo, hi = min(window), max(window)
    close_basis = 100.0 * (CLOSES[-1] - lo) / (hi - lo)
    assert close_basis == 100.0

    true_range = _stochastic_k(CLOSES, HIGHS, LOWS, period=3)
    assert 0.0 < true_range < 100.0


def test_a_monotonic_advance_never_pins_the_true_range_oscillator():
    closes = [100.0 + i for i in range(40)]
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]

    # 100 * (close - (close-13-1)) / ((close+1) - (close-13-1)) = 100 * 14/15
    assert _stochastic_k(closes, highs, lows) == pytest.approx(100.0 * 14.0 / 15.0)
    # ...whereas every warmed close-basis reading on this series is exactly 100.
    assert 100.0 * (closes[-1] - min(closes[-14:])) / (max(closes[-14:]) - min(closes[-14:])) == 100.0


def test_a_flat_window_reads_the_midpoint_rather_than_dividing_by_zero():
    flat = [50.0] * 20
    assert _stochastic_k(flat, flat, flat) == 50.0


# --- absence, not substitution -----------------------------------------------------------------


def test_missing_extremes_yield_none_not_a_close_basis_fallback():
    """scan_job.py's memory backfill builds a Series with highs=[] and lows=[].

    Falling back to closes there would write close-basis rows into the same `stochastic_k` column
    that live rows write true-range values to, and memory.py would then match one against the other.
    """
    assert _stochastic_k(CLOSES, None, None, period=3) is None
    assert _stochastic_k(CLOSES, [], [], period=3) is None


def test_a_nulled_bar_inside_the_window_yields_none():
    highs = list(HIGHS)
    highs[3] = None  # inside the last-3 window
    assert _stochastic_k(CLOSES, highs, LOWS, period=3) is None


def test_a_nulled_bar_outside_the_window_is_ignored():
    """The check is scoped to the window, not the whole history — an old hole must not blind today."""
    highs = list(HIGHS)
    highs[1] = None  # outside the last-3 window
    assert _stochastic_k(CLOSES, highs, LOWS, period=3) == pytest.approx(100.0 * 6.0 / 9.0)


def test_a_non_finite_bar_inside_the_window_yields_none():
    lows = list(LOWS)
    lows[-1] = math.nan
    assert _stochastic_k(CLOSES, HIGHS, lows, period=3) is None


def test_an_incoherent_bar_with_high_below_low_yields_none():
    highs = list(HIGHS)
    highs[-1] = 1.0  # below its own low of 11
    assert _stochastic_k(CLOSES, highs, LOWS, period=3) is None


def test_misaligned_extremes_yield_none_rather_than_a_guessed_alignment():
    assert _stochastic_k(CLOSES, HIGHS[:-1], LOWS, period=3) is None


def test_a_series_shorter_than_the_period_yields_none():
    assert _stochastic_k(CLOSES, HIGHS, LOWS, period=14) is None


# --- the summarize() contract ------------------------------------------------------------------


def _series(closes, highs, lows):
    return Series(
        symbol="TEST",
        closes=closes,
        opens=[None] * len(closes),
        volumes=[None] * len(closes),
        dates=[f"2026080{i}" for i in range(len(closes))],
        fifty_two_high=max(closes),
        fifty_two_low=min(closes),
        currency="USD",
        highs=highs,
        lows=lows,
    )


def test_summarize_reports_stochastic_k_when_bars_carry_extremes():
    closes = [100.0 + (i % 5) for i in range(30)]
    highs = [c + 2.0 for c in closes]
    lows = [c - 2.0 for c in closes]
    snap = summarize(_series(closes, highs, lows), None)
    assert snap["stochastic_k"] is not None


def test_summarize_reports_none_when_the_series_has_no_extremes():
    """The exact shape scan_job.py:437 builds for the memory backfill."""
    closes = [100.0 + (i % 5) for i in range(30)]
    snap = summarize(_series(closes, [], []), None)
    assert snap["stochastic_k"] is None
    # The key must still be present — memory.py's schema and _SANDBOX_TECH_KEYS both expect it, and
    # an absent key is not the same claim as a null one.
    assert "stochastic_k" in snap


def test_summarize_survives_a_stub_series_exposing_only_closes():
    """tests/test_portfolio_snapshot.py monkeypatches fetch_series with exactly this stub.

    swing.py's module docstring records that reading `.highs` directly off summarize()'s argument
    kills that test with an AttributeError, in a test that has nothing to do with oscillators.
    """

    class _Stub:
        symbol = "STUB"
        closes = [100.0] * 60
        dates = ["20260801"] * 60
        currency = "USD"
        fifty_two_high = 100.0
        fifty_two_low = 100.0

    snap = summarize(_Stub(), None)
    assert snap["stochastic_k"] is None
