"""summarize()'s volatility keys: atr14 and atr14_pct.

Until 2026-08-31 the LLM-facing snapshot carried no magnitude of movement at all, so the analyst
placed entry zones, stops and targets with no idea whether a name travels 0.8% or 7% in an ordinary
session — while PLAN_SYSTEM already asked it to sanity-check a 1.5 risk:reward.

The cases that matter here are the absent ones. summarize() is called with duck-typed stubs exposing
only `closes` and with scan_job's truncated Series whose extremes are empty lists, and in both the
answer must be null rather than a range derived from closes.
"""

import math

import pytest

from app.market import Series, _round_sig, summarize
from app.swing import clean_tail_atr


def _series(closes, highs=None, lows=None):
    return Series(
        symbol="TEST",
        closes=closes,
        opens=[None] * len(closes),
        volumes=[None] * len(closes),
        dates=[f"2026{(i % 12) + 1:02d}{(i % 28) + 1:02d}" for i in range(len(closes))],
        fifty_two_high=max(closes),
        fifty_two_low=min(closes),
        currency="USD",
        highs=highs if highs is not None else [],
        lows=lows if lows is not None else [],
    )


def _ohlc(n, start=100.0, step=0.5, band=1.0):
    closes = [start + i * step for i in range(n)]
    return closes, [c + band for c in closes], [c - band for c in closes]


# --- the measurement ---------------------------------------------------------------------------


def test_atr_is_published_when_the_bars_carry_extremes():
    closes, highs, lows = _ohlc(40)
    snap = summarize(_series(closes, highs, lows), None)
    assert snap["atr14"] is not None
    assert snap["atr14"] > 0
    assert snap["atr14_pct"] == pytest.approx(snap["atr14"] / closes[-1] * 100.0, abs=0.01)


def test_atr14_matches_the_swing_helper_exactly():
    """One number, one implementation — the snapshot and the mechanical scan must not diverge."""
    closes, highs, lows = _ohlc(40)
    snap = summarize(_series(closes, highs, lows), None)
    assert snap["atr14"] == pytest.approx(clean_tail_atr(highs, lows, closes), rel=1e-3)


def test_a_hole_in_old_history_does_not_cost_the_measurement():
    """atr() refuses the whole series over one bad bar; the clean tail is why the wrapper exists."""
    closes, highs, lows = _ohlc(60)
    highs[10] = None
    snap = summarize(_series(closes, highs, lows), None)
    assert snap["atr14"] is not None


def test_a_hole_in_the_latest_bar_yields_no_measurement():
    """An ATR from the previous session must never sit beside this session's price."""
    closes, highs, lows = _ohlc(60)
    highs[-1] = None
    assert summarize(_series(closes, highs, lows), None)["atr14"] is None


# --- absence, not substitution -----------------------------------------------------------------


def test_a_closes_only_stub_yields_null_without_raising():
    class _Stub:
        symbol = "STUB"
        closes = [100.0] * 60
        dates = ["20260801"] * 60
        currency = "USD"
        fifty_two_high = 100.0
        fifty_two_low = 100.0

    snap = summarize(_Stub(), None)
    assert snap["atr14"] is None
    assert snap["atr14_pct"] is None
    # Present-and-null, not missing: an absent key is not the same claim as a null one.
    assert "atr14" in snap and "atr14_pct" in snap


def test_the_truncated_backfill_series_yields_null():
    """The exact shape scan_job.py builds when the source series carries no extremes."""
    closes, _, _ = _ohlc(60)
    assert summarize(_series(closes, [], []), None)["atr14"] is None


def test_none_extremes_are_absorbed_rather_than_raising():
    """swing._align() takes len() with no guard, so a None reaching it is a TypeError."""
    assert clean_tail_atr(None, None, [100.0] * 60) is None


def test_fewer_than_fifteen_clean_bars_yields_null():
    """ATR(14) needs period + 1 bars: fourteen true ranges take fifteen bars."""
    c14, h14, l14 = _ohlc(14)
    c15, h15, l15 = _ohlc(15)
    assert clean_tail_atr(h14, l14, c14) is None
    assert clean_tail_atr(h15, l15, c15) is not None


# --- the two cases the adversarial review caught --------------------------------------------


def test_a_sub_penny_asset_keeps_its_magnitude_instead_of_rounding_to_zero():
    """Decimal rounding would publish 0.0 — a measured flat range — for a name moving ~6% a day.

    round(7.4e-07, 4) is 0.0. The snapshot would then carry atr14 0.0 next to an atr14_pct of about
    6, two contradictory claims in one dict, and any check reading atr14 would conclude there is no
    range to size a stop against. Significant figures hold the magnitude at every price scale.
    """
    closes = [0.0000120 + i * 0.0000001 for i in range(40)]
    highs = [c * 1.03 for c in closes]
    lows = [c * 0.97 for c in closes]
    snap = summarize(_series(closes, highs, lows), None)

    assert snap["atr14"] is not None
    assert snap["atr14"] > 0.0, "a sub-penny ATR must not round away to a measured zero"
    assert snap["atr14_pct"] > 1.0
    # And the two keys must agree with each other.
    assert snap["atr14"] / closes[-1] * 100.0 == pytest.approx(snap["atr14_pct"], abs=0.5)


def test_a_mixed_split_basis_is_refused_rather_than_quoted():
    """ATR is the metric a split break corrupts hardest — a pre-split bar beside a post-split one
    reads as one enormous session's range. swing.implausible_jump guards the market-scan path but is
    never reached from summarize(), so without this the analyst is handed an ATR near 100% of price
    as fact, and any stop would be measured against it.
    """
    closes, highs, lows = _ohlc(40, start=100.0, step=0.0)
    # A 10-for-1 split applied to the tail only: the break is not a price move.
    for i in range(20, 40):
        closes[i] /= 10.0
        highs[i] /= 10.0
        lows[i] /= 10.0
    snap = summarize(_series(closes, highs, lows), None)
    assert snap["atr14"] is None
    assert snap["atr14_pct"] is None


# --- a measured zero is a measurement ------------------------------------------------------------


def test_a_genuinely_flat_series_reports_zero_not_null():
    closes = [50.0] * 40
    snap = summarize(_series(closes, list(closes), list(closes)), None)
    assert snap["atr14"] == 0.0
    assert snap["atr14_pct"] == 0.0


def test_a_zero_price_drops_the_percentage_but_keeps_the_atr():
    closes, highs, lows = _ohlc(40)
    closes[-1] = 0.0
    snap = summarize(_series(closes, highs, lows), None)
    assert snap["atr14_pct"] is None


# --- the rounding helper -------------------------------------------------------------------------


def test_round_sig_keeps_small_magnitudes_and_is_absent_on_garbage():
    assert _round_sig(7.3712e-07, 4) == pytest.approx(7.371e-07, rel=1e-9)
    assert _round_sig(123.456789, 4) == pytest.approx(123.5)
    assert _round_sig(0.0, 4) == 0.0
    assert _round_sig(None) is None
    assert _round_sig(math.nan) is None
    assert _round_sig(math.inf) is None
