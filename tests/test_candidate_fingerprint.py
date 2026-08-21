"""What the model was SHOWN, recorded so a claim about it can be checked.

Every record this account keeps describes what was DECIDED -- the trade log, the posture, the review
verdict, the settings changelog. Nothing recorded what was SEEN.

Measured 2026-08-21: an order cited "rsi14w 65" for GLD, and the review model rejected the
surrounding logic partly on the grounds that GLD "appears nowhere in the candidate data". Rebuilding
that day's pool by hand showed GLD present with an rsi_14w of 58.6 -- so the analyst's number was
wrong AND the reviewer's reason was wrong, and neither could be established from disk. A critic whose
claims cannot be checked is another unverified source, not an audit.
"""
from __future__ import annotations

from app.sandbox_job import candidate_fingerprint

GLD = {
    "symbol": "gld", "source": "core", "price": 423.36, "exposure_group": "GOLD",
    "technicals": {"rsi14": 71.16, "pct_vs_sma50": 10.53, "pct_off_52w_high": -16.94,
                   "stochastic_k": 100.0, "macd_hist": 3.14, "golden_cross": True},
    "long_term": {"rsi_14w": 58.6, "mayer_multiple": 1.02, "zone": "above",
                  "price_vs_200w_sma_pct": 58.2, "cagr_3y_pct": 33.5, "below_line": False},
    # The verbose blocks a real row carries. Kept realistic so the size assertion below measures
    # something true rather than a ratio against an already-trimmed fixture.
    "gap": {"gap_pct": 1.27, "direction": "up", "size_bucket": "1-2%", "volume_ratio": 1.56,
            "catalyst_likely": True, "filled_today": False,
            "historical_fill_rate_10d_pct": 69.5, "measured_edge": "none"},
    "track_record": {"analogues": {"n": 40, "median_fwd_20d_pct": 1.06,
                                   "positive_rate_20d": 0.55, "median_fwd_5d_pct": -0.68,
                                   "vs_benchmark": {"n": 39, "median_excess_20d_pct": 0.72,
                                                    "beat_rate_20d": 0.64}}},
}


def test_it_captures_the_numbers_a_reason_actually_quotes():
    # The exact case: "rsi14w 65" was claimed, 58.6 was shown. That comparison is only possible
    # because the shown value is now on disk.
    r = candidate_fingerprint([GLD])[0]
    assert r["rsi_14w"] == 58.6
    assert r["mayer_multiple"] == 1.02
    assert r["price"] == 423.36
    assert r["rsi14"] == 71.16          # the daily one too -- both get quoted, and they differ


def test_presence_alone_settles_the_other_half_of_that_dispute():
    # The reviewer's claim was that GLD was absent entirely. A symbol list is enough to settle it.
    syms = [r["symbol"] for r in candidate_fingerprint([GLD])]
    assert syms == ["GLD"]              # and normalised, so a case difference cannot fake an absence


def test_the_zone_string_is_kept_even_though_it_is_not_a_number():
    # "long_term zone above" was itself one of the quoted claims, so it has to be checkable.
    assert candidate_fingerprint([GLD])[0]["zone"] == "above"


def test_it_is_a_subset_not_the_whole_payload():
    # A full candidate row is ~230 tokens and 89 of them is ~80KB a tick. Verbose fields that no
    # reason has ever cited are dropped on purpose.
    import json
    r = candidate_fingerprint([GLD])[0]
    assert "track_record" not in r          # the largest block, never quoted in a reason
    assert "macd_hist" not in r
    assert "golden_cross" not in r
    # Size is the claim, not key count: the point is bytes on disk per tick, and a flat row of
    # scalars can carry more KEYS than a nested one while still being much smaller.
    assert len(json.dumps(r)) < len(json.dumps(GLD)) / 2      # measured at ~30% of a real row


def test_a_candidate_missing_a_field_omits_it_rather_than_writing_a_null():
    # A null would be indistinguishable from a measured zero once it is on disk, which is the same
    # absent-versus-zero confusion this codebase keeps having to fix.
    thin = {"symbol": "IBIT", "source": "core", "price": 67.0, "technicals": {"rsi14": 50.0}}
    r = candidate_fingerprint([thin])[0]
    assert r["rsi14"] == 50.0
    assert "mayer_multiple" not in r


def test_an_empty_pool_yields_an_empty_list():
    assert candidate_fingerprint([]) == []
    assert candidate_fingerprint(None) == []
