"""The dip radar has to say what it REJECTED, and why — and never call missing data "no dip".

Two incidents sit behind this file.

The first is structural: `_dip_tier` returned a tier or None, so a name that missed a threshold by
0.8 of a point and a name trading at its high were both simply ABSENT from the results. A screener
that shows only its winners is something you have to take on faith; one that shows what it turned
down, with the number it turned it down on, is something you can check against your own eyes.

The second is the live app bug. `DipListScreen` renders "No dips right now — nothing you track is
notably off its highs" whenever `scan?.results` is null, which includes the case where the FETCH
ITSELF FAILED. "We looked and the market is calm" and "we could not look" rendered identically, and
the reassuring one was the lie. The backend contract these tests pin down:

  * every reject carries a reason naming its own numbers — never a bare "did not qualify";
  * NEAR MISSES (within `_NEAR_MISS_MARGIN_PP` of the nearest tier) are a separate bucket from the
    names that are nowhere near a dip;
  * a symbol whose data failed to fetch is UNMEASURED, never "no dip";
  * the five counters partition the scanned set exactly, so the totals can be checked by adding up;
  * a scan that RAN and found nothing has empty reject lists, while "there is no scan" answers with
    nulls and `scan_available: False`. Those two must never render the same.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from app import main
from app.scan_job import (
    _NEAR_MISS_MARGIN_PP,
    _dip_tier,
    classify_dip,
    dip_verdicts,
)


def _closes(pct_off_high: float, bars: int = 63) -> list[float]:
    """A price series whose last bar sits `pct_off_high`% under a flat 100.00 three-month high."""
    return [100.0] * (bars - 1) + [round(100.0 * (1 + pct_off_high / 100.0), 4)]


def _row(symbol: str, **kw) -> dict:
    """A scan result row shaped like `_score` returns one."""
    base = {"symbol": symbol, "signal": "hold"}
    base.update(kw)
    return base


# ------------------------------------------------------------------ near misses name the number

def test_a_name_just_under_a_tier_is_a_near_miss_that_names_what_it_missed_by():
    """THE case. 4.2% off its three-month high needs 5% — 0.8 of a point away. Before this it was
    indistinguishable from a name sitting at its all-time high: both were simply not in the list."""
    out = _dip_tier(_closes(-4.2), -6.0, False, False)
    assert out["dip"] is None
    assert out["dip_near_miss"] is True
    assert out["dip_gap_pp"] == 0.8
    reason = out["dip_reject_reason"]
    assert "4.2%" in reason and "3-month high" in reason
    assert "5%" in reason, reason          # the threshold it missed
    assert "0.8" in reason, reason         # and by how much


def test_a_near_miss_on_the_52_week_measure_names_the_20_percent_threshold():
    """A name 18.6% off its 52-week high but barely off its 3-month one misses mega_dip, not
    pullback_5, and the reason has to name the measure that was actually closest."""
    out = _dip_tier(_closes(-1.0), -18.6, False, False)
    assert out["dip_near_miss"] is True
    assert out["dip_gap_pp"] == 1.4
    assert "52-week high" in out["dip_reject_reason"]
    assert "20%" in out["dip_reject_reason"]


def test_a_name_hovering_just_above_its_200_week_line_is_a_near_miss():
    """mungbeans' signal is the cross BELOW the 200-week line, so 0.9% above it is one ordinary
    session from qualifying — exactly what the near-miss bucket exists to surface."""
    out = _dip_tier(_closes(-0.5), -2.0, False, False, {"price_vs_200w_sma_pct": 0.9, "rsi_14w": 44.0})
    assert out["dip_near_miss"] is True
    assert out["dip_gap_pp"] == 0.9
    assert "200-week line" in out["dip_reject_reason"]


def test_the_near_miss_margin_is_the_only_thing_deciding_the_boundary():
    """Pinned so the constant stays load-bearing: a name exactly at the margin is a near miss and one
    a tenth of a point past it is not. If someone widens the margin, this test moves with it."""
    at = _dip_tier(_closes(-(5.0 - _NEAR_MISS_MARGIN_PP)), None, False, False)
    past = _dip_tier(_closes(-(5.0 - _NEAR_MISS_MARGIN_PP - 0.1)), None, False, False)
    assert at["dip_gap_pp"] == pytest.approx(_NEAR_MISS_MARGIN_PP)
    assert at["dip_near_miss"] is True
    assert past["dip_near_miss"] is False


# ------------------------------------------------------------------ nowhere near still gets a reason

def test_a_name_nowhere_near_a_dip_gets_a_reason_and_is_not_a_near_miss():
    """The flat middle of the watchlist. It still has to say WHY — with its own numbers on every
    measure the screener looked at — but it must not be dressed up as almost-qualifying."""
    out = _dip_tier(_closes(-1.0), -3.0, False, False, {"price_vs_200w_sma_pct": 42.0, "rsi_14w": 61.0})
    assert out["dip"] is None
    assert out["dip_near_miss"] is False
    assert out["dip_gap_pp"] == 4.0
    reason = out["dip_reject_reason"]
    for fragment in ("1.0% off its 3-month high", "3.0% off its 52-week high",
                     "42.0% above its 200-week line", "14-week RSI 61", "no dip on any measure"):
        assert fragment in reason, reason


def test_a_name_at_its_high_reads_as_at_its_high_not_as_minus_zero_percent_off():
    out = _dip_tier(_closes(0.0), -0.0, False, False)
    assert "at its 3-month high" in out["dip_reject_reason"]
    assert "-0.0%" not in out["dip_reject_reason"]


def test_no_reject_reason_is_ever_a_bare_did_not_qualify():
    """Every reason must be grounded in an actual measurement, i.e. must contain a number."""
    for out in (
        _dip_tier(_closes(-4.2), -6.0, False, False),
        _dip_tier(_closes(-1.0), -3.0, False, False, {"price_vs_200w_sma_pct": 42.0, "rsi_14w": 61.0}),
        _dip_tier(_closes(-0.1), None, None, None),
    ):
        assert any(ch.isdigit() for ch in out["dip_reject_reason"]), out["dip_reject_reason"]


def test_a_qualifying_name_carries_no_reject_fields_at_all():
    """`dip_near_miss` is None, not False, on a qualifier. A client rendering "near miss: no" for a
    name that actually QUALIFIED would be the same defect pointing the other way."""
    out = _dip_tier(_closes(-12.0), -14.0, False, False)
    assert out["dip"] == "pullback_10"
    assert out["dip_reject_reason"] is None
    assert out["dip_near_miss"] is None
    assert out["dip_gap_pp"] is None
    assert out["dip_measured"] is True


# ------------------------------------------------------------------ unmeasured is not "no dip"

def test_a_symbol_whose_data_could_not_be_fetched_is_unmeasured_never_no_dip():
    """THE defect this task exists to fix. A failed fetch used to fall out of the results entirely,
    which on the screen is the same picture as a calm tape."""
    rejects, counts = dip_verdicts([_row("ACME", error="not enough history")])
    assert counts["unmeasured"] == 1
    assert counts["nowhere_near"] == 0 and counts["near_miss"] == 0 and counts["qualified"] == 0
    assert [r["symbol"] for r in rejects["unmeasured"]] == ["ACME"]
    assert rejects["nowhere_near"] == []
    assert "not enough history" in rejects["unmeasured"][0]["reason"]


def test_a_symbol_with_no_usable_price_history_is_unmeasured_too():
    """No bars at all means no dip criterion could be evaluated. The distance fields stay None —
    a 0.0 there would read as "sitting exactly on its 3-month high", which is a price claim."""
    out = _dip_tier([], None, None, None)
    assert out["dip"] is None
    assert out["dip_measured"] is False
    assert out["pct_off_recent_high"] is None
    assert classify_dip(_row("VOID", **out)) == "unmeasured"


def test_a_row_with_no_dip_measurement_at_all_falls_to_unmeasured_not_to_no_dip():
    """Defensive: an unrecognised row shape must default to the honest bucket, not the calm one."""
    assert classify_dip({"symbol": "WAT"}) == "unmeasured"


def test_the_unmeasured_reason_never_leaks_the_upstream_url():
    """These reasons are rendered in the app, and httpx puts the full request URL in its errors."""
    err = "Server error '502 Bad Gateway' for url 'https://query1.finance.yahoo.com/v8/chart/ACME?x=1'"
    rejects, _ = dip_verdicts([_row("ACME", error=err)])
    reason = rejects["unmeasured"][0]["reason"]
    assert "yahoo.com" not in reason and "http" not in reason
    assert "502" in reason


# ------------------------------------------------------------------ the counters must add up

def _scan_rows() -> list[dict]:
    """One of each: a qualifier, a near miss, a nowhere-near, and a failed fetch."""
    return [
        _row("DIPCO", **_dip_tier(_closes(-12.0), -14.0, False, False)),
        _row("CLOSE", **_dip_tier(_closes(-4.2), -6.0, False, False)),
        _row("FLAT", **_dip_tier(_closes(-1.0), -3.0, False, False,
                                 {"price_vs_200w_sma_pct": 42.0, "rsi_14w": 61.0})),
        _row("GONE", error="not enough history"),
    ]


def test_the_counters_partition_the_scanned_set_exactly():
    """Five different facts, and the only way to check them is that four of them sum to the fifth."""
    rejects, counts = dip_verdicts(_scan_rows())
    assert counts == {"scanned": 4, "qualified": 1, "near_miss": 1, "nowhere_near": 1, "unmeasured": 1}
    assert counts["scanned"] == (
        counts["qualified"] + counts["near_miss"] + counts["nowhere_near"] + counts["unmeasured"]
    )
    assert sum(len(v) for v in rejects.values()) == counts["scanned"] - counts["qualified"]


def test_the_reject_lists_never_contain_a_qualifying_name():
    """The whole point of the field name: nothing in `dip_rejects` is a dip."""
    rejects, _ = dip_verdicts(_scan_rows())
    listed = {r["symbol"] for bucket in rejects.values() for r in bucket}
    assert "DIPCO" not in listed
    assert listed == {"CLOSE", "FLAT", "GONE"}


def test_every_reject_carries_a_reason():
    rejects, _ = dip_verdicts(_scan_rows())
    for bucket, rows in rejects.items():
        for r in rows:
            assert r["reason"], f"{bucket}/{r['symbol']} has no reason"


# ------------------------------------------------------------------ empty scan vs no scan at all

def test_an_empty_scan_produces_empty_reject_lists_not_null():
    """A scan that RAN over nothing is a real (if boring) claim, and its lists are real lists."""
    rejects, counts = dip_verdicts([])
    assert rejects == {"near_miss": [], "nowhere_near": [], "unmeasured": []}
    assert counts == {"scanned": 0, "qualified": 0, "near_miss": 0, "nowhere_near": 0, "unmeasured": 0}


def _latest(tmp_path, monkeypatch, payload: str | None):
    """Point the endpoint at an isolated scan file (absent when `payload` is None) and call it."""
    path = tmp_path / "scan_latest.json"
    if payload is not None:
        path.write_text(payload)
    monkeypatch.setattr(main, "LATEST", path)
    return asyncio.run(main.scan_latest())


def test_a_scan_that_ran_and_found_nothing_is_available_with_empty_lists(tmp_path, monkeypatch):
    rejects, counts = dip_verdicts([])
    body = _latest(tmp_path, monkeypatch, json.dumps(
        {"generated_at": 1.0, "results": [], "flips": [], "dip_alerts": [],
         "dip_rejects": rejects, "dip_counts": counts},
    ))
    assert body["scan_available"] is True
    assert body["results"] == []
    assert body["dip_rejects"] == {"near_miss": [], "nowhere_near": [], "unmeasured": []}
    assert body["dip_counts"]["scanned"] == 0


def test_no_scan_at_all_is_distinguishable_from_a_scan_that_found_nothing(tmp_path, monkeypatch):
    """The live app bug. `results ?: emptyList()` collapsed "we could not look" into "the market is
    calm", so the client needs a flag it can branch on and nulls rather than empty lists."""
    body = _latest(tmp_path, monkeypatch, None)
    assert body["scan_available"] is False
    assert body["results"] is None
    assert body["dip_rejects"] is None
    assert body["dip_counts"] is None
    assert body["generated_at"] is None
    assert body["unavailable_reason"]


def test_an_unreadable_scan_file_reports_unavailable_rather_than_empty(tmp_path, monkeypatch):
    """A half-written or corrupt file is a failure to read the scan, not an empty scan."""
    body = _latest(tmp_path, monkeypatch, "{not json")
    assert body["scan_available"] is False
    assert body["results"] is None
    assert "could not be read" in body["unavailable_reason"]


def test_a_scan_written_before_this_feature_reports_null_rejects_not_zeros(tmp_path, monkeypatch):
    """The deployed CT is serving last night's file until the next run. Its missing counters must
    read as "unknown", never as "0 near misses" — that would be a confident claim about a scan that
    never measured one."""
    body = _latest(tmp_path, monkeypatch, json.dumps({"generated_at": 1.0, "results": [], "flips": []}))
    assert body["scan_available"] is True
    assert body["dip_rejects"] is None
    assert body["dip_counts"] is None
