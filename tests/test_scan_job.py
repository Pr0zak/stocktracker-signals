"""OPS-1 — an all-failure scan night must not erase the diff baseline.

Verified in the CT 237 journal on 2026-09-10: every analyst call failed that night, yet `scan_job`
logged `scanned 54 · unmeasured 54`, exited 0, and overwrote data/scan_latest.json with the empty
night. On 2026-09-11, `_prev_state()` kept only rows carrying a "signal" key, so the previous state
was `{}` — and `_diff_vs_prev`'s `dip_new` check (`r.get("dip") is not None and p.get("dip") !=
r.get("dip")`) fired for every symbol that had been sitting in a dip tier for weeks. The phone got
"Good time to add" pushes for names that had been dipping for a month, not names that had just
started.

The fix has two load-bearing pieces, both exercised here:

  * `_prev_state()` falls back to the persisted `last_measured` carry-forward map when the
    immediately-previous run measured nothing at all.
  * `_diff_vs_prev` — the exact function that mis-fired on 2026-09-11 — must not mark a dip as new
    when the "previous" reading it is handed came from that fallback and already carries the same
    dip.

`_finalize_payload` is the third piece: it is the thing that decides, on a total-failure night, to
keep last night's payload (with `last_run_failed`/`last_error` stamped on it, mirroring
app/macro_job.py's degraded-flag pattern) instead of publishing tonight's empty cross-section —
which is what makes the `_prev_state()` fallback unnecessary on the very next night, and load-bearing
only across TWO or more consecutive all-failure nights, or when the file predates this fix.
"""
from __future__ import annotations

import json

from app import scan_job


def _measured_row(symbol: str, *, dip: str | None = None, signal: str = "hold", **over) -> dict:
    """A scan result row shaped like `_score` returns one — carries a real "signal" key and
    `dip_measured: True` so `classify_dip` files it as qualified/nowhere_near rather than
    unmeasured (see app/scan_job.py's `classify_dip`)."""
    row = {"symbol": symbol, "signal": signal, "squeeze": None, "below_200wma": False, "dip": dip,
           "dip_measured": True}
    row.update(over)
    return row


def _error_row(symbol: str, msg: str = "analyst call failed") -> dict:
    """A row shaped like `one()` returns when `_score` raised — no "signal" key at all."""
    return {"symbol": symbol, "error": msg}


def _write_latest(monkeypatch, tmp_path, blob: dict):
    path = tmp_path / "scan_latest.json"
    path.write_text(json.dumps(blob))
    monkeypatch.setattr(scan_job, "LATEST", path)
    return path


# --------------------------------------------------------------- _prev_state's carried fallback

def test_prev_state_falls_back_to_last_measured_after_an_all_error_night(tmp_path, monkeypatch):
    """THE case: last night's `results` are all error rows (no "signal" key on any of them), so
    the normal path returns {}. `last_measured` — carried from the last GOOD night — is what's
    left, and it is what tonight's diff must be run against."""
    _write_latest(monkeypatch, tmp_path, {
        "results": [_error_row("ACME"), _error_row("GLOBEX")],
        "last_measured": {
            "ACME": {"signal": "buy", "squeeze": None, "below_200wma": False, "dip": "pullback_10"},
            "GLOBEX": {"signal": "hold", "squeeze": "fuel", "below_200wma": True, "dip": None},
        },
        "last_run_failed": True,
        "last_error": "every symbol failed to measure",
    })

    prev = scan_job._prev_state()

    assert prev["ACME"]["dip"] == "pullback_10"
    assert prev["GLOBEX"]["squeeze"] == "fuel"


def test_prev_state_prefers_real_results_over_the_carried_map_when_both_exist(tmp_path, monkeypatch):
    """The carried map is a FALLBACK, not a substitute for a night that actually measured
    something — a fresher real reading must win over a stale carried one."""
    _write_latest(monkeypatch, tmp_path, {
        "results": [_measured_row("ACME", dip="pullback_5")],
        "last_measured": {"ACME": {"signal": "hold", "squeeze": None, "below_200wma": False,
                                    "dip": "pullback_10"}},
    })

    prev = scan_job._prev_state()

    assert prev["ACME"]["dip"] == "pullback_5"


def test_prev_state_is_empty_with_no_file_at_all(tmp_path, monkeypatch):
    monkeypatch.setattr(scan_job, "LATEST", tmp_path / "nope.json")
    assert scan_job._prev_state() == {}


def test_prev_state_with_an_all_error_night_and_no_carried_map_is_still_empty(tmp_path, monkeypatch):
    """Without the carry-forward this IS the 2026-09-10 failure mode: nothing to fall back to, so
    the baseline really is empty. Pinned so the fallback reads as ADDITIVE — it recovers a baseline
    that was actually recorded, it does not manufacture one that never existed."""
    _write_latest(monkeypatch, tmp_path, {"results": [_error_row("ACME")]})
    assert scan_job._prev_state() == {}


def test_prev_state_ignores_a_carried_map_when_last_night_measured_something(tmp_path, monkeypatch):
    """The fallback is gated on "measured nothing", not "always prefer the carried map merged with
    results" — a stale carried entry for a symbol dropped from tonight's results must not leak in
    when the run was otherwise a real, successful measurement."""
    _write_latest(monkeypatch, tmp_path, {
        "results": [_measured_row("ACME", dip=None)],
        "last_measured": {"ZOMBIE": {"signal": "buy", "squeeze": None, "below_200wma": False,
                                      "dip": "pullback_10"}},
    })

    prev = scan_job._prev_state()

    assert "ZOMBIE" not in prev
    assert prev["ACME"]["dip"] is None


# --------------------------------------------------------------------------- the diff itself

def test_a_standing_dip_survives_an_all_error_night_via_the_carried_map():
    """The exact incident, isolated to the diff function itself. ACME has been in a
    `pullback_10` dip for weeks; `prev` here is what `_prev_state()`'s fallback hands back after an
    all-error night. Tonight ACME measures the SAME dip again — this must not be `dip_new`."""
    prev = {"ACME": {"signal": "hold", "squeeze": None, "below_200wma": False, "dip": "pullback_10"}}

    out = scan_job._diff_vs_prev(_measured_row("ACME", dip="pullback_10"), prev)

    assert out["dip_new"] is False
    assert out["prev_dip"] == "pullback_10"


def test_a_genuinely_new_dip_is_still_flagged_against_the_carried_map():
    """The fallback must not swallow real signal either: ACME was not dipping before, and now is —
    fed via the same carried-map shape, this must still fire."""
    prev = {"ACME": {"signal": "hold", "squeeze": None, "below_200wma": False, "dip": None}}

    out = scan_job._diff_vs_prev(_measured_row("ACME", dip="pullback_10"), prev)

    assert out["dip_new"] is True


def test_end_to_end_prev_state_into_the_diff_across_a_kept_failed_night(tmp_path, monkeypatch):
    """Wires `_prev_state()`'s fallback straight into `_diff_vs_prev`, the way `run_scan`'s `one()`
    does, so the whole OPS-1 fix is pinned as one behavior rather than two independently-passing
    halves."""
    _write_latest(monkeypatch, tmp_path, {
        "results": [_error_row("ACME"), _error_row("GLOBEX")],
        "last_measured": {
            "ACME": {"signal": "hold", "squeeze": None, "below_200wma": False, "dip": "pullback_10"},
            "GLOBEX": {"signal": "hold", "squeeze": None, "below_200wma": False, "dip": None},
        },
        "last_run_failed": True,
        "last_error": "every symbol failed to measure",
    })

    prev = scan_job._prev_state()
    acme = scan_job._diff_vs_prev(_measured_row("ACME", dip="pullback_10"), prev)
    globex = scan_job._diff_vs_prev(_measured_row("GLOBEX", dip=None), prev)

    assert acme["dip_new"] is False, "a dip that was already standing must not read as new"
    assert globex["dip_new"] is False


# --------------------------------------------------------- _finalize_payload's total-failure guard

def test_a_total_failure_night_keeps_the_previous_payload_and_stamps_it_degraded():
    """The other half of the fix: when nothing was measured, the good payload from before is kept
    byte-for-byte (its `results`, `dip_counts`, everything) except for the degraded stamp."""
    prior = {
        "generated_at": 111.0, "results": [_measured_row("ACME", dip="pullback_10")],
        "dip_counts": {"scanned": 1, "qualified": 1, "near_miss": 0, "nowhere_near": 0, "unmeasured": 0},
        "last_measured": {"ACME": {"signal": "hold", "squeeze": None, "below_200wma": False,
                                    "dip": "pullback_10"}},
    }
    results = [_error_row("ACME", "every provider timed out")]
    _, counts = scan_job.dip_verdicts(results)

    out = scan_job._finalize_payload(
        results=results, dip_rejects={"near_miss": [], "nowhere_near": [], "unmeasured": []},
        dip_counts=counts, date_alerts=[], scored=0, seeded={}, prior_payload=prior,
    )

    assert out["results"] == prior["results"]           # untouched — not tonight's empty scan
    assert out["dip_counts"] == prior["dip_counts"]
    assert out["last_run_failed"] is True
    assert "every provider timed out" in out["last_error"]


def test_a_total_failure_night_still_updates_last_measured_when_returning_fresh_payload_shape():
    """`last_measured` merges old + new even in the branches that don't end up publishing the
    merged payload verbatim — nothing here should ever go backwards or drop a symbol silently."""
    prior = {"last_measured": {"ACME": {"signal": "hold", "squeeze": None,
                                         "below_200wma": False, "dip": "pullback_10"}}}
    results = [_measured_row("ACME", dip="pullback_5"), _measured_row("GLOBEX", dip=None)]
    _, counts = scan_job.dip_verdicts(results)

    out = scan_job._finalize_payload(
        results=results, dip_rejects={"near_miss": [], "nowhere_near": [], "unmeasured": []},
        dip_counts=counts, date_alerts=[], scored=0, seeded={}, prior_payload=prior,
    )

    assert out["last_run_failed"] is False
    assert out["last_measured"]["ACME"]["dip"] == "pullback_5"    # fresh reading wins
    assert out["last_measured"]["GLOBEX"]["dip"] is None


def test_a_partial_failure_is_not_a_total_failure():
    """One bad fetch among many successes must publish tonight's real results, not fall back to
    keeping last night's payload — `total_failure` requires EVERY symbol to be unmeasured."""
    prior = {"results": [_measured_row("ACME", dip=None)], "last_measured": {}}
    results = [_measured_row("ACME", dip="pullback_10"), _error_row("GLOBEX")]
    _, counts = scan_job.dip_verdicts(results)

    out = scan_job._finalize_payload(
        results=results, dip_rejects={"near_miss": [], "nowhere_near": [], "unmeasured": ["GLOBEX"]},
        dip_counts=counts, date_alerts=[], scored=0, seeded={}, prior_payload=prior,
    )

    assert out["last_run_failed"] is False
    assert out["results"] == results


def test_an_empty_watchlist_is_not_a_total_failure():
    """0 scanned / 0 unmeasured trivially satisfies unmeasured == scanned, but a genuinely empty
    watchlist is a real, boring, successful run — not "every symbol failed"."""
    _, counts = scan_job.dip_verdicts([])

    out = scan_job._finalize_payload(
        results=[], dip_rejects={"near_miss": [], "nowhere_near": [], "unmeasured": []},
        dip_counts=counts, date_alerts=[], scored=0, seeded={}, prior_payload={},
    )

    assert out["last_run_failed"] is False
    assert out["last_error"] is None
