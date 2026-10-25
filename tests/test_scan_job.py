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


# --------------------------------------------------------------- flip debounce (two-run confirmation)

def _sig_row(symbol, signal):
    return {"symbol": symbol, "signal": signal, "dip": None}


def _carry(row):
    """The per-symbol state the next run's `_prev_state()` hands back for this row."""
    return {row["symbol"]: {k: row.get(k) for k in
                            ("signal", "squeeze", "below_200wma", "dip", "stable_signal", "pending_signal")}}


def test_a_one_run_change_is_not_a_flip_yet():
    prev = {"XOM": {"signal": "hold", "stable_signal": "hold", "pending_signal": None}}
    out = scan_job._diff_vs_prev(_sig_row("XOM", "sell"), prev)
    assert out["flipped"] is False
    assert out["stable_signal"] == "hold" and out["pending_signal"] == "sell"


def test_the_same_change_on_the_next_run_confirms_the_flip():
    prev = {"XOM": {"signal": "sell", "stable_signal": "hold", "pending_signal": "sell"}}
    out = scan_job._diff_vs_prev(_sig_row("XOM", "sell"), prev)
    assert out["flipped"] is True and out["flipped_from"] == "hold"
    assert out["stable_signal"] == "sell" and out["pending_signal"] is None


def test_hold_sell_hold_sell_churn_never_flips():
    prev = {"CVX": {"signal": "hold", "stable_signal": "hold", "pending_signal": None}}
    flips = []
    for sig in ("sell", "hold", "sell", "hold", "sell"):
        row = scan_job._diff_vs_prev(_sig_row("CVX", sig), prev)
        flips.append(row["flipped"])
        prev = _carry(row)
    assert flips == [False] * 5
    assert prev["CVX"]["stable_signal"] == "hold"


def test_a_different_second_change_restarts_the_wait():
    prev = {"CVX": {"signal": "sell", "stable_signal": "hold", "pending_signal": "sell"}}
    out = scan_job._diff_vs_prev(_sig_row("CVX", "buy"), prev)
    assert out["flipped"] is False
    assert out["stable_signal"] == "hold" and out["pending_signal"] == "buy"


def test_a_row_from_before_the_debounce_takes_its_last_signal_as_confirmed():
    prev = {"AAPL": {"signal": "hold"}}  # an old payload: no stable/pending fields
    same = scan_job._diff_vs_prev(_sig_row("AAPL", "hold"), prev)
    assert same["flipped"] is False and same["stable_signal"] == "hold"
    changed = scan_job._diff_vs_prev(_sig_row("AAPL", "buy"), prev)
    assert changed["flipped"] is False and changed["pending_signal"] == "buy"


def test_first_ever_sighting_is_not_a_flip():
    out = scan_job._diff_vs_prev(_sig_row("NEW", "buy"), {})
    assert out["flipped"] is False and out["stable_signal"] == "buy"


def test_debounce_state_survives_into_last_measured():
    row = scan_job._diff_vs_prev(_sig_row("XOM", "sell"),
                                 {"XOM": {"signal": "hold", "stable_signal": "hold"}})
    payload = scan_job._finalize_payload(
        results=[row], dip_rejects={}, date_alerts=[], scored=0, seeded={}, prior_payload={},
        dip_counts={"scanned": 1, "qualified": 0, "near_miss": 0, "nowhere_near": 1, "unmeasured": 0},
    )
    assert payload["last_measured"]["XOM"]["pending_signal"] == "sell"
    assert payload["last_measured"]["XOM"]["stable_signal"] == "hold"


# --------------------------------------------------------------- crypto-only runs on closed days

def _prior_with(rows, generated_at=1000.0):
    return {"generated_at": generated_at, "results": rows}


def test_carried_stock_rows_keep_their_read_and_lose_every_change_flag():
    prior = _prior_with([
        {"symbol": "XOM", "signal": "sell", "flipped": True, "dip": "pullback_10", "dip_new": True,
         "squeeze_changed": True, "crossed_below_200wma": True, "cost_usd": 0.01,
         "stable_signal": "sell"},
        {"symbol": "ETH-USD", "signal": "hold", "flipped": True},
        {"symbol": "GONE", "signal": "buy", "flipped": True},
    ])
    out = scan_job._carry_stock_rows(prior, ["XOM", "AAPL"])
    assert [r["symbol"] for r in out] == ["XOM"]  # crypto is re-scored; delisted names drop out
    xom = out[0]
    assert xom["signal"] == "sell" and xom["dip"] == "pullback_10" and xom["stable_signal"] == "sell"
    assert not any(xom[k] for k in ("flipped", "dip_new", "squeeze_changed", "crossed_below_200wma"))
    assert xom["cost_usd"] == 0.0 and xom["carried_from"] == 1000.0


def test_a_second_closed_day_keeps_the_scan_that_really_measured_the_row():
    prior = _prior_with([{"symbol": "XOM", "signal": "sell", "carried_from": 900.0}], generated_at=1000.0)
    assert scan_job._carry_stock_rows(prior, ["XOM"])[0]["carried_from"] == 900.0


def test_carried_rows_do_not_mask_a_night_every_crypto_fetch_failed():
    results = [
        {"symbol": "XOM", "signal": "sell", "dip_measured": True, "carried_from": 1000.0},
        {"symbol": "ETH-USD", "error": "boom"},
    ]
    _, counts = scan_job.dip_verdicts(results)
    payload = scan_job._finalize_payload(
        results=results, dip_rejects={}, dip_counts=counts, date_alerts=[], scored=0, seeded={},
        prior_payload={"generated_at": 1000.0, "results": [{"symbol": "XOM", "signal": "sell"}]},
    )
    assert payload["last_run_failed"] is True


def test_monday_diffs_against_the_carried_rows_not_an_empty_baseline():
    """A standing dip carried over the weekend must not read as new on Monday's full scan."""
    carried = scan_job._carry_stock_rows(
        _prior_with([{"symbol": "XOM", "signal": "hold", "dip": "pullback_10", "stable_signal": "hold"}]),
        ["XOM"],
    )
    prev = scan_job._prev_state({"results": carried})
    monday = scan_job._diff_vs_prev({"symbol": "XOM", "signal": "hold", "dip": "pullback_10"}, prev)
    assert monday["dip_new"] is False and monday["flipped"] is False
