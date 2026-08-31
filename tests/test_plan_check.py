"""plan_check: verifying the two rules PLAN_SYSTEM asks for and nothing checked.

Three of these tests exist because an adversarial review of the design caught defects before it was
written — a route-level 500 on sub-penny assets, a false `stop_ok: false` on the same, and a severity
switch that was silently always lenient because pydantic's model_dump() returns the enum member and
`str(PlanAction.buy_now)` is "PlanAction.buy_now". They are marked where they appear.
"""

import math

import pytest

from app.analyst import EntryPlan, PlanAction
from app.plan_check import ERROR, INFO, WARNING, _snap_for, annotate, annotate_picks, check


def plan(action="buy_now", entry_low=100.0, entry_high=100.0, stop=95.0, target=107.5):
    return {"action": action, "entry_low": entry_low, "entry_high": entry_high,
            "stop": stop, "target": target, "symbol": "TEST"}


def snap(atr=2.0, symbol="TEST"):
    return {"symbol": symbol, "atr14": atr, "price": 100.0}


def types(res):
    return [m["type"] for m in res["messages"]]


# --- the measurement ---------------------------------------------------------------------------


def test_a_sound_plan_measures_clean_and_says_nothing():
    r = check(plan(), snap())
    assert r["entry_mid"] == 100.0
    assert r["r_usd"] == 5.0
    assert r["reward_usd"] == 7.5
    assert r["rr_ratio"] == 1.5
    assert r["rr_ok"] is True
    assert r["stop_atr"] == 2.5
    assert r["stop_ok"] is True
    assert r["messages"] == []


def test_the_ratio_boundary_is_the_rounded_one_the_card_displays():
    """"at least ~1.5" — a 1.498 shown as 1.5 and flagged as a failure contradicts itself."""
    assert check(plan(target=107.49), snap())["rr_ok"] is True     # 1.498 -> displays 1.5
    r = check(plan(target=107.40), snap())                          # 1.48
    assert r["rr_ratio"] == 1.48
    assert r["rr_ok"] is False
    assert types(r) == [WARNING]


def test_a_stop_inside_one_average_session_is_flagged():
    r = check(plan(stop=99.0), snap(atr=2.0))   # risk 1.0 against a 2.0 ATR
    assert r["stop_atr"] == 0.5
    assert r["stop_ok"] is False
    assert WARNING in types(r)


def test_the_two_checks_are_independent():
    """A plan can fail the ratio and pass the noise test, and each carries its own reason."""
    r = check(plan(target=101.0, stop=90.0), snap(atr=2.0))
    assert r["rr_ok"] is False
    assert r["stop_ok"] is True
    assert types(r) == [WARNING]


# --- absence, never a substituted number --------------------------------------------------------


def test_a_zero_stop_is_absence_and_never_a_zero_risk():
    """EntryPlan.stop is a non-nullable float, so a model that cannot justify one emits 0.0."""
    r = check(plan(stop=0.0), snap())
    assert r["r_usd"] is None
    assert r["rr_ratio"] is None and r["rr_ok"] is None
    assert r["stop_atr"] is None and r["stop_ok"] is None
    assert types(r) == [ERROR]


def test_a_stop_inside_the_entry_zone_is_an_error_with_nothing_derived():
    """Tested against entry_low, not the midpoint: a fill at the bottom of the zone is already under
    a stop that merely clears the middle."""
    r = check(plan(entry_low=100.0, entry_high=110.0, stop=104.0), snap())
    assert r["r_usd"] is None
    assert ERROR in types(r)


def test_a_missing_snapshot_still_measures_the_ratio():
    r = check(plan(), None)
    assert r["rr_ok"] is True
    assert r["atr14"] is None and r["stop_atr"] is None and r["stop_ok"] is None
    assert types(r) == [INFO]


def test_an_unmeasurable_atr_reports_its_reason():
    r = check(plan(), snap(atr=None))
    assert r["stop_ok"] is None
    assert types(r) == [INFO]


def test_a_measured_zero_atr_is_reported_as_a_measurement_not_a_failure():
    r = check(plan(), snap(atr=0.0))
    assert r["atr14"] == 0.0
    assert r["stop_ok"] is None, "no range to size against is not the same as a stop that is too tight"
    assert types(r) == [INFO]


@pytest.mark.parametrize("bad", [None, 0.0, -5.0, math.nan, math.inf, True])
def test_garbage_levels_produce_absence_rather_than_a_flag(bad):
    r = check(plan(entry_low=bad), snap())
    assert r["entry_mid"] is None
    assert r["rr_ok"] is None and r["stop_ok"] is None


def test_an_inverted_zone_is_refused():
    r = check(plan(entry_low=110.0, entry_high=100.0), snap())
    assert r["entry_mid"] is None
    assert types(r) == [ERROR]


def test_a_flag_is_never_false_without_a_measurement_behind_it():
    """rr_ok and stop_ok are assigned only in the branch that assigned their ratio."""
    for p in (plan(stop=0.0), plan(target=0.0), plan(entry_low=0.0), plan(stop=101.0)):
        r = check(p, snap())
        if r["rr_ok"] is not None:
            assert r["rr_ratio"] is not None
        if r["stop_ok"] is not None:
            assert r["stop_atr"] is not None


# --- caught by the adversarial review, before implementation ------------------------------------


def test_a_sub_penny_asset_does_not_crash_the_route():
    """CAUGHT IN REVIEW. The design guarded on an unrounded difference but stored a 4dp-rounded one,
    so a sub-penny plan divided by a stored 0.0 and raised ZeroDivisionError inside a route with no
    try/except — a 500 on /plan and /recommendations. Money is now never rounded before division."""
    p = plan(entry_low=0.0000120, entry_high=0.0000126, stop=0.0000110, target=0.0000150)
    r = check(p, snap(atr=7.4e-07))
    assert r["r_usd"] is not None and r["r_usd"] > 0
    assert r["stop_atr"] is not None


def test_a_sub_penny_stop_is_not_falsely_flagged_as_noise():
    """CAUGHT IN REVIEW. The same rounding produced stop_atr 0.0 and stop_ok False for a stop that
    was genuinely 1.36 ATR wide — a false flag where true was the measurement."""
    p = plan(entry_low=0.0000120, entry_high=0.0000126, stop=0.0000110, target=0.0000150)
    r = check(p, snap(atr=7.4e-07))
    assert r["stop_atr"] > 1.0
    assert r["stop_ok"] is True


def test_the_severity_switch_reads_the_enum_value_not_its_repr():
    """CAUGHT IN REVIEW. model_dump() without mode="json" returns the ENUM MEMBER, and PlanAction
    subclasses str, so str(PlanAction.buy_now) is "PlanAction.buy_now" on Python 3.11+. Comparing the
    stringified member never matched, and every plan was graded with the lenient severity."""
    assert str(PlanAction.buy_now) == "PlanAction.buy_now"  # the trap, pinned

    actionable = {"action": PlanAction.buy_now, "entry_low": 100.0, "entry_high": 100.0,
                  "stop": 0.0, "target": 110.0}
    assert types(check(actionable, snap())) == [ERROR]

    waiting = {**actionable, "action": PlanAction.wait}
    assert types(check(waiting, snap())) == [INFO]


# --- per-symbol correctness ---------------------------------------------------------------------


def test_a_pick_is_annotated_with_its_own_symbols_volatility():
    picks = [{**plan(), "symbol": "AAA"}, {**plan(), "symbol": "BBB"}]
    snaps = [{"symbol": "AAA", "atr14": 1.0}, {"symbol": "BBB", "atr14": 50.0}]
    out = annotate_picks(picks, snaps)
    assert out[0]["plan_check"]["atr14"] == 1.0
    assert out[1]["plan_check"]["atr14"] == 50.0


def test_an_unmatched_pick_gets_absence_rather_than_a_neighbours_number():
    out = annotate_picks([{**plan(), "symbol": "ZZZ"}], [{"symbol": "AAA", "atr14": 1.0}])
    assert out[0]["plan_check"]["atr14"] is None
    assert out[0]["plan_check"]["rr_ok"] is True, "the ratio half needs only the plan's own levels"


def test_symbol_matching_is_exact_with_one_crypto_suffix_retry():
    snaps = [{"symbol": "BTC-USD", "atr14": 1000.0}, {"symbol": "AAA", "atr14": 1.0}]
    assert _snap_for("BTC", snaps)["atr14"] == 1000.0
    assert _snap_for(" aaa ", snaps)["atr14"] == 1.0
    assert _snap_for("AA", snaps) is None, "a near match must not become a wrong number"
    assert _snap_for(None, snaps) is None
    assert _snap_for("AAA", None) is None


# --- the envelope --------------------------------------------------------------------------------


def test_plan_check_is_always_present_and_always_an_object():
    out = annotate(plan(entry_low=None), None)
    assert isinstance(out["plan_check"], dict)
    for k in ("entry_mid", "r_usd", "reward_usd", "rr_ratio", "rr_ok", "atr14", "stop_atr",
              "stop_ok", "messages"):
        assert k in out["plan_check"], f"{k} must be present-and-null, not missing"


def test_it_attaches_to_the_dumped_dict_because_the_model_rejects_unknown_fields():
    e = EntryPlan(symbol="TEST", action=PlanAction.buy_now, entry_low=100.0, entry_high=100.0,
                  stop=95.0, target=107.5, allocation_usd=100.0, suggested_shares=1,
                  conviction=60, timing="now", thesis="t")
    with pytest.raises(ValueError):
        e.plan_check = {}
    assert "plan_check" in annotate(e.model_dump(), snap())


def test_messages_serialise_deterministically():
    p = plan(stop=0.0, target=0.0)
    assert check(p, snap())["messages"] == check(p, snap())["messages"]
