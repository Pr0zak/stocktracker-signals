"""How far the book is from its own standing plan, computed rather than left to the model.

The daily model received the plan's targets and the book's positions as two separate structures and
was left to diff them. Measured 2026-08-17, three weeks in: MSFT, COST and JNJ sat at exactly 0.0%
against targets of 8%, 7% and 6% -- 21 points of a standing plan the tick had never once acted on --
while cash ran at 43.5% against a 12% target. No skip row exists for any of them. Nothing blocked
those buys; they were simply never proposed.

Only shortfalls are returned. An overweight group is not something a buy can fix, and listing it
invites a sell to correct a number, which is exactly the churn this account is steered away from.
"""
from __future__ import annotations

from app.sandbox_job import target_gaps

GROUPS = {"VTI": "US_EQUITY", "VOO": "US_EQUITY"}


def _group(sym: str) -> str:
    return GROUPS.get(sym.upper(), sym.upper())


def _pos(sym, shares, price):
    return {"symbol": sym, "shares": shares, "avg_cost": price, "last_price": price}


def _plan(**targets):
    return {"targets": [{"exposure_group": g, "target_pct": p} for g, p in targets.items()]}


def _px(prices):
    return lambda s: prices.get(s.upper())


def test_shortfalls_come_back_largest_first():
    # The ordering IS the feature: the model reads top-down, and the biggest unmet target is the one
    # worth its attention first.
    # 24 VTI at $100 is $2,400 of a $10,000 book: US_EQUITY nearly met at 24% against 25%, so its
    # 1-point gap sorts BEHIND the two untouched targets. That ordering is the whole point -- a
    # nearly-filled target must not sit above one the book has never held at all.
    gaps = target_gaps(
        [_pos("VTI", 24, 100.0)], equity=10_000.0,
        plan=_plan(US_EQUITY=25.0, MSFT=8.0, JNJ=6.0),
        group_of=_group, price_of=_px({"VTI": 100.0}))
    assert [g["exposure_group"] for g in gaps] == ["MSFT", "JNJ", "US_EQUITY"]
    assert gaps[0]["gap_pct"] == 8.0
    assert gaps[0]["actual_pct"] == 0.0


def test_the_gap_is_reported_in_dollars_as_well_as_points():
    # A percentage of a book this size is not an order size. The dollar figure is what a buy is
    # actually written in, and making the model convert it is another step it can skip.
    gaps = target_gaps(
        [], equity=10_000.0, plan=_plan(MSFT=8.0), group_of=_group, price_of=_px({}))
    assert gaps[0]["gap_dollars"] == 800.0


def test_a_met_target_is_not_reported():
    gaps = target_gaps(
        [_pos("VTI", 25, 100.0)], equity=10_000.0, plan=_plan(US_EQUITY=25.0),
        group_of=_group, price_of=_px({"VTI": 100.0}))
    assert gaps == []


def test_an_overweight_group_is_not_reported_as_a_gap():
    # Deliberately omitted. A buy cannot fix an overweight, and surfacing it as a "gap" would read
    # as something to act on -- the action being a sell, to correct a number rather than a thesis.
    gaps = target_gaps(
        [_pos("VTI", 40, 100.0)], equity=10_000.0, plan=_plan(US_EQUITY=25.0),
        group_of=_group, price_of=_px({"VTI": 100.0}))
    assert gaps == []


def test_half_a_point_of_drift_is_not_a_shortfall():
    # Prices move. A target reported as unmet by 0.3 points every single day is noise that would
    # train the reader -- and the model -- to skim the list.
    gaps = target_gaps(
        [_pos("VTI", 247, 100.0)], equity=100_000.0, plan=_plan(US_EQUITY=25.0),
        group_of=_group, price_of=_px({"VTI": 100.0}))
    assert gaps == []


def test_positions_in_the_same_group_are_summed():
    # VTI and VOO are one exposure. Counted separately, each would look short and the pair would
    # invite buying the group to double its target.
    gaps = target_gaps(
        [_pos("VTI", 10, 100.0), _pos("VOO", 10, 100.0)], equity=10_000.0,
        plan=_plan(US_EQUITY=25.0), group_of=_group, price_of=_px({"VTI": 100.0, "VOO": 100.0}))
    assert gaps[0]["actual_pct"] == 20.0
    assert gaps[0]["gap_pct"] == 5.0


def test_no_plan_or_no_equity_yields_nothing_rather_than_dividing_by_zero():
    assert target_gaps([], equity=10_000.0, plan=None, group_of=_group, price_of=_px({})) == []
    assert target_gaps([], equity=0.0, plan=_plan(MSFT=8.0),
                       group_of=_group, price_of=_px({})) == []


def test_a_target_with_an_unparseable_percentage_is_skipped_not_crashed():
    # Plans are model output. One malformed target must not take the whole gap list with it.
    plan = {"targets": [{"exposure_group": "MSFT", "target_pct": "eight"},
                        {"exposure_group": "JNJ", "target_pct": 6.0}]}
    gaps = target_gaps([], equity=10_000.0, plan=plan, group_of=_group, price_of=_px({}))
    assert [g["exposure_group"] for g in gaps] == ["JNJ"]
