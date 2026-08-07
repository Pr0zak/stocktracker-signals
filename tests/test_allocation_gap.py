"""The weekly plan has to account for every dollar.

Measured 2026-08-04: the strategist's targets summed to 64% against a 22% cash target, leaving 14
points with no owner. The book sat at 42% cash and the ENTIRE shortfall against the S&P was that idle
cash — $193 of opportunity cost against a $187 gap, meaning the stock selection was net positive and
the uninvested money lost the race by itself.

Deliberately an AUDIT, not a repair: normalising short targets upward would push them through the
per-group cap and produce orders the daily tick can never fill. Only the next review can add groups
or honestly raise the cash target, so the finding is recorded and fed back to it.
"""
from __future__ import annotations

from app.sandbox_job import allocation_gap


def _note(targets, cash=22.0):
    return {"cash_target_pct": cash,
            "targets": [{"exposure_group": g, "target_pct": p} for g, p in targets]}


# ---------------------------------------------------------------- the plan that motivated this

def test_the_real_short_plan_is_flagged():
    gap = allocation_gap(_note([("VTI", 25), ("SP500", 21), ("VXUS", 12), ("AMZN", 6)], cash=22.0))
    assert gap is not None
    assert gap["targets_sum_pct"] == 64.0
    assert gap["investable_pct"] == 78.0
    assert gap["unallocated_pct"] == 14.0


def test_it_reports_how_many_groups_are_needed():
    """78% investable under a 25% cap needs at least 4 groups — this plan had 4 but only used 64%."""
    gap = allocation_gap(_note([("VTI", 25), ("SP500", 21), ("VXUS", 12), ("AMZN", 6)], cash=22.0))
    assert gap["groups"] == 4
    assert gap["groups_needed"] == 4


def test_too_few_groups_to_cover_the_plan_is_flagged():
    """Two groups cannot cover 78% under a 25% cap however the percentages are written."""
    gap = allocation_gap(_note([("VTI", 25), ("SP500", 25)], cash=22.0))
    assert gap is not None
    assert gap["groups_needed"] == 4


# ---------------------------------------------------------------- sound plans pass

def test_a_fully_allocated_plan_passes():
    gap = allocation_gap(_note([("VTI", 25), ("SP500", 25), ("VXUS", 18), ("AMZN", 10)], cash=22.0))
    assert gap is None


def test_high_cash_is_fine_when_it_is_DECLARED():
    """Holding 60% cash is a legitimate stance. Leaving 60% unassigned is not the same thing."""
    assert allocation_gap(_note([("VTI", 25), ("SP500", 15)], cash=60.0)) is None


def test_small_rounding_slack_is_tolerated():
    gap = allocation_gap(_note([("VTI", 25), ("SP500", 25), ("VXUS", 16), ("AMZN", 10)], cash=22.0))
    assert gap is None   # sums to 76 vs 78 investable — inside the slack


def test_slack_boundary_is_enforced():
    gap = allocation_gap(_note([("VTI", 25), ("SP500", 25), ("VXUS", 10), ("AMZN", 10)], cash=22.0))
    assert gap is not None and gap["unallocated_pct"] == 8.0


# ---------------------------------------------------------------- unreachable targets

def test_a_target_above_the_cap_is_flagged_as_unreachable():
    """The tick can never fill it, so it fires blocked orders against that group forever."""
    gap = allocation_gap(_note([("VTI", 40), ("SP500", 25), ("VXUS", 13)], cash=22.0),
                         max_position_pct=25.0)
    assert gap is not None
    assert gap["targets_over_cap"] == ["VTI"]


def test_two_labels_for_one_group_are_caught_as_a_cap_breach():
    """The case that slipped through. After VTI/SPY/VOO were merged into US_EQUITY, the standing plan
    still said "VTI 25%" and "SP500 21%" — each fine alone, 46% of one group together, so the tick
    fired a blocked VTI buy every day and deployed nothing while the audit reported no cap problem."""
    groups = {"VTI": "US_EQUITY", "SP500": "US_EQUITY", "VXUS": "VXUS", "AMZN": "AMZN"}
    note = _note([("VTI", 25), ("SP500", 21), ("VXUS", 12), ("AMZN", 6)], cash=22.0)
    assert allocation_gap(note, max_position_pct=25.0)["targets_over_cap"] == []  # old behaviour
    gap = allocation_gap(note, max_position_pct=25.0, group_of=lambda s: groups.get(s, s))
    assert gap["targets_over_cap"] == ["US_EQUITY"]


def test_grouping_does_not_invent_a_breach_that_is_not_there():
    groups = {"VTI": "US_EQUITY", "VXUS": "VXUS", "SCHD": "SCHD", "AMZN": "AMZN"}
    note = _note([("VTI", 25), ("VXUS", 25), ("SCHD", 18), ("AMZN", 10)], cash=22.0)
    assert allocation_gap(note, max_position_pct=25.0, group_of=lambda s: groups.get(s, s)) is None


def test_the_cap_is_configurable():
    note = _note([("VTI", 40), ("SP500", 38)], cash=22.0)
    assert allocation_gap(note, max_position_pct=40.0) is None
    assert allocation_gap(note, max_position_pct=25.0) is not None


# ---------------------------------------------------------------- degrading safely

def test_no_note_is_not_a_gap():
    assert allocation_gap(None) is None


def test_an_empty_plan_is_flagged_rather_than_ignored():
    gap = allocation_gap(_note([], cash=22.0))
    assert gap is not None and gap["unallocated_pct"] == 78.0


def test_a_full_cash_stance_needs_no_targets():
    assert allocation_gap(_note([], cash=100.0)) is None
