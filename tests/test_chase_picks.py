"""SWT-15 — the Ideas screen shows entry zones beside live prices with no chase read.

SWT-3 built the read and wired it into `/plan` only. `/recommendations` returns the same `EntryPlan`
shape, so the Ideas screen has been printing "Entry $95–$100" next to a live price of $118 with
nothing saying the trade on offer is no longer the trade being shown. The app side has been ready and
silent since app v1.0: `ChaseLine(ChaseState.fromPick(pick))` renders nothing while the fields are
absent, which is the correct behaviour for an old server and the wrong outcome forever.

The contract these pin, inherited from `chase.annotate`: all four keys ALWAYS present, null included.
A client seeing `chase_status` present and null knows the read was attempted and could not be taken,
where a missing key is ambiguous with an older server. Never 0%, never "ok".
"""

import pytest

from app import chase

_KEYS = {"chase_pct", "chase_status", "chase_warning", "chase_price"}


def _pick(symbol="AAPL", low=95.0, high=100.0):
    return {"symbol": symbol, "entry_low": low, "entry_high": high, "action": "buy_on_pullback"}


# --- each pick gets ITS OWN symbol's price --------------------------------------------------------


def test_each_pick_is_read_against_its_own_price():
    """One call ranks many symbols. Annotating a pick with a neighbour's price would be worse than
    not annotating it — a wrong number is invisible where an absent one is not."""
    picks = [_pick("AAPL", 95.0, 100.0), _pick("MSFT", 400.0, 420.0)]
    out = chase.annotate_picks(picks, {"AAPL": 118.0, "MSFT": 410.0})

    assert out[0]["chase_price"] == 118.0
    assert out[0]["chase_status"] == chase.TOO_DEEP
    assert out[1]["chase_price"] == 410.0
    assert out[1]["chase_status"] == chase.IN_ZONE


def test_a_symbol_with_no_price_is_annotated_absent_not_comfortable():
    out = chase.annotate_picks([_pick("AAPL")], {})
    assert _KEYS <= out[0].keys(), "the keys must be present so the client can tell attempted-and-failed"
    assert out[0]["chase_pct"] is None
    assert out[0]["chase_status"] is None
    assert out[0]["chase_price"] is None


def test_the_lookup_is_case_and_whitespace_insensitive():
    out = chase.annotate_picks([{"symbol": " aapl ", "entry_low": 95.0, "entry_high": 100.0}],
                               {"AAPL": 118.0})
    assert out[0]["chase_price"] == 118.0


def test_a_pick_with_no_entry_zone_still_carries_all_four_keys():
    out = chase.annotate_picks([{"symbol": "AAPL", "action": "wait"}], {"AAPL": 118.0})
    assert _KEYS <= out[0].keys()
    assert out[0]["chase_status"] is None
    # The price IS known even when the zone is not — the read failed, the quote did not.
    assert out[0]["chase_price"] == 118.0


def test_annotation_reads_the_pick_s_own_zone_not_a_nested_plan():
    """`chase.annotate` reads payload["plan"]; a pick carries the zone on itself. Using the wrong one
    silently produces an absent read on every pick."""
    out = chase.annotate_pick(_pick(low=95.0, high=100.0), 90.0)
    assert out["chase_status"] == chase.BELOW_ZONE


# --- the states, on a pick ------------------------------------------------------------------------


@pytest.mark.parametrize("price,expected", [
    (90.0, chase.BELOW_ZONE),    # cheaper than the analyst asked for
    (97.5, chase.IN_ZONE),       # inside it
    (101.0, chase.OK),           # a shade above, still fine
    (140.0, chase.TOO_DEEP),     # the trade on offer is not the trade being shown
])
def test_the_four_states_survive_the_pick_shaped_call(price, expected):
    assert chase.annotate_pick(_pick(), price)["chase_status"] == expected


def test_an_empty_pick_list_is_not_an_error():
    assert chase.annotate_picks([], {"AAPL": 118.0}) == []
