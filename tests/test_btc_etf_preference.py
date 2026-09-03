"""Bitcoin-ETF routing — honouring "which fund" without breaking "which trade".

Every spot-bitcoin ETF holds the same asset, so which one you own is a custody/liquidity/fee question
that belongs to the user. Routing a buy onto their pick is easy; doing it without corrupting the order
is the part with edges — the funds trade at very different share prices (IBIT ~$36 vs FBTC ~$55), so
share counts and entry zones do NOT carry across.
"""
from __future__ import annotations

import pytest

from app.sandbox_job import prefer_btc_etf

PRICES = {"IBIT": 36.70, "FBTC": 54.71, "AAPL": 220.0}


def _px(sym):
    return PRICES.get(sym.upper())


def _buy(sym, **over):
    o = {"symbol": sym, "side": "buy", "dollars": 1000.0, "shares": 27.0,
         "conviction": 70, "reason": "bitcoin sleeve"}
    o.update(over)
    return o


def _pos(sym, shares=10.0):
    return {"symbol": sym, "shares": shares, "avg_cost": 30.0}


# --------------------------------------------------------------------------- the happy path

def test_buy_is_routed_to_the_preferred_fund():
    out, notes = prefer_btc_etf([_buy("IBIT")], preferred="FBTC", positions=[], price_of=_px)
    assert out[0]["symbol"] == "FBTC"
    assert notes == ["IBIT→FBTC (preferred bitcoin ETF)"]


def test_routing_is_recorded_in_the_reason():
    out, _ = prefer_btc_etf([_buy("IBIT")], preferred="FBTC", positions=[], price_of=_px)
    assert "routed IBIT→FBTC" in out[0]["reason"]
    assert "bitcoin sleeve" in out[0]["reason"]   # the model's own reasoning survives


def test_an_order_already_on_the_preferred_fund_is_untouched():
    o = _buy("FBTC")
    out, notes = prefer_btc_etf([o], preferred="FBTC", positions=[], price_of=_px)
    assert out == [o] and notes == []


def test_non_crypto_orders_are_untouched():
    o = _buy("AAPL")
    out, notes = prefer_btc_etf([o], preferred="FBTC", positions=[], price_of=_px)
    assert out == [o] and notes == []


def test_inputs_are_not_mutated():
    o = _buy("IBIT")
    prefer_btc_etf([o], preferred="FBTC", positions=[], price_of=_px)
    assert o["symbol"] == "IBIT" and "shares" in o


# --------------------------------------------------------------------------- sells

def test_sells_are_never_rerouted():
    """You can only sell what you hold — rewriting a sell turns a valid exit into a 'not held' skip."""
    o = {"symbol": "IBIT", "side": "sell", "shares": 4.0, "reason": "trim"}
    out, notes = prefer_btc_etf([o], preferred="FBTC", positions=[_pos("IBIT")], price_of=_px)
    assert out == [o] and notes == []


# --------------------------------------------------------------------------- don't fragment

def test_adding_to_a_held_vehicle_is_left_alone():
    """Splitting one exposure across two funds costs spread and leaves the cap logic two positions
    where it expects one. The preference applies to NEW exposure."""
    out, notes = prefer_btc_etf([_buy("IBIT")], preferred="FBTC",
                                positions=[_pos("IBIT")], price_of=_px)
    assert out[0]["symbol"] == "IBIT" and notes == []


def test_routing_resumes_once_the_preferred_fund_is_also_held():
    """Holding both, consolidate onto the preference rather than feeding the other one."""
    out, _ = prefer_btc_etf([_buy("IBIT")], preferred="FBTC",
                            positions=[_pos("IBIT"), _pos("FBTC")], price_of=_px)
    assert out[0]["symbol"] == "FBTC"


def test_a_closed_position_does_not_block_routing():
    out, _ = prefer_btc_etf([_buy("IBIT")], preferred="FBTC",
                            positions=[_pos("IBIT", shares=0.0)], price_of=_px)
    assert out[0]["symbol"] == "FBTC"


# --------------------------------------------------------------------------- sizing across funds

def test_dollar_notional_is_preserved_not_the_share_count():
    """THE sizing trap. IBIT ~$36 vs FBTC ~$55 — carrying 27 shares across would buy ~1.5x the
    intended notional."""
    out, _ = prefer_btc_etf([_buy("IBIT", dollars=1000.0, shares=27.0)],
                            preferred="FBTC", positions=[], price_of=_px)
    assert out[0]["dollars"] == 1000.0
    assert "shares" not in out[0]


def test_a_shares_only_order_is_converted_to_dollars():
    """Dropping `shares` without converting would leave neither field, and the buy path skips that
    order outright ('specified neither dollars nor shares')."""
    o = _buy("IBIT", dollars=None, shares=10.0)
    out, _ = prefer_btc_etf([o], preferred="FBTC", positions=[], price_of=_px)
    assert out[0]["dollars"] == pytest.approx(367.0)   # 10 × IBIT's price, not FBTC's
    assert "shares" not in out[0]


def test_an_unsizeable_order_is_left_alone():
    o = _buy("IBIT", dollars=None, shares=None)
    out, notes = prefer_btc_etf([o], preferred="FBTC", positions=[], price_of=_px)
    assert out == [o] and notes == []


def test_the_entry_zone_is_dropped():
    """The zone was priced against IBIT's ~$36 share price. Carried onto FBTC at ~$55 it would sit
    permanently 'above the zone' and block every routed buy forever."""
    out, _ = prefer_btc_etf([_buy("IBIT", entry_low=34.0, entry_high=37.0)],
                            preferred="FBTC", positions=[], price_of=_px)
    assert "entry_low" not in out[0] and "entry_high" not in out[0]


# --------------------------------------------------------------------------- degrading safely

def test_no_preference_is_a_no_op():
    o = _buy("IBIT")
    for pref in ("", None, "   "):
        out, notes = prefer_btc_etf([o], preferred=pref, positions=[], price_of=_px)
        assert out == [o] and notes == []


def test_an_unknown_preference_is_ignored_rather_than_obeyed():
    o = _buy("IBIT")
    out, notes = prefer_btc_etf([o], preferred="NOTATICKER", positions=[], price_of=_px)
    assert out == [o] and notes == []


def test_an_unpriceable_preference_leaves_the_order_fillable():
    """Routing onto something with no price converts a fillable order into a guaranteed skip."""
    out, notes = prefer_btc_etf([_buy("IBIT")], preferred="FBTC", positions=[],
                                price_of=lambda s: None if s == "FBTC" else 36.70)
    assert out[0]["symbol"] == "IBIT" and notes == []


def test_preference_is_case_insensitive():
    out, _ = prefer_btc_etf([_buy("ibit")], preferred="fbtc", positions=[], price_of=_px)
    assert out[0]["symbol"] == "FBTC"


def test_mixed_batch_routes_only_what_it_should():
    orders = [_buy("IBIT"), _buy("AAPL"), {"symbol": "IBIT", "side": "sell", "shares": 2.0}]
    out, notes = prefer_btc_etf(orders, preferred="FBTC", positions=[], price_of=_px)
    assert [o["symbol"] for o in out] == ["FBTC", "AAPL", "IBIT"]
    assert len(notes) == 1


# ------------------------------------------------------------- the symmetry (2026-09-02)
#
# The consolidate rule ran on one side only. `prefer_vehicle` returned early on `sym == pref`, before
# the check — so with IBIT held and FBTC preferred, an order naming IBIT consolidated onto IBIT while
# an order naming FBTC opened a SECOND position in the same exposure. Same book, same dollars, same
# preference; the outcome turned on nothing but which ticker the model happened to write.
#
# Measured on the gold side of the same function on 2026-09-02: the baseline sandbox arm bought $260
# of GLDM on a stated rationale of "replacing GLD (60bp fee savings)", never sold the GLD, and closed
# the day holding both at a blended 0.347% expense ratio — with gold at 12.93% of equity instead of
# the 10.6% a swap would have left. It reached the outcome `intra_group_swaps` exists to refuse by
# proposing one leg instead of two.


def test_ordering_the_preferred_fund_while_holding_the_other_does_not_open_a_second_position():
    """The regression. Mirror of test_adding_to_a_held_vehicle_is_left_alone, which always passed."""
    out, notes = prefer_btc_etf([_buy("FBTC")], preferred="FBTC",
                                positions=[_pos("IBIT")], price_of=_px)
    assert out[0]["symbol"] == "IBIT", "new money must join the vehicle already held"
    assert notes, "a symbol rewrite that the log does not mention is a silent rewrite"


def test_the_two_orderings_of_the_same_decision_agree():
    """Stated as the invariant rather than as two expected values, so neither side can drift alone:
    with one vehicle held, the fund bought must not depend on which the model named."""
    held = [_pos("IBIT")]
    named_incumbent, _ = prefer_btc_etf([_buy("IBIT")], preferred="FBTC", positions=held, price_of=_px)
    named_preferred, _ = prefer_btc_etf([_buy("FBTC")], preferred="FBTC", positions=held, price_of=_px)
    assert named_incumbent[0]["symbol"] == named_preferred[0]["symbol"]


def test_the_reroute_says_which_rule_moved_it():
    """"preferred fund" and "already holding it" are different reasons and the log must not conflate
    them — one is honouring a setting, the other is overriding it."""
    to_pref, n1 = prefer_btc_etf([_buy("IBIT")], preferred="FBTC", positions=[], price_of=_px)
    assert to_pref[0]["symbol"] == "FBTC"
    assert "preferred" in n1[0]

    to_held, n2 = prefer_btc_etf([_buy("FBTC")], preferred="FBTC", positions=[_pos("IBIT")], price_of=_px)
    assert to_held[0]["symbol"] == "IBIT"
    assert "already holding" in n2[0]


def test_consolidation_outranks_the_preference_but_only_while_the_incumbent_is_open():
    """Closing the old vehicle is what lets the preference take effect — no sale is ever forced."""
    out, _ = prefer_btc_etf([_buy("FBTC")], preferred="FBTC",
                            positions=[_pos("IBIT", shares=0.0)], price_of=_px)
    assert out[0]["symbol"] == "FBTC"


def test_an_unfillable_incumbent_leaves_the_order_alone_rather_than_guaranteeing_a_skip():
    """Same rule the preference path already had: never reroute onto something with no price."""
    out, notes = prefer_btc_etf([_buy("FBTC")], preferred="FBTC", positions=[_pos("IBIT")],
                                price_of=lambda s: None if s == "IBIT" else 55.0)
    assert out[0]["symbol"] == "FBTC" and notes == []


def test_the_consolidate_rule_applies_even_with_no_preference_set():
    """The second position breaks the cap logic whether or not the user expressed a preference, and
    the early return on an empty preference used to skip the check entirely."""
    out, _ = prefer_btc_etf([_buy("FBTC")], preferred="", positions=[_pos("IBIT")], price_of=_px)
    assert out[0]["symbol"] == "IBIT"


def test_dollar_sizing_survives_a_consolidating_reroute():
    """The two funds trade at very different share prices, so a shares count carried across would buy
    the wrong notional — the same arithmetic the preference path documents."""
    out, _ = prefer_btc_etf([{"symbol": "FBTC", "side": "buy", "shares": 4.0, "reason": "add"}],
                            preferred="FBTC", positions=[_pos("IBIT")], price_of=_px)
    assert out[0]["symbol"] == "IBIT"
    assert "shares" not in out[0], "a raw share count would buy the wrong notional in the other fund"
    assert out[0]["dollars"] == round(4.0 * _px("FBTC"), 2)


def test_a_sell_of_the_preferred_fund_is_still_never_rerouted():
    """The consolidate branch must not reach sells — rerouting an exit turns it into a 'not held' skip."""
    o = {"symbol": "FBTC", "side": "sell", "shares": 2.0, "reason": "trim"}
    out, notes = prefer_btc_etf([o], preferred="FBTC", positions=[_pos("IBIT"), _pos("FBTC")],
                                price_of=_px)
    assert out == [o] and notes == []


# ------------------------------------------------- an exclusion outranks consolidation (2026-09-02)
#
# Found by the adversarial review of the symmetry fix above, which introduced this. Consolidation is
# a DEFAULT; an exclusion is an explicit instruction, and "stop adding to IBIT" plus "prefer FBTC" is
# exactly how a user asks to migrate off a fund without selling it. Routed naively, those two
# settings combined into "no buys in this family at all": the FBTC order consolidated onto the held
# IBIT and was then skipped by validate_and_fill as an excluded ticker — under a symbol the model
# never named — and stayed that way until the IBIT position happened to close.


def test_an_excluded_incumbent_is_not_a_consolidation_target():
    out, notes = prefer_btc_etf([_buy("FBTC")], preferred="FBTC", positions=[_pos("IBIT")],
                                price_of=_px, exclude={"IBIT"})
    assert out[0]["symbol"] == "FBTC", "the buy must not be routed into the fund the user excluded"
    assert notes == []


def test_the_exclusion_holds_with_no_preference_configured_too():
    """The consolidate rule now runs without a preference, so the dead end reproduces there as well."""
    out, _ = prefer_btc_etf([_buy("FBTC")], preferred="", positions=[_pos("IBIT")],
                            price_of=_px, exclude={"IBIT"})
    assert out[0]["symbol"] == "FBTC"


def test_an_excluded_preference_does_not_capture_new_exposure():
    """Excluding the preferred fund while holding nothing must leave the order where the model put
    it, not route it onto a ticker that is then refused."""
    out, _ = prefer_btc_etf([_buy("IBIT")], preferred="FBTC", positions=[], price_of=_px,
                            exclude={"FBTC"})
    assert out[0]["symbol"] == "IBIT"


def test_excluding_the_whole_family_leaves_every_order_untouched():
    """Nothing to route onto. The order must reach the exclusion check as the model wrote it, so the
    skip names the ticker the model actually proposed."""
    out, notes = prefer_btc_etf([_buy("FBTC")], preferred="FBTC", positions=[_pos("IBIT")],
                                price_of=_px, exclude={"IBIT", "FBTC"})
    assert out[0]["symbol"] == "FBTC" and notes == []


def test_consolidation_still_applies_to_tickers_that_are_not_excluded():
    """The exclusion must not switch the rule off wholesale — an unrelated exclusion changes nothing."""
    out, _ = prefer_btc_etf([_buy("FBTC")], preferred="FBTC", positions=[_pos("IBIT")],
                            price_of=_px, exclude={"TSLA"})
    assert out[0]["symbol"] == "IBIT"


def test_the_migration_settings_actually_migrate():
    """The end-to-end intent, stated once: hold the old fund, exclude it, prefer the new one — and
    new money reaches the new one instead of dying between two rules."""
    held_old = [_pos("IBIT")]
    out, _ = prefer_btc_etf([_buy("FBTC")], preferred="FBTC", positions=held_old, price_of=_px,
                            exclude={"IBIT"})
    assert out[0]["symbol"] == "FBTC"
    # ...and the old fund is not sold to get there.
    sell = {"symbol": "IBIT", "side": "sell", "shares": 4.0, "reason": "x"}
    still, _ = prefer_btc_etf([sell], preferred="FBTC", positions=held_old, price_of=_px,
                              exclude={"IBIT"})
    assert still == [sell]
