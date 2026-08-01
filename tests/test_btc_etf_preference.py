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
