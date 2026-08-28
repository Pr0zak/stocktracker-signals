"""Which gold vehicle to own is the user's call, and the fee gap only steers NEW money.

Every ETF in the GOLD group holds the same bullion, so choosing between them changes cost, liquidity
and custody — never exposure. That is the identical situation the spot-bitcoin shelf already solved
with `preferred_btc_etf`, so gold reuses the same implementation (`prefer_vehicle`) rather than a
second copy that drifts from it. The spread is wider here than it is for bitcoin: GLD charges 0.400%
for what GLDM holds at 0.100% and IAUM at 0.090%.

Measured 2026-08-28: the fee map had priced GLD at 0.400% the day before, and the `fast` arm still
bought GLD — because GLDM was not on `_SANDBOX_CORE` and therefore was never a candidate. Pricing a
cheaper vehicle the model cannot choose changes nothing, so the shelf now stocks it and the
preference filter keeps exactly one of the pair.

The rules these pin are `prefer_vehicle`'s, exercised through the gold family. Their bitcoin
equivalents live in test_btc_etf_preference.py and both must keep passing off the one implementation.
"""
from __future__ import annotations

from app.sandbox_job import GOLD_ETFS, group_representative, prefer_gold_etf

PRICES = {"GLD": 409.38, "GLDM": 91.17, "IAUM": 45.93, "IAU": 86.62}


def _px(sym: str):
    return PRICES.get(sym.upper())


def _buy(sym: str, **kw):
    return {"symbol": sym, "side": "buy", "dollars": 800.0, "conviction": 70,
            "reason": "close the gold gap", **kw}


def _pos(sym: str, shares: float = 2.0):
    return {"symbol": sym, "shares": shares, "avg_cost": PRICES[sym]}


def test_a_gld_buy_is_routed_to_the_cheaper_gldm():
    out, notes = prefer_gold_etf([_buy("GLD")], preferred="GLDM", positions=[], price_of=_px)
    assert out[0]["symbol"] == "GLDM"
    assert notes == ["GLD→GLDM (preferred gold ETF)"]


def test_the_notional_survives_the_share_price_difference():
    """GLD is ~$409 and GLDM ~$91. A raw share count carried across would buy 4.5x the intended
    exposure, which is the whole reason this routes in dollars."""
    out, _ = prefer_gold_etf([{"symbol": "GLD", "side": "buy", "shares": 2.0}],
                             preferred="GLDM", positions=[], price_of=_px)
    assert out[0]["symbol"] == "GLDM"
    assert out[0]["dollars"] == round(2.0 * 409.38, 2)
    assert "shares" not in out[0]


def test_an_add_to_an_existing_gld_position_is_left_alone():
    """Consolidate, don't fragment — and the case that matters, because every arm holds GLD today.
    Routing this would leave two positions where the cap logic expects one, and the preference is
    about new exposure rather than about the gold already owned."""
    out, notes = prefer_gold_etf([_buy("GLD")], preferred="GLDM",
                                 positions=[_pos("GLD")], price_of=_px)
    assert out[0]["symbol"] == "GLD"
    assert notes == []


def test_a_sell_is_never_rerouted():
    """You can only sell what you hold; rewriting the symbol turns a valid exit into a not-held skip."""
    out, notes = prefer_gold_etf([{"symbol": "GLD", "side": "sell", "shares": 2.0}],
                                 preferred="GLDM", positions=[_pos("GLD")], price_of=_px)
    assert out[0]["symbol"] == "GLD"
    assert notes == []


def test_no_route_onto_something_that_cannot_fill():
    """An unpriced preference would convert a valid order into a guaranteed skip."""
    out, notes = prefer_gold_etf([_buy("GLD")], preferred="GLDM", positions=[],
                                 price_of=lambda s: None)
    assert out[0]["symbol"] == "GLD"
    assert notes == []


def test_the_entry_zone_does_not_travel_with_the_symbol():
    """A zone priced against $409 GLD would gate a $91 GLDM against a band from another instrument,
    and with respect_entry_zones on that blocks every substituted buy forever."""
    out, _ = prefer_gold_etf([_buy("GLD", entry_low=400.0, entry_high=415.0)],
                             preferred="GLDM", positions=[], price_of=_px)
    assert "entry_low" not in out[0] and "entry_high" not in out[0]


def test_an_unknown_preference_is_ignored_rather_than_obeyed():
    out, notes = prefer_gold_etf([_buy("GLD")], preferred="NOTATICKER",
                                 positions=[], price_of=_px)
    assert out[0]["symbol"] == "GLD"
    assert notes == []


def test_non_gold_orders_pass_through_untouched():
    orders = [_buy("AMZN"), _buy("VTI")]
    out, notes = prefer_gold_etf(orders, preferred="GLDM", positions=[], price_of=_px)
    assert [o["symbol"] for o in out] == ["AMZN", "VTI"]
    assert notes == []


def test_the_substitution_is_stated_in_the_reason():
    """A silent symbol rewrite would make the trade log disagree with what the model proposed."""
    out, _ = prefer_gold_etf([_buy("GLD")], preferred="GLDM", positions=[], price_of=_px)
    assert "routed GLD→GLDM" in out[0]["reason"]
    assert "close the gold gap" in out[0]["reason"]


def test_the_mechanical_engine_honours_the_same_preference():
    """rules_decision must not quietly pick a different vehicle than the LLM path would."""
    assert group_representative("GOLD", positions=[], price_of=_px,
                                group_of=lambda s: "GOLD" if s.upper() in GOLD_ETFS else s.upper(),
                                preferred_gold_etf="GLDM") == "GLDM"


def test_the_mechanical_engine_still_prefers_what_is_already_held():
    """Held-first outranks the preference, for the same anti-fragmentation reason."""
    assert group_representative("GOLD", positions=[_pos("GLD")], price_of=_px,
                                group_of=lambda s: "GOLD" if s.upper() in GOLD_ETFS else s.upper(),
                                preferred_gold_etf="GLDM") == "GLD"
