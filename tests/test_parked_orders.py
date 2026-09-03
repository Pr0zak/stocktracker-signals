"""A buy refused only on price is parked, not discarded.

`respect_entry_zones` defers a buy when the market sits above the zone the analyst named. With one
decision per trading day that made the zone nearly decorative: the price had to be inside it at
15:35 ET, or the buy waited a full session and was re-derived from scratch the next afternoon. An
entry zone is a limit order in everything but name, and nothing was holding the order between looks.

Parked orders are same-day only. A zone describes today's setup; carrying one into tomorrow would
execute a thesis nobody re-examined.
"""
from __future__ import annotations

import datetime as dt

from app.sandbox_job import ET, validate_and_fill

# Every call below pins now_ts, so the parked date is derived from THAT, not from the wall clock.
# A test that compared against "today" would pass on the day it was written and drift afterwards.
FIXED_TS = 1_786_000_000.0
FIXED_DATE = dt.datetime.fromtimestamp(FIXED_TS, ET).date().isoformat()


def _group(sym: str) -> str:
    return sym.upper()


def _blob(cash=10_000.0):
    return {
        "cash": cash, "positions": [], "realized_pl_total": 0.0, "funded_total": 10_000.0,
        "benchmark": {"symbol": "^GSPC", "shares": 1.0, "cost_basis": 1.0},
        "settings": {
            "max_position_pct": 25.0, "cash_floor_pct": 5.0, "slippage_bps": 5,
            "min_conviction_to_trade": 55, "max_trades_per_tick": 4,
            "max_new_positions_per_tick": 2, "max_turnover_pct": 0.0,
            "respect_entry_zones": True, "allow_crypto": False, "allow_crypto_etf": True,
            "account_type": "cash", "avoid_wash_sales": False, "preferred_btc_etf": "FBTC",
        },
        "unsettled": [], "recent_loss_sales": {},
    }


def _buy(sym="AAPL", dollars=1_000.0, low=200.0, high=210.0):
    return [{"symbol": sym, "side": "buy", "shares": 0.0, "dollars": dollars,
             "conviction": 80, "reason": "t", "entry_low": low, "entry_high": high}]


def test_a_buy_above_its_zone_is_parked_with_the_price_that_refused_it():
    nb, filled, skipped = validate_and_fill(
        _blob(), _buy(), lambda s: 230.0, group_of=_group, now_ts=FIXED_TS)
    assert filled == []
    assert "parked for today" in skipped[0]["skip_reason"]
    parked = nb["parked_orders"]
    assert len(parked) == 1
    assert parked[0]["symbol"] == "AAPL"
    assert parked[0]["parked_price"] == 230.0
    assert parked[0]["parked_date"] == FIXED_DATE
    assert parked[0]["entry_high"] == 210.0        # the zone travels with the order


def test_a_buy_inside_its_zone_fills_and_parks_nothing():
    nb, filled, _ = validate_and_fill(
        _blob(), _buy(), lambda s: 205.0, group_of=_group, now_ts=FIXED_TS)
    assert filled and filled[0]["shares"] > 0
    assert nb["parked_orders"] == []


def test_a_price_below_the_zone_is_not_parked():
    # Cheaper than the analyst wanted to pay is not a reason to wait. Only ABOVE the zone defers.
    nb, filled, _ = validate_and_fill(
        _blob(), _buy(), lambda s: 150.0, group_of=_group, now_ts=FIXED_TS)
    assert filled and filled[0]["shares"] > 0
    assert nb["parked_orders"] == []


def test_parking_is_replaced_each_tick_not_accumulated():
    # Yesterday's parked order must not survive into today's list. Appending would build a queue of
    # theses nobody re-examined, and the oldest would be the least examined of all.
    blob = _blob()
    blob["parked_orders"] = [{"symbol": "STALE", "side": "buy", "parked_date": "2020-01-01"}]
    nb, _, _ = validate_and_fill(
        blob, _buy(), lambda s: 230.0, group_of=_group, now_ts=FIXED_TS)
    assert [o["symbol"] for o in nb["parked_orders"]] == ["AAPL"]


def test_nothing_is_parked_when_entry_zones_are_switched_off():
    blob = _blob()
    blob["settings"]["respect_entry_zones"] = False
    nb, filled, _ = validate_and_fill(
        blob, _buy(), lambda s: 230.0, group_of=_group, now_ts=FIXED_TS)
    assert filled and filled[0]["shares"] > 0     # chases the price, as configured
    assert nb["parked_orders"] == []


def test_an_order_refused_for_a_reason_other_than_price_is_not_parked():
    # Parking says "right idea, wrong moment". A conviction floor rejection is not that -- it says
    # the idea itself did not clear the bar, and re-checking the price later would not change it.
    blob = _blob()
    orders = _buy()
    orders[0]["conviction"] = 10
    nb, filled, skipped = validate_and_fill(
        blob, orders, lambda s: 230.0, group_of=_group, now_ts=FIXED_TS)
    assert filled == []
    assert "conviction" in skipped[0]["skip_reason"]
    assert nb["parked_orders"] == []


def test_a_ticker_excluded_after_parking_is_not_bought_by_the_sweep():
    """The parked sweep at main.py's /sandbox/fill_parked omitted `exclude`, alone among the three
    validate_and_fill call sites.

    validate_and_fill reads every OTHER live setting off the blob itself — a raised conviction floor,
    a zeroed trade cap or a closed gate all block a parked fill — so exclusions was the single
    constraint that could be added between the 14:35 tick and the 14:40 sweep and still be ignored.
    `exclusions` is documented as the tickers the AI must never buy.
    """
    order = {"symbol": "SCHD", "side": "buy", "dollars": 400.0, "reason": "parked earlier",
             "conviction": 70}
    blob = _blob(cash=5_000.0)
    blob["settings"]["exclusions"] = ["SCHD"]

    _, filled, skipped = validate_and_fill(
        blob, [dict(order)], lambda s: 20.0, group_of=lambda s: s, source="parked_fill",
        exclude={str(x).upper() for x in blob["settings"]["exclusions"]},
    )
    assert filled == []
    assert [r["symbol"] for r in skipped] == ["SCHD"]
    assert "excluded" in skipped[0]["skip_reason"]
