"""An entry zone the market is nowhere near is a broken order, not a patient one.

Measured 2026-08-21: an order to buy GLD carried an entry zone of $170-180 while GLD traded at
$423.43 -- the market 135% above the top of its own limit. The review model found the cause in the
same tick: the order cited technicals "for a ticker that appears nowhere in the candidate data", so
the model had written specific price levels for a symbol it had no price for.

Parking assumes the thesis is sound and only the moment is wrong. That assumption does not survive a
135% gap, so the order is refused rather than held for an intraday re-check that could never fill.
"""
from __future__ import annotations

from app.sandbox_job import MAX_ZONE_DISTANCE_PCT, validate_and_fill

NOW = 1_787_000_000.0


def _group(sym):
    return sym.upper()


def _blob():
    return {
        "cash": 10_000.0, "positions": [], "realized_pl_total": 0.0, "funded_total": 10_000.0,
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


def _buy(low, high, dollars=1_000.0):
    return [{"symbol": "GLD", "side": "buy", "shares": 0.0, "dollars": dollars,
             "conviction": 80, "reason": "t", "entry_low": low, "entry_high": high}]


def _run(orders, price):
    return validate_and_fill(_blob(), orders, lambda s: price, group_of=_group, now_ts=NOW)


def test_the_gld_order_that_prompted_this_is_refused_not_parked():
    nb, filled, skipped = _run(_buy(170.0, 180.0), 423.43)
    assert filled == []
    assert "the zone is the error, not the timing" in skipped[0]["skip_reason"]
    # Crucially NOT parked: an intraday re-check could never fill this, so holding it would just
    # carry a broken order around all afternoon.
    assert nb["parked_orders"] == []


def test_a_plausible_limit_is_still_parked_rather_than_refused():
    # VXUS the same day: zone 72-78 against a market of 87.71, about 12% away. That is a real
    # patient bid and must keep behaving like one.
    nb, filled, skipped = _run(_buy(72.0, 78.0), 87.71)
    assert filled == []
    assert "parked for today" in skipped[0]["skip_reason"]
    assert len(nb["parked_orders"]) == 1


def test_a_zone_far_above_the_market_is_refused_too():
    # The other direction currently FILLS, since buying below the zone is cheaper than asked. But
    # "cheaper than intended" is no comfort when the intention was formed against the wrong number.
    _, filled, skipped = _run(_buy(400.0, 420.0), 100.0)
    assert filled == []
    assert "the zone is the error" in skipped[0]["skip_reason"]


def test_the_boundary_is_wide_enough_to_never_catch_a_real_bid():
    # Just inside the threshold parks; just outside is refused. 50% is deliberately loose -- nothing
    # legitimate is half the share price away from the market.
    inside = 100.0 * (1 + (MAX_ZONE_DISTANCE_PCT - 5) / 100.0)
    outside = 100.0 * (1 + (MAX_ZONE_DISTANCE_PCT + 5) / 100.0)
    nb, _, skipped = _run(_buy(90.0, 100.0), inside)
    assert "parked for today" in skipped[0]["skip_reason"] and nb["parked_orders"]
    nb, _, skipped = _run(_buy(90.0, 100.0), outside)
    assert "the zone is the error" in skipped[0]["skip_reason"] and not nb["parked_orders"]


def test_an_order_with_no_zone_is_unaffected():
    _, filled, _ = _run(_buy(None, None), 423.43)
    assert filled and filled[0]["shares"] > 0
