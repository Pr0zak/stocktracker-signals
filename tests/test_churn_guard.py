"""A small loss on a days-old position is noise, not a broken thesis.

Measured 2026-08-18: MSFT was bought on the 14th because the plan wanted 8% of it, and sold four days
later at -2.5% for "short-term loss harvesting" -- $12.33 of loss in a PAPER account with no tax to
harvest, costing two legs of slippage and locking the name out for 30 days under the wash-sale rule.
The prompt covered the mirror image ("up 19% in four days is not a reason to sell") and said nothing
about the downside version, so the model walked through the gap.

TWO conditions, not one. A blanket ban on short-hold loss sales would block the legitimate case:
protecting from a genuine downturn also realises a loss and can also be right on day two. What
separates it from churn is the SIZE -- a position down 15% in a week is a thesis failing, one down
2.5% is noise being traded.
"""
from __future__ import annotations

from app.sandbox_job import MIN_HOLD_DAYS_FOR_SMALL_LOSS, SMALL_LOSS_PCT, validate_and_fill

NOW = 1_787_000_000.0
DAY = 86_400.0


def _group(sym: str) -> str:
    return sym.upper()


def _blob(avg_cost=495.56, bought_days_ago=4.0, shares=1.0):
    return {
        "cash": 5_000.0,
        "positions": [{"symbol": "MSFT", "shares": shares, "avg_cost": avg_cost,
                       "exposure_group": "MSFT", "opened_at": NOW - bought_days_ago * DAY,
                       "last_add_at": NOW - bought_days_ago * DAY}],
        "realized_pl_total": 0.0, "funded_total": 10_000.0,
        "benchmark": {"symbol": "^GSPC", "shares": 1.0, "cost_basis": 1.0},
        "settings": {
            "max_position_pct": 25.0, "cash_floor_pct": 5.0, "slippage_bps": 5,
            "min_conviction_to_trade": 55, "max_trades_per_tick": 4,
            "max_new_positions_per_tick": 2, "max_turnover_pct": 0.0,
            "respect_entry_zones": False, "allow_crypto": False, "allow_crypto_etf": True,
            "account_type": "cash", "avoid_wash_sales": False, "preferred_btc_etf": "FBTC",
        },
        "unsettled": [], "recent_loss_sales": {},
    }


def _sell(shares=1.0):
    return [{"symbol": "MSFT", "side": "sell", "shares": shares, "dollars": 0.0,
             "conviction": 70, "reason": "short-term loss harvesting"}]


def _run(blob, price, liquidation=False):
    return validate_and_fill(blob, _sell(), lambda s: price, group_of=_group,
                             now_ts=NOW, liquidation=liquidation)


def test_the_msft_trade_that_prompted_this_is_now_refused():
    # The exact case: bought at 495.56, sold four days later at 483.23 for -2.5%.
    nb, filled, skipped = _run(_blob(), 483.23)
    assert filled == []
    assert "churn guard" in skipped[0]["skip_reason"]
    assert nb["positions"][0]["shares"] == 1.0          # the position survives untouched
    assert nb["cash"] == 5_000.0                        # and so does the cash


def test_a_real_downturn_can_still_be_cut_on_day_two():
    # The legitimate case the guard must not block. Down 20% two days in is a thesis failing, and
    # protecting from a downturn is an explicitly allowed reason to sell.
    _, filled, _ = _run(_blob(bought_days_ago=2.0), 396.00)
    assert filled and filled[0]["shares"] == 1.0
    assert filled[0]["realized_pl"] < 0


def test_the_same_small_loss_is_allowed_once_the_position_is_old_enough():
    # The guard is about churn, not about losses. A week later the same trade goes through.
    _, filled, _ = _run(_blob(bought_days_ago=MIN_HOLD_DAYS_FOR_SMALL_LOSS + 1), 483.23)
    assert filled and filled[0]["shares"] == 1.0


def test_selling_at_a_profit_is_never_blocked():
    # Only losses are in scope. Taking profit is governed by the extension rules, not by this.
    _, filled, _ = _run(_blob(bought_days_ago=1.0), 520.00)
    assert filled and filled[0]["realized_pl"] > 0


def test_an_exit_date_liquidation_is_exempt():
    # A scheduled flatten is the user's instruction, not the strategy's choice. Throttling it would
    # leave the account holding positions it was told to be out of.
    _, filled, _ = _run(_blob(), 483.23, liquidation=True)
    assert filled and filled[0]["shares"] == 1.0


def test_the_boundary_is_the_loss_size_not_just_the_age():
    # Just inside the loss threshold on a new position: blocked. Just outside: allowed. Pins that
    # BOTH conditions are required, so a future edit cannot quietly make it age-only.
    cost = 100.0
    inside = cost * (1 - (SMALL_LOSS_PCT - 1) / 100.0)
    outside = cost * (1 - (SMALL_LOSS_PCT + 1) / 100.0)
    _, filled, skipped = _run(_blob(avg_cost=cost, bought_days_ago=1.0), inside)
    assert filled == [] and skipped
    _, filled, _ = _run(_blob(avg_cost=cost, bought_days_ago=1.0), outside)
    assert filled
