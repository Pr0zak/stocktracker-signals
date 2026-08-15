"""A broken price feed must not be able to corrupt the ledger or the equity curve.

Found 2026-08-14 by adversarial fuzzing of `validate_and_fill` -- 4000 scenarios built from boundary
values rather than uniform randoms, because the codebase had already learned that random floats never
land on the boundaries where this arithmetic breaks.

The failure chain, all of which the existing guards permitted:

  1. `not px or px <= 0` rejects an absent, zero or negative quote -- but a feed glitch does not
     politely return exactly 0.0. A quote of 1e-9 passed every check.
  2. A $100 order at 1e-9 filled 99,950,024,987 shares and logged its price as $0.00, which is 1e-9
     rounded to two decimals. Cash was conserved throughout, so the cash-conservation assert that
     guards this module -- the one designed to catch exactly this class of damage -- saw nothing.
  3. The damage lands the NEXT day. Once the feed recovers, marking that position produced a NAV
     point of $23,128,435,791,892 against $10,000 funded.
  4. sandbox_nav.jsonl is APPEND-ONLY. That point could never be retracted, so one bad quote
     permanently destroys the equity curve and every benchmark comparison drawn from it.

Two guards, at different layers, because either alone leaves the other hole open: a price floor on
fills, and a refusal to write a non-finite NAV point however it arose.
"""
from __future__ import annotations

import math

import pytest

from app.sandbox_job import MIN_FILL_PRICE, nav_row, positions_value, validate_and_fill

GROUPS = {"VTI": "US_EQUITY", "VOO": "US_EQUITY", "FBTC": "BTC"}


def _group(sym: str) -> str:
    return GROUPS.get(sym.upper(), sym.upper())


def _blob(cash=10_000.0, positions=None):
    return {
        "cash": cash, "positions": positions or [], "realized_pl_total": 0.0,
        "funded_total": 10_000.0, "benchmark": {"symbol": "^GSPC", "shares": 1.0, "cost_basis": 1.0},
        "settings": {
            "max_position_pct": 25.0, "cash_floor_pct": 5.0, "slippage_bps": 5,
            "min_conviction_to_trade": 55, "max_trades_per_tick": 4,
            "max_new_positions_per_tick": 2, "max_turnover_pct": 0.0,
            "respect_entry_zones": False, "allow_crypto": True, "allow_crypto_etf": True,
            "account_type": "cash", "avoid_wash_sales": False, "preferred_btc_etf": "FBTC",
        },
        "unsettled": [], "recent_loss_sales": {},
    }


def _buy(sym, dollars=100.0):
    return [{"symbol": sym, "side": "buy", "shares": 0.0, "dollars": dollars,
             "conviction": 80, "reason": "t"}]


def _fill(orders, px, blob=None):
    return validate_and_fill(blob or _blob(), orders, lambda s: px,
                             group_of=_group, now_ts=1_786_000_000.0)


def test_a_sub_penny_quote_on_an_equity_is_refused():
    # The exact fuzz repro. Before the guard this filled 99,950,024,987 shares.
    _, filled, skipped = _fill(_buy("AAPL"), 1e-9)
    assert filled == []
    assert "sub-penny floor" in skipped[0]["skip_reason"]


def test_the_floor_is_a_penny_and_a_penny_itself_still_trades():
    # A real penny quote is a real price. The guard rejects BELOW the floor, not at it.
    _, filled, _ = _fill(_buy("AAPL"), MIN_FILL_PRICE)
    assert filled and filled[0]["shares"] > 0
    _, filled, skipped = _fill(_buy("AAPL"), MIN_FILL_PRICE - 0.001)
    assert filled == [] and skipped


def test_crypto_is_exempt_because_it_fills_fractionally():
    # SHIB trades around $0.00001. There is no minimum lot, so a sub-penny coin is not a broken
    # quote -- applying an equity floor here would make a legitimate asset permanently unbuyable.
    _, filled, _ = _fill(_buy("SHIB-USD"), 0.00001)
    assert filled and filled[0]["shares"] > 0


def test_a_sell_is_refused_at_a_sub_penny_quote_too():
    # Not just buys. Selling a real position into a broken quote books a near-zero credit and
    # destroys the shares -- worse than the buy case, because the position cannot be recovered.
    held = [{"symbol": "AAPL", "shares": 10.0, "avg_cost": 200.0,
             "exposure_group": "AAPL", "opened_at": 1.0, "last_add_at": 1.0}]
    orders = [{"symbol": "AAPL", "side": "sell", "shares": 10.0, "dollars": 0.0,
               "conviction": 90, "reason": "t"}]
    nb, filled, skipped = _fill(orders, 1e-9, blob=_blob(positions=held))
    assert filled == []
    assert "sub-penny floor" in skipped[0]["skip_reason"]
    assert nb["positions"][0]["shares"] == 10.0      # the position survives intact


def test_a_normal_quote_is_untouched():
    _, filled, _ = _fill(_buy("AAPL", dollars=1000.0), 231.40)
    assert filled[0]["shares"] == 4.0                 # 1000 / 231.52 after slippage


def test_nav_row_refuses_to_write_a_non_finite_point():
    """The second layer. A price of inf still reaches positions_value, and the log is append-only.

    There is no sensible substitute value: 0 reads as a wiped-out account, and carrying the last
    known equity forward is a fabrication. So fail closed and let the caller abort with the ledger
    unsaved -- a missing point is recoverable on the next tick, a poisoned one never is.
    """
    blob = _blob()
    for bad in (float("inf"), float("nan")):
        with pytest.raises(AssertionError, match="non-finite NAV point"):
            nav_row(blob, positions_val=bad, spy_price=6800.0)


def test_nav_row_still_writes_ordinary_points():
    row = nav_row(_blob(cash=1_000.0), positions_val=500.0, spy_price=6800.0)
    assert row["equity"] == 1_500.0
    assert math.isfinite(row["equity"])


def test_positions_value_falls_back_to_cost_on_an_unusable_quote():
    # Already correct, pinned because the nav_row guard above is the only thing behind it. NaN,
    # negative and zero all fall back to cost basis rather than marking the position at nothing.
    pos = [{"symbol": "AAPL", "shares": 10.0, "avg_cost": 100.0}]
    for px in (float("nan"), -50.0, 0.0, None):
        assert positions_value(pos, lambda s: px) == 1_000.0


def test_a_sub_penny_crypto_fill_is_logged_at_its_real_price():
    """The audit log must not record a real purchase as having happened at zero.

    Crypto is exempt from the fill floor above, correctly -- SHIB trades near $0.000012 and fills
    fractionally. But the trade row rounded price to 4dp, so that fill was written to the append-only
    log as `price: 0.0` with `avg_cost_after: 0.0`. The POSITION kept full precision, so no money was
    lost and unrealized P/L stayed right; the damage was confined to the record, which is exactly the
    "confident number that is simply untrue" class this codebase keeps having to fix.
    """
    blob = _blob()
    orders = [{"symbol": "SHIB-USD", "side": "buy", "shares": 0.0, "dollars": 500.0,
               "conviction": 80, "reason": "t"}]
    nb, filled, _ = validate_and_fill(blob, orders, lambda s: 0.00001208,
                                      group_of=_group, now_ts=1_786_000_000.0)
    row = filled[0]
    assert row["gross"] == 500.0
    assert row["price"] > 0, "a real fill must not be logged at zero"
    assert row["price"] == pytest.approx(0.00001208, rel=1e-3)
    assert row["avg_cost_after"] > 0
    # The position was always right; pinned so a future rounding change cannot break it either.
    assert nb["positions"][0]["avg_cost"] == pytest.approx(0.00001208, rel=1e-2)


def test_ordinary_prices_keep_four_decimals():
    # The sub-penny branch must not change anything at or above a penny. 5bps of buy-side slippage
    # on 231.4012345 is 231.516870..., which still rounds to four decimals as it always did.
    _, filled, _ = _fill(_buy("AAPL", dollars=1000.0), 231.4012345)
    assert filled[0]["price"] == 231.5169
