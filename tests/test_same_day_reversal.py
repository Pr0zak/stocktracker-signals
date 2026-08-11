"""The tick may not sell back what it bought earlier the same day.

Measured 2026-08-11. The 14:35 tick bought 7 VXUS toward a 22% target. A second tick 54 minutes
later re-ran the weekly strategy review, which rewrote that target to 12%, and then sold 5 of those
shares back. Neither decision was wrong against the plan in front of it. The account still paid the
spread twice and booked $3.84 of short-term gain — realised against the position's $86.19 average
cost, so a round trip worth nine cents on the shares themselves became a taxable event forty times
that size.

`intra_group_swaps` cannot see this: the legs are in different ticks and never share an order list.
The link is the position's `last_add_at`, which the fill path already stamps on every buy.
"""
from __future__ import annotations

import datetime as dt

from app import sandbox_job
from app.sandbox_job import ET, same_day_reversals


def _ts(y, m, d, hh, mm) -> float:
    return dt.datetime(y, m, d, hh, mm, tzinfo=ET).timestamp()


NOW = _ts(2026, 8, 11, 15, 29)          # the forced tick that did it
EARLIER_TODAY = _ts(2026, 8, 11, 14, 35)  # the scheduled tick that bought
YESTERDAY = _ts(2026, 8, 10, 14, 35)


def _pos(sym, last_add_at):
    return {"symbol": sym, "shares": 21.0, "avg_cost": 86.19, "last_add_at": last_add_at}


def _order(side, sym):
    return {"side": side, "symbol": sym, "shares": 5.0}


# ---------------------------------------------------------------- the trade that motivated this

def test_the_real_reversal_is_blocked():
    orders = [_order("sell", "VXUS")]
    positions = [_pos("VXUS", EARLIER_TODAY)]
    assert same_day_reversals(orders, positions, now_ts=NOW) == {0}


def test_a_sell_of_something_bought_yesterday_is_fine():
    # An overnight decision is ordinary. Only the same session is churn.
    assert same_day_reversals(
        [_order("sell", "VXUS")], [_pos("VXUS", YESTERDAY)], now_ts=NOW) == set()


def test_the_boundary_is_the_trading_day_not_24_hours():
    """An hours-based window would block a normal next-morning sell and pass a 23-hour-old add."""
    yesterday_late = _ts(2026, 8, 10, 15, 55)     # 23.5h before NOW, but a DIFFERENT session
    assert same_day_reversals(
        [_order("sell", "VXUS")], [_pos("VXUS", yesterday_late)], now_ts=NOW) == set()
    today_early = _ts(2026, 8, 11, 9, 31)          # under 6h, same session
    assert same_day_reversals(
        [_order("sell", "VXUS")], [_pos("VXUS", today_early)], now_ts=NOW) == {0}


def test_the_day_is_measured_in_exchange_time():
    # 00:30 UTC on the 12th is still the evening of the 11th in New York. A UTC date would call this
    # a new day and wave the reversal through.
    late = dt.datetime(2026, 8, 12, 0, 30, tzinfo=dt.timezone.utc).timestamp()
    assert same_day_reversals(
        [_order("sell", "VXUS")], [_pos("VXUS", EARLIER_TODAY)], now_ts=late) == {0}


# ---------------------------------------------------------------- it must not over-block

def test_buys_are_never_blocked():
    # Adding again to something bought this morning is just following the plan.
    assert same_day_reversals(
        [_order("buy", "VXUS")], [_pos("VXUS", EARLIER_TODAY)], now_ts=NOW) == set()


def test_selling_an_untouched_position_is_fine():
    orders = [_order("sell", "AMZN")]
    positions = [_pos("VXUS", EARLIER_TODAY), _pos("AMZN", YESTERDAY)]
    assert same_day_reversals(orders, positions, now_ts=NOW) == set()


def test_only_the_offending_leg_is_dropped():
    orders = [_order("sell", "VXUS"), _order("sell", "AMZN"), _order("buy", "SCHD")]
    positions = [_pos("VXUS", EARLIER_TODAY), _pos("AMZN", YESTERDAY)]
    assert same_day_reversals(orders, positions, now_ts=NOW) == {0}


def test_a_position_with_no_timestamp_is_not_treated_as_bought_today():
    # Ledger rows predating last_add_at must not become unsellable.
    assert same_day_reversals(
        [_order("sell", "VXUS")], [{"symbol": "VXUS", "shares": 1.0}], now_ts=NOW) == set()
    assert same_day_reversals(
        [_order("sell", "VXUS")], [_pos("VXUS", 0)], now_ts=NOW) == set()


def test_symbols_match_case_insensitively():
    assert same_day_reversals(
        [{"side": "SELL", "symbol": "vxus"}], [_pos("VXUS", EARLIER_TODAY)], now_ts=NOW) == {0}


def test_empty_inputs_are_not_errors():
    assert same_day_reversals([], [], now_ts=NOW) == set()
    assert same_day_reversals([_order("sell", "VXUS")], [], now_ts=NOW) == set()


# ---------------------------------------------------------------- through the real validator


def _settings(**over):
    s = {"master_enabled": True, "max_position_pct": 25.0, "cash_floor_pct": 5.0,
         "min_conviction_to_trade": 55, "max_trades_per_tick": 4, "max_new_positions_per_tick": 2,
         "max_turnover_pct": 0.0, "account_type": "margin", "respect_entry_zones": False,
         "slippage_bps": 5, "avoid_wash_sales": False}
    s.update(over)
    return s


def _blob(last_add_at):
    return {"cash": 6000.0, "funded_total": 10500.0, "realized_pl_total": 87.83,
            "settings": _settings(),
            "positions": [{"symbol": "VXUS", "shares": 21.0, "avg_cost": 86.19,
                           "exposure_group": "VXUS", "opened_at": YESTERDAY,
                           "last_add_at": last_add_at}]}


def _sell(n=5.0):
    return {"side": "sell", "symbol": "VXUS", "shares": n, "conviction": 68,
            "reason": "Trim to 12% target"}


def test_the_reversal_is_refused_and_audited_not_swallowed():
    """The exact 2026-08-11 trade: no fill, no realised gain, and a skip row explaining why."""
    new_blob, filled, skipped = sandbox_job.validate_and_fill(
        _blob(EARLIER_TODAY), [_sell()], lambda s: {"VXUS": 86.9565}.get(s.upper()),
        group_of=lambda s: s.upper(), source="test", now_ts=NOW,
    )
    assert filled == []
    assert [r["symbol"] for r in skipped] == ["VXUS"]
    assert "bought earlier today" in skipped[0]["skip_reason"]
    assert new_blob["realized_pl_total"] == 87.83            # the $3.84 is never booked
    assert new_blob["positions"][0]["shares"] == 21.0        # nothing sold back


def test_the_same_sell_executes_the_next_day():
    _, filled, _ = sandbox_job.validate_and_fill(
        _blob(YESTERDAY), [_sell()], lambda s: {"VXUS": 86.9565}.get(s.upper()),
        group_of=lambda s: s.upper(), source="test", now_ts=NOW,
    )
    assert filled and filled[0]["symbol"] == "VXUS" and filled[0]["shares"] == 5.0


def test_an_exit_date_liquidation_is_never_blocked_by_this():
    """A flatten is the user asking to be out. Churn protection must not leave them holding stock."""
    _, filled, _ = sandbox_job.validate_and_fill(
        _blob(EARLIER_TODAY), [_sell(21.0)], lambda s: {"VXUS": 86.9565}.get(s.upper()),
        group_of=lambda s: s.upper(), source="exit_date", now_ts=NOW, liquidation=True,
    )
    assert filled and filled[0]["shares"] == 21.0
