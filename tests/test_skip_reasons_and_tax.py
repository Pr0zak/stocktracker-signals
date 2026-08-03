"""Blocked-trade diagnostics and holding-period annotation.

Both exist for the same reason: the weekly strategy review reads these, so a wrong one teaches the
strategist that the wrong thing is binding and it re-plans around a limit that was never in the way.
"""
from __future__ import annotations

import tempfile
import time

import pytest

from app import sandbox_job


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setenv("SIGNALS_DATA_DIR", tempfile.mkdtemp())


def _settings(**over):
    s = {
        "master_enabled": True, "max_position_pct": 25.0, "cash_floor_pct": 5.0,
        "min_conviction_to_trade": 55, "max_trades_per_tick": 4, "max_new_positions_per_tick": 2,
        "max_turnover_pct": 0.0, "account_type": "cash", "respect_entry_zones": False,
        "slippage_bps": 5, "avoid_wash_sales": False,
    }
    s.update(over)
    return s


def _blob(cash=10000.0, positions=None, **over):
    b = {"cash": cash, "positions": positions or [], "settings": _settings(**over.pop("settings", {})),
         "realized_pl_total": 0.0, "funded_total": cash}
    b.update(over)
    return b


def _run(blob, orders, prices):
    return sandbox_job.validate_and_fill(
        blob, orders, lambda s: prices.get(s.upper()),
        group_of=lambda s: s.upper(), source="test",
    )


def _skip_reason(blob, orders, prices):
    _, _, skipped = _run(blob, orders, prices)
    assert skipped, "expected the order to be skipped"
    return skipped[0]["skip_reason"]


# ------------------------------------------------------------------ the mislabel that started this

def test_an_undersized_order_does_not_blame_the_cash_floor():
    """Measured 2026-08-03: a VTI buy logged "cash floor left room for less than one share" while
    cash held $5,000 against a $534 floor — thirteen shares' worth. The order itself was simply
    smaller than one share, and the catch-all `else` named the cash floor regardless."""
    why = _skip_reason(
        _blob(cash=10000.0), [{"symbol": "VTI", "side": "buy", "dollars": 200.0, "conviction": 80}],
        {"VTI": 374.0},
    )
    assert "no limit was binding" in why
    assert "cash floor" not in why


def test_the_undersized_message_names_the_numbers():
    why = _skip_reason(
        _blob(cash=10000.0), [{"symbol": "VTI", "side": "buy", "dollars": 200.0, "conviction": 80}],
        {"VTI": 374.0},
    )
    assert "$200" in why and "374" in why


# ------------------------------------------------------------------ the real limits still name themselves

def test_the_cash_floor_is_named_when_it_actually_binds():
    """Plenty of cap room, but the floor really does leave under a share."""
    why = _skip_reason(
        _blob(cash=500.0, settings={"cash_floor_pct": 50.0, "max_position_pct": 100.0}),
        [{"symbol": "VTI", "side": "buy", "dollars": 5000.0, "conviction": 80}],
        {"VTI": 374.0},
    )
    assert "cash floor" in why


def test_the_exposure_cap_is_named_when_it_binds():
    blob = _blob(cash=10000.0, positions=[{"symbol": "VTI", "shares": 6.0, "avg_cost": 362.0,
                                           "exposure_group": "VTI"}],
                 settings={"max_position_pct": 21.0})
    why = _skip_reason(blob, [{"symbol": "VTI", "side": "buy", "dollars": 5000.0, "conviction": 80}],
                       {"VTI": 374.0})
    assert "cap" in why


def test_the_turnover_cap_is_named_when_it_binds():
    why = _skip_reason(
        _blob(cash=10000.0, settings={"max_turnover_pct": 2.0}),
        [{"symbol": "VTI", "side": "buy", "dollars": 5000.0, "conviction": 80}],
        {"VTI": 374.0},
    )
    assert "turnover" in why


# ------------------------------------------------------------------ holding period / capital gains

def test_a_young_position_is_marked_short_term():
    now = time.time()
    book = [{"symbol": "AMZN"}]
    sandbox_job.annotate_holding_period(
        book, [{"symbol": "AMZN", "last_add_at": now - 4 * 86_400}], now_ts=now)
    assert book[0]["holding_days"] == 4
    assert book[0]["capital_gains"] == "short_term"
    assert book[0]["days_to_long_term"] == sandbox_job._LONG_TERM_DAYS - 4


def test_a_seasoned_position_is_marked_long_term_with_no_countdown():
    now = time.time()
    book = [{"symbol": "SPY"}]
    sandbox_job.annotate_holding_period(
        book, [{"symbol": "SPY", "last_add_at": now - 400 * 86_400}], now_ts=now)
    assert book[0]["capital_gains"] == "long_term"
    assert "days_to_long_term" not in book[0]


def test_the_boundary_is_more_than_one_year():
    now = time.time()

    def status(days):
        book = [{"symbol": "X"}]
        sandbox_job.annotate_holding_period(
            book, [{"symbol": "X", "last_add_at": now - days * 86_400}], now_ts=now)
        return book[0]["capital_gains"]

    assert status(364) == "short_term"
    assert status(365) == "short_term"   # "more than one year", not "one year"
    assert status(366) == "long_term"


def test_adding_to_a_position_restarts_the_clock():
    """Each tax lot is clocked separately; using opened_at would overstate what qualifies."""
    now = time.time()
    book = [{"symbol": "VTI"}]
    sandbox_job.annotate_holding_period(book, [{
        "symbol": "VTI", "opened_at": now - 400 * 86_400, "last_add_at": now - 10 * 86_400,
    }], now_ts=now)
    assert book[0]["capital_gains"] == "short_term"
    assert book[0]["holding_days"] == 10


def test_a_position_missing_from_the_ledger_is_left_untouched():
    book = [{"symbol": "GHOST"}]
    sandbox_job.annotate_holding_period(book, [], now_ts=time.time())
    assert "holding_days" not in book[0]


def test_annotation_never_gates_a_sell():
    """Informational only — a 1-day-old winner must still be sellable when the model decides to.
    Tax efficiency is a preference for the prompt to weigh, never a rule the validator enforces."""
    now = time.time()
    blob = _blob(cash=1000.0, positions=[{"symbol": "AMZN", "shares": 5.0, "avg_cost": 238.0,
                                          "exposure_group": "AMZN", "last_add_at": now - 86_400}])
    _, filled, _ = _run(blob, [{"symbol": "AMZN", "side": "sell", "shares": 5.0}], {"AMZN": 284.0})
    assert filled and filled[0]["symbol"] == "AMZN"


def test_a_one_share_position_cannot_be_trimmed_and_says_so():
    """Stocks round to whole shares, so a partial trim of a 1-share position is 0. The model spent an
    order on an arithmetically impossible trim on 2026-08-03."""
    blob = _blob(cash=1000.0, positions=[{"symbol": "AMZN", "shares": 1.0, "avg_cost": 238.0,
                                          "exposure_group": "AMZN"}])
    why = _skip_reason(blob, [{"symbol": "AMZN", "side": "sell", "shares": 0.4}], {"AMZN": 284.0})
    assert "nothing to sell" in why
