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


# ------------------------------------------------------------------ rescuing the last share
#
# Measured 2026-08-13: a GOOGL buy of $346 against a $346.39 fill floored to zero shares and vanished,
# while the book held $6,032 of idle cash and an $862 target for that name. Flooring is right
# everywhere else — it can only ever leave a limit further away — but on the final share it converts
# an order into no order at all.

# GOOGL that day. `price_of` returns the raw quote; the 5bp slippage puts the fill just over $346.39,
# which is the whole point — the order was written under the price it would actually pay.
_GOOGL = 346.22


def _fill_row(blob, orders, prices):
    _, filled, skipped = _run(blob, orders, prices)
    assert filled, f"expected a fill, got skip: {skipped and skipped[0]['skip_reason']}"
    return filled[0]


def test_an_order_a_few_cents_under_one_share_rounds_up():
    row = _fill_row(_blob(cash=10000.0),
                    [{"symbol": "GOOGL", "side": "buy", "dollars": 346.0, "conviction": 62}],
                    {"GOOGL": _GOOGL})
    assert row["shares"] == 1.0


def test_the_rounded_up_fill_says_so_on_its_face():
    """A row showing $346.39 against a $346 order has to explain itself, or the trade log is
    quietly answering a question nobody can see it was asked."""
    row = _fill_row(_blob(cash=10000.0),
                    [{"symbol": "GOOGL", "side": "buy", "dollars": 346.0, "conviction": 62}],
                    {"GOOGL": _GOOGL})
    assert row["size_note"] and "rounded up" in row["size_note"]
    assert "$346" in row["size_note"]


def test_an_ordinary_fill_carries_no_size_note():
    row = _fill_row(_blob(cash=10000.0),
                    [{"symbol": "GOOGL", "side": "buy", "dollars": 2000.0, "conviction": 62}],
                    {"GOOGL": _GOOGL})
    assert row["shares"] == 5.0
    assert row["size_note"] is None


def test_the_rescue_does_not_reach_a_deliberately_fractional_order():
    """$200 of a $374 share is not a near-miss, it is a half-position — and rounding it up would be
    the server overriding the analyst's size rather than honouring it. This is the same case the
    2026-08-03 skip-reason work already ruled on; it must still skip."""
    _, filled, skipped = _run(
        _blob(cash=10000.0), [{"symbol": "VTI", "side": "buy", "dollars": 200.0, "conviction": 80}],
        {"VTI": 374.0})
    assert not filled
    assert "no limit was binding" in skipped[0]["skip_reason"]


def test_the_rescue_is_refused_when_the_cash_floor_cannot_fund_a_whole_share():
    blob = _blob(cash=400.0, settings={"cash_floor_pct": 50.0, "max_position_pct": 100.0})
    _, filled, skipped = _run(
        blob, [{"symbol": "GOOGL", "side": "buy", "dollars": 346.0, "conviction": 62}],
        {"GOOGL": _GOOGL})
    assert not filled, "rounding up must not spend through the cash floor"
    assert "cash floor" in skipped[0]["skip_reason"]


def test_the_rescue_is_refused_when_the_exposure_cap_cannot_take_a_whole_share():
    """The near-miss branch must not outrank a real cap: the order was one share's worth, so the cap
    is what stopped it and the cap is what the log has to say."""
    blob = _blob(cash=10000.0, settings={"max_position_pct": 3.0})
    _, filled, skipped = _run(
        blob, [{"symbol": "GOOGL", "side": "buy", "dollars": 346.0, "conviction": 62}],
        {"GOOGL": _GOOGL})
    assert not filled, "rounding up must not breach the per-group cap"
    assert "cap" in skipped[0]["skip_reason"]
    assert "no limit was binding" not in skipped[0]["skip_reason"]


def test_the_rescue_is_refused_when_the_turnover_budget_cannot_take_a_whole_share():
    blob = _blob(cash=10000.0, settings={"max_turnover_pct": 3.0})
    _, filled, skipped = _run(
        blob, [{"symbol": "GOOGL", "side": "buy", "dollars": 346.0, "conviction": 62}],
        {"GOOGL": _GOOGL})
    assert not filled, "rounding up must not breach the turnover budget"
    assert "turnover" in skipped[0]["skip_reason"]


def test_a_rounded_up_buy_conserves_cash():
    """The rescue adds shares the order did not pay for unless the extra cost is actually debited."""
    blob = _blob(cash=10000.0)
    after, filled, _ = _run(blob, [{"symbol": "GOOGL", "side": "buy", "dollars": 346.0,
                                    "conviction": 62}], {"GOOGL": _GOOGL})
    row = filled[0]
    assert after["cash"] == pytest.approx(10000.0 - row["gross"], abs=0.01)
    assert row["gross"] == pytest.approx(_GOOGL * 1.0005, abs=0.01)


def test_a_rounded_up_buy_lands_inside_the_cap_it_was_measured_against():
    """fill <= cap_room is checked before the round-up; this is the after-the-fact proof."""
    blob = _blob(cash=10000.0, settings={"max_position_pct": 4.0})
    after, filled, _ = _run(blob, [{"symbol": "GOOGL", "side": "buy", "dollars": 346.0,
                                    "conviction": 62}], {"GOOGL": _GOOGL})
    assert filled, "4% of $10,000 is $400 — one $346 share fits"
    pos = after["positions"][0]
    equity = after["cash"] + pos["shares"] * _GOOGL
    assert pos["shares"] * _GOOGL <= 0.04 * equity


def test_a_multi_share_order_priced_at_the_last_price_does_not_lose_a_share():
    """The same arithmetic as the GOOGL miss, one size up. Sizing N shares the obvious way — N times
    the last price — divides by a fill that slippage has already pushed above that price, landing at
    N - 0.0005. Flooring made every such order N-1 shares and left a share's worth of cash unspent."""
    for n in (2, 5, 10):
        row = _fill_row(_blob(cash=100_000.0),
                        [{"symbol": "GOOGL", "side": "buy", "dollars": n * _GOOGL, "conviction": 62}],
                        {"GOOGL": _GOOGL})
        assert row["shares"] == float(n), f"{n} shares' worth filled {row['shares']}"


def test_a_genuinely_fractional_multi_share_order_still_floors():
    """5.5 shares' worth is not a near-miss on 6 — it floors to 5, as every other size does."""
    row = _fill_row(_blob(cash=100_000.0),
                    [{"symbol": "GOOGL", "side": "buy", "dollars": 5.5 * _GOOGL, "conviction": 62}],
                    {"GOOGL": _GOOGL})
    assert row["shares"] == 5.0
    assert row["size_note"] is None


def test_a_limit_cut_order_never_rounds_up_into_the_limit():
    """When a limit is what sized the order, the remainder is an artefact of the limit — rounding up
    would spend straight through the thing that just cut it."""
    blob = _blob(cash=10_000.0, settings={"max_turnover_pct": 34.7})  # ~$3,470: 10 shares less a hair
    after, filled, _ = _run(blob, [{"symbol": "GOOGL", "side": "buy", "dollars": 100_000.0,
                                    "conviction": 62}], {"GOOGL": _GOOGL})
    row = filled[0]
    assert row["gross"] <= 3_470.0, "filled past the turnover budget"
    assert row["size_note"] is None


def test_an_undersized_order_under_a_tighter_cap_names_the_cap():
    """Both are true — the order was under a share AND the cap could not take one. The cap is the
    one that would still bind if the order were sized correctly, so the cap is what gets named."""
    why = _skip_reason(_blob(cash=10000.0, settings={"max_position_pct": 1.0}),
                       [{"symbol": "GOOGL", "side": "buy", "dollars": 50.0, "conviction": 62}],
                       {"GOOGL": _GOOGL})
    assert "cap" in why
    assert "no limit was binding" not in why


def test_the_named_limit_is_the_tightest_one_not_the_first_tested():
    """A 100% cap is inert; the cash floor is what actually leaves under a share. Naming the cap
    because it happens to be checked earlier would send the strategist after the wrong number."""
    why = _skip_reason(
        _blob(cash=400.0, settings={"cash_floor_pct": 50.0, "max_position_pct": 100.0}),
        [{"symbol": "GOOGL", "side": "buy", "dollars": 5000.0, "conviction": 62}], {"GOOGL": _GOOGL})
    assert "cash floor" in why
    assert "cap" not in why


def test_blocked_reasons_for_a_recurring_limit_are_identical_strings():
    """memory.blocked_summary() GROUPs BY this string. A per-event dollar figure in it splits one
    recurring problem into a column of counts of 1, which is how a standing blockage hides."""
    seen = set()
    for cash in (9_000.0, 9_500.0, 10_000.0):
        seen.add(_skip_reason(_blob(cash=cash, settings={"max_position_pct": 3.0}),
                              [{"symbol": "GOOGL", "side": "buy", "dollars": 5000.0, "conviction": 62}],
                              {"GOOGL": _GOOGL}))
    assert len(seen) == 1, f"cap reason varies per event: {seen}"


def test_t1_is_only_blamed_when_settling_would_actually_fix_it():
    """"Frees up next session" is a forecast. If the cash floor alone leaves under a share, next
    session looks identical and the strategist is told to wait for something that will not arrive."""
    # Sell frees $500 into T+1, but a 95% floor means even fully settled cash cannot buy a share.
    blob = _blob(cash=600.0, positions=[{"symbol": "OLD", "shares": 5.0, "avg_cost": 100.0,
                                         "exposure_group": "OLD"}],
                 settings={"cash_floor_pct": 95.0, "max_position_pct": 100.0})
    orders = [{"symbol": "OLD", "side": "sell", "shares": 5.0, "conviction": 90},
              {"symbol": "GOOGL", "side": "buy", "dollars": 5000.0, "conviction": 62}]
    _, _, skipped = _run(blob, orders, {"OLD": 100.0, "GOOGL": _GOOGL})
    why = [s for s in skipped if s["symbol"] == "GOOGL"][0]["skip_reason"]
    assert "cash floor" in why
    assert "next session" not in why


def test_t1_is_still_blamed_when_settling_really_would_fix_it():
    """The complement — the guard must not swing so far that a genuine T+1 hold reads as a floor."""
    blob = _blob(cash=100.0, positions=[{"symbol": "OLD", "shares": 20.0, "avg_cost": 100.0,
                                         "exposure_group": "OLD"}],
                 settings={"cash_floor_pct": 5.0, "max_position_pct": 100.0})
    orders = [{"symbol": "OLD", "side": "sell", "shares": 20.0, "conviction": 90},
              {"symbol": "GOOGL", "side": "buy", "dollars": 1000.0, "conviction": 62}]
    _, _, skipped = _run(blob, orders, {"OLD": 100.0, "GOOGL": _GOOGL})
    why = [s for s in skipped if s["symbol"] == "GOOGL"][0]["skip_reason"]
    assert "T+1" in why


def test_a_negative_quote_cannot_book_a_buy():
    """A negative fill makes the round-up guard trivially true and turns `cash -= cost` into a
    deposit — the account would gain both a share and money."""
    blob = _blob(cash=10_000.0)
    after, filled, skipped = _run(
        blob, [{"symbol": "GOOGL", "side": "buy", "dollars": 346.0, "conviction": 62}],
        {"GOOGL": -346.22})
    assert not filled
    assert after["cash"] == 10_000.0
    assert "no fresh price" in skipped[0]["skip_reason"]


def test_crypto_blocked_by_a_real_limit_names_that_limit():
    """Crypto fills fractionally, so a zero-share crypto buy is nearly always a limit, not the size."""
    blob = _blob(cash=10_000.0, settings={"allow_crypto": True, "max_turnover_pct": 0.0001})
    why = _skip_reason(blob, [{"symbol": "BTC-USD", "side": "buy", "dollars": 5000.0,
                               "conviction": 62}], {"BTC-USD": 90_000.0})
    assert "turnover" in why


def test_crypto_is_untouched_because_it_fills_fractionally():
    """There is no last share to rescue when sixth-decimal fractions fill."""
    blob = _blob(cash=10000.0, settings={"allow_crypto": True})
    row = _fill_row(blob, [{"symbol": "BTC-USD", "side": "buy", "dollars": 346.0, "conviction": 62}],
                    {"BTC-USD": 90000.0})
    assert 0 < row["shares"] < 1
    assert row["size_note"] is None


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
