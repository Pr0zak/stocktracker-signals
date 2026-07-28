"""The rebalance plan reaches a broker ticket. Its numbers must be checked, not trusted.

Before this, `plan.model_dump()` went straight out: nothing verified that a sell was of shares the
user holds, that the buys were affordable, that `dollars` agreed with shares x price, or that the
moves produced the `resulting_top_weight_pct` the plan claimed.
"""
import pytest
from app.rebalance_check import validate_plan


def book(**kw):
    return {"positions": [
        {"symbol": "AAPL", "shares": 10.0, "price": 100.0, "value": 1000.0},
        {"symbol": "NVDA", "shares": 5.0, "price": 200.0, "value": 1000.0},
    ], **kw}


def test_a_sell_larger_than_the_position_is_capped_not_passed_through():
    # The one that reaches a broker as a SHORT.
    moves = [{"symbol": "AAPL", "action": "sell", "shares": 40.0, "dollars": 4000.0, "reason": "trim"}]
    kept, derived, warns = validate_plan(book(), moves, cash=0.0, max_position_pct=25.0)
    assert kept[0]["shares"] == 10.0
    assert kept[0]["dollars"] == 1000.0
    assert any("only 10" in w for w in warns)


def test_buys_cannot_spend_more_than_cash_plus_proceeds():
    moves = [
        {"symbol": "AAPL", "action": "sell", "shares": 5.0, "dollars": 500.0, "reason": "trim"},
        {"symbol": "NVDA", "action": "buy", "shares": 25.0, "dollars": 5000.0, "reason": "add"},
    ]
    kept, derived, warns = validate_plan(book(), moves, cash=100.0, max_position_pct=25.0)
    buy = next(m for m in kept if m["action"] == "buy")
    assert buy["dollars"] == 600.0            # 100 cash + 500 proceeds
    assert buy["shares"] == pytest.approx(3.0)
    assert any("more than the cash" in w for w in warns)


def test_dollars_that_contradict_shares_times_price_are_corrected():
    moves = [{"symbol": "NVDA", "action": "sell", "shares": 2.0, "dollars": 9999.0, "reason": "trim"}]
    kept, derived, warns = validate_plan(book(), moves, cash=0.0, max_position_pct=25.0)
    assert kept[0]["dollars"] == 400.0        # 2 x $200
    assert any("corrected" in w for w in warns)


def test_a_symbol_the_user_does_not_hold_is_dropped():
    moves = [{"symbol": "TSLA", "action": "buy", "shares": 5.0, "dollars": 1000.0, "reason": "new"}]
    kept, derived, warns = validate_plan(book(), moves, cash=5000.0, max_position_pct=25.0)
    assert kept == []
    assert any("not a holding" in w for w in warns)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -3.0, 0.0, "twelve", None])
def test_nonsense_share_counts_never_reach_the_user(bad):
    moves = [{"symbol": "AAPL", "action": "sell", "shares": bad, "dollars": 100.0, "reason": "x"}]
    kept, derived, warns = validate_plan(book(), moves, cash=0.0, max_position_pct=25.0)
    assert kept == [] and warns


def test_the_after_state_is_recomputed_not_taken_on_faith():
    # Model claims are irrelevant here — validate_plan derives these from the kept moves.
    moves = [{"symbol": "AAPL", "action": "sell", "shares": 5.0, "dollars": 500.0, "reason": "trim"}]
    kept, derived, warns = validate_plan(book(), moves, cash=0.0, max_position_pct=25.0)
    # after: AAPL 500, NVDA 1000, cash 500 -> total 2000, top = NVDA 50.0%
    assert derived["cash_after"] == 500.0
    assert derived["resulting_top_weight_pct"] == 50.0


def test_an_unweightable_book_reports_no_top_weight():
    moves = [{"symbol": "AAPL", "action": "hold", "shares": 0.0, "dollars": 0.0, "reason": "keep"}]
    kept, derived, warns = validate_plan(book(unvalued=["VXUS"]), moves, cash=0.0, max_position_pct=25.0)
    assert derived["resulting_top_weight_pct"] is None
    assert derived["weights_computable"] is False


def test_a_clean_plan_passes_through_unchanged_and_silent():
    # max_position_pct=60 is ACHIEVABLE on this two-position book. At 25 it is not — no plan over two
    # holdings can put the top weight under 25% — so the target-miss warning would correctly fire and
    # this would not be testing the silent path at all.
    moves = [
        {"symbol": "AAPL", "action": "sell", "shares": 2.0, "dollars": 200.0, "reason": "trim"},
        {"symbol": "NVDA", "action": "buy", "shares": 1.0, "dollars": 200.0, "reason": "add"},
    ]
    kept, derived, warns = validate_plan(book(), moves, cash=0.0, max_position_pct=60.0)
    assert warns == []
    assert [m["shares"] for m in kept] == [2.0, 1.0]


def test_one_symbol_gets_one_instruction():
    # A live plan returned both "sell AAPL 4" and "hold AAPL" — two rows saying opposite things
    # about the same position.
    moves = [
        {"symbol": "AAPL", "action": "sell", "shares": 4.0, "dollars": 400.0, "reason": "trim"},
        {"symbol": "AAPL", "action": "hold", "shares": 0.0, "dollars": 0.0, "reason": "keep"},
    ]
    kept, derived, warns = validate_plan(book(), moves, cash=0.0, max_position_pct=25.0)
    assert [m["action"] for m in kept] == ["sell"]
    assert any("redundant hold" in w for w in warns)


def test_a_plan_that_both_buys_and_sells_one_symbol_is_dropped():
    moves = [
        {"symbol": "AAPL", "action": "sell", "shares": 2.0, "dollars": 200.0, "reason": "trim"},
        {"symbol": "AAPL", "action": "buy", "shares": 2.0, "dollars": 200.0, "reason": "add"},
    ]
    kept, derived, warns = validate_plan(book(), moves, cash=1000.0, max_position_pct=25.0)
    assert not any(m["symbol"] == "AAPL" for m in kept)
    assert any("both bought and sold" in w for w in warns)


def test_a_plan_that_misses_its_own_target_says_so():
    # Two equal $1000 positions, no cash: doing nothing leaves the top weight at 50%, not 25%.
    moves = [{"symbol": "AAPL", "action": "hold", "shares": 0.0, "dollars": 0.0, "reason": "keep"}]
    kept, derived, warns = validate_plan(book(), moves, cash=0.0, max_position_pct=25.0)
    assert derived["resulting_top_weight_pct"] == 50.0
    assert any("does not fully rebalance" in w for w in warns)
