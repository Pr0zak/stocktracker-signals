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


def _btc_book():
    # One exposure, two wrappers: each row looks like 33%, the real BTC exposure is 67%.
    return {"positions": [
        {"symbol": "IBIT", "shares": 10.0, "price": 100.0, "value": 1000.0, "exposure_group": "BTC"},
        {"symbol": "FBTC", "shares": 10.0, "price": 100.0, "value": 1000.0, "exposure_group": "BTC"},
        {"symbol": "AAPL", "shares": 10.0, "price": 100.0, "value": 1000.0, "exposure_group": "AAPL"},
    ]}


def test_the_cap_is_judged_on_combined_exposure_not_per_position():
    moves = [{"symbol": "AAPL", "action": "hold", "shares": 0.0, "dollars": 0.0, "reason": "keep"}]
    kept, derived, warns = validate_plan(_btc_book(), moves, cash=0.0, max_position_pct=25.0)
    # Per position the top is 33.3%; per exposure it is 66.7% — nearly 3x the cap.
    assert derived["resulting_top_weight_pct"] == pytest.approx(66.7, abs=0.1)
    assert derived["resulting_top_exposure"] == "BTC"
    assert any("does not fully rebalance" in w for w in warns)


def test_selling_one_wrapper_reduces_the_whole_exposure():
    moves = [{"symbol": "IBIT", "action": "sell", "shares": 10.0, "dollars": 1000.0, "reason": "trim"}]
    kept, derived, warns = validate_plan(_btc_book(), moves, cash=0.0, max_position_pct=40.0)
    # after: FBTC 1000 (BTC), AAPL 1000, cash 1000 -> BTC 33.3%
    assert derived["resulting_top_weight_pct"] == pytest.approx(33.3, abs=0.1)
    assert not any("does not fully rebalance" in w for w in warns)


# ---------------------------------------------------------------- review actions

from app.rebalance_check import validate_actions   # noqa: E402


def test_review_actions_naming_symbols_the_user_does_not_hold_are_dropped():
    acts = [{"symbol": "AAPL", "action": "trim", "reason": "heavy"},
            {"symbol": "TSLA", "action": "add", "reason": "nice setup"}]
    kept, warns = validate_actions(book(), acts)
    assert [a["symbol"] for a in kept] == ["AAPL"]
    assert any("not a holding" in w for w in warns)


def test_an_invented_action_string_is_dropped():
    kept, warns = validate_actions(book(), [{"symbol": "AAPL", "action": "YOLO", "reason": "x"}])
    assert kept == [] and any("unknown action" in w for w in warns)


def test_an_unpriced_holding_cannot_be_trimmed_or_added():
    # No price means no setup to judge; only hold/watch is honest.
    pf = {**book(), "unpriced": [{"symbol": "NVDA", "shares": 5.0, "value_at_cost": 600.0}]}
    kept, warns = validate_actions(pf, [{"symbol": "NVDA", "action": "trim", "reason": "heavy"}])
    assert kept[0]["action"] == "watch"
    assert any("downgraded" in w for w in warns)


def test_one_action_per_holding():
    acts = [{"symbol": "AAPL", "action": "trim", "reason": "a"},
            {"symbol": "AAPL", "action": "add", "reason": "b"}]
    kept, warns = validate_actions(book(), acts)
    assert len(kept) == 1 and any("second action" in w for w in warns)


# ---------------------------------------------------------------- ordering regressions
# All three reproduced against the original ordering (validate -> merge -> drop, never re-checked).

def test_two_sells_of_one_symbol_cannot_exceed_the_shares_held():
    """The merge used to run AFTER the cap and simply SUM the shares.

    Each 5-share sell passed the 6-share cap individually, then merged to 10 — an instruction to
    sell 4 shares that do not exist, typed straight into a broker as a short.
    """
    b = {"positions": [{"symbol": "AAPL", "shares": 6.0, "price": 100.0, "value": 600.0}]}
    moves = [{"symbol": "AAPL", "action": "sell", "shares": 5.0, "dollars": 500.0, "reason": "a"},
             {"symbol": "AAPL", "action": "sell", "shares": 5.0, "dollars": 500.0, "reason": "b"}]
    kept, derived, warns = validate_plan(b, moves, cash=0.0, max_position_pct=25.0)
    assert kept[0]["shares"] == 6.0, f"instructed a short: {kept}"
    assert kept[0]["dollars"] == 600.0
    assert any("only 6" in w for w in warns)


def test_a_buy_is_not_funded_by_proceeds_from_a_sell_that_was_dropped():
    """Affordability used to be computed BEFORE dedupe could drop the sell that funded it."""
    b = {"positions": [{"symbol": "AAPL", "shares": 100.0, "price": 100.0, "value": 10000.0},
                       {"symbol": "MSFT", "shares": 10.0, "price": 50.0, "value": 500.0}]}
    moves = [{"symbol": "AAPL", "action": "sell", "shares": 50.0, "dollars": 5000.0, "reason": "a"},
             {"symbol": "AAPL", "action": "buy", "shares": 10.0, "dollars": 1000.0, "reason": "b"},
             {"symbol": "MSFT", "action": "buy", "shares": 100.0, "dollars": 5000.0, "reason": "c"}]
    kept, derived, warns = validate_plan(b, moves, cash=0.0, max_position_pct=25.0)
    assert not any(m["action"] == "buy" for m in kept), f"unfunded buy survived: {kept}"
    assert derived["cash_after"] >= 0.0, "a plan cannot spend cash that does not exist"


@pytest.mark.parametrize("sell_act,buy_act", [("SELL", "Buy"), ("Sell", "BUY"), (" sell ", " buy ")])
def test_the_models_casing_cannot_skip_every_check(sell_act, buy_act):
    """action was lower-cased into a local for the whitelist but never on the dict, so every branch
    after it compared the raw string and matched nothing — no cap, no affordability, and _derive
    returned the BEFORE state as the after state."""
    b = {"positions": [{"symbol": "AAPL", "shares": 6.0, "price": 100.0, "value": 600.0},
                       {"symbol": "MSFT", "shares": 10.0, "price": 50.0, "value": 500.0}]}
    moves = [{"symbol": "AAPL", "action": sell_act, "shares": 6.0, "dollars": 600.0, "reason": "a"},
             {"symbol": "MSFT", "action": buy_act, "shares": 100.0, "dollars": 5000.0, "reason": "b"}]
    kept, derived, warns = validate_plan(b, moves, cash=0.0, max_position_pct=25.0)
    assert all(m["action"] in ("buy", "sell", "hold") for m in kept), kept
    # Selling all of AAPL must be reflected: it cannot still be the top exposure afterwards.
    assert derived["resulting_top_exposure"] != "AAPL", derived
    # And a $5,000 buy on $0 cash + $600 proceeds must be trimmed or dropped, not waved through.
    buys = [m for m in kept if m["action"] == "buy"]
    assert sum(float(m["dollars"]) for m in buys) <= 600.0 + 1e-6, buys


def test_unpriced_holdings_belong_in_the_weight_denominator():
    """They live in `unpriced`, not `positions`, so they were missing from the total entirely.

    That INFLATES every remaining weight — a true 50% position was reported as 100% — and fires a
    false "does not fully rebalance" warning off a book that is half invisible.
    """
    pf = {"positions": [{"symbol": "AAPL", "shares": 10, "price": 100.0, "value": 1000.0,
                         "exposure_group": "AAPL"}],
          "unpriced": [{"symbol": "VXUS", "shares": 10, "value_at_cost": 1000.0}]}
    moves = [{"symbol": "AAPL", "action": "hold", "shares": 0, "dollars": 0, "reason": "x"}]
    kept, derived, warns = validate_plan(pf, moves, cash=0.0, max_position_pct=25.0)
    assert derived["resulting_top_weight_pct"] == 50.0, derived
    assert derived["weights_approximate"] is True


def test_a_fully_priced_book_is_not_marked_approximate():
    kept, derived, warns = validate_plan(book(), [], cash=0.0, max_position_pct=60.0)
    assert derived.get("weights_approximate") is False


def test_an_action_on_an_unpriced_holding_is_downgraded_not_dropped():
    """Unpriced holdings live in `unpriced`, not `positions`, so membership of `held` alone dropped
    them as "not a holding in this book" — wrong (they ARE held) and it made the trim/add -> watch
    downgrade below it unreachable."""
    pf = {"positions": [{"symbol": "AAPL", "shares": 10, "price": 100.0, "value": 1000.0}],
          "unpriced": [{"symbol": "NVDA", "shares": 5, "value_at_cost": 600.0}]}
    kept, warns = validate_actions(pf, [{"symbol": "NVDA", "action": "trim", "reason": "heavy"}])
    assert [k["symbol"] for k in kept] == ["NVDA"]
    assert kept[0]["action"] == "watch", "no price means no setup to trim on"
    assert any("downgraded" in w for w in warns)
    assert not any("not a holding" in w for w in warns)


def test_a_symbol_that_really_is_absent_is_still_dropped():
    pf = {"positions": [{"symbol": "AAPL", "shares": 10, "price": 100.0, "value": 1000.0}]}
    kept, warns = validate_actions(pf, [{"symbol": "TSLA", "action": "add", "reason": "x"}])
    assert kept == [] and any("not a holding" in w for w in warns)
