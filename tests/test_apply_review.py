"""The second-opinion layer's effect on the order list.

`validate_and_fill` is the mechanical authority and checks arithmetic: caps, cash, oversell,
turnover. Both of this account's worst decisions passed it cleanly. On 2026-08-03 it sold IBIT on a
three-month momentum read while bitcoin sat below its 200-week line; on 2026-08-06 it sold all 3 SPY
to buy VTI on expense-ratio grounds, realising a short-term gain to save about $1.50 a year, and the
replacement buy was then cap-blocked so 39.9% cash became 58.3%. Two prompt rules forbade the second
explicitly. Nothing between the model and the ledger was reading the argument.

This function is the deterministic half of that layer: given a verdict, decide what survives. The
LLM half is judgement and cannot be unit-tested; this part can, and it is the part that decides
whether real orders disappear.
"""
from __future__ import annotations

from app.sandbox_job import apply_review


def _orders(*symbols):
    return [{"symbol": s, "side": "buy", "shares": 1.0, "dollars": 100.0,
             "conviction": 70, "reason": "t"} for s in symbols]


def test_named_symbols_are_dropped_and_the_rest_proceed():
    # The proportionate case, and the one the reviewer is told to prefer: three orders sound, one
    # not. Rejecting the batch over one bad order costs three good trades.
    kept, dropped = apply_review(
        _orders("VTI", "SPY", "AMZN"),
        {"approve": True, "drop_symbols": ["SPY"]})
    assert [o["symbol"] for o in kept] == ["VTI", "AMZN"]
    assert dropped == ["SPY"]


def test_a_blanket_rejection_drops_everything_and_names_it():
    # Every symbol is reported, not just a count -- the warning and the trade log have to say what
    # was lost, or a rejected tick is indistinguishable from a quiet one.
    kept, dropped = apply_review(_orders("VTI", "SPY"), {"approve": False})
    assert kept == []
    assert dropped == ["VTI", "SPY"]


def test_approve_false_wins_over_named_drops():
    # A reviewer that rejects the decision AND names symbols meant the stronger thing.
    kept, dropped = apply_review(
        _orders("VTI", "SPY", "AMZN"),
        {"approve": False, "drop_symbols": ["SPY"]})
    assert kept == []
    assert dropped == ["VTI", "SPY", "AMZN"]


def test_named_drops_are_honoured_even_when_approve_was_left_true():
    # The likely malformed verdict: specific findings, forgotten flag. Taken at its narrower word.
    # Reading approve=true as overriding the drops would discard the reviewer's only real finding.
    kept, dropped = apply_review(
        _orders("VTI", "SPY"), {"approve": True, "drop_symbols": ["SPY"]})
    assert [o["symbol"] for o in kept] == ["VTI"]
    assert dropped == ["SPY"]


def test_an_approving_verdict_changes_nothing():
    orders = _orders("VTI", "SPY")
    kept, dropped = apply_review(orders, {"approve": True, "drop_symbols": []})
    assert kept == orders
    assert dropped == []


def test_the_reviewer_cannot_add_an_order():
    # It is a check, not a second trader. A verdict naming a symbol that was never proposed must not
    # conjure one -- and must not crash trying to drop it either.
    kept, dropped = apply_review(
        _orders("VTI"), {"approve": True, "drop_symbols": ["NVDA", "TSLA"]})
    assert [o["symbol"] for o in kept] == ["VTI"]
    assert dropped == []


def test_symbol_matching_is_case_insensitive():
    kept, dropped = apply_review(_orders("VTI"), {"approve": True, "drop_symbols": ["vti"]})
    assert kept == []
    assert dropped == ["VTI"]


def test_an_empty_order_list_is_left_alone():
    # A decision to hold has no argument to check. The tick skips the review call entirely in this
    # case; this pins the function against a caller that does not.
    assert apply_review([], {"approve": False}) == ([], [])


def test_a_verdict_missing_its_fields_defaults_to_approving():
    # A malformed or empty verdict must not silently cancel the day's trading. The review is an
    # ADDITIONAL check: its failure returns the account to the behaviour it had without it.
    orders = _orders("VTI")
    assert apply_review(orders, {}) == (orders, [])
