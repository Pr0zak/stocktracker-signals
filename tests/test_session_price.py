"""Fills must be priced at the session actually in progress.

`regularMarketPrice` freezes at the 4pm close, so using it to fill an extended-hours order books a
trade at a price nobody could have got. Measured 2026-08-03: BLZE closed at 15.59 and traded to 18.25
after hours — a 17% gap the ledger would have banked as a real gain.
"""
from __future__ import annotations

from app.market_now import session_price


def _row(state, price=15.59, post=None, pre=None):
    return {"state": state, "price": price, "post_price": post, "pre_price": pre}


# ------------------------------------------------------------------ the case that motivated this

def test_after_hours_fill_uses_the_after_hours_price():
    assert session_price(_row("POST", price=15.59, post=18.25)) == 18.25


def test_the_gap_is_not_small():
    """Sanity on the magnitude — this is why it matters rather than a rounding nicety."""
    row = _row("POST", price=15.59, post=18.25)
    assert (session_price(row) / row["price"] - 1) > 0.15


def test_postpost_still_prefers_the_last_extended_print():
    """After 20:00 ET the last after-hours trade is still the most recent real one."""
    assert session_price(_row("POSTPOST", price=15.59, post=18.25)) == 18.25


def test_premarket_uses_the_premarket_price():
    assert session_price(_row("PRE", price=15.59, pre=16.10)) == 16.10


# ------------------------------------------------------------------ the ordinary cases

def test_regular_session_uses_the_regular_price():
    assert session_price(_row("REGULAR", price=15.59, post=18.25)) == 15.59


def test_closed_uses_the_regular_price():
    assert session_price(_row("CLOSED", price=15.59)) == 15.59


# ------------------------------------------------------------------ degrading safely

def test_missing_extended_price_falls_back_to_regular():
    """The common case: most names have no extended print, and the close is the right mark there."""
    assert session_price(_row("POST", price=15.59, post=None)) == 15.59


def test_a_zero_extended_price_is_not_used():
    """0.0 is not a price. Marking a holding at zero is the fabrication `mark_price` exists to stop."""
    assert session_price(_row("POST", price=15.59, post=0.0)) == 15.59


def test_absent_state_uses_the_regular_price():
    assert session_price({"price": 15.59, "post_price": 18.25}) == 15.59


def test_state_is_case_insensitive():
    assert session_price(_row("post", price=15.59, post=18.25)) == 18.25


def test_an_empty_row_is_none_not_zero():
    """An unpriceable symbol must stay None so mark_price falls back rather than booking a zero."""
    assert session_price({}) is None
