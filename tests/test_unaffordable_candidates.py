"""A candidate the account can never buy should never occupy a candidate slot.

One BRK-A share is around $700,000. The sandbox book is $10,800. The model cannot see that: it reads
a well-run company with real technicals, proposes it, and the per-group cap refuses the order --
then the same thing happens tomorrow, because nothing about the refusal changes what the screen
returns. It is not a bad idea that lost on merit; it is an idea this account is structurally
incapable of acting on, and it costs a slot and a line of trade-log noise every day it survives.

The ceiling is the POSITION CAP, not the cash balance. Cash is today's constraint and moves around;
the cap is the most this account may ever put into one exposure group, so a share priced above it can
never fill however the book grows.
"""
from __future__ import annotations

from app.sandbox_job import unaffordable

PRICES = {
    "BRK-A": 712_000.0,
    "BRK-B": 505.0,
    "AAPL": 231.4,
    "NVDA": 178.0,
    "BTC-USD": 96_204.0,
    "NOQUOTE": None,
}


def _price_of(sym: str):
    return PRICES.get(sym.upper())


def _call(syms, equity=10_800.0, cap=25.0):
    return unaffordable(syms, price_of=_price_of, equity=equity, max_position_pct=cap)


def test_a_share_priced_above_the_position_cap_is_dropped():
    # 25% of $10,800 = $2,700. One BRK-A share is 260x that and can never fill.
    assert _call(["AAPL", "BRK-A", "NVDA"]) == ["BRK-A"]


def test_the_ceiling_is_the_cap_not_the_cash_balance():
    # BRK-B at $505 clears the $2,700 cap and stays, even though it is a large slice of a small book.
    assert _call(["BRK-B"]) == []
    # The same ticker becomes affordable purely by the book growing, because the ceiling scales with
    # equity. At $1m the cap is $250k -- under one $712k share, so still dropped.
    assert _call(["BRK-A"], equity=1_000_000.0) == ["BRK-A"]
    # At $10m the cap is $2.5m, which clears one share, so it is a legitimate candidate again.
    assert _call(["BRK-A"], equity=10_000_000.0) == []


def test_crypto_is_never_dropped_because_it_fills_fractionally():
    # A whole bitcoin costs far more than the cap, but there is no minimum lot -- the order simply
    # buys a fraction. Filtering it would remove a perfectly fillable position.
    assert _call(["BTC-USD"]) == []


def test_an_unpriced_candidate_is_kept():
    # Absent is not the same as expensive. Dropping unpriced names would quietly shrink the universe
    # every time a quote failed, and a quote failure says nothing about affordability.
    assert _call(["NOQUOTE"]) == []


def test_a_zero_or_missing_cap_disables_the_filter_rather_than_dropping_everything():
    # A cap of 0 means "no per-group limit configured", not "nothing may be bought". Reading it the
    # other way would empty the entire candidate list on a misconfiguration.
    assert _call(["BRK-A", "AAPL"], cap=0.0) == []
    assert _call(["BRK-A", "AAPL"], equity=0.0) == []
