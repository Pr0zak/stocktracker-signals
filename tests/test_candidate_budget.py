"""The watchlist and the market screen get separate candidate budgets.

Two channels feed the daily tick's candidate pool: the user's watchlist, and the live market screen.
Sharing one budget between them means one of the two always loses, and both ways of losing shipped:

1. A flat slice over a watchlist-first list starved the SCREEN. Every symbol added to the watchlist
   took another slot off it, so the only channel that can surface an unfamiliar name shrank precisely
   as the user curated more. Measured 2026-08-14: a 25-name watchlist filled 20 of 24 slots and 4 of
   the 8 screened names were fetched, ranked and thrown away before the model saw one.
2. Reserving the screen's slice fixed that and moved the loss onto the WATCHLIST. The same afternoon
   the list grew 25 -> 49 and 21 names -- every ticker added that day -- fell off the end. That is
   the worse failure: adding a ticker silently evicted another one the user had also chosen.

So neither channel is trimmed to fit the other. The watchlist is bounded only by a runaway ceiling,
the screen always has room of its own, and the total grows with the list.

These tests also pin the reporting of anything trimmed, because a truncated list and a short list are
indistinguishable downstream -- "the model passed on it" and "the model never saw it" leave the same
trace in the ledger.
"""
from __future__ import annotations

from app.sandbox_job import select_candidates


def _group(sym: str) -> str:
    return {"IBIT": "BTC", "FBTC": "BTC", "FETH": "ETH"}.get(sym.upper(), sym.upper())


def _select(watchlist, discovered, **over):
    kw = dict(
        watchlist=watchlist, discovered=discovered, held=[], exclusions=[],
        allow_crypto=False, allow_crypto_etf=True, group_of=_group,
        max_watchlist=60, max_discovered=12,
    )
    kw.update(over)
    return select_candidates(**kw)


def test_the_pool_grows_with_the_watchlist_instead_of_evicting_from_it():
    # Regression (2): 49 watchlist names must all survive, not the 22 a shared 34-slot budget left.
    watch = [f"W{i}" for i in range(49)]
    disc = [f"D{i}" for i in range(12)]
    picked, dropped = _select(watch, disc)
    assert dropped == []
    assert len(picked) == 61
    assert [s for s in picked if s.startswith("W")] == watch


def test_a_long_watchlist_cannot_crowd_out_the_market_screen():
    # Regression (1): the screen keeps all 12 no matter how long the watchlist gets.
    picked, _ = _select([f"W{i}" for i in range(58)], [f"D{i}" for i in range(12)])
    assert [s for s in picked if s.startswith("D")] == [f"D{i}" for i in range(12)]


def test_the_two_budgets_are_independent_in_both_directions():
    # A short watchlist does not hand its unused room to the screen, and vice versa. Each channel
    # takes what it has, up to its own ceiling -- there is no borrowing to reason about.
    picked, _ = _select(["AAPL"], [f"D{i}" for i in range(30)], max_watchlist=10, max_discovered=4)
    assert picked == ["AAPL", "D0", "D1", "D2", "D3"]
    picked, _ = _select([f"W{i}" for i in range(10)], [], max_watchlist=10, max_discovered=4)
    assert picked == [f"W{i}" for i in range(10)]


def test_the_watchlist_ceiling_still_holds_and_reports_what_it_cut():
    # The ceiling is a runaway guard, not a working limit -- but when it does bite, the names come
    # back to the caller. A count alone would not let the tick say which tickers went unseen.
    watch = [f"W{i}" for i in range(63)]
    picked, dropped = _select(watch, ["D0"])
    assert len(picked) == 61                       # 60 watchlist + 1 screened
    assert dropped == ["W60", "W61", "W62"]


def test_the_screeners_being_down_costs_the_watchlist_nothing():
    # discover() returns [] on failure. Separate budgets mean there is no slot to hold open.
    picked, dropped = _select([f"W{i}" for i in range(40)], [])
    assert picked == [f"W{i}" for i in range(40)]
    assert dropped == []


def test_held_names_and_exclusions_never_reach_the_pool():
    picked, _ = _select(["AAPL", "MSFT", "GME"], ["NVDA", "TSLA"],
                        held=["AAPL"], exclusions=["gme", "TSLA"])
    assert picked == ["MSFT", "NVDA"]


def test_vehicle_filters_apply_to_both_channels():
    # Direct spot crypto off: -USD names go, from the watchlist AND the screen.
    picked, _ = _select(["AAPL", "BTC-USD"], ["ETH-USD", "NVDA"], allow_crypto=False)
    assert picked == ["AAPL", "NVDA"]
    # Crypto ETFs off: the ETF goes but direct spot is governed by the other flag.
    picked, _ = _select(["IBIT", "AAPL"], ["FETH"], allow_crypto=True, allow_crypto_etf=False)
    assert picked == ["AAPL"]


def test_a_symbol_on_both_lists_is_not_counted_twice():
    # discover() already excludes the watchlist, but the dedupe must not depend on that -- a
    # duplicate would consume two slots and be sent to the model twice.
    picked, _ = _select(["AAPL", "NVDA"], ["NVDA", "AMD"])
    assert picked == ["AAPL", "NVDA", "AMD"]


def test_the_watchlist_keeps_its_stored_order():
    # The order is the user's own (drag-to-reorder in the app), and the model reads the list top
    # down. Re-sorting it here would quietly discard a preference the user expressed by hand.
    picked, _ = _select(["ZZZ", "AAA", "MMM"], [])
    assert picked == ["ZZZ", "AAA", "MMM"]
