"""The market screen gets a reserved slice of the candidate budget.

Two channels feed the daily tick's candidate pool: the user's watchlist, and the live market screen.
They compete for one budget, and whichever one is truncated last is the one that survives.

It used to be a flat slice over a watchlist-first list, which meant the screen got the leftovers.
That degrades in the worst possible direction — every symbol added to the watchlist takes another
slot off the screen, so the only channel that can surface an unfamiliar name shrinks precisely as the
user curates more. Measured 2026-08-14: a 25-name watchlist filled 20 of 24 slots and 4 of the 8
screened names were fetched, ranked and then thrown away before the model saw one.

These tests pin the reservation, and pin the reporting of anything trimmed — because a truncated list
and a short list are indistinguishable downstream, and "the model passed on it" and "the model never
saw it" leave the same trace in the ledger.
"""
from __future__ import annotations

from app.sandbox_job import select_candidates


def _group(sym: str) -> str:
    return {"IBIT": "BTC", "FBTC": "BTC", "FETH": "ETH"}.get(sym.upper(), sym.upper())


def _select(watchlist, discovered, **over):
    kw = dict(
        watchlist=watchlist, discovered=discovered, held=[], exclusions=[],
        allow_crypto=False, allow_crypto_etf=True, group_of=_group,
        max_candidates=34, max_discovered=12,
    )
    kw.update(over)
    return select_candidates(**kw)


def test_a_long_watchlist_cannot_crowd_out_the_market_screen():
    # The regression this exists for. 30 watchlist names against a 34 budget: under the old
    # watchlist-first slice the screen would have got 4 of its 12. It must get all 12.
    watch = [f"W{i}" for i in range(30)]
    disc = [f"D{i}" for i in range(12)]
    picked, dropped = _select(watch, disc)
    assert len(picked) == 34
    assert [s for s in picked if s.startswith("D")] == disc      # every screened name survives
    assert len(dropped) == 8                                     # the watchlist yields instead
    assert dropped == watch[22:]


def test_the_watchlist_gets_the_whole_budget_when_the_screeners_are_down():
    # discover() returns [] on failure. The reservation must not hold slots open for names that do
    # not exist — that would shrink the pool on exactly the day it is already degraded.
    watch = [f"W{i}" for i in range(40)]
    picked, dropped = _select(watch, [])
    assert len(picked) == 34
    assert picked == watch[:34]
    assert dropped == watch[34:]


def test_nothing_is_trimmed_when_everything_fits():
    picked, dropped = _select(["AAPL", "MSFT"], ["NVDA"])
    assert picked == ["AAPL", "MSFT", "NVDA"]
    assert dropped == []


def test_trimmed_symbols_are_returned_not_silently_discarded():
    watch = [f"W{i}" for i in range(6)]
    picked, dropped = _select(watch, ["D0", "D1"], max_candidates=5, max_discovered=2)
    assert picked == ["W0", "W1", "W2", "D0", "D1"]
    # The caller needs the names, not just a count, to say anything useful about what was skipped.
    assert dropped == ["W3", "W4", "W5"]


def test_held_names_and_exclusions_never_reach_the_pool():
    picked, _ = _select(["AAPL", "MSFT", "GME"], ["NVDA", "TSLA"],
                        held=["AAPL"], exclusions=["gme", "TSLA"])
    assert picked == ["MSFT", "NVDA"]


def test_vehicle_filters_apply_to_both_channels():
    # Direct spot crypto off: -USD names go, from the watchlist AND the screen.
    picked, _ = _select(["AAPL", "BTC-USD"], ["ETH-USD", "NVDA"], allow_crypto=False)
    assert picked == ["AAPL", "NVDA"]
    # Crypto ETFs off: the ETF goes but the direct spot name is governed by the other flag.
    picked, _ = _select(["IBIT", "AAPL"], ["FETH"], allow_crypto=True, allow_crypto_etf=False)
    assert picked == ["AAPL"]


def test_a_symbol_on_both_lists_is_not_counted_twice():
    # discover() already excludes the watchlist, but the dedupe must not depend on that — a duplicate
    # would otherwise consume two slots and be sent to the model twice.
    picked, _ = _select(["AAPL", "NVDA"], ["NVDA", "AMD"])
    assert picked == ["AAPL", "NVDA", "AMD"]


def test_the_screen_slice_is_capped_even_when_the_budget_is_empty():
    # More discovered names than the reservation: it takes its slice and no more, so the watchlist is
    # never starved by an unusually generous screener run.
    picked, dropped = _select(["W0", "W1"], [f"D{i}" for i in range(20)],
                              max_candidates=10, max_discovered=4)
    assert picked == ["W0", "W1", "D0", "D1", "D2", "D3"]
    assert dropped == []
