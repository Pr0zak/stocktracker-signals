"""The market-cap floor on the sandbox's candidate screen.

The universe reaches the whole market now, and the screens that give it that breadth (actives,
gainers, growth) are also where the speculative end lives -- a single day's pool carried ONDS, IONQ,
BYND and AAOI. This is the dial between "everything above the junk floor" and "established companies
only".

The case worth pinning is the MISSING one. Yahoo does not always report marketCap, and a name with no
cap must not be read as a name with a small cap: absent is a fact about the response, not about the
company. Keeping unknowns means a raised floor leaks the occasional unreported name; dropping them
would silently discard real companies on a data gap, which is the worse trade for a universe this
setting exists to shape rather than to shrink.
"""
from __future__ import annotations

import asyncio

import httpx

from app import discover as disc_mod


def _quote(sym: str, cap: float | None, qt: str = "EQUITY", price: float = 100.0) -> dict:
    q = {"symbol": sym, "quoteType": qt, "regularMarketPrice": price}
    if cap is not None:
        q["marketCap"] = cap
    return q


def _run(quotes: list[dict], **kw) -> list[str]:
    """Drive discover() with a stubbed screener so no network is touched."""
    async def fake_screen(client, scr, count=15):
        return quotes

    orig = disc_mod._screen
    disc_mod._screen = fake_screen
    try:
        return asyncio.run(disc_mod.discover(
            httpx.AsyncClient(), exclude=set(), screens=("most_actives",), **kw))
    finally:
        disc_mod._screen = orig


def test_the_default_floor_is_two_billion():
    out = _run([_quote("BIG", 50e9), _quote("MID", 3e9), _quote("SMALL", 900e6)], cap=10)
    assert out == ["BIG", "MID"]


def test_raising_the_floor_drops_the_smaller_names():
    quotes = [_quote("MEGA", 500e9), _quote("BIG", 50e9), _quote("MID", 3e9)]
    assert _run(quotes, cap=10, min_market_cap=10e9) == ["MEGA", "BIG"]
    assert _run(quotes, cap=10, min_market_cap=100e9) == ["MEGA"]


def test_a_zero_floor_admits_everything_above_the_price_filter():
    # 0 is "no size floor", a legitimate choice -- not "admit nothing". The $5 price filter is
    # separate and still applies, because a sub-$5 quote is a liquidity problem rather than a size one.
    out = _run([_quote("MID", 3e9), _quote("TINY", 50e6), _quote("PENNY", 900e6, price=2.0)],
               cap=10, min_market_cap=0.0)
    assert out == ["MID", "TINY"]


def test_a_name_with_no_reported_cap_survives_any_floor():
    # Absent is not small. This is the leak the docstring describes, pinned deliberately so a future
    # change to "drop unknowns" has to argue with a test rather than slip through.
    out = _run([_quote("NOCAP", None), _quote("MID", 3e9)], cap=10, min_market_cap=100e9)
    assert out == ["NOCAP"]


def test_etfs_are_never_filtered_by_company_size():
    # An ETF reports AUM, not market cap. Filtering a broad index fund by company size is a category
    # error, and it would empty the sandbox's core shelf the moment the floor was raised.
    out = _run([_quote("VTI", None, qt="ETF"), _quote("SPY", 0.0, qt="ETF"), _quote("MID", 3e9)],
               cap=10, min_market_cap=100e9)
    assert out == ["VTI", "SPY"]


def test_a_strict_floor_returns_a_short_list_rather_than_the_fallback_universe():
    """A filter doing its job is not a failure.

    The fallback fired whenever fewer than five names survived, which conflated "the screener API is
    unreachable" with "your floor is strict". A high min_market_cap would therefore be silently
    replaced by a hardcoded mega-cap list that the floor had never been applied to — the setting
    would stop being honoured with nothing on screen to say so.
    """
    out = _run([_quote("MEGA", 500e9), _quote("MID", 3e9), _quote("SMALL", 900e6)],
               cap=10, min_market_cap=100e9)
    assert out == ["MEGA"]
    assert "AAPL" not in out          # i.e. FALLBACK did not quietly take over


def test_the_fallback_still_covers_an_unreachable_screener():
    # The case the fallback actually exists for: nothing came back at all.
    out = _run([], cap=6)
    assert out[:3] == ["AAPL", "MSFT", "GOOGL"]
