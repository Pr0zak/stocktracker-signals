"""Per-bar highs/lows must live on the SAME price basis as the closes beside them.

`Series.closes` has always been the split/dividend-ADJUSTED series (`_adjusted_closes`), because
raw closes make a 10:1 split look like a -90% day and wreck every moving average. Yahoo's
`indicators.quote[0]` high/low/open arrays, however, are RAW. Adding highs/lows for the swing
metrics (ADR20, ATR14, CLV, 52-week distance) therefore walked straight into a basis mismatch: a
pre-split NVDA bar would have carried a ~$1,200 high against a ~$120 adjusted close, and
`(high - low) / close` would have read about 10x too wide for every bar older than the split. ADR20
would then rank a placid mega-cap as one of the most volatile names in the universe, and CLV would
be computed from a bar whose "high" sits an order of magnitude above its own close.

The fix is per-bar rescaling by that bar's own adjusted/raw close ratio — exactly the cumulative
factor Yahoo already applied to the close. The split test below is the one that matters: it asserts
the derived range stays stable ACROSS the split boundary, which is precisely what the naive version
gets wrong.

The second half of this file pins the new `fallback=False` switch. The bulk swing scan sweeps
3,000+ symbols, and letting each Yahoo failure fall through to Webull's unofficial search+chart
endpoints costs ~57s per name for warrants a swing scan does not want. `fallback` defaults True so
the ~35 existing call sites keep the rescue they have today.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from app import market, webull


def _payload(bars: list[dict], *, adjclose: bool = True) -> dict:
    """A Yahoo chart response built from [{ts, o, h, l, c, adj, v}] rows.

    Mirrors the real shape closely enough to exercise `_adjusted_closes`: raw OHLC under
    `indicators.quote[0]`, the adjusted series under `indicators.adjclose[0]`.
    """
    quote = {
        "open": [b.get("o") for b in bars],
        "high": [b.get("h") for b in bars],
        "low": [b.get("l") for b in bars],
        "close": [b.get("c") for b in bars],
        "volume": [b.get("v", 1_000_000) for b in bars],
    }
    indicators: dict = {"quote": [quote]}
    if adjclose:
        indicators["adjclose"] = [{"adjclose": [b.get("adj", b.get("c")) for b in bars]}]
    return {
        "chart": {
            "error": None,
            "result": [{
                "meta": {"currency": "USD", "fiftyTwoWeekHigh": 120.0, "fiftyTwoWeekLow": 40.0},
                "timestamp": [b["ts"] for b in bars],
                "indicators": indicators,
            }],
        }
    }


def _fetch(handler, symbol: str = "TEST", **kw) -> market.Series:
    """Run fetch_series against a mocked transport (the repo tests async with asyncio.run rather
    than taking a pytest-asyncio dependency)."""
    async def _run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await market.fetch_series(c, symbol, **kw)
    return asyncio.run(_run())


def _ok(payload):
    return lambda request: httpx.Response(200, json=payload)


# --------------------------------------------------------------------- alignment with closes

def test_highs_and_lows_are_index_aligned_with_closes():
    """Anything reading `series.highs[i]` alongside `series.closes[i]` needs one shared index."""
    bars = [
        {"ts": 1_780_000_000 + i * 86400, "o": 100.0 + i, "h": 102.0 + i, "l": 98.0 + i,
         "c": 100.5 + i, "v": 5_000_000}
        for i in range(30)
    ]
    s = _fetch(_ok(_payload(bars)))
    assert len(s.highs) == len(s.closes) == len(s.lows) == 30
    assert s.highs[0] == pytest.approx(102.0)
    assert s.lows[-1] == pytest.approx(98.0 + 29)


def test_bars_yahoo_nulls_out_are_skipped_from_highs_too_so_the_index_never_drifts():
    """A null adjusted close makes the loop `continue`. If highs/lows were appended before that
    check they would end up one bar longer and silently off-by-one against closes."""
    bars = [
        {"ts": 1_780_000_000, "o": 10.0, "h": 11.0, "l": 9.0, "c": 10.5},
        {"ts": 1_780_086_400, "o": None, "h": None, "l": None, "c": None, "adj": None},
        {"ts": 1_780_172_800, "o": 12.0, "h": 13.0, "l": 11.0, "c": 12.5},
    ]
    s = _fetch(_ok(_payload(bars)))
    assert len(s.closes) == 2
    assert len(s.highs) == len(s.lows) == 2
    assert s.highs == [pytest.approx(11.0), pytest.approx(13.0)]


# --------------------------------------------------------------------- THE split test

def _split_series() -> market.Series:
    """Six bars around a synthetic 2:1 split. The first three are pre-split: raw prices are double
    what the adjusted series shows (close 100 raw -> 50 adjusted). The last three are post-split and
    need no adjustment. Every bar has the same TRUE 4%-of-close range, so a correct implementation
    reports one constant (high - low) / close on both sides of the boundary."""
    pre = [{"ts": 1_780_000_000 + i * 86400, "o": 100.0, "h": 102.0, "l": 98.0, "c": 100.0,
            "adj": 50.0} for i in range(3)]
    post = [{"ts": 1_780_000_000 + (3 + i) * 86400, "o": 50.0, "h": 51.0, "l": 49.0, "c": 50.0,
             "adj": 50.0} for i in range(3)]
    return _fetch(_ok(_payload(pre + post)))


def test_a_split_does_not_widen_the_pre_split_range():
    """THE assertion. Raw high 102 next to adjusted close 50 reads as an 8% daily range; the true
    range is 4%. Every pre-split bar must match its post-split twin."""
    s = _split_series()
    ranges = [(h - lo) / c for h, lo, c in zip(s.highs, s.lows, s.closes)]
    assert len(ranges) == 6
    for r in ranges:
        assert r == pytest.approx(0.04, abs=1e-9), ranges


def test_the_pre_split_high_is_scaled_onto_the_adjusted_basis():
    """Spelled out on the raw numbers, so a regression names itself: 102 raw at a 0.5 factor is 51,
    not 102."""
    s = _split_series()
    assert s.closes[0] == pytest.approx(50.0)
    assert s.highs[0] == pytest.approx(51.0)
    assert s.lows[0] == pytest.approx(49.0)
    # ...and the post-split bars are untouched, because their factor is exactly 1.0.
    assert s.highs[-1] == pytest.approx(51.0)
    assert s.lows[-1] == pytest.approx(49.0)


def test_a_high_never_ends_up_below_its_own_close():
    """The cheap invariant that the un-scaled version violates in the opposite direction: with a
    REVERSE split the raw high sits far BELOW the adjusted close, which would drive CLV out of its
    -1..+1 range."""
    pre = [{"ts": 1_780_000_000 + i * 86400, "o": 10.0, "h": 10.2, "l": 9.8, "c": 10.0,
            "adj": 100.0} for i in range(3)]
    post = [{"ts": 1_780_000_000 + (3 + i) * 86400, "o": 100.0, "h": 102.0, "l": 98.0, "c": 100.0,
             "adj": 100.0} for i in range(3)]
    s = _fetch(_ok(_payload(pre + post)))
    for h, lo, c in zip(s.highs, s.lows, s.closes):
        assert lo <= c <= h, (lo, c, h)


def test_symbols_with_no_adjclose_array_keep_their_raw_highs():
    """Indices and crypto carry no adjclose at all — raw already IS adjusted there, so the scaling
    factor must come out 1.0 rather than mangling the bars."""
    bars = [{"ts": 1_780_000_000 + i * 86400, "o": 300.0, "h": 310.0, "l": 295.0, "c": 305.0}
            for i in range(5)]
    s = _fetch(_ok(_payload(bars, adjclose=False)))
    assert s.highs == [pytest.approx(310.0)] * 5
    assert s.lows == [pytest.approx(295.0)] * 5


# --------------------------------------------------------------------- absent is not zero

def test_a_null_high_or_low_is_none_not_zero():
    """House rule, and a real correctness trap: a 0.0 high would make (high - low) / close a large
    negative number and quietly poison ADR20's mean instead of being counted as unmeasurable."""
    bars = [
        {"ts": 1_780_000_000, "o": 10.0, "h": 11.0, "l": 9.0, "c": 10.5},
        {"ts": 1_780_086_400, "o": 10.5, "h": None, "l": None, "c": 10.6},
    ]
    s = _fetch(_ok(_payload(bars)))
    assert s.highs[1] is None and s.lows[1] is None
    assert s.highs[1] != 0.0 and s.lows[1] != 0.0


def test_a_missing_raw_close_leaves_the_bar_unscaled_rather_than_dividing_by_zero():
    """Yahoo occasionally nulls a raw close while still publishing adjclose for that bar. There is
    no factor to compute, so the bar is stored as-is — never dropped, never a ZeroDivisionError."""
    bars = [
        {"ts": 1_780_000_000, "o": 10.0, "h": 11.0, "l": 9.0, "c": None, "adj": 10.5},
        {"ts": 1_780_086_400, "o": 10.5, "h": 11.5, "l": 10.0, "c": 10.6, "adj": 10.6},
    ]
    s = _fetch(_ok(_payload(bars)))
    assert len(s.closes) == 2
    assert s.highs[0] == pytest.approx(11.0)


# --------------------------------------------------------------------- the fallback switch

def _boom(request):
    return httpx.Response(500, text="yahoo is down")


def test_fallback_false_reraises_instead_of_reaching_for_webull(monkeypatch):
    """The bulk scan's whole reason for existing: ~60 failures out of 3,000 symbols must not each
    spend ~27s more inside an unofficial endpoint after already burning the Yahoo timeouts."""
    called: list[str] = []

    async def _never(client, symbol):
        called.append(symbol)
        raise AssertionError("the Webull fallback must not run when fallback=False")

    monkeypatch.setattr(market, "_webull_series", _never)
    with pytest.raises(RuntimeError):
        _fetch(_boom, fallback=False)
    assert called == []


def test_fallback_true_still_reaches_webull(monkeypatch):
    """The default is unchanged, because ~35 existing call sites depend on the warrant/OTC rescue."""
    called: list[str] = []

    async def _rescue(client, symbol):
        called.append(symbol)
        return market.Series(
            symbol=symbol.upper(), closes=[1.0], opens=[1.0], volumes=[1.0], dates=["20260101"],
            fifty_two_high=1.0, fifty_two_low=1.0, currency="USD", source="webull",
        )

    monkeypatch.setattr(market, "_webull_series", _rescue)
    s = _fetch(_boom, symbol="GME.WS")
    assert called == ["GME.WS"]
    assert s.source == "webull"


def test_the_default_call_signature_is_untouched(monkeypatch):
    """`fallback` is keyword-only on purpose: `fetch_series(client, sym, "2y")` positional callers
    must not silently start passing a range string into the new flag."""
    async def _rescue(client, symbol):
        return market.Series(
            symbol=symbol.upper(), closes=[1.0], opens=[1.0], volumes=[1.0], dates=["20260101"],
            fifty_two_high=1.0, fifty_two_low=1.0, currency="USD", source="webull",
        )

    monkeypatch.setattr(market, "_webull_series", _rescue)
    assert _fetch(_boom).source == "webull"  # no fallback= given -> rescue still happens


# --------------------------------------------------------------------- the webull path carries h/l

def test_the_webull_fallback_lifts_its_own_highs_and_lows_through(monkeypatch):
    """Webull already parses h/l out of its bar CSV. A Series that reached us via the fallback would
    otherwise arrive with empty highs, and every swing metric would report "not measurable" for
    exactly the thin names where the range matters most."""
    bars = [
        {"t": 1_780_000_000_000, "o": 2.0, "c": 2.1, "h": 2.3, "l": 1.9, "v": 100_000.0},
        {"t": 1_780_086_400_000, "o": 2.1, "c": 2.2, "h": 2.4, "l": 2.05, "v": 120_000.0},
    ]

    async def _history(client, symbol, count=800):
        return bars

    # `_webull_series` imports `webull` lazily inside the function, so the patch target is the
    # module object itself rather than a name bound on `market`.
    monkeypatch.setattr(webull, "history", _history)
    s = _fetch(_boom, symbol="GME.WS")
    assert s.source == "webull"
    assert s.highs == [pytest.approx(2.3), pytest.approx(2.4)]
    assert s.lows == [pytest.approx(1.9), pytest.approx(2.05)]
    assert len(s.highs) == len(s.closes)
