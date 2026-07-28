"""`_build_portfolio_snapshot` must never quietly misrepresent the book.

This is the shared structure behind BOTH the sandbox's weekly strategy review AND the user's real
`/portfolio/review` + `/portfolio/rebalance` — screens acted on with real money at Fidelity. It prices
each holding by fetching a series per symbol, so a transient upstream failure is routine, not exotic.

Two failure modes, neither previously covered by any test:
  1. ONE holding unpriceable -> `_price` returns None and the row is dropped, with no signal anywhere.
     `total_value` then omits that position, so every SURVIVING weight_pct and cash_pct is inflated.
     The analyst sets targets against a book that is missing a position and looks more cash-heavy and
     more concentrated than the account really is.
  2. ALL holdings unpriceable -> HTTPException(502), which the sandbox tick catches with a bare
     `except Exception` and replaces with `{"cash_pct": 100.0, "positions": []}` — a book asserting
     the account is entirely in cash while it holds shares. That note then persists for SEVEN DAYS.
"""
import asyncio
import pytest
from app import main as m


class _Series:
    closes = [100.0] * 60


def _fake_summarize(price):
    return lambda series, bench: {"price": price, "rsi14": 50.0, "pct_vs_sma50": 1.0}


def _run(coro):
    # asyncio.run, NOT get_event_loop(): the latter passes when this file runs alone and raises
    # "no current event loop" once any earlier test has closed the loop. A test that only passes in
    # isolation is not a test.
    return asyncio.run(coro)


@pytest.fixture
def priced(monkeypatch):
    """Price AAPL at $100; make any symbol in `broken` raise."""
    broken: set[str] = set()

    async def fake_fetch(http, sym, *a, **k):
        if sym.upper() in broken:
            raise RuntimeError(f"upstream down for {sym}")
        return _Series()

    monkeypatch.setattr(m, "_http", object())
    monkeypatch.setattr(m, "fetch_series", fake_fetch)
    monkeypatch.setattr(m, "summarize", lambda series, bench: {"price": 100.0, "rsi14": 50.0})
    return broken


def test_a_dropped_holding_does_not_silently_inflate_the_others(priced):
    # avg_cost == the live price here, so carrying the unpriceable holding at cost is EXACT and the
    # assertion can be exact. (When cost and price differ it is an estimate — see the next test.)
    priced.add("VXUS")
    holdings = [m.Holding(symbol="AAPL", shares=10, avg_cost=90.0),     # $1000 at $100
                m.Holding(symbol="VXUS", shares=10, avg_cost=100.0)]    # $1000, unpriceable
    book = _run(m._build_portfolio_snapshot(holdings, cash=1000.0))

    # Truth: 1000 + 1000 + 1000 cash = 3000 -> AAPL 33.3%, cash 33.3%.
    # Dropping VXUS made the denominator 2000 -> AAPL 50.0%, cash 50.0%.
    aapl = next(p for p in book["positions"] if p["symbol"] == "AAPL")
    assert aapl["weight_pct"] == pytest.approx(33.3, abs=0.2), (
        f"a dropped holding inflated AAPL's weight to {aapl['weight_pct']}% — the analyst sizes "
        f"against this")
    assert book["cash_pct"] == pytest.approx(33.3, abs=0.2), (
        f"a dropped holding made the account look {book['cash_pct']}% cash")


def test_an_unpriceable_holding_is_carried_at_cost_not_at_zero(priced):
    # Cost basis is a rough mark, but the alternative the code had was valuing it at ZERO, which is
    # never closer to the truth. Bounds the residual error rather than pretending it is gone.
    priced.add("VXUS")
    holdings = [m.Holding(symbol="AAPL", shares=10, avg_cost=90.0),     # $1000 at $100
                m.Holding(symbol="VXUS", shares=10, avg_cost=80.0)]     # really $1000, marked $800
    book = _run(m._build_portfolio_snapshot(holdings, cash=1000.0))
    assert book["total_value"] == 2800.0            # was 2000.0 when the holding was dropped
    assert book["unpriced"][0]["value_at_cost"] == 800.0
    aapl = next(p for p in book["positions"] if p["symbol"] == "AAPL")
    assert 33.3 <= aapl["weight_pct"] <= 36.0, "should land near truth (33.3%), never back at 50%"


def test_a_dropped_holding_is_reported_rather_than_vanishing(priced):
    priced.add("VXUS")
    holdings = [m.Holding(symbol="AAPL", shares=10, avg_cost=90.0),
                m.Holding(symbol="VXUS", shares=10, avg_cost=80.0)]
    book = _run(m._build_portfolio_snapshot(holdings, cash=1000.0))
    assert book.get("unpriced"), "a holding vanished from the book with no signal at all"
    assert [u["symbol"] for u in book["unpriced"]] == ["VXUS"]


def test_everything_unpriceable_still_says_so(priced):
    priced.update({"AAPL", "VXUS"})
    holdings = [m.Holding(symbol="AAPL", shares=10, avg_cost=90.0),
                m.Holding(symbol="VXUS", shares=10, avg_cost=80.0)]
    with pytest.raises(Exception):
        _run(m._build_portfolio_snapshot(holdings, cash=1000.0))


def test_the_happy_path_is_unchanged(priced):
    holdings = [m.Holding(symbol="AAPL", shares=10, avg_cost=90.0)]
    book = _run(m._build_portfolio_snapshot(holdings, cash=1000.0))
    assert book["total_value"] == 2000.0
    assert book["cash_pct"] == 50.0
    assert not book.get("unpriced")


def test_a_holding_with_no_cost_basis_is_not_valued_at_zero(priced):
    """avg_cost can legitimately be 0 — the user never entered a basis.

    Valuing at cost is then valuing at ZERO, which reproduces the exact weight inflation the
    carry-at-cost fallback exists to prevent: measured 33.3% -> 50.0%. A book containing a holding
    with neither a price nor a basis cannot be weighted at all, so it reports no weights rather than
    confident wrong ones.
    """
    priced.add("VXUS")
    holdings = [m.Holding(symbol="AAPL", shares=10, avg_cost=90.0),
                m.Holding(symbol="VXUS", shares=10, avg_cost=0.0)]   # no basis, unpriceable
    book = _run(m._build_portfolio_snapshot(holdings, cash=1000.0))

    assert book["unvalued"] == ["VXUS"]
    assert book["weights_approximate"] is True
    aapl = next(p for p in book["positions"] if p["symbol"] == "AAPL")
    assert aapl["weight_pct"] is None, (
        f"AAPL reported {aapl['weight_pct']}% off a denominator missing a whole position")
    assert book["cash_pct"] is None


def test_a_priceable_book_still_reports_real_weights(priced):
    # The suppression must be narrow: only an UNVALUED holding kills the weights.
    holdings = [m.Holding(symbol="AAPL", shares=10, avg_cost=90.0),
                m.Holding(symbol="VXUS", shares=10, avg_cost=0.0)]   # no basis but priceable
    book = _run(m._build_portfolio_snapshot(holdings, cash=1000.0))
    assert all(p["weight_pct"] is not None for p in book["positions"])
    assert book["cash_pct"] == pytest.approx(33.3, abs=0.2)
    assert not book.get("unvalued")


def test_the_same_symbol_sent_twice_becomes_one_position(priced):
    """Two rows for one ticker made the book diversified against itself.

    It produced AAPL at 66.7% and AAPL at 33.3% of the same book, an equivalent_exposures entry of
    {"AAPL": ["AAPL", "AAPL"]} (advice to consolidate AAPL into AAPL), and a per-exposure cap judged
    against half the real position.
    """
    holdings = [m.Holding(symbol="AAPL", shares=10, avg_cost=90.0),
                m.Holding(symbol="AAPL", shares=5, avg_cost=120.0)]
    book = _run(m._build_portfolio_snapshot(holdings, cash=0.0))
    assert len(book["positions"]) == 1
    pos = book["positions"][0]
    assert pos["shares"] == 15.0
    assert pos["avg_cost"] == pytest.approx(100.0)     # (10*90 + 5*120) / 15
    assert pos["weight_pct"] == 100.0
    assert not book.get("equivalent_exposures")


def test_mixed_currencies_are_named_not_silently_summed(priced, monkeypatch):
    """No FX rates here, so a GBP holding entered the USD total at face value.

    The app already refuses to present such a sum without a caveat; the backend handed the analyst
    the bare number and every weight computed off it.
    """
    def by_symbol(series, bench):
        return {"price": 100.0, "rsi14": 50.0, "currency": series.cur}
    seen = {}

    class S:
        closes = [100.0] * 60
        cur = "USD"

    async def fetch(http, sym, *a, **k):
        s = S()
        s.cur = "GBP" if sym.upper() == "VOD" else "USD"
        return s

    monkeypatch.setattr(m, "fetch_series", fetch)
    monkeypatch.setattr(m, "summarize", by_symbol)
    holdings = [m.Holding(symbol="AAPL", shares=10, avg_cost=90.0),
                m.Holding(symbol="VOD", shares=10, avg_cost=80.0)]
    book = _run(m._build_portfolio_snapshot(holdings, cash=0.0))
    assert book["mixed_currencies"] == ["GBP"]


def test_a_single_currency_book_says_nothing(priced):
    holdings = [m.Holding(symbol="AAPL", shares=10, avg_cost=90.0)]
    book = _run(m._build_portfolio_snapshot(holdings, cash=0.0))
    assert not book.get("mixed_currencies")
