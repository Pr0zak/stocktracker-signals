"""Tests for the OC-3 single-contract re-pricer — the `GET /option_quote/{symbol}` FastAPI route in
app.main. The app's call-position tracker calls this to show live P/L on a contract already bought.

Run from the repo root:

    .venv/bin/python -m pytest tests/test_option_quote.py -v
    .venv/bin/python -m pytest tests/ -v -m "not live"   # skip the network test

The offline tests monkeypatch `options.fetch_chain` with a synthetic chain, so they're deterministic
and need no network. `test_live_option_quote_aapl` hits Yahoo (crumb dance + a real AAPL chain), picks
an actual (expiry_ts, strike) from it, and SKIPS itself on any network failure — exactly like the
other live smoke tests — so an offline/flaky run stays green. No Anthropic key: this path is LLM-free.
"""
from __future__ import annotations

import asyncio
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from app import options as o
from app.main import app
from app.options import ExpiryChain, OptionChain, OptionContract

# --- the exact response contract OC-3 (the app) depends on ---
TOP_KEYS = {"symbol", "spot", "as_of", "quote_delayed", "dte", "contract"}
CONTRACT_KEYS = {
    "contract_symbol", "type", "strike", "expiration", "bid", "ask", "last_price", "mid",
    "limit_price", "implied_volatility", "delta", "theta", "open_interest", "in_the_money",
    "spread_pct",
}


def _synthetic_chain(now, *, spot=103.0, dte=30, strikes=(95.0, 100.0, 105.0, 110.0),
                     quote_delayed=False):
    """A small, plausible AAPL-shaped chain: calls + puts at a handful of strikes, quoted with IV so
    annotate_expiry can compute mid/spread/greeks."""
    exp_ts = int(now + dte * 86400)
    iso = time.strftime("%Y-%m-%d", time.gmtime(exp_ts))

    def contract(kind, k):
        intrinsic = max(0.0, spot - k) if kind == "call" else max(0.0, k - spot)
        base = intrinsic + 2.0
        return OptionContract(
            type=kind, contract_symbol=f"AAPL{exp_ts}{'C' if kind == 'call' else 'P'}{int(k * 1000):08d}",
            strike=float(k), expiration=exp_ts, bid=base - 0.05, ask=base + 0.05, last_price=base,
            implied_volatility=0.40, open_interest=1200, volume=80,
            in_the_money=(k < spot if kind == "call" else k > spot),
        )

    chain = OptionChain(
        symbol="AAPL", spot=spot, currency="USD", market_state="REGULAR",
        quote_delayed=quote_delayed, expirations=[{"ts": exp_ts, "iso": iso}], strikes=list(strikes),
    )
    chain.expiry = ExpiryChain(
        expiration=exp_ts, expiration_iso=iso,
        calls=[contract("call", k) for k in strikes],
        puts=[contract("put", k) for k in strikes],
    )
    return chain


def _patch_fetch(monkeypatch, chain):
    async def fake_fetch(client, symbol, expiry_ts=None):
        return chain
    monkeypatch.setattr(o, "fetch_chain", fake_fetch)


# ======================================================================================
# Offline route tests (synthetic chain — deterministic, no network)
# ======================================================================================

def test_option_quote_crypto_returns_400_not_500():
    """A -USD symbol short-circuits before any network call — a clean 400, never a 500."""
    with TestClient(app) as client:
        r = client.get("/option_quote/BTC-USD", params={"expiry_ts": 1893456000, "strike": 100})
        assert r.status_code == 400
        assert "crypto" in r.json()["detail"].lower()


def test_option_quote_found_offline(monkeypatch):
    """A real strike resolves to the matching contract with the exact response shape the app wants."""
    now = time.time()
    chain = _synthetic_chain(now)
    _patch_fetch(monkeypatch, chain)
    want = next(c for c in chain.expiry.calls if c.strike == 100.0)

    with TestClient(app) as client:
        r = client.get("/option_quote/AAPL",
                       params={"expiry_ts": chain.expiry.expiration, "strike": 100.0, "type": "call"})
        assert r.status_code == 200, r.text
        body = r.json()

        # --- exact top-level + contract shape ---
        assert set(body) == TOP_KEYS, f"top-level key drift: {set(body) ^ TOP_KEYS}"
        c = body["contract"]
        assert set(c) == CONTRACT_KEYS, f"contract key drift: {set(c) ^ CONTRACT_KEYS}"

        # --- the matched contract ---
        assert body["symbol"] == "AAPL"
        assert body["spot"] == round(chain.spot, 2)
        assert body["quote_delayed"] is False
        assert body["dte"] == pytest.approx(30.0, abs=0.01)
        assert c["type"] == "call"
        assert c["strike"] == 100.0
        assert c["contract_symbol"] == want.contract_symbol
        assert c["expiration"] == chain.expiry.expiration
        assert c["in_the_money"] is True  # spot 103 > strike 100

        # --- re-price: both sides quoted -> mid, and limit_price tracks it ---
        assert c["bid"] == pytest.approx(want.bid, abs=0.005)
        assert c["ask"] == pytest.approx(want.ask, abs=0.005)
        assert c["mid"] == pytest.approx((want.bid + want.ask) / 2, abs=0.01)
        assert c["limit_price"] == pytest.approx(c["mid"], abs=0.01)  # mid preferred over last

        # --- greeks present + sane for an ITM call ---
        assert 0.0 < c["delta"] < 1.0
        assert c["theta"] is not None and c["theta"] <= 0.0
        assert c["implied_volatility"] == pytest.approx(0.40, abs=1e-6)
        assert c["open_interest"] == 1200
        assert c["spread_pct"] is not None and c["spread_pct"] >= 0.0


def test_option_quote_put_side_offline(monkeypatch):
    """type=put resolves against the puts list; put delta is negative."""
    now = time.time()
    chain = _synthetic_chain(now)
    _patch_fetch(monkeypatch, chain)

    with TestClient(app) as client:
        r = client.get("/option_quote/AAPL",
                       params={"expiry_ts": chain.expiry.expiration, "strike": 110.0, "type": "put"})
        assert r.status_code == 200, r.text
        c = r.json()["contract"]
        assert c["type"] == "put"
        assert c["strike"] == 110.0
        assert c["in_the_money"] is True  # strike 110 > spot 103
        assert -1.0 < c["delta"] < 0.0


def test_option_quote_nonexistent_strike_returns_404(monkeypatch):
    """A strike not in the chain is a clean 404 (never a 500), naming the type/strike/symbol/expiry."""
    now = time.time()
    chain = _synthetic_chain(now)
    _patch_fetch(monkeypatch, chain)

    with TestClient(app) as client:
        r = client.get("/option_quote/AAPL",
                       params={"expiry_ts": chain.expiry.expiration, "strike": 999.0, "type": "call"})
        assert r.status_code == 404
        detail = r.json()["detail"].lower()
        assert "no call at strike" in detail and "999" in detail and "aapl" in detail


def test_option_quote_no_chain_returns_400(monkeypatch):
    """A chain with no loaded expiry (illiquid/unknown symbol) degrades to 400, not 500."""
    async def fake_fetch(client, symbol, expiry_ts=None):
        return OptionChain(symbol="ZZZZ", spot=None, currency="USD", market_state="CLOSED",
                           quote_delayed=True, expirations=[], strikes=[])
    monkeypatch.setattr(o, "fetch_chain", fake_fetch)

    with TestClient(app) as client:
        r = client.get("/option_quote/ZZZZ", params={"expiry_ts": 1893456000, "strike": 100.0})
        assert r.status_code == 400
        assert "no option chain" in r.json()["detail"].lower()


def test_option_quote_fetch_error_returns_400(monkeypatch):
    """A raised fetch (network/handshake failure) is caught and returned as a generic 400, never 500,
    and the crumb-bearing URL never leaks into the client-facing detail."""
    async def boom(client, symbol, expiry_ts=None):
        raise RuntimeError("https://query1.finance.yahoo.com/...crumb=SECRET failed")
    monkeypatch.setattr(o, "fetch_chain", boom)

    with TestClient(app) as client:
        r = client.get("/option_quote/AAPL", params={"expiry_ts": 1893456000, "strike": 100.0})
        assert r.status_code == 400
        assert "crumb" not in r.json()["detail"].lower()
        assert "aren't available" in r.json()["detail"].lower()


# ======================================================================================
# Live route test (network — skips itself on failure)
# ======================================================================================

@pytest.mark.live
def test_live_option_quote_aapl():
    """Fetch a REAL AAPL chain, pick an actual (expiry_ts, strike) from it, re-price it through the
    endpoint, and assert the contract block matches. SKIPS (not fails) if Yahoo is unreachable."""
    async def run():
        async with httpx.AsyncClient(follow_redirects=True) as client:
            return await o.fetch_chain(client, "AAPL")

    try:
        chain = asyncio.run(run())
    except Exception as e:  # noqa: BLE001 — network/handshake trouble -> skip, don't fail
        pytest.skip(f"live Yahoo options fetch unavailable: {e}")
    if chain.expiry is None or not chain.expiry.calls:
        pytest.skip("no calls in the default AAPL expiry")

    # Pick a real, near-the-money call — most likely to carry a live quote.
    spot = chain.spot or 0.0
    call = min(chain.expiry.calls, key=lambda c: abs(c.strike - spot))
    expiry_ts = chain.expiry.expiration
    strike = call.strike

    with TestClient(app) as client:
        r = client.get("/option_quote/AAPL",
                       params={"expiry_ts": expiry_ts, "strike": strike, "type": "call"})
        if r.status_code != 200:
            pytest.skip(f"live /option_quote/AAPL unavailable: {r.status_code} {r.text[:120]}")
        body = r.json()

        # --- exact shape ---
        assert set(body) == TOP_KEYS, f"top-level key drift: {set(body) ^ TOP_KEYS}"
        c = body["contract"]
        assert set(c) == CONTRACT_KEYS, f"contract key drift: {set(c) ^ CONTRACT_KEYS}"

        # --- it re-priced the SAME contract we picked from the raw chain ---
        assert body["symbol"] == "AAPL"
        assert isinstance(body["spot"], (int, float)) and body["spot"] > 0
        assert body["dte"] >= 0
        assert c["type"] == "call"
        assert abs(c["strike"] - strike) < 0.01
        assert c["contract_symbol"] == call.contract_symbol
        assert c["expiration"] == expiry_ts
        assert isinstance(body["quote_delayed"], bool)

        # --- numeric fields are nullable but, when present, sane ---
        if c["delta"] is not None:
            assert 0.0 <= c["delta"] <= 1.0
        for money in ("bid", "ask", "last_price", "mid", "limit_price"):
            assert c[money] is None or c[money] >= 0.0
        # limit_price is the re-price: mid when both-sided, else last trade — one of them when quoted.
        if c["mid"] is not None:
            assert c["limit_price"] == pytest.approx(c["mid"], abs=0.01)
