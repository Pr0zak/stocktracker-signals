"""Tests for app.options — Black-Scholes greeks, derived helpers, and a live chain smoke test.

Run from the repo root so `app` is importable:

    .venv/bin/python -m pytest tests/ -v          # everything
    .venv/bin/python -m pytest tests/ -v -m "not live"   # skip the network smoke test

The greeks/helper tests are pure and offline. `test_live_chain_smoke` actually hits Yahoo (crumb
dance) and SKIPS itself if the network call fails, so a flaky/offline run never fails the suite.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from app.options import (
    RISK_FREE_RATE,
    annotate_expiry,
    black_scholes_greeks,
    breakeven,
    expected_move,
    fetch_chain,
    mid_price,
    norm_cdf,
    norm_pdf,
    spread_pct,
)


# ----------------------------------------------------------------------------------------
# Normal CDF / PDF
# ----------------------------------------------------------------------------------------

def test_norm_cdf_pdf_reference():
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_cdf(1.96) == pytest.approx(0.975, abs=1e-3)   # classic 2-sigma value
    assert norm_cdf(-1.96) == pytest.approx(0.025, abs=1e-3)
    assert norm_pdf(0.0) == pytest.approx(0.3989422804, abs=1e-9)  # 1/sqrt(2π)
    assert norm_pdf(1.0) == pytest.approx(0.2419707245, abs=1e-9)


# ----------------------------------------------------------------------------------------
# Black-Scholes greeks — reference values and invariants
# ----------------------------------------------------------------------------------------

def test_greeks_reference_call():
    """Textbook case S=K=100, T=1y, sigma=20%, r=5%. Hand values: d1=0.35, d2=0.15."""
    g = black_scholes_greeks(100.0, 100.0, 1.0, 0.20, r=0.05, is_call=True)
    assert g["delta"] == pytest.approx(0.63683, abs=5e-4)   # N(0.35)
    assert g["gamma"] == pytest.approx(0.018762, abs=2e-5)  # φ(d1)/(S·σ·√T)
    assert g["vega"] == pytest.approx(0.37524, abs=1e-3)    # per 1 vol point
    assert g["theta"] == pytest.approx(-0.017571, abs=5e-4)  # per calendar day


def test_greeks_reference_put_parity():
    """Same inputs, put side. Put delta = call delta - 1; gamma/vega identical."""
    call = black_scholes_greeks(100.0, 100.0, 1.0, 0.20, r=0.05, is_call=True)
    put = black_scholes_greeks(100.0, 100.0, 1.0, 0.20, r=0.05, is_call=False)
    assert put["delta"] == pytest.approx(-0.36317, abs=5e-4)  # N(0.35) - 1
    # put-call parity of deltas
    assert call["delta"] - put["delta"] == pytest.approx(1.0, abs=1e-9)
    # gamma and vega are the same for a call and put at the same strike
    assert call["gamma"] == pytest.approx(put["gamma"], abs=1e-12)
    assert call["vega"] == pytest.approx(put["vega"], abs=1e-12)


@pytest.mark.parametrize("is_call", [True, False])
def test_greeks_delta_ranges(is_call):
    """Call delta ∈ [0,1]; put delta ∈ [-1,0], across a spread of moneyness."""
    for strike in (60, 80, 100, 120, 140):
        g = black_scholes_greeks(100.0, float(strike), 0.25, 0.35, is_call=is_call)
        if is_call:
            assert 0.0 <= g["delta"] <= 1.0
        else:
            assert -1.0 <= g["delta"] <= 0.0


def test_atm_call_delta_band():
    """An ATM ~35-DTE call should have delta ≈ 0.5-0.6 (slightly above 0.5 from carry)."""
    t = 35.0 / 365.0
    g = black_scholes_greeks(100.0, 100.0, t, 0.30, r=RISK_FREE_RATE, is_call=True)
    assert 0.50 <= g["delta"] <= 0.60


def test_theta_negative_vega_positive():
    """A long call and a long put both bleed time value (theta<0) and gain on rising IV (vega>0)."""
    for is_call in (True, False):
        g = black_scholes_greeks(100.0, 100.0, 0.10, 0.30, is_call=is_call)
        assert g["theta"] < 0.0
        assert g["vega"] > 0.0
        assert g["gamma"] > 0.0


def test_greeks_degenerate_inputs():
    """At/after expiry (or zero vol) greeks collapse to intrinsic: delta is the ITM indicator."""
    # expired ITM call
    assert black_scholes_greeks(120.0, 100.0, 0.0, 0.3, is_call=True)["delta"] == 1.0
    # expired OTM call
    assert black_scholes_greeks(80.0, 100.0, 0.0, 0.3, is_call=True)["delta"] == 0.0
    # expired ITM put (spot below strike)
    assert black_scholes_greeks(80.0, 100.0, 0.0, 0.3, is_call=False)["delta"] == -1.0
    g = black_scholes_greeks(100.0, 100.0, -1.0, 0.0, is_call=True)
    assert g["gamma"] == 0.0 and g["theta"] == 0.0 and g["vega"] == 0.0


# ----------------------------------------------------------------------------------------
# Derived helpers
# ----------------------------------------------------------------------------------------

def test_mid_and_spread():
    assert mid_price(9.0, 10.0) == pytest.approx(9.5)
    assert spread_pct(9.0, 10.0) == pytest.approx(1.0 / 9.5, abs=1e-9)
    # a 0/0 quote (market closed) yields no mid/spread
    assert mid_price(0.0, 0.0) is None
    assert spread_pct(None, 10.0) is None


def test_breakeven():
    assert breakeven(420.0, 6.70, is_call=True) == pytest.approx(426.70)
    assert breakeven(420.0, 6.70, is_call=False) == pytest.approx(413.30)
    assert breakeven(100.0, None, is_call=True) is None


def test_expected_move():
    # spot 100, IV 30%, 365 DTE -> 1 sigma == 30
    assert expected_move(100.0, 0.30, 365.0) == pytest.approx(30.0, abs=1e-9)
    # quarter year scales by sqrt(0.25)=0.5
    assert expected_move(100.0, 0.30, 365.0 / 4) == pytest.approx(15.0, abs=1e-9)
    assert expected_move(None, 0.3, 30) is None
    assert expected_move(100.0, None, 30) is None


# ----------------------------------------------------------------------------------------
# Live chain smoke test (network — skips itself on failure)
# ----------------------------------------------------------------------------------------

@pytest.mark.live
def test_live_chain_smoke():
    """Fetch a real, liquid chain (AAPL) via the crumb dance and sanity-check the shape + greeks.

    SKIPS (not fails) if the network fetch errors, so an offline CI run stays green.
    """
    async def run():
        async with httpx.AsyncClient(follow_redirects=True) as client:
            return await fetch_chain(client, "AAPL")

    try:
        chain = asyncio.run(run())
    except Exception as e:  # noqa: BLE001 — network/handshake trouble -> skip, don't fail
        pytest.skip(f"live Yahoo options fetch unavailable: {e}")

    # --- shape ---
    assert chain.spot is not None and chain.spot > 0, "spot price should be positive"
    assert len(chain.expirations) >= 1, "expected at least one expiration"
    assert all("ts" in e and "iso" in e for e in chain.expirations)
    assert chain.expiry is not None, "a default expiry should be loaded"
    assert len(chain.expiry.calls) >= 1, "calls should be present"

    # at least one call carries bid/ask/IV (during/after a trading session)
    quoted = [
        c for c in chain.expiry.calls
        if c.implied_volatility and (c.bid is not None or c.ask is not None or c.last_price)
    ]
    assert quoted, "expected calls with IV + a quote"

    # --- annotate + greeks sanity ---
    expiry = annotate_expiry(chain)
    call_deltas = [c.delta for c in expiry.calls if c.delta is not None]
    put_deltas = [p.delta for p in expiry.puts if p.delta is not None]
    assert call_deltas, "expected computed call deltas"
    assert all(0.0 <= d <= 1.0 for d in call_deltas), "call deltas must be in [0,1]"
    if put_deltas:
        assert all(-1.0 <= d <= 0.0 for d in put_deltas), "put deltas must be in [-1,0]"
    # a deep-ITM call (low strike) should have delta near 1; a far-OTM call near 0
    assert max(call_deltas) > 0.6, "deep-ITM call should have a high delta"
    # a mid/spread and breakeven got attached to at least one contract
    assert any(c.mid is not None and c.breakeven is not None for c in expiry.calls)
