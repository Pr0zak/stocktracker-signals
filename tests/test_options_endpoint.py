"""Tests for the OC-1 no-LLM call suggester — the pure suggester layer in app.options and the
`GET /options/{symbol}` FastAPI route in app.main.

Run from the repo root:

    .venv/bin/python -m pytest tests/test_options_endpoint.py -v
    .venv/bin/python -m pytest tests/ -v -m "not live"   # skip the network smoke test

The suggester logic (directional read, expiry pick, strike pick, assembly) is exercised offline with
a synthetic chain — those tests are deterministic. `test_live_options_aapl` actually hits Yahoo (the
crumb dance + a real AAPL chain) and SKIPS itself on any network failure, exactly like OC-0's live
smoke test, so an offline/flaky run stays green. No Anthropic key is needed: this path is LLM-free.
"""
from __future__ import annotations

import re
import time

import pytest
from fastapi.testclient import TestClient

from app import options as o
from app.main import app
from app.options import ExpiryChain, OptionChain, OptionContract

# --- the exact response contract OC-2 (the app) depends on ---
TOP_KEYS = {
    "symbol", "spot", "as_of", "quote_delayed", "light", "light_reason", "expiry",
    "expected_move", "structure", "structure_note", "candidates", "warnings", "earnings",
}
CAND_KEYS = {
    "profile", "contract_symbol", "strike", "limit_price", "cost", "max_loss", "contracts",
    "breakeven", "breakeven_pct", "delta", "theta", "iv", "spread_pct", "open_interest",
    "expected_move", "order_ticket",
}
TICKET_RE = re.compile(r"^BUY \d+ \S+ \d{2}/\d{2}/\d{2} [\d.]+ C @ \d+\.\d{2} LMT$")


# ======================================================================================
# directional_read — the mechanical no-LLM stand-in for the Signals-card direction
# ======================================================================================

def test_directional_read_bullish():
    s = {"golden_cross": True, "pct_vs_sma20": 3.1, "pct_vs_sma50": 6.2, "macd_hist": 0.5,
         "rel_strength_3mo_vs_benchmark": 0.04, "rsi14": 61.0, "pct_off_52w_high": -2.0}
    d = o.directional_read(s)
    assert d["signal"] == "bullish" and d["bullish"] is True
    assert d["score"] >= o.BULLISH_SCORE


def test_directional_read_bearish():
    s = {"golden_cross": False, "pct_vs_sma20": -3.1, "pct_vs_sma50": -6.2, "macd_hist": -0.5,
         "rel_strength_3mo_vs_benchmark": -0.04, "rsi14": 40.0, "pct_off_52w_high": -30.0}
    d = o.directional_read(s)
    assert d["signal"] == "bearish" and d["bullish"] is False
    assert d["score"] <= 45


def test_directional_read_empty_is_neutral():
    """No technicals -> neutral 50, which is NOT bullish (so the light defaults to caution/red)."""
    d = o.directional_read({})
    assert d["signal"] == "neutral" and d["score"] == 50 and d["bullish"] is False


# ======================================================================================
# select_expiry — window / target-date / earnings-straddle logic
# ======================================================================================

def _bare_chain(dtes, *, now, spot=100.0, quote_delayed=False):
    exps = [{"ts": int(now + d * 86400), "iso": time.strftime("%Y-%m-%d", time.gmtime(now + d * 86400))}
            for d in dtes]
    return OptionChain(symbol="TEST", spot=spot, currency="USD", market_state="REGULAR",
                       quote_delayed=quote_delayed, expirations=exps, strikes=[])


def test_select_expiry_prefers_60():
    now = time.time()
    chosen, warns = o.select_expiry(_bare_chain([7, 30, 60, 75, 120], now=now), now=now)
    assert chosen is not None and chosen["dte"] == 60 and warns == []


def test_select_expiry_skips_earnings_straddle():
    now = time.time()
    # 50/60/75 are all in the 45-90 window; earnings at day 55 straddles 60 & 75 but NOT 50.
    earn = time.strftime("%Y-%m-%d", time.gmtime(now + 55 * 86400))
    chosen, warns = o.select_expiry(_bare_chain([30, 50, 60, 75], now=now), now=now, earnings_date=earn)
    assert chosen["dte"] == 50
    assert not any("straddle" in w for w in warns)


def test_select_expiry_target_date_forces_later_and_warns():
    now = time.time()
    tgt = time.strftime("%Y-%m-%d", time.gmtime(now + 80 * 86400))
    chosen, warns = o.select_expiry(_bare_chain([7, 30, 60, 75, 120], now=now), now=now, target_date=tgt)
    assert chosen["dte"] == 120  # only the 120-DTE expiry clears the target (+buffer)
    assert any("45-90" in w for w in warns)


def test_select_expiry_none_when_nothing_expires_in_time():
    now = time.time()
    # all in the past -> nothing to pick
    assert o.select_expiry(_bare_chain([-5, -1], now=now), now=now)[0] is None


# ======================================================================================
# Full assembly on a synthetic chain
# ======================================================================================

def _synthetic_annotated_chain(now, *, spot=100.0, oi=500, quote_delayed=False, dte=60):
    exp_ts = int(now + dte * 86400)
    chain = _bare_chain([dte], now=now, spot=spot, quote_delayed=quote_delayed)
    calls = []
    for k in (80, 85, 90, 95, 100, 105, 110, 115, 120, 130):
        intrinsic = max(0.0, spot - k)
        base = intrinsic + 2.0  # a plausible premium with time value
        calls.append(OptionContract(
            type="call", contract_symbol=f"TEST{exp_ts}C{int(k * 1000):08d}", strike=float(k),
            expiration=exp_ts, bid=base - 0.03, ask=base + 0.03, last_price=base,
            implied_volatility=0.35, open_interest=oi, volume=100, in_the_money=k < spot,
        ))
    chain.expiry = ExpiryChain(expiration=exp_ts,
                               expiration_iso=time.strftime("%Y-%m-%d", time.gmtime(exp_ts)),
                               calls=calls, puts=[])
    o.annotate_expiry(chain)
    return chain


def _assert_candidate_shape(c, spot):
    assert set(c) == CAND_KEYS
    assert 0.0 < c["delta"] < 1.0
    assert c["limit_price"] > 0 and c["cost"] == pytest.approx(c["limit_price"] * 100, abs=0.5)
    assert c["breakeven"] > spot  # a long call's breakeven is above spot (strike + premium)
    assert TICKET_RE.match(c["order_ticket"]), c["order_ticket"]


def test_assemble_green_when_bullish_and_clean():
    now = time.time()
    chain = _synthetic_annotated_chain(now, oi=800)
    summary = {"golden_cross": True, "pct_vs_sma20": 3.1, "pct_vs_sma50": 6.2, "macd_hist": 0.5,
               "rel_strength_3mo_vs_benchmark": 0.04, "rsi14": 61.0, "pct_off_52w_high": -2.0}
    chosen, _ = o.select_expiry(chain, now=now)
    resp = o.assemble_suggestion(chain, chain.expiry, summary, chosen=chosen, style="balanced",
                                 budget=3000.0, now=now)
    assert set(TOP_KEYS).issubset(resp)
    assert resp["light"] == "green"  # bullish + tight spread + fat OI + no earnings
    assert resp["structure"] == "long_call"
    assert resp["candidates"][0]["profile"] == "balanced"  # requested style first
    for c in resp["candidates"]:
        _assert_candidate_shape(c, resp["spot"])
    # deltas ordered by profile: safer >= balanced >= cheaper
    by = {c["profile"]: c["delta"] for c in resp["candidates"]}
    assert by["safer"] >= by["balanced"] >= by["cheaper"]


def test_assemble_red_when_not_bullish():
    now = time.time()
    chain = _synthetic_annotated_chain(now)
    summary = {"golden_cross": False, "pct_vs_sma20": -4.0, "pct_vs_sma50": -6.0, "macd_hist": -1.0,
               "rel_strength_3mo_vs_benchmark": -0.05, "rsi14": 38.0, "pct_off_52w_high": -35.0}
    chosen, _ = o.select_expiry(chain, now=now)
    resp = o.assemble_suggestion(chain, chain.expiry, summary, chosen=chosen, now=now)
    assert resp["light"] == "red"
    assert "wait/sell" in resp["light_reason"]
    # candidates are still returned (the app decides whether to show them); schema stays stable.
    assert 1 <= len(resp["candidates"]) <= 3


def test_assemble_yellow_on_earnings_in_window():
    now = time.time()
    chain = _synthetic_annotated_chain(now, oi=800)
    summary = {"golden_cross": True, "pct_vs_sma20": 3.1, "pct_vs_sma50": 6.2, "macd_hist": 0.5,
               "rel_strength_3mo_vs_benchmark": 0.04, "rsi14": 61.0, "pct_off_52w_high": -2.0}
    chosen, _ = o.select_expiry(chain, now=now)  # 60 DTE; earnings at day 30 falls before it
    earn = time.strftime("%Y-%m-%d", time.gmtime(now + 30 * 86400))
    resp = o.assemble_suggestion(chain, chain.expiry, summary, chosen=chosen, earnings_date=earn, now=now)
    assert resp["light"] == "yellow"
    assert resp["earnings"] == {"date": earn, "in_window": True}
    assert any("earnings" in w for w in resp["warnings"])


def test_assemble_budget_sizes_contracts():
    now = time.time()
    chain = _synthetic_annotated_chain(now)
    summary = {"golden_cross": True, "macd_hist": 0.5, "rel_strength_3mo_vs_benchmark": 0.04}
    chosen, _ = o.select_expiry(chain, now=now)
    resp = o.assemble_suggestion(chain, chain.expiry, summary, chosen=chosen, budget=1500.0, now=now)
    for c in resp["candidates"]:
        assert c["contracts"] is not None and c["contracts"] >= 0
        # never buy more than the budget allows
        assert c["cost"] * c["contracts"] <= 1500.0 + 1e-6
        assert c["max_loss"] == pytest.approx(c["cost"] * c["contracts"], abs=0.01)


def test_assemble_no_budget_leaves_contracts_none():
    now = time.time()
    chain = _synthetic_annotated_chain(now)
    summary = {"golden_cross": True, "macd_hist": 0.5}
    chosen, _ = o.select_expiry(chain, now=now)
    resp = o.assemble_suggestion(chain, chain.expiry, summary, chosen=chosen, budget=None, now=now)
    for c in resp["candidates"]:
        assert c["contracts"] is None
        assert c["max_loss"] == c["cost"]  # per-contract premium is the max loss


# ======================================================================================
# Route-level tests
# ======================================================================================

def test_options_crypto_returns_400_not_500():
    """crypto=true short-circuits before any network call — a clean 400, never a 500."""
    with TestClient(app) as client:
        r = client.get("/options/BTC-USD", params={"crypto": True})
        assert r.status_code == 400
        assert "crypto" in r.json()["detail"].lower()
        # a -USD symbol is treated as crypto even without the flag
        assert client.get("/options/ETH-USD").status_code == 400


# ======================================================================================
# Regression tests for the reviewed defect fixes (F1, F4, F5, F9, F10)
# ======================================================================================

_BULLISH = {"golden_cross": True, "pct_vs_sma20": 3.1, "pct_vs_sma50": 6.2, "macd_hist": 0.5,
            "rel_strength_3mo_vs_benchmark": 0.04, "rsi14": 61.0, "pct_off_52w_high": -2.0}


def _ticket_qty(c):
    m = re.match(r"^BUY (\d+) ", c["order_ticket"])
    assert m, c["order_ticket"]
    return int(m.group(1))


def test_f1_max_loss_equals_cost_times_ticket_quantity():
    """F1: the invariant max_loss == cost × (the quantity on the order ticket) holds for all three
    budget regimes, and `contracts` is NEVER 0 while a BUY-n ticket exists."""
    now = time.time()
    chain = _synthetic_annotated_chain(now)
    chosen, _ = o.select_expiry(chain, now=now)

    # (a) no budget: contracts None, ticket buys 1, max_loss == a single premium (cost × 1).
    resp = o.assemble_suggestion(chain, chain.expiry, _BULLISH, chosen=chosen, budget=None, now=now)
    assert resp["candidates"]
    for c in resp["candidates"]:
        assert c["contracts"] is None
        assert _ticket_qty(c) == 1
        assert c["max_loss"] == pytest.approx(c["cost"] * _ticket_qty(c), abs=0.01)
        assert c["max_loss"] == pytest.approx(c["cost"], abs=0.01)

    # (b) budget affords >= 1 contract: contracts == affordable == ticket qty, max_loss scales with it.
    resp = o.assemble_suggestion(chain, chain.expiry, _BULLISH, chosen=chosen, budget=3000.0, now=now)
    for c in resp["candidates"]:
        assert c["contracts"] is not None and c["contracts"] >= 1
        assert _ticket_qty(c) == c["contracts"]
        assert c["cost"] * c["contracts"] <= 3000.0 + 1e-6
        assert c["max_loss"] == pytest.approx(c["cost"] * _ticket_qty(c), abs=0.01)

    # (c) budget below one contract: contracts falls back to 1 (NOT 0), max_loss == one premium,
    #     and a "above your budget" warning is emitted (never a silent 0 max_loss).
    resp = o.assemble_suggestion(chain, chain.expiry, _BULLISH, chosen=chosen, budget=25.0, now=now)
    for c in resp["candidates"]:
        assert c["cost"] > 25.0            # precondition: can't afford even one
        assert c["contracts"] == 1          # the bug produced 0 here
        assert _ticket_qty(c) == 1
        assert c["max_loss"] == pytest.approx(c["cost"], abs=0.01)  # never 0.0
        assert c["max_loss"] == pytest.approx(c["cost"] * _ticket_qty(c), abs=0.01)
    assert any("budget" in w for w in resp["warnings"])


def test_f4_boundary_dte_window_matches_display():
    """F4: an expiry at 44.6 DTE displays as 45 (in-window); the window classification must use that
    SAME int, so no 'outside 45-90' warning fires and the rationale calls it in the sweet spot."""
    now = time.time()
    # select_expiry side: displayed int is 45, and NO fallback/out-of-window warning.
    chosen, warns = o.select_expiry(_bare_chain([44.6], now=now), now=now)
    assert chosen["dte"] == 45
    assert not any("45-90" in w for w in warns)

    # assemble side: rationale agrees (sweet spot), and no outside-window warning.
    chain = _synthetic_annotated_chain(now, dte=44.6)
    chosen, warns = o.select_expiry(chain, now=now)
    assert chosen["dte"] == 45
    resp = o.assemble_suggestion(chain, chain.expiry, _BULLISH, chosen=chosen, now=now)
    assert "sweet spot" in resp["expiry"]["rationale"]
    assert not any("outside the preferred 45-90" in w for w in resp["warnings"])


def test_f5_red_reason_no_candidates_even_when_bullish():
    """F5: with a bullish direction but ZERO quotable contracts, the red light must cite the missing
    contracts, not the (false) 'wait/sell' directional reason."""
    now = time.time()
    exp_ts = int(now + 60 * 86400)
    chain = _bare_chain([60], now=now, spot=100.0)
    # No IV -> annotate can't compute delta -> nothing selectable -> zero candidates.
    calls = [OptionContract(type="call", contract_symbol=f"X{exp_ts}C{int(k * 1000):08d}",
                            strike=float(k), expiration=exp_ts, bid=1.0, ask=1.2, last_price=1.1,
                            implied_volatility=None, open_interest=500)
             for k in (90, 100, 110)]
    chain.expiry = ExpiryChain(expiration=exp_ts,
                               expiration_iso=time.strftime("%Y-%m-%d", time.gmtime(exp_ts)),
                               calls=calls, puts=[])
    o.annotate_expiry(chain)
    chosen, _ = o.select_expiry(chain, now=now)
    resp = o.assemble_suggestion(chain, chain.expiry, _BULLISH, chosen=chosen, now=now)

    assert o.directional_read(_BULLISH)["bullish"] is True   # direction IS bullish
    assert resp["candidates"] == []                          # but nothing quotable
    assert resp["light"] == "red"
    assert resp["light_reason"] == "no quotable contracts for the chosen expiry"


def test_f10_requested_style_dropped_when_unquotable_warns():
    """F10: when the requested style's strike is unquotable and gets dropped, warn that we're showing
    a different profile first (rather than silently violating style-first)."""
    now = time.time()
    chain = _synthetic_annotated_chain(now)
    # Blank the quotes on whatever contract the SAFER profile would pick, then re-derive: it now has
    # no mid and no last -> no limit price -> build_candidates drops it.
    pick = o._pick_by_delta(chain.expiry.calls, o.TARGET_DELTAS["safer"])
    pick.bid = pick.ask = pick.last_price = None
    o.annotate_expiry(chain)
    chosen, _ = o.select_expiry(chain, now=now)
    resp = o.assemble_suggestion(chain, chain.expiry, _BULLISH, chosen=chosen, style="safer", now=now)

    assert not any(c["profile"] == "safer" for c in resp["candidates"])
    assert any("safer" in w and "wasn't quotable" in w for w in resp["warnings"])


def test_f9_budget_validation_returns_422_offline():
    """F9: a non-positive or non-finite budget is a 422 BEFORE any chain fetch (offline, no network)."""
    with TestClient(app) as client:
        for bad in ("-5", "inf", "nan", "0"):
            r = client.get("/options/AAPL", params={"budget": bad})
            assert r.status_code == 422, f"budget={bad} -> {r.status_code}"
            assert "budget" in r.json()["detail"].lower()


@pytest.mark.live
def test_live_options_aapl():
    """Hit the real endpoint for AAPL end-to-end. SKIPS (not fails) if Yahoo is unreachable."""
    with TestClient(app) as client:
        r = client.get("/options/AAPL", params={"budget": 2000, "style": "balanced"})
        if r.status_code != 200:
            pytest.skip(f"live /options/AAPL unavailable: {r.status_code} {r.text[:120]}")
        body = r.json()

        # --- shape / stable contract ---
        assert set(TOP_KEYS).issubset(body), f"missing keys: {TOP_KEYS - set(body)}"
        assert isinstance(body["spot"], (int, float)) and body["spot"] > 0
        assert body["light"] in {"green", "yellow", "red"}
        assert isinstance(body["light_reason"], str) and body["light_reason"]
        assert body["structure"] == "long_call"
        assert set(body["expiry"]) == {"ts", "iso", "dte", "rationale"}
        assert body["expiry"]["dte"] > 0

        # --- candidates: 1-3, sane deltas, requested style first, valid tickets ---
        cands = body["candidates"]
        assert 1 <= len(cands) <= 3
        assert cands[0]["profile"] == "balanced"  # the requested style leads
        profiles = [c["profile"] for c in cands]
        assert len(profiles) == len(set(profiles))  # de-duped
        for c in cands:
            assert set(c) == CAND_KEYS, f"candidate key drift: {set(c) ^ CAND_KEYS}"
            assert 0.0 < c["delta"] < 1.0, f"insane delta {c['delta']}"
            assert c["strike"] > 0 and c["limit_price"] > 0
            assert c["breakeven"] > 0
            assert TICKET_RE.match(c["order_ticket"]), c["order_ticket"]
        # delta ordering by profile (higher delta = safer)
        by = {c["profile"]: c["delta"] for c in cands}
        if "safer" in by and "balanced" in by:
            assert by["safer"] >= by["balanced"]
        if "balanced" in by and "cheaper" in by:
            assert by["balanced"] >= by["cheaper"]
