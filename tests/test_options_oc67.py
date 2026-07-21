"""Tests for OC-6 (ATM-IV logging + IV rank + debit-spread alternative) and OC-7 (optional Opus
analyst paragraph) — the additive enhancements to the `/options` call suggester.

Run from the repo root:

    .venv/bin/python -m pytest tests/test_options_oc67.py -v
    .venv/bin/python -m pytest tests/ -q -m "not live"   # skip the network smoke tests

Everything here is offline + deterministic: `iv_rank` is pure math, the debit-spread block is built on
a synthetic annotated chain, and the route tests monkeypatch `options.fetch_chain`, `main.fetch_series`,
`main.fetch_context`, and `main.options_note` (the LLM) — no network, no Anthropic key. The CRITICAL
guarantee under test is schema stability: the ORIGINAL OC-1 response keys must all still be present.
"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app import options as o
from app.main import app
from app.options import ExpiryChain, OptionChain, OptionContract

# --- the ORIGINAL OC-1 contract the app already depends on (must stay present) ---
ORIGINAL_TOP_KEYS = {
    "symbol", "spot", "as_of", "quote_delayed", "light", "light_reason", "expiry",
    "expected_move", "structure", "structure_note", "candidates", "warnings", "earnings",
}
# --- the additive OC-6/OC-7 fields (new, nullable) ---
NEW_TOP_KEYS = {"iv_rank", "alternative", "recommend_alternative", "analyst"}
CAND_KEYS = {
    "profile", "contract_symbol", "strike", "limit_price", "cost", "max_loss", "contracts",
    "breakeven", "breakeven_pct", "delta", "theta", "iv", "spread_pct", "open_interest",
    "expected_move", "order_ticket",
}
ALT_KEYS = {
    "structure", "long_strike", "short_strike", "net_debit", "cost", "max_profit", "max_loss",
    "breakeven", "note",
}

_BULLISH = {"golden_cross": True, "pct_vs_sma20": 3.1, "pct_vs_sma50": 6.2, "macd_hist": 0.5,
            "rel_strength_3mo_vs_benchmark": 0.04, "rsi14": 61.0, "pct_off_52w_high": -2.0}


# ======================================================================================
# Synthetic chain — premiums DECREASE with strike so a debit spread has a positive net debit.
# (annotate_expiry derives delta from IV, not from premium, so the strike ladder stays realistic.)
# ======================================================================================

def _chain(now, *, spot=100.0, dte=60, oi=500, iv=0.35, quote_delayed=False):
    exp_ts = int(now + dte * 86400)
    iso = time.strftime("%Y-%m-%d", time.gmtime(exp_ts))
    exps = [{"ts": exp_ts, "iso": iso}]
    chain = OptionChain(symbol="TEST", spot=spot, currency="USD", market_state="REGULAR",
                        quote_delayed=quote_delayed, expirations=exps, strikes=[])
    calls = []
    for k in (80, 85, 90, 95, 100, 105, 110, 115, 120, 130):
        intrinsic = max(0.0, spot - k)
        tv = max(0.30, (130 - k) * 0.10)     # time value decreasing in strike -> higher K is cheaper
        premium = intrinsic + tv
        calls.append(OptionContract(
            type="call", contract_symbol=f"TEST{exp_ts}C{int(k * 1000):08d}", strike=float(k),
            expiration=exp_ts, bid=premium - 0.03, ask=premium + 0.03, last_price=premium,
            implied_volatility=iv, open_interest=oi, volume=100, in_the_money=k < spot,
        ))
    chain.expiry = ExpiryChain(expiration=exp_ts, expiration_iso=iso, calls=calls, puts=[])
    o.annotate_expiry(chain)
    return chain


# ======================================================================================
# OC-6a — iv_rank math
# ======================================================================================

def test_iv_rank_math_and_clamp():
    vals = [0.20, 0.30, 0.40, 0.50] * 6           # 24 points, min 0.20, max 0.50
    assert o.iv_rank(vals) == 100.0                # default current = last (0.50) = max
    assert o.iv_rank(vals, current=0.35) == 50.0   # (0.35-0.20)/(0.50-0.20)*100
    assert o.iv_rank(vals, current=0.20) == 0.0
    assert o.iv_rank(vals, current=0.95) == 100.0  # above the window -> clamped, not >100
    assert o.iv_rank(vals, current=0.05) == 0.0    # below the window -> clamped, not <0


def test_iv_rank_too_few_points_is_null():
    assert o.iv_rank([0.2, 0.3] * 9) is None        # 18 points < 20 -> "building"
    assert o.iv_rank([], current=0.4) is None


def test_iv_rank_flat_window_is_null():
    assert o.iv_rank([0.30] * 25) is None           # max == min -> undefined -> null


def test_iv_rank_ignores_nonpositive_points():
    vals = [0.0, -1.0] + [0.20, 0.40] * 12          # junk stripped, 24 valid points remain
    assert o.iv_rank(vals, current=0.30) == 50.0


# ======================================================================================
# OC-6a — iv_history file I/O (tolerates missing/short/corrupt; one line/symbol/day)
# ======================================================================================

def test_iv_history_append_load_and_windowing(monkeypatch, tmp_path):
    monkeypatch.setattr(o, "IV_HISTORY", tmp_path / "iv_history.jsonl")

    assert o.load_iv_history("AAPL") == []                       # missing file -> []
    assert o.append_iv_history("AAPL", 0.40, date_str="2026-07-01") is True
    assert o.append_iv_history("AAPL", 0.41, date_str="2026-07-01") is False  # dedupe: same symbol/day
    assert o.append_iv_history("AAPL", 0.50, date_str="2026-07-02") is True
    assert o.append_iv_history("MSFT", 0.30, date_str="2026-07-01") is True
    assert o.append_iv_history("AAPL", None, date_str="2026-07-03") is False  # no IV -> skip

    assert o.load_iv_history("AAPL") == [0.40, 0.50]             # oldest -> newest, MSFT excluded
    assert o.load_iv_history("MSFT") == [0.30]
    assert o.load_iv_history("AAPL", window=1) == [0.50]         # window keeps the newest N
    assert o.load_iv_history("NVDA") == []                       # unknown symbol -> []


def test_iv_history_tolerates_corrupt_lines(monkeypatch, tmp_path):
    p = tmp_path / "iv_history.jsonl"
    p.write_text('{"symbol":"AAPL","date":"2026-07-01","atm_iv":0.4}\nNOT JSON\n\n'
                 '{"symbol":"AAPL","date":"2026-07-02","atm_iv":0.5}\n')
    monkeypatch.setattr(o, "IV_HISTORY", p)
    assert o.load_iv_history("AAPL") == [0.4, 0.5]               # corrupt/blank lines skipped


# ======================================================================================
# OC-6b — the debit-spread alternative block
# ======================================================================================

def test_debit_spread_block_math():
    now = time.time()
    chain = _chain(now)
    chosen, _ = o.select_expiry(chain, now=now)
    resp = o.assemble_suggestion(chain, chain.expiry, _BULLISH, chosen=chosen, now=now)

    alt = resp["alternative"]
    assert alt is not None and set(alt) == ALT_KEYS
    assert alt["structure"] == "debit_call_spread"
    assert alt["short_strike"] > alt["long_strike"]              # SELL leg is a higher strike

    long_c = next(c for c in chain.expiry.calls if c.strike == alt["long_strike"])
    short_c = next(c for c in chain.expiry.calls if c.strike == alt["short_strike"])
    net = o._limit_price(long_c) - o._limit_price(short_c)
    width = alt["short_strike"] - alt["long_strike"]
    assert alt["net_debit"] == pytest.approx(net, abs=0.01) and net > 0
    assert alt["cost"] == pytest.approx(net * 100, abs=0.5)
    assert alt["max_loss"] == pytest.approx(net * 100, abs=0.5)
    assert alt["max_loss"] == alt["cost"]
    assert alt["max_profit"] == pytest.approx((width - net) * 100, abs=0.5)
    assert alt["breakeven"] == pytest.approx(alt["long_strike"] + net, abs=0.01)


def test_recommend_alternative_gates_on_iv_rank():
    now = time.time()
    chain = _chain(now)
    chosen, _ = o.select_expiry(chain, now=now)

    def rec(rank):
        r = o.assemble_suggestion(chain, chain.expiry, _BULLISH, chosen=chosen, now=now, iv_rank=rank)
        assert r["alternative"] is not None  # a spread IS derivable, so the toggle depends only on rank
        return r["recommend_alternative"], r["iv_rank"]

    assert rec(60.0) == (True, 60.0)      # rich IV (>= 50) -> recommend the spread
    assert rec(50.0) == (True, 50.0)      # boundary is inclusive
    assert rec(10.0) == (False, 10.0)     # cheap IV -> plain long call
    assert rec(None) == (False, None)     # unknown IV rank -> don't recommend


def test_iv_rank_null_notes_building_in_structure_note():
    now = time.time()
    chain = _chain(now)
    chosen, _ = o.select_expiry(chain, now=now)
    resp = o.assemble_suggestion(chain, chain.expiry, _BULLISH, chosen=chosen, now=now, iv_rank=None)
    assert resp["iv_rank"] is None
    assert "building" in resp["structure_note"].lower()

    resp2 = o.assemble_suggestion(chain, chain.expiry, _BULLISH, chosen=chosen, now=now, iv_rank=72.0)
    assert resp2["iv_rank"] == 72.0
    assert "IV rank ~72" in resp2["structure_note"]


def test_no_spread_when_long_leg_is_top_strike():
    """If the balanced (long) leg lands on the highest strike, there's no higher strike to SELL —
    the alternative is None and recommend_alternative is False even at high IV rank."""
    now = time.time()
    exp_ts = int(now + 60 * 86400)
    iso = time.strftime("%Y-%m-%d", time.gmtime(exp_ts))
    chain = OptionChain(symbol="TEST", spot=100.0, currency="USD", market_state="REGULAR",
                        quote_delayed=False, expirations=[{"ts": exp_ts, "iso": iso}], strikes=[])
    # Only two strikes; the balanced ~0.50-delta pick is the higher one, leaving nothing above it.
    calls = [
        OptionContract(type="call", contract_symbol=f"T{exp_ts}C1", strike=95.0, expiration=exp_ts,
                       bid=6.0, ask=6.1, last_price=6.0, implied_volatility=0.35, open_interest=500),
        OptionContract(type="call", contract_symbol=f"T{exp_ts}C2", strike=100.0, expiration=exp_ts,
                       bid=3.0, ask=3.1, last_price=3.0, implied_volatility=0.35, open_interest=500),
    ]
    chain.expiry = ExpiryChain(expiration=exp_ts, expiration_iso=iso, calls=calls, puts=[])
    o.annotate_expiry(chain)
    chosen, _ = o.select_expiry(chain, now=now)
    resp = o.assemble_suggestion(chain, chain.expiry, _BULLISH, chosen=chosen, now=now, iv_rank=90.0)
    # balanced pick is the 100 strike (delta ~0.55); nothing higher is quotable -> no spread
    assert resp["alternative"] is None
    assert resp["recommend_alternative"] is False


# ======================================================================================
# Schema stability — the ORIGINAL keys survive; the new ones are added
# ======================================================================================

def test_schema_stability_original_keys_intact():
    now = time.time()
    chain = _chain(now)
    chosen, _ = o.select_expiry(chain, now=now)
    resp = o.assemble_suggestion(chain, chain.expiry, _BULLISH, chosen=chosen, budget=3000.0, now=now)

    assert ORIGINAL_TOP_KEYS.issubset(resp), f"missing original keys: {ORIGINAL_TOP_KEYS - set(resp)}"
    assert NEW_TOP_KEYS.issubset(resp), f"missing new keys: {NEW_TOP_KEYS - set(resp)}"
    assert resp["structure"] == "long_call"               # structure unchanged
    for c in resp["candidates"]:
        assert set(c) == CAND_KEYS                         # candidate schema untouched


# ======================================================================================
# Route-level (OC-7) — deep=false leaves analyst null with NO LLM call; deep=true calls it once
# ======================================================================================

def _patch_route(monkeypatch, chain):
    """Offline route wiring: synthetic chain, no network for the directional snapshot."""
    async def fake_fetch(client, symbol, expiry_ts=None):
        return chain
    monkeypatch.setattr(o, "fetch_chain", fake_fetch)

    async def boom_series(client, symbol):               # skip the network snapshot (warning path)
        raise RuntimeError("offline")
    monkeypatch.setattr("app.main.fetch_series", boom_series)

    async def no_context(client, symbol):
        return {}
    monkeypatch.setattr("app.main.fetch_context", no_context)


def test_options_route_deep_false_no_llm_call(monkeypatch):
    calls = {"n": 0}

    async def spy(context, *, deep=True):
        calls["n"] += 1
        return ("para", {"model": "x", "input_tokens": 1, "output_tokens": 1,
                         "cache_read_tokens": 0, "cost_usd": 0.0})
    monkeypatch.setattr("app.main.options_note", spy)
    _patch_route(monkeypatch, _chain(time.time()))

    with TestClient(app) as client:
        r = client.get("/options/TEST")                  # deep defaults to false
        assert r.status_code == 200, r.text
        body = r.json()

    assert body["analyst"] is None                       # null on the free path
    assert calls["n"] == 0                                # and NO LLM call was made
    assert ORIGINAL_TOP_KEYS.issubset(body)              # schema stability at the route layer
    assert NEW_TOP_KEYS.issubset(body)


def test_options_route_deep_true_calls_llm_once(monkeypatch):
    calls = {"n": 0}

    async def spy(context, *, deep=True):
        calls["n"] += 1
        assert context["symbol"] == "TEST"               # the contract context is threaded through
        return ("Buy the 100 call ...", {"model": "claude-opus-4-8", "input_tokens": 5,
                                          "output_tokens": 9, "cache_read_tokens": 0, "cost_usd": 0.0})
    monkeypatch.setattr("app.main.options_note", spy)
    monkeypatch.setattr("app.main.usage_store.record", lambda *a, **k: None)  # don't touch data/usage.jsonl
    _patch_route(monkeypatch, _chain(time.time()))

    with TestClient(app) as client:
        r = client.get("/options/TEST", params={"deep": True})
        assert r.status_code == 200, r.text
        body = r.json()

    assert body["analyst"] == "Buy the 100 call ..."
    assert calls["n"] == 1
    assert ORIGINAL_TOP_KEYS.issubset(body) and NEW_TOP_KEYS.issubset(body)


def test_options_route_deep_true_llm_failure_leaves_analyst_null(monkeypatch):
    """A failing analyst call must NOT 500 — the deterministic body returns with analyst=null."""
    async def boom(context, *, deep=True):
        raise RuntimeError("anthropic down")
    monkeypatch.setattr("app.main.options_note", boom)
    _patch_route(monkeypatch, _chain(time.time()))

    with TestClient(app) as client:
        r = client.get("/options/TEST", params={"deep": True})
        assert r.status_code == 200, r.text
        assert r.json()["analyst"] is None
