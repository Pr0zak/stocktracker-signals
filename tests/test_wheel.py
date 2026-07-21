"""Tests for the OC-8 wheel suggesters — the no-LLM cash-secured-PUT (`GET /puts/{symbol}`) and
covered-CALL (`GET /covered_call/{symbol}`) endpoints in app.main, plus the pure suggester layer in
app.options.

Run from the repo root:

    .venv/bin/python -m pytest tests/test_wheel.py -v
    .venv/bin/python -m pytest tests/ -v -m "not live"   # skip the network smoke tests

The suggester logic (short-dated expiry pick, |delta| strike pick, sizing, called-away math) is
exercised offline with a synthetic annotated chain — deterministic, no network. The route tests
monkeypatch `options.fetch_chain` (and `main.fetch_context`) with synthetic data. `test_live_*`
actually hit Yahoo and SKIP themselves on any network failure, exactly like the OC-0/OC-1 live
smoke tests, so an offline/flaky run stays green. No Anthropic key: this path is LLM-free.
"""
from __future__ import annotations

import re
import time

import pytest
from fastapi.testclient import TestClient

from app import options as o
from app.main import app
from app.options import ExpiryChain, OptionChain, OptionContract

# --- the exact response contracts the app depends on ---
PUT_TOP_KEYS = {"symbol", "spot", "as_of", "quote_delayed", "expiry", "candidates", "warnings",
                "earnings", "note"}
PUT_CAND_KEYS = {"profile", "contract_symbol", "strike", "limit_price", "premium_income",
                 "net_cost_per_share", "discount_vs_spot_pct", "cash_to_reserve", "contracts",
                 "static_yield_pct", "annualized_yield_pct", "assignment_prob_pct", "breakeven",
                 "delta", "theta", "iv", "open_interest", "spread_pct", "order_ticket"}
CC_TOP_KEYS = {"symbol", "spot", "as_of", "quote_delayed", "shares", "contracts", "expiry",
               "candidate", "warnings", "note"}
CC_CAND_KEYS = {"contract_symbol", "strike", "limit_price", "premium_income", "premium_yield_pct",
                "annualized_yield_pct", "assignment_prob_pct", "called_away_gain_from_here",
                "delta", "theta", "iv", "open_interest", "spread_pct", "order_ticket"}
PUT_TICKET_RE = re.compile(r"^SELL \d+ \S+ \d{2}/\d{2}/\d{2} [\d.]+ P @ \d+\.\d{2} LMT$")
CC_TICKET_RE = re.compile(r"^SELL \d+ \S+ \d{2}/\d{2}/\d{2} [\d.]+ C @ \d+\.\d{2} LMT$")


# ======================================================================================
# Synthetic chain — puts + calls with IV so annotate_expiry can compute deltas.
# ======================================================================================

def _wheel_chain(now, *, spot=103.0, dte=35, iv=0.40, oi=800, quote_delayed=False):
    """A plausible ladder around spot=103 chosen so the three put profiles resolve to DISTINCT
    strikes: |delta| ~0.43 (K102, aggressive), ~0.31 (K98, balanced), ~0.18 (K93, conservative), and
    a ~0.30-delta OTM call at K110."""
    exp_ts = int(now + dte * 86400)
    iso = time.strftime("%Y-%m-%d", time.gmtime(exp_ts))
    strikes = (85, 90, 93, 95, 98, 100, 102, 105, 110, 115, 120)

    def contract(kind, k):
        intrinsic = max(0.0, spot - k) if kind == "call" else max(0.0, k - spot)
        base = intrinsic + 2.0  # plausible premium with time value
        return OptionContract(
            type=kind,
            contract_symbol=f"TEST{exp_ts}{'C' if kind == 'call' else 'P'}{int(k * 1000):08d}",
            strike=float(k), expiration=exp_ts, bid=base - 0.03, ask=base + 0.03, last_price=base,
            implied_volatility=iv, open_interest=oi, volume=100,
            in_the_money=(k < spot if kind == "call" else k > spot),
        )

    chain = OptionChain(symbol="TEST", spot=spot, currency="USD", market_state="REGULAR",
                        quote_delayed=quote_delayed, expirations=[{"ts": exp_ts, "iso": iso}],
                        strikes=list(map(float, strikes)))
    chain.expiry = ExpiryChain(expiration=exp_ts, expiration_iso=iso,
                               calls=[contract("call", k) for k in strikes],
                               puts=[contract("put", k) for k in strikes])
    return chain


# ======================================================================================
# select_wheel_expiry — short-dated window / earnings-straddle logic
# ======================================================================================

def _bare_chain(dtes, *, now, spot=103.0):
    exps = [{"ts": int(now + d * 86400), "iso": time.strftime("%Y-%m-%d", time.gmtime(now + d * 86400))}
            for d in dtes]
    return OptionChain(symbol="TEST", spot=spot, currency="USD", market_state="REGULAR",
                       quote_delayed=False, expirations=exps, strikes=[])


def test_select_wheel_expiry_prefers_target():
    now = time.time()
    chosen, warns = o.select_wheel_expiry(_bare_chain([7, 21, 35, 45, 90], now=now),
                                          low=25, high=50, target=35, now=now)
    assert chosen is not None and chosen["dte"] == 35 and warns == []


def test_select_wheel_expiry_skips_earnings_straddle():
    now = time.time()
    # 28/35/45 are all in the 25-50 window; earnings at day 30 straddles 35 & 45 but NOT 28.
    earn = time.strftime("%Y-%m-%d", time.gmtime(now + 30 * 86400))
    chosen, warns = o.select_wheel_expiry(_bare_chain([21, 28, 35, 45], now=now),
                                          low=25, high=50, target=35, now=now, earnings_date=earn)
    assert chosen["dte"] == 28
    assert not any("straddle" in w for w in warns)


def test_select_wheel_expiry_falls_back_and_warns():
    now = time.time()
    chosen, warns = o.select_wheel_expiry(_bare_chain([7, 14, 120], now=now),
                                          low=25, high=50, target=35, now=now)
    assert chosen["dte"] == 14  # nearest >= WHEEL_MIN_DTE (14) to the 35 target
    assert any("25-50 day window" in w for w in warns)


def test_select_wheel_expiry_none_when_all_past():
    now = time.time()
    assert o.select_wheel_expiry(_bare_chain([-5, -1], now=now), low=25, high=50, target=35, now=now)[0] is None


# ======================================================================================
# /puts — pure assembly (cash-secured put suggester)
# ======================================================================================

def _assemble_puts(now, *, cash, style="aggressive", earnings_date=None, iv=0.40, oi=800,
                   quote_delayed=False):
    chain = _wheel_chain(now, iv=iv, oi=oi, quote_delayed=quote_delayed)
    o.annotate_expiry(chain, now_ts=now)
    chosen, _ = o.select_wheel_expiry(chain, low=o.PUT_DTE_LOW, high=o.PUT_DTE_HIGH,
                                      target=o.PUT_DTE_TARGET, now=now)
    return o.assemble_put_suggestion(chain, chain.expiry, chosen=chosen, cash=cash, style=style,
                                     earnings_date=earnings_date, now=now)


def test_puts_three_profiles_ordered_by_assignment_prob():
    now = time.time()
    body = _assemble_puts(now, cash=30_000.0, style="aggressive")
    assert set(body) == PUT_TOP_KEYS, f"top-level key drift: {set(body) ^ PUT_TOP_KEYS}"
    cands = body["candidates"]
    assert len(cands) == 3
    profiles = [c["profile"] for c in cands]
    assert profiles == ["aggressive", "balanced", "conservative"]  # style=aggressive => assignment-prob order
    # assignment probability strictly decreases aggressive -> balanced -> conservative
    probs = [c["assignment_prob_pct"] for c in cands]
    assert probs[0] > probs[1] > probs[2], probs
    # and the mapping holds independent of array order
    by = {c["profile"]: c["assignment_prob_pct"] for c in cands}
    assert by["aggressive"] > by["balanced"] > by["conservative"]
    for c in cands:
        assert set(c) == PUT_CAND_KEYS, f"candidate key drift: {set(c) ^ PUT_CAND_KEYS}"
        assert -1.0 < c["delta"] < 0.0                       # put delta is negative
        assert c["assignment_prob_pct"] == round(abs(c["delta"]) * 100)
        assert PUT_TICKET_RE.match(c["order_ticket"]), c["order_ticket"]


def test_puts_sizing_invariant_consistent():
    """net_cost, cash_to_reserve, contracts and premium_income are internally consistent, and no
    figure reads $0 while a SELL-n ticket exists (the OC-1 sizing invariant)."""
    now = time.time()
    cash = 30_000.0
    body = _assemble_puts(now, cash=cash, style="aggressive")
    for c in body["candidates"]:
        strike, limit, n = c["strike"], c["limit_price"], c["contracts"]
        assert n >= 1                                          # never 0
        assert int(re.match(r"^SELL (\d+) ", c["order_ticket"]).group(1)) == n  # ticket qty == contracts
        assert c["cash_to_reserve"] == pytest.approx(strike * 100 * n, abs=0.5)
        assert c["cash_to_reserve"] <= cash + 1e-6            # never reserve more cash than you have
        assert c["premium_income"] == pytest.approx(limit * 100 * n, abs=0.5)
        assert c["net_cost_per_share"] == pytest.approx(strike - limit, abs=0.01)
        assert c["breakeven"] == pytest.approx(strike - limit, abs=0.01)
        assert c["premium_income"] > 0 and c["cash_to_reserve"] > 0   # no $0 on a live ticket
        # an OTM put's net cost (strike - premium) is below today's price -> a genuine discount
        assert c["discount_vs_spot_pct"] < 0
        assert c["static_yield_pct"] == pytest.approx(limit / strike * 100, abs=0.02)
        assert c["annualized_yield_pct"] == pytest.approx(
            c["static_yield_pct"] * 365 / body["expiry"]["dte"], abs=0.2)


def test_puts_cash_below_one_contract_warns_and_sizes_to_one():
    """Cash that can't secure even the cheapest strike -> contracts falls back to 1 (NOT 0), and a
    'below ... reserve' warning fires (never a silent $0)."""
    now = time.time()
    body = _assemble_puts(now, cash=5_000.0, style="aggressive")  # < the ~$9,300 cheapest reserve
    for c in body["candidates"]:
        assert c["strike"] * 100 > 5_000.0        # precondition: can't secure one contract
        assert c["contracts"] == 1                 # sized to the 1-contract minimum, never 0
        assert c["premium_income"] > 0
    assert any("reserve" in w for w in body["warnings"])


def test_puts_earnings_in_window_populates_block_and_warns():
    now = time.time()
    earn = time.strftime("%Y-%m-%d", time.gmtime(now + 20 * 86400))  # before the ~35 DTE expiry
    body = _assemble_puts(now, cash=30_000.0, earnings_date=earn)
    assert body["earnings"] == {"date": earn, "in_window": True}
    assert any("earnings" in w for w in body["warnings"])


def test_puts_high_iv_is_noted():
    now = time.time()
    body = _assemble_puts(now, cash=30_000.0, iv=0.95)  # >= HIGH_IV (0.80)
    assert any("IV is elevated" in w for w in body["warnings"])


def test_puts_note_and_not_advice():
    now = time.time()
    body = _assemble_puts(now, cash=30_000.0)
    assert "cash-secured" in body["note"].lower()
    assert "not investment advice" in body["note"].lower()


# ======================================================================================
# /covered_call — pure assembly (covered call suggester)
# ======================================================================================

def _assemble_cc(now, *, shares, target=None, iv=0.40, oi=800):
    chain = _wheel_chain(now, iv=iv, oi=oi)
    o.annotate_expiry(chain, now_ts=now)
    chosen, _ = o.select_wheel_expiry(chain, low=o.CALL_DTE_LOW, high=o.CALL_DTE_HIGH,
                                      target=o.CALL_DTE_TARGET, now=now)
    return o.assemble_covered_call(chain, chain.expiry, shares=shares, chosen=chosen, target=target, now=now)


def test_covered_call_default_delta_pick_and_shape():
    now = time.time()
    body = _assemble_cc(now, shares=300)
    assert set(body) == CC_TOP_KEYS, f"top-level key drift: {set(body) ^ CC_TOP_KEYS}"
    assert body["shares"] == 300 and body["contracts"] == 3
    c = body["candidate"]
    assert set(c) == CC_CAND_KEYS, f"candidate key drift: {set(c) ^ CC_CAND_KEYS}"
    assert c["strike"] > body["spot"]                 # ~0.30 delta call is OTM
    assert 0.0 < c["delta"] < 1.0
    assert c["assignment_prob_pct"] == round(c["delta"] * 100)
    assert CC_TICKET_RE.match(c["order_ticket"]), c["order_ticket"]
    # called-away gain = upside to the strike + premium, from today's price
    expected = round((c["strike"] - body["spot"]) * 100 * body["contracts"] + c["premium_income"], 2)
    assert c["called_away_gain_from_here"] == pytest.approx(expected, abs=0.5)
    assert c["premium_income"] == pytest.approx(c["limit_price"] * 100 * body["contracts"], abs=0.5)
    assert c["premium_yield_pct"] == pytest.approx(c["limit_price"] / body["spot"] * 100, abs=0.02)


def test_covered_call_target_picks_nearest_strike_at_or_above():
    now = time.time()
    body = _assemble_cc(now, shares=200, target=115.0)
    assert body["contracts"] == 2
    assert body["candidate"]["strike"] == 115.0     # exact strike at the target
    # positive called-away gain: strike (115) is above spot (103)
    assert body["candidate"]["called_away_gain_from_here"] > 0


def test_covered_call_target_above_all_strikes_falls_back_and_warns():
    now = time.time()
    body = _assemble_cc(now, shares=100, target=500.0)  # above every listed strike
    assert body["contracts"] == 1
    assert any("above every listed strike" in w for w in body["warnings"])


def test_covered_call_note_and_not_advice():
    now = time.time()
    body = _assemble_cc(now, shares=100)
    assert "covered call" in body["note"].lower()
    assert "not investment advice" in body["note"].lower()


# ======================================================================================
# Route-level tests (crypto/validation short-circuits + monkeypatched happy paths)
# ======================================================================================

def _patch_fetch(monkeypatch, chain):
    async def fake_fetch(client, symbol, expiry_ts=None):
        return chain
    monkeypatch.setattr(o, "fetch_chain", fake_fetch)

    async def no_context(client, symbol):
        return {}
    monkeypatch.setattr("app.main.fetch_context", no_context)


def test_puts_crypto_returns_400_not_500():
    with TestClient(app) as client:
        r = client.get("/puts/BTC-USD", params={"cash": 10_000})
        assert r.status_code == 400
        assert "crypto" in r.json()["detail"].lower()
        assert client.get("/puts/ETH-USD", params={"cash": 10_000}).status_code == 400


def test_puts_bad_cash_returns_422_offline():
    """A non-positive / non-finite cash reserve is a 422 BEFORE any chain fetch."""
    with TestClient(app) as client:
        for bad in ("-5", "inf", "nan", "0"):
            r = client.get("/puts/AAPL", params={"cash": bad})
            assert r.status_code == 422, f"cash={bad} -> {r.status_code}"
            assert "cash" in r.json()["detail"].lower()


def test_puts_route_happy_path(monkeypatch):
    now = time.time()
    _patch_fetch(monkeypatch, _wheel_chain(now))
    with TestClient(app) as client:
        r = client.get("/puts/TEST", params={"cash": 30_000, "style": "aggressive"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert set(body) == PUT_TOP_KEYS
        assert [c["profile"] for c in body["candidates"]] == ["aggressive", "balanced", "conservative"]
        assert all(set(c) == PUT_CAND_KEYS for c in body["candidates"])


def test_covered_call_requires_100_shares_returns_400():
    """< 100 shares is a clean 400 (before any network), naming the requirement."""
    with TestClient(app) as client:
        r = client.get("/covered_call/AAPL", params={"shares": 50})
        assert r.status_code == 400
        assert "at least 100 shares" in r.json()["detail"].lower()
        assert client.get("/covered_call/AAPL", params={"shares": 99}).status_code == 400


def test_covered_call_crypto_returns_400_not_500():
    with TestClient(app) as client:
        r = client.get("/covered_call/BTC-USD", params={"shares": 100})
        assert r.status_code == 400
        assert "crypto" in r.json()["detail"].lower()


def test_covered_call_route_happy_path(monkeypatch):
    now = time.time()
    _patch_fetch(monkeypatch, _wheel_chain(now))
    with TestClient(app) as client:
        r = client.get("/covered_call/TEST", params={"shares": 300, "target": 115})
        assert r.status_code == 200, r.text
        body = r.json()
        assert set(body) == CC_TOP_KEYS
        assert body["shares"] == 300 and body["contracts"] == 3
        assert body["candidate"]["strike"] == 115.0
        assert set(body["candidate"]) == CC_CAND_KEYS


# ======================================================================================
# Live route tests (network — skip themselves on failure)
# ======================================================================================

@pytest.mark.live
def test_live_puts_aapl():
    with TestClient(app) as client:
        r = client.get("/puts/AAPL", params={"cash": 50_000, "style": "balanced"})
        if r.status_code != 200:
            pytest.skip(f"live /puts/AAPL unavailable: {r.status_code} {r.text[:120]}")
        body = r.json()
        assert set(body) == PUT_TOP_KEYS, f"missing/extra keys: {set(body) ^ PUT_TOP_KEYS}"
        assert isinstance(body["spot"], (int, float)) and body["spot"] > 0
        assert set(body["expiry"]) == {"ts", "iso", "dte", "rationale"}
        assert body["expiry"]["dte"] > 0
        cands = body["candidates"]
        assert 1 <= len(cands) <= 3
        for c in cands:
            assert set(c) == PUT_CAND_KEYS, f"candidate key drift: {set(c) ^ PUT_CAND_KEYS}"
            assert -1.0 < c["delta"] < 0.0
            assert c["strike"] > 0 and c["limit_price"] > 0
            assert c["contracts"] >= 1
            assert PUT_TICKET_RE.match(c["order_ticket"]), c["order_ticket"]
        # assignment probability decreases as strikes get deeper OTM (higher |delta| -> more likely)
        by = {c["profile"]: c["assignment_prob_pct"] for c in cands}
        if "aggressive" in by and "balanced" in by:
            assert by["aggressive"] >= by["balanced"]
        if "balanced" in by and "conservative" in by:
            assert by["balanced"] >= by["conservative"]


@pytest.mark.live
def test_live_covered_call_aapl():
    with TestClient(app) as client:
        r = client.get("/covered_call/AAPL", params={"shares": 100})
        if r.status_code != 200:
            pytest.skip(f"live /covered_call/AAPL unavailable: {r.status_code} {r.text[:120]}")
        body = r.json()
        assert set(body) == CC_TOP_KEYS, f"missing/extra keys: {set(body) ^ CC_TOP_KEYS}"
        assert body["shares"] == 100 and body["contracts"] == 1
        c = body["candidate"]
        assert set(c) == CC_CAND_KEYS, f"candidate key drift: {set(c) ^ CC_CAND_KEYS}"
        assert 0.0 < c["delta"] < 1.0
        assert c["strike"] > 0 and c["limit_price"] > 0
        assert CC_TICKET_RE.match(c["order_ticket"]), c["order_ticket"]
