"""The long-calls-and-puts arms (app/options_arm.py): contract identity, liquidity gates, sizing,
the rule exits, expiry settlement, fills that conserve cash, the flip rule, the no-model twin and the
score in dollars. Plus the seams with the share ledger: a contract row must be invisible to every
reader that filters on buy/sell, and the list endpoints hide these arms until the app can draw them.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import importlib
import tempfile
import time
from pathlib import Path

import pytest

from app import options, options_arm, sandbox_job, sandbox_store, sandbox_today

TODAY = dt.date.today()


def _occ(root: str, expiry: dt.date, right: str, strike: float) -> str:
    return f"{root}{expiry.strftime('%y%m%d')}{'C' if right == 'call' else 'P'}{int(round(strike * 1000)):08d}"


def _expiry(days: int) -> tuple[dt.date, int]:
    d = TODAY + dt.timedelta(days=days)
    ts = int(dt.datetime.combine(d, dt.time(0, 0), tzinfo=dt.timezone.utc).timestamp())
    return d, ts


def _quote(und: str, right: str, strike: float, *, bid: float, ask: float, delta: float, oi: int = 5000,
           days: int = 62, spot: float = 750.0, state: str = "REGULAR", root: str | None = None) -> dict:
    d, ts = _expiry(days)
    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else None
    return {
        "contract_symbol": _occ(root or und, d, right, strike), "underlying": und, "right": right,
        "strike": strike, "expiry_ts": ts, "expiry": d.isoformat(), "dte": float(days),
        "bid": bid, "ask": ask, "mid": mid, "last_price": mid, "last_trade_epoch": int(time.time()),
        "iv": 0.2, "delta": delta if right == "call" else -abs(delta), "theta": -0.05,
        "open_interest": oi, "volume": 100,
        "spread_pct": round((ask - bid) / mid * 100, 2) if mid else None,
        "breakeven": strike + ask if right == "call" else strike - ask,
        "cost_one": round(ask * 100, 2), "spot": spot, "market_state": state,
    }


def _cand(und: str = "SPY", *, spot: float = 750.0, score: int = 62, is_fund: bool = True,
          earnings_known: bool = True, straddles: bool = False, state: str = "REGULAR",
          atm_call=16.45, otm_call=7.11, atm_put=18.63, otm_put=13.70) -> dict:
    calls = {"atm": _quote(und, "call", 760, bid=16.30, ask=atm_call, delta=0.51, spot=spot, state=state) if atm_call else None,
             "otm": _quote(und, "call", 780, bid=7.03, ask=otm_call, delta=0.33, spot=spot, state=state) if otm_call else None}
    puts = {"atm": _quote(und, "put", 750, bid=18.46, ask=atm_put, delta=0.49, spot=spot, state=state) if atm_put else None,
            "otm": _quote(und, "put", 740, bid=13.56, ask=otm_put, delta=0.33, spot=spot, state=state) if otm_put else None}
    d, _ = _expiry(62)
    return {"symbol": und, "price": spot, "market_state": state, "exposure_group": "US_EQUITY",
            "is_fund": is_fund, "technicals": {"rsi14": 60},
            "directional": {"signal": "bullish" if score >= 55 else "bearish" if score <= 45 else "neutral",
                            "score": score, "reasons": ["MACD positive"]},
            "earnings": {"date": (TODAY + dt.timedelta(days=30)).isoformat() if straddles else None,
                         "known": earnings_known, "straddles": straddles},
            "expiry": d.isoformat(), "dte": 62.0, "atm_iv": 0.18, "iv_rank": None, "expected_move_pct": 7.5,
            "calls": calls, "puts": puts}


def _contract(und: str = "SPY", right: str = "call", *, strike: float = 780, days: int = 62, count: int = 1,
              premium: float = 7.11, spot_at_open: float = 750.0, spy_at_open: float = 750.0,
              opened_days_ago: int = 10) -> dict:
    d, ts = _expiry(days)
    return {"contract_symbol": _occ(und, d, right, strike), "underlying": und, "right": right, "strike": strike,
            "expiry_ts": ts, "expiry": d.isoformat(), "count": count, "avg_premium": premium,
            "cost_total": round(premium * 100 * count, 2), "opened_at": time.time() - opened_days_ago * 86400,
            "opened_date": (TODAY - dt.timedelta(days=opened_days_ago)).isoformat(),
            "spot_at_open": spot_at_open, "spy_at_open": spy_at_open, "last_mark": premium,
            "last_mark_date": (TODAY - dt.timedelta(days=1)).isoformat()}


def _blob(cash: float = 10_000.0, contracts: list[dict] | None = None, **settings) -> dict:
    s = {**sandbox_store.DEFAULT_SETTINGS, "universe": "options", "cash_floor_pct": 50.0,
         "min_conviction_to_trade": 55, "master_enabled": True, **settings}
    return {"arm": "options", "engine": "rules", "cash": cash, "funded_total": cash, "realized_pl_total": 0.0,
            "positions": [], "benchmark": {"symbol": "^GSPC", "shares": cash / 750.0, "cost_basis": cash},
            "settings": s, "options": {"contracts": list(contracts or []), "signals": {}, "pause_ticks_left": 0,
                                       "marks_as_of": None}}


# ------------------------------------------------------------------ identity and settings

def test_defaults_match_the_store_settings():
    """The limits ride in the arm's settings block; the two tables must agree or an arm created
    without overrides would run on numbers the module's own defaults do not describe."""
    for k, v in options_arm.DEFAULTS.items():
        assert sandbox_store.DEFAULT_SETTINGS[k] == v, k


def test_only_the_exact_universe_is_an_options_arm():
    assert options_arm.is_options_arm({"universe": "options"})
    assert not options_arm.is_options_arm({"universe": "etf"})
    assert not options_arm.is_options_arm({"universe": "all"})
    assert not options_arm.is_options_arm(None)
    assert "options" in __import__("app.etf_arm", fromlist=["x"]).UNIVERSES


def test_occ_root_and_standard_contracts_only():
    d, _ = _expiry(62)
    assert options_arm.occ_root(_occ("SPY", d, "call", 780)) == "SPY"
    assert options_arm.is_standard(_occ("SPY", d, "call", 780), "spy")
    assert not options_arm.is_standard(_occ("SPY1", d, "call", 780), "SPY")
    assert options_arm.occ_root("short") == ""


# ------------------------------------------------------------------ quotes and gates

def test_quote_from_a_real_annotated_chain():
    d, ts = _expiry(62)
    call = options.OptionContract(type="call", contract_symbol=_occ("SPY", d, "call", 780), strike=780.0,
                                  expiration=ts, bid=7.03, ask=7.11, last_price=7.05, implied_volatility=0.18,
                                  open_interest=15794, volume=200)
    put = options.OptionContract(type="put", contract_symbol=_occ("SPY", d, "put", 740), strike=740.0,
                                 expiration=ts, bid=13.56, ask=13.70, last_price=13.6, implied_volatility=0.2,
                                 open_interest=14677, volume=100)
    chain = options.OptionChain(symbol="SPY", spot=754.37, currency="USD", market_state="REGULAR",
                                quote_delayed=False, expirations=[{"ts": ts, "iso": d.isoformat()}], strikes=[],
                                expiry=options.ExpiryChain(expiration=ts, expiration_iso=d.isoformat(),
                                                           calls=[call], puts=[put]))
    options.annotate_expiry(chain, now_ts=time.time())
    qc = options_arm.quote_from(call, underlying="SPY", spot=chain.spot, market_state="REGULAR", dte=62.0)
    qp = options_arm.quote_from(put, underlying="SPY", spot=chain.spot, market_state="REGULAR", dte=62.0)
    assert 0 < qc["delta"] < 1 and qp["delta"] < 0
    assert qc["cost_one"] == 711.0 and qc["mid"] == 7.07
    assert 0 < qc["spread_pct"] < 2            # percent, not a fraction
    assert qc["expiry"] == d.isoformat() and qc["breakeven"] == pytest.approx(787.07, abs=0.01)


def test_quote_gate_names_what_binds():
    s = sandbox_store.DEFAULT_SETTINGS
    assert options_arm.quote_gate(_quote("SPY", "call", 780, bid=7.0, ask=7.2, delta=0.33), settings=s, underlying="SPY") is None
    assert "two-sided" in options_arm.quote_gate(_quote("SPY", "call", 780, bid=0, ask=7.2, delta=0.33), settings=s, underlying="SPY")
    assert "open interest" in options_arm.quote_gate(_quote("SPY", "call", 780, bid=7.0, ask=7.2, delta=0.33, oi=100), settings=s, underlying="SPY")
    assert "spread" in options_arm.quote_gate(_quote("F", "call", 11.85, bid=0.63, ask=0.90, delta=0.5), settings=s, underlying="F")
    assert "days to expiry" in options_arm.quote_gate(_quote("SPY", "call", 780, bid=7.0, ask=7.2, delta=0.33, days=20), settings=s, underlying="SPY")
    assert "non-standard" in options_arm.quote_gate(_quote("SPY", "call", 780, bid=7.0, ask=7.2, delta=0.33, root="SPY1"), settings=s, underlying="SPY")


def test_pick_contracts_uses_absolute_delta_for_puts_and_collapses_duplicates():
    s = sandbox_store.DEFAULT_SETTINGS
    puts = [_quote("SPY", "put", 750, bid=18.4, ask=18.6, delta=0.49),
            _quote("SPY", "put", 740, bid=13.5, ask=13.7, delta=0.33),
            _quote("SPY", "put", 700, bid=0, ask=3.0, delta=0.15)]      # one-sided: never offered
    picked = options_arm.pick_contracts(puts, settings=s, underlying="SPY")
    assert picked["atm"]["strike"] == 750 and picked["otm"]["strike"] == 740
    one = options_arm.pick_contracts([puts[0]], settings=s, underlying="SPY")
    assert one["atm"]["strike"] == 750 and one["otm"] is None


# ------------------------------------------------------------------ sizing

def test_size_open_prefers_the_atm_contract_when_it_fits_else_the_cheaper_else_nothing():
    s = sandbox_store.DEFAULT_SETTINGS | {"cash_floor_pct": 50.0}
    c = _cand()                       # atm call $1,645, otm call $711; cap at $10,000 = $1,500
    q, n, why = options_arm.size_open(c, "call", equity=10_000, cash=10_000, at_risk=0, conviction=100, settings=s)
    assert q["strike"] == 780 and n == 2 and why is None       # floor(1500 / 711)
    q, n, _ = options_arm.size_open(c, "call", equity=10_000, cash=10_000, at_risk=0, conviction=55, settings=s)
    assert n == 1                                              # floor(825 / 711)
    q, n, _ = options_arm.size_open(c, "call", equity=12_000, cash=12_000, at_risk=0, conviction=100, settings=s)
    assert q["strike"] == 760 and n == 1                       # cap $1,800 now admits the 0.50 at $1,645
    c2 = _cand(otm_call=None)
    q, n, why = options_arm.size_open(c2, "call", equity=10_000, cash=10_000, at_risk=0, conviction=100, settings=s)
    assert q is None and n == 0 and "per-contract cap of $1,500" in why


def test_size_open_respects_the_at_risk_cap_and_the_cash_floor():
    s = sandbox_store.DEFAULT_SETTINGS | {"cash_floor_pct": 50.0}
    c = _cand()
    q, n, why = options_arm.size_open(c, "call", equity=10_000, cash=7_100, at_risk=2_900, conviction=100, settings=s)
    assert q is None and "left under the at-risk cap" in why       # $100 of room, one costs $711
    q, n, why = options_arm.size_open(c, "call", equity=10_000, cash=7_100, at_risk=3_000, conviction=100, settings=s)
    assert q is None and "at its 30% cap" in why
    q, n, why = options_arm.size_open(c, "call", equity=10_000, cash=5_000, at_risk=0, conviction=100, settings=s)
    assert q is None and "cash floor" in why
    q, n, why = options_arm.size_open(_cand(atm_put=None, otm_put=None), "put", equity=10_000, cash=10_000,
                                      at_risk=0, conviction=90, settings=s)
    assert q is None and "no liquid put" in why


# ------------------------------------------------------------------ exits

def test_rule_exits_days_to_expiry_fires_without_a_mark_stop_and_target_need_a_fresh_one():
    s = sandbox_store.DEFAULT_SETTINGS
    now = time.time()
    near = _contract("SPY", days=20)
    stopped = _contract("QQQ", days=62, premium=10.0)
    winner = _contract("IWM", days=62, premium=5.0)
    stale = _contract("AAPL", days=62, premium=10.0)
    stale["last_mark"] = 1.0                       # looks like a stop, but no fresh mark today
    marks = {stopped["contract_symbol"]: _quote("QQQ", "call", 780, bid=4.9, ask=5.1, delta=0.3),
             winner["contract_symbol"]: _quote("IWM", "call", 780, bid=10.4, ask=10.6, delta=0.7)}
    out = options_arm.rule_exits([near, stopped, winner, stale], marks, now_ts=now, settings=s)
    rules = {o["underlying"]: o["rule"] for o in out}
    assert rules == {"SPY": "dte", "QQQ": "stop", "IWM": "target"}
    assert all(o["action"] == "close" and o["symbol"] == o["underlying"] for o in out)


def test_settle_expired_holds_without_a_close_and_settles_intrinsic_with_one():
    b = _blob(cash=1_000.0)
    itm = _contract("SPY", "call", strike=780, days=-3, premium=7.11)     # expired three days ago
    otm = _contract("QQQ", "put", strike=600, days=-3, premium=5.0)
    unknown = _contract("IWM", "call", strike=280, days=-3, premium=3.0)
    b["options"]["contracts"] = [itm, otm, unknown]
    rows = options_arm.settle_expired(b, {itm["contract_symbol"]: 790.0, otm["contract_symbol"]: 650.0},
                                      now_ts=time.time(), spy_price=760.0)
    by = {r["underlying"]: r for r in rows}
    assert by["SPY"]["side"] == "settled" and by["SPY"]["gross"] == 1000.0 and by["SPY"]["realized_pl"] == pytest.approx(1000 - 711)
    assert by["QQQ"]["side"] == "expired" and by["QQQ"]["gross"] == 0.0 and by["QQQ"]["realized_pl"] == -500.0
    assert by["IWM"]["status"] == "skipped" and "awaiting" in by["IWM"]["skip_reason"]
    assert [c["underlying"] for c in b["options"]["contracts"]] == ["IWM"]
    assert b["cash"] == 2_000.0 and b["realized_pl_total"] == pytest.approx(289 - 500)
    assert by["SPY"]["index_alt_usd"] == pytest.approx(711 * (760 / 750 - 1), abs=0.01)
    assert by["SPY"]["direction_right"] is True and by["QQQ"]["direction_right"] is True


# ------------------------------------------------------------------ fills

def _fill(b, orders, cands, marks=None, spy=750.0, today=None):
    return options_arm.fill(b, orders, candidates={c["symbol"]: c for c in cands}, marks=marks or {},
                            spy_price=spy, now_ts=time.time(), source="rules_tick", settings=b["settings"],
                            today=today or TODAY.isoformat())


def test_fill_opens_at_the_ask_and_conserves_cash():
    b = _blob()
    nb, filled, skipped = _fill(b, [{"symbol": "SPY", "underlying": "SPY", "right": "call", "action": "open",
                                     "conviction": 70, "reason": "up"}], [_cand()])
    assert len(filled) == 1 and not skipped
    r = filled[0]
    assert r["side"] == "buy_to_open" and r["instrument"] == "option" and r["shares"] == 1
    assert r["price"] == 7.11 and r["gross"] == 711.0 and nb["cash"] == 10_000 - 711
    c = nb["options"]["contracts"][0]
    assert c["strike"] == 780 and c["avg_premium"] == 7.11 and c["cost_total"] == 711.0
    assert c["spot_at_open"] == 750.0 and c["spy_at_open"] == 750.0 and c["last_mark"] == 7.07
    assert nb["options"]["signals"]["SPY"] == {"right": "call", "date": TODAY.isoformat()}
    assert b["cash"] == 10_000 and b["options"]["contracts"] == []        # the input was not mutated


def test_fill_closes_at_the_bid_first_then_opens_and_scores_the_close():
    held = _contract("SPY", "call", strike=780, premium=7.11, spot_at_open=750.0, spy_at_open=740.0)
    b = _blob(contracts=[held])
    marks = {held["contract_symbol"]: _quote("SPY", "call", 780, bid=12.0, ask=12.3, delta=0.6, spot=775.0)}
    orders = [{"symbol": "SPY", "underlying": "SPY", "right": "call", "action": "close", "rule": "target",
               "reason": "up 69% — target"},
              {"symbol": "QQQ", "underlying": "QQQ", "right": "put", "action": "open", "conviction": 80, "reason": "down"}]
    nb, filled, skipped = _fill(b, orders, [_cand("QQQ", spot=680.0, score=38)], marks=marks, spy=755.0)
    assert [r["side"] for r in filled] == ["sell_to_close", "buy_to_open"]
    close = filled[0]
    assert close["price"] == 12.0 and close["gross"] == 1200.0 and close["realized_pl"] == pytest.approx(1200 - 711)
    assert close["direction_right"] is True and close["spot_at_close"] == 775.0
    assert close["index_alt_usd"] == pytest.approx(711 * (755 / 740 - 1), abs=0.01)
    assert nb["realized_pl_total"] == pytest.approx(489.0)
    assert nb["cash"] == pytest.approx(10_000 + 1200 - filled[1]["gross"])
    assert [c["underlying"] for c in nb["options"]["contracts"]] == ["QQQ"]


def test_fill_refuses_a_second_contract_on_one_underlying_and_counts_the_caps():
    b = _blob(contracts=[_contract("SPY")], opt_max_new_per_tick=2, opt_max_new_week_pct=0)
    orders = [{"symbol": "SPY", "underlying": "SPY", "right": "put", "action": "open", "conviction": 90, "reason": "x"},
              {"symbol": "QQQ", "underlying": "QQQ", "right": "call", "action": "open", "conviction": 80, "reason": "x"},
              {"symbol": "IWM", "underlying": "IWM", "right": "call", "action": "open", "conviction": 70, "reason": "x"},
              {"symbol": "AAPL", "underlying": "AAPL", "right": "call", "action": "open", "conviction": 60, "reason": "x"}]
    cands = [_cand("QQQ", spot=680), _cand("IWM", spot=277), _cand("AAPL", spot=356, is_fund=False)]
    nb, filled, skipped = _fill(b, orders, cands)
    assert [r["underlying"] for r in filled] == ["QQQ", "IWM"]            # two new per run
    why = {r["underlying"]: r["skip_reason"] for r in skipped}
    assert "one per underlying" in why["SPY"] and "per-run cap" in why["AAPL"]


def test_fill_refuses_stale_quotes_unknown_earnings_and_straddled_earnings_but_not_funds():
    b = _blob()
    orders = [{"symbol": s, "underlying": s, "right": "call", "action": "open", "conviction": 80, "reason": "x"}
              for s in ("SPY", "AAPL", "NVDA", "IWM")]
    cands = [_cand("SPY", state="CLOSED"), _cand("AAPL", spot=356, is_fund=False, earnings_known=False),
             _cand("NVDA", spot=163, is_fund=False, straddles=True),
             _cand("IWM", spot=277, is_fund=True, earnings_known=False)]
    nb, filled, skipped = _fill(b, orders, cands)
    assert [r["underlying"] for r in filled] == ["IWM"]
    why = {r["underlying"]: r["skip_reason"] for r in skipped}
    assert "not live" in why["SPY"] and "earnings date unknown" in why["AAPL"] and "spans earnings" in why["NVDA"]


def test_a_direction_flip_waits_one_run_then_trades():
    yesterday = (TODAY - dt.timedelta(days=1)).isoformat()
    b = _blob()
    b["options"]["signals"] = {"SPY": {"right": "call", "date": yesterday}}
    put = [{"symbol": "SPY", "underlying": "SPY", "right": "put", "action": "open", "conviction": 80, "reason": "x"}]
    nb, filled, skipped = _fill(b, put, [_cand("SPY", score=40)])
    assert not filled and "flipped from call to put" in skipped[0]["skip_reason"]
    assert nb["options"]["signals"]["SPY"]["right"] == "put"             # today's read is recorded
    tomorrow = (TODAY + dt.timedelta(days=1)).isoformat()
    nb2, filled2, skipped2 = _fill(nb, put, [_cand("SPY", score=40)], today=tomorrow)
    assert len(filled2) == 1 and not skipped2
    # the first read of a name is never a flip; the same direction again is never a flip
    nb3, f3, s3 = _fill(_blob(), put, [_cand("SPY", score=40)])
    assert len(f3) == 1
    b4 = _blob(); b4["options"]["signals"] = {"SPY": {"right": "put", "date": yesterday}}
    nb4, f4, s4 = _fill(b4, put, [_cand("SPY", score=40)])
    assert len(f4) == 1
    # a flip older than the window is a fresh read
    b5 = _blob(); b5["options"]["signals"] = {"SPY": {"right": "call", "date": (TODAY - dt.timedelta(days=40)).isoformat()}}
    nb5, f5, s5 = _fill(b5, put, [_cand("SPY", score=40)])
    assert len(f5) == 1


def test_a_close_with_no_bid_is_held_and_a_pause_blocks_opens_not_closes():
    held = _contract("SPY")
    b = _blob(contracts=[held])
    b["options"]["pause_ticks_left"] = 3
    marks = {held["contract_symbol"]: _quote("SPY", "call", 780, bid=0, ask=5.0, delta=0.3)}
    orders = [{"symbol": "SPY", "underlying": "SPY", "action": "close", "right": "call", "reason": "stop"},
              {"symbol": "QQQ", "underlying": "QQQ", "right": "call", "action": "open", "conviction": 80, "reason": "x"}]
    nb, filled, skipped = _fill(b, orders, [_cand("QQQ", spot=680)], marks=marks)
    assert not filled
    why = {r["underlying"]: r["skip_reason"] for r in skipped}
    assert "no bid" in why["SPY"] and "paused" in why["QQQ"]
    assert len(nb["options"]["contracts"]) == 1
    marks2 = {held["contract_symbol"]: _quote("SPY", "call", 780, bid=4.9, ask=5.0, delta=0.3)}
    nb2, filled2, _ = _fill(b, orders[:1], [], marks=marks2)
    assert filled2[0]["side"] == "sell_to_close" and nb2["cash"] == 10_000 + 490.0


def test_fill_rejects_low_conviction_and_exclusions():
    b = _blob()
    orders = [{"symbol": "SPY", "underlying": "SPY", "right": "call", "action": "open", "conviction": 40, "reason": "x"},
              {"symbol": "QQQ", "underlying": "QQQ", "right": "call", "action": "open", "conviction": 90, "reason": "x"}]
    nb, filled, skipped = options_arm.fill(b, orders, candidates={"SPY": _cand(), "QQQ": _cand("QQQ")}, marks={},
                                           spy_price=750.0, now_ts=time.time(), source="haiku_tick",
                                           settings=b["settings"], exclude={"QQQ"})
    assert not filled
    why = {r["underlying"]: r["skip_reason"] for r in skipped}
    assert "under the 55 floor" in why["SPY"] and "exclusion" in why["QQQ"]


# ------------------------------------------------------------------ the no-model twin

def test_rules_decision_reads_calls_above_55_puts_below_45_strongest_first():
    s = sandbox_store.DEFAULT_SETTINGS
    cands = [_cand("SPY", score=60), _cand("QQQ", score=50), _cand("IWM", score=30),
             _cand("AAPL", score=80, atm_call=None, otm_call=None)]          # bullish, but no liquid call
    d = options_arm.rules_decision(cands, settings=s)
    assert [(o["underlying"], o["right"], o["conviction"]) for o in d["orders"]] == [("IWM", "put", 70), ("SPY", "call", 60)]
    assert "1 call, 1 put" in d["posture"]
    assert options_arm.rules_decision([_cand("SPY", score=50)], settings=s)["orders"] == []


def test_rules_twin_skips_names_the_ledger_would_refuse_instead_of_spending_its_slots():
    """Measured 2026-11-15: the twin's two slots went to AAPL (score 85) and SOFI (16), both stocks
    with a failed earnings lookup; the ledger refused both and the run filled nothing while QQQ and
    SPY puts at 81 and 77 sat unproposed."""
    s = sandbox_store.DEFAULT_SETTINGS
    cands = [_cand("AAPL", score=85, is_fund=False, earnings_known=False),
             _cand("SOFI", score=16, is_fund=False, earnings_known=False),
             _cand("NVDA", score=18, is_fund=False, straddles=True),
             _cand("QQQ", score=19), _cand("SPY", score=23), _cand("IWM", score=35, state="CLOSED")]
    d = options_arm.rules_decision(cands, settings=s)
    assert [(o["underlying"], o["right"], o["conviction"]) for o in d["orders"]] == [("QQQ", "put", 81), ("SPY", "put", 77)]
    assert options_arm.entry_block(_cand("SPY")) is None
    assert "not live" in options_arm.entry_block(_cand("SPY", state="CLOSED"))


def test_rank_underlyings_index_first_then_strongest_reads_no_crypto_no_penny():
    rows = [{"symbol": "BTC-USD", "price": 90000, "technicals": {"macd_hist": 1}},
            {"symbol": "F", "price": 4.5, "technicals": {"macd_hist": 1}},
            {"symbol": "AAPL", "price": 356, "technicals": {"golden_cross": True, "macd_hist": 1, "rel_strength_3mo_vs_benchmark": 0.1}},
            {"symbol": "XYZ", "price": 50, "technicals": {}},
            {"symbol": "SPY", "price": 754, "technicals": {}}]
    out = [r["symbol"] for r in options_arm.rank_underlyings(rows, limit=3)]
    assert out == ["SPY", "AAPL", "XYZ"]


# ------------------------------------------------------------------ score, pause, payload

def test_scorecard_is_dollars_and_counts_right_but_lost():
    trades = [
        {"instrument": "option", "status": "filled", "side": "sell_to_close", "cost_total": 711.0, "realized_pl": 489.0,
         "index_alt_usd": 14.0, "direction_right": True},
        {"instrument": "option", "status": "filled", "side": "expired", "cost_total": 500.0, "realized_pl": -500.0,
         "index_alt_usd": -3.0, "direction_right": True},
        {"instrument": "option", "status": "filled", "side": "settled", "cost_total": 300.0, "realized_pl": 50.0,
         "direction_right": False},
        {"instrument": "option", "status": "filled", "side": "buy_to_open", "gross": 711.0},     # an open: not scored
        {"status": "filled", "side": "sell", "realized_pl": 999.0},                               # a share row: ignored
    ]
    sc = options_arm.scorecard(trades, [_contract()])
    assert sc == {"closed": 3, "premium_deployed_usd": 1511.0, "pl_usd": 39.0, "index_alt_usd": 11.0, "index_alt_n": 2,
                  "direction_right": 2, "direction_right_but_lost": 1, "open": 1, "premium_at_risk_usd": 711.0}
    assert "win_rate" not in sc


def test_update_pause_starts_on_a_deep_drawdown_and_counts_down():
    b = _blob()
    assert options_arm.update_pause(b, current_drawdown_pct=None, settings=b["settings"]) is None
    assert options_arm.update_pause(b, current_drawdown_pct=-4.0, settings=b["settings"]) is None
    note = options_arm.update_pause(b, current_drawdown_pct=-12.0, settings=b["settings"])
    assert "paused for 5 runs" in note and b["options"]["pause_ticks_left"] == 5
    for left in (4, 3, 2, 1, 0):
        assert options_arm.update_pause(b, current_drawdown_pct=-12.0, settings=b["settings"]) is None
        assert b["options"]["pause_ticks_left"] == left
    # the same drawdown does not re-arm the pause ...
    for _ in range(8):
        assert options_arm.update_pause(b, current_drawdown_pct=-13.0, settings=b["settings"]) is None
        assert b["options"]["pause_ticks_left"] == 0
    # ... a recovery to the peak and a NEW drawdown does
    assert options_arm.update_pause(b, current_drawdown_pct=-0.5, settings=b["settings"]) is None
    assert options_arm.update_pause(b, current_drawdown_pct=-11.0, settings=b["settings"]) is not None


def test_analyst_payload_carries_limits_rows_and_unknown_iv_rank_as_null():
    held = _contract("SPY", premium=7.11)
    b = _blob(contracts=[held])
    marks = {held["contract_symbol"]: _quote("SPY", "call", 780, bid=8.0, ask=8.2, delta=0.4, spot=760.0)}
    p = options_arm.analyst_payload(b, [_cand("QQQ", spot=680)], marks, equity=10_000 + 810, cash=10_000,
                                    spy_price=755.0, now_ts=time.time(), settings=b["settings"],
                                    scorecard={"closed": 0})
    assert p["limits"]["per_contract_usd"] == pytest.approx(0.15 * 10_810, abs=0.01)
    assert p["limits"]["exits_by_code"] == {"days_to_expiry": 21, "stop_pct": -50.0, "target_pct": 100.0}
    oc = p["open_contracts"][0]
    assert oc["mark"] == 8.1 and oc["mark_stale"] is False and oc["pl_pct"] == pytest.approx(13.9, abs=0.1)
    c = p["candidates"][0]
    assert c["iv_rank"] is None and c["call"]["cost_one"] == 1645.0 and c["call_cheaper"]["cost_one"] == 711.0
    assert "your_closed_contracts_in_dollars" not in p and "new_entries_paused_runs_left" not in p
    assert "birth_date" not in p and "settings" not in p


def test_recent_activity_keeps_only_option_rows_inside_the_window():
    today = TODAY.isoformat()
    old = (TODAY - dt.timedelta(days=20)).isoformat()
    trades = [{"ts": 3, "date": today, "instrument": "option", "display": "SPY $780 call", "side": "buy_to_open",
               "status": "filled", "shares": 1, "price": 7.11, "gross": 711.0, "reason": "up"},
              {"ts": 2, "date": today, "instrument": "option", "side": "buy_to_open", "status": "skipped",
               "skip_reason": "cap"},
              {"ts": 1, "date": old, "instrument": "option", "side": "buy_to_open", "status": "filled"},
              {"ts": 4, "date": today, "side": "deposit", "status": "filled"}]
    out = options_arm.recent_activity(trades, today=today)
    assert [r.get("what") for r in out] == ["SPY $780 call", None]
    assert out[1]["why_not"] == "cap"


# ------------------------------------------------------------------ the seams with the share ledger

def test_contract_rows_are_invisible_to_the_sold_symbols_quote_batch():
    rows = [{"status": "filled", "side": "sell_to_close", "symbol": "SPY270115C00780000", "date": "2026-11-20"},
            {"status": "filled", "side": "settled", "symbol": "QQQ270115P00650000", "date": "2026-11-20"},
            {"status": "filled", "side": "sell", "symbol": "GLD", "date": "2026-11-19"}]
    assert sandbox_job.sold_symbols([rows]) == ["GLD"]


def test_today_card_folds_option_sides_and_shows_the_display_name():
    arms = [{"arm": "options", "label": "Options", "engine": "llm", "universe": "options", "enabled": True,
             "last_tick_date": "2026-11-20", "last_posture": "one call"}]
    trades = {"options": [
        {"date": "2026-11-20", "status": "filled", "side": "buy_to_open", "instrument": "option",
         "symbol": "SPY270115C00780000", "display": "SPY $780 call exp 2027-01-15", "shares": 1, "price": 7.11, "gross": 711.0},
        {"date": "2026-11-20", "status": "filled", "side": "sell_to_close", "instrument": "option",
         "symbol": "QQQ270115P00650000", "display": "QQQ $650 put exp 2027-01-15", "shares": 1, "price": 20.0, "gross": 2000.0},
        {"date": "2026-11-20", "status": "filled", "side": "deposit", "symbol": "CASH", "gross": 500.0}]}
    out = sandbox_today.summarize(arms, trades, {"options": 1.0})
    a = out["arms"][0]
    assert [(o["side"], o["symbol"], o["instrument"]) for o in a["filled"]] == [
        ("sell", "QQQ $650 PUT EXP 2027-01-15", "option"), ("buy", "SPY $780 CALL EXP 2027-01-15", "option")]
    assert out["totals"] == {"buys": 1, "sells": 1, "skipped": 0, "bought": 711.0, "sold": 2000.0}


@pytest.fixture()
def store(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setenv("SIGNALS_DATA_DIR", d)
    from app import sandbox_store as s
    importlib.reload(s)
    s._DATA_DIR = Path(d)
    s._cache = {s.MAIN_ARM: s._load(s.MAIN_ARM)}
    return s


def test_list_endpoints_hide_options_arms_unless_asked(store):
    from app import main
    store.create_arm("options", engine="llm", label="Options", settings={"universe": "options"})
    store.create_arm("plain", engine="rules", label="Plain")
    assert "options" not in main._list_arms(False) and "plain" in main._list_arms(False)
    assert "options" in main._list_arms(True)


def test_options_arm_cycle_without_a_snapshot_marks_nav_and_opens_nothing(store, monkeypatch):
    from app import main
    store.create_arm("options-rules", engine="rules", label="Options (rules)",
                     settings={"universe": "options", "master_enabled": True, "monthly_deposit": 0.0})
    b = store.get("options-rules")
    b["cash"] = 10_000.0; b["funded_total"] = 10_000.0
    b["benchmark"] = {"symbol": "^GSPC", "shares": 10_000 / 750.0, "cost_basis": 10_000.0}
    store.save(b, "options-rules")
    monkeypatch.setattr(main.sandbox_job, "tick_gate", lambda blob, **kw: (True, "ok"))
    out = asyncio.run(main._run_options_arm("options-rules", now=sandbox_job.now_et(), spy_price=755.0,
                                            macro_block=None, force=True, opt_ctx=None))
    assert out["status"] == "ok" and out["orders_filled"] == []
    assert out["nav"]["equity"] == pytest.approx(10_000.0, abs=5.0)   # cash, plus a day of interest at most
    assert out["nav"]["num_positions"] == 0
    assert any("chains unavailable" in w for w in out["warnings"])
    assert "No chain snapshot" in out["posture"]
    assert store.get("options-rules")["last_tick_date"] == sandbox_job.today_et_str()


def test_options_arm_cycle_with_a_snapshot_fills_the_rules_pick_and_writes_rows(store, monkeypatch):
    from app import main
    store.create_arm("options-rules", engine="rules", label="Options (rules)",
                     settings={"universe": "options", "master_enabled": True, "monthly_deposit": 0.0,
                               "cash_floor_pct": 50.0, "cash_apy_pct": 0.0})
    b = store.get("options-rules")
    b["cash"] = 10_000.0; b["funded_total"] = 10_000.0
    b["benchmark"] = {"symbol": "^GSPC", "shares": 10_000 / 750.0, "cost_basis": 10_000.0}
    store.save(b, "options-rules")
    monkeypatch.setattr(main.sandbox_job, "tick_gate", lambda blob, **kw: (True, "ok"))
    cands = [_cand("SPY", score=64), _cand("QQQ", spot=680, score=50)]
    ctx = {"marks": {}, "candidates": cands, "by_underlying": {c["symbol"]: c for c in cands}, "errors": [], "as_of": time.time()}
    out = asyncio.run(main._run_options_arm("options-rules", now=sandbox_job.now_et(), spy_price=755.0,
                                            macro_block=None, force=True, opt_ctx=ctx))
    assert out["status"] == "ok" and len(out["orders_filled"]) == 1
    r = out["orders_filled"][0]
    assert r["underlying"] == "SPY" and r["right"] == "call" and r["side"] == "buy_to_open"
    # conviction 64 of a $1,500 cap = $960 -> one $711 contract; NAV marks it at the mid
    assert r["shares"] == 1 and r["gross"] == 711.0
    assert out["nav"]["cash"] == 10_000 - 711 and out["nav"]["positions_value"] == pytest.approx(707.0)
    assert out["nav"]["num_positions"] == 1
    saved = store.get("options-rules")
    assert saved["positions"] == [] and len(saved["options"]["contracts"]) == 1
    rows = store.read_trades(10, "options-rules")
    assert any(t.get("side") == "buy_to_open" for t in rows)
    assert all(t.get("side") not in ("buy", "sell") for t in rows if t.get("instrument") == "option")


def test_the_tick_snapshot_never_raises_whatever_fails(store, monkeypatch):
    """It runs inside main's tick before the eleven comparison arms; an exception would cost all of
    them their day. Both the chain fetch and the arm filter itself are covered."""
    from app import main
    store.create_arm("options", engine="llm", label="Options",
                     settings={"universe": "options", "master_enabled": True})
    monkeypatch.setattr(main.sandbox_job, "tick_gate", lambda blob, **kw: (True, "ok"))

    async def boom(*a, **k):
        raise RuntimeError("yahoo down")
    monkeypatch.setattr(main, "_options_context", boom)
    out = asyncio.run(main._maybe_options_context(now=sandbox_job.now_et(), force=True, candidates=[],
                                                  held_rows=[], bench_series=None))
    assert out is None

    def broken_list():
        raise OSError("disk")
    monkeypatch.setattr(main.sandbox_store, "list_arms", broken_list)
    out = asyncio.run(main._maybe_options_context(now=sandbox_job.now_et(), force=True, candidates=[],
                                                  held_rows=[], bench_series=None))
    assert out is None


def test_no_options_arm_means_no_chain_fetch(store, monkeypatch):
    from app import main
    called = []

    async def spy(*a, **k):
        called.append(1)
        return {}
    monkeypatch.setattr(main, "_options_context", spy)
    out = asyncio.run(main._maybe_options_context(now=sandbox_job.now_et(), force=True, candidates=[],
                                                  held_rows=[], bench_series=None))
    assert out is None and called == []


# ------------------------------------------------------------------ weekly pacing (2026-11-17)

def _open(und, conv=93):
    return {"symbol": und, "underlying": und, "right": "call", "action": "open", "conviction": conv, "reason": "x"}


def test_weekly_budget_lets_one_contract_through_then_refuses_with_the_date_room_returns():
    """The 2026-11-16/17 case: WELL ($1,410) on Monday, DUK ($1,230) on Tuesday. With a 15% weekly
    budget Monday fills and Tuesday is refused, naming the spend and when room comes back."""
    b = _blob(cash_floor_pct=0.0)
    mon = TODAY.isoformat()
    nb, filled, _ = _fill(b, [_open("QQQ")], [_cand("QQQ", spot=680, atm_call=14.10, otm_call=6.0)], today=mon)
    assert [r["gross"] for r in filled] == [1410.0]
    assert nb["options"]["spend_log"] == [{"date": mon, "cost": 1410.0}]
    tue = (TODAY + dt.timedelta(days=1)).isoformat()
    nb2, filled2, skipped2 = _fill(nb, [_open("IWM")], [_cand("IWM", spot=277, atm_call=12.30, otm_call=9.0)], today=tue)
    assert not filled2
    why = skipped2[0]["skip_reason"]
    assert "weekly options budget" in why and "$1,410 of $1,5" in why
    assert (TODAY + dt.timedelta(days=7)).isoformat() in why
    # seven days on, Monday's spend no longer counts and the same read fills
    later = (TODAY + dt.timedelta(days=7)).isoformat()
    nb3, filled3, _ = _fill(nb2, [_open("IWM")], [_cand("IWM", spot=277, atm_call=12.30, otm_call=9.0)], today=later)
    assert len(filled3) == 1
    assert [r["date"] for r in nb3["options"]["spend_log"]] == [later]   # pruned


def test_weekly_budget_counts_a_contract_closed_inside_the_week():
    """Spending is spending: a stop-out on Tuesday does not hand Monday's money back to the week."""
    st = options_arm.state({"options": {"contracts": [], "spend_log": [{"date": TODAY.isoformat(), "cost": 1400.0}]}})
    wk = options_arm.week_budget(st, equity=10_000, settings={}, today=TODAY.isoformat())
    assert wk["spent_usd"] == 1400.0 and wk["room_usd"] == 100.0


def test_weekly_budget_seeds_from_held_contracts_and_can_be_switched_off():
    held = _contract("WELL") | {"opened_date": TODAY.isoformat(), "cost_total": 1410.0}
    st = options_arm.state({"options": {"contracts": [held]}})
    assert st["spend_log"] == [{"date": TODAY.isoformat(), "cost": 1410.0}]
    assert options_arm.week_budget(st, equity=10_000, settings={"opt_max_new_week_pct": 0}, today=TODAY.isoformat()) is None


def test_one_new_contract_a_run_by_default_and_the_rules_twin_offers_spares():
    assert options_arm.DEFAULTS["opt_max_new_per_tick"] == 1
    b = _blob(cash_floor_pct=0.0, opt_max_new_week_pct=0)
    nb, filled, skipped = _fill(b, [_open("QQQ"), _open("IWM", 80)], [_cand("QQQ", spot=680), _cand("IWM", spot=277)])
    assert [r["underlying"] for r in filled] == ["QQQ"] and "per-run cap" in skipped[0]["skip_reason"]
    cands = [_cand(s, score=sc) for s, sc in (("A", 90), ("B", 80), ("C", 70), ("D", 60))]
    orders = options_arm.rules_decision(cands, settings=sandbox_store.DEFAULT_SETTINGS)["orders"]
    assert [o["symbol"] for o in orders] == ["A", "B", "C"]
