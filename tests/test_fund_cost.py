"""FC-1 — fund fees and the near-copies each fund is compared with (app/fund_cost.py)."""
from __future__ import annotations

import asyncio
import importlib

import pytest
from fastapi.testclient import TestClient

from app import fund_cost as fc

SAVED = {"SPY": 0.095, "SPYM": 0.020, "VOO": 0.030, "FXAIX": 9.99}
SAVED_ON = "2026-08-27"


@pytest.fixture(autouse=True)
def _cold_cache():
    fc._cache.clear()
    yield
    fc._cache.clear()


def _yahoo(rows: dict[str, dict]):
    """A stand-in for market_now.fetch_quotes that answers from [rows] and records every ask."""
    asked: list[list[str]] = []

    async def fetch(_client, symbols):
        asked.append(list(symbols))
        return {s: rows[s] for s in symbols if s in rows}
    fetch.asked = asked
    return fetch


def _down():
    async def fetch(_client, _symbols):
        raise RuntimeError("Yahoo quote fetch failed")
    return fetch


def _etf(name, fee):
    return {"long_name": name, "quote_type": "ETF", "expense_ratio_pct": fee}


def _mf(name, fee):
    return {"long_name": name, "quote_type": "MUTUALFUND", "expense_ratio_pct": fee}


LIVE = {
    "VOO": _etf("Vanguard S&P 500 ETF", 0.03),
    "IVV": _etf("iShares Core S&P 500 ETF", 0.03),
    # What Yahoo really says about SPYM: an EQUITY with no fee.
    "SPYM": {"long_name": "State Street SPDR Portfolio S&P 500 ETF", "quote_type": "EQUITY",
             "expense_ratio_pct": None},
    "SPY": _etf("State Street SPDR S&P 500 ETF Trust", 0.0945),
    "FXAIX": _mf("Fidelity 500 Index", 0.015),
    "FNILX": _mf("Fidelity ZERO Large Cap Index", 0.0),
    "AAPL": {"long_name": "Apple Inc.", "quote_type": "EQUITY", "expense_ratio_pct": None},
    "ETHE": _etf("Grayscale Ethereum Staking ETF", 2.5),
}


def _run(symbols, fetch, now=1_000.0, saved=SAVED):
    return asyncio.run(fc.lookup(None, symbols, saved=saved, saved_as_of=SAVED_ON, fetch=fetch, now=now))


# --- the groups themselves ----------------------------------------------------------------------

def test_no_fund_sits_in_two_groups():
    seen: dict[str, str] = {}
    for g in fc.GROUPS:
        assert len(g.members) >= 2, f"{g.id} compares a fund with nothing"
        for m in g.members:
            assert m not in seen, f"{m} is in both {seen[m]} and {g.id}"
            seen[m] = g.id


def test_every_flagged_fund_is_actually_compared():
    grouped = {m for g in fc.GROUPS for m in g.members}
    # A flag on a fund no group contains would never reach the app — dead data that looks live.
    assert fc.FIDELITY <= grouped
    assert fc.MUTUAL_FUNDS <= fc.FIDELITY      # every mutual fund offered is Fidelity's own
    assert fc.FIDELITY_ONLY <= fc.MUTUAL_FUNDS


def test_look_alikes_that_failed_the_measurement_stay_apart():
    # Each of these LOOKS like the other and was measured not to be (see the module docstring).
    for a, b in (("VEA", "IEFA"), ("VWO", "IEMG"), ("IEMG", "EEM"), ("IEMG", "FPADX"), ("VGT", "XLK"),
                 ("SCHD", "VYM"), ("SPMO", "MTUM"), ("IWF", "VUG"), ("QQQ", "ONEQ")):
        ga, gb = fc.group_of(a), fc.group_of(b)
        assert ga is None or gb is None or ga.id != gb.id, f"{a} and {b} grouped as the same thing"


# --- fees ---------------------------------------------------------------------------------------

def test_live_fees_and_the_group_cheapest_first_with_itself_included():
    out = _run(["SPY"], _yahoo(LIVE))
    assert out["live"] is True
    spy = out["funds"]["SPY"]
    assert spy["expense_ratio_pct"] == 0.0945 and spy["fee_source"] == "yahoo"
    assert spy["fee_checked_at"] == 1_000.0
    assert spy["kind"] == "etf"
    order = [r["symbol"] for r in spy["group"]["funds"]]
    assert order == ["FNILX", "FXAIX", "SPYM", "IVV", "VOO", "SPY"]
    assert spy["group"]["label"].startswith("the S&P 500")


def test_a_zero_fee_is_zero_not_unknown():
    out = _run(["FNILX"], _yahoo(LIVE))
    z = out["funds"]["FNILX"]
    assert z["expense_ratio_pct"] == 0.0 and z["fee_source"] == "yahoo"
    assert z["kind"] == "mutual_fund" and z["fidelity"] and z["fidelity_only"]


def test_yahoo_calls_spym_an_equity_with_no_fee_and_the_issuer_figure_fills_it():
    spym = _run(["SPYM"], _yahoo(LIVE))["funds"]["SPYM"]
    assert spym["kind"] == "etf"                 # grouped fund, whatever Yahoo's quoteType says
    assert spym["expense_ratio_pct"] == 0.02
    assert spym["fee_source"] == "issuer" and spym["fee_dated"] == fc.ISSUER_FEES_CHECKED
    assert spym["fee_checked_at"] is None


def test_yahoo_down_falls_back_to_the_saved_table_with_its_date():
    spy = _run(["SPY"], _down())["funds"]["SPY"]
    assert spy["expense_ratio_pct"] == 0.095
    assert spy["fee_source"] == "saved" and spy["fee_dated"] == SAVED_ON


def test_an_etf_listed_at_zero_is_not_believed():
    rows = {
        "HODL": _etf("VanEck Bitcoin ETF", 0.0),      # waiver over; the issuer says 0.20%
        "ZZZZ": _etf("Some Other ETF", 0.0),           # no issuer figure on file
        "IBIT": _etf("iShares Bitcoin Trust ETF", 0.25),
    }
    out = _run(["HODL", "ZZZZ"], _yahoo(rows), saved={"ZZZZ": 0.0})
    hodl, z = out["funds"]["HODL"], out["funds"]["ZZZZ"]
    assert hodl["expense_ratio_pct"] == fc.ISSUER_FEES["HODL"] and hodl["fee_source"] == "issuer"
    assert hodl["fee_dated"] == fc.ISSUER_FEES_CHECKED and hodl["listed_zero"] is True
    # Unknown, not free — and a zero in the saved table is the same stale waiver, so not used either.
    assert z["expense_ratio_pct"] is None and z["fee_source"] is None and z["listed_zero"] is True
    # The 0% listing no longer sorts first in the comparison as "the cheapest bitcoin fund".
    assert all(r["expense_ratio_pct"] != 0.0 for r in hodl["group"]["funds"])
    assert hodl["group"]["funds"][0]["symbol"] == "HODL"   # 0.20 beats BRRR 0.25 and IBIT 0.25


def test_a_real_fee_cut_on_yahoo_beats_the_issuer_figure():
    out = _run(["HODL"], _yahoo({"HODL": _etf("VanEck Bitcoin ETF", 0.15)}))
    assert out["funds"]["HODL"]["expense_ratio_pct"] == 0.15
    assert out["funds"]["HODL"]["fee_source"] == "yahoo"


def test_the_saved_table_is_never_used_for_a_mutual_fund():
    # SAVED carries a bogus FXAIX figure on purpose: the table is ETF-only, because the source it was
    # built from got Fidelity's mutual funds badly wrong.
    fx = _run(["FXAIX"], _down())["funds"]["FXAIX"]
    assert fx["expense_ratio_pct"] is None and fx["fee_source"] is None


def test_yahoo_down_with_nothing_cached_is_unknown_not_free():
    out = _run(["VTI"], _down(), saved={})
    assert out["live"] is False
    vti = out["funds"]["VTI"]
    assert vti["expense_ratio_pct"] is None and vti["fee_source"] is None
    assert all(r["expense_ratio_pct"] is None for r in vti["group"]["funds"])


def test_yahoo_down_serves_the_last_live_fee_with_its_own_time():
    _run(["SPY"], _yahoo(LIVE), now=1_000.0)
    later = 1_000.0 + fc.FEE_TTL_SECONDS + 5
    out = _run(["SPY"], _down(), now=later)
    spy = out["funds"]["SPY"]
    assert out["live"] is False
    assert spy["expense_ratio_pct"] == 0.0945 and spy["fee_source"] == "yahoo"
    assert spy["fee_checked_at"] == 1_000.0      # when it was read, not when it was served


def test_fresh_fees_are_not_fetched_again():
    y = _yahoo(LIVE)
    _run(["SPY"], y, now=1_000.0)
    _run(["VOO"], y, now=2_000.0)                # same group, all cached
    assert len(y.asked) == 1


def test_a_single_stock_is_other_with_no_fee_and_no_group():
    a = _run(["AAPL"], _yahoo(LIVE))["funds"]["AAPL"]
    assert a["kind"] == "other" and a["expense_ratio_pct"] is None and a["group"] is None


def test_staking_is_read_from_the_fund_name_and_the_ether_group_says_why():
    e = _run(["ETHE"], _yahoo(LIVE))["funds"]["ETHE"]
    assert e["staking"] is True
    assert "staking" in (e["group"]["note"] or "").lower()


def test_request_is_deduplicated_and_capped():
    out = _run(["spy", "SPY", " voo "] + [f"X{i}" for i in range(fc.MAX_SYMBOLS + 20)], _yahoo(LIVE))
    assert list(out["funds"])[:2] == ["SPY", "VOO"]
    assert len(out["funds"]) == fc.MAX_SYMBOLS


# --- the route ----------------------------------------------------------------------------------

def test_route_serves_fund_costs(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    import app.main as m
    importlib.reload(m)
    y = _yahoo(LIVE)
    monkeypatch.setattr(m.fund_cost.market_now, "fetch_quotes", y)
    # lookup's default was bound at import, so route the default through the patched function too.
    real = m.fund_cost.lookup

    async def lookup(client, symbols, **kw):
        return await real(client, symbols, fetch=y, **kw)
    monkeypatch.setattr(m.fund_cost, "lookup", lookup)
    with TestClient(m.app) as c:
        r = c.get("/fund_costs", params={"symbols": "SPY,AAPL"})
        assert r.status_code == 200
        body = r.json()
        assert body["funds"]["SPY"]["expense_ratio_pct"] == 0.0945
        assert body["funds"]["AAPL"]["kind"] == "other"
        spym = next(f for f in body["funds"]["SPY"]["group"]["funds"] if f["symbol"] == "SPYM")
        assert spym["expense_ratio_pct"] == 0.02 and spym["fee_source"] == "issuer"
        assert c.get("/fund_costs").json()["funds"] == {}


def test_an_empty_answer_does_not_pin_every_symbol_as_unknown():
    async def empty(_client, symbols):
        return {}
    _run(["VOO"], empty, now=1_000.0)
    assert "VOO" not in fc._cache                       # nothing learned, nothing cached
    y = _yahoo(LIVE)
    _run(["VOO"], y, now=1_001.0)                       # so the next ask goes straight back to Yahoo
    assert y.asked
