"""FUND-1..6 — what a fund covers, pairwise overlap, "same bet" sets, and performance."""
from __future__ import annotations

import asyncio
import datetime as dt
import importlib
import math

import pytest
from fastapi.testclient import TestClient

from app import fund_cost, fund_overlap as fo
from app.market import Series


@pytest.fixture(autouse=True)
def _cold_caches():
    for c in (fund_cost._cache, fund_cost._spreads, fo._profiles, fo._series):
        c.clear()
    yield
    for c in (fund_cost._cache, fund_cost._spreads, fo._profiles, fo._series):
        c.clear()


def _series(sym: str, closes: list[float], start=dt.date(2021, 9, 27)) -> Series:
    """Business-day series from [start]."""
    dates, d = [], start
    while len(dates) < len(closes):
        if d.weekday() < 5:
            dates.append(d.strftime("%Y%m%d"))
        d += dt.timedelta(days=1)
    return Series(symbol=sym, closes=closes, opens=closes, volumes=[1.0] * len(closes), dates=dates,
                  fifty_two_high=None, fifty_two_low=None, currency="USD")


def _wave(n: int, phase: float = 0.0, drift: float = 0.0004, amp: float = 0.03, noise=None) -> list[float]:
    out, p = [], 100.0
    for i in range(n):
        r = drift + amp * math.sin(i / 7.0 + phase) / 10 + (noise(i) if noise else 0.0)
        p *= 1 + r
        out.append(p)
    return out


def _quotes(rows: dict[str, dict]):
    async def fetch(_client, symbols):
        return {s: rows[s] for s in symbols if s in rows}
    return fetch


def _profile_result(sectors: dict[str, float], holdings: list[tuple[str, float]], category="Large Blend",
                    stock=0.99, bond=0.0) -> dict:
    return {
        "topHoldings": {
            "stockPosition": {"raw": stock}, "bondPosition": {"raw": bond},
            "sectorWeightings": [{k: {"raw": v}} for k, v in sectors.items()],
            "holdings": [{"symbol": s, "holdingName": s + " Inc", "holdingPercent": {"raw": w}} for s, w in holdings],
        },
        "fundProfile": {"categoryName": category, "family": "Test"},
    }


# --- what a fund covers -----------------------------------------------------------------------

def test_region_reads_the_category_before_the_stock_split():
    assert fo.region("Foreign Large Blend", 0.97, 0.0) == "international"
    assert fo.region("Diversified Emerging Mkts", 0.98, 0.0) == "international"
    assert fo.region("Digital Assets", 0.0, 0.0) == "crypto"
    assert fo.region("Intermediate Core Bond", 0.0, 0.99) == "bonds"
    assert fo.region("World Large-Stock Blend", 0.99, 0.0) == "world"
    assert fo.region("Trading--Leveraged Equity", 0.9, 0.0) == "leveraged"
    assert fo.region("Moderate Allocation", 0.6, 0.4) == "mixed"
    assert fo.region("Technology", 0.99, 0.0) == "us"
    assert fo.region(None, None, None) is None


def test_profile_normalises_sectors_and_counts_alphabet_once():
    p = fo.parse_profile(_profile_result(
        {"technology": 0.30, "healthcare": 0.10, "utilities": 0.0},
        [("NVDA", 0.08), ("GOOGL", 0.03), ("GOOG", 0.025), ("AAPL", 0.07)],
    ))
    assert [s["label"] for s in p["sectors"]] == ["Tech", "Health care"]       # zero weight dropped
    assert p["sectors"][0]["pct"] == 75.0 and p["sectors"][1]["pct"] == 25.0   # of the stock portion
    tops = {h["symbol"]: h["pct"] for h in p["top_holdings"]}
    assert "GOOG" not in tops and tops["GOOGL"] == 5.5                        # two classes, one company
    assert [h["symbol"] for h in p["top_holdings"]][0] == "NVDA"
    assert p["region"] == "us" and p["stock_pct"] == 99.0


def test_a_fund_with_no_stocks_has_empty_lists_not_missing_ones():
    p = fo.parse_profile(_profile_result({}, [], category="Digital Assets", stock=0.0))
    assert p["sectors"] == [] and p["top_holdings"] == [] and p["top10_pct"] is None
    assert p["region"] == "crypto"


# --- pairs ------------------------------------------------------------------------------------

def test_identical_funds_correlate_at_one_and_the_basis_is_weekly():
    closes = _wave(520)
    c, n = fo.correlation(_series("A", closes), _series("B", closes), step=5)
    assert c == pytest.approx(1.0) and n >= 100


def test_too_little_shared_history_is_unknown_not_zero():
    a = _series("A", _wave(520))
    b = _series("B", _wave(60), start=dt.date(2026, 7, 1))    # a fund three months old
    c, n = fo.correlation(a, b, step=5)
    assert c is None and n < 26


def test_shared_top_is_a_floor_and_says_so_in_its_name():
    a = fo.parse_profile(_profile_result({"technology": 1.0}, [("NVDA", 0.08), ("AAPL", 0.07), ("KO", 0.01)]))
    b = fo.parse_profile(_profile_result({"technology": 1.0}, [("NVDA", 0.09), ("AAPL", 0.05), ("AMD", 0.04)]))
    s = fo.shared_top(a, b)
    assert s["shared_top"] == ["NVDA", "AAPL"] and s["shared_top_count"] == 2
    assert s["shared_top_min_pct"] == 13.0                   # 8 + 5: the smaller weight each time
    assert fo.shared_top(a, None)["shared_top_count"] == 0


def test_sector_likeness_needs_both_sides():
    a = fo.parse_profile(_profile_result({"technology": 0.5, "energy": 0.5}, []))
    b = fo.parse_profile(_profile_result({"technology": 1.0}, []))
    assert fo.sector_alike(a, b) == 50.0
    assert fo.sector_alike(a, fo.parse_profile(_profile_result({}, []))) is None


def test_same_bets_use_complete_linkage_and_never_merge_on_unknown():
    corr = {
        ("VOO", "SPY"): 1.0, ("VOO", "QQQM"): 0.95, ("SPY", "QQQM"): 0.95,
        ("VOO", "SPMO"): 0.871, ("SPY", "SPMO"): 0.87, ("QQQM", "SPMO"): 0.915,
        ("VOO", "SCHD"): 0.52, ("SPY", "SCHD"): 0.52, ("QQQM", "SCHD"): 0.33, ("SPMO", "SCHD"): 0.4,
        ("FBTC", "IBIT"): None,                               # not measured: must not merge
    }
    for x in ("VOO", "SPY", "QQQM", "SPMO", "SCHD"):
        for y in ("FBTC", "IBIT"):
            corr[(x, y)] = 0.3
    bets = fo.same_bets(["VOO", "SPY", "QQQM", "SPMO", "SCHD", "FBTC", "IBIT"], corr)
    assert bets[0] == ["VOO", "SPY", "QQQM"]
    # SPMO clears the bar with QQQM but not with VOO: complete linkage keeps it apart, where single
    # linkage would have chained it in and called five funds one.
    assert ["SPMO"] in bets and ["SCHD"] in bets
    assert ["FBTC"] in bets and ["IBIT"] in bets


# --- performance ------------------------------------------------------------------------------

def test_returns_are_none_where_the_history_does_not_reach():
    closes = [100.0 * (1.001 ** i) for i in range(640)]      # about 2.5 years of business days
    r = fo.returns(_series("X", closes, start=dt.date(2024, 1, 11)))
    assert r["1y"] is not None and r["2y"] is not None
    assert r["3y"] is None and r["5y"] is None


def test_five_years_tolerates_where_the_range_happens_to_start():
    # Yahoo's 5y range can begin a couple of days after the exact five-year date.
    s = _series("X", [100.0 + i * 0.1 for i in range(1305)], start=dt.date(2021, 9, 27))
    assert fo.returns(s)["5y"] is not None


def test_worst_drop_finds_the_deepest_fall_and_its_dates():
    closes = [100, 120, 90, 110, 60, 80, 130]
    w = fo.worst_drop(_series("X", [float(c) for c in closes]))
    assert w["worst_drop_pct"] == -50.0                       # 120 -> 60
    assert w["worst_drop_from"] < w["worst_drop_to"]
    up = fo.worst_drop(_series("Y", [1.0, 2.0, 3.0]))
    assert up["worst_drop_pct"] == 0.0 and up["worst_drop_from"] is None


def test_the_chart_starts_where_every_fund_has_a_price_and_shares_its_days():
    old = _series("OLD", [100.0 + i for i in range(300)])
    young = _series("NEW", [50.0 + i * 0.5 for i in range(100)], start=dt.date(2022, 6, 1))
    ch = fo.aligned_chart({"OLD": old, "NEW": young})
    assert ch["dates"][0] >= "2022-06-01"                    # nothing before the youngest fund
    assert ch["lines"]["OLD"][0] == 0.0 and ch["lines"]["NEW"][0] == 0.0
    assert len(ch["lines"]["OLD"]) == len(ch["lines"]["NEW"]) == len(ch["dates"])
    # A day one fund is missing drops out for both, instead of shifting one line off the other.
    gappy = _series("GAP", [100.0 + i for i in range(300)])
    gappy.dates.pop(-3)
    gappy.closes.pop(-3)
    ch2 = fo.aligned_chart({"OLD": old, "GAP": gappy})
    assert len(ch2["lines"]["OLD"]) == len(ch2["lines"]["GAP"])
    assert fo.aligned_chart({}) is None


# --- end to end with fakes --------------------------------------------------------------------

QUOTES = {
    "VOO": {"long_name": "Vanguard S&P 500 ETF", "quote_type": "ETF", "expense_ratio_pct": 0.03,
            "net_assets": 1.7e12, "state": "REGULAR", "bid": 600.00, "ask": 600.02},
    "SPY": {"long_name": "SPDR S&P 500 ETF Trust", "quote_type": "ETF", "expense_ratio_pct": 0.0945,
            "net_assets": 8.1e11, "state": "REGULAR", "bid": 650.0, "ask": 650.01},
    "FXAIX": {"long_name": "Fidelity 500 Index", "quote_type": "MUTUALFUND", "expense_ratio_pct": 0.015,
              "net_assets": 8.6e11, "state": "REGULAR", "bid": 0.0, "ask": 0.0},
    "SCHD": {"long_name": "Schwab U.S. Dividend Equity ETF", "quote_type": "ETF", "expense_ratio_pct": 0.06,
             "net_assets": 1.1e11, "state": "REGULAR", "bid": 27.0, "ask": 27.01},
    "AAPL": {"long_name": "Apple Inc.", "quote_type": "EQUITY", "expense_ratio_pct": None},
}


def _run_overlap(symbols, *, profiles=None, histories=None, phase="CLOSED", quotes=QUOTES):
    base = _wave(1300)
    histories = histories if histories is not None else {
        "VOO": _series("VOO", base), "SPY": _series("SPY", base), "FXAIX": _series("FXAIX", base),
        "SCHD": _series("SCHD", _wave(1300, phase=2.0, noise=lambda i: 0.004 * math.cos(i * 1.3))),
    }
    profiles = profiles if profiles is not None else {
        s: fo.parse_profile(_profile_result({"technology": 0.4, "healthcare": 0.6}, [("NVDA", 0.08)])) for s in histories
    }

    async def fp(_c, s):
        if isinstance(profiles.get(s), Exception):
            raise profiles[s]
        return profiles.get(s)

    async def fh(_c, s):
        h = histories.get(s)
        if h is None:
            raise RuntimeError("no history")
        return h
    return asyncio.run(fo.overlap(None, symbols, saved={}, saved_as_of="2026-08-27", fetch_profile=fp,
                                  fetch_history=fh, quotes=_quotes(quotes), now=1_000.0, phase=phase))


def test_overlap_sorts_out_stocks_and_groups_the_same_fund():
    out = _run_overlap(["VOO", "SPY", "FXAIX", "SCHD", "AAPL"])
    assert out["not_funds"] == ["AAPL"]
    assert set(out["funds"]) == {"VOO", "SPY", "FXAIX", "SCHD"}
    assert out["same_bets"][0] == ["VOO", "SPY", "FXAIX"] or set(out["same_bets"][0]) == {"VOO", "SPY", "FXAIX"}
    pair = next(p for p in out["pairs"] if {p["a"], p["b"]} == {"VOO", "FXAIX"})
    assert pair["corr_basis"] == "4-weekly"                  # a mutual fund is involved
    voo = out["funds"]["VOO"]
    assert voo["group_id"] == "sp500" and voo["region_label"] == "US stocks" and voo["profile_ok"]
    assert voo["net_assets"] == 1.7e12


def test_a_failed_holdings_lookup_is_unknown_not_empty():
    base = _wave(1300)
    out = _run_overlap(["VOO", "SPY"], histories={"VOO": _series("VOO", base), "SPY": _series("SPY", base)},
                       profiles={"VOO": RuntimeError("yahoo down"),
                                 "SPY": fo.parse_profile(_profile_result({"technology": 1.0}, [("NVDA", 0.08)]))})
    voo = out["funds"]["VOO"]
    assert voo["profile_ok"] is False and voo["sectors"] is None and voo["top_holdings"] is None
    assert out["pairs"][0]["corr"] == pytest.approx(1.0)     # prices still answer the real question


def test_a_failed_price_history_leaves_the_pair_unmeasured():
    base = _wave(1300)
    out = _run_overlap(["VOO", "SPY"], histories={"VOO": _series("VOO", base)})
    p = out["pairs"][0]
    assert p["corr"] is None and p["corr_points"] == 0
    assert out["same_bets"] == [["SPY"], ["VOO"]]           # unknown never merges


def test_spreads_are_measured_only_during_the_session():
    after_hours = {s: {**q, "state": "POST"} for s, q in QUOTES.items()}
    closed = _run_overlap(["VOO", "SPY"], phase="CLOSED", quotes=after_hours)
    assert closed["funds"]["VOO"]["spread_pct"] is None      # a closed market's bid/ask is not a cost
    fund_cost._cache.clear()
    fo._profiles.clear()
    fo._series.clear()
    open_ = _run_overlap(["VOO", "SPY", "FXAIX"], phase="REGULAR")
    assert open_["funds"]["VOO"]["spread_pct"] == pytest.approx(0.02 / 600.01 * 100, rel=1e-6)
    assert open_["funds"]["VOO"]["spread_at"] == 1_000.0
    assert open_["funds"]["FXAIX"]["spread_pct"] is None      # a mutual fund has no spread


def test_performance_says_unavailable_instead_of_zero():
    async def fh(_c, s):
        if s == "BAD":
            raise RuntimeError("no data")
        return _series(s, [100.0 + i for i in range(600)])
    out = asyncio.run(fo.performance(None, ["VOO", "BAD"], include_series=True, fetch_history=fh, now=5.0))
    assert out["funds"]["BAD"] == {"symbol": "BAD", "available": False}
    v = out["funds"]["VOO"]
    assert v["available"] and v["returns"]["1y"] is not None
    assert set(out["chart"]["lines"]) == {"VOO"}             # the unreadable fund has no line


# --- routes -----------------------------------------------------------------------------------

def test_routes_answer_empty_requests_and_delegate(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    import app.main as m
    importlib.reload(m)

    async def fake_overlap(client, symbols, **kw):
        return {"funds": {s: {} for s in symbols}, "not_funds": [], "pairs": [], "same_bets": [],
                "same_bet_corr": 0.9, "live": True, "as_of": 1.0}

    async def fake_perf(client, symbols, include_series=False, **kw):
        return {"funds": {s: {"series": include_series} for s in symbols}, "as_of": 1.0}

    async def fake_groups(client, **kw):
        return {"groups": [{"id": "sp500"}], "live": True, "as_of": 1.0}
    monkeypatch.setattr(m.fund_overlap, "overlap", fake_overlap)
    monkeypatch.setattr(m.fund_overlap, "performance", fake_perf)
    monkeypatch.setattr(m.fund_cost, "groups_overview", fake_groups)
    with TestClient(m.app) as c:
        assert c.get("/funds/overlap").json()["funds"] == {}
        assert set(c.get("/funds/overlap", params={"symbols": "voo,spy"}).json()["funds"]) == {"VOO", "SPY"}
        assert c.get("/funds/performance", params={"symbols": "VOO", "series": "true"}).json()["funds"]["VOO"]["series"] is True
        assert c.get("/funds/groups").json()["groups"][0]["id"] == "sp500"


# --- review fixes (2026-09-26) ------------------------------------------------------------------

def test_returns_side_by_side_end_on_the_same_day():
    # Identical paths; the ETF carries today's live bar (+1.5%) that the mutual fund's NAV lacks.
    base = [100.0 * (1.0004 ** i) for i in range(600)]
    etf = _series("VOO", base + [base[-1] * 1.015])
    mf = _series("FXAIX", base)

    async def fh(_c, s):
        return {"VOO": etf, "FXAIX": mf}[s]
    out = asyncio.run(fo.performance(None, ["VOO", "FXAIX"], fetch_history=fh, now=1.0))
    assert out["funds"]["VOO"]["returns"]["2y"] == out["funds"]["FXAIX"]["returns"]["2y"]
    assert out["aligned_to"] == out["funds"]["FXAIX"]["last_date"]
    # Measured on its own last bar the ETF would have been a whole session ahead.
    assert fo.returns(etf)["2y"] > out["funds"]["VOO"]["returns"]["2y"]


def test_a_zero_close_is_dropped_not_divided_by():
    s = _series("Z", [0.0] + [100.0 + i for i in range(300)])
    clean = fo._positive(s)
    assert 0.0 not in clean.closes and len(clean.closes) == 300

    async def fh(_c, sym):
        return s if sym == "Z" else _series(sym, [100.0 + i for i in range(301)])
    out = asyncio.run(fo.performance(None, ["Z", "VOO"], include_series=True, fetch_history=fh, now=2.0))
    assert out["funds"]["Z"]["available"] and out["funds"]["VOO"]["available"]
    assert out["chart"] is not None


def test_an_unanswered_symbol_is_unknown_not_a_stock():
    out = _run_overlap(["VOO", "SPY", "ZZZZ"], quotes={k: v for k, v in QUOTES.items()})
    assert "ZZZZ" in out["unknown"] and "ZZZZ" not in out["not_funds"]
    assert out["not_funds"] == []


def test_funds_past_the_cap_are_unmeasured_not_stocks():
    many = [f"F{i:02d}" for i in range(fo.MAX_FUNDS + 2)]
    quotes = {s: {"long_name": s, "quote_type": "ETF", "expense_ratio_pct": 0.1} for s in many}
    base = _wave(1300)
    out = _run_overlap(many, quotes=quotes, histories={s: _series(s, base) for s in many})
    assert len(out["funds"]) == fo.MAX_FUNDS
    assert out["unmeasured"] == many[fo.MAX_FUNDS:]
    assert out["not_funds"] == []


def test_a_profile_missing_its_holdings_module_is_a_failed_read():
    class R:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"quoteSummary": {"result": [{"fundProfile": {"categoryName": "Large Blend"}}]}}

    class C:
        async def get(self, *a, **k):
            return R()

    async def auth(_c, **k):
        return "crumb"
    orig = fo.options._ensure_auth
    fo.options._ensure_auth = auth
    try:
        assert asyncio.run(fo._yahoo_profile(C(), "VOO")) is None
    finally:
        fo.options._ensure_auth = orig
