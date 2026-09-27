"""FUND-8 — the Explore catalogue: plain names, one shared end day, windowed drops, cheaper copies."""
from __future__ import annotations

import asyncio
import datetime as dt
from collections import Counter

import pytest

from app import fund_catalog as fc, fund_cost, fund_overlap as fo
from app.market import Series


@pytest.fixture(autouse=True)
def _cold():
    for c in (fund_cost._cache, fund_cost._spreads, fo._series):
        c.clear()
    fc._built = None
    fc._task = None
    yield
    for c in (fund_cost._cache, fund_cost._spreads, fo._series):
        c.clear()
    fc._built = None
    fc._task = None


def _series(sym: str, closes: list[float], end=dt.date(2026, 9, 25)) -> Series:
    """Business-day series ENDING on [end]."""
    dates, d = [], end
    while len(dates) < len(closes):
        if d.weekday() < 5:
            dates.append(d.strftime("%Y%m%d"))
        d -= dt.timedelta(days=1)
    dates.reverse()
    return Series(symbol=sym, closes=closes, opens=closes, volumes=[1.0] * len(closes), dates=dates,
                  fifty_two_high=None, fifty_two_low=None, currency="USD")


def _quotes(rows: dict[str, dict]):
    async def fetch(_client, symbols):
        return {s: rows[s] for s in symbols if s in rows}
    return fetch


def test_the_catalogue_has_no_duplicates_known_types_and_short_plain_names():
    c = Counter(e.symbol for e in fc.CATALOG)
    assert [s for s, n in c.items() if n > 1] == []
    cats = {k for k, _ in fc.CATEGORIES}
    assert all(e.category in cats for e in fc.CATALOG)
    assert max(len(e.name) for e in fc.CATALOG) <= 26
    # Every measured look-alike is listed, so a cheaper copy is always a fund you can open here.
    members = {m for g in fund_cost.GROUPS for m in g.members}
    assert members <= set(c)


def test_windows_measure_return_and_drop_over_the_same_days():
    # 300 business days: a fall from 100 to 50 in the first 30, then a steady climb to 120.
    closes = [100 - i * 50 / 30 for i in range(30)] + [50 + i * 70 / 269 for i in range(270)]
    s = _series("X", closes)
    rets, drops = fc.window_stats(s, dt.date(2026, 9, 25))
    # One year back lands after the fall: the 1-year drop never sees it.
    assert rets["1y"] is not None and rets["1y"] > 0
    assert drops["1y"] == 0.0
    # The fund is younger than three years: no figure, not a flattering one.
    assert rets["3y"] is None and drops["3y"] is None
    assert rets["5y"] is None and drops["5y"] is None


def test_a_young_fund_has_no_five_year_drop_to_win_a_ranking_with():
    old = _series("OLD", [100.0] * 900 + [60.0] * 10 + [100.0] * 400)
    young = _series("NEW", [100.0] * 200)
    _, d_old = fc.window_stats(old, dt.date(2026, 9, 25))
    _, d_new = fc.window_stats(young, dt.date(2026, 9, 25))
    assert d_old["5y"] == -40.0
    assert d_new["5y"] is None


def test_cheaper_copy_prefers_the_cheapest_etf_over_a_cheaper_mutual_fund():
    rows = {
        "VTI": {"symbol": "VTI", "name": "Vanguard Total", "kind": "etf", "expense_ratio_pct": 0.03, "fidelity_only": False},
        "ITOT": {"symbol": "ITOT", "name": "iShares Core", "kind": "etf", "expense_ratio_pct": 0.03, "fidelity_only": False},
        "SCHB": {"symbol": "SCHB", "name": "Schwab", "kind": "etf", "expense_ratio_pct": 0.03, "fidelity_only": False},
        "SPTM": {"symbol": "SPTM", "name": "SPDR", "kind": "etf", "expense_ratio_pct": 0.03, "fidelity_only": False},
        "FSKAX": {"symbol": "FSKAX", "name": "Fidelity Total", "kind": "mutual_fund", "expense_ratio_pct": 0.015, "fidelity_only": False},
        "FZROX": {"symbol": "FZROX", "name": "Fidelity ZERO", "kind": "mutual_fund", "expense_ratio_pct": 0.0, "fidelity_only": True},
    }
    # No ETF is cheaper than VTI, so the cheapest mutual fund is offered, marked Fidelity-only.
    c = fc.cheaper_copy("VTI", rows)
    assert c["symbol"] == "FZROX" and c["fidelity_only"] and c["mutual_fund"]
    assert c["saves_per_10k"] == 3.0
    # The cheapest is offered nothing.
    assert fc.cheaper_copy("FZROX", rows) is None
    # An unknown fee has no cheaper copy: there is nothing to compare.
    rows["VTI"]["expense_ratio_pct"] = None
    assert fc.cheaper_copy("VTI", rows) is None
    # A fund outside every measured group has none either.
    assert fc.cheaper_copy("SCHD", rows) is None


def test_a_stale_fund_is_left_out_instead_of_dragging_the_end_day_back():
    hs = {
        "A": _series("A", [1.0, 2.0]),
        "B": _series("B", [1.0, 2.0], end=dt.date(2026, 9, 24)),     # a mutual fund's NAV, a day behind
        "C": _series("C", [1.0, 2.0], end=dt.date(2026, 8, 1)),      # stopped updating
    }
    end, stale = fc._end_date(hs)
    assert end == dt.date(2026, 9, 24)
    assert stale == {"C"}


def test_build_measures_to_one_day_and_says_what_it_could_not_read():
    n = 1320
    series = {s: _series(s, [100.0 + i * 0.05 for i in range(n)]) for s in ("VOO", "IVV", "SPY")}
    series["FXAIX"] = _series("FXAIX", [100.0 + i * 0.05 for i in range(n)], end=dt.date(2026, 9, 24))

    async def fetch_history(_client, s):
        if s in series:
            return series[s]
        raise RuntimeError("no data")

    quotes = _quotes({
        "VOO": {"long_name": "Vanguard S&P 500 ETF", "quote_type": "ETF", "expense_ratio_pct": 0.03},
        "SPY": {"long_name": "SPDR S&P 500", "quote_type": "ETF", "expense_ratio_pct": 0.0945},
        "IVV": {"long_name": "iShares Core S&P 500", "quote_type": "ETF", "expense_ratio_pct": 0.03},
        "SPYM": {"long_name": "SPDR Portfolio S&P 500", "quote_type": "EQUITY"},
        "FXAIX": {"long_name": "Fidelity 500 Index", "quote_type": "MUTUALFUND", "expense_ratio_pct": 0.015},
    })
    out = asyncio.run(fc.build(None, saved={}, saved_as_of="2026-08-27", fetch_history=fetch_history,
                               quotes=quotes, now=1_790_000_000.0))
    by = {f["symbol"]: f for f in out["funds"]}
    assert out["aligned_to"] == "2026-09-24"          # the mutual fund's day, not the ETFs' newer one
    assert by["VOO"]["available"] and by["VOO"]["returns"]["5y"] is not None
    assert by["VOO"]["name"] == "S&P 500" and by["VOO"]["category"] == "us"
    # SPY's cheaper copy is the cheapest ETF: SPYM at the issuer's 0.02%, not a mutual fund.
    assert by["SPY"]["cheaper"]["symbol"] == "SPYM"
    assert by["SPY"]["cheaper"]["saves_per_10k"] == pytest.approx(7.45)
    # No history: unavailable, with every figure unknown rather than zero.
    assert by["QQQ"]["available"] is False
    assert by["QQQ"]["returns"] == {"1y": None, "3y": None, "5y": None}
    assert by["QQQ"]["expense_ratio_pct"] is None
    cats = {c["id"]: c["count"] for c in out["categories"]}
    assert sum(cats.values()) == len(fc.CATALOG)
    assert out["measured"] == 4


def test_explore_serves_the_last_build_and_refreshes_it_in_the_background(monkeypatch):
    calls = []

    async def fake_build(_client, *, saved, saved_as_of, **_):
        calls.append(1)
        return {"funds": [], "categories": [], "aligned_to": None, "measured": len(calls), "live": True,
                "built_at": 1000.0 * len(calls)}

    monkeypatch.setattr(fc, "build", fake_build)

    async def run():
        first = await fc.explore(None, saved={}, saved_as_of="x", now=1000.0)
        assert first["built_at"] == 1000.0 and len(calls) == 1
        # Fresh: served as-is, nothing rebuilt.
        again = await fc.explore(None, saved={}, saved_as_of="x", now=1000.0 + 60)
        assert len(calls) == 1 and again["refreshing"] is False
        # Old: the old build is served at once while a new one runs.
        old = await fc.explore(None, saved={}, saved_as_of="x", now=1000.0 + fc.EXPLORE_TTL_SECONDS + 1)
        assert old["built_at"] == 1000.0 and old["refreshing"] is True
        await fc._task
        assert fc._built["built_at"] == 2000.0

    asyncio.run(run())


def test_a_build_that_measured_nothing_does_not_replace_a_good_one(monkeypatch):
    async def empty_build(_client, **_):
        return {"funds": [], "categories": [], "aligned_to": None, "measured": 0, "live": False, "built_at": 5.0}

    monkeypatch.setattr(fc, "build", empty_build)
    fc._built = {"measured": 150, "built_at": 1.0}
    asyncio.run(fc._build_and_keep(None, {}, "x"))
    assert fc._built["measured"] == 150
