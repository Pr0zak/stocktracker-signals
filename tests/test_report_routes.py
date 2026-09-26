"""RPT-1 — assembling, storing and serving the weekly and monthly report."""
import datetime as dt
import importlib

import pytest
from fastapi.testclient import TestClient

from app.market import Series

D = dt.date
DATES = ["20260917", "20260918", "20260921", "20260922", "20260923", "20260924", "20260925"]


def _series(sym, start, end, dates=DATES):
    """A series that moves linearly from `start` on the 18th to `end` on the 25th."""
    n = len(dates)
    closes = [start] * 2 + [start + (end - start) * i / (n - 2) for i in range(1, n - 1)]
    return Series(symbol=sym, closes=closes, opens=closes, volumes=[1.0] * n, dates=dates,
                  fifty_two_high=None, fifty_two_low=None, currency="USD")


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    import app.report_job as rj
    import app.sandbox_store as ss
    import app.daily_pick_store as dps
    import app.settings_store as st
    for mod in (st, ss, dps, rj):
        importlib.reload(mod)
    import app.main as m
    importlib.reload(m)
    with TestClient(m.app) as c:
        yield m, rj, c


def _inputs(rj, *, sp_end=103.0):
    from app import report
    series = {s: _series(s, 100.0, 101.0) for _, s, _, _, _ in report.INDEXES}
    series["^GSPC"] = _series("^GSPC", 100.0, sp_end)
    series["RSP"] = _series("RSP", 100.0, 99.0)
    for sym, _ in report.SECTORS:
        series[sym] = _series(sym, 100.0, 102.0 if sym == "XLK" else 99.0)
    for i, sym in enumerate(report.ETFS):
        series[sym] = _series(sym, 100.0, 100.0 + (i % 7) - 3)
    big = [{"symbol": f"S{i}", "name": f"Company {i}, Inc. - Common Stock", "market_cap": 2e10}
           for i in range(20)]
    for i, b in enumerate(big):
        series[b["symbol"]] = _series(b["symbol"], 50.0, 50.0 + i - 12)
    nav = [{"date": "2026-09-18", "equity": 1000.0, "funded_total": 1000.0, "benchmark_value": 1000.0, "cash": 50.0},
           {"date": "2026-09-25", "equity": 990.0, "funded_total": 1000.0, "benchmark_value": 1012.0, "cash": 40.0}]
    return series, big, nav


def test_assemble_measures_every_section(env):
    _, rj, _ = env
    series, big, nav = _inputs(rj)
    rep = rj.assemble("week", D(2026, 9, 18), D(2026, 9, 25), series, [], big,
                      nav_by_arm={"main": nav, "fast": nav}, labels={"main": "Baseline"},
                      main_trades=[], pick_runs=[], now=1.0)
    assert rep["id"] == "week-2026-09-25" and rep["label"] == "Sep 21 – 25"
    sp = next(i for i in rep["market"]["indexes"] if i["key"] == "sp500")
    assert sp["pct"] == 3.0
    assert rep["market"]["typical_stock"]["pct"] == -1.0
    assert rep["market"]["sectors"][0] == {"symbol": "XLK", "name": "Tech", "pct": 2.0}
    stocks = rep["market"]["stocks"]
    assert stocks["best"][0]["symbol"] == "S19" and stocks["best"][0]["name"] == "Company 19"
    assert stocks["worst"][0]["symbol"] == "S0" and stocks["measured"] == 20
    # The sector funds have their own tile and never appear among the ETF movers.
    assert not {r["symbol"] for r in rep["market"]["etfs"]["best"] + rep["market"]["etfs"]["worst"]} & {"XLK", "XLU"}
    assert rep["sandbox"]["available"] and rep["sandbox"]["main"]["change_usd"] == -10.0
    assert rep["headline"] == "Big tech lifted the S&P. Most stocks fell." or rep["headline"].startswith("The biggest")
    assert rep["market"]["daily"] is None  # the calendar is monthly only


def test_a_period_without_the_sp_is_refused_not_stored(env):
    _, rj, _ = env
    series, big, nav = _inputs(rj)
    del series["^GSPC"]
    with pytest.raises(rj.ReportError):
        rj.assemble("week", D(2026, 9, 18), D(2026, 9, 25), series, ["^GSPC"], big,
                    nav_by_arm={"main": nav}, labels={}, main_trades=[], pick_runs=[])


def test_too_few_big_companies_is_refused(env):
    _, rj, _ = env
    series, big, nav = _inputs(rj)
    for b in big[:15]:
        del series[b["symbol"]]
    with pytest.raises(rj.ReportError, match="5 of 20"):
        rj.assemble("week", D(2026, 9, 18), D(2026, 9, 25), series, [], big,
                    nav_by_arm={"main": nav}, labels={}, main_trades=[], pick_runs=[])


def test_no_sandbox_record_is_said_not_zeroed(env):
    _, rj, _ = env
    series, big, _ = _inputs(rj)
    rep = rj.assemble("week", D(2026, 9, 18), D(2026, 9, 25), series, [], big,
                      nav_by_arm={"main": []}, labels={}, main_trades=[], pick_runs=[])
    assert rep["sandbox"]["available"] is False and rep["sandbox"]["main"] is None and rep["sandbox"]["reason"]


def test_routes_serve_stored_reports_newest_first(env):
    _, rj, c = env
    assert c.get("/report/latest?kind=week").json()["available"] is False
    series, big, nav = _inputs(rj)
    for end, start in ((D(2026, 9, 25), D(2026, 9, 18)),):
        rj.save(rj.assemble("week", start, end, series, [], big, nav_by_arm={"main": nav}, labels={},
                            main_trades=[], pick_runs=[]))
    older = rj.assemble("week", D(2026, 9, 18), D(2026, 9, 25), series, [], big, nav_by_arm={"main": nav},
                        labels={}, main_trades=[], pick_runs=[])
    older["id"], older["end"] = "week-2026-09-18", "2026-09-18"
    rj.save(older)
    rows = c.get("/reports").json()["reports"]
    assert [r["id"] for r in rows] == ["week-2026-09-25", "week-2026-09-18"]
    assert rows[0]["sp500_pct"] == 3.0 and rows[0]["sandbox_pct"] == -1.0
    latest = c.get("/report/latest?kind=week").json()
    assert latest["available"] is True and latest["id"] == "week-2026-09-25"
    assert c.get("/report/week-2026-09-18").json()["id"] == "week-2026-09-18"
    assert c.get("/report/week-2026-01-02").status_code == 404
    assert c.get("/report/..%2Fsettings").status_code == 404
    assert c.get("/report/latest?kind=year").status_code == 422


def test_run_reports_a_refusal_as_failed(env, monkeypatch):
    m, rj, c = env

    async def refuse(kind, **kw):
        raise rj.ReportError("the S&P 500 could not be measured for this period")
    monkeypatch.setattr(m.report_job, "build", refuse)
    body = c.post("/report/run?kind=auto").json()
    assert [r["status"] for r in body["results"]] == ["failed", "failed"]
    assert c.post("/report/run?kind=auto&end=2026-09-25").status_code == 422


def test_report_routes_need_the_token_from_off_box(env):
    m, _, _ = env
    with TestClient(m.app, client=("203.0.113.50", 5000)) as remote:
        assert remote.get("/reports").status_code == 401
        assert remote.get("/report/latest").status_code == 401
