"""DP-3 — the Daily Pick's routes: what the card is served, and when a run repeats.

The honesty rules pinned here are the ones the app cannot enforce on its own:
  * yesterday's pick is served with its own date and `stale: true`, never as today's;
  * a failed run is served as `failed` with its error, never as a quiet "no pick";
  * a completed day does not re-run without `force`, but a FAILED day does, so the retry timer can
    recover a lost morning.
"""
import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    import app.daily_pick_store as dps
    import app.settings_store as st
    importlib.reload(st)
    importlib.reload(dps)
    import app.main as m
    importlib.reload(m)

    async def fake_quotes(http, syms):
        return {s: {"price": 100.0, "pct": 1.0, "state": "REGULAR"} for s in syms}
    monkeypatch.setattr(m.market_now, "fetch_quotes", fake_quotes)
    with TestClient(m.app) as c:
        yield m, dps, c


def _pick_run(date, symbol="AAA"):
    return {"date": date, "ts": 0, "status": "pick", "pick": {
        "symbol": symbol, "conviction": 72, "levels": {"entry_low": 98.0, "entry_high": 101.0,
                                                       "stop": 94.0, "target": 112.0}}}


def test_nothing_run_yet_is_unavailable(env):
    _, _, c = env
    body = c.get("/daily_pick").json()
    assert body["available"] is False and body["reason"]


def test_todays_pick_is_fresh_with_a_live_chase_read(env):
    m, dps, c = env
    dps.append_run(_pick_run(m._et_now().date().isoformat()))
    body = c.get("/daily_pick").json()
    assert body["stale"] is False and body["status"] == "pick"
    assert body["live"]["price"] == 100.0
    assert body["chase"]["status"] == "in_zone"


def test_yesterdays_pick_is_served_stale_with_its_own_date(env):
    _, dps, c = env
    dps.append_run(_pick_run("2020-01-02"))
    body = c.get("/daily_pick").json()
    assert body["stale"] is True and body["date"] == "2020-01-02"


def test_a_failed_run_is_failed_not_none(env):
    m, dps, c = env
    today = m._et_now().date().isoformat()
    dps.append_run({"date": today, "ts": 0, "status": "failed", "error": "the analyst call failed"})
    body = c.get("/daily_pick").json()
    assert body["status"] == "failed" and body["error"] and body["live"] is None


def test_a_later_run_on_the_same_day_wins(env):
    m, dps, c = env
    today = m._et_now().date().isoformat()
    dps.append_run({"date": today, "ts": 0, "status": "failed", "error": "x"})
    dps.append_run(_pick_run(today, "BBB"))
    assert c.get("/daily_pick").json()["pick"]["symbol"] == "BBB"


def test_completed_day_does_not_rerun_but_a_failed_day_does(env, monkeypatch):
    m, dps, c = env
    calls = []

    async def fake_compute(today, now_et):
        calls.append(today)
        return {"date": today, "ts": 0, "status": "none", "none_reason": "quiet"}
    monkeypatch.setattr(m, "_daily_pick_compute", fake_compute)
    monkeypatch.setattr(m.market_calendar, "is_trading_day", lambda d: True)
    today = m._et_now().date().isoformat()
    dps.append_run({"date": today, "ts": 0, "status": "failed", "error": "x"})
    assert c.post("/daily_pick/run", json={}).json()["status"] == "none"
    assert c.post("/daily_pick/run", json={}).json()["status"] == "already_ran"
    assert c.post("/daily_pick/run", json={"force": True}).json()["status"] == "none"
    assert len(calls) == 2


def test_a_crashed_run_is_recorded_as_failed(env, monkeypatch):
    m, dps, c = env

    async def boom(today, now_et):
        raise RuntimeError("kaboom")
    monkeypatch.setattr(m, "_daily_pick_compute", boom)
    monkeypatch.setattr(m.market_calendar, "is_trading_day", lambda d: True)
    assert c.post("/daily_pick/run", json={}).json()["status"] == "failed"
    assert c.get("/daily_pick").json()["status"] == "failed"


def test_settings_validate_the_universe(env):
    _, _, c = env
    assert c.post("/daily_pick/settings", json={"universe": "everything"}).status_code == 422
    assert c.post("/daily_pick/settings", json={"universe": "watchlist"}).json()["universe"] == "watchlist"
    assert c.get("/daily_pick/settings").json()["universe"] == "watchlist"


def test_history_reports_unwritten_marks_as_none(env):
    _, dps, c = env
    dps.append_run({**_pick_run("2026-09-01"), "rule_pick": {"symbol": "BBB", "as_of_date": "20260831"}})
    body = c.get("/daily_pick/history").json()
    item = body["items"][0]
    assert item["marks"]["20d"] is None and item["rule_marks"]["20d"] is None
    assert body["comparison"]["20d"]["n_days"] == 0
    assert body["report_cards"] == []


def test_recheck_needs_a_morning_pick(env, monkeypatch):
    m, dps, c = env
    body = c.post("/daily_pick/recheck").json()
    assert body["status"] == "failed" and "morning pick" in body["error"]
    assert body["graded"] is False


def test_recheck_cooldown_returns_the_last_one(env, monkeypatch):
    m, dps, c = env
    calls = []

    async def fake(today, now_et):
        calls.append(1)
        return {"date": today, "ts": m.time.time(), "status": "pick", "graded": False,
                "pick": {"symbol": "AAA"}, "same_as_morning": True}
    monkeypatch.setattr(m, "_daily_pick_recheck_compute", fake)
    first = c.post("/daily_pick/recheck").json()
    second = c.post("/daily_pick/recheck").json()
    assert len(calls) == 1 and first["cooldown_seconds"] == 0 and second["cooldown_seconds"] > 0


def test_a_failed_recheck_does_not_block_a_retry(env, monkeypatch):
    m, dps, c = env
    calls = []

    async def fake(today, now_et):
        calls.append(1)
        return {"date": today, "ts": m.time.time(), "status": "failed", "error": "x", "graded": False}
    monkeypatch.setattr(m, "_daily_pick_recheck_compute", fake)
    c.post("/daily_pick/recheck"); c.post("/daily_pick/recheck")
    assert len(calls) == 2


def test_the_card_carries_todays_recheck_beside_the_pick(env):
    m, dps, c = env
    today = m._et_now().date().isoformat()
    dps.append_run(_pick_run(today))
    dps.append_recheck({"date": today, "ts": 0, "status": "pick", "pick": {"symbol": "BBB"}, "same_as_morning": False})
    body = c.get("/daily_pick").json()
    assert body["pick"]["symbol"] == "AAA"          # the morning pick is untouched
    assert body["recheck"]["pick"]["symbol"] == "BBB"
