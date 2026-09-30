"""TODAY-1 — the all-arms "today" summary (app/sandbox_today.py) and its endpoint."""
from __future__ import annotations

from app import sandbox_today as st


def _arm(arm, day="2026-09-30", **kw):
    return {"arm": arm, "label": kw.get("label", arm), "engine": "llm", "universe": kw.get("u", "all"),
            "enabled": True, "last_tick_date": day, "last_posture": kw.get("posture", "")}


def _t(sym, side="buy", status="filled", day="2026-09-30", gross=100.0, **kw):
    return {"date": day, "symbol": sym, "side": side, "status": status, "shares": kw.get("shares", 1.0),
            "price": kw.get("price", gross), "gross": gross, "skip_reason": kw.get("skip_reason")}


def test_ran_and_traded_ran_and_held_and_did_not_run_stay_distinct():
    arms = [_arm("main"), _arm("etf", u="etf"), _arm("old", day="2026-09-25")]
    trades = {
        "main": [_t("CASH", side="interest", gross=0.05)],
        "etf": [_t("VEA", gross=353.65), _t("VONV", gross=1091.55),
                _t("VEU", status="skipped", gross=None,
                   skip_reason="turnover cap (25% of equity) left room for less than one share")],
        "old": [_t("AAPL", day="2026-09-25")],
    }
    out = st.summarize(arms, trades, {"main": 1.0, "etf": 2.0})
    by = {a["arm"]: a for a in out["arms"]}
    assert out["date"] == "2026-09-30" and out["ran_at"] == 2.0
    assert by["main"]["ran"] and by["main"]["filled"] == [], "interest is not a trade"
    assert [o["symbol"] for o in by["etf"]["filled"]] == ["VONV", "VEA"], "biggest first"
    assert by["etf"]["skipped"] == [{"symbol": "VEU", "side": "buy", "shares": 1.0, "price": None,
                                     "gross": None, "reason": "daily trade limit"}]
    assert by["old"]["ran"] is False and by["old"]["filled"] == [], "an old day's trade is not today's"
    assert out["totals"] == {"buys": 2, "sells": 0, "skipped": 1, "bought": 1445.2, "sold": 0.0}


def test_sells_are_counted_apart_from_buys():
    out = st.summarize([_arm("main")], {"main": [_t("XOM", side="sell", gross=640.0), _t("VTI", gross=500.0)]}, {})
    assert out["totals"]["sells"] == 1 and out["totals"]["sold"] == 640.0
    assert out["totals"]["buys"] == 1 and out["totals"]["bought"] == 500.0


def test_no_run_ever_is_an_empty_answer_not_zero_trades_today():
    out = st.summarize([_arm("main", day=None)], {}, {})
    assert out["date"] is None and out["arms"] == []


def test_short_reasons_are_plain_and_bounded():
    assert st.short_reason("wash-sale window (24d left since the loss sale)") == "wash-sale wait"
    assert st.short_reason("cash floor (0% of equity) left under one share at $382.55") == "not enough cash"
    assert st.short_reason("review model dropped this order — I dropped GLDM because…") == "reviewer dropped it"
    assert st.short_reason(None) == "skipped"
    long = st.short_reason("x" * 80)
    assert len(long) <= 48 and long.endswith("…")


def test_endpoint_is_token_gated_and_answers(monkeypatch):
    from fastapi.testclient import TestClient
    from app import sandbox_store
    from app.main import app

    monkeypatch.setattr(sandbox_store, "list_arms", lambda: ["main"])
    monkeypatch.setattr(sandbox_store, "get", lambda a="main": {
        "label": "Baseline", "engine": "llm", "settings": {"master_enabled": True},
        "last_tick_date": "2026-09-30", "last_posture": "Hold"})
    monkeypatch.setattr(sandbox_store, "read_trades", lambda n, a="main": [_t("VTI", gross=376.0)])
    monkeypatch.setattr(sandbox_store, "read_nav", lambda d=None, a="main": [{"date": "2026-09-30", "ts": 5.0}])
    with TestClient(app) as c:
        body = c.get("/sandbox/today").json()
    assert body["date"] == "2026-09-30" and body["ran_at"] == 5.0
    assert body["arms"][0]["enabled"] is True and body["arms"][0]["filled"][0]["symbol"] == "VTI"
    with TestClient(app, client=("203.0.113.9", 5000)) as c:
        assert c.get("/sandbox/today").status_code == 401
