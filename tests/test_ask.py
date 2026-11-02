"""Ask AI: the context the model answers from, and the check on what it may suggest."""
from __future__ import annotations

import asyncio

from app import ask


BUILD = {"funds": [
    {"symbol": "QQQM", "name": "Nasdaq-100", "category": "us", "expense_ratio_pct": 0.15,
     "group_label": "the 100 biggest Nasdaq companies", "returns": {"1y": 10.9}, "drops": {"3y": -22.0}},
    {"symbol": "GLDM", "name": "Gold", "category": "commodities", "expense_ratio_pct": 0.10,
     "returns": {"1y": -3.5}, "drops": {"3y": -27.0}},
    {"symbol": "VEA", "name": "Rich countries", "category": "world", "expense_ratio_pct": 0.03},
]}
ROWS = {
    "QQQM": {"price": 250.0, "pct_off_52w_high": -8.53, "pct_vs_sma50": -4.12, "pct_vs_sma200": 2.2,
             "above_sma200": True, "mom_126d": 6.0},
    "GLDM": {"price": 80.0, "pct_off_52w_high": -26.56, "pct_vs_sma50": -6.67, "pct_vs_sma200": -11.84,
             "above_sma200": False, "mom_126d": -9.0},
    "VEA": None,                                    # history unreadable
}
QUOTES = {"QQQM": {"price": 251.65, "pct": 0.66}, "GLDM": {"price": 81.0, "pct": 2.06}}


def test_board_flags_sound_pullbacks_and_keeps_unmeasured_funds():
    b = {r["symbol"]: r for r in ask.etf_board(BUILD, ROWS, QUOTES, ["QQQM", "GLDM", "VEA"])}
    assert b["QQQM"]["dip"] is True                 # 8.5% off, 200-day intact
    assert b["GLDM"]["dip"] is False                # deeper fall, but below its 200-day: broken
    # No history: still on the board, but its dip is unknown — not "no dip".
    assert b["VEA"]["dip"] is None and b["VEA"]["pct_off_52w_high"] is None
    assert b["QQQM"]["price"] == 251.65 and b["QQQM"]["today_pct"] == 0.66
    # No live quote: today's move is unknown, never 0.
    assert b["VEA"]["today_pct"] is None


def test_ideas_only_for_symbols_the_context_priced():
    ctx = {"etfs": ask.etf_board(BUILD, ROWS, QUOTES, ["QQQM", "GLDM"]),
           "etf_pick": {"symbol": "XLV"}}
    ideas = [{"symbol": "qqqm", "kind": "etf"}, {"symbol": "ARKK", "kind": "etf"},
             {"symbol": "QQQM", "kind": "etf"}, {"symbol": "XLV", "kind": "etf"},
             {"symbol": "GLDM", "kind": "etf"}]
    kept, dropped = ask.check_ideas(ideas, ctx)
    assert [k["symbol"] for k in kept] == ["QQQM", "XLV", "GLDM"]   # deduped, capped at 3
    assert dropped == ["ARKK"]
    # Re-stamped from the context, not the model's numbers.
    assert kept[0]["price"] == 251.65 and kept[0]["pct_off_52w_high"] == -8.53
    assert kept[0]["fee_pct"] == 0.15
    assert kept[1]["price"] is None                 # the pick card carries no price row


def test_question_cleaning_and_holdings_filter():
    assert ask.clean_question("  which   ETF\n today? ") == "which ETF today?"
    assert ask.clean_question(None) == ""
    assert len(ask.clean_question("x" * 900)) == ask.MAX_QUESTION_CHARS
    h = ask.holdings_rows([{"symbol": "voo", "weight_pct": 41.26}, {"symbol": "bad ticker!"},
                           {"symbol": ""}])
    assert h == [{"symbol": "VOO", "weight_pct": 41.3}]


def test_pick_brief_absent_when_never_run():
    assert ask.pick_brief({"available": False}) is None
    b = ask.pick_brief({"available": True, "date": "2026-11-02", "stale": False, "status": "none",
                        "pick": {"runners_up": [{"symbol": "XLE", "why_not": "oil spike"}],
                                 "closest": {"symbol": "XLV"}}})
    assert b["closest"] == "XLV" and b["runners_up"][0]["symbol"] == "XLE" and b["symbol"] is None


def test_endpoint_rejects_empty_question_and_restamps(monkeypatch):
    import pytest
    from fastapi import HTTPException

    from app import analyst, main

    async def fake_ctx(holdings):
        return {"today": "2026-11-02", "missing": ["macro"],
                "etfs": ask.etf_board(BUILD, ROWS, QUOTES, ["QQQM"])}

    async def fake_ask(q, ctx, deep=True):
        return analyst.AskAnswer(answer="QQQM is 8.5% off its high.", ideas=[
            analyst.AskIdea(symbol="QQQM", kind="etf", headline="h", why="w", watch_out="x"),
            analyst.AskIdea(symbol="TSLA", kind="stock", headline="h", why="w", watch_out="x")]), \
            {"model": "m", "cost_usd": 0.01}

    monkeypatch.setattr(main, "_ask_context", fake_ctx)
    monkeypatch.setattr(main, "analyst_ask", fake_ask)
    monkeypatch.setattr(main.usage_store, "record", lambda *a, **k: None)
    with pytest.raises(HTTPException) as e:
        asyncio.run(main.ask_endpoint(main.AskRequest(question="   ")))
    assert e.value.status_code == 422
    out = asyncio.run(main.ask_endpoint(main.AskRequest(question="Find an ETF to invest in today")))
    assert [i["symbol"] for i in out["ideas"]] == ["QQQM"] and out["dropped"] == ["TSLA"]
    assert out["missing"] == ["macro"] and out["none_reason"] is None
