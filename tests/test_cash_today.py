"""The `cash_today` block: what the book can spend now and when the next deposit lands.

Added after the 2026-10-30 review found the LLM arms proposing buys against deposits that had already
been paid (picky: 25 cash skips against 5 fills in October). These pin the dates and the arithmetic
the model is now told, since a wrong "next deposit" date would mislead it in the same way.
"""
from __future__ import annotations

import datetime as dt

from app.sandbox_job import ET, cash_for_prompt, next_deposit


def _at(day: str) -> dt.datetime:
    return dt.datetime.fromisoformat(day + "T15:45:00").replace(tzinfo=ET)


SEMI = {"monthly_deposit": 250.0, "deposit_frequency": "semimonthly", "cash_floor_pct": 0.0}


def _nd(settings, day, last):
    return next_deposit({"settings": settings, "last_deposit_period": last}, now=_at(day))


def test_semimonthly_after_the_15th_is_next_month_first_trading_day():
    # 2026-11-01 is a Sunday, so the deposit lands Monday the 2nd — the case behind this file.
    assert _nd(SEMI, "2026-10-30", "2026-10-H2") == {"date": "2026-11-02", "weekday": "Monday",
                                                    "days_away": 3, "dollars": 250.0}


def test_open_but_unpaid_window_lands_on_the_next_tick():
    # Sunday the 1st: H1 is due but no tick runs until Monday. Read naively from the calendar this
    # says the 16th, which is the bug the live replay caught.
    assert _nd(SEMI, "2026-11-01", "2026-10-H2")["date"] == "2026-11-02"
    # A trading day whose deposit the tick deferred (no benchmark quote): the next tick, not today.
    assert _nd(SEMI, "2026-11-02", "2026-10-H2")["date"] == "2026-11-03"


def test_semimonthly_before_the_15th_is_the_15th_or_next_trading_day():
    assert _nd(SEMI, "2026-10-02", "2026-10-H1")["date"] == "2026-10-15"
    # 2026-11-15 is a Sunday.
    assert _nd(SEMI, "2026-11-03", "2026-11-H1")["date"] == "2026-11-16"


def test_monthly_skips_the_15th_and_rolls_the_year():
    s = {"monthly_deposit": 500.0, "deposit_frequency": "monthly"}
    assert _nd(s, "2026-10-02", "2026-10")["date"] == "2026-11-02"
    # 2027-01-01 is New Year's Day, a market holiday.
    assert _nd(s, "2026-12-20", "2026-12")["date"] == "2027-01-04"


def test_no_deposit_means_no_next_deposit():
    assert next_deposit({"settings": {"monthly_deposit": 0}}, now=_at("2026-10-30")) is None
    out = cash_for_prompt({"cash": 50.0, "settings": {"cash_floor_pct": 0.0}}, equity=1000.0,
                          now=_at("2026-10-30"))
    assert out == {"spendable_now_usd": 50.0}


def test_spendable_takes_off_unsettled_proceeds_and_the_floor():
    blob = {"cash": 1500.0, "settings": dict(SEMI, cash_floor_pct=5.0), "last_deposit_period": "2026-10-H2",
            "unsettled": [{"amount": 300.0, "settles_on": "2026-11-02"},
                          {"amount": 999.0, "settles_on": "2026-10-30"}]}   # settled today: spendable
    out = cash_for_prompt(blob, equity=10000.0, now=_at("2026-10-30"))
    assert out["spendable_now_usd"] == 700.0          # 1500 - 300 unsettled - 500 floor
    assert out["unsettled_usd"] == 300.0
    assert out["next_deposit"]["date"] == "2026-11-02"


def test_margin_account_can_spend_unsettled_and_never_goes_negative():
    blob = {"cash": 100.0, "settings": {"account_type": "margin", "cash_floor_pct": 10.0},
            "unsettled": [{"amount": 80.0, "settles_on": "2026-11-02"}]}
    assert cash_for_prompt(blob, equity=5000.0, now=_at("2026-10-30"))["spendable_now_usd"] == 0.0


def test_jit_arm_gets_no_block():
    blob = {"cash": 0.0, "settings": dict(SEMI, funding_mode="jit")}
    assert cash_for_prompt(blob, equity=1500.0, now=_at("2026-10-30")) is None


def test_review_and_decision_payloads_carry_the_new_blocks(monkeypatch):
    import asyncio
    import json

    from app import analyst

    seen = []

    async def fake_parse(system, prompt, model_cls, **kw):
        seen.append(json.loads(prompt.split("\n", 1)[1].rsplit("\n\n", 1)[0]))
        return model_cls.model_validate({"approve": True} if model_cls is analyst.ReviewVerdict
                                        else {"posture": "hold", "orders": []}), {}

    monkeypatch.setattr(analyst, "_parse", fake_parse)
    book = {"total_value": 1000.0, "cash_pct": 5.0, "positions": []}
    macro = {"risk_level": "elevated", "catalysts": [{"title": "Hormuz"}]}
    dec, _ = asyncio.run(analyst.sandbox_decision(
        book, [], cash=50.0, settings={}, strategy_note=None, macro=macro,
        cash_today={"spendable_now_usd": 50.0}))
    asyncio.run(analyst.review_decision(dec, book, [], cash=50.0, settings={},
                                        strategy_note=None, macro=macro))
    assert seen[0]["cash_today"] == {"spendable_now_usd": 50.0}
    assert seen[1]["macro"] == macro
    # Absent stays absent: no empty block asserting a calm backdrop.
    asyncio.run(analyst.review_decision(dec, book, [], cash=50.0, settings={}, strategy_note=None))
    assert "macro" not in seen[2]


def test_next_deposit_spells_the_date_three_ways():
    """The model misread an ISO date once ("2026-11-16" became "11 Nov"); weekday + count can't be."""
    nxt = _nd(SEMI, "2026-11-05", "2026-11-H1")
    assert nxt["date"] == "2026-11-16"
    assert nxt["weekday"] == "Monday"
    assert nxt["days_away"] == 11
