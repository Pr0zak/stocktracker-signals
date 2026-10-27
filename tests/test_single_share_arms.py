from __future__ import annotations

import importlib
import tempfile
from pathlib import Path

import pytest

from app import sandbox_job, sandbox_store


@pytest.fixture()
def store(monkeypatch):
    """A fresh sandbox_store bound to a throwaway data dir."""
    d = tempfile.mkdtemp()
    monkeypatch.setenv("SIGNALS_DATA_DIR", d)
    importlib.reload(sandbox_store)
    sandbox_store._DATA_DIR = Path(d)
    sandbox_store._cache = {sandbox_store.MAIN_ARM: sandbox_store._load(sandbox_store.MAIN_ARM)}
    return sandbox_store


def test_single_share_rules_pick_selects_top_technical_candidate():
    candidates = [
        {
            "symbol": "WEAK",
            "price": 50.0,
            "technicals": {
                "rel_strength_3mo_vs_benchmark": -5.0,
                "golden_cross": False,
                "pct_vs_sma50": -3.0,
                "rsi14": 35.0,
                "macd_hist": -0.5,
            },
        },
        {
            "symbol": "STRONG",
            "price": 120.0,
            "technicals": {
                "rel_strength_3mo_vs_benchmark": 18.0,
                "golden_cross": True,
                "pct_vs_sma50": 6.5,
                "rsi14": 56.0,
                "macd_hist": 1.2,
            },
        },
        {
            "symbol": "OVERBOUGHT",
            "price": 200.0,
            "technicals": {
                "rel_strength_3mo_vs_benchmark": 12.0,
                "golden_cross": True,
                "pct_vs_sma50": 8.0,
                "rsi14": 82.0,
                "macd_hist": 0.8,
            },
        },
    ]

    res = sandbox_job.single_share_rules_pick(
        candidates,
        held=[],
        exclude=set(),
        price_of=lambda s: {"WEAK": 50.0, "STRONG": 120.0, "OVERBOUGHT": 200.0}.get(s),
    )
    orders = res["orders"]
    assert len(orders) == 1
    assert orders[0]["symbol"] == "STRONG"
    assert orders[0]["side"] == "buy"
    assert orders[0]["shares"] == 1.0
    assert orders[0]["conviction"] >= 60
    assert "Mechanical 1-share buy" in res["posture"]


def test_single_share_rules_pick_respects_exclusions():
    candidates = [
        {
            "symbol": "STRONG",
            "price": 120.0,
            "technicals": {
                "rel_strength_3mo_vs_benchmark": 25.0,
                "golden_cross": True,
                "pct_vs_sma50": 10.0,
                "rsi14": 55.0,
                "macd_hist": 1.0,
            },
        },
    ]
    res = sandbox_job.single_share_rules_pick(
        candidates,
        held=[],
        exclude={"STRONG"},
        price_of=lambda s: 120.0,
    )
    assert res["orders"] == []
    assert "No candidate cleared" in res["posture"]


def test_apply_jit_deposit():
    blob = {
        "cash": 5.0,
        "funded_total": 5.0,
        "benchmark": {"symbol": "^GSPC", "shares": 0.01, "cost_basis": 5.0},
    }
    trade = sandbox_job.apply_jit_deposit(
        blob,
        amount=195.50,
        spy_price=500.0,
        symbol="NVDA",
    )
    assert blob["cash"] == 200.50
    assert blob["funded_total"] == 200.50
    assert blob["benchmark"]["shares"] == pytest.approx(0.01 + 195.50 / 500.0, abs=1e-5)
    assert blob["benchmark"]["cost_basis"] == 200.50
    assert trade["side"] == "deposit"
    assert trade["source"] == "jit_funding"
    assert trade["gross"] == 195.50
    assert trade["cash_after"] == 200.50
    assert "NVDA" in trade["reason"]


@pytest.mark.parametrize("price", [35.1, 35.1049, 34.99, 162.4449, 79.59, 104.8246])
def test_jit_amount_always_covers_the_fill_validate_and_fill_charges(price):
    # 2026-10-27: SCHD at $35.1049 with 5 bps slippage fills at $35.1224; the old top-up rounded that
    # to $35.12 and validate_and_fill refused the buy. Whatever the price, cash after the top-up must
    # buy the share.
    cash = 0.0
    need = sandbox_job.jit_amount_needed(price, 5, cash)
    blob = {
        "cash": round(cash + need, 2),
        "funded_total": round(cash + need, 2),
        "positions": [],
        "settings": {"single_share": True, "funding_mode": "jit", "allow_sells": False,
                     "cash_floor_pct": 0.0, "slippage_bps": 5},
    }
    orders = [{"symbol": "SCHD", "side": "buy", "conviction": 75, "shares": 1.0}]
    _, filled, skipped = sandbox_job.validate_and_fill(
        blob, orders, price_of=lambda s: price, group_of=lambda s: "US")
    assert skipped == [] and len(filled) == 1, skipped
    assert need - price * 1.0005 < 0.01  # never more than a cent over


def test_jit_amount_is_zero_when_cash_already_covers_the_share():
    assert sandbox_job.jit_amount_needed(35.10, 5, 40.0) == 0.0
    assert sandbox_job.jit_amount_needed(35.0, 0, 35.0) == 0.0
    assert sandbox_job.jit_amount_needed(35.0, 0, 34.0) == 1.0


def test_validate_and_fill_single_share_buys_fixed_one_share_and_uncapped():
    # Account has $2,000 cash. Stock is $1,800.
    # Normal cap of 20% would allow only $400, but single_share is uncapped.
    blob = {
        "cash": 2000.0,
        "funded_total": 2000.0,
        "positions": [],
        "settings": {
            "single_share": True,
            "funding_mode": "jit",
            "allow_sells": False,
            "max_position_pct": 20.0,  # normally binds at $400
            "cash_floor_pct": 10.0,    # normally reserves $200
            "slippage_bps": 0,
        },
    }
    orders = [{"symbol": "EXPENSIVE", "side": "buy", "conviction": 90, "dollars": 9999.0}]
    new_blob, filled, skipped = sandbox_job.validate_and_fill(
        blob,
        orders,
        price_of=lambda s: 1800.0,
        group_of=lambda s: "TECH",
    )
    assert len(filled) == 1
    assert filled[0]["symbol"] == "EXPENSIVE"
    assert filled[0]["shares"] == 1.0
    assert filled[0]["price"] == 1800.0
    assert filled[0]["gross"] == 1800.0
    assert new_blob["cash"] == 200.0
    assert len(new_blob["positions"]) == 1
    assert new_blob["positions"][0]["symbol"] == "EXPENSIVE"
    assert new_blob["positions"][0]["shares"] == 1.0


def test_validate_and_fill_single_share_disallows_sells():
    blob = {
        "cash": 100.0,
        "funded_total": 500.0,
        "positions": [{"symbol": "AAPL", "shares": 2.0, "avg_cost": 200.0, "exposure_group": "TECH"}],
        "settings": {
            "single_share": True,
            "allow_sells": False,
        },
    }
    orders = [{"symbol": "AAPL", "side": "sell", "shares": 1.0, "conviction": 80}]
    new_blob, filled, skipped = sandbox_job.validate_and_fill(
        blob,
        orders,
        price_of=lambda s: 220.0,
        group_of=lambda s: "TECH",
    )
    assert len(filled) == 0
    assert len(skipped) == 1
    assert "sells are disabled" in skipped[0]["skip_reason"]
    assert new_blob["positions"][0]["shares"] == 2.0
    assert new_blob["cash"] == 100.0


def test_validate_and_fill_single_share_limits_to_one_buy_per_tick():
    blob = {
        "cash": 1000.0,
        "funded_total": 1000.0,
        "positions": [],
        "settings": {
            "single_share": True,
            "slippage_bps": 0,
        },
    }
    orders = [
        {"symbol": "FIRST", "side": "buy", "conviction": 90, "shares": 1.0},
        {"symbol": "SECOND", "side": "buy", "conviction": 80, "shares": 1.0},
    ]
    new_blob, filled, skipped = sandbox_job.validate_and_fill(
        blob,
        orders,
        price_of=lambda s: 100.0,
        group_of=lambda s: "US_EQUITY",
    )
    assert len(filled) == 1
    assert filled[0]["symbol"] == "FIRST"
    assert filled[0]["shares"] == 1.0
    assert len(skipped) == 1
    assert skipped[0]["symbol"] == "SECOND"
    assert skipped[0]["skip_reason"] == "max_trades_per_tick reached"


def test_ensure_single_share_arms(store):
    store.ensure_single_share_arms()
    arms = store.list_arms()
    assert "oneshare" in arms
    assert "oneshare-rules" in arms

    arm_llm = store.get("oneshare")
    assert arm_llm["engine"] == "llm"
    assert arm_llm["settings"]["single_share"] is True
    assert arm_llm["settings"]["funding_mode"] == "jit"
    assert arm_llm["settings"]["allow_sells"] is False
    assert arm_llm["settings"]["master_enabled"] is True

    arm_rules = store.get("oneshare-rules")
    assert arm_rules["engine"] == "rules"
    assert arm_rules["settings"]["single_share"] is True
    assert arm_rules["settings"]["funding_mode"] == "jit"
    assert arm_rules["settings"]["allow_sells"] is False
    assert arm_rules["settings"]["master_enabled"] is True

    # Idempotent
    store.ensure_single_share_arms()
    assert store.list_arms().count("oneshare") == 1
    assert store.list_arms().count("oneshare-rules") == 1
