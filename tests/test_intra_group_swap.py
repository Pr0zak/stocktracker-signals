"""A sell-and-rebuy inside one exposure group must never execute.

2026-08-06, first live tick after US index funds were merged into one group: the model sold all 3 SPY
to buy VTI, arguing 0.095% is a worse expense ratio than 0.030%. Correct for new money, wrong for an
existing holding — it banked $85.56 of SHORT-TERM gains to save ~$1.50/yr in fees, and the buy was
then blocked by the group cap so the proceeds never reached VTI. Cash went 39.9% -> 58.3%.

The prompt has said "NEVER sell one to buy its equivalent" since the sandbox was built. It was
ignored, which is the whole argument for enforcing it in the validator rather than a paragraph.
"""
from __future__ import annotations

import tempfile

import pytest

from app import sandbox_job


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setenv("SIGNALS_DATA_DIR", tempfile.mkdtemp())


GROUPS = {"SPY": "US_EQUITY", "VTI": "US_EQUITY", "VOO": "US_EQUITY",
          "VXUS": "VXUS", "FBTC": "BTC", "AMZN": "AMZN"}


def group_of(sym):
    return GROUPS.get(sym.upper(), sym.upper())


def _blocked(orders):
    return sandbox_job.intra_group_swaps(orders, group_of=group_of)


def buy(sym, **kw):
    return {"symbol": sym, "side": "buy", "dollars": 1000.0, "conviction": 70, **kw}


def sell(sym, **kw):
    return {"symbol": sym, "side": "sell", "shares": 3.0, "conviction": 70, **kw}


# ------------------------------------------------------------------ the trade that happened

def test_the_spy_to_vti_swap_is_blocked_on_both_legs():
    orders = [sell("SPY"), buy("VTI")]
    assert _blocked(orders) == {0, 1}


def test_a_three_way_shuffle_inside_one_group_is_blocked():
    orders = [sell("SPY"), buy("VTI"), buy("VOO")]
    assert _blocked(orders) == {0, 1, 2}


# ------------------------------------------------------------------ legitimate activity survives

def test_a_lone_sell_is_untouched():
    """Trimming a group that is over its cap is legitimate and must still work."""
    assert _blocked([sell("SPY")]) == set()


def test_a_lone_buy_is_untouched():
    assert _blocked([buy("VTI")]) == set()


def test_selling_one_group_to_buy_a_DIFFERENT_one_is_untouched():
    """That is a real allocation change — governed by the prompt's sell rules, not this one."""
    assert _blocked([sell("SPY"), buy("VXUS")]) == set()


def test_unrelated_groups_in_the_same_tick_are_untouched():
    assert _blocked([buy("VXUS"), buy("FBTC"), sell("AMZN")]) == set()


def test_only_the_offending_group_is_blocked():
    orders = [sell("SPY"), buy("VTI"), buy("FBTC")]
    assert _blocked(orders) == {0, 1}


def test_case_is_normalised():
    assert _blocked([sell("spy"), buy("vti")]) == {0, 1}


def test_unrecognised_sides_are_ignored():
    assert _blocked([{"symbol": "SPY", "side": "hold"}, buy("VTI")]) == set()


def test_a_blank_symbol_does_not_crash():
    assert _blocked([{"symbol": "", "side": "sell"}, buy("VTI")]) == set()


# ------------------------------------------------------------------ end to end through the ledger

def _settings(**over):
    s = {"master_enabled": True, "max_position_pct": 25.0, "cash_floor_pct": 5.0,
         "min_conviction_to_trade": 55, "max_trades_per_tick": 4, "max_new_positions_per_tick": 2,
         "max_turnover_pct": 0.0, "account_type": "margin", "respect_entry_zones": False,
         "slippage_bps": 5, "avoid_wash_sales": False}
    s.update(over)
    return s


def test_neither_leg_reaches_the_ledger_and_both_are_audited():
    blob = {"cash": 5000.0, "funded_total": 10000.0, "realized_pl_total": 0.0,
            "settings": _settings(),
            "positions": [{"symbol": "SPY", "shares": 3.0, "avg_cost": 740.0,
                           "exposure_group": "US_EQUITY"}]}
    prices = {"SPY": 768.57, "VTI": 381.0}
    new_blob, filled, skipped = sandbox_job.validate_and_fill(
        blob, [sell("SPY"), buy("VTI")], lambda s: prices.get(s.upper()),
        group_of=group_of, source="test",
    )
    assert filled == []                                   # nothing executed
    assert {r["symbol"] for r in skipped} == {"SPY", "VTI"}   # both recorded, not swallowed
    assert all("same-exposure swap" in r["skip_reason"] for r in skipped)
    assert new_blob["realized_pl_total"] == 0.0           # no tax event
    assert len(new_blob["positions"]) == 1                # SPY still held


def test_a_cap_trim_with_no_rebuy_still_executes():
    blob = {"cash": 0.0, "funded_total": 10000.0, "realized_pl_total": 0.0,
            "settings": _settings(),
            "positions": [{"symbol": "SPY", "shares": 3.0, "avg_cost": 740.0,
                           "exposure_group": "US_EQUITY"}]}
    _, filled, _ = sandbox_job.validate_and_fill(
        blob, [sell("SPY")], lambda s: {"SPY": 768.57}.get(s.upper()),
        group_of=group_of, source="test",
    )
    assert filled and filled[0]["symbol"] == "SPY"
