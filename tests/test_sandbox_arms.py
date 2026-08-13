"""Comparison arms, and the mechanical control arm that runs in one.

Why these exist: with a single ledger every change to the sandbox is a guess measured against noise.
The account has been ~52% cash against a 12% target for weeks, and nothing about one equity curve can
say whether that is the strategist's plan or the tick's mechanics. Two ledgers on the same tick, same
quotes, same day, differing in exactly one thing, can.

So the properties worth testing are mostly about ISOLATION (an arm must not be able to touch another
arm's money or history) and IDENTITY (an arm id becomes a directory name).
"""
from __future__ import annotations

import importlib
import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture()
def store(monkeypatch):
    """A fresh sandbox_store bound to a throwaway data dir. Re-imported per test because the module
    resolves its data directory and loads `main` at import time."""
    d = tempfile.mkdtemp()
    monkeypatch.setenv("SIGNALS_DATA_DIR", d)
    from app import sandbox_store as s
    importlib.reload(s)
    s._DATA_DIR = Path(d)          # the module-level constant was bound before the reload
    s._cache = {s.MAIN_ARM: s._load(s.MAIN_ARM)}
    return s


# The REAL exposure map, not a stub. A hand-written one here would have said VTI's group is "VTI",
# which is the whole thing the US_EQUITY merge exists to deny — and the fragmentation test below
# would then have passed against a fixture that already encoded the answer rather than against the
# code. That failure mode has bitten this repo before (2026-08-07, the allocation-gap aliasing).
from app.main import _exposure_group as _group  # noqa: E402


# ------------------------------------------------------------------ arm ids reach the filesystem

@pytest.mark.parametrize("bad", [
    "../escape", "..", "a/b", "/abs", "arm\x00", "", " ", "-lead", "x" * 33,
    "a b", "arm.json", "./x", "..\\win",
])
def test_a_hostile_arm_id_is_refused_before_it_becomes_a_path(store, bad):
    """The id is joined to a data path, so validation is the only thing between a query string and
    the filesystem. Whitelisted rather than blacklisted — every one of these fails by construction
    instead of by having been thought of."""
    with pytest.raises(store.ArmError):
        store.validate_arm(bad)


@pytest.mark.parametrize("ok", ["rules", "a", "arm-1", "arm_2", "0", "x" * 32])
def test_reasonable_arm_ids_are_accepted(store, ok):
    assert store.validate_arm(ok) == ok


def test_arm_ids_are_case_folded_to_one_directory(store):
    """Deliberate: on a case-insensitive filesystem `Rules` and `rules` would be one directory but
    two cache keys, and the second would silently write over the first."""
    assert store.validate_arm("Rules") == "rules"
    assert store.validate_arm(" RULES ") == "rules"


def test_main_is_not_deletable(store):
    """It is the real account with the real history."""
    with pytest.raises(store.ArmError):
        store.delete_arm("main")


def test_an_arm_cannot_be_created_twice(store):
    store.create_arm("rules", engine="rules")
    with pytest.raises(store.ArmError):
        store.create_arm("rules", engine="rules")


def test_an_unknown_engine_is_refused(store):
    with pytest.raises(store.ArmError):
        store.create_arm("weird", engine="vibes")


# ------------------------------------------------------------------ isolation

def test_main_keeps_its_original_paths_so_nothing_migrates(store):
    """The live ledger has weeks of history. The safest migration is the one that does not happen."""
    store.save({**store.get("main"), "cash": 1234.0}, "main")
    assert (store._DATA_DIR / "sandbox.json").exists()
    assert not (store._DATA_DIR / "arms" / "main").exists()


def test_two_arms_do_not_share_a_ledger(store):
    store.create_arm("rules", engine="rules")
    store.save({**store.get("main"), "cash": 100.0}, "main")
    store.save({**store.get("rules"), "cash": 900.0}, "rules")
    assert store.get("main")["cash"] == 100.0
    assert store.get("rules")["cash"] == 900.0


def test_two_arms_do_not_share_a_trade_log(store):
    store.create_arm("rules", engine="rules")
    store.append_trade({"symbol": "AAA", "side": "buy"}, "main")
    store.append_trade({"symbol": "BBB", "side": "buy"}, "rules")
    store.append_trade({"symbol": "CCC", "side": "buy"}, "rules")
    assert [t["symbol"] for t in store.read_trades(50, "main")] == ["AAA"]
    assert sorted(t["symbol"] for t in store.read_trades(50, "rules")) == ["BBB", "CCC"]


def test_two_arms_do_not_share_a_nav_series(store):
    store.create_arm("rules", engine="rules")
    store.append_nav({"date": "2026-08-13", "equity": 1.0}, "main")
    store.append_nav({"date": "2026-08-13", "equity": 2.0}, "rules")
    assert [r["equity"] for r in store.read_nav(None, "main")] == [1.0]
    assert [r["equity"] for r in store.read_nav(None, "rules")] == [2.0]


def test_saving_a_blob_lands_on_the_arm_it_came_from(store):
    """A caller that read one arm and wrote it back must not be able to land it on another."""
    store.create_arm("rules", engine="rules")
    b = store.get("rules")
    b["cash"] = 555.0
    store.save(b)                                   # no explicit arm
    assert store.get("rules")["cash"] == 555.0
    assert store.get("main")["cash"] == 0.0


def test_the_path_wins_over_a_blob_that_claims_another_arm(store):
    """A hand-edited or wrongly-restored file must not be able to rename itself into another book."""
    store.create_arm("rules", engine="rules")
    f = store._paths("rules")[0]
    d = json.loads(f.read_text()); d["arm"] = "main"
    f.write_text(json.dumps(d))
    store._cache.pop("rules", None)
    assert store.get("rules")["arm"] == "rules"


def test_resetting_an_arm_keeps_its_identity(store):
    """A reset arm is the same experiment starting over. Reverting `engine` to the default would
    quietly turn the control arm into a second copy of main."""
    store.create_arm("rules", engine="rules", label="Mechanical")
    store.reset("rules")
    b = store.get("rules")
    assert b["engine"] == "rules" and b["label"] == "Mechanical" and b["cash"] == 0.0


def test_a_new_arm_starts_disabled_and_unfunded(store):
    """Seeded settings must not carry `master_enabled` over from a live account."""
    store.save({**store.get("main"), "settings": {**store.get("main")["settings"],
                                                  "master_enabled": True}}, "main")
    b = store.create_arm("rules", engine="rules")
    assert b["settings"]["master_enabled"] is False
    assert b["cash"] == 0.0 and b["funded_total"] == 0.0


def test_a_new_arm_inherits_the_settings_it_is_being_compared_against(store):
    """Otherwise the A/B has forty uncontrolled variables instead of one."""
    m = store.get("main")
    m["settings"]["max_position_pct"] = 33.0
    store.save(m, "main")
    b = store.create_arm("rules", engine="rules", settings={"cash_floor_pct": 1.0})
    assert b["settings"]["max_position_pct"] == 33.0     # inherited
    assert b["settings"]["cash_floor_pct"] == 1.0        # the variable under test


def test_a_cloned_arm_starts_from_the_same_book_and_the_same_starting_line(store):
    """An arm started empty today against one invested for three weeks differs by a head start as
    well as by its strategy — and the head start is the larger effect for months."""
    m = store.get("main")
    m.update({"cash": 5_618.47, "funded_total": 10_500.0, "realized_pl_total": 91.67,
              "positions": [{"symbol": "VTI", "shares": 7.0, "avg_cost": 365.0}],
              "benchmark": {"symbol": "^GSPC", "shares": 1.84, "cost_basis": 10_500.0},
              "last_strategy_note": {"cash_target_pct": 12.0, "targets": []}})
    store.save(m, "main")
    b = store.create_arm("rules", engine="rules", clone_from="main")
    assert b["cash"] == 5_618.47 and b["funded_total"] == 10_500.0
    assert [p["symbol"] for p in b["positions"]] == ["VTI"]
    assert b["benchmark"]["shares"] == 1.84      # same starting line for excess return
    assert b["last_strategy_note"] is not None


def test_a_cloned_arm_does_not_inherit_the_day_cursor(store):
    """Inheriting today's last_tick_date would gate the new arm out of its own first tick."""
    store.save({**store.get("main"), "last_tick_date": "2026-08-13",
                "last_decision_date": "2026-08-13"}, "main")
    b = store.create_arm("rules", engine="rules", clone_from="main")
    assert b["last_tick_date"] is None and b["last_decision_date"] is None


def test_a_cloned_arm_does_not_inherit_trades_it_never_made(store):
    """Unsettled T+1 proceeds and the wash-sale clock belong to the other arm's history."""
    store.save({**store.get("main"), "unsettled": [{"amount": 500.0, "settles_on": "2026-08-14"}],
                "recent_loss_sales": {"VXUS": 1786000000.0}}, "main")
    b = store.create_arm("rules", engine="rules", clone_from="main")
    assert b["unsettled"] == [] and b["recent_loss_sales"] == {}


def test_cloning_copies_the_book_not_a_reference_to_it(store):
    store.save({**store.get("main"),
                "positions": [{"symbol": "VTI", "shares": 7.0, "avg_cost": 365.0}]}, "main")
    store.create_arm("rules", engine="rules", clone_from="main")
    m = store.get("main")
    m["positions"][0]["shares"] = 99.0
    store.save(m, "main")
    assert store.get("rules")["positions"][0]["shares"] == 7.0


def test_deleting_an_arm_removes_its_data(store):
    store.create_arm("rules", engine="rules")
    store.append_trade({"symbol": "AAA"}, "rules")
    store.delete_arm("rules")
    assert "rules" not in store.list_arms()
    assert store.read_trades(50, "rules") == []


def test_list_arms_always_includes_main_first(store):
    store.create_arm("zzz", engine="rules")
    store.create_arm("aaa", engine="rules")
    assert store.list_arms() == ["main", "aaa", "zzz"]


# ------------------------------------------------------------------ the mechanical arm

from app import sandbox_job  # noqa: E402

PLAN = {"cash_target_pct": 12.0, "targets": [
    {"exposure_group": "US_EQUITY", "target_pct": 25.0},
    {"exposure_group": "SCHD", "target_pct": 12.0},
    {"exposure_group": "GOOGL", "target_pct": 8.0},
]}
PX = {"VTI": 384.42, "SPLG": 78.0, "SCHD": 34.42, "GOOGL": 346.22, "^GSPC": 6000.0}


def _blob(cash=10_000.0, positions=None, **over):
    s = {"cash_floor_pct": 5.0, "max_trades_per_tick": 4, "preferred_btc_etf": "FBTC"}
    s.update(over.pop("settings", {}))
    return {"cash": cash, "positions": positions or [], "settings": s}


def _decide(blob, plan=PLAN, px=None):
    p = px or PX
    return sandbox_job.rules_decision(
        blob, plan=plan, group_of=_group, price_of=lambda s: p.get(s.upper()))


def test_it_buys_toward_the_underweight_targets():
    d = _decide(_blob())
    assert d["orders"], d["posture"]
    assert {o["symbol"] for o in d["orders"]} <= {"SPLG", "SCHD", "GOOGL"}
    assert all(o["side"] == "buy" for o in d["orders"])


def test_it_never_sells():
    """Buy-only on purpose. Selling to reach a target makes this a rebalancer instead of a deployment
    baseline, and inherits the exact behaviour the account already bans for the analyst."""
    over = _blob(cash=100.0, positions=[{"symbol": "SCHD", "shares": 900.0, "avg_cost": 30.0}])
    assert all(o["side"] == "buy" for o in _decide(over)["orders"])


def test_it_holds_the_plan_s_cash_target_rather_than_the_lower_floor():
    """The floor is a risk limit and the target is an intention; a mechanical arm may not overrule
    its own instructions by deploying through either."""
    d = _decide(_blob(cash=10_000.0))
    spend = sum(o["dollars"] for o in d["orders"])
    assert spend <= 10_000.0 - 0.12 * 10_000.0 + 0.01


def test_the_floor_binds_when_it_is_the_higher_of_the_two():
    d = _decide(_blob(cash=10_000.0, settings={"cash_floor_pct": 40.0}))
    assert sum(o["dollars"] for o in d["orders"]) <= 6_000.01


def test_it_adds_to_the_vehicle_already_held_rather_than_fragmenting_the_group():
    """Topping up a US_EQUITY target held as VTI by buying SPLG would split one exposure across two
    vehicles — the fragmentation the grouping exists to prevent."""
    d = _decide(_blob(positions=[{"symbol": "VTI", "shares": 1.0, "avg_cost": 380.0}]))
    syms = {o["symbol"] for o in d["orders"]}
    assert "VTI" in syms and "SPLG" not in syms


def test_it_picks_the_cheapest_vehicle_for_a_group_it_does_not_yet_hold():
    d = _decide(_blob())
    assert "SPLG" in {o["symbol"] for o in d["orders"]}


def test_a_target_worth_less_than_one_share_is_left_alone():
    """A group within a share of target would otherwise emit a blocked order every single day."""
    plan = {"cash_target_pct": 0.0, "targets": [{"exposure_group": "GOOGL", "target_pct": 1.0}]}
    d = _decide(_blob(cash=10_000.0), plan=plan)     # 1% of ~10k = $100 vs a $346 share
    assert d["orders"] == []


def test_no_plan_is_reported_as_no_plan_not_as_on_target():
    assert "No standing plan" in _decide(_blob(), plan=None)["posture"]
    assert _decide(_blob(), plan={"targets": []})["orders"] == []


def test_a_fully_allocated_book_proposes_nothing():
    d = _decide(_blob(cash=0.0, positions=[{"symbol": "VTI", "shares": 10.0, "avg_cost": 380.0}]))
    assert d["orders"] == []


def test_it_respects_the_per_tick_trade_cap():
    plan = {"cash_target_pct": 0.0, "targets": [
        {"exposure_group": g, "target_pct": 10.0} for g in ("SCHD", "GOOGL", "US_EQUITY")]}
    d = _decide(_blob(cash=100_000.0, settings={"max_trades_per_tick": 2}), plan=plan)
    assert len(d["orders"]) == 2


def test_two_plan_labels_for_one_group_are_collapsed_before_sizing():
    """A plan written before a regrouping names labels that are the same group today; sizing them
    separately would ask for twice the intended weight."""
    plan = {"cash_target_pct": 0.0, "targets": [
        {"exposure_group": "SCHD", "target_pct": 10.0},
        {"exposure_group": "SCHD", "target_pct": 10.0}]}
    d = _decide(_blob(cash=10_000.0), plan=plan)
    assert len(d["orders"]) == 1
    assert d["orders"][0]["dollars"] == pytest.approx(2_000.0, abs=1.0)


def test_its_orders_survive_the_validator_and_conserve_cash():
    """The mechanical arm proposes; validate_and_fill is still the sole authority, exactly as for the
    analyst. If the two disagree about what is affordable, the validator wins."""
    blob = {**_blob(cash=10_000.0), "settings": {
        "cash_floor_pct": 5.0, "max_trades_per_tick": 4, "max_position_pct": 100.0,
        "min_conviction_to_trade": 55, "slippage_bps": 5, "respect_entry_zones": True,
        "account_type": "margin", "avoid_wash_sales": False, "max_turnover_pct": 0.0}}
    d = _decide(blob)
    after, filled, _ = sandbox_job.validate_and_fill(
        blob, d["orders"], lambda s: PX.get(s.upper()), group_of=_group, source="rules_tick")
    assert filled
    # The ledger's own running number, not a sum of per-row `gross` — those are each rounded to the
    # cent for display and re-adding them reintroduces the rounding the ledger deliberately avoids.
    # (validate_and_fill's internal conservation assert is the real guard; reaching here means it
    # held.)
    assert after["cash"] == pytest.approx(filled[-1]["cash_after"], abs=0.001)
    assert after["cash"] < 10_000.0


def test_a_mechanical_order_is_labelled_as_one():
    """The reason line is read on a phone next to the analyst's. It must not look like a judgement."""
    o = _decide(_blob())["orders"][0]
    assert "mechanical" in o["reason"].lower()
    assert o["conviction"] == 100
