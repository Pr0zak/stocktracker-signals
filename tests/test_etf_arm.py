"""The ETF-only arms: cap groups, cheapest-copy routing, the fund-list guard and the no-AI mix.

Built against the REAL maps (etf_arm.ETF_GROUPS, fund_cost.GROUPS, the fund catalogue), not stubs.
A hand-written group map in a fixture can encode the very answer a test exists to check — this repo
has been bitten by that before (2026-08-07, the allocation-gap aliasing) — so the invariants below
are asserted over the data the arms actually run on.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import importlib
import tempfile
from pathlib import Path

import pytest

from app import etf_arm, fund_catalog, fund_cost, sandbox_job

GOF = etf_arm.group_of


# ------------------------------------------------------------------ the group map

def test_every_grouped_ticker_is_a_fund_the_arm_can_buy():
    """A typo in ETF_GROUPS would give a real fund no group and a fake one a cap."""
    uni = etf_arm.universe_set()
    stray = [s for members in etf_arm.ETF_GROUPS.values() for s in members if s not in uni]
    assert stray == []


def test_no_ticker_sits_in_two_groups():
    seen: dict[str, str] = {}
    for g, members in etf_arm.ETF_GROUPS.items():
        for s in members:
            assert s not in seen, f"{s} is in both {seen[s]} and {g}"
            seen[s] = g


def test_funds_holding_the_same_index_always_share_one_cap():
    """The cheapest-copy router moves a buy between members of a fund_cost group. If two of them
    sat in different cap groups, routing would move money from one cap to another behind the
    analyst's back — and a book could hold the S&P twice, under two caps."""
    uni = etf_arm.universe_set()
    for g in fund_cost.GROUPS:
        caps = {GOF(m) for m in g.members if m in uni}
        assert len(caps) <= 1, f"{g.id} spans cap groups {caps}"


def test_mutual_funds_are_not_on_the_list():
    """They price once a day at the closing NAV; this ledger fills near the close at a live price."""
    assert not (etf_arm.universe_set() & fund_cost.MUTUAL_FUNDS)
    assert "VOO" in etf_arm.universe_set() and "FXAIX" not in etf_arm.universe_set()


def test_broad_groups_and_representatives_name_real_groups():
    assert etf_arm.BROAD_GROUPS <= set(etf_arm.ETF_GROUPS)
    for g, reps in etf_arm.REPRESENTATIVES.items():
        assert g in etf_arm.ETF_GROUPS
        assert all(GOF(r) == g for r in reps), g


def test_a_fund_outside_the_map_is_its_own_group():
    assert GOF("ARKK") == "ARKK"
    assert GOF("vti") == "US_EQUITY"


def test_growth_funds_share_the_s_and_p_cap():
    """The case the map exists for: four growth funds each under 20% would be 80% in one bet."""
    assert {GOF(s) for s in ("VUG", "SCHG", "IWF", "VONG", "VOO", "VTI")} == {"US_EQUITY"}


def test_a_same_group_swap_is_refused_in_the_etf_vocabulary():
    """Selling VOO to buy SPYM is the 2026-08-06 SPY->VTI trade again; the swap guard must see it."""
    orders = [{"symbol": "VOO", "side": "sell"}, {"symbol": "SPYM", "side": "buy"}]
    assert sandbox_job.intra_group_swaps(orders, group_of=GOF) == {0, 1}


# ------------------------------------------------------------------ caps

def test_broad_groups_get_the_broad_cap_and_the_rest_keep_the_narrow_one():
    cap = etf_arm.cap_pct_of({"max_position_pct": 20.0, "broad_position_pct": 60.0})
    assert cap("US_EQUITY") == 60.0 and cap("INTL") == 60.0 and cap("US_BONDS") == 60.0
    assert cap("US_TECH") == 20.0 and cap("GOLD") == 20.0 and cap("ARKK") == 20.0


def test_the_broad_cap_is_never_tighter_than_the_narrow_one():
    cap = etf_arm.cap_pct_of({"max_position_pct": 30.0, "broad_position_pct": 10.0})
    assert cap("US_EQUITY") == 30.0


def _book(cash=10_000.0, positions=None, **settings):
    base = {"max_position_pct": 20.0, "cash_floor_pct": 0.0, "min_conviction_to_trade": 55,
            "max_trades_per_tick": 6, "max_new_positions_per_tick": 6, "max_turnover_pct": 0.0,
            "slippage_bps": 0, "respect_entry_zones": False, "universe": "etf",
            "broad_position_pct": 60.0}
    base.update(settings)
    return {"cash": cash, "positions": positions or [], "settings": base, "realized_pl_total": 0.0}


PRICES = {"VTI": 300.0, "VOO": 550.0, "SPYM": 70.0, "IVV": 600.0, "SPY": 600.0, "XLK": 250.0,
          "VXUS": 70.0, "VEU": 65.0, "IXUS": 75.0, "BND": 74.0, "AGG": 99.0, "SCHZ": 23.0,
          "ITOT": 130.0, "SCHB": 25.0, "AAPL": 230.0, "FBTC": 90.0, "IBIT": 60.0, "QQQM": 220.0,
          "QQQ": 530.0}


def _px(s):
    return PRICES.get(s.upper())


def _fill(blob, orders):
    return sandbox_job.validate_and_fill(
        blob, orders, _px, group_of=GOF, cap_pct_of=etf_arm.cap_pct_of(blob["settings"]),
        allowed_buys=etf_arm.universe_set(), now_ts=1_790_000_000.0)


def test_a_broad_index_fund_can_be_bought_past_the_narrow_cap():
    """A 45% whole-market position is an ordinary index portfolio, not a concentration."""
    _, filled, skipped = _fill(_book(), [
        {"symbol": "VTI", "side": "buy", "dollars": 4_500.0, "conviction": 90}])
    assert skipped == [] and round(filled[0]["gross"]) == 4_500


def test_a_sector_fund_still_stops_at_the_narrow_cap():
    _, filled, skipped = _fill(_book(), [
        {"symbol": "XLK", "side": "buy", "dollars": 4_500.0, "conviction": 90}])
    assert filled[0]["gross"] <= 2_000.0 + 1e-6


def test_look_alike_funds_count_against_one_cap():
    held = [{"symbol": "VOO", "shares": 10.0, "avg_cost": 550.0, "exposure_group": "US_EQUITY"}]
    # 5,500 already in US_EQUITY on 15,500 equity; 60% is 9,300, so 3,800 of room is left.
    _, filled, _ = _fill(_book(cash=10_000.0, positions=held), [
        {"symbol": "VTI", "side": "buy", "dollars": 9_000.0, "conviction": 90}])
    assert filled[0]["gross"] <= 3_800.0 + 1e-6


def test_a_stock_buy_is_refused_on_an_etf_arm():
    _, filled, skipped = _fill(_book(), [
        {"symbol": "AAPL", "side": "buy", "dollars": 1_000.0, "conviction": 90}])
    assert filled == [] and skipped[0]["skip_reason"] == sandbox_job.ALLOWED_BUYS_SKIP


def test_a_position_off_the_list_can_still_be_sold():
    """The list governs buys. A holding the arm cannot buy must never become one it cannot sell."""
    held = [{"symbol": "AAPL", "shares": 2.0, "avg_cost": 200.0, "exposure_group": "AAPL",
             "opened_at": 0, "last_add_at": 0}]
    _, filled, skipped = _fill(_book(positions=held), [
        {"symbol": "AAPL", "side": "sell", "shares": 2, "conviction": 90}])
    assert skipped == [] and filled[0]["side"] == "sell"


def test_no_list_means_no_restriction():
    """Every other arm passes allowed_buys=None and must behave exactly as before."""
    _, filled, skipped = sandbox_job.validate_and_fill(
        _book(), [{"symbol": "AAPL", "side": "buy", "dollars": 1_000.0, "conviction": 90}], _px,
        group_of=GOF, now_ts=1_790_000_000.0)
    assert skipped == [] and filled


# ------------------------------------------------------------------ cheapest copy

FEES = {"VOO": (0.03, 1.4e12), "IVV": (0.03, 6e11), "SPYM": (0.02, 7e10), "SPY": (0.0945, 6.5e11),
        "VTI": (0.03, 1.9e12), "ITOT": (0.03, 7e10), "SCHB": (0.03, 3.5e10), "SPTM": (0.03, 1e10),
        "VXUS": (0.05, 5e11), "VEU": (0.04, 6e10), "IXUS": (0.07, 4e10), "BND": (0.03, 1.3e12),
        "AGG": (0.03, 1.2e11), "SCHZ": (0.03, 8e9), "QQQ": (0.20, 3e11), "QQQM": (0.15, 5e10),
        "FBTC": (0.25, 2e10), "IBIT": (0.25, 8e10)}


def _route(orders, positions=(), exclude=None, price_of=_px):
    return etf_arm.route_to_cheapest(orders, positions=list(positions), fees=FEES,
                                     price_of=price_of, exclude=exclude)


def test_a_buy_lands_on_the_cheapest_fund_holding_the_same_index():
    out, notes = _route([{"symbol": "SPY", "side": "buy", "dollars": 1_000.0, "reason": "fill"}])
    assert out[0]["symbol"] == "SPYM" and out[0]["dollars"] == 1_000.0
    assert "SPY→SPYM" in notes[0] and "routed" in out[0]["reason"]


def test_new_money_joins_the_copy_already_held_rather_than_opening_a_second():
    held = [{"symbol": "VOO", "shares": 3.0}]
    out, _ = _route([{"symbol": "SPYM", "side": "buy", "dollars": 500.0}], positions=held)
    assert out[0]["symbol"] == "VOO"


def test_a_tie_on_fee_goes_to_the_bigger_fund():
    """VTI, ITOT, SCHB and SPTM all charge 0.03%; the biggest has the tightest spread."""
    out, _ = _route([{"symbol": "SCHB", "side": "buy", "dollars": 500.0}])
    assert out[0]["symbol"] == "VTI"


def test_sells_are_never_routed():
    held = [{"symbol": "SPY", "shares": 3.0}]
    out, notes = _route([{"symbol": "SPY", "side": "sell", "shares": 3}], positions=held)
    assert out[0]["symbol"] == "SPY" and notes == []


def test_it_never_routes_onto_an_excluded_fund():
    out, _ = _route([{"symbol": "SPY", "side": "buy", "dollars": 900.0}], exclude={"SPYM"})
    assert out[0]["symbol"] in ("VOO", "IVV")


def test_an_unpriced_cheapest_fund_is_passed_over():
    out, _ = _route([{"symbol": "SPY", "side": "buy", "dollars": 900.0}],
                    price_of=lambda s: None if s == "SPYM" else _px(s))
    assert out[0]["symbol"] == "VOO"


def test_an_unknown_fee_is_never_taken_for_a_free_one():
    fees = {**FEES, "SPYM": (None, 1e12)}
    out, _ = etf_arm.route_to_cheapest([{"symbol": "SPY", "side": "buy", "dollars": 900.0}],
                                       positions=[], fees=fees, price_of=_px)
    assert out[0]["symbol"] == "VOO"


def test_bitcoin_is_left_to_the_user_s_chosen_fund():
    """preferred_btc_etf is the user's call; a fee comparison here would override it."""
    out, notes = _route([{"symbol": "IBIT", "side": "buy", "dollars": 500.0}])
    assert out[0]["symbol"] == "IBIT" and notes == []


def test_a_share_count_is_carried_across_as_dollars():
    """VOO ~$550 vs SPYM ~$70: carrying '2 shares' across would buy an eighth of the intent."""
    out, _ = _route([{"symbol": "VOO", "side": "buy", "shares": 2}])
    assert out[0]["symbol"] == "SPYM" and out[0]["dollars"] == 1_100.0 and "shares" not in out[0]


# ------------------------------------------------------------------ the no-AI mix

@pytest.mark.parametrize("years,expect", [(30, 90.0), (25, 90.0), (0, 50.0), (18, 78.8), (-7, 30.0),
                                          (-20, 30.0)])
def test_the_glide_path_shape(years, expect):
    assert etf_arm.glidepath_stock_pct(years, "balanced") == pytest.approx(expect)


def test_risk_tolerance_moves_the_line_and_stays_in_bounds():
    assert etf_arm.glidepath_stock_pct(18, "conservative") == pytest.approx(63.8)
    assert etf_arm.glidepath_stock_pct(30, "aggressive") == 95.0
    assert etf_arm.glidepath_stock_pct(-20, "conservative") == 20.0


def test_the_plan_fills_exactly_the_investable_share():
    """Targets short of (100 - cash) silently become cash — the 2026-08-04 lesson."""
    plan = etf_arm.glidepath_plan({"birth_date": "1978-08-18", "retirement_age": 65,
                                   "risk_tolerance": "balanced", "cash_floor_pct": 2.0},
                                  dt.date(2026, 9, 28))
    total = sum(t["target_pct"] for t in plan["targets"])
    assert total == pytest.approx(100.0 - plan["cash_target_pct"], abs=0.05)
    assert [t["exposure_group"] for t in plan["targets"]] == ["US_EQUITY", "INTL", "US_BONDS"]
    # 48 on that date, 17 years out: 50 + 40*17/25 = 77.2% stocks.
    us, intl, bonds = (t["target_pct"] for t in plan["targets"])
    assert us / (us + intl) == pytest.approx(0.60, abs=0.005)
    assert (us + intl) / 98.0 == pytest.approx(0.772, abs=0.002)
    assert sandbox_job.allocation_gap(plan, max_position_pct=20.0, group_of=GOF,
                                      cap_pct_of=etf_arm.cap_pct_of({"max_position_pct": 20.0})) is None


def test_the_same_plan_is_unreachable_under_a_single_narrow_cap():
    """Why the broad cap exists: without it an index portfolio fails its own audit."""
    plan = etf_arm.glidepath_plan({"current_age": 48, "retirement_age": 65}, dt.date(2026, 9, 28))
    gap = sandbox_job.allocation_gap(plan, max_position_pct=20.0, group_of=GOF)
    assert gap and "US_EQUITY" in gap["targets_over_cap"]


def test_the_no_ai_arm_buys_the_whole_mix_through_the_real_pipeline():
    """rules_decision -> route_to_cheapest -> validate_and_fill, as the tick runs them. Every order
    fills, nothing hits a cap, and cash is conserved."""
    blob = _book(cash=11_250.0, birth_date="1978-08-18", retirement_age=65)
    plan = etf_arm.glidepath_plan(blob["settings"], dt.date(2026, 9, 28))
    d = sandbox_job.rules_decision(blob, plan=plan, group_of=GOF, price_of=_px,
                                   representatives=etf_arm.REPRESENTATIVES)
    assert {o["symbol"] for o in d["orders"]} == {"VTI", "VXUS", "BND"}
    orders, _ = etf_arm.route_to_cheapest(d["orders"], positions=[], fees=FEES, price_of=_px)
    assert {o["symbol"] for o in orders} == {"VTI", "VEU", "BND"}
    new, filled, skipped = _fill(blob, orders)
    assert skipped == [] and len(filled) == 3
    assert new["cash"] == pytest.approx(11_250.0 - sum(r["gross"] for r in filled), abs=0.01)
    assert {GOF(p["symbol"]) for p in new["positions"]} == {"US_EQUITY", "INTL", "US_BONDS"}


def test_without_representatives_the_bond_target_has_nothing_to_buy():
    """The reason rules_decision takes a representatives map: US_BONDS is not a ticker."""
    blob = _book(cash=11_250.0, current_age=48, retirement_age=65)
    plan = etf_arm.glidepath_plan(blob["settings"], dt.date(2026, 9, 28))
    d = sandbox_job.rules_decision(blob, plan=plan, group_of=GOF, price_of=_px)
    assert "BND" not in {o["symbol"] for o in d["orders"]}


# ------------------------------------------------------------------ what the analyst sees

def _build():
    funds = []
    for e in fund_catalog.CATALOG:
        fee = FEES.get(e.symbol, (0.10, 1e9))
        funds.append({"symbol": e.symbol, "name": e.name, "category": e.category,
                      "expense_ratio_pct": fee[0], "net_assets": fee[1],
                      "returns": {"1y": 10.0, "3y": 30.0, "5y": None},
                      "drops": {"1y": -8.0, "3y": -20.0, "5y": None}, "history_start": "2019-01-02"})
    return {"funds": funds, "aligned_to": "2026-09-25"}


def test_the_pool_shows_one_fund_per_index_and_it_is_the_cheapest():
    pool = etf_arm.pool_symbols(_build())
    assert "SPYM" in pool and not ({"VOO", "IVV", "SPY"} & set(pool))
    assert "QQQM" in pool and "QQQ" not in pool
    by_group = {}
    for s in pool:
        g = fund_cost.group_of(s)
        if g:
            assert g.id not in by_group, f"{s} and {by_group[g.id]} hold the same index"
            by_group[g.id] = s
    assert set(pool) <= etf_arm.universe_set()


def test_the_user_s_chosen_bitcoin_and_gold_funds_are_the_ones_shown():
    pool = etf_arm.pool_symbols(_build(), prefer=["FBTC", "GLDM"])
    assert "FBTC" in pool and "IBIT" not in pool and "GLDM" in pool


def test_the_pool_respects_exclusions_and_the_crypto_switch():
    pool = etf_arm.pool_symbols(_build(), exclude={"SPYM"}, allow_crypto_etf=False)
    assert "SPYM" not in pool and ({"VOO", "IVV"} & set(pool))
    assert not [s for s in pool if GOF(s) in ("BTC", "ETH")]


def test_an_unknown_fee_stays_unknown_on_the_row():
    b = _build()
    for f in b["funds"]:
        if f["symbol"] == "ARKK":
            f["expense_ratio_pct"] = None
    row = etf_arm.candidate_row("ARKK", funds=etf_arm.funds_by_symbol(b),
                                tech_row={"price": 60.0, "technicals": {"rsi14": 50}})
    assert row["fee_pct"] is None and row["fee_per_10k_usd"] is None
    assert row["return_pct"]["5y"] is None


def test_a_fund_with_no_price_is_not_shown():
    assert etf_arm.candidate_row("VTI", funds={}, tech_row=None) is None
    assert etf_arm.candidate_row("VTI", funds={}, tech_row={"price": None}) is None


def test_every_group_the_plan_can_name_has_its_cap_and_its_history():
    summary = etf_arm.group_summary(_build(), settings={"max_position_pct": 20.0})
    assert summary["US_EQUITY"]["cap_pct"] == 60.0 and summary["US_TECH"]["cap_pct"] == 20.0
    assert summary["US_EQUITY"]["funds"][0]["symbol"] == "SPYM"
    assert summary["US_EQUITY"]["worst_drop_pct"]["3y"] == -20.0
    assert summary["US_EQUITY"]["measured_on"] == "SPYM"


# ------------------------------------------------------------------ other arms are untouched

def test_other_arms_prompts_do_not_see_the_etf_settings():
    """A stray broad_position_pct of 60 in main's prompt would read as a 60% cap it does not have."""
    s = {"universe": "all", "broad_position_pct": 60.0, "max_position_pct": 20.0}
    out = sandbox_job.settings_for_prompt(s)
    assert "broad_position_pct" not in out and "universe" not in out
    etf = sandbox_job.settings_for_prompt({**s, "universe": "etf"})
    assert etf["broad_position_pct"] == 60.0 and etf["universe"] == "etf"


@pytest.fixture()
def store(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setenv("SIGNALS_DATA_DIR", d)
    from app import sandbox_store as s
    importlib.reload(s)
    s._DATA_DIR = Path(d)
    s._cache = {s.MAIN_ARM: s._load(s.MAIN_ARM)}
    return s


def test_adding_etf_arms_does_not_move_the_original_arms_comparison_start(store):
    """The ETF arms start weeks after the others. One common start for every arm would throw the
    shared history away; the original arms keep theirs and the ETF arms get their own."""
    import time
    from app import main
    now = time.time()
    store.create_arm("rules", engine="rules")
    store.create_arm("etf", engine="llm", settings={"universe": "etf"})
    for d in ("2026-09-01", "2026-09-02", "2026-09-28", "2026-09-29"):
        store.append_nav({"date": d, "equity": 100.0, "ts": now}, "main")
        store.append_nav({"date": d, "equity": 100.0, "ts": now}, "rules")
    for d in ("2026-09-28", "2026-09-29"):
        store.append_nav({"date": d, "equity": 100.0, "ts": now}, "etf")
    r = asyncio.run(main.sandbox_arms_nav_endpoint(180))
    assert r["common_start"] == "2026-09-01"
    assert r["cohorts"]["etf"]["common_start"] == "2026-09-28"
    assert set(r["cohorts"]["etf"]["arms"]) == {"etf", "main"}
    assert {a["arm"]: a["universe"] for a in r["arms"]}["etf"] == "etf"


def test_every_fill_path_judges_an_etf_arm_by_the_etf_rules():
    """The daily tick and the parked-order sweep both take their fill rules from one helper, so an
    ETF arm's parked index-fund buy is not re-judged at 14:40 by main's single cap and group map."""
    from app import main
    etf = main._arm_fill_rules({"universe": "etf", "max_position_pct": 20.0})
    assert etf["group_of"]("QQQM") == "US_TECH"
    assert etf["cap_pct_of"]("US_EQUITY") == 60.0
    assert "AAPL" not in etf["allowed_buys"] and "VTI" in etf["allowed_buys"]
    other = main._arm_fill_rules({"universe": "all"})
    assert set(other) == {"group_of"} and other["group_of"]("QQQM") == "US_EQUITY"
    assert main._arm_fill_rules(None) == {"group_of": main._exposure_group}


# ------------------------------------------------------------------ plan targets bind buys

PLAN = {"cash_target_pct": 5.0, "targets": [
    {"exposure_group": "US_EQUITY", "target_pct": 40.0},
    {"exposure_group": "SHORT_TREASURY", "target_pct": 12.0}]}


def _fill_planned(blob, orders, plan=PLAN):
    return sandbox_job.validate_and_fill(
        blob, orders, lambda s: {"VGSH": 58.0, **PRICES}.get(s.upper()), group_of=GOF,
        cap_pct_of=etf_arm.cap_pct_of(blob["settings"]), allowed_buys=etf_arm.universe_set(),
        target_limit_of=etf_arm.target_limit_of(plan, blob["settings"]), now_ts=1_790_000_000.0)


def test_an_oversized_buy_is_cut_to_its_plan_target():
    """The 2026-09-28 dry run: $4,237 asked for a 12% target on an $11,250 book."""
    _, filled, _ = _fill_planned(_book(cash=11_250.0), [
        {"symbol": "VGSH", "side": "buy", "dollars": 4_237.5, "conviction": 82}])
    assert filled[0]["gross"] <= (12.0 + etf_arm.PLAN_TARGET_SLACK_PCT) / 100 * 11_250.0 + 1e-6
    assert filled[0]["gross"] > 0.12 * 11_250.0 - 58.0


def test_a_group_at_its_plan_target_takes_no_more():
    held = [{"symbol": "VGSH", "shares": 30.0, "avg_cost": 58.0, "exposure_group": "SHORT_TREASURY"}]
    _, filled, skipped = _fill_planned(_book(cash=10_000.0, positions=held), [
        {"symbol": "VGSH", "side": "buy", "dollars": 1_000.0, "conviction": 82}])
    assert filled == [] and skipped[0]["skip_reason"] == sandbox_job.PLAN_TARGET_SKIP


def test_an_unplanned_broad_group_gets_only_the_narrow_cap():
    """The broad cap exists so a PLAN of index funds is reachable, not for unplanned T-bills."""
    _, filled, _ = _fill_planned(_book(cash=10_000.0), [
        {"symbol": "BND", "side": "buy", "dollars": 5_000.0, "conviction": 82}])
    assert filled[0]["gross"] <= 2_000.0 + 1e-6


def test_no_plan_means_only_the_caps_apply():
    assert etf_arm.target_limit_of(None, {}) is None
    assert etf_arm.target_limit_of({"targets": []}, {}) is None


# ------------------------------------------------------------------ plan vocabulary

def test_plain_word_group_names_resolve_to_the_real_group():
    assert GOF("BONDS") == "US_BONDS" and GOF("international") == "INTL"
    assert GOF("SP500") == "US_EQUITY" and GOF("Emerging_Markets") == "EM"


def test_no_alias_shadows_a_fund_on_the_list():
    """An alias equal to a ticker would silently move that fund into another group's cap."""
    for alias in etf_arm._ALIASES:
        assert alias not in etf_arm.universe_set(), alias


def test_a_plan_label_outside_the_vocabulary_is_dropped_not_left_as_a_permanent_gap(monkeypatch):
    """A target no fund belongs to would read as 0% forever and keep the daily model filling it."""
    from app import main, memory
    from app.analyst import StrategyNote, TargetWeight
    notes = []
    monkeypatch.setattr(memory, "add_note", lambda kind, body, **k: notes.append((kind, body)))

    async def fake_review(context, *, settings, deep=True, extra_system=None):
        assert "exposure_groups" in context and extra_system == "X"
        return StrategyNote(stance="neutral", cash_target_pct=5.0, targets=[
            TargetWeight(exposure_group="US_EQUITY", target_pct=55.0),
            TargetWeight(exposure_group="BONDS", target_pct=20.0),          # alias -> US_BONDS
            TargetWeight(exposure_group="MOON_FUNDS", target_pct=20.0)]), {}
    monkeypatch.setattr(main, "strategy_review", fake_review)
    blob = {"settings": {"max_position_pct": 20.0, "universe": "etf"}, "cash": 1000.0,
            "funded_total": 1000.0, "benchmark": {"shares": 0.0}}
    vocab = {"US_EQUITY": ["VTI"], "US_BONDS": ["BND"]}
    ran = asyncio.run(main._maybe_weekly_review(
        blob, {"total_value": 1000.0, "positions": []}, blob["settings"], arm="etf",
        group_of=GOF, vocabulary=vocab, extra_system="X", note_kind="strategy@etf",
        cap_pct_of=etf_arm.cap_pct_of(blob["settings"])))
    assert ran
    assert [t["exposure_group"] for t in blob["last_strategy_note"]["targets"]] == ["US_EQUITY", "US_BONDS"]
    assert any(k == "strategy@etf_gap" and "MOON_FUNDS" in b for k, b in notes)
    assert all(k.startswith("strategy@etf") for k, _ in notes), "must never write main's note kinds"


# ------------------------------------------------------------------ review round 1 (money lens)

@pytest.mark.parametrize("age", range(20, 91, 3))
@pytest.mark.parametrize("risk", ["conservative", "balanced", "aggressive"])
@pytest.mark.parametrize("floor,broad", [(0.0, 60.0), (5.0, 60.0), (2.0, 20.0), (0.0, 35.0)])
def test_no_glidepath_target_is_ever_over_its_cap(age, risk, floor, broad):
    """A target over its cap is re-proposed and refused every tick while the money idles."""
    s = {"current_age": age, "retirement_age": 65, "risk_tolerance": risk, "cash_floor_pct": floor,
         "max_position_pct": 20.0, "broad_position_pct": broad}
    plan = etf_arm.glidepath_plan(s, dt.date(2026, 9, 28))
    caps = etf_arm.cap_pct_of(s)
    for t in plan["targets"]:
        assert t["target_pct"] <= caps(t["exposure_group"]) + 0.05, (t, plan)
    total = sum(t["target_pct"] for t in plan["targets"]) + plan["cash_target_pct"]
    assert total == pytest.approx(100.0, abs=0.05)
    assert plan["cash_target_pct"] >= floor


def test_the_bitcoin_and_gold_fund_lists_cover_every_copy_the_arm_can_buy():
    """A copy missing from BTC_ETFS / GOLD_ETFS skips the user's chosen fund and splits the holding."""
    uni = etf_arm.universe_set()
    for gid, family in (("bitcoin", sandbox_job.BTC_ETFS), ("gold", sandbox_job.GOLD_ETFS)):
        g = next(x for x in fund_cost.GROUPS if x.id == gid)
        assert {m for m in g.members if m in uni} <= family, gid


def test_a_fee_route_that_would_leave_less_than_one_share_is_undone():
    """$200 of ITOT ($130) fills; routed to VTI ($300) for the same 0.03% it would not."""
    out, notes = _route([{"symbol": "ITOT", "side": "buy", "dollars": 200.0}])
    assert out[0]["symbol"] == "ITOT" and notes == []


def test_a_route_onto_a_held_copy_stands_even_when_it_cannot_fill():
    held = [{"symbol": "VTI", "shares": 2.0}]
    out, _ = _route([{"symbol": "ITOT", "side": "buy", "dollars": 200.0}], positions=held)
    assert out[0]["symbol"] == "VTI"
