"""DP-1/2/11/12/16 — the Daily Pick's pure rules.

What these pin, in the order a run meets them:
  * the screen's hard filters, and that an unmeasured metric is left OUT of the score, never read as 0;
  * the gate RAISES the conviction bar rather than forbidding a pick (it failed 17 of 22 sessions in
    Aug-Sep 2026, so a hard block would have produced "no pick" nearly every day and left nothing to
    grade the gate's value by);
  * reconcile() treats the model as a proposal: off-list symbols, unmeasured factors, a case with no
    reason against, and price levels that cannot be true are all removed before the card sees them;
  * the AI-vs-rule comparison counts only days where both marks exist, and a same-name day is a tie;
  * portfolio fit refuses to compute a weight over a partial book.
"""
from __future__ import annotations

import datetime as dt

from app import daily_pick as dp


def _row(sym="AAA", **kw):
    base = {
        "symbol": sym, "price": 50.0, "dollar_volume_20d": 1e8, "bars": 400,
        "above_sma50": 1, "above_sma200": 1, "ma_stacked": 1,
        "rel_strength_3mo": 8.0, "rel_strength_3mo_pctile": 90.0,
        "mom_60d": 12.0, "mom_60d_pctile": 85.0,
        "ema20_slope_pct_pctile": 80.0, "pct_vs_sma50": 5.0, "rsi14": 60.0,
        "rsi14_pctile": 70.0, "adr20_pct": 2.0, "adr20_pct_pctile": 50.0,
        "pct_off_52w_high": -3.0, "pct_off_52w_high_pctile": 85.0,
        "rel_volume": 1.2, "rel_volume_pctile": 60.0,
    }
    base.update(kw)
    return base


# ------------------------------------------------------------------ DP-1 screen

def test_hard_filters_reject_in_order():
    assert dp.reject_reason(_row("BTC-USD")) == dp.REJECT_NOT_EQUITY
    assert dp.reject_reason(_row("^GSPC")) == dp.REJECT_NOT_EQUITY
    assert dp.reject_reason(_row(price=3.0)) == dp.REJECT_PRICE
    assert dp.reject_reason(_row(price=None)) == dp.REJECT_PRICE
    assert dp.reject_reason(_row(dollar_volume_20d=5e6)) == dp.REJECT_ILLIQUID
    assert dp.reject_reason(_row(bars=100)) == dp.REJECT_HISTORY
    assert dp.reject_reason(_row()) is None


def test_trend_component_keeps_unmeasured_as_none():
    assert dp.trend_component(_row()) == 100.0
    assert dp.trend_component(_row(ma_stacked=0)) == 70.0
    assert dp.trend_component(_row(ma_stacked=0, above_sma50=0)) == 40.0
    assert dp.trend_component(_row(ma_stacked=0, above_sma200=0)) == 20.0
    assert dp.trend_component(_row(ma_stacked=0, above_sma50=0, above_sma200=0)) == 0.0
    assert dp.trend_component(_row(above_sma200=None)) is None


def test_missing_component_is_left_out_not_zeroed():
    full = dp.score_row(_row())
    no_slope = dp.score_row(_row(ema20_slope_pct_pctile=None))
    zero_slope = dp.score_row(_row(ema20_slope_pct_pctile=0.0))
    assert no_slope["partial"] is True and no_slope["missing"] == ["slope"]
    # Renormalised over what was measured: an absent slope must not drag the score to where a
    # measured 0th-percentile slope puts it.
    assert no_slope["score"] > zero_slope["score"]
    assert full["partial"] is False


def test_unscorable_without_trend_or_without_both_momentum_inputs():
    assert dp.score_row(_row(above_sma200=None)) is None
    assert dp.score_row(_row(rel_strength_3mo_pctile=None, mom_60d_pctile=None)) is None
    assert dp.score_row(_row(rel_strength_3mo_pctile=None)) is not None


def test_extension_and_volatility_are_penalised():
    calm = dp.score_row(_row())
    stretched = dp.score_row(_row(pct_vs_sma50=25.0, rsi14=82.0, adr20_pct_pctile=97.0))
    assert stretched["penalty"] == 35
    assert stretched["score"] == round(calm["score"] - 35, 2)
    assert len(stretched["penalty_reasons"]) == 3


def test_shortlist_ranks_counts_rejects_and_respects_the_universe():
    rows = [_row("AAA"), _row("BBB", rel_strength_3mo_pctile=10.0), _row("CCC", price=1.0),
            _row("DDD", above_sma200=None)]
    sl = dp.shortlist(rows)
    assert [c["symbol"] for c in sl["ranked"]] == ["AAA", "BBB"]
    assert sl["rejects"] == {dp.REJECT_PRICE: 1, dp.REJECT_UNSCORABLE: 1}
    assert sl["scanned"] == 4 and sl["eligible"] == 2
    only = dp.shortlist(rows, only={"BBB"})
    assert [c["symbol"] for c in only["ranked"]] == ["BBB"]
    assert only["scanned"] == 1 and only["rejects"] == {}


def test_shortlist_ties_break_on_symbol():
    sl = dp.shortlist([_row("ZZZ"), _row("AAA")])
    assert [c["symbol"] for c in sl["ranked"]] == ["AAA", "ZZZ"]


def test_sessions_until_counts_trading_days():
    weekday = lambda d: d.weekday() < 5  # noqa: E731
    assert dp.sessions_until("2026-09-22", "2026-09-22", weekday) == 0
    assert dp.sessions_until("2026-09-25", "2026-09-28", weekday) == 1   # Fri -> Mon
    assert dp.sessions_until("2026-09-22", "2026-09-21", weekday) is None
    assert dp.sessions_until("2026-09-22", None, weekday) is None


# ------------------------------------------------------------------ factors

def test_unmeasured_factors_are_absent_not_zero():
    f = dp.factors_for(_row(rsi14=None, rel_strength_3mo=None), {})
    assert "rsi" not in f and "rel_strength" not in f
    assert f["momentum"]["pctile"] == 85.0
    assert "long_cycle" not in f and "track_record" not in f


def test_a_failed_earnings_lookup_is_absent_not_quiet():
    assert "earnings" not in dp.factors_for(_row(), {}, earnings={"ok": False, "date": None})
    quiet = dp.factors_for(_row(), {}, earnings={"ok": True, "date": None, "window_days": 7})
    assert quiet["earnings"]["display"] == "no earnings report in the next 7 days"


def test_regime_factor_reads_all_three_gate_states():
    shut = {"available": True, "passed": False, "failing": ["Breadth > 55%"], "market_score": 69.5}
    assert "narrow" in dp.factors_for(_row(), {}, gate=shut)["regime"]["display"]
    assert "could not be measured" in dp.factors_for(_row(), {}, gate={"available": True, "passed": None})["regime"]["display"]
    assert "regime" not in dp.factors_for(_row(), {}, gate={"available": False})


# ------------------------------------------------------------------ DP-2 reconcile

def _cands(sym="AAA", price=100.0, gate=None):
    return {sym: {"price": price, "factors": dp.factors_for(_row(sym), {}, gate=gate)}}


def _choice(**kw):
    base = {
        "symbol": "AAA", "conviction": 72, "thesis": "Strong and steady.",
        "invalidation": "a close under 95",
        "reasons": [
            {"factor": "rel_strength", "stance": "supports", "text": "Beat the S&P by 8 points."},
            {"factor": "trend", "stance": "supports", "text": "Averages stacked."},
            {"factor": "rsi", "stance": "against", "text": "RSI is getting warm."},
        ],
        "entry_low": 98.0, "entry_high": 101.0, "stop": 94.0, "target": 112.0,
        "runners_up": [],
    }
    base.update(kw)
    return base


def test_a_clean_pick_passes_with_server_numbers():
    out = dp.reconcile(_choice(), candidates=_cands(), gate=None)
    assert out["status"] == dp.STATUS_PICK
    assert out["levels"] == {"entry_low": 98.0, "entry_high": 101.0, "stop": 94.0, "target": 112.0}
    assert out["risk_reward"]["rr_ratio"] == round((112 - 99.5) / (99.5 - 94), 2)


def test_null_symbol_is_no_pick_with_the_models_reason():
    out = dp.reconcile(_choice(symbol=None, none_reason="Nothing clean today."), candidates=_cands(), gate=None)
    assert out["status"] == dp.STATUS_NONE and out["none_reason"] == "Nothing clean today."


def test_off_list_symbol_is_discarded():
    out = dp.reconcile(_choice(symbol="ZZZ"), candidates=_cands(), gate=None)
    assert out["status"] == dp.STATUS_NONE and out["rejected_symbol"] == "ZZZ"


def test_reasons_about_unmeasured_factors_are_dropped():
    reasons = _choice()["reasons"] + [{"factor": "insider", "stance": "supports", "text": "Insiders buying."}]
    out = dp.reconcile(_choice(reasons=reasons), candidates=_cands(), gate=None)
    assert all(r["factor"] != "insider" for r in out["reasons"])
    assert out["reasons_dropped"] == 1


def test_no_reason_against_means_no_pick():
    only_for = [r for r in _choice()["reasons"] if r["stance"] == "supports"]
    out = dp.reconcile(_choice(reasons=only_for), candidates=_cands(), gate=None)
    assert out["status"] == dp.STATUS_NONE and "no reason against" in out["none_reason"]


def test_a_case_citing_only_unmeasured_factors_is_no_pick():
    fake = [{"factor": "insider", "stance": "supports", "text": "x"},
            {"factor": "rsi", "stance": "against", "text": "y"}]
    out = dp.reconcile(_choice(reasons=fake), candidates=_cands(), gate=None)
    assert out["status"] == dp.STATUS_NONE


def test_a_shut_gate_raises_the_floor_and_adds_itself_as_a_reason_against():
    shut = {"available": True, "passed": False, "failing": ["Breadth > 55%"]}
    low = dp.reconcile(_choice(conviction=65), candidates=_cands(gate=shut), gate=shut)
    assert low["status"] == dp.STATUS_NONE and low["conviction_floor"] == 70
    assert low["none_reason"].startswith("The best candidate, AAA, scored 65 out of 100")
    assert "fewer than 55% of stocks are in uptrends" in low["none_reason"]
    ok = dp.reconcile(_choice(conviction=74), candidates=_cands(gate=shut), gate=shut)
    assert ok["status"] == dp.STATUS_PICK
    assert any(r["factor"] == "regime" and r["stance"] == "against" for r in ok["reasons"])


def test_an_open_or_unknown_gate_keeps_the_normal_floor():
    for g in (None, {"available": True, "passed": True}, {"available": True, "passed": None},
              {"available": False, "passed": None}):
        out = dp.reconcile(_choice(conviction=62), candidates=_cands(gate=g), gate=g)
        assert out["status"] == dp.STATUS_PICK, g


def test_conviction_is_clamped():
    assert dp.reconcile(_choice(conviction=250), candidates=_cands(), gate=None)["conviction"] == 100


def test_runners_up_must_be_on_the_list_and_not_the_pick():
    c = {**_cands("AAA"), **_cands("BBB")}
    ru = [{"symbol": "BBB", "why_not": "weaker"}, {"symbol": "AAA", "why_not": "self"},
          {"symbol": "ZZZ", "why_not": "off list"}, {"symbol": "bbb", "why_not": "dup"}]
    out = dp.reconcile(_choice(runners_up=ru), candidates=c, gate=None)
    assert [r["symbol"] for r in out["runners_up"]] == ["BBB"]


# ------------------------------------------------------------------ price levels

def test_a_zone_far_from_price_is_dropped():
    lv, notes = dp.sanitize_levels({"entry_low": 170, "entry_high": 180, "stop": 160, "target": 200}, 423.0)
    assert lv["entry_low"] is None and lv["entry_high"] is None
    assert notes


def test_a_zone_containing_the_price_is_kept():
    lv, notes = dp.sanitize_levels({"entry_low": 98, "entry_high": 102, "stop": 94, "target": 110}, 100.0)
    assert lv == {"entry_low": 98.0, "entry_high": 102.0, "stop": 94.0, "target": 110.0} and not notes


def test_inverted_zone_is_swapped():
    lv, _ = dp.sanitize_levels({"entry_low": 102, "entry_high": 98}, 100.0)
    assert (lv["entry_low"], lv["entry_high"]) == (98.0, 102.0)


def test_impossible_stop_and_target_are_dropped_not_zeroed():
    lv, notes = dp.sanitize_levels({"entry_low": 98, "entry_high": 102, "stop": 99, "target": 101}, 100.0)
    assert lv["stop"] is None and lv["target"] is None and len(notes) == 2


def test_zero_levels_are_absent():
    lv, _ = dp.sanitize_levels({"entry_low": 0, "entry_high": 0, "stop": 0, "target": 0}, 100.0)
    assert lv == {"entry_low": None, "entry_high": None, "stop": None, "target": None}


def test_risk_reward_never_divides_by_an_absent_stop():
    rr = dp.risk_reward({"entry_low": 98, "entry_high": 102, "stop": None, "target": 110})
    assert rr["rr_ratio"] is None and rr["reward_per_share"] == 10.0


# ------------------------------------------------------------------ DP-11 comparison

def test_comparison_counts_only_days_with_both_marks_and_ties_same_name():
    days = [
        {"ai_status": "pick", "ai_excess": 2.0, "rule_excess": 1.0, "same": False},
        {"ai_status": "pick", "ai_excess": -1.0, "rule_excess": 0.5, "same": False},
        {"ai_status": "pick", "ai_excess": 1.0, "rule_excess": 1.0, "same": True},
        {"ai_status": "pick", "ai_excess": None, "rule_excess": 1.0, "same": False},
        {"ai_status": "none", "ai_excess": None, "rule_excess": -2.0, "same": False},
    ]
    c = dp.paired_comparison(days, 20)
    assert c["n_days"] == 3
    assert (c["ai_better"], c["rule_better"], c["ties"]) == (1, 1, 1)
    assert c["median_diff_pp"] == round((1.0 + -1.5) / 2, 2)
    assert c["declined_days"] == 1 and c["declined_rule_median_excess_pp"] == -2.0


def test_empty_comparison_reports_none_not_zero():
    c = dp.paired_comparison([], 5)
    assert c["n_days"] == 0 and c["median_diff_pp"] is None


# ------------------------------------------------------------------ DP-16 repeats

def test_repeat_counts_only_actual_picks():
    hist = [{"status": "pick", "symbol": "XOM"}, {"status": "none", "symbol": None},
            {"status": "pick", "symbol": "CVX"}, {"status": "pick", "symbol": "XOM"}]
    grp = lambda s: "ENERGY" if s in ("XOM", "CVX") else s  # noqa: E731
    r = dp.repeat_counts(hist, "XOM", group_of=grp)
    assert r["symbol"] == 2 and r["group"] == 3 and r["window_runs"] == 4


# ------------------------------------------------------------------ DP-12 fit

_grp = lambda s: {"IBIT": "BTC", "FBTC": "BTC"}.get(s, s)  # noqa: E731


def test_fit_refuses_weights_over_a_partial_book():
    fit = dp.portfolio_fit("XOM", None, [{"symbol": "CVX", "value": 1000}, {"symbol": "AAPL", "value": None}],
                           group_of=_grp, sectors={"XOM": "Energy", "CVX": "Energy"})
    assert fit["available"] is False and fit["weight_pct"] is None and fit["unpriced"] == ["AAPL"]
    assert "could not be priced" in dp.fit_sentence(fit)


def test_fit_with_no_holdings_is_unknown():
    fit = dp.portfolio_fit("XOM", None, [], group_of=_grp, sectors={})
    assert fit["available"] is False and "unknown" in dp.fit_sentence(fit)


def test_fit_sector_weight_before_and_after_a_median_buy():
    held = [{"symbol": "CVX", "value": 1000}, {"symbol": "AAPL", "value": 3000}, {"symbol": "SPY", "value": 6000}]
    fit = dp.portfolio_fit("XOM", None, held, group_of=_grp,
                           sectors={"XOM": "Energy", "CVX": "Energy", "AAPL": "Technology", "SPY": None})
    assert fit["available"] is True and fit["already_held"] is False
    assert fit["sector_weight_pct"] == 10.0
    assert fit["reference_buy_usd"] == 3000.0
    assert fit["sector_weight_after_pct"] == round(4000 / 13000 * 100, 1)
    assert fit["unclassified"] == ["SPY"]
    s = dp.fit_sentence(fit)
    assert "Energy is 10.0%" in s and "no sector on file" in s


def test_fit_sees_the_same_exposure_through_another_ticker():
    fit = dp.portfolio_fit("IBIT", None, [{"symbol": "FBTC", "value": 500}, {"symbol": "AAPL", "value": 500}],
                           group_of=_grp, sectors={})
    assert fit["group_weight_pct"] == 50.0 and fit["group_members_held"] == ["FBTC"]
    assert "FBTC" in dp.fit_sentence(fit)


def test_fit_never_returns_a_share_count():
    fit = dp.portfolio_fit("XOM", 100.0, [{"symbol": "CVX", "value": 1000}], group_of=_grp, sectors={})
    assert not any("share" in k for k in fit)


def test_gate_failures_read_as_plain_words():
    assert dp.gate_failing_words({"failing": ["VIX < 20"]}) == "the fear index (VIX) is above 20"
    two = dp.gate_failing_words({"failing": ["Breadth > 55%", "SPY 20-day momentum > 0"]})
    assert two.startswith("2 market checks failed")
    assert dp.gate_failing_words({"failing": ["Something new"]}) == "Something new"
    assert dp.gate_failing_words(None) == "a market check failed"
