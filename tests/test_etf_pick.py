"""The ETF pick's screen and score: the sound, low-cost fund with the strongest momentum (etf_pick.py)."""
from __future__ import annotations

import importlib
import tempfile
from pathlib import Path

import pytest

from app import daily_pick, etf_pick


def _closes(n=260, start=80.0, end=100.0, dip=0.0):
    """A steady rise from `start` to `end`; `dip` pulls the final 20 bars back by that many percent, so
    the fund is still up over 6 months but below its high (a pause on a leader, not a break)."""
    xs = [start + (end - start) * i / (n - 1) for i in range(n)]
    if dip:
        xs = xs[:-20] + [x * (1.0 - dip / 100.0) for x in xs[-20:]]
    return xs


def _falling(n=260, start=100.0, end=80.0):
    return [start + (end - start) * i / (n - 1) for i in range(n)]


def _bench(n=260, start=90.0, end=100.0):
    return [start + (end - start) * i / (n - 1) for i in range(n)]


def _fund(sym="VTI", fee=0.03, d3=-20.0, d5=-25.0, cat="us", r1y=12.0):
    return {"symbol": sym, "name": "Whole US market", "category": cat, "expense_ratio_pct": fee,
            "returns": {"1y": r1y, "3y": 40.0, "5y": 60.0}, "drops": {"1y": -9.0, "3y": d3, "5y": d5}}


def test_tech_row_reads_six_month_momentum_and_the_benchmark():
    row = etf_pick.tech_row(_closes(), bench_closes=_bench())
    assert row["above_sma200"] is True
    assert row["mom_126d"] == pytest.approx(10.8, abs=0.2)
    assert row["bench_mom_126d"] == pytest.approx(5.2, abs=0.2)
    assert row["rel_strength_6mo"] == pytest.approx(row["mom_126d"] - row["bench_mom_126d"], abs=0.02)
    assert etf_pick.tech_row([100.0] * 150) is None
    assert "rel_strength_6mo" not in etf_pick.tech_row(_closes())


def test_a_sound_rising_fund_is_eligible_even_a_little_off_its_high():
    assert etf_pick.reject_reason(_fund(), etf_pick.tech_row(_closes())) is None
    assert etf_pick.reject_reason(_fund(), etf_pick.tech_row(_closes(dip=8.0))) is None


@pytest.mark.parametrize("fund,closes,why", [
    (_fund("FBTC", cat="crypto"), _closes(), etf_pick.REJECT_EXCLUDED),
    (_fund("SGOV"), _closes(), etf_pick.REJECT_EXCLUDED),
    (_fund(fee=None), _closes(), etf_pick.REJECT_FEE),
    (_fund(fee=0.95), _closes(), etf_pick.REJECT_FEE),
    (_fund(d3=None), _closes(), etf_pick.REJECT_HISTORY),
    (_fund(), _falling(), etf_pick.REJECT_BELOW_200),
    # Above its 200-day only because of an old rise, but down over the last 6 months.
    (_fund(), _closes(n=130, start=60.0, end=120.0) + _closes(n=130, start=120.0, end=117.0), etf_pick.REJECT_NO_MOMENTUM),
])
def test_each_screen_rule(fund, closes, why):
    assert etf_pick.reject_reason(fund, etf_pick.tech_row(closes)) == why


def test_an_unmeasured_fund_is_rejected_not_scored_as_zero():
    assert etf_pick.reject_reason(_fund(), None) == etf_pick.REJECT_UNMEASURED


def test_stronger_momentum_scores_higher_and_a_pullback_is_not_punished():
    strong = etf_pick.tech_row(_closes(end=110.0), bench_closes=_bench())
    weak = etf_pick.tech_row(_closes(end=96.0), bench_closes=_bench())
    assert etf_pick.score(_fund(), strong)["score"] > etf_pick.score(_fund(), weak)["score"]
    # 2% off its high with the same 6-month gain: no "health" or "discount" term moves the score.
    paused = etf_pick.tech_row(_closes(end=110.0, dip=2.0), bench_closes=_bench())
    s_strong, s_paused = etf_pick.score(_fund(), strong), etf_pick.score(_fund(), paused)
    assert set(s_strong["parts"]) == {"gain_6m", "gain_12m", "rel_6m", "calm", "cost"}
    assert s_paused["parts"]["calm"] == s_strong["parts"]["calm"]
    assert s_paused["parts"]["cost"] == s_strong["parts"]["cost"]


def test_calmer_cheaper_funds_score_higher_at_the_same_momentum():
    row = etf_pick.tech_row(_closes(), bench_closes=_bench())
    calm = etf_pick.score(_fund(fee=0.03, d3=-15.0), row)["score"]
    wild = etf_pick.score(_fund(fee=0.45, d3=-45.0), row)["score"]
    assert calm > wild


def test_a_fund_too_young_for_a_12_month_gain_scores_zero_there_not_rejected():
    row = etf_pick.tech_row(_closes(), bench_closes=_bench())
    young = _fund(r1y=None)
    assert etf_pick.reject_reason(young, row) is None
    assert etf_pick.score(young, row)["parts"]["gain_12m"] == 0.0
    assert "gain_12m" not in etf_pick.fund_factors(young, row)


def test_no_benchmark_means_level_not_behind():
    row = etf_pick.tech_row(_closes())
    assert etf_pick.score(_fund(), row)["parts"]["rel_6m"] == 50.0
    assert "vs" not in etf_pick.fund_factors(_fund(), row)["gain_6m"]["display"]


def test_every_part_is_between_0_and_100():
    row = etf_pick.tech_row(_closes(end=140.0), bench_closes=_bench(end=80.0))
    for v in etf_pick.score(_fund(r1y=90.0), row)["parts"].values():
        assert 0.0 <= v <= 100.0


def test_shortlist_ranks_by_momentum_and_counts_rejects():
    build = {"funds": [_fund("VTI"), _fund("XLK", fee=0.08, d3=-30.0, cat="sector"), _fund("IBIT", cat="crypto"),
                       _fund("BND", cat="bonds")]}
    rows = {"VTI": etf_pick.tech_row(_closes(end=105.0)), "XLK": etf_pick.tech_row(_closes(end=125.0)),
            "IBIT": etf_pick.tech_row(_closes()), "BND": etf_pick.tech_row(_falling())}
    sl = etf_pick.shortlist(build, rows, pool=["VTI", "XLK", "IBIT", "BND"])
    assert [c["symbol"] for c in sl["ranked"]] == ["XLK", "VTI"]
    assert sl["rejects"] == {etf_pick.REJECT_EXCLUDED: 1, etf_pick.REJECT_BELOW_200: 1} and sl["eligible"] == 2


def test_the_no_pick_reason_names_the_common_case():
    assert "up over the last 6 months" in etf_pick.none_reason(
        {"scanned": 90, "rejects": {etf_pick.REJECT_NO_MOMENTUM: 70, etf_pick.REJECT_FEE: 5}})
    assert "200-day" in etf_pick.none_reason(
        {"scanned": 90, "rejects": {etf_pick.REJECT_BELOW_200: 70, etf_pick.REJECT_NO_MOMENTUM: 10}})
    assert etf_pick.none_reason({"scanned": 0}) == "The fund list could not be read this morning"


def test_fund_factors_are_real_factor_rows_and_absent_when_unmeasured():
    row = etf_pick.tech_row(_closes(), bench_closes=_bench())
    f = etf_pick.fund_factors(_fund(), row)
    assert set(f) == {"gain_6m", "gain_12m", "fee", "worst_drop"} and all(k in daily_pick.FACTOR_KEYS for k in f)
    assert f["gain_6m"]["display"] == (f"{row['mom_126d']:+.1f}% over the last 6 months, "
                                       f"vs {row['bench_mom_126d']:+.1f}% for the S&P 500")
    assert f["gain_12m"]["display"] == "+12.0% over the last 12 months"
    assert "$3 per $10,000" in f["fee"]["display"]
    assert etf_pick.fund_factors({"expense_ratio_pct": None, "drops": {}}) == {}


def test_the_analyst_may_cite_the_fund_factors():
    from app.analyst import PickFactor
    assert {"fee", "worst_drop", "gain_6m", "gain_12m"} <= {p.value for p in PickFactor}


def test_stock_and_etf_runs_never_share_a_log(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setenv("SIGNALS_DATA_DIR", d)
    from app import daily_pick_store as st
    importlib.reload(st)
    st.append_run({"date": "2026-09-28", "status": "none"}, kind="stock")
    st.append_run({"date": "2026-09-28", "status": "pick", "pick": {"symbol": "VTI"}}, kind="etf")
    assert st.run_for("2026-09-28")["status"] == "none"
    assert st.run_for("2026-09-28", kind="etf")["pick"]["symbol"] == "VTI"
    with pytest.raises(ValueError):
        st.runs(kind="bonds")
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(Path(d)))


def test_funds_that_move_together_take_one_slot():
    """XLF and FNCL (0.994) took two of eight slots on the first live run."""
    build = {"funds": [_fund("XLF", cat="sector"), _fund("FNCL", fee=0.084, cat="sector"), _fund("VTI")]}
    row = etf_pick.tech_row(_closes())
    sl = etf_pick.shortlist(build, {s: row for s in ("XLF", "FNCL", "VTI")}, pool=["XLF", "FNCL", "VTI"])
    groups = [etf_pick.etf_arm.group_of(c["symbol"]) for c in sl["ranked"]]
    assert len(groups) == len(set(groups)) == 2
