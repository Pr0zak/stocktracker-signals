"""The ETF pick's screen and score: a sound, low-cost fund at a better price (etf_pick.py)."""
from __future__ import annotations

import importlib
import tempfile
from pathlib import Path

import pytest

from app import daily_pick, etf_pick


def _closes(n=260, top=110.0, now=100.0, floor=80.0):
    """A long rise from `floor` to `top`, then a pullback to `now`, ending the series."""
    up = [floor + (top - floor) * i / (n - 30) for i in range(n - 30)]
    down = [top + (now - top) * i / 29 for i in range(30)]
    return up + down


def _fund(sym="VTI", fee=0.03, d3=-20.0, d5=-25.0, cat="us"):
    return {"symbol": sym, "name": "Whole US market", "category": cat, "expense_ratio_pct": fee,
            "returns": {"1y": 10.0, "3y": 40.0, "5y": 60.0}, "drops": {"1y": -9.0, "3y": d3, "5y": d5}}


def test_tech_row_reads_a_pullback_above_the_200_day():
    row = etf_pick.tech_row(_closes(), high_52w=110.0)
    assert row["above_sma200"] is True
    assert row["pct_off_52w_high"] == pytest.approx(-9.09, abs=0.01)
    assert etf_pick.tech_row([100.0] * 150) is None


def test_a_sound_fund_on_sale_is_eligible():
    row = etf_pick.tech_row(_closes(), high_52w=110.0)
    assert etf_pick.reject_reason(_fund(), row) is None


@pytest.mark.parametrize("fund,row_kw,why", [
    (_fund("FBTC", cat="crypto"), {}, etf_pick.REJECT_EXCLUDED),
    (_fund("SGOV"), {}, etf_pick.REJECT_EXCLUDED),
    (_fund(fee=None), {}, etf_pick.REJECT_FEE),
    (_fund(fee=0.95), {}, etf_pick.REJECT_FEE),
    (_fund(d3=None), {}, etf_pick.REJECT_HISTORY),
    (_fund(), {"now": 108.0}, etf_pick.REJECT_NOT_DOWN),
    (_fund(), {"now": 80.0, "floor": 20.0}, etf_pick.REJECT_BROKEN),
])
def test_each_screen_rule(fund, row_kw, why):
    row = etf_pick.tech_row(_closes(**row_kw), high_52w=110.0)
    assert etf_pick.reject_reason(fund, row) == why


def test_below_the_200_day_is_a_broken_trend_not_a_bargain():
    xs = [100.0] * 200 + [100 - i * 0.5 for i in range(60)]
    row = etf_pick.tech_row(xs, high_52w=100.0)
    assert etf_pick.reject_reason(_fund(), row) in (etf_pick.REJECT_BELOW_200, etf_pick.REJECT_BROKEN)


def test_an_unmeasured_fund_is_rejected_not_scored_as_zero():
    assert etf_pick.reject_reason(_fund(), None) == etf_pick.REJECT_UNMEASURED


def test_calmer_cheaper_funds_score_higher_at_the_same_discount():
    row = etf_pick.tech_row(_closes(), high_52w=110.0)
    calm = etf_pick.score(_fund(fee=0.03, d3=-15.0), row)["score"]
    wild = etf_pick.score(_fund(fee=0.45, d3=-45.0), row)["score"]
    assert calm > wild


def test_every_part_is_between_0_and_100():
    row = etf_pick.tech_row(_closes(), high_52w=110.0)
    for v in etf_pick.score(_fund(), row)["parts"].values():
        assert 0.0 <= v <= 100.0


def test_shortlist_ranks_and_counts_rejects():
    build = {"funds": [_fund("VTI"), _fund("XLK", fee=0.08, d3=-30.0, cat="sector"), _fund("IBIT", cat="crypto")]}
    on_sale = etf_pick.tech_row(_closes(), high_52w=110.0)
    rows = {"VTI": on_sale, "XLK": on_sale, "IBIT": on_sale}
    sl = etf_pick.shortlist(build, rows, pool=["VTI", "XLK", "IBIT"])
    assert [c["symbol"] for c in sl["ranked"]] == ["VTI", "XLK"]
    assert sl["rejects"] == {etf_pick.REJECT_EXCLUDED: 1} and sl["eligible"] == 2


def test_the_no_pick_reason_names_the_common_case():
    assert "5% or more off its high" in etf_pick.none_reason(
        {"scanned": 90, "rejects": {etf_pick.REJECT_NOT_DOWN: 70, etf_pick.REJECT_FEE: 5}})


def test_fund_factors_are_real_factor_rows_and_absent_when_unmeasured():
    f = etf_pick.fund_factors(_fund())
    assert set(f) == {"fee", "worst_drop"} and all(k in daily_pick.FACTOR_KEYS for k in f)
    assert "$3 per $10,000" in f["fee"]["display"]
    assert etf_pick.fund_factors({"expense_ratio_pct": None, "drops": {}}) == {}


def test_the_analyst_may_cite_the_fund_factors():
    from app.analyst import PickFactor
    assert {"fee", "worst_drop"} <= {p.value for p in PickFactor}


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
