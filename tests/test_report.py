"""RPT-1 — the weekly and monthly report's arithmetic (app/report.py)."""
import datetime as dt

import pytest

from app import report

D = dt.date


# --- periods ---------------------------------------------------------------------------------------

def test_week_bounds_are_the_close_before_monday_to_the_last_session():
    assert report.bounds("week", D(2026, 9, 23)) == (D(2026, 9, 18), D(2026, 9, 25))


def test_labor_day_week_starts_from_the_friday_before_and_skips_the_holiday():
    start, end = report.bounds("week", D(2026, 9, 9))
    assert (start, end) == (D(2026, 9, 4), D(2026, 9, 11))
    assert report.sessions_in(start, end) == 4
    assert report.period_label("week", start, end) == "Sep 8 – 11"


def test_good_friday_week_ends_on_thursday():
    assert report.bounds("week", D(2026, 4, 1)) == (D(2026, 3, 27), D(2026, 4, 2))


def test_month_bounds_use_the_months_last_session():
    assert report.bounds("month", D(2026, 9, 10)) == (D(2026, 8, 31), D(2026, 9, 30))
    # May 2026 ends on a Sunday; its last session is Friday the 29th.
    assert report.bounds("month", D(2026, 5, 3))[1] == D(2026, 5, 29)


def test_latest_completed_is_the_period_just_closed_or_the_one_before():
    assert report.latest_completed("week", D(2026, 9, 25)) == (D(2026, 9, 18), D(2026, 9, 25))
    assert report.latest_completed("week", D(2026, 9, 23)) == (D(2026, 9, 11), D(2026, 9, 18))
    assert report.latest_completed("month", D(2026, 9, 25)) == (D(2026, 7, 31), D(2026, 8, 31))
    assert report.latest_completed("month", D(2026, 9, 30)) == (D(2026, 8, 31), D(2026, 9, 30))


def test_labels_read_the_way_people_say_them():
    assert report.period_label("week", D(2026, 9, 18), D(2026, 9, 25)) == "Sep 21 – 25"
    assert report.period_label("week", D(2026, 9, 25), D(2026, 10, 2)) == "Sep 28 – Oct 2"
    assert report.period_label("month", D(2026, 8, 31), D(2026, 9, 30)) == "September 2026"
    assert report.report_id("week", D(2026, 9, 25)) == "week-2026-09-25"


def test_unknown_kind_is_refused():
    with pytest.raises(ValueError):
        report.bounds("year", D(2026, 9, 25))


# --- series ----------------------------------------------------------------------------------------

DATES = ["20260917", "20260918", "20260921", "20260922", "20260923", "20260924", "20260925"]


def test_change_is_measured_between_the_two_bounding_closes():
    closes = [99.0, 100.0, 101.0, 102.0, 100.5, 101.5, 103.0]
    c = report.change(DATES, closes, D(2026, 9, 18), D(2026, 9, 25))
    assert c == {"close": 103.0, "start": 100.0, "pct": 3.0}


def test_a_missing_end_bar_is_a_missing_number_not_the_day_before():
    closes = [99.0, 100.0, 101.0, 102.0, 100.5, 101.5, None]
    assert report.change(DATES, closes, D(2026, 9, 18), D(2026, 9, 25)) is None
    assert report.change(DATES[:-1], closes[:-1], D(2026, 9, 18), D(2026, 9, 25)) is None


def test_points_unit_reports_the_level_change():
    closes = [4.9, 5.0, 5.05, 5.1, 5.12, 5.15, 5.18]
    c = report.change(DATES, closes, D(2026, 9, 18), D(2026, 9, 25), unit="pts")
    assert c["change"] == pytest.approx(0.18) and "pct" not in c


def test_daily_moves_start_from_the_close_before_the_period():
    closes = [99.0, 100.0, 101.0, None, 100.0, 101.0, 102.01]
    moves = report.daily_moves(DATES, closes, D(2026, 9, 18), D(2026, 9, 25))
    # The 22nd has no bar, so neither the 22nd nor the 23rd (measured against it) is drawn.
    assert [m["date"] for m in moves] == ["2026-09-21", "2026-09-24", "2026-09-25"]
    assert moves[0]["pct"] == 1.0 and moves[1]["pct"] == 1.0


# --- movers, breadth, names ------------------------------------------------------------------------

def test_movers_keep_one_fund_per_holding_and_never_list_a_name_twice():
    rows = [{"symbol": "SOXX", "name": "Chip makers", "pct": 7.4},
            {"symbol": "SMH", "name": "Chip makers", "pct": 5.9},
            {"symbol": "UNG", "name": "Natural gas", "pct": 6.9},
            {"symbol": "XOP", "name": "Oil drillers", "pct": -4.8}]
    m = report.movers(rows, n=5, dedupe_key="name")
    assert [r["symbol"] for r in m["best"]] == ["SOXX", "UNG", "XOP"]
    # SMH was dropped from "best" as SOXX's twin; it must not reappear as one of the worst.
    assert m["worst"] == []


def test_movers_ignore_unmeasured_rows():
    rows = [{"symbol": "A", "pct": None}, {"symbol": "B", "pct": 1.0}, {"symbol": "C", "pct": -2.0}]
    m = report.movers(rows, n=1)
    assert m["best"][0]["symbol"] == "B" and m["worst"][0]["symbol"] == "C"


def test_breadth_counts_over_what_was_measured():
    b = report.breadth([{"pct": 1.0}, {"pct": -1.0}, {"pct": 0.0}, {"pct": None}])
    assert (b["up"], b["down"], b["flat"], b["measured"]) == (1, 1, 1, 3)
    assert report.breadth([])["up_share"] is None


@pytest.mark.parametrize("raw,clean", [
    ("Moderna, Inc. - Common Stock", "Moderna"),
    ("Everpure, Inc. Class A common stock", "Everpure"),
    ("Bending Spoons S.p.A. - Ordinary Shares", "Bending Spoons"),
    ("Charter Communications, Inc. - Class A Common Stock", "Charter Communications"),
    ("Alibaba Group Holding Limited American Depositary Shares", "Alibaba"),
    ("10x Genomics, Inc. - Class A Common Stock", "10x Genomics"),
])
def test_clean_name(raw, clean):
    assert report.clean_name(raw) == clean


# --- headline --------------------------------------------------------------------------------------

def test_headline_names_a_narrow_rally():
    assert report.headline("week", 1.21, 0.38, tech_led=True) == "Big tech lifted the S&P. Most stocks fell."
    assert report.headline("week", 1.21, 0.38) == "The biggest companies lifted the S&P. Most stocks fell."


def test_headline_other_shapes():
    assert report.headline("month", 2.0, 0.7) == "A good month: the S&P rose and most stocks did too."
    assert report.headline("week", -1.0, 0.6) == "The S&P slipped, but most stocks rose."
    assert report.headline("week", -1.0, 0.3) == "A down week for the S&P and most stocks."
    assert report.headline("week", 0.1, 0.3) == "A flat week for the S&P. Most stocks fell."
    assert report.headline("week", None, 0.3) == "The S&P could not be measured this week."


# --- sandbox ---------------------------------------------------------------------------------------

def _nav(date, eq, funded, bench, cash=100.0):
    return {"date": date, "equity": eq, "funded_total": funded, "benchmark_value": bench, "cash": cash}


def test_sandbox_period_takes_deposits_out_of_the_result():
    nav = [_nav("2026-09-18", 1000.0, 1000.0, 1000.0),
           _nav("2026-09-21", 1010.0, 1000.0, 1020.0),
           # a $100 deposit lands with the market flat for the book
           _nav("2026-09-22", 1110.0, 1100.0, 1120.0),
           _nav("2026-09-25", 1099.0, 1100.0, 1131.2)]
    p = report.sandbox_period(nav, D(2026, 9, 18), D(2026, 9, 25))
    assert p["deposits"] == 100.0
    assert p["change_usd"] == -1.0
    # Time-weighted: the deposit day's return is measured with the deposit taken out, so the $100
    # neither counts as a gain nor dilutes the base. Here the book ends almost exactly flat.
    twr = (1010 / 1000) * ((1110 - 100) / 1010) * (1099 / 1110) - 1
    assert p["change_pct"] == pytest.approx(round(twr * 100, 2))
    bench = (1020 / 1000) * ((1120 - 100) / 1020) * (1131.2 / 1120) - 1
    assert p["bench_pct"] == pytest.approx(round(bench * 100, 2))
    assert p["vs_pts"] == pytest.approx(p["change_pct"] - p["bench_pct"])
    assert p["measured_through_end"] is True
    assert p["cash_pct"] == pytest.approx(9.1, abs=0.1)


def test_sandbox_period_says_when_the_last_check_was_early():
    nav = [_nav("2026-09-18", 1000.0, 1000.0, 1000.0), _nav("2026-09-24", 990.0, 1000.0, 1010.0)]
    p = report.sandbox_period(nav, D(2026, 9, 18), D(2026, 9, 25))
    assert p["end_date"] == "2026-09-24" and p["measured_through_end"] is False


def test_sandbox_period_without_coverage_is_none():
    assert report.sandbox_period([_nav("2026-09-22", 1000.0, 1000.0, 1000.0)], D(2026, 9, 18), D(2026, 9, 25)) is None
    assert report.sandbox_period([], D(2026, 9, 18), D(2026, 9, 25)) is None


def test_period_trades_age_sales_first_in_first_out_and_flag_quick_losses():
    trades = [
        {"date": "2026-08-10", "side": "buy", "symbol": "SCHD", "status": "filled", "shares": 5, "price": 35.0},
        {"date": "2026-09-13", "side": "buy", "symbol": "SCHD", "status": "filled", "shares": 5, "price": 34.6},
        {"date": "2026-09-24", "side": "sell", "symbol": "SCHD", "status": "filled", "shares": 10,
         "price": 33.1, "gross": 331.0, "realized_pl": -14.87},
        {"date": "2026-09-24", "side": "buy", "symbol": "VXUS", "status": "filled", "shares": 4,
         "price": 85.58, "gross": 342.33},
        {"date": "2026-09-23", "side": "buy", "symbol": "BRK-B", "status": "skipped",
         "skip_reason": "cash floor (0% of equity) left under one share at $508.71"},
        {"date": "2026-09-25", "side": "buy", "symbol": "BRK-B", "status": "skipped",
         "skip_reason": "cash floor (0% of equity) left under one share at $504.96"},
        {"date": "2026-09-25", "side": "interest", "symbol": "CASH", "status": "filled", "gross": 0.05},
        {"date": "2026-09-11", "side": "buy", "symbol": "XOM", "status": "filled", "shares": 1, "price": 150.0},
    ]
    t = report.period_trades(trades, D(2026, 9, 18), D(2026, 9, 25))
    sale = next(f for f in t["fills"] if f["side"] == "sell")
    # The oldest lot sold went back to Aug 10 — 45 days, so this sale is not a "quick" one.
    assert sale["held_days"] == 45 and sale["flag"] is None
    assert (t["buys"], t["sells"], t["blocked_count"]) == (1, 1, 2)
    assert t["blocked"][0] == {"symbol": "BRK-B", "side": "buy", "count": 2,
                               "dates": ["2026-09-23", "2026-09-25"], "reason": "Not enough cash for one share"}
    assert t["interest"] == 0.05
    assert all(f["symbol"] != "XOM" for f in t["fills"])  # before the period


def test_a_quick_sale_at_a_loss_is_flagged():
    trades = [
        {"date": "2026-09-13", "side": "buy", "symbol": "SCHD", "status": "filled", "shares": 10, "price": 34.6},
        {"date": "2026-09-24", "side": "sell", "symbol": "SCHD", "status": "filled", "shares": 10,
         "price": 33.1, "realized_pl": -14.87},
    ]
    sale = report.period_trades(trades, D(2026, 9, 18), D(2026, 9, 25))["fills"][0]
    assert sale["held_days"] == 11 and sale["flag"] == "quick_loss"


@pytest.mark.parametrize("raw,plain", [
    ("wash-sale window (29d left since the loss sale)", "Waiting out the 30-day wash-sale rule"),
    ("regime gate did not pass — standing aside from new risk (sells unaffected)", "Market checks said stand aside"),
    ("unsettled proceeds (T+1) — frees up next session", "Waiting for sale cash to settle"),
    (None, "Blocked"),
])
def test_plain_skip_reasons(raw, plain):
    assert report.plain_skip_reason(raw) == plain


# --- daily pick ------------------------------------------------------------------------------------

def _none_run(date):
    return {"date": date, "status": "none", "gate": {
        "failing": ["Breadth > 55%"],
        "legs": [{"name": "Breadth > 55%", "key": "breadth_55", "ok": False}]}}


def test_a_pickless_week_names_the_usual_reason():
    runs = [_none_run(f"2026-09-{d}") for d in (21, 22, 23, 24, 25)] + [_none_run("2026-09-18")]
    s = report.daily_pick_summary(runs, D(2026, 9, 18), D(2026, 9, 25))
    assert (s["runs"], s["picks"], s["reason"]) == (5, 0, "Too few stocks were rising")


def test_picks_are_listed_oldest_first_and_carry_no_reason():
    runs = [{"date": "2026-09-24", "status": "pick", "pick": {"symbol": "DK"}},
            {"date": "2026-09-22", "status": "pick", "pick": {"symbol": "VLO"}},
            _none_run("2026-09-23")]
    s = report.daily_pick_summary(runs, D(2026, 9, 18), D(2026, 9, 25))
    assert s["symbols"] == ["VLO", "DK"] and s["reason"] is None


# --- price change vs adjusted ----------------------------------------------------------------------

def test_a_dividend_week_quotes_the_price_change_not_the_total_return():
    raw = [99.0, 100.0, 100.5, 101.0, 100.4, 100.0, 99.44]
    # ex-dividend on the 23rd: every earlier adjusted bar is scaled down by ~0.4%
    adj = [r * 0.996 for r in raw[:4]] + raw[4:]
    c = report.change(DATES, adj, D(2026, 9, 18), D(2026, 9, 25), raw=raw)
    assert c["pct"] == -0.56 and "split" not in c


def test_a_split_inside_the_period_falls_back_to_adjusted_prices():
    raw = [200.0, 200.0, 202.0, 101.0, 102.0, 103.0, 104.0]   # 2-for-1 on the 22nd
    adj = [100.0, 100.0, 101.0, 101.0, 102.0, 103.0, 104.0]
    c = report.change(DATES, adj, D(2026, 9, 18), D(2026, 9, 25), raw=raw)
    assert c["pct"] == 4.0 and c["split"] is True
