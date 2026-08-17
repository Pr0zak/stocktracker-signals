"""Peak-to-trough risk from the equity curve.

The account reported "+2.48% return, -1.80pp vs the S&P" and nothing else. That is a return without
its risk: a book that got there in a straight line and one that was 20% underwater on the way look
identical in those two numbers. The curve needed to say the other half was already on disk.

The case that matters most here is the SHORT series. A one-point curve has no drawdown, and the
honest answer is None rather than 0.0 -- "no drawdown yet" and "measured, and it was zero" are
different claims, and only the second is an achievement.
"""
from __future__ import annotations

from app.sandbox_job import drawdown_stats


def _curve(*equities):
    return [{"date": f"2026-08-{i + 1:02d}", "equity": e} for i, e in enumerate(equities)]


def test_worst_peak_to_trough_fall():
    # 10,500 -> 9,800 is the worst fall: 700/10500 = 6.67%.
    d = drawdown_stats(_curve(10_000, 10_500, 9_800, 10_200, 11_000, 10_600))
    assert d["max_drawdown_pct"] == 6.67
    assert d["max_drawdown_from"] == "2026-08-02"     # the peak it fell from
    assert d["max_drawdown_to"] == "2026-08-03"       # the trough


def test_current_drawdown_is_measured_from_the_all_time_high():
    # Ends at 10,600 against an 11,000 high: 400/11000 = 3.64%. Distinct from max_drawdown, which
    # is a historical worst and does not shrink as the account recovers.
    d = drawdown_stats(_curve(10_000, 10_500, 9_800, 10_200, 11_000, 10_600))
    assert d["current_drawdown_pct"] == 3.64
    assert d["peak_equity"] == 11_000.0


def test_both_are_positive_magnitudes():
    # A drawdown is a magnitude. Signing it invites a reader to add it to a return, which is how you
    # get "+2.48% and -6.67%" read as a net figure.
    d = drawdown_stats(_curve(10_000, 8_000))
    assert d["max_drawdown_pct"] > 0
    assert d["current_drawdown_pct"] > 0


def test_a_curve_that_only_rises_has_no_drawdown():
    d = drawdown_stats(_curve(10_000, 10_100, 10_400, 11_000))
    assert d["max_drawdown_pct"] == 0.0
    assert d["current_drawdown_pct"] == 0.0
    assert d["days_underwater"] == 0


def test_days_underwater_counts_nav_points_not_calendar_days():
    # Trading days, because the NAV log has one point per trading day. Calendar days would
    # overstate every stretch by the weekends inside it.
    d = drawdown_stats(_curve(10_000, 11_000, 10_500, 10_400, 10_900))
    assert d["days_underwater"] == 3


def test_a_series_too_short_to_have_a_drawdown_returns_none_not_zero():
    # The distinction the codebase keeps having to defend: absent is not zero. A brand-new account
    # showing "max drawdown 0.00%" claims a risk measurement it has not earned.
    for rows in ([], _curve(10_000)):
        d = drawdown_stats(rows)
        assert d["max_drawdown_pct"] is None
        assert d["current_drawdown_pct"] is None
        assert d["days_underwater"] is None


def test_non_finite_equity_points_are_ignored_rather_than_poisoning_the_series():
    # nav_row now refuses to write these, but the log is append-only and predates that guard, so a
    # historical row could still carry one. One bad point must not take the whole metric with it.
    rows = _curve(10_000, 11_000, 9_000)
    rows.insert(2, {"date": "bad", "equity": float("inf")})
    d = drawdown_stats(rows)
    assert d["max_drawdown_pct"] == 18.18       # 2000/11000, as if the bad row were absent
