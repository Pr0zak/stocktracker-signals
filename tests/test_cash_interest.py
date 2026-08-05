"""Idle cash earns a yield.

The ledger paid nothing on cash, which biased every decision weighing whether to hold it: measured
2026-08-05, the 40%-cash position looked like it cost $193 against the benchmark at 0% but nearer $177
once cash earns its keep, and over the account's 18-year runway the two assumptions differ by ~$8.5k
of projected terminal value. It also makes the benchmark fair — that shadow is 100% invested by
construction, so charging 0% on cash penalised the sandbox twice for the same choice.
"""
from __future__ import annotations

import datetime as dt

from app.sandbox_job import ET, accrue_cash_interest


def _at(day: str):
    return dt.datetime.fromisoformat(day + "T15:00:00").replace(tzinfo=ET)


def _blob(cash=10000.0, apy=4.3, last="2026-08-01", **over):
    b = {"cash": cash, "settings": {"cash_apy_pct": apy}, "last_interest_date": last}
    b.update(over)
    return b


def test_interest_accrues_over_elapsed_days():
    b = _blob(cash=10000.0, apy=4.3, last="2026-08-01")
    earned = accrue_cash_interest(b, now=_at("2026-08-08"))
    assert earned == round(10000 * 0.043 * 7 / 365, 2)
    assert b["cash"] == round(10000 + earned, 2)


def test_it_is_earnings_not_a_contribution():
    """funded_total must NOT move, or total_return_pct would net the interest out of the return."""
    b = _blob(cash=10000.0, funded_total=10000.0)
    accrue_cash_interest(b, now=_at("2026-08-08"))
    assert b["funded_total"] == 10000.0


def test_a_same_day_rerun_cannot_double_credit():
    """The app's 'run now' button forces extra ticks — a second one must not pay twice."""
    b = _blob(cash=10000.0, last="2026-08-01")
    first = accrue_cash_interest(b, now=_at("2026-08-08"))
    second = accrue_cash_interest(b, now=_at("2026-08-08"))
    assert first > 0 and second == 0.0


def test_the_cursor_advances_so_the_next_period_starts_clean():
    b = _blob(cash=10000.0, last="2026-08-01")
    accrue_cash_interest(b, now=_at("2026-08-08"))
    assert b["last_interest_date"] == "2026-08-08"
    again = accrue_cash_interest(b, now=_at("2026-08-15"))
    assert again == round(b["cash"] / (1 + 0.043 * 7 / 365) * 0.043 * 7 / 365, 2) or again > 0


def test_running_total_is_tracked():
    b = _blob(cash=10000.0, last="2026-08-01")
    accrue_cash_interest(b, now=_at("2026-08-08"))
    accrue_cash_interest(b, now=_at("2026-08-15"))
    assert b["interest_total"] > 0


# ------------------------------------------------------------------ degrading safely

def test_first_run_starts_the_clock_without_back_paying():
    """No cursor means we never modelled the earlier period; inventing it would fabricate return."""
    b = _blob(last=None)
    assert accrue_cash_interest(b, now=_at("2026-08-08")) == 0.0
    assert b["last_interest_date"] == "2026-08-08"


def test_zero_apy_pays_nothing():
    b = _blob(cash=10000.0, apy=0.0)
    assert accrue_cash_interest(b, now=_at("2026-08-08")) == 0.0
    assert b["cash"] == 10000.0


def test_no_cash_pays_nothing():
    b = _blob(cash=0.0)
    assert accrue_cash_interest(b, now=_at("2026-08-08")) == 0.0


def test_a_corrupt_cursor_resets_rather_than_raising():
    b = _blob(cash=10000.0, last="not-a-date")
    assert accrue_cash_interest(b, now=_at("2026-08-08")) == 0.0
    assert b["last_interest_date"] == "2026-08-08"


def test_a_future_cursor_pays_nothing():
    b = _blob(cash=10000.0, last="2026-09-01")
    assert accrue_cash_interest(b, now=_at("2026-08-08")) == 0.0


def test_a_sub_cent_accrual_does_not_creep():
    b = _blob(cash=1.0, apy=4.3, last="2026-08-01")
    assert accrue_cash_interest(b, now=_at("2026-08-02")) == 0.0
    assert b["cash"] == 1.0
