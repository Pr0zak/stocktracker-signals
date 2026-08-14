"""The account's age is derived from a birth date, not stored as a number.

`current_age: 47` was typed once and then aged silently. It is not a cosmetic field — the glidepath
that governs the whole strategy is (retirement_age - current_age), so a number entered by hand keeps
reporting the runway it had on the day it was entered, growing a year too long for every year that
passes. Storing the birth date instead makes the age a derived value that cannot drift.

Two properties matter beyond the arithmetic:

1. The stored `current_age` remains the fallback, so an account configured before `birth_date`
   existed keeps behaving exactly as it did until someone sets a date.
2. `birth_date` never reaches the model. The entire settings dict is json.dumps'd into both the
   daily and weekly prompts, so anything left in it is sent verbatim; the model reasons about
   runway, which the derived age already carries in full.
"""
from __future__ import annotations

import datetime as dt

from app.sandbox_job import age_on, effective_age, settings_for_prompt


def test_age_before_and_after_the_birthday_this_year():
    # The day before the birthday the age has not turned over yet; on the day, it has.
    assert age_on("1979-08-20", dt.date(2026, 8, 19)) == 46
    assert age_on("1979-08-20", dt.date(2026, 8, 20)) == 47
    assert age_on("1979-08-20", dt.date(2026, 8, 21)) == 47


def test_age_advances_with_the_calendar():
    """The whole point: the same stored value reads differently a year later."""
    born = "1979-03-01"
    assert age_on(born, dt.date(2026, 8, 14)) == 47
    assert age_on(born, dt.date(2027, 8, 14)) == 48
    assert age_on(born, dt.date(2044, 8, 14)) == 65


def test_leap_day_birth_turns_over_on_march_1_in_a_common_year():
    # 2027 is not a leap year, so Feb 29 does not exist in it. Comparing (month, day) tuples puts the
    # turnover on Mar 1 without a special case.
    assert age_on("2000-02-29", dt.date(2027, 2, 28)) == 26
    assert age_on("2000-02-29", dt.date(2027, 3, 1)) == 27
    # In a leap year the birthday itself exists and counts.
    assert age_on("2000-02-29", dt.date(2028, 2, 29)) == 28


def test_unusable_birth_dates_return_none_rather_than_a_number():
    # None, not 0 and not a guess. A wrong age is worse than an absent one — it silently lengthens or
    # shortens the runway, and nothing downstream can tell it was fabricated.
    for bad in (None, "", "   ", "not-a-date", "1979-13-01", "20/08/1979", 47):
        assert age_on(bad) is None
    # A birth date in the future is not an age.
    assert age_on("2030-01-01", dt.date(2026, 8, 14)) is None


def test_stored_age_is_the_fallback_and_the_date_wins_when_both_exist():
    today = dt.date(2026, 8, 14)
    # Legacy account: no date, so the stored number is used unchanged.
    assert effective_age({"current_age": 47}, today) == 47
    # Date set: derived, and the stale number is ignored even when it disagrees.
    assert effective_age({"birth_date": "1979-03-01", "current_age": 40}, today) == 47
    # Neither: absent, not zero.
    assert effective_age({}, today) is None
    # Age 0 is a real answer, so it must survive the fallback's None-check rather than be treated as
    # missing and replaced by the stored number.
    assert effective_age({"birth_date": "2026-01-01", "current_age": 47}, today) == 0


def test_the_prompt_gets_the_derived_age_and_never_the_birth_date():
    today = dt.date(2027, 8, 14)
    s = {"birth_date": "1979-03-01", "current_age": None, "retirement_age": 65, "cash_floor_pct": 5.0}
    out = settings_for_prompt(s, today)
    assert "birth_date" not in out
    assert out["current_age"] == 48                     # derived for THIS day, not the stored None
    assert out["retirement_age"] == 65                  # everything else passes through untouched
    assert out["cash_floor_pct"] == 5.0
    # The input is not mutated — the caller keeps writing the real settings back to the ledger.
    assert s["birth_date"] == "1979-03-01"
    assert s["current_age"] is None


def test_prompt_settings_omit_the_age_entirely_when_it_is_unknown():
    # Absent, not invented. A key present with a wrong value would be read by the model as fact.
    out = settings_for_prompt({"retirement_age": 65}, dt.date(2026, 8, 14))
    assert "current_age" not in out
    assert "birth_date" not in out
