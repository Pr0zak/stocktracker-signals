"""The recurring deposit, paid once or twice a month.

The sandbox's DCA used to fire exactly once per ET calendar month, cursored on `last_deposit_month`.
Paying it twice — the 1st and the 15th, the way a semi-monthly paycheque lands — needs a period key
finer than a month, and the whole risk of that change is double-crediting: cash that arrives twice
for one window is money the account never earned, and it would flatter every return figure measured
against the benchmark shadow. So these pin the cursor, not just the arithmetic.
"""
from __future__ import annotations

import datetime as dt
import importlib
import json
import tempfile
from pathlib import Path

import pytest

from app.sandbox_job import (
    ET, apply_recurring_deposit, deposit_frequency, due_deposit_periods, monthly_deposit_total,
)


def _at(day: str) -> dt.datetime:
    return dt.datetime.fromisoformat(day + "T15:45:00").replace(tzinfo=ET)


def _blob(amount=250.0, frequency="monthly", **over) -> dict:
    b = {
        "cash": 1000.0,
        "funded_total": 1000.0,
        "benchmark": {"symbol": "^GSPC", "shares": 1.0, "cost_basis": 1000.0},
        "settings": {"monthly_deposit": amount, "deposit_frequency": frequency},
    }
    b.update(over)
    return b


# ------------------------------------------------------------------ monthly: unchanged behaviour

def test_monthly_pays_once_per_calendar_month():
    b = _blob(frequency="monthly", last_deposit_period=None)
    assert due_deposit_periods(b, now=_at("2026-09-01")) == ["2026-09"]


def test_monthly_does_not_pay_twice_in_the_same_month():
    b = _blob(frequency="monthly", last_deposit_period="2026-09")
    assert due_deposit_periods(b, now=_at("2026-09-20")) == []


def test_monthly_pays_again_next_month():
    b = _blob(frequency="monthly", last_deposit_period="2026-09")
    assert due_deposit_periods(b, now=_at("2026-10-01")) == ["2026-10"]


def test_an_existing_ledger_keeps_its_old_cursor_without_migration():
    """Live blobs on disk carry `last_deposit_month`. Ignoring it would re-pay the current month."""
    b = _blob(frequency="monthly", last_deposit_month="2026-09")
    assert due_deposit_periods(b, now=_at("2026-09-20")) == []


def test_a_zero_amount_is_off():
    assert due_deposit_periods(_blob(amount=0.0), now=_at("2026-09-01")) == []


# ------------------------------------------------------------------ twice a month

def test_twice_monthly_pays_at_the_start_of_the_month():
    b = _blob(frequency="semimonthly", last_deposit_period="2026-08-H2")
    assert due_deposit_periods(b, now=_at("2026-09-01")) == ["2026-09-H1"]


def test_the_second_instalment_is_not_due_before_the_fifteenth():
    b = _blob(frequency="semimonthly", last_deposit_period="2026-09-H1")
    assert due_deposit_periods(b, now=_at("2026-09-14")) == []


def test_the_second_instalment_falls_due_on_the_fifteenth():
    b = _blob(frequency="semimonthly", last_deposit_period="2026-09-H1")
    assert due_deposit_periods(b, now=_at("2026-09-15")) == ["2026-09-H2"]


def test_neither_half_pays_twice():
    b = _blob(frequency="semimonthly", last_deposit_period="2026-09-H2")
    assert due_deposit_periods(b, now=_at("2026-09-16")) == []
    assert due_deposit_periods(b, now=_at("2026-09-30")) == []


def test_a_missed_first_half_is_caught_up_not_silently_dropped():
    """A month-opening outage must not cost the account a quarter of its yearly contributions."""
    b = _blob(frequency="semimonthly", last_deposit_period="2026-08-H2")
    assert due_deposit_periods(b, now=_at("2026-09-16")) == ["2026-09-H1", "2026-09-H2"]


def test_enabling_mid_month_funds_the_month_in_full():
    """Same as monthly, which credits the whole month's amount whenever it is first switched on."""
    b = _blob(frequency="semimonthly", last_deposit_period=None)
    assert due_deposit_periods(b, now=_at("2026-09-20")) == ["2026-09-H1", "2026-09-H2"]


# ------------------------------------------------------------------ switching frequency mid-month

def test_switching_to_twice_monthly_tops_the_month_up_rather_than_repaying_it():
    """A monthly cursor already paid this month's first half; only the mid-month half is left."""
    b = _blob(frequency="semimonthly", last_deposit_period="2026-09")
    assert due_deposit_periods(b, now=_at("2026-09-20")) == ["2026-09-H2"]


def test_switching_to_twice_monthly_before_the_fifteenth_pays_nothing_more():
    b = _blob(frequency="semimonthly", last_deposit_period="2026-09")
    assert due_deposit_periods(b, now=_at("2026-09-10")) == []


def test_switching_back_to_monthly_does_not_re_pay_the_month():
    b = _blob(frequency="monthly", last_deposit_period="2026-09-H1")
    assert due_deposit_periods(b, now=_at("2026-09-20")) == []


def test_an_unknown_frequency_falls_back_to_monthly():
    assert deposit_frequency({"deposit_frequency": "fortnightly"}) == "monthly"
    assert deposit_frequency({}) == "monthly"
    b = _blob(frequency="fortnightly", last_deposit_period=None)
    assert due_deposit_periods(b, now=_at("2026-09-20")) == ["2026-09"]


# ------------------------------------------------------------------ applying one instalment

def test_the_instalment_credits_cash_funding_and_the_benchmark_together():
    b = _blob(frequency="semimonthly", last_deposit_period="2026-09-H1")
    row = apply_recurring_deposit(b, amount=250.0, spy_price=5000.0, period="2026-09-H2",
                                  now=_at("2026-09-15"), ts=1.0)
    assert b["cash"] == 1250.0
    assert b["funded_total"] == 1250.0
    assert b["benchmark"]["shares"] == 1.05          # 250 / 5000 of an index share
    assert b["benchmark"]["cost_basis"] == 1250.0
    assert row["gross"] == 250.0 and row["cash_after"] == 1250.0
    assert row["side"] == "deposit" and row["source"] == "recurring"


def test_the_row_says_which_instalment_it_was():
    """Two identical $250 rows a fortnight apart cannot be told from one double-credited by a bug."""
    b = _blob(frequency="semimonthly")
    first = apply_recurring_deposit(b, amount=250.0, spy_price=5000.0, period="2026-09-H1",
                                    now=_at("2026-09-01"), ts=1.0)
    second = apply_recurring_deposit(b, amount=250.0, spy_price=5000.0, period="2026-09-H2",
                                     now=_at("2026-09-15"), ts=2.0)
    assert "1 of 2" in first["reason"] and "2 of 2" in second["reason"]
    monthly = apply_recurring_deposit(_blob(), amount=500.0, spy_price=5000.0, period="2026-09",
                                      now=_at("2026-09-01"), ts=3.0)
    assert monthly["reason"] == "Recurring monthly deposit $500"


def test_applying_advances_the_cursor_and_keeps_the_old_one_in_step():
    """The legacy key is written too: a rollback to the release before this one knows only that key,
    and would re-deposit the whole month across every book if it found none."""
    b = _blob(frequency="semimonthly", last_deposit_month="2026-08")
    apply_recurring_deposit(b, amount=250.0, spy_price=5000.0, period="2026-09-H2",
                            now=_at("2026-09-15"), ts=1.0)
    assert b["last_deposit_period"] == "2026-09-H2"
    assert b["last_deposit_month"] == "2026-09"
    assert due_deposit_periods(b, now=_at("2026-09-20")) == []


def test_an_outage_spanning_whole_months_is_not_repaid_in_arrears():
    """Settling months of arrears at one afternoon's quote would put a fiction in both legs of the
    benchmark comparison. A missed month stays missed, exactly as the monthly deposit always did."""
    b = _blob(frequency="semimonthly", last_deposit_period="2026-06-H2")
    assert due_deposit_periods(b, now=_at("2026-09-20")) == ["2026-09-H1", "2026-09-H2"]
    monthly = _blob(frequency="monthly", last_deposit_period="2026-06")
    assert due_deposit_periods(monthly, now=_at("2026-09-20")) == ["2026-09"]


def test_both_halves_of_a_caught_up_month_land_once_each():
    """The tick loops over the due list — that loop must leave the month fully paid, not partly."""
    b = _blob(frequency="semimonthly", last_deposit_period="2026-08-H2")
    now = _at("2026-09-16")
    for i, period in enumerate(due_deposit_periods(b, now=now)):
        apply_recurring_deposit(b, amount=250.0, spy_price=5000.0, period=period, now=now, ts=float(i))
    assert b["cash"] == 1500.0
    assert b["funded_total"] == 1500.0
    assert due_deposit_periods(b, now=now) == []


def test_the_monthly_total_reflects_the_number_of_instalments():
    assert monthly_deposit_total({"monthly_deposit": 250.0, "deposit_frequency": "semimonthly"}) == 500.0
    assert monthly_deposit_total({"monthly_deposit": 250.0, "deposit_frequency": "monthly"}) == 250.0
    assert monthly_deposit_total({}) == 0.0


# ------------------------------------------------------------------ the ledger on disk

@pytest.fixture()
def store(monkeypatch):
    """A fresh sandbox_store bound to a throwaway data dir — the module resolves its data directory
    at import time, so it is re-imported and re-bound per test (tests/test_sandbox_arms.py)."""
    d = tempfile.mkdtemp()
    monkeypatch.setenv("SIGNALS_DATA_DIR", d)
    from app import sandbox_store as s
    importlib.reload(s)
    s._DATA_DIR = Path(d)
    s._cache = {s.MAIN_ARM: s._load(s.MAIN_ARM)}
    return s


def test_the_new_setting_ships_off(store):
    """Every existing account must keep depositing exactly as it did before the setting existed."""
    assert store.DEFAULT_SETTINGS["deposit_frequency"] == "monthly"
    assert store.get("main")["settings"]["deposit_frequency"] == "monthly"


def test_a_pre_feature_ledger_loads_with_the_monthly_default_and_keeps_its_cursor(store):
    """There is no versioned migration here — the defaults merge in _load IS the migration, so a
    blob written before the setting existed has to come back monthly with its cursor intact."""
    legacy = {
        "arm": "main", "cash": 1000.0, "funded_total": 1000.0,
        "benchmark": {"symbol": "^GSPC", "shares": 1.0, "cost_basis": 1000.0},
        "settings": {"monthly_deposit": 500.0},      # no deposit_frequency key at all
        "last_deposit_month": "2026-09",
    }
    store._paths("main")[0].write_text(json.dumps(legacy))
    store._cache.pop("main", None)
    blob = store.get("main")
    assert blob["settings"]["deposit_frequency"] == "monthly"
    assert blob["last_deposit_month"] == "2026-09"
    assert due_deposit_periods(blob, now=_at("2026-09-25")) == []
    assert due_deposit_periods(blob, now=_at("2026-10-01")) == ["2026-10"]


def test_a_cloned_arm_inherits_the_deposit_cursor(store):
    """The opposite polarity to the day cursor, which is deliberately NOT inherited: an arm that
    started its life with an empty deposit cursor would re-pay the current period on its first tick,
    handing the clone money the account it is being compared against never received."""
    store.save({**store.get("main"), "last_deposit_period": "2026-09-H1"}, "main")
    assert store.create_arm("rules", engine="rules",
                            clone_from="main")["last_deposit_period"] == "2026-09-H1"


def test_a_clone_of_a_pre_feature_ledger_inherits_the_old_cursor_too(store):
    b = dict(store.get("main"))
    b.pop("last_deposit_period", None)
    b["last_deposit_month"] = "2026-09"
    store.save(b, "main")
    assert store.create_arm("fast", engine="llm",
                            clone_from="main")["last_deposit_period"] == "2026-09"
