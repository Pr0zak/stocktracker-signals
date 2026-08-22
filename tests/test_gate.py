"""The five-leg market gate, and the one thing it must never do: guess.

The gate exists because every buy-side consumer in this service is a ranker. A ranker fed a bleeding
tape still returns its top five, with the same confidence it had in April. gate.py is the lookup that
lets those consumers stand aside for a reason they can name — SPY and QQQ over their 50-EMAs, more
than 55% of the scanned universe over its own 50-SMA, VIX under 20, and SPY's 20-day return positive.

The defect this file is really about is the one the app-ui-honesty-invariants note keeps recording:
absence rendered as a confident reading. Four of the five legs come off a network fetch and the fifth
comes out of a nightly systemd oneshot, so "we could not measure this" is a NORMAL operating state,
not an edge case — a fresh install has no scan at all, and `scan_store.breadth()` says so with
`available: False` and a null reading. If a null breadth becomes `passed: false`, the gate is
indistinguishable from a bearish market and every consumer learns to ignore a real closure. If it
becomes `passed: true`, the gate has invented an all-clear out of an empty table.

So the invariant is tested from both ends here: an unmeasurable leg's `ok` is None, its NAME lands in
`unmeasured` and NOT in `failing`, and `passed` is None rather than False. The one case where an
unknown does not win is when a measurable leg explicitly failed — we know enough to stand aside and
the reason we report is real.

`test_a_null_passed_survives_the_round_trip_as_none_and_not_as_false` covers the same distinction on
disk. SQLite has no boolean; `bool(0)` and `bool(None)` are both False in Python, so the third state
is lost by any generic restore, and the history would claim the gate failed on a day it specifically
declined to decide.

Isolation is `SIGNALS_DATA_DIR` plus `importlib.reload()`, matching tests/test_scan_store.py: the
store binds `_DATA_DIR` at import time. `importlib.reload` re-executes into the SAME module object,
so gate.py's `from . import scan_store` still points at the isolated store without reloading gate.
"""
from __future__ import annotations

import asyncio
import importlib

import pytest

from app import gate


# --------------------------------------------------------------------------------- fixtures/stubs

class _Bars:
    """A duck-typed stand-in for market.Series: gate reads `closes` (and `dates`) and nothing else.

    Deliberately not a real Series — legs_from() is pure by contract, and a test that needed the
    dataclass would be a test that could drift into needing the fetch stack behind it.
    """

    def __init__(self, closes: list[float], dates: list[str] | None = None):
        self.closes = closes
        self.dates = dates or []


def _bars(last: float, level: float = 100.0, n: int = 260) -> _Bars:
    """`n` flat bars at `level` with a final close of `last`.

    Flat history makes both price legs analytically obvious: the 50-EMA of a constant series is that
    constant, so the last bar sets the margin over it, and the 20-day return is `last` against
    `level`. That is what lets the market_score tests below assert on a MARGIN rather than on a
    number that happens to fall out of a fixture.
    """
    return _Bars([level] * (n - 1) + [last], dates=[f"2026{(i % 12) + 1:02d}01" for i in range(n)])


def _breadth(pct: float | None = 65.0, *, available: bool = True) -> dict:
    """A scan_store.breadth() reading, in the shape the real one returns."""
    return {"available": available, "as_of": "20260821", "n": 3000,
            "pct_above_sma50": pct, "age_hours": 11.0}


def _all_pass(**over):
    """The four inputs for a market clearing every line, overridable one leg at a time."""
    kw = {"spy": _bars(103.0), "qqq": _bars(104.0), "vix": 15.0, "breadth": _breadth(65.0)}
    kw.update(over)
    return gate.legs_from(kw["spy"], kw["qqq"], kw["vix"], kw["breadth"])


def _by_key(legs: list[dict], key: str) -> dict:
    return next(l for l in legs if l["key"] == key)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A fresh, isolated scan.db per test — the gate table included."""
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    from app import scan_store as ss

    importlib.reload(ss)
    yield ss
    if ss._conn is not None:
        ss._conn.close()
        ss._conn = None


# ------------------------------------------------------------------------------------- the legs

def test_the_five_legs_appear_in_the_published_order_with_the_published_keys():
    legs = _all_pass()
    assert [l["key"] for l in legs] == [
        "spy_above_ema50", "qqq_above_ema50", "breadth_55", "vix_under_20", "spy_mom_20d",
    ]
    assert all({"name", "key", "ok", "value", "threshold", "note"} <= set(l) for l in legs)


def test_a_market_clearing_every_line_passes_the_gate_with_nothing_failing_or_unknown():
    legs = _all_pass()
    passed, failing, unmeasured = gate.verdict(legs)
    assert [l["ok"] for l in legs] == [True, True, True, True, True]
    assert passed is True
    assert failing == [] and unmeasured == []


def test_a_single_failing_leg_closes_the_gate_and_is_named_while_nothing_is_unmeasured():
    legs = _all_pass(vix=26.0)
    passed, failing, unmeasured = gate.verdict(legs)
    assert _by_key(legs, "vix_under_20")["ok"] is False
    assert passed is False
    assert failing == ["VIX < 20"]
    # The point of this assertion: a failing leg is a MEASUREMENT, and it must never leak into the
    # bucket that means "we could not look".
    assert unmeasured == []


def test_a_breadth_reading_that_does_not_exist_is_unknown_and_never_a_silent_pass_or_fail(store):
    """THE invariant, driven by the real store rather than a hand-written dict.

    `scan_store.breadth()` on a database with no scan in it returns `available: False` with every
    reading None. That is the state of a fresh install and of any morning the nightly oneshot did not
    run, so this path is reached in normal operation — it is not a defensive branch.
    """
    reading = store.breadth()
    assert reading["available"] is False and reading["pct_above_sma50"] is None

    legs = _all_pass(breadth=reading)
    passed, failing, unmeasured = gate.verdict(legs)
    leg = _by_key(legs, "breadth_55")

    assert leg["ok"] is None, "an unreadable leg must be None — not False, not True"
    assert leg["value"] is None, "and it must not report a number it did not measure"
    assert passed is None, "unknown is not a closure: passed must be None, never False"
    assert leg["name"] in unmeasured
    assert leg["name"] not in failing
    assert failing == [], "nothing measurable failed, so nothing may be reported as failing"


def test_a_failing_leg_outranks_an_unknown_one_because_the_reason_to_stand_aside_is_real():
    legs = _all_pass(vix=26.0, breadth=_breadth(None, available=False))
    passed, failing, unmeasured = gate.verdict(legs)
    assert passed is False
    assert failing == ["VIX < 20"]
    assert unmeasured == ["Breadth > 55%"]
    # Both facts survive: the gate is closed AND we still say which leg we could not read, so a
    # consumer is never told the market failed on evidence that was missing.
    assert _by_key(legs, "breadth_55")["ok"] is None


def test_a_failed_price_fetch_makes_its_legs_unknown_rather_than_bearish():
    """A None series is an upstream outage, not a stock below its average."""
    legs = gate.legs_from(None, _bars(104.0), 15.0, _breadth(65.0))
    passed, failing, unmeasured = gate.verdict(legs)
    assert _by_key(legs, "spy_above_ema50")["ok"] is None
    assert _by_key(legs, "spy_mom_20d")["ok"] is None
    assert passed is None and failing == []
    assert unmeasured == ["SPY > 50-EMA", "SPY 20-day momentum > 0"]


def test_a_series_too_short_for_a_fifty_bar_ema_reports_unknown_instead_of_seeding_one():
    short = _Bars([100.0] * 30 + [103.0])
    legs = gate.legs_from(short, _bars(104.0), 15.0, _breadth(65.0))
    assert _by_key(legs, "spy_above_ema50")["ok"] is None
    assert _by_key(legs, "spy_above_ema50")["threshold"] is None
    # 31 bars is plenty for a 20-day return, so that leg IS measurable — the two legs fail
    # independently, which is the point of measuring them independently.
    assert _by_key(legs, "spy_mom_20d")["ok"] is True


def test_a_scan_that_exists_but_could_not_call_breadth_is_also_unknown():
    """`available: True` with a None percentage — scan_store's coverage floor, not an empty file."""
    legs = _all_pass(breadth=_breadth(None, available=True))
    assert _by_key(legs, "breadth_55")["ok"] is None
    assert gate.verdict(legs)[0] is None


def test_breadth_exactly_on_its_threshold_does_not_clear_it():
    assert _by_key(_all_pass(breadth=_breadth(55.0)), "breadth_55")["ok"] is False
    assert _by_key(_all_pass(breadth=_breadth(55.1)), "breadth_55")["ok"] is True


# ---------------------------------------------------------------------------------- market_score

def test_the_score_is_none_when_any_leg_is_unmeasurable_rather_than_averaging_a_partial_set():
    assert gate.market_score(_all_pass()) is not None
    assert gate.market_score(_all_pass(breadth=_breadth(None, available=False))) is None
    assert gate.market_score(_all_pass(vix=None)) is None
    assert gate.market_score(gate.legs_from(None, None, None, None)) is None


def test_a_market_scraping_over_its_lines_scores_far_below_one_clearing_them_comfortably():
    """The reason the score is not "legs passing / 5 * 100": both of these pass all five legs."""
    barely = gate.market_score(gate.legs_from(
        _bars(100.05), _bars(100.05), 19.9, _breadth(55.5)))
    comfortably = gate.market_score(gate.legs_from(
        _bars(112.0), _bars(112.0), 13.0, _breadth(85.0)))

    assert gate.verdict(gate.legs_from(_bars(100.05), _bars(100.05), 19.9, _breadth(55.5)))[0] is True
    assert gate.verdict(gate.legs_from(_bars(112.0), _bars(112.0), 13.0, _breadth(85.0)))[0] is True
    assert comfortably > barely + 25, (barely, comfortably)
    # Half the credit is for crossing the line at all, so an all-pass floor is ~50 and a full sweep
    # is 100. Pinning the ends stops a future re-weighting from silently changing what a score means.
    assert 45.0 <= barely <= 60.0
    assert comfortably >= 95.0
    assert 0.0 <= barely <= 100.0 and 0.0 <= comfortably <= 100.0


def test_a_failing_leg_contributes_nothing_to_the_score():
    good = gate.market_score(_all_pass())
    bad = gate.market_score(_all_pass(vix=26.0))
    assert bad < good
    # A failing leg earns neither the crossing credit nor margin credit, so it costs a full fifth.
    assert bad <= good - 10.0


# ---------------------------------------------------------------------------------------- history

def test_a_gate_row_round_trips_through_the_store_with_every_field_intact(store):
    legs = _all_pass()
    passed, failing, unmeasured = gate.verdict(legs)
    row = {"passed": passed, "market_score": gate.market_score(legs), "legs": legs,
           "failing": failing, "unmeasured": unmeasured, "evaluated_at": 1787353715.3}

    assert store.record_gate(row, d="20260821") is True

    back = store.gate_for("20260821")
    assert back is not None
    assert back["d"] == "20260821"
    assert back["passed"] is True
    assert back["market_score"] == row["market_score"]
    assert back["failing"] == [] and back["unmeasured"] == []
    assert [l["key"] for l in back["legs"]] == [l["key"] for l in legs]
    assert back["ts"] == 1787353715.3


def test_a_null_passed_survives_the_round_trip_as_none_and_not_as_false(store):
    """The third state, on disk.

    SQLite has no boolean and Python's `bool(None)` is False, so a generic restore collapses "the
    gate could not decide" into "the gate failed" — a bearish claim about a day on which we made no
    claim at all. Recorded from the real unmeasurable-breadth path so the value under test is the one
    the gate actually produces.
    """
    legs = _all_pass(breadth=_breadth(None, available=False))
    passed, failing, unmeasured = gate.verdict(legs)
    assert passed is None

    store.record_gate({"passed": passed, "market_score": gate.market_score(legs), "legs": legs,
                       "failing": failing, "unmeasured": unmeasured}, d="20260820")

    back = store.gate_for("20260820")
    assert back["passed"] is None
    assert back["passed"] is not False, "None must not be restored as an explicit failure"
    assert back["market_score"] is None, "a partial set has no score to store"
    assert back["failing"] == []
    assert back["unmeasured"] == ["Breadth > 55%"]

    # And the raw column really is NULL, not 0 — the distinction has to hold in SQL too, because a
    # future query that counts closed days would otherwise count this one.
    raw = store._db().execute("SELECT passed FROM gate WHERE d = ?", ("20260820",)).fetchone()
    assert raw["passed"] is None


def test_history_comes_back_newest_first_and_keeps_each_day_distinct(store):
    for day, vix in (("20260818", 15.0), ("20260819", 26.0), ("20260820", 15.0)):
        legs = _all_pass(vix=vix)
        p, f, u = gate.verdict(legs)
        store.record_gate({"passed": p, "market_score": gate.market_score(legs), "legs": legs,
                           "failing": f, "unmeasured": u}, d=day)

    hist = store.gate_history(limit=10)
    assert [r["d"] for r in hist] == ["20260820", "20260819", "20260818"]
    assert [r["passed"] for r in hist] == [True, False, True]
    assert hist[1]["failing"] == ["VIX < 20"]


def test_re_recording_a_day_updates_it_in_place_rather_than_duplicating_it(store):
    legs = _all_pass()
    p, f, u = gate.verdict(legs)
    store.record_gate({"passed": p, "market_score": 91.0, "legs": legs, "failing": f,
                       "unmeasured": u}, d="20260821")
    closed = _all_pass(vix=26.0)
    p2, f2, u2 = gate.verdict(closed)
    store.record_gate({"passed": p2, "market_score": 40.0, "legs": closed, "failing": f2,
                       "unmeasured": u2}, d="20260821")

    hist = store.gate_history()
    assert len(hist) == 1
    assert hist[0]["passed"] is False and hist[0]["failing"] == ["VIX < 20"]


def test_a_day_that_was_never_recorded_is_none_and_not_an_empty_gate(store):
    """Absent is not "the gate ran and could not decide" — the caller must be able to tell."""
    assert store.gate_for("20260101") is None
    assert store.gate_history() == []


# --------------------------------------------------------------------------------------- evaluate

def test_evaluate_assembles_the_contract_shape_and_records_it(store, monkeypatch):
    """Async tested as a sync def + asyncio.run — no pytest-asyncio in this repo."""
    async def fake_series(client, symbol, *a, **kw):
        return _bars(103.0 if symbol == "SPY" else 104.0)

    async def fake_quotes(client, symbols):
        return {"^VIX": {"price": 14.5, "state": "REGULAR"}}

    monkeypatch.setattr(gate.market, "fetch_series", fake_series)
    monkeypatch.setattr(gate.market_now, "fetch_quotes", fake_quotes)
    store.insert_night([{"symbol": "AAA", "price": 10.0, "above_sma50": True},
                        {"symbol": "BBB", "price": 10.0, "above_sma50": True}], d="20260821")

    out = asyncio.run(gate.evaluate(None))

    assert out["passed"] is True
    assert out["available"] is True
    assert out["failing"] == [] and out["unmeasured"] == []
    assert isinstance(out["evaluated_at"], float)
    assert out["market_score"] is not None
    assert isinstance(out["note"], str) and out["note"]
    assert len(out["as_of"]) == 8 and out["as_of"].isdigit()
    # It filed itself: the history is the point of evaluating on a schedule.
    assert store.gate_for(out["as_of"])["passed"] is True


def test_evaluate_degrades_to_unavailable_when_every_input_fails_and_writes_no_row(store, monkeypatch):
    """Total outage. Nothing measurable, so nothing is claimed — and nothing is filed either.

    The row is keyed on the day and upserted, so recording a five-null evaluation would let a
    30-second network blip overwrite a real morning reading with "we know nothing".
    """
    async def boom(*a, **kw):
        raise RuntimeError("yahoo is down")

    monkeypatch.setattr(gate.market, "fetch_series", boom)
    monkeypatch.setattr(gate.market_now, "fetch_quotes", boom)

    out = asyncio.run(gate.evaluate(None, d="20260821"))

    assert out["available"] is False
    assert out["passed"] is None, "an outage is not a bearish market"
    assert out["market_score"] is None
    assert out["failing"] == []
    assert len(out["unmeasured"]) == 5
    assert [l["ok"] for l in out["legs"]] == [None] * 5
    assert store.gate_for("20260821") is None
