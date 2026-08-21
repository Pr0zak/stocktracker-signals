"""The nightly cross-section store, and the two ways a store like this lies.

The first lie is REPLACE. `memory.record_verdict` uses `INSERT OR REPLACE`, which is safe there
because one writer owns every column of the row. scan.db is not that shape: the nightly job writes
the measured metrics, and a later pass writes derived columns — percentiles — onto rows that already
exist. REPLACE is a DELETE followed by an INSERT, so the second run of a night would silently drop
every percentile computed after the first, and nothing anywhere would report that it had. That is
what `test_reinserting_a_night_preserves_a_column_written_after_the_original_insert` pins down, and
it is the entire reason `insert_night` uses ON CONFLICT ... DO UPDATE with an explicit column list.

The second lie is zero. A breadth reading of `pct_above_sma50: 0.0` is not "we have no scan" — it is
"none of the market is above its 50-day average", which is about as bearish as a single number gets.
This codebase has shipped that defect before in other AI cards, so `breadth()` carries an
`available` flag and returns None for every reading when there is nothing to report, including on a
database it cannot even open.

Isolation is `SIGNALS_DATA_DIR` plus `importlib.reload()`: the module binds `_DATA_DIR` at import
time, so setting the env var without reloading gets the previous test's database.
"""
from __future__ import annotations

import importlib
import time

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A fresh, isolated scan.db per test."""
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    from app import scan_store as ss

    importlib.reload(ss)
    yield ss
    if ss._conn is not None:
        ss._conn.close()
        ss._conn = None


def _row(symbol: str, **over) -> dict:
    """A row shaped the way swing.metrics() delivers one: every key present, None where unmeasured."""
    row = {
        "symbol": symbol,
        "price": 100.0,
        "sma20": 99.0, "sma50": 95.0, "sma150": 90.0, "sma200": 85.0,
        "ema20": 98.0, "ema50": 94.0,
        "above_sma50": True, "above_sma200": True, "ma_stacked": True,
        "adr20_pct": 2.5, "atr14": 3.0, "atr14_pct": 3.0, "adx14": 25.0,
        "clv": 0.4, "rel_volume": 1.2, "dollar_volume_20d": 5_000_000.0,
        "mom_20d": 4.0, "mom_60d": 9.0, "rsi14": 58.0,
        "pct_off_52w_high": -6.0, "pct_vs_sma50": 5.2, "pct_vs_sma200": 17.6,
        "rel_strength_3mo": 3.0, "ema20_slope_pct": 1.1,
        "bars": 300,
        "unmeasured": [],
    }
    row.update(over)
    return row


# ---- round trip

def test_a_night_round_trips_through_insert_and_read(store):
    """What the scan wrote is what every reader gets back, with types intact."""
    assert store.insert_night([_row("AAPL"), _row("MSFT", price=200.0)], d="20260820") == 2

    assert store.latest_date() == "20260820"
    section = store.cross_section()
    assert [r["symbol"] for r in section] == ["AAPL", "MSFT"]

    aapl = store.symbol_row("aapl")            # case-folded on the way in
    assert aapl is not None
    assert aapl["d"] == "20260820"
    assert aapl["price"] == 100.0
    assert aapl["bars"] == 300
    assert aapl["above_sma50"] is True         # restored as a bool, not the stored 1
    assert aapl["unmeasured"] == []

    st = store.stats()
    assert (st["rows"], st["nights"], st["symbols"]) == (2, 1, 2)
    assert st["retain_days"] == store.retention_days() == 90


def test_a_symbol_absent_from_a_night_reads_as_none_rather_than_an_empty_row(store):
    store.insert_night([_row("AAPL")], d="20260820")
    assert store.symbol_row("NVDA") is None
    assert store.symbol_row("AAPL", "20260101") is None
    assert store.symbol_history("NVDA") == []


def test_history_returns_the_most_recent_nights_first(store):
    """`limit` has to mean the newest N; ascending order would silently return the oldest N."""
    for i, d in enumerate(("20260818", "20260819", "20260820")):
        store.insert_night([_row("AAPL", price=100.0 + i)], d=d)

    hist = store.symbol_history("AAPL", limit=2)
    assert [r["d"] for r in hist] == ["20260820", "20260819"]


# ---- the reason REPLACE was rejected

def test_reinserting_a_night_preserves_a_column_written_after_the_original_insert(store):
    """A percentile computed after the scan must survive a re-run of that same night.

    INSERT OR REPLACE deletes the row and inserts a new one, so every column the statement does not
    name goes back to its default. The nightly job names only the metrics it measured; the derived
    columns belong to a later pass. Under REPLACE, one re-run of tonight's scan wipes them and the
    reader downstream sees NULLs it will happily treat as "not measurable".
    """
    store.insert_night([_row("AAPL", adx14=25.0)], d="20260820")

    # A later pass adds and fills a derived column, exactly as the percentile step will.
    db = store._db()
    db.execute("ALTER TABLE scan ADD COLUMN rs_pct REAL")
    db.execute("UPDATE scan SET rs_pct = 88.0 WHERE d = ? AND symbol = ?", ("20260820", "AAPL"))
    db.commit()

    # The night is re-run — a manual re-scan, a retry after a partial failure.
    assert store.insert_night([_row("AAPL", adx14=31.0)], d="20260820") == 1

    row = store.symbol_row("AAPL")
    assert row["rs_pct"] == 88.0, "the derived column was wiped — this is the REPLACE bug"
    assert row["adx14"] == 31.0, "the re-run's own measurement should still win"
    assert store.stats()["rows"] == 1, "the re-run must update the night, not duplicate it"


def test_a_rerun_clears_a_metric_that_stopped_being_measurable(store):
    """The flip side: a column the writer DOES name must be nulled, not left at yesterday's value.

    swing.metrics() always returns every key, None included, so a metric that fell out of reach
    (too few bars after a data revision) arrives as an explicit None and has to overwrite. Carrying
    the previous value forward would report a stale number as a current one.
    """
    store.insert_night([_row("AAPL", adx14=25.0)], d="20260820")
    store.insert_night([_row("AAPL", adx14=None)], d="20260820")
    assert store.symbol_row("AAPL")["adx14"] is None


# ---- absent is never zero

def test_none_values_round_trip_as_none_and_never_as_zero(store):
    """An unmeasurable metric comes back None. 0.0 would be a claim; None is the truth."""
    store.insert_night(
        [_row("AAPL", adx14=None, rsi14=None, above_sma200=None, above_sma50=False,
              bars=None, unmeasured=["adx14", "rsi14", "above_sma200"])],
        d="20260820",
    )
    row = store.symbol_row("AAPL")
    assert row["adx14"] is None
    assert row["rsi14"] is None
    assert row["bars"] is None
    assert row["above_sma200"] is None          # unknown
    assert row["above_sma50"] is False          # measured, and false — a different thing entirely
    assert row["unmeasured"] == ["above_sma200", "adx14", "rsi14"]


def test_non_finite_numbers_are_stored_as_unmeasured_rather_than_as_nan(store):
    """NaN survives SQLite, poisons every aggregate it touches, and prints as "NaN" in the UI."""
    store.insert_night([_row("AAPL", atr14=float("nan"), mom_20d=float("inf"))], d="20260820")
    row = store.symbol_row("AAPL")
    assert row["atr14"] is None
    assert row["mom_20d"] is None


# ---- breadth

def test_breadth_with_no_scan_reports_unavailable_rather_than_zero_percent(store):
    """The defect this flag exists to prevent: 0% above the 50-day is a maximally bearish reading.

    An empty table means "we did not scan", and every reading has to say so. A caller that forgets
    to check `available` then renders a dash, not a crash warning the market never gave.
    """
    b = store.breadth()
    assert b["available"] is False
    assert b["n"] == 0
    for key in ("pct_above_sma50", "pct_above_sma200", "advancers", "decliners",
                "new_52w_highs", "new_52w_lows", "age_hours", "as_of"):
        assert b[key] is None, f"{key} must be None when there is no scan, got {b[key]!r}"


def test_breadth_computes_percentages_over_the_rows_that_could_be_measured(store):
    """Three of four above the 50-day is 75%, and the unmeasurable row is not counted as a no."""
    rows = [
        _row("AAA", above_sma50=True, above_sma200=True),
        _row("BBB", above_sma50=True, above_sma200=False),
        _row("CCC", above_sma50=True, above_sma200=False),
        _row("DDD", above_sma50=False, above_sma200=False),
    ]
    assert store.insert_night(rows, d="20260820") == 4

    b = store.breadth()
    assert b["available"] is True
    assert b["as_of"] == "20260820"
    assert b["n"] == 4
    assert b["pct_above_sma50"] == 75.0
    assert b["pct_above_sma200"] == 25.0
    assert b["age_hours"] is not None and b["age_hours"] < 1.0


def test_breadth_reports_none_when_too_little_of_the_universe_could_be_measured(store):
    """A rate over a minority of the names is a sample wearing the word "breadth"."""
    rows = [_row(f"S{i}", above_sma50=None) for i in range(9)]
    rows.append(_row("YES", above_sma50=True))
    store.insert_night(rows, d="20260820")

    b = store.breadth()
    assert b["available"] is True
    assert b["pct_above_sma50"] is None, "1 measurable row out of 10 is not 100% breadth"


def test_new_52w_highs_counts_closing_highs_and_lows_are_reported_as_unknown(store):
    """`pct_off_52w_high` is 0 at a new closing high; there is no low-side metric to count at all."""
    store.insert_night(
        [_row("AAA", pct_off_52w_high=0.0), _row("BBB", pct_off_52w_high=-0.004),
         _row("CCC", pct_off_52w_high=-8.0), _row("DDD", pct_off_52w_high=-40.0)],
        d="20260820",
    )
    b = store.breadth()
    assert b["new_52w_highs"] == 2, "the -0.004 row is float rounding at the high, not a laggard"
    assert b["new_52w_lows"] is None, "no low-side metric exists — None, never 0"


def test_the_near_high_band_is_reported_because_the_strict_count_reads_zero_most_nights(store):
    """The strict count compares a CLOSE against an INTRADAY 52-week high, so it only fires when a
    name closes on the exact high tick of its year. Measured against the live universe on
    2026-08-21: 0 names qualified, the closest sat 0.02% away, and 87 were within 1%. A field
    called `new_52w_highs` printing 0 every night reads as "nothing is leading", which is false.

    Both numbers are therefore reported, and they must stay distinct: the strict count keeps its
    exact meaning, and the band answers the question a reader is actually asking.
    """
    store.insert_night(
        [_row("ATHIGH", pct_off_52w_high=0.0),    # a genuine new closing high
         _row("NEAR", pct_off_52w_high=-0.5),     # 0.5% off — inside the band, not a new high
         _row("EDGE", pct_off_52w_high=-1.0),     # exactly on the boundary, inclusive
         _row("OUT", pct_off_52w_high=-1.5),      # outside it
         _row("FAR", pct_off_52w_high=-40.0)],
        d="20260820",
    )
    b = store.breadth()
    assert b["new_52w_highs"] == 1, "only the row at a new closing high counts as one"
    assert b["near_52w_high"] == 3, "ATHIGH, NEAR and EDGE are all within 1% of the high"
    assert b["near_52w_high_pct"] == 1.0, "the band must be published, not left for a caller to guess"


def test_the_near_high_band_is_labelled_even_when_there_is_no_scan(store):
    """The threshold belongs to this module, not to the scan, so an empty state can still say what
    band it would have counted. The COUNT stays None — absent is never zero."""
    b = store.breadth()
    assert b["available"] is False
    assert b["near_52w_high"] is None, "no scan means no count, not a count of zero"
    assert b["near_52w_high_pct"] == 1.0


def test_advancers_are_unknown_on_the_first_night_and_counted_from_the_second(store):
    """Zero advancers and zero decliners is a flat tape, which is not what one night of data means."""
    store.insert_night([_row("UP", price=10.0), _row("DOWN", price=10.0)], d="20260819")
    first = store.breadth()
    assert first["advancers"] is None and first["decliners"] is None

    store.insert_night([_row("UP", price=11.0), _row("DOWN", price=9.0)], d="20260820")
    second = store.breadth()
    assert (second["advancers"], second["decliners"]) == (1, 1)


def test_a_split_sized_move_is_counted_as_neither_an_advance_nor_a_decline(store):
    """Adjusted closes are re-adjusted retroactively, so a 4:1 split reads as -75% between nights."""
    store.insert_night([_row("SPLIT", price=400.0), _row("CALM", price=10.0)], d="20260819")
    store.insert_night([_row("SPLIT", price=100.0), _row("CALM", price=10.5)], d="20260820")

    b = store.breadth()
    assert b["decliners"] == 0, "the split must not be reported as the worst decline of the day"
    assert b["advancers"] == 1


def test_breadth_can_be_asked_for_an_older_night(store):
    store.insert_night([_row("AAA", above_sma50=True)], d="20260819")
    store.insert_night([_row("AAA", above_sma50=False)], d="20260820")

    assert store.breadth("20260819")["pct_above_sma50"] == 100.0
    assert store.breadth("2026-08-20")["pct_above_sma50"] == 0.0     # dashes tolerated on the way in
    assert store.breadth("20260101")["available"] is False


# ---- top / filtering

def test_top_sorts_descending_by_default_and_ascending_on_a_minus_prefix(store):
    store.insert_night(
        [_row("LOW", rel_strength_3mo=-2.0), _row("MID", rel_strength_3mo=4.0),
         _row("HIGH", rel_strength_3mo=9.0), _row("NONE", rel_strength_3mo=None)],
        d="20260820",
    )
    assert [r["symbol"] for r in store.top()] == ["HIGH", "MID", "LOW", "NONE"]
    assert [r["symbol"] for r in store.top(sort="-rel_strength")][:3] == ["LOW", "MID", "HIGH"]
    assert [r["symbol"] for r in store.top(limit=2)] == ["HIGH", "MID"]


def test_top_applies_min_max_and_boolean_filters(store):
    store.insert_night(
        [_row("A", adx14=30.0, above_sma200=True, rel_strength_3mo=1.0),
         _row("B", adx14=10.0, above_sma200=True, rel_strength_3mo=2.0),
         _row("C", adx14=40.0, above_sma200=False, rel_strength_3mo=3.0)],
        d="20260820",
    )
    assert [r["symbol"] for r in store.top(min_adx14=20)] == ["C", "A"]
    assert [r["symbol"] for r in store.top(above_sma200=True)] == ["B", "A"]
    assert [r["symbol"] for r in store.top(min_adx14=20, above_sma200=True)] == ["A"]
    assert [r["symbol"] for r in store.top(max_adx14=15)] == ["B"]


def test_an_unknown_filter_returns_nothing_rather_than_an_unfiltered_list(store):
    """A filter that silently does nothing hands back MORE rows and calls them filtered."""
    store.insert_night([_row("A"), _row("B")], d="20260820")
    assert store.top(min_adx14=1) != []
    assert store.top(min_nonsense=1) == []
    assert store.top(sort="'; DROP TABLE scan --") == []
    assert store.stats()["rows"] == 2, "the table is still there"


# ---- retention

def test_prune_removes_exactly_the_nights_past_the_window_and_leaves_the_rest(store):
    """Cut on the bar date, so a night that arrived late is still aged by when it happened."""
    day = 86_400
    nights = [time.strftime("%Y%m%d", time.gmtime(time.time() - n * day)) for n in (200, 120, 30, 1)]
    for d in nights:
        store.insert_night([_row("AAA"), _row("BBB")], d=d)
    assert store.stats()["nights"] == 4

    removed = store.prune()
    assert removed == 4, "the 200- and 120-day-old nights, two rows each"

    st = store.stats()
    assert st["nights"] == 2
    assert st["oldest"] == nights[2] and st["latest"] == nights[3]
    assert store.prune() == 0, "a second pass has nothing left to do"


def test_prune_accepts_a_shorter_window_than_the_default(store):
    day = 86_400
    recent, older = (time.strftime("%Y%m%d", time.gmtime(time.time() - n * day)) for n in (2, 40))
    store.insert_night([_row("AAA")], d=older)
    store.insert_night([_row("AAA")], d=recent)

    assert store.prune(retain_days=10) == 1
    assert store.latest_date() == recent
    assert store.stats()["rows"] == 1


def test_the_store_is_created_with_incremental_auto_vacuum(store):
    """auto_vacuum cannot be turned on after the first table exists without a full VACUUM.

    Nothing on this host ever runs one, so a 90-day table that turns over four times a year would
    park the file at its high-water mark forever. The pragma has to be set on the empty database.
    """
    store.insert_night([_row("AAA")], d="20260820")
    assert store._db().execute("PRAGMA auto_vacuum").fetchone()[0] == 2


def test_the_schema_and_the_migration_agree_on_the_column_list(store):
    """A column added to only one of _SCHEMA / _migrate diverges by machine age, not by code."""
    store.insert_night([_row("AAA")], d="20260820")
    have = {r["name"] for r in store._db().execute("PRAGMA table_info(scan)")}
    assert set(store._WRITABLE) <= have
    assert store._db().execute("PRAGMA user_version").fetchone()[0] == store._USER_VERSION


# ---- degraded database

def test_a_corrupt_database_reads_as_the_neutral_shape_rather_than_raising(store, tmp_path):
    """An unreadable file must look like "no scan", not like a 500 on every panel that reads it."""
    store._conn = None
    (tmp_path / "scan.db").write_bytes(b"this is not a database, it is a text file" * 64)

    b = store.breadth()
    assert b["available"] is False
    assert b["pct_above_sma50"] is None and b["advancers"] is None

    assert store.cross_section() == []
    assert store.top() == []
    assert store.symbol_row("AAPL") is None
    assert store.symbol_history("AAPL") == []
    assert store.latest_date() is None
    assert store.prune() == 0
    assert store.insert_night([_row("AAA")], d="20260820") == 0
    assert store.stats() == {"error": "unavailable"}


def test_a_row_without_a_symbol_is_refused_rather_than_stored_under_a_blank_key(store):
    """Silently storing it would put a row under symbol '' that every reader counts and none names."""
    assert store.insert_night([_row("AAA"), {"price": 5.0}, _row("BBB")], d="20260820") == 2
    assert [r["symbol"] for r in store.cross_section()] == ["AAA", "BBB"]


def test_an_unusable_night_key_writes_nothing_at_all(store):
    """A cross-section filed under a date nobody will look for is worse than a failed write."""
    assert store.insert_night([_row("AAA")], d="not-a-date") == 0
    assert store.latest_date() is None


# ---- retiring rows a full re-scan no longer produces

def test_a_full_rescan_retires_rows_it_stopped_producing(store):
    """insert_night upserts, so a re-run cannot remove a row by omission. That is correct for a
    partial writer and wrong for a full re-scan: when the integrity gate landed on 2026-08-21 it
    began rejecting twelve mixed-split-basis names, a forced re-run stored 3,101 rows instead of
    3,113, and all twelve corrupt rows stayed exactly where they were — still sorting to the top of
    every momentum ranking, because "stopped being measurable" had no way to mean "remove".
    """
    store.insert_night([_row("GOOD"), _row("CORRUPT")], d="20260820")
    assert len(store.cross_section("20260820")) == 2

    store.insert_night([_row("GOOD")], d="20260820")
    assert len(store.cross_section("20260820")) == 2, "the upsert alone leaves the stale row"

    assert store.retire_absent("20260820", ["GOOD"]) == 1
    assert [r["symbol"] for r in store.cross_section("20260820")] == ["GOOD"]


def test_retiring_against_an_empty_symbol_set_is_refused(store):
    """The one caller that could pass an empty set by accident is a run that measured nothing at
    all, and for that run "retire everything you did not produce" means "delete the night"."""
    store.insert_night([_row("AAA"), _row("BBB")], d="20260820")
    assert store.retire_absent("20260820", []) == 0
    assert store.retire_absent("20260820", None) == 0
    assert len(store.cross_section("20260820")) == 2


def test_retiring_one_night_leaves_every_other_night_alone(store):
    # The night is part of the key. A retire that reached across nights would silently truncate the
    # retention window from the wrong end.
    store.insert_night([_row("AAA"), _row("OLD")], d="20260819")
    store.insert_night([_row("AAA")], d="20260820")

    assert store.retire_absent("20260820", ["AAA"]) == 0
    assert len(store.cross_section("20260819")) == 2, "yesterday is not this run's business"
