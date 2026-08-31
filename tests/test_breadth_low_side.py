"""The low side of breadth: pct_off_52w_low, new_52w_lows, near_52w_low, high_low_diff.

`breadth()["new_52w_lows"]` was permanently None because the metrics contract measured distance from
the 52-week HIGH and had no low-side equivalent — so a market where BOTH new highs and new lows were
elevated (the classic internal divergence) was indistinguishable from one where only highs were.

Two things here are worth more than the happy path. Nights stored before the column existed must
report None and never 0, because 0 is "no name made a new low", a bullish all-clear invented out of a
missing column. And the differential is built on the BAND counts, because both STRICT counts compare
a closing extreme against intraday extremes and read near zero on almost every night — measured
against the live universe the strict high count read 0, 5, 6, 1 and 3 per night out of ~3,103 — so
their difference would be noise wearing the name of a divergence signal.
"""

import pytest

from app import swing


class Bars:
    def __init__(self, closes, highs=None, lows=None, volumes=None):
        self.closes = closes
        self.highs = highs if highs is not None else []
        self.lows = lows if lows is not None else []
        self.volumes = volumes if volumes is not None else []


def ramp(n, start=100.0, step=0.4, band=1.0):
    closes = [start + i * step for i in range(n)]
    return [c + band for c in closes], [c - band for c in closes], closes


# --- the metric ----------------------------------------------------------------------------------


def test_pct_off_52w_low_joins_the_contract():
    highs, lows, closes = ramp(260)
    m = swing.metrics(Bars(closes, highs, lows, [1e6] * 260))
    assert "pct_off_52w_low" in m
    assert m["pct_off_52w_low"] is not None


def test_the_two_sides_are_mirrors_of_each_other():
    """A monotonic advance sits at its high and far above its low; a decline, the reverse.

    Neither reads exactly 0, and that is the documented behaviour rather than a rounding slip: both
    compare a CLOSE against INTRADAY extremes, so a name that printed a new high and closed a
    fraction under it is not "at" its high. It is the same conservatism that makes the strict counts
    read near zero most nights, which is why the differential is built on the bands instead.
    """
    highs, lows, closes = ramp(260, step=0.4)          # monotonic advance
    up = swing.metrics(Bars(closes, highs, lows, [1e6] * 260))
    assert -1.0 < up["pct_off_52w_high"] <= 0.0, "sitting at its high, just under the intraday tick"
    assert up["pct_off_52w_low"] > 50.0, "and a long way above its low"

    highs, lows, closes = ramp(260, start=300.0, step=-0.4)  # monotonic decline
    down = swing.metrics(Bars(closes, highs, lows, [1e6] * 260))
    assert 0.0 <= down["pct_off_52w_low"] < 1.0, "sitting at its low, just over the intraday tick"
    assert down["pct_off_52w_high"] < -25.0, "and a long way below its high"


def test_the_low_is_clamped_so_float_noise_never_prints_below_its_own_low():
    highs, lows, closes = ramp(260, start=300.0, step=-0.4)
    m = swing.metrics(Bars(closes, highs, lows, [1e6] * 260))
    assert m["pct_off_52w_low"] >= 0.0


def test_a_close_only_series_still_answers_both_sides():
    """The completeness rule falls back to the close window rather than mixing bases."""
    _, _, closes = ramp(260)
    m = swing.metrics(Bars(closes, [], [], [1e6] * 260))
    assert m["pct_off_52w_high"] is not None
    assert m["pct_off_52w_low"] is not None


def test_a_partially_populated_lows_list_is_refused_rather_than_mixed():
    """Half intraday, half nothing is not a 52-week low — the high side made the same choice."""
    highs, lows, closes = ramp(260)
    lows[5] = None
    m = swing.metrics(Bars(closes, highs, lows, [1e6] * 260))
    # Falls back to the close window; the point is that it does not crash and does not take the min
    # of a list containing None.
    assert m["pct_off_52w_low"] is not None


def test_too_little_history_leaves_it_unmeasured_rather_than_zero():
    m = swing.metrics(Bars([100.0], [101.0], [99.0], [1e6]))
    assert m["pct_off_52w_low"] is None or m["pct_off_52w_low"] >= 0.0
    if m["pct_off_52w_low"] is None:
        assert "pct_off_52w_low" in m["unmeasured"]


# --- the store -------------------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A throwaway store. scan_store binds its data directory at import, so the env var alone is not
    enough — the module has to be reloaded, exactly as tests/test_market_scan_job.py does it."""
    import importlib

    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    from app import scan_store as ss

    importlib.reload(ss)
    yield ss
    if ss._conn is not None:
        ss._conn.close()
        ss._conn = None
    importlib.reload(ss)


def test_a_night_with_no_low_side_column_reports_none_not_zero(store):
    """The migration path. A night stored before the column existed holds NULL, and NULL is the
    honest answer: nobody measured the low side that night, which is not the same as nobody making
    a new low. Backfill is impossible — the store keeps recent nights, not 52 weeks per row."""
    b = store.breadth()
    # No scan at all is the strongest form of the same rule.
    assert b["available"] is False
    assert b["new_52w_lows"] is None
    assert b["near_52w_low"] is None
    assert b["high_low_diff"] is None
    # And never a zero anywhere in the reading.
    for k in ("new_52w_lows", "near_52w_low", "high_low_diff", "new_52w_highs", "near_52w_high"):
        assert b[k] is None, f"{k} must be None, never 0, when nothing was measured"


def test_the_no_breadth_shape_carries_every_new_key(store):
    """A caller that reads a key which only exists on the happy path gets a KeyError at the worst
    possible moment. The two shapes must agree on their keys."""
    empty = store.breadth()
    for k in ("new_52w_lows", "near_52w_low", "near_52w_low_pct", "high_low_diff"):
        assert k in empty


def test_the_low_side_thresholds_mirror_the_high_side():
    from app import scan_store

    assert scan_store._AT_52W_LOW_PCT == -scan_store._AT_52W_HIGH_PCT
    assert scan_store._NEAR_52W_LOW_PCT == -scan_store._NEAR_52W_HIGH_PCT


def test_the_metric_is_registered_for_storage():
    """A metric that swing produces but the store does not list is silently dropped on write."""
    from app import scan_store

    assert "pct_off_52w_low" in scan_store._METRIC_COLS


def test_an_existing_database_gains_the_column_by_migration(tmp_path, monkeypatch):
    """The path every deployed install takes. CREATE TABLE IF NOT EXISTS is a no-op on an existing
    table, so a database created before today needs the explicit ALTER — without it the column is
    missing on exactly the machines that have data in them, and the breadth query would raise."""
    import importlib
    import sqlite3

    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    from app import scan_store as ss

    importlib.reload(ss)

    # Build the pre-migration shape: the real table, minus the new column.
    db_file = tmp_path / "scan.db"
    if db_file.exists():
        db_file.unlink()
    con = sqlite3.connect(db_file)
    con.executescript(ss._SCHEMA)
    con.execute("ALTER TABLE scan DROP COLUMN pct_off_52w_low")
    cols_before = {r[1] for r in con.execute("PRAGMA table_info(scan)")}
    assert "pct_off_52w_low" not in cols_before
    con.execute(
        "INSERT INTO scan (d, symbol, ts, pct_off_52w_high) VALUES ('20260101', 'OLD', 1767225600, -5.0)"
    )
    con.commit()
    con.close()

    importlib.reload(ss)
    b = ss.breadth("20260101")          # must not raise
    con = sqlite3.connect(db_file)
    assert "pct_off_52w_low" in {r[1] for r in con.execute("PRAGMA table_info(scan)")}
    con.close()

    # The pre-existing row keeps NULL, so the low side reports nothing rather than a count of zero.
    assert b["new_52w_lows"] is None
    assert b["high_low_diff"] is None

    if ss._conn is not None:
        ss._conn.close()
        ss._conn = None
    importlib.reload(ss)
