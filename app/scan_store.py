"""Persistence for the nightly market-wide cross-section — one row per (night, symbol).

The swing scanner walks ~3,200 names a night and measures the same ~25 numbers on each. Those
numbers are only worth keeping if they can be compared ACROSS the market (percentiles, breadth) and
ACROSS time (is this name's relative strength improving?), which is a query, not a JSON blob. This
module is that store, and nothing else: it holds rows and answers questions about them. It computes
no metrics — `app/swing.py` owns that contract.

Why a NEW FILE (data/scan.db) and not another table inside memory.db
--------------------------------------------------------------------
This is a lifecycle decision, not a performance one. The write is ~96ms either way; that number
justifies nothing and is not the argument.

1. **Retention is irreconcilable, and it is a published contract.** `memory.prune()` applies ONE
   cutoff to every table it owns, and `memory.retention_days()` returns ONE number that
   POST /memory/backfill validates user input against with a 422. Verdicts want 730 days because a
   verdict cannot be re-derived; a market cross-section wants ~90 because it is bulk and cheap. Two
   retention policies cannot live behind one number without one of them becoming a lie.

2. **Opposite replaceability.** `verdicts` holds two years of scored calls that CANNOT be
   regenerated — the forward returns are measurable only because the verdict was recorded at the
   time it was made. A scan cross-section is fully reproducible by re-running tonight's scan. One
   bad migration should not be able to endanger the irreplaceable half on behalf of the disposable
   half.

3. **File high-water mark.** Nothing in this service ever VACUUMs a database. A table with 90-day
   retention churns its entire contents quarterly and parks its host file at the high-water mark,
   ~100MB+. Inside scan.db that is self-limiting and expected. Inside memory.db it means
   `memory.stats()["db_bytes"]` — the number the ops dashboard puts in front of the user — jumps
   roughly 50x for reasons that have nothing to do with memory.

4. **One lock, one connection.** memory.py serialises ALL access on a single module-level
   threading.Lock, so WAL buys it nothing for in-process concurrency. Two files means two locks:
   the nightly scan's bulk insert and a verdict write never queue behind each other.

THE SINGLE-WRITER INVARIANT — read this before adding a write
-------------------------------------------------------------
`app/market_scan_job.py`, running as its own systemd oneshot, is the SOLE WRITER of scan.db. The API
process is a READ-ONLY consumer of it. That is the entire basis for there being no cross-process
lock here: SQLite's own file locking plus WAL is enough for many readers and one writer, and this
module's threading.Lock only serialises threads inside whichever process is holding it.

If you are about to add a second writer — an endpoint that backfills a night, a percentile pass that
runs in the API process — that invariant is gone, and `timeout=5.0` becomes the only thing standing
between you and an intermittent "database is locked" in a request handler. Either keep the write in
the nightly job, or change this docstring and think about it properly first.

THE SECOND WRITER (record_gate) — added deliberately, and here is the reasoning
------------------------------------------------------------------------------
`record_gate()` breaks the invariant above: `app/gate.py` evaluates in the API process, off a
request or a sandbox tick, and files one row per day. So scan.db now has TWO writing processes.

What that actually costs. SQLite locks the FILE, not the table — `gate` and `scan` living in
separate tables buys exactly nothing here, and anyone reasoning "different tables, no contention"
is wrong. In WAL mode readers never block, but the two writers serialise on the single write lock,
and the only thing between them is `timeout=5.0` (SQLite's busy handler, which retries for five
seconds before raising "database is locked").

Why it is acceptable anyway, in the order the numbers matter:
  1. The shapes are wildly asymmetric. The gate write is ONE upserted row of ~400 bytes, sub-
     millisecond. The nightly job's write holds the lock for the ~100ms of a 3,200-row commit. A
     100ms hold against a 5,000ms budget is a 50x margin, and the gate write is not retried into a
     queue — it either lands immediately or waits out a commit that is already ending.
  2. The windows barely overlap. The scan is a systemd oneshot at a fixed nightly hour; the gate is
     consulted during the session and by the sandbox tick. Collisions are possible, not routine.
  3. The failure is contained and honest. `record_gate` swallows into a logged False, so a lock
     timeout costs one row of gate HISTORY — never the evaluation itself, which has already been
     returned to the caller, and never a scan row, since the nightly writer is the one holding the
     lock it would be waiting on.
What is NOT acceptable, and must not be added on the back of this precedent: a bulk write from the
API process. A percentile pass or a backfill endpoint writing thousands of rows would hold the lock
for exactly as long as the nightly job does, in the middle of the trading day, and the argument
above would no longer hold — it rests entirely on this writer being one small row.

THE THIRD WRITER (write_percentiles) — the paragraph above named it, so here is the answer
-------------------------------------------------------------------------------------------
SWT-4's percentile pass (app/percentiles.py) runs in TWO places, and only one of them is new.

  * At the tail of the nightly job, in the SAME process that just wrote the night. That is not a
    second writer at all — it is the sole writer doing one more thing before it exits, and it is
    where the pass belongs.
  * From POST /market_scan/percentiles, to rank a night that is already stored. THIS is the bulk
    write from the API process the paragraph above forbids, and it is admitted under one condition:
    `write_percentiles` commits in `_PCT_CHUNK`-row transactions rather than one. The objection was
    never "thousands of rows", it was "one lock held for the length of thousands of rows"; a chunked
    writer holds it for ~200 UPDATEs at a time and lets go in between, so the worst any concurrent
    writer waits is one chunk. It is an explicit POST, it is idempotent, and it writes only columns
    no other writer owns. A bulk INSERT from the API process is still forbidden, for the reason
    stated above and unchanged.

Threading model
---------------
Every public function here is SYNCHRONOUS and blocking. Callers in async code wrap them:
`await asyncio.to_thread(scan_store.breadth)`, the pattern main.py already uses for
usage_store/selfupdate/observability. This module deliberately does not repeat what
`GET /memory/stats` does today — `memory.stats()` is called straight from an `async def`, so its
SQLite work runs ON the event loop of a single-worker uvicorn and blocks every other request for its
duration. That is survivable for a few-thousand-row aggregate over memory.db; over a 288k-row
cross-section it would not be.

Closed 2026-08-31: `breadth()["new_52w_lows"]` used to be permanently None, because the metrics
contract measured distance from the 52-week HIGH and had no low-side equivalent. `pct_off_52w_low`
now joins the contract, so the low side is measured, and `high_low_diff` reports the difference
between the two BAND counts — the strict counts compare a closing extreme against intraday extremes
and read ~0 on almost every night, so their difference would be noise, not divergence.

Nights stored before that date hold NULL in the new column and report None here. They cannot be
backfilled: this store keeps recent nights, not 52 weeks of history per row.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from collections.abc import Iterable
from typing import Any

log = logging.getLogger("signals.scan_store")

_DATA_DIR = Path(os.environ.get("SIGNALS_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
_FILE = _DATA_DIR / "scan.db"


def _env_int(name: str, default: int) -> int:
    """An int from the environment, falling back to `default` rather than failing at import.

    A typo in a unit file must not take the whole service down at import time; a wrong retention
    window is a far smaller problem than a process that will not start.
    """
    try:
        return max(1, int(os.environ.get(name, "") or default))
    except (TypeError, ValueError):
        log.warning("scan_store: ignoring unparseable %s", name)
        return default


# 90 days. The cross-section is bulk, fully reproducible, and read for "where does this name sit
# relative to the market TODAY" — a question a quarter of history answers as well as a decade would.
# It is deliberately NOT memory.py's 730: see the retention argument in the module docstring.
_RETAIN_DAYS = _env_int("SIGNALS_SCAN_RETAIN_DAYS", 90)

# Bumped when a one-time repair is added below. Stamped into PRAGMA user_version so the repair runs
# once per database instead of once per process start.
_USER_VERSION = 1

# A count or a rate computed over a fraction of the universe is not the market-wide number its name
# claims to be: "42 new highs" over 300 measurable rows out of 3,200 is not breadth, it is a sample
# presented as a census. Below this share of the night's rows we report None instead. Half is a
# judgement call, not a derived threshold — it is set where a reader would still call the answer
# "the market" rather than "some of it".
_MIN_COVERAGE = 0.5

# `pct_off_52w_high` is (close / max(52w highs) - 1) * 100, so a close that takes out every prior
# print of the year lands at exactly 0.0 — but it is rounded on the way in, so an `== 0` test would
# drop rows that stored as -0.0001. This is a FLOAT ROUNDING tolerance and nothing more; it is not a
# "near the highs" band, and widening it would silently turn the count into a different statistic.
#
# Note what this therefore counts: CLOSING 52-week highs measured against intraday highs — a name
# that printed a new high and closed below it is not counted. That is a conservative definition and
# it will read lower than the exchange-published new-high counts, which use intraday prints.
_AT_52W_HIGH_PCT = -0.01

# The companion band, and the reason it exists. Because the strict count above compares a CLOSE
# against an INTRADAY high, it can only fire when a name closes on the exact high tick of its year,
# and it therefore reads 0 on most nights: measured against the live universe on 2026-08-21, 0 names
# qualified while the closest sat 0.02% away and 87 were within 1%. A field named `new_52w_highs`
# that prints 0 every night is read as "nothing is leading" — which is false, and is the same
# absent-rendered-as-a-confident-number failure this codebase keeps having to correct.
#
# Reporting BOTH is the honest resolution: the strict count keeps its exact meaning, and this band
# answers the question a reader is actually asking. It is named for what it measures — proximity,
# not an event — so the two can never be confused for one another.
_NEAR_52W_HIGH_PCT = -1.0

# The low-side mirrors, with the SAME reasoning inverted. `pct_off_52w_low` is
# (close / min(52w lows) - 1) * 100, so a close that undercuts every prior print of the year lands at
# exactly 0.0; the tolerance is float rounding and nothing else.
#
# And the strict count has the SAME defect the high side documents: a CLOSING low measured against
# INTRADAY lows can only fire when a name closes on the exact low tick of its year, so it reads ~0 on
# most nights. Measured against the live universe, the strict high-side count read 0/5/6/1/3 per
# night out of ~3,103. A high-low DIFFERENTIAL built on two numbers that are both nearly always zero
# is not a divergence signal, it is noise — so the differential below is built on the BAND form,
# which is the only one of the pair that moves.
_AT_52W_LOW_PCT = 0.01
_NEAR_52W_LOW_PCT = 1.0

# Two adjusted closes from two different nightly fetches are not always comparable: Yahoo re-adjusts
# a whole series retroactively, so a split between last night and tonight shows up here as a -75%
# "decline" for a 4:1 and a +900% "advance" for a 1:10 reverse. Pairs that moved more than this in
# one session are excluded from BOTH counts — they are unknown, not down. The cost is a handful of
# genuine 50%+ movers (biotech readouts, squeezes) going uncounted; the alternative is every split
# in the universe being reported as a crash, every night, forever.
_MAX_SESSION_MOVE = 0.5

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


# --------------------------------------------------------------------------------------- columns

# One column per swing.metrics() numeric/bool key. Flat and typed rather than a JSON blob because
# every question asked of this table — percentile of adx14, count above sma50, top 50 by relative
# strength — is a scan or an aggregate over ONE of these, and json_extract() over 288k rows is not.
_METRIC_REAL: tuple[str, ...] = (
    "price",
    "sma20", "sma50", "sma150", "sma200",
    "ema20", "ema50",
    "adr20_pct", "atr14", "atr14_pct", "adx14",
    "clv", "rel_volume", "dollar_volume_20d",
    "mom_20d", "mom_60d", "rsi14",
    "pct_off_52w_high", "pct_off_52w_low", "pct_vs_sma50", "pct_vs_sma200",
    "rel_strength_3mo", "ema20_slope_pct",
)
_METRIC_INT: tuple[str, ...] = ("bars",)
# SQLite has no BOOL: stored 1/0/NULL and restored to True/False/None on the way out. NULL is a
# third state and means "not measurable" — never False. A name with fewer than 200 bars has no
# opinion about its 200-day average; saying `above_sma200 = False` about it is a bearish claim we
# have no basis for.
_METRIC_BOOL: tuple[str, ...] = ("above_sma50", "above_sma200", "ma_stacked")

_METRIC_COLS: tuple[str, ...] = _METRIC_REAL + _METRIC_INT + _METRIC_BOOL
_BOOL_SET = frozenset(_METRIC_BOOL)
_INT_SET = frozenset(_METRIC_INT)
# Every column insert_night() knows how to write. Anything else in the table — a percentile added by
# a later pass, a column from a future migration — is untouched by a write, which is the whole point
# of the ON CONFLICT below.
_WRITABLE: tuple[str, ...] = ("d", "symbol", "ts") + _METRIC_COLS + ("unmeasured",)

# ------------------------------------------------------------------ derived: percentile ranks

# SWT-4. These are written by a LATER PASS (app/percentiles.py) over a night that is already stored,
# and they are deliberately NOT in `_METRIC_COLS` or `_WRITABLE`: the nightly job measures, the pass
# ranks, and neither may overwrite the other's columns. That separation is what
# `test_reinserting_a_night_preserves_a_column_written_after_the_original_insert` has been pinning
# down since before these columns existed.
#
# The suffix is `_pctile`, not `_pct`: `_pct` already means "this number is expressed in percent" on
# a third of the metric columns, and `adr20_pct_pct` names nothing at all.
PCT_SUFFIX = "_pctile"

# The ten metrics worth a cross-sectional rank. Everything else in `_METRIC_COLS` is left unranked
# ON PURPOSE, in three groups:
#
#   * PER-SHARE DOLLAR LEVELS — price, sma20/50/150/200, ema20/50, atr14. A percentile over these
#     ranks share PRICE, which says nothing about a company: a $480 name is not "in the 99th
#     percentile" of anything a reader cares about, it just has few shares outstanding.
#   * THE BOOLEANS and `bars` — above_sma50, above_sma200, ma_stacked have two states, not a
#     distribution, and `bars` measures how much history OUR FETCH returned, which is a fact about
#     the pipeline rather than about the market.
#   * THE NEAR-DUPLICATES — atr14_pct (adr20_pct already carries the volatility axis), clv,
#     pct_vs_sma50, pct_vs_sma200 (pct_off_52w_high already carries the extension axis). These are
#     rankable and simply are not read on any screen today; each is one entry in this tuple plus one
#     column in two places below, whenever one of them is.
PCT_METRICS: tuple[str, ...] = (
    "rel_strength_3mo",
    "rel_volume",
    "adr20_pct",
    "adx14",
    "mom_20d",
    "mom_60d",
    "rsi14",
    "pct_off_52w_high",
    "ema20_slope_pct",
    "dollar_volume_20d",
)

# metric -> the column its rank is stored in. Published so a caller can build the percentile half of
# a response without hardcoding the suffix, and so a metric that is not ranked is visibly absent
# from the mapping rather than silently returning nothing.
PCT_COLUMNS: dict[str, str] = {m: m + PCT_SUFFIX for m in PCT_METRICS}
_PCT_COLS: tuple[str, ...] = tuple(PCT_COLUMNS.values())

# Rows per transaction in write_percentiles(). See its docstring: this number is the entire reason a
# backfill from the API process is defensible where a bulk insert would not be.
_PCT_CHUNK = 200


# `sort="rel_strength"` is the ergonomic name callers use; the column is `rel_strength_3mo` because
# the contract pins the horizon into the metric's name. Aliased in one place rather than letting
# every caller learn the difference.
_SORT_ALIASES = {"rel_strength": "rel_strength_3mo"}

# The public vocabulary of `sort` and of `**filters`, published so a caller can validate BEFORE it
# calls. An HTTP handler owes a 422 that NAMES the bad parameter, and it cannot get one from `top()`:
# top's blanket except turns the ValueError for an unknown sort/filter into an empty list, which on a
# screen is indistinguishable from "the scan ran and nothing matched". Derived from the column tuples
# above rather than written out again, so a new metric joins both without anyone remembering to.
SORT_COLUMNS: frozenset[str] = frozenset(_METRIC_COLS) | frozenset(_SORT_ALIASES)
BOOL_FILTERS: frozenset[str] = frozenset(_METRIC_BOOL)
FILTER_NAMES: frozenset[str] = (
    frozenset(f"min_{c}" for c in _METRIC_COLS)
    | frozenset(f"max_{c}" for c in _METRIC_COLS)
    | BOOL_FILTERS
)


# ---------------------------------------------------------------------------------------- schema

# WITHOUT ROWID, and the PRIMARY KEY is the access pattern. Measured ~275 B/row against ~305 for a
# rowid table plus a unique index on (d, symbol): one b-tree instead of two, no integer key stored
# alongside a key we already have, and rows physically CLUSTERED by night — which is exactly how
# breadth() and the percentile pass read them (whole night, in one contiguous sweep).
#
# Sizing this file: ~3,200 rows/night x 90 nights of retention ~= 288k rows ~= 80-90 MB, plus the
# secondary index. That is the steady state to expect on CT 237's 12 GB disk, and it is why
# retention lives here and not in memory.db.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS scan (
    d       TEXT NOT NULL,          -- YYYYMMDD of the last bar, matching Series.dates
    symbol  TEXT NOT NULL,
    ts      REAL NOT NULL,          -- epoch seconds when the row was produced

    price               REAL,
    sma20               REAL,
    sma50               REAL,
    sma150              REAL,
    sma200              REAL,
    ema20               REAL,
    ema50               REAL,
    adr20_pct           REAL,
    atr14               REAL,
    atr14_pct           REAL,
    adx14               REAL,
    clv                 REAL,
    rel_volume          REAL,
    dollar_volume_20d   REAL,
    mom_20d             REAL,
    mom_60d             REAL,
    rsi14               REAL,
    pct_off_52w_high    REAL,
    pct_off_52w_low     REAL,
    pct_vs_sma50        REAL,
    pct_vs_sma200       REAL,
    rel_strength_3mo    REAL,
    ema20_slope_pct     REAL,
    bars                INTEGER,
    above_sma50         INTEGER,    -- 1 / 0 / NULL, where NULL is "not measurable"
    above_sma200        INTEGER,
    ma_stacked          INTEGER,

    unmeasured TEXT,                -- JSON array of the metric names that came back None

    -- SWT-4 percentile ranks, written by app/percentiles.py AFTER the night is stored, never by
    -- insert_night. NULL means "not ranked" — a metric we could not measure, or a metric too little
    -- of the night could be measured on. It NEVER means the 0th percentile, which would be a
    -- confident claim that this name is the worst in the market. Also add any new one to _migrate().
    rel_strength_3mo_pctile   REAL,
    rel_volume_pctile         REAL,
    adr20_pct_pctile          REAL,
    adx14_pctile              REAL,
    mom_20d_pctile            REAL,
    mom_60d_pctile            REAL,
    rsi14_pctile              REAL,
    pct_off_52w_high_pctile   REAL,
    ema20_slope_pct_pctile    REAL,
    dollar_volume_20d_pctile  REAL,

    PRIMARY KEY (d, symbol)
) WITHOUT ROWID;

-- "how has THIS name's cross-sectional standing moved" — symbol_history()'s only query, and the
-- one access pattern the (d, symbol) clustering is wrong for.
CREATE INDEX IF NOT EXISTS ix_scan_symbol ON scan(symbol, d);

-- One row per day: what app/gate.py said about the market that day. It lives HERE, next to the
-- cross-section, because one of the gate's five legs IS the cross-section (breadth) — a gate row and
-- the night it was computed against belong to the same retention story and the same file.
--
-- `passed` is deliberately NULLABLE and three-valued. NULL is not "unknown row", it is a RECORDED
-- STATE: the gate ran, and at least one leg could not be measured, so it refused to claim the market
-- was clear either way. Storing that as 0 would rewrite "we could not tell" into "the market failed
-- the gate" the moment it hit disk, which is precisely the distinction gate.py exists to preserve.
CREATE TABLE IF NOT EXISTS gate (
    d            TEXT NOT NULL,     -- YYYYMMDD the gate describes
    ts           REAL NOT NULL,     -- epoch seconds of the evaluation
    passed       INTEGER,           -- 1 / 0 / NULL, where NULL means "not decidable", never False
    market_score REAL,              -- 0-100, NULL when any leg was unmeasurable
    failing      TEXT,              -- JSON array of leg names that were explicitly False
    unmeasured   TEXT,              -- JSON array of leg names whose ok was NULL
    legs         TEXT,              -- JSON blob of the full five-leg list, as evaluated

    PRIMARY KEY (d)
) WITHOUT ROWID;
"""


def _db() -> sqlite3.Connection:
    """Open (once) and return the connection. WAL so a long read never blocks the nightly write."""
    global _conn
    if _conn is None:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        # Built LOCALLY and published only once initialisation fully succeeds. This ordering is
        # copied verbatim from memory.py, where assigning the global first was a real production
        # bug: a failure part-way through left a live but un-migrated connection in place, and since
        # `_conn is None` was then false, every later call skipped schema and migration silently,
        # forever. The failure surfaced once, at startup, and then looked healthy.
        # check_same_thread=False + the module lock: callers reach this from asyncio.to_thread, so
        # the connection is touched from many threads and serialising ourselves is simpler to reason
        # about than one connection per thread.
        conn = sqlite3.connect(_FILE, check_same_thread=False, timeout=5.0)
        try:
            conn.row_factory = sqlite3.Row
            # DEVIATION 3 from memory.py. auto_vacuum can only be set on a database that has no
            # tables yet — changing it later requires a full VACUUM, which needs free disk equal to
            # the file and is exactly the maintenance nobody performs on this box. So it goes FIRST,
            # before _SCHEMA creates anything. With 90-day retention this table deletes its entire
            # contents four times a year; without it the file would sit forever at its high-water
            # mark. prune() calls `PRAGMA incremental_vacuum` to actually hand the pages back.
            conn.execute("PRAGMA auto_vacuum = INCREMENTAL")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(_SCHEMA)
            _migrate(conn)
            conn.commit()
        except Exception:
            conn.close()
            raise
        try:
            os.chmod(_FILE, 0o600)
        except OSError:
            pass
        _conn = conn
    return _conn


def _migrate(db: sqlite3.Connection) -> None:
    """Add columns introduced after a database was first created, and run one-time repairs once.

    `CREATE TABLE IF NOT EXISTS` is a no-op on an existing table, so a new column needs an explicit
    ALTER, guarded by PRAGMA table_info because a fresh install already has it. A new column must be
    added in BOTH places — `_SCHEMA` for new databases and the list here for existing ones — or the
    two diverge and the difference only shows up on whichever machine has the older file.

    DEVIATION 2 from memory.py: nothing in here may be an UNCONDITIONAL full-table statement.
    memory._migrate runs a dedupe DELETE across `verdicts` on every process start — ~459ms at 414k
    rows, forever, and selfupdate restarts the service on a button press. A repair is a one-time
    event and belongs behind the user_version gate below, where it costs one integer read per start
    instead of a full scan.
    """
    have = {r["name"] for r in db.execute("PRAGMA table_info(scan)")}
    for col, decl in tuple((c, "REAL") for c in _PCT_COLS) + (("pct_off_52w_low", "REAL"),):
        # also add it to _SCHEMA above
        if col not in have:
            # Nights stored before this column existed keep NULL in it, and that is the correct
            # value: the low side genuinely was not measured on those nights. It cannot be
            # backfilled — the store keeps recent nights, not 52 weeks of history per row — so
            # anything reading it must render "-" for them and must never substitute 0, which would
            # claim no name made a new low on a night nobody looked.
            db.execute(f"ALTER TABLE scan ADD COLUMN {col} {decl}")

    # Same treatment for the gate table, which arrived after scan did: a database created before it
    # existed gets the table from CREATE TABLE IF NOT EXISTS above, but a COLUMN added to it later
    # needs its own ALTER here or the two definitions diverge by machine age.
    have_gate = {r["name"] for r in db.execute("PRAGMA table_info(gate)")}
    for col, decl in ():           # ("new_col", "REAL"), ... — also add it to _SCHEMA above
        if col not in have_gate:
            db.execute(f"ALTER TABLE gate ADD COLUMN {col} {decl}")

    ver = int(db.execute("PRAGMA user_version").fetchone()[0] or 0)
    if ver < _USER_VERSION:
        # One-time data repairs go HERE, inside the gate — never as a bare statement above it.
        # There is none to run at version 1; the gate is stamped from day one so that the first
        # repair anyone writes has an obvious place to go that is not "on every start".
        db.execute(f"PRAGMA user_version = {_USER_VERSION}")


# ---------------------------------------------------------------------------------------- writes

def insert_night(rows: list[dict], *, d: str) -> int:
    """Store one night's cross-section. Returns the number of rows written.

    `d` is the night, taken from the caller rather than from each row, because a cross-section is
    only comparable if every row in it was measured off the same bar date. A row carrying its own
    `d` does not get to disagree with the batch it arrived in.

    Columns absent from a row dict are LEFT ALONE rather than nulled. swing.metrics() always returns
    every key (None when it could not be measured), so a real nightly re-run still clears a metric
    that stopped being measurable — absence here means "this writer had no opinion about that
    column", which is what a partial writer (a percentile pass) means and what a scan never says.
    """
    try:
        night = _norm_d(d)
        now = time.time()
        # Grouped by which columns the row actually carries: rows with the same shape share one
        # statement, and a partial writer never drags the full column list along with it.
        groups: dict[tuple[str, ...], list[list[Any]]] = {}
        skipped = 0
        for r in rows or ():
            sym = str(r.get("symbol") or "").strip().upper()
            if not sym:
                skipped += 1
                continue
            rec: dict[str, Any] = {"d": night, "symbol": sym, "ts": _num(r.get("ts")) or now}
            for col in _METRIC_COLS:
                if col in r:
                    rec[col] = _coerce(col, r[col])
            if "unmeasured" in r:
                rec["unmeasured"] = _dump_unmeasured(r["unmeasured"])
            groups.setdefault(tuple(rec), []).append(list(rec.values()))

        if skipped:
            # Counted and named rather than silently dropped: a night that is quietly 200 rows short
            # produces breadth that looks fine and is wrong.
            log.warning("scan_store: %s row(s) for %s had no symbol and were not stored", skipped, night)

        written = 0
        with _lock:
            db = _db()
            for cols, values in groups.items():
                # DEVIATION 1 from memory.py, which uses INSERT OR REPLACE. REPLACE is a DELETE
                # followed by an INSERT, so every column the statement does not name reverts to its
                # default — a percentile computed and written after the original insert would be
                # silently wiped by a re-run of the same night, and nothing would report that it
                # had happened. The explicit DO UPDATE SET touches only what this writer brought.
                sets = ",".join(f"{c}=excluded.{c}" for c in cols if c not in ("d", "symbol"))
                sql = (
                    f"INSERT INTO scan ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)}) "
                    f"ON CONFLICT(d, symbol) DO UPDATE SET {sets}"
                )
                db.executemany(sql, values)
                written += len(values)
            db.commit()
        return written
    except Exception:  # noqa: BLE001 — a failed nightly write must not take down the scan job that
        # produced the rows; the night is reproducible by re-running it, and 0 says plainly that
        # nothing landed.
        log.warning("scan_store: insert_night failed for %s", d, exc_info=True)
        return 0


def retire_absent(d: str, symbols: Iterable[str]) -> int:
    """Delete rows for night `d` whose symbol is NOT in `symbols`. Returns the number removed.

    `insert_night` deliberately upserts, which means a re-run never removes a row it has stopped
    producing. That is correct for a partial writer and wrong for a full re-scan: when the integrity
    gate was added on 2026-08-21 it began rejecting twelve mixed-split-basis names, and a forced
    re-run left all twelve of the previous run's corrupt rows in place — they still sorted to the top
    of every momentum ranking, because "stopped being measured" had no way to mean "remove".

    ONLY a caller that measured the WHOLE universe may use this. A partial run (`--limit 400`)
    calling it would delete the other 2,700 rows and silently replace a night's cross-section with a
    sample of itself — worse than the problem it fixes, and invisible afterwards.

    An empty `symbols` is refused rather than read as "retire everything", because the one caller
    that could pass it by accident is a run that failed to measure anything at all.
    """
    try:
        keep = {str(s).strip().upper() for s in (symbols or ()) if str(s).strip()}
        if not keep:
            log.warning("scan_store: refusing to retire rows for %s against an empty symbol set", d)
            return 0
        night = _norm_d(d)
        with _lock:
            db = _db()
            have = [r["symbol"] for r in db.execute("SELECT symbol FROM scan WHERE d = ?", (night,))]
            gone = [s for s in have if s not in keep]
            # Chunked so the parameter list cannot exceed SQLite's variable limit on a big universe.
            for i in range(0, len(gone), 400):
                batch = gone[i:i + 400]
                db.execute(
                    f"DELETE FROM scan WHERE d = ? AND symbol IN ({','.join('?' for _ in batch)})",
                    (night, *batch),
                )
            db.commit()
        if gone:
            log.warning("scan_store: retired %s row(s) from %s no longer produced", len(gone), night)
        return len(gone)
    except Exception:  # noqa: BLE001 — housekeeping must never sink the run that calls it; a night
        # carrying a few stale rows is recoverable, a job that dies before writing its summary is not.
        log.warning("scan_store: retire_absent failed for %s", d, exc_info=True)
        return 0


def write_percentiles(d: str, values: dict[str, dict[str, Any]]) -> int:
    """Write the SWT-4 percentile columns onto rows of night `d` THAT ALREADY EXIST. Rows updated.

    `values` is {symbol: {percentile_column: rank_or_None}}. app/percentiles.py computes it; this
    function only stores it, exactly as this module stores metrics it does not compute.

    UPDATE, not the ON CONFLICT upsert `insert_night` uses, and the difference is load-bearing. An
    upsert for a symbol that is not in this night would INSERT a row carrying nothing but ranks — an
    all-null phantom name that lands in the cross-section, counts in breadth's coverage denominator,
    and is indistinguishable afterwards from a stock we tried and failed to measure. An UPDATE that
    matches no row is a silent no-op, which is the correct answer to "rank a symbol this night does
    not contain".

    EVERY percentile column is named on EVERY row, so a metric the caller passes nothing for is
    CLEARED. That is deliberate. When a metric that ranked cleanly last night falls under its
    coverage floor tonight, NULL is the honest state; leaving yesterday's rank sitting in the row
    would show a stale market position with tonight's date stamped on it, which is the exact defect
    class the rest of this file is built to avoid.

    CHUNKED, and this is the paragraph to read before reusing the pattern. The module docstring
    above rules out a bulk write from the API process because it would hold the single write lock
    for the ~100ms of a 3,200-row commit in the middle of the trading day. This writer holds it for
    `_PCT_CHUNK` rows at a time and releases between chunks, so a concurrent `record_gate` waits out
    one small chunk instead of a whole pass. The price is that a backfill is NOT atomic: an
    interrupted one leaves the remainder of the night NULL, which reads as "not ranked yet" and is
    repaired by running it again. An absent rank is recoverable; a wrong one is not.
    """
    try:
        night = _norm_d(d)
        # Columns come from _PCT_COLS, never from the caller's keys: the names are interpolated into
        # SQL, and a whitelist is the only version of that which is safe by construction.
        sql = (f"UPDATE scan SET {','.join(f'{c}=?' for c in _PCT_COLS)}"
               f" WHERE d = ? AND symbol = ?")
        params: list[list[Any]] = []
        for sym, cols in (values or {}).items():
            s = str(sym or "").strip().upper()
            if not s:
                continue
            got = cols or {}
            params.append([_num(got.get(c)) for c in _PCT_COLS] + [night, s])

        updated = 0
        for i in range(0, len(params), _PCT_CHUNK):
            with _lock:
                db = _db()
                cur = db.executemany(sql, params[i:i + _PCT_CHUNK])
                db.commit()
                # Read INSIDE the lock: rowcount belongs to this cursor, and a count that is only
                # correct because no other thread happened to be writing is not a count.
                updated += max(0, int(cur.rowcount or 0))
        return updated
    except Exception:  # noqa: BLE001 — a failed rank pass must not sink the nightly job whose rows
        # are already committed, nor 500 a backfill request. The measurements are intact either way
        # and the ranks are re-derivable from them by running the pass again; 0 says nothing landed.
        log.warning("scan_store: write_percentiles failed for %s", d, exc_info=True)
        return 0


# ---------------------------------------------------------------------------- gate history (rw)

# Kept together rather than split across the writes/reads sections above: all three share the
# second-writer caveat in the module docstring, and the round-trip of `passed` — 1/0/NULL out and
# True/False/None back — is only checkable if the two halves sit next to each other.

def record_gate(row: dict, *, d: str) -> bool:
    """File one day's gate evaluation. Returns whether it landed. Never raises.

    `row` is `gate.evaluate()`'s dict, taken whole, so the recorded row cannot drift from the shape
    the consumer was handed. `d` comes from the caller for the same reason `insert_night` takes it:
    the day a reading is filed under is the caller's claim, not something a payload gets to restate.

    `passed` round-trips through THREE states. None is stored as SQL NULL and read back as None, and
    the coercion below is written out longhand precisely because `int(bool(None))` and
    `1 if x else 0` both quietly turn "we could not tell" into "the gate failed" — a bearish claim
    about a day on which the gate specifically declined to make one.

    Upsert, not REPLACE, and last-write-wins within a day: the row is what the gate most recently
    reported. `evaluate()` refuses to call this when it measured nothing at all, so a transient
    outage cannot overwrite a real morning reading with five nulls.
    """
    try:
        night = _norm_d(d)
        passed = row.get("passed")
        # Explicit three-way. Anything that is not exactly True or False is stored as NULL rather
        # than coerced — a truthy string or a 1 arriving here is a caller bug, and NULL ("we do not
        # know") is the only safe way to record a bug in a field that means "is the market clear".
        stored = 1 if passed is True else (0 if passed is False else None)
        vals = (
            night,
            _num(row.get("evaluated_at")) or time.time(),
            stored,
            _num(row.get("market_score")),
            _dump_names(row.get("failing")),
            _dump_names(row.get("unmeasured")),
            json.dumps(row.get("legs") or [], default=str),
        )
        with _lock:
            db = _db()
            db.execute(
                "INSERT INTO gate (d, ts, passed, market_score, failing, unmeasured, legs)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(d) DO UPDATE SET ts=excluded.ts, passed=excluded.passed,"
                "   market_score=excluded.market_score, failing=excluded.failing,"
                "   unmeasured=excluded.unmeasured, legs=excluded.legs",
                vals,
            )
            db.commit()
        return True
    except Exception:  # noqa: BLE001 — history is a record of a decision that has ALREADY been
        # returned to the caller. A locked database (see the second-writer note in the module
        # docstring) or an unserialisable leg must cost one history row, never the evaluation.
        log.warning("scan_store: record_gate failed for %s", d, exc_info=True)
        return False


def gate_history(limit: int = 30) -> list[dict]:
    """The recent gate rows, NEWEST FIRST. [] on any failure.

    Newest first for the same reason symbol_history() is: `limit` has to mean "the most recent N",
    and an ascending order would silently hand back the oldest N instead.

    Note these rows are NOT pruned with the cross-section. `prune()` cuts `scan` on a 90-day window
    because a night is bulk and fully reproducible; a gate row is neither. It is a record of what we
    decided and why, ~400 bytes a day, and its entire value is being readable long after the night
    it was computed against is gone — the same argument that keeps verdicts for two years.
    """
    try:
        with _lock:
            rows = _db().execute(
                "SELECT * FROM gate ORDER BY d DESC LIMIT ?", (max(1, min(int(limit), 1000)),)
            ).fetchall()
        return [_gate_rowdict(r) for r in rows]
    except Exception:  # noqa: BLE001 — a history panel that cannot load renders as "no history";
        # raising would take the gate card it sits on down with it.
        log.warning("scan_store: gate_history failed", exc_info=True)
        return []


def gate_for(d: str) -> dict | None:
    """One day's gate row, or None if the gate was never recorded for that day.

    None means "no row", which is a different statement from a row whose `passed` is None ("the gate
    ran and could not decide"). A caller must be able to tell those apart, so this never manufactures
    an empty row to return.
    """
    try:
        with _lock:
            row = _db().execute("SELECT * FROM gate WHERE d = ?", (_norm_d(d),)).fetchone()
        return _gate_rowdict(row) if row is not None else None
    except Exception:  # noqa: BLE001 — an unreadable row is indistinguishable from an absent one to
        # every caller of this, and None is what both of them already handle.
        log.warning("scan_store: gate_for failed for %s", d, exc_info=True)
        return None


def _gate_rowdict(r: sqlite3.Row) -> dict:
    """A gate row as callers expect it: `passed` back to True/False/None, JSON columns parsed.

    `passed` is the whole reason this is not `_rowdict`: `bool(0)` and `bool(None)` are both False in
    Python, so restoring this column with the generic helper would collapse the third state on the
    way OUT after the write took such care to keep it on the way in.
    """
    passed = r["passed"]
    return {
        "d": r["d"],
        "ts": r["ts"],
        "passed": None if passed is None else bool(passed),
        "market_score": r["market_score"],
        # [] and None are different here too, exactly as in _load_unmeasured: [] means the gate
        # checked and nothing was failing, None means no writer ever said.
        "failing": _load_unmeasured(r["failing"]),
        "unmeasured": _load_unmeasured(r["unmeasured"]),
        "legs": _load_legs(r["legs"]),
    }


def _dump_names(v: Any) -> str | None:
    """A list of leg names as JSON, IN THE ORDER GIVEN. None stays None.

    Deliberately not `_dump_unmeasured`, which sorts: these lists are ordered by the gate's leg
    order, and a `failing` list that comes back alphabetised no longer matches the leg list stored
    beside it in the same row.
    """
    if v is None:
        return None
    if not isinstance(v, (list, tuple)):
        return None
    return json.dumps([str(x) for x in v])


def _load_legs(v: Any) -> list[dict] | None:
    """The stored leg blob back as a list of dicts, or None if it is missing or unparseable."""
    if v is None:
        return None
    try:
        out = json.loads(v)
    except (TypeError, ValueError):
        return None
    return [x for x in out if isinstance(x, dict)] if isinstance(out, list) else None


# ----------------------------------------------------------------------------------------- reads

def breadth(d: str | None = None) -> dict:
    """Market-wide participation for one night. Never raises.

    `available` is the first thing to read: when it is False every reading below it is None, and the
    caller must render "no scan", not a number.
    """
    try:
        with _lock:
            db = _db()
            night = _resolve_d(db, d)
            if night is None:
                return _no_breadth()
            row = db.execute(
                "SELECT COUNT(*) n, MAX(ts) ts,"
                "  SUM(above_sma50 IS NOT NULL) m50, SUM(above_sma50) a50,"
                "  SUM(above_sma200 IS NOT NULL) m200, SUM(above_sma200) a200,"
                "  SUM(pct_off_52w_high IS NOT NULL) mhh,"
                "  SUM(pct_off_52w_high >= ?) nh,"
                "  SUM(pct_off_52w_high >= ?) nearh,"
                "  SUM(pct_off_52w_low IS NOT NULL) mll,"
                "  SUM(pct_off_52w_low <= ?) nl,"
                "  SUM(pct_off_52w_low <= ?) nearl"
                " FROM scan WHERE d = ?",
                (_AT_52W_HIGH_PCT, _NEAR_52W_HIGH_PCT, _AT_52W_LOW_PCT, _NEAR_52W_LOW_PCT, night),
            ).fetchone()
            n = int(row["n"] or 0)
            if not n:
                return _no_breadth()
            adv, dcl = _advance_decline(db, night, n)
            latest_ts = float(row["ts"]) if row["ts"] is not None else None
            # Bound once: _count applies the coverage floor, so calling it twice per side inside the
            # dict below would compute the same guard four times and read as if the two sides were
            # independently derived.
            _near_h = _count(row["nearh"], row["mhh"], n)
            _near_l = _count(row["nearl"], row["mll"], n)

        return {
            "available": True,
            "as_of": night,
            "n": n,
            "pct_above_sma50": _rate_pct(row["a50"], row["m50"], n),
            "pct_above_sma200": _rate_pct(row["a200"], row["m200"], n),
            "advancers": adv,
            "decliners": dcl,
            "new_52w_highs": _count(row["nh"], row["mhh"], n),
            # Proximity, not an event — see _NEAR_52W_HIGH_PCT. Read alongside new_52w_highs, never
            # instead of it: one counts names that closed at a new yearly high, this counts names
            # trading within 1% of one.
            "near_52w_high": _near_h,
            "near_52w_high_pct": abs(_NEAR_52W_HIGH_PCT),
            # The low side, live since 2026-08-31. Nights stored before then hold NULL in
            # `pct_off_52w_low` and therefore report None here — not 0, which would claim no name
            # made a new low on a night nobody measured one. There is no backfill: the store keeps
            # recent nights, not a year of history per row.
            "new_52w_lows": _count(row["nl"], row["mll"], n),
            "near_52w_low": _near_l,
            "near_52w_low_pct": abs(_NEAR_52W_LOW_PCT),
            # The differential, and the reason it is built on the BAND counts rather than the strict
            # ones: both strict counts read ~0 on almost every night by construction, so their
            # difference is noise dressed as a divergence signal. This is the classic internal
            # divergence — an index making ground while more names sit near their lows than near
            # their highs — and it is the reading the gate's participation leg cannot see.
            # None whenever either side is unmeasured, so a one-sided night reports nothing at all.
            "high_low_diff": (
                (_near_h - _near_l) if (_near_h is not None and _near_l is not None) else None
            ),
            "age_hours": round((time.time() - latest_ts) / 3600.0, 2) if latest_ts else None,
        }
    except Exception:  # noqa: BLE001 — breadth is decoration on every screen that shows it; an
        # unreadable database must degrade to "we do not know", never to a market reading.
        log.warning("scan_store: breadth failed", exc_info=True)
        return _no_breadth()


def _no_breadth() -> dict:
    """The shape returned when there is no scan to report on.

    Every percentage and count is None, NOT 0. This is the defect class this codebase keeps
    catching: `pct_above_sma50: 0.0` is not "we did not scan", it is "zero percent of the market is
    above its 50-day average" — the single most bearish breadth reading that exists, rendered from
    an empty table. A caller that forgets to check `available` gets a dash on the screen instead of
    a crash warning it never earned.
    """
    return {
        "available": False,
        "as_of": None,
        "n": 0,                       # a genuine count of the rows we hold, which really is zero
        "pct_above_sma50": None,
        "pct_above_sma200": None,
        "advancers": None,
        "decliners": None,
        "new_52w_highs": None,
        "near_52w_high": None,
        # The threshold is a property of this module, not of the scan, so it stays populated even
        # when there is nothing to count — it lets a caller label an empty state ("— within 1%")
        # without having to hardcode the band its own copy of.
        "near_52w_high_pct": abs(_NEAR_52W_HIGH_PCT),
        "new_52w_lows": None,
        "near_52w_low": None,
        "near_52w_low_pct": abs(_NEAR_52W_LOW_PCT),
        "high_low_diff": None,
        "age_hours": None,
    }


def _advance_decline(db: sqlite3.Connection, night: str, n: int) -> tuple[int | None, int | None]:
    """Advancers/decliners against the most recent EARLIER night we hold.

    The store is the only thing that knows both sides, so this is computed here rather than asked of
    the metrics contract (which has no one-session return). "The previous night we stored" is not
    always the previous session — a missed run makes the comparison span several days — so treat
    this as a change since the last observation, which is what it is.

    Returns (None, None) rather than (0, 0) when there is no earlier night: on the first night ever
    scanned, zero advancers and zero decliners is not a flat tape, it is no data.
    """
    prev = db.execute("SELECT MAX(d) p FROM scan WHERE d < ?", (night,)).fetchone()["p"]
    if not prev:
        return None, None
    row = db.execute(
        "SELECT SUM(CASE WHEN c.price > p.price THEN 1 END) adv,"
        "       SUM(CASE WHEN c.price < p.price THEN 1 END) dcl,"
        "       COUNT(*) pairs"
        "  FROM scan c JOIN scan p ON p.symbol = c.symbol AND p.d = ?"
        " WHERE c.d = ? AND c.price IS NOT NULL AND p.price > 0"
        "   AND ABS(c.price / p.price - 1.0) < ?",
        (prev, night, _MAX_SESSION_MOVE),
    ).fetchone()
    pairs = int(row["pairs"] or 0)
    if not pairs or pairs < n * _MIN_COVERAGE:
        return None, None
    return int(row["adv"] or 0), int(row["dcl"] or 0)


def _rate_pct(hits: Any, measured: Any, n: int) -> float | None:
    """A percentage over the rows that could be measured, or None if too few of them could be."""
    m = int(measured or 0)
    if not m or m < n * _MIN_COVERAGE:
        return None
    return round(float(hits or 0) / m * 100.0, 1)


def _count(hits: Any, measured: Any, n: int) -> int | None:
    """A count, or None when it would be a count over a minority of the universe."""
    m = int(measured or 0)
    if not m or m < n * _MIN_COVERAGE:
        return None
    return int(hits or 0)


def cross_section(d: str | None = None) -> list[dict]:
    """Every row for one night, symbol-ordered. [] when that night was never scanned.

    This is the whole night — ~3,200 rows — because the percentile pass that consumes it needs the
    full distribution. Callers wrap it in asyncio.to_thread; it is not a request-path read.
    """
    try:
        with _lock:
            db = _db()
            night = _resolve_d(db, d)
            if night is None:
                return []
            rows = db.execute("SELECT * FROM scan WHERE d = ? ORDER BY symbol", (night,)).fetchall()
        return [_rowdict(r) for r in rows]
    except Exception:  # noqa: BLE001 — a missing cross-section is "not scanned yet", which every
        # caller already has to handle; raising would turn it into a 500 on a dashboard.
        log.warning("scan_store: cross_section failed", exc_info=True)
        return []


def top(d: str | None = None, *, sort: str = "rel_strength", limit: int = 50, **filters) -> list[dict]:
    """The night's leaders on one metric, optionally filtered. [] on any failure.

    `sort` is a metric name, descending, NULLs last — prefix it with "-" for ascending (`"-atr14_pct"`
    for the calmest names rather than the wildest). Ties break on symbol so repeated calls agree.

    `filters` are `min_<metric>` / `max_<metric>` bounds, plus a bare boolean metric name for an
    exact match (`above_sma200=True`). An unrecognised filter name is a caller bug, and it is
    refused rather than ignored: a filter that silently does nothing returns a LONGER list than
    asked for and presents it as filtered, which is the failure mode worth being loud about. The
    blanket except below turns that into [] — wrong-and-empty, never wrong-and-plausible.
    """
    try:
        col, asc = _sort_column(sort)
        where, params = _filter_sql(filters)
        with _lock:
            db = _db()
            night = _resolve_d(db, d)
            if night is None:
                return []
            sql = (
                f"SELECT * FROM scan WHERE d = ?{where}"
                f" ORDER BY {col} IS NULL, {col} {'ASC' if asc else 'DESC'}, symbol"
                f" LIMIT ?"
            )
            rows = db.execute(sql, (night, *params, max(1, min(int(limit), 1000)))).fetchall()
        return [_rowdict(r) for r in rows]
    except Exception:  # noqa: BLE001 — see the docstring: failing to an empty list is the safe
        # direction for a screen, because an over-broad list looks like a real answer.
        log.warning("scan_store: top failed (sort=%s filters=%s)", sort, sorted(filters), exc_info=True)
        return []


def count(d: str | None = None, **filters) -> int | None:
    """How many of one night's rows match `filters`. None when we could not look.

    `top()` answers "the best N"; anything that SHOWS those N also has to say what they are N of, and
    `len(top(...))` can only ever report back the cap it was handed — "50 names are trending" when
    812 were. Same filter vocabulary as `top()`, same refusal of an unknown filter name.

    Returns None, never 0, on a failed query or a night that was never scanned. "Nothing matched your
    filter" and "we could not count" are different claims about the market, and 0 is only the first.
    """
    try:
        where, params = _filter_sql(filters)
        with _lock:
            db = _db()
            night = _resolve_d(db, d)
            if night is None:
                return None
            row = db.execute(
                f"SELECT COUNT(*) n FROM scan WHERE d = ?{where}", (night, *params)
            ).fetchone()
        return int(row["n"] or 0)
    except Exception:  # noqa: BLE001 — matching top()'s contract: a broken filter must not 500 a
        # dashboard, and None (unlike 0) cannot be misread as a real, empty cross-section.
        log.warning("scan_store: count failed (filters=%s)", sorted(filters), exc_info=True)
        return None


def symbol_row(symbol: str, d: str | None = None) -> dict | None:
    """One name's row for one night, or None if it was not in that night's scan."""
    try:
        sym = str(symbol or "").strip().upper()
        if not sym:
            return None
        with _lock:
            db = _db()
            if d is None:
                # The latest night THIS SYMBOL has, not the latest night overall: a name that was
                # skipped tonight (halted, delisted, fetch failed) should report its last real
                # observation rather than nothing, and `d` in the returned row says which night it
                # actually is.
                row = db.execute(
                    "SELECT * FROM scan WHERE symbol = ? ORDER BY d DESC LIMIT 1", (sym,)
                ).fetchone()
            else:
                row = db.execute(
                    "SELECT * FROM scan WHERE symbol = ? AND d = ?", (sym, _norm_d(d))
                ).fetchone()
        return _rowdict(row) if row is not None else None
    except Exception:  # noqa: BLE001 — one missing row must not break a symbol page.
        log.warning("scan_store: symbol_row failed for %s", symbol, exc_info=True)
        return None


def symbol_history(symbol: str, *, limit: int = 90) -> list[dict]:
    """One name across the nights we hold, NEWEST FIRST. [] on any failure.

    Newest first because `limit` has to mean "the most recent N" — an ascending order would make a
    limit silently return the OLDEST N, which is the opposite of every question asked of this.
    """
    try:
        sym = str(symbol or "").strip().upper()
        if not sym:
            return []
        with _lock:
            rows = _db().execute(
                "SELECT * FROM scan WHERE symbol = ? ORDER BY d DESC LIMIT ?",
                (sym, max(1, min(int(limit), 1000))),
            ).fetchall()
        return [_rowdict(r) for r in rows]
    except Exception:  # noqa: BLE001 — one name's history is a panel, not the page; an empty
        # list renders as "no history" while an exception would take the whole symbol view with it.
        log.warning("scan_store: symbol_history failed for %s", symbol, exc_info=True)
        return []


def latest_date() -> str | None:
    """The most recent night stored, or None if nothing has been scanned yet."""
    try:
        with _lock:
            return _db().execute("SELECT MAX(d) d FROM scan").fetchone()["d"]
    except Exception:  # noqa: BLE001 — "we have no scan" is exactly what an unreadable file means
        # to every caller of this, so None is the honest answer rather than an exception.
        log.warning("scan_store: latest_date failed", exc_info=True)
        return None


# ----------------------------------------------------------------------------------- housekeeping

def retention_days() -> int:
    """How many days of cross-sections are kept. Distinct from memory.retention_days() by design."""
    return _RETAIN_DAYS


def prune(retain_days: int | None = None) -> int:
    """Drop nights past the retention window and hand the freed pages back. Returns rows removed.

    Cut on `d` — the bar date — and not on `ts`, the moment the row was written. A night backfilled
    today is old data that arrived late; retaining it for 90 days from its INSERT would keep a
    six-month-old cross-section alive and let it into a percentile that is supposed to be current.
    """
    try:
        days = _RETAIN_DAYS if retain_days is None else max(1, int(retain_days))
        # UTC. Bar dates are exchange dates and the two disagree by hours; at a 90-day boundary that
        # is not a difference worth carrying a timezone dependency for.
        cutoff = time.strftime("%Y%m%d", time.gmtime(time.time() - days * 86_400))
        with _lock:
            db = _db()
            n = db.execute("DELETE FROM scan WHERE d < ?", (cutoff,)).rowcount
            db.commit()
            if n:
                # The other half of auto_vacuum=INCREMENTAL (see _db). Without this call the pages
                # are on the freelist and the file never shrinks; with it, a quarterly full turnover
                # of the table does not ratchet the file size upward forever. It is a no-op on a
                # database created before auto_vacuum was set, which is the right failure.
                db.execute("PRAGMA incremental_vacuum")
                db.commit()
        return n
    except Exception:  # noqa: BLE001 — housekeeping that fails must not take the nightly job with
        # it; the rows stay, the file grows, and the next run tries again.
        log.warning("scan_store: prune failed", exc_info=True)
        return 0


def stats() -> dict:
    """Shape of what the store holds — for the ops dashboard, so it is inspectable, not a black box."""
    try:
        with _lock:
            db = _db()
            row = db.execute(
                "SELECT COUNT(*) rows, COUNT(DISTINCT d) nights, COUNT(DISTINCT symbol) symbols,"
                "       MIN(d) oldest, MAX(d) latest FROM scan"
            ).fetchone()
        return {
            "rows": row["rows"] or 0,
            "nights": row["nights"] or 0,
            "symbols": row["symbols"] or 0,
            "oldest": row["oldest"],
            "latest": row["latest"],
            "retain_days": _RETAIN_DAYS,
            "db_bytes": _db_bytes(),
        }
    except Exception:  # noqa: BLE001 — /api/status must render even when this file is unreadable;
        # a named error is more useful there than a 500 that hides every other panel.
        log.warning("scan_store: stats failed", exc_info=True)
        return {"error": "unavailable"}


# --------------------------------------------------------------------------------------- helpers

def _db_bytes() -> int:
    """On-disk footprint of the store — the main file PLUS the write-ahead log.

    The -wal file holds committed pages until a checkpoint folds them back, and after a 3,200-row
    night it is not a rounding error. Reporting only scan.db would understate the footprint on the
    ops dashboard at exactly the moment someone is looking at it because the disk is filling up.
    """
    total = 0
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(_FILE) + suffix)
        if p.exists():
            total += p.stat().st_size
    return total


def _resolve_d(db: sqlite3.Connection, d: str | None) -> str | None:
    """The night to read, or None when there is nothing to read.

    A caller that asks for no particular night gets the most recent one; a caller that names a night
    gets that one, existence unchecked here — the query that follows returns nothing and every
    caller already treats that as "not scanned".
    """
    if d is not None:
        return _norm_d(d)
    return db.execute("SELECT MAX(d) d FROM scan").fetchone()["d"]


def _norm_d(d: Any) -> str:
    """`d` as the bare YYYYMMDD that Series.dates uses. Raises on anything that is not a date.

    Tolerates "2026-08-21" and datetime.date because both turn up at call sites, but refuses to
    guess: a night key that is silently wrong writes a whole cross-section under a date nothing will
    ever look for.
    """
    digits = "".join(c for c in str(d or "") if c.isdigit())
    if len(digits) < 8:
        raise ValueError(f"scan_store: not a YYYYMMDD date: {d!r}")
    return digits[:8]


def _sort_column(sort: str) -> tuple[str, bool]:
    """(column, ascending) for an ORDER BY, validated against the known columns.

    Whitelisted rather than escaped: the column name is interpolated into SQL, and a whitelist is
    the only version of that which is safe by construction rather than by review.
    """
    s = str(sort or "").strip()
    asc = s.startswith("-")
    s = s.lstrip("-")
    col = _SORT_ALIASES.get(s, s)
    if col not in _METRIC_COLS:
        raise ValueError(f"scan_store: unknown sort column {sort!r}")
    return col, asc


def _filter_sql(filters: dict) -> tuple[str, list[Any]]:
    """Turn min_/max_/bare-bool kwargs into a WHERE fragment. Raises on an unknown name."""
    parts: list[str] = []
    params: list[Any] = []
    for key, val in sorted(filters.items()):
        if val is None:
            continue                       # an unset bound is not a bound
        if key.startswith("min_") and key[4:] in _METRIC_COLS:
            parts.append(f" AND {key[4:]} >= ?")
            params.append(_coerce(key[4:], val))
        elif key.startswith("max_") and key[4:] in _METRIC_COLS:
            parts.append(f" AND {key[4:]} <= ?")
            params.append(_coerce(key[4:], val))
        elif key in _BOOL_SET:
            parts.append(f" AND {key} = ?")
            params.append(1 if val else 0)
        else:
            raise ValueError(f"scan_store: unknown filter {key!r}")
    return "".join(parts), params


def _coerce(col: str, v: Any) -> Any:
    """A Python value as the column stores it: bools to 1/0, ints to int, everything else to float."""
    if col in _BOOL_SET:
        return None if v is None else (1 if v else 0)
    if col in _INT_SET:
        return _int(v)
    return _num(v)


def _rowdict(r: sqlite3.Row) -> dict:
    """A DB row as the dict callers expect: bools restored, `unmeasured` parsed back to a list."""
    out: dict[str, Any] = {}
    for k in r.keys():
        v = r[k]
        if k in _BOOL_SET:
            out[k] = None if v is None else bool(v)
        elif k == "unmeasured":
            out[k] = _load_unmeasured(v)
        else:
            out[k] = v
    return out


def _dump_unmeasured(v: Any) -> str | None:
    """The unmeasured-key list as JSON. None stays None — see _load_unmeasured for why it matters."""
    if v is None:
        return None
    try:
        names = [str(x) for x in v] if isinstance(v, (list, tuple, set)) else [str(v)]
    except TypeError:
        return None
    return json.dumps(sorted(names))


def _load_unmeasured(v: Any) -> list[str] | None:
    """The stored JSON array back as a list.

    NULL comes back as None, NOT []. They are different claims: [] means the producer checked and
    everything was measurable, None means no producer ever said. Collapsing the second into the
    first would report a row of unknown provenance as fully measured.
    """
    if v is None:
        return None
    try:
        out = json.loads(v)
        return [str(x) for x in out] if isinstance(out, list) else None
    except (TypeError, ValueError):
        return None


def _num(v: Any) -> float | None:
    """A finite float, or None. NaN and +-inf are NOT numbers this store will keep: they survive
    SQLite, break every aggregate they touch, and print as "NaN" in a UI that promised a price."""
    try:
        if v is None or isinstance(v, bool):
            return None
        f = float(v)
        return f if f == f and f not in (float("inf"), float("-inf")) else None
    except (TypeError, ValueError):
        return None


def _int(v: Any) -> int | None:
    f = _num(v)
    return int(f) if f is not None else None
