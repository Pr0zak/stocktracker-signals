"""Long-term memory for the analyst: what it said, and what actually happened next.

Every verdict this service produces is fire-and-forget today — the model re-derives its view of a
name from scratch each time and never learns that the last four "buy" calls on it went nowhere. This
module records each verdict alongside the numeric setup that produced it, scores the *realized*
forward return once enough bars exist, and hands the calibrated result back on the next call.

Why SQLite + a numeric k-NN and not a vector database
-----------------------------------------------------
The obvious reach is embeddings, and it is the wrong tool here on two counts:

1. **Hardware.** CT 237 has 2 cores and 2 GB of RAM, already hosting uvicorn. `sentence-transformers`
   pulls ~2 GB of torch onto a 12 GB disk and holds hundreds of MB resident. Measured on this host,
   the reasoning workload is already out of reach locally (an 8B model needs ~22 min for one sandbox
   tick vs ~20 s today); adding a second model process to serve retrieval buys nothing.

2. **The query isn't semantic.** "Find past setups like this one" is a question about *numbers* —
   RSI, distance from the 50-day, relative strength, how far off the 52-week high. A weighted
   nearest-neighbour over ~8 normalized dimensions answers that exactly, in microseconds, with no
   dependencies and no embedding drift. Cosine similarity over prose *about* those numbers is a
   lossy proxy for a distance we can just compute directly.

Prose memory (strategy notes, blocked-trade reasons, research findings) lives in a plain `notes`
table. It briefly carried an FTS5 index; that was removed once it became clear nothing needed fuzzy
text recall — the only question actually asked of it is "which rules blocked trades, and how often",
which is a GROUP BY. One file, zero dependencies, no index without a reader.

Statistical honesty
-------------------
A hit rate over 3 samples is noise. `similar_setups` refuses to emit a track record below
`_MIN_SAMPLES` and always reports `n`, so the model can discount it. This mirrors how the
seasonality and short-interest features already flag weak samples rather than laundering them into
confident-sounding numbers.

Nothing in here may break a verdict: every public entry point is wrapped by the caller in a
try/except, and writes are best-effort by design.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

log = logging.getLogger("signals.memory")

_DATA_DIR = Path(os.environ.get("SIGNALS_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
_FILE = _DATA_DIR / "memory.db"

# Below this many comparable, already-scored setups we report nothing rather than a number that
# reads as evidence but is noise.
_MIN_SAMPLES = 5

# Forward-return horizons, in BARS of the symbol's own series (sessions for an equity, calendar days
# for crypto). 5 and 20 are the original pair. 63 (a quarter) and 252 (one trading year — also the
# short/long-term capital-gains boundary the sandbox is told to respect) were added 2026-09-04: the
# sandbox's objective is measured in years and nothing here looked past a month, so the strategist
# was being handed a 20-day scorecard in the same prompt that told it churn destroys terminal value.
# A row is graded at 5+20 together first (`scored_at` marks that), and each long horizon fills in on
# its own as the bars arrive. Readers derive their per-horizon keys from these tuples.
HORIZONS: tuple[int, ...] = (5, 20, 63, 252)
LONG_HORIZONS: tuple[int, ...] = (63, 252)
# Calendar days a row must age before a horizon is worth fetching a series for. A floor, not a
# promise — an equity needs ~365 calendar days for 252 sessions once holidays are counted — so a row
# flagged early is simply not scorable yet and is revisited the next night at no cost but a fetch.
_HORIZON_MIN_AGE_DAYS: dict[int, int] = {20: 5, 63: 85, 252: 355}

# Rows older than this are pruned. It was two years until 2026-09-04, which would have started
# deleting the oldest backfill rows on 2026-09-24 — the ONLY rows old enough to carry a one-year
# mark, three weeks before the first of those marks could be written. A row is fully graded only
# after a year and is most valuable after that, so it is kept for five: ~300 bytes each, so even
# 100k rows is 30 MB on a 12 GB disk.
_RETAIN_DAYS = 1826
# Prose notes have no horizon to wait for; two years remains plenty.
_NOTES_RETAIN_DAYS = 730

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


# --------------------------------------------------------------------------------------- schema

_SCHEMA = """
CREATE TABLE IF NOT EXISTS verdicts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol       TEXT    NOT NULL,
    ts           REAL    NOT NULL,          -- epoch seconds when the verdict was produced
    asof_date    TEXT    NOT NULL,          -- YYYYMMDD of the last bar the verdict saw,
                                            -- matching Series.dates so scoring can index straight in
    signal       TEXT,                      -- normalised lower-case value: buy / hold / sell / ...
    conviction   INTEGER,
    deep         INTEGER NOT NULL DEFAULT 0,
    model        TEXT,
    -- 'model' = a real verdict this service produced. 'backfill' = a setup replayed from price
    -- history with NO verdict attached, so the pattern's base rate is measurable on day one instead
    -- of 20 sessions from now. Backfilled rows must NEVER count toward the model's own calibration.
    origin       TEXT NOT NULL DEFAULT 'model',
    price        REAL,
    rule_score   INTEGER,
    thesis       TEXT,
    -- the numeric setup, stored flat so k-NN is a plain scan
    rsi14            REAL,
    pct_vs_sma20     REAL,
    pct_vs_sma50     REAL,
    macd_hist_pct    REAL,                  -- macd_hist / price * 100, so it compares across names
    bollinger_pct_b  REAL,
    stochastic_k     REAL,
    pct_off_52w_high REAL,
    rel_strength     REAL,
    -- filled in later by score_symbol()
    fwd_5d       REAL,
    fwd_20d      REAL,
    -- the benchmark's return over the SAME window, so the track record can report excess rather
    -- than market drift dressed up as skill
    bench_fwd_5d  REAL,
    bench_fwd_20d REAL,
    -- set once fwd_5d/fwd_20d are written; the long horizons below fill in on their own later
    scored_at    REAL,
    fwd_63d        REAL,
    fwd_252d       REAL,
    bench_fwd_63d  REAL,
    bench_fwd_252d REAL
);
CREATE INDEX IF NOT EXISTS ix_verdicts_symbol ON verdicts(symbol, ts DESC);
CREATE INDEX IF NOT EXISTS ix_verdicts_unscored ON verdicts(scored_at) WHERE scored_at IS NULL;

CREATE TABLE IF NOT EXISTS notes (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       REAL NOT NULL,
    kind     TEXT NOT NULL,                 -- strategy | blocked | research | trade
    symbol   TEXT,
    body     TEXT NOT NULL,
    meta     TEXT                           -- optional JSON blob
);
CREATE INDEX IF NOT EXISTS ix_notes_kind ON notes(kind, ts DESC);
"""

# Full-text search used to live here (an FTS5 virtual table plus insert/delete triggers). It was
# removed rather than left in place: nothing ever needed fuzzy text recall. The one question worth
# asking of this data — "how often did a rule block a trade, and which rule?" — is a GROUP BY, and
# "find setups like this one" is answered far better by the numeric k-NN above than by matching prose
# about the numbers. An index with no reader is just a maintenance surface and a lie about intent.
_DROP_FTS = """
DROP TRIGGER IF EXISTS notes_ai;
DROP TRIGGER IF EXISTS notes_ad;
DROP TABLE IF EXISTS notes_fts;
"""


def _db() -> sqlite3.Connection:
    """Open (once) and return the connection. WAL so a long read never blocks the tick's write."""
    global _conn
    if _conn is None:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        # Built LOCALLY and published only once initialisation fully succeeds. Assigning the global
        # first meant a failure part-way through left a live but un-migrated connection in place —
        # and since `_conn is None` was then false, every later call skipped schema and migration
        # silently, forever. The failure surfaced once, at startup, and then looked healthy.
        # check_same_thread=False + the module-level lock: uvicorn runs handlers on a threadpool, and
        # serialising ourselves is simpler to reason about than one connection per thread.
        conn = sqlite3.connect(_FILE, check_same_thread=False, timeout=5.0)
        try:
            conn.row_factory = sqlite3.Row
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
    """Add columns introduced after a database was first created.

    `CREATE TABLE IF NOT EXISTS` is a no-op on an existing table, so new columns need an explicit
    ALTER. Each is guarded individually — a fresh install already has them and would otherwise raise.
    """
    db.executescript(_DROP_FTS)  # retire the unused full-text index (see note above)
    # Collapse pre-existing duplicate bars (keep the newest row) so the unique index can be built.
    db.execute(
        "DELETE FROM verdicts WHERE id NOT IN ("
        "  SELECT MAX(id) FROM verdicts GROUP BY symbol, asof_date, origin)"
    )
    # Created HERE, after the dedupe — never in _SCHEMA, which runs first and would fail against a
    # database that still has duplicates (exactly what happened on the first deploy of this change).
    # One row per bar per origin: without it a `refresh=true` verdict, a manual /scan/run alongside
    # the timer, or a repeated backfill each inserted another row for the identical bar, so `n`
    # counted verdict PRODUCTIONS rather than decisions and repeated bars skewed every median.
    db.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_verdicts_bar "
        "ON verdicts(symbol, asof_date, origin)"
    )
    # Repair rows written before signals were normalised: "Signal.buy" -> "buy". Left alone they are
    # invisible to every buy filter, so the scorecard would under-count real calls indefinitely.
    db.execute(
        "UPDATE verdicts SET signal = LOWER(SUBSTR(signal, INSTR(signal, '.') + 1)) "
        "WHERE signal LIKE 'Signal.%'"
    )
    have = {r["name"] for r in db.execute("PRAGMA table_info(verdicts)")}
    for col, decl in (
        ("bench_fwd_5d", "REAL"),
        ("bench_fwd_20d", "REAL"),
        ("origin", "TEXT NOT NULL DEFAULT 'model'"),
        # 2026-09-04: the long horizons. NULL on every existing row; score_symbol fills them in
        # as each row ages past the mark, so a database from before this change needs no re-seed.
        *((f"fwd_{h}d", "REAL") for h in LONG_HORIZONS),
        *((f"bench_fwd_{h}d", "REAL") for h in LONG_HORIZONS),
    ):
        if col not in have:
            db.execute(f"ALTER TABLE verdicts ADD COLUMN {col} {decl}")


# ------------------------------------------------------------------------------------- features

# Each dimension is divided by a fixed, domain-chosen scale so the distance is comparable across
# names and stable as the table grows. Deriving scales from the data instead would make yesterday's
# neighbours silently change meaning as new rows land.
# `veto` marks the axes that DEFINE what a setup is: two rows that disagree sharply on any of them
# are not neighbours, however well the rest line up. It is listed explicitly rather than derived from
# `weight`, because deriving it silently exempted rel_strength (weight 0.8 < the 1.0 threshold) — the
# single dimension the module docstring names as the point of the whole exercise. That is verbatim
# the failure the veto was added to prevent, reintroduced by coupling two unrelated ideas.
_FEATURES: tuple[tuple[str, float, float, bool], ...] = (
    # (column,            scale,  weight, veto)
    ("rsi14",             25.0,   1.0,    True),
    ("pct_vs_sma20",       6.0,   1.0,    True),
    ("pct_vs_sma50",      12.0,   1.0,    True),
    ("macd_hist_pct",      1.5,   0.7,    False),   # a confirmer; allowed to disagree
    ("bollinger_pct_b",    0.4,   0.7,    False),
    # BASIS CHANGE 2026-08-30: stochastic_k switched from a close-basis window to Pine's ta.stoch
    # over bar highs/lows. Rows written before that date hold the old basis, which saturated at 0 or
    # 100 on roughly 30% of bars, so a historical neighbour can look far closer or further on this
    # axis than it really was. It is weight 0.5 and non-veto, and _distance skips dimensions a row
    # lacks, so the damage is bounded — but the mixed population is real until those rows are either
    # re-derived or nulled.
    ("stochastic_k",      30.0,   0.5,    False),
    ("pct_off_52w_high",  15.0,   1.0,    True),
    ("rel_strength",       0.12,  0.8,    True),    # weaker weight, but still setup-defining
)
# A neighbour further than this (mean weighted units) isn't the same setup in any useful sense.
_MAX_DISTANCE = 1.15
# ...but a mean alone is too forgiving: dimensions that happen to agree drag it down and let an
# OPPOSITE setup through. Measured: RSI 29 vs RSI 79 — oversold vs overbought, about as different as
# two setups get — still scored 0.94 because MACD/Bollinger/RS matched. So any single core dimension
# this far apart vetoes the match outright, no matter how well the rest line up.
_VETO_DISTANCE = 1.5
# (which dimensions veto is declared per-feature above, not inferred from weight)


def features_from_summary(summary: dict) -> dict[str, float | None]:
    """Project a `_snapshot` dict onto the stored feature columns."""
    price = _num(summary.get("price"))
    hist = _num(summary.get("macd_hist"))
    # MACD histogram is in price units, so a $500 name dwarfs a $20 one. Percent-of-price makes the
    # dimension mean the same thing everywhere.
    macd_pct = (hist / price * 100.0) if (hist is not None and price) else None
    return {
        "rsi14": _num(summary.get("rsi14")),
        "pct_vs_sma20": _num(summary.get("pct_vs_sma20")),
        "pct_vs_sma50": _num(summary.get("pct_vs_sma50")),
        "macd_hist_pct": macd_pct,
        "bollinger_pct_b": _num(summary.get("bollinger_pct_b")),
        "stochastic_k": _num(summary.get("stochastic_k")),
        "pct_off_52w_high": _num(summary.get("pct_off_52w_high")),
        "rel_strength": _num(summary.get("rel_strength_3mo_vs_benchmark")),
    }


def _distance(a: dict[str, float | None], row: sqlite3.Row) -> float | None:
    """Weighted mean absolute distance over the dimensions both sides actually have.

    Returns None when too little overlaps to be a fair comparison — a row matching on two of eight
    dimensions is not a near neighbour, it's an artefact of missing data.
    """
    total = 0.0
    wsum = 0.0
    used = 0
    for col, scale, weight, veto in _FEATURES:
        x, y = a.get(col), row[col]
        if x is None or y is None:
            continue
        d = abs(float(x) - float(y)) / scale
        if veto and d > _VETO_DISTANCE:
            return None  # opposite on an axis that defines the setup — not a neighbour at all
        total += weight * d
        wsum += weight
        used += 1
    if used < 4 or wsum <= 0:
        return None
    return total / wsum


# --------------------------------------------------------------------------------------- writes

def record_verdict(
    *,
    symbol: str,
    summary: dict,
    verdict: dict,
    model: str | None = None,
    deep: bool = False,
    origin: str = "model",
    ts: float | None = None,
) -> int | None:
    """Persist one verdict + the setup that produced it. Returns the row id, or None on failure.

    `origin="backfill"` records a setup replayed from price history (no real verdict); `ts` lets the
    backfill stamp the historical bar's own time rather than now, so retention and ordering behave.
    """
    try:
        feats = features_from_summary(summary)
        asof = _asof_date(summary)
        row = {
            "symbol": (symbol or "").upper(),
            "ts": ts if ts is not None else time.time(),
            "asof_date": asof,
            "signal": _signal_str(verdict.get("signal")),
            "conviction": _int(verdict.get("conviction")),
            "deep": 1 if deep else 0,
            "model": model,
            "origin": origin,
            "price": _num(summary.get("price")),
            "rule_score": _int(summary.get("rule_score")),
            "thesis": (verdict.get("thesis") or verdict.get("summary") or "")[:1200] or None,
            **feats,
        }
        cols = ",".join(row)
        marks = ",".join("?" for _ in row)
        with _lock:
            db = _db()
            # REPLACE, not IGNORE: a re-run reflects a newer read of the same bar and should win.
            cur = db.execute(
                f"INSERT OR REPLACE INTO verdicts ({cols}) VALUES ({marks})", list(row.values()))
            db.commit()
            return cur.lastrowid
    except Exception:  # noqa: BLE001 — memory must never break a verdict
        log.warning("memory: record_verdict failed", exc_info=True)
        return None


def add_note(kind: str, body: str, *, symbol: str | None = None, meta: dict | None = None) -> None:
    """Store a prose memory (weekly strategy, a blocked trade and why, a research finding)."""
    if not (body or "").strip():
        return
    try:
        with _lock:
            db = _db()
            db.execute(
                "INSERT INTO notes (ts, kind, symbol, body, meta) VALUES (?,?,?,?,?)",
                (time.time(), kind, (symbol or None) and symbol.upper(),
                 body.strip()[:4000], json.dumps(meta) if meta else None),
            )
            db.commit()
    except Exception:  # noqa: BLE001
        log.warning("memory: add_note failed", exc_info=True)


# --------------------------------------------------------------------------------------- recall

def similar_setups(symbol: str, summary: dict, *, k: int = 40) -> dict | None:
    """What happened the last time this setup appeared?

    Returns two independently-reported cohorts, because they answer different questions and
    averaging them would hide the distinction:

      - `this_symbol`: prior verdicts on *this* name (does the model read this ticker well?)
      - `analogues`:   the nearest setups across every other name (does this *pattern* work?)

    Either is omitted when it has fewer than `_MIN_SAMPLES` scored rows. Inside each, the 20-day
    numbers are as they always were; `vs_benchmark_63d` and `vs_benchmark_252d` appear once at least
    `_MIN_SAMPLES` neighbours carry that mark, each with its own `n` and `n_symbols`.
    """
    try:
        feats = features_from_summary(summary)
        if sum(1 for v in feats.values() if v is not None) < 4:
            return None
        sym = (symbol or "").upper()
        with _lock:
            db = _db()
            # Only scored rows can teach anything. The veto axes are applied IN SQL, not just in
            # _distance: until 2026-09-04 the scan was capped at the 4,000 newest rows, which was
            # fine while every horizon was 20 days but excludes precisely the rows old enough to
            # carry a one-year mark. Pre-filtering on the same veto the Python check applies keeps
            # the candidate set small however large the table grows (measured on the live 9k-row
            # database: 75 ms for the capped scan, single-digit ms with the prefilter), so the cap
            # can be a safety valve rather than a silent bias toward recent rows.
            where, params = _veto_sql(feats)
            rows = db.execute(
                f"SELECT * FROM verdicts WHERE scored_at IS NOT NULL AND fwd_20d IS NOT NULL "
                f"{where} ORDER BY ts DESC LIMIT 20000", params,
            ).fetchall()

        mine: list[tuple[float, sqlite3.Row]] = []
        others: list[tuple[float, sqlite3.Row]] = []
        for r in rows:
            d = _distance(feats, r)
            if d is None or d > _MAX_DISTANCE:
                continue
            (mine if r["symbol"] == sym else others).append((d, r))

        out: dict[str, Any] = {}
        # Pass the FULL sorted list; `_cohort` truncates to k for the overall base rate but selects
        # the model's own rows separately. Truncating first crowded them out entirely: backfill
        # outnumbers live verdicts by ~140:1 per symbol, so the 40-nearest window was all replayed
        # setups and `when_model_said_buy` could never reach its minimum sample — and auto-seeding
        # made it progressively worse every night.
        same = _cohort(sorted(mine, key=lambda t: t[0]), k=k)
        if same:
            out["this_symbol"] = same
        near = _cohort(sorted(others, key=lambda t: t[0]), k=k)
        if near:
            out["analogues"] = near
        return out or None
    except Exception:  # noqa: BLE001
        log.warning("memory: similar_setups failed", exc_info=True)
        return None


def _cohort(pairs: Sequence[tuple[float, sqlite3.Row]], *, k: int = 40) -> dict | None:
    """Summarize matched neighbours into a track record, or None if the sample is too thin.

    Reports EXCESS return vs the benchmark over the same window wherever the benchmark is known.
    Raw forward return is nearly useless on its own: equities drift up, so ~55-60% of any 20-day
    window is positive regardless of setup, and a raw "62% positive" reads as edge when it is drift.
    """
    if len(pairs) < _MIN_SAMPLES:
        return None
    nearest = pairs[:k]                       # base rate: the k closest, whatever their origin
    r20 = [float(r["fwd_20d"]) for _, r in nearest if r["fwd_20d"] is not None]
    r5 = [float(r["fwd_5d"]) for _, r in nearest if r["fwd_5d"] is not None]
    if len(r20) < _MIN_SAMPLES:
        return None
    out: dict[str, Any] = {
        "n": len(r20),
        # Bars are not independent observations: backfill samples every third session, so two
        # adjacent rows on one name share almost their whole forward window, and at a one-year
        # horizon forty bars from two names is closer to two observations than forty. The reader
        # gets both counts and the prompt tells it which one to respect at which horizon.
        "n_symbols": _n_symbols(nearest),
        "median_fwd_20d_pct": round(_median(r20), 2),
        "positive_rate_20d": round(sum(1 for x in r20 if x > 0) / len(r20), 2),
    }
    if len(r5) >= _MIN_SAMPLES:
        out["median_fwd_5d_pct"] = round(_median(r5), 2)
    ex = _excess(nearest)
    if ex:
        out["vs_benchmark"] = ex
    # The long horizons pick THEIR OWN k nearest from the rows that carry the mark. Taking the same
    # `nearest` window would report nothing for a year — the closest rows are usually the newest,
    # and the newest are exactly the ones without a one-year outcome yet.
    for h in LONG_HORIZONS:
        ex_h = _excess(_having(pairs, h)[:k], horizon=h)
        if ex_h:
            out[f"vs_benchmark_{h}d"] = ex_h
    # Only REAL verdicts count here — a replayed setup has no opinion attached, and letting backfill
    # rows in would silently turn "how well does the model call this setup" into "how does this setup
    # drift", which is a different question wearing the same label.
    said_buy = _directional(pairs, BUY_SIGNALS, bullish=True, k=k)
    if said_buy:
        out["when_model_said_buy"] = said_buy
    said_sell = _directional(pairs, SELL_SIGNALS, bullish=False, k=k)
    if said_sell:
        out["when_model_said_sell"] = said_sell
    return out


def _having(pairs: Sequence[tuple[float, sqlite3.Row]], horizon: int) -> list[tuple[float, sqlite3.Row]]:
    """The subset of `pairs` (already sorted by distance) that carries a mark at `horizon`."""
    col = f"fwd_{horizon}d"
    return [(d, r) for d, r in pairs if r[col] is not None]


def _n_symbols(pairs: Sequence[tuple[float, sqlite3.Row]]) -> int:
    return len({r["symbol"] for _, r in pairs})


def _veto_sql(feats: dict[str, float | None]) -> tuple[str, list[float]]:
    """The veto axes as a SQL prefilter, equivalent to the early return in `_distance`.

    A row that is NULL on an axis is kept (the Python check skips missing dimensions too), and an
    axis the QUERY lacks contributes no predicate. This is the same rule as `_distance`, so nothing
    the prefilter removes could have been a neighbour — it only stops the Python loop from visiting
    every row in the table to reject most of them.
    """
    clauses: list[str] = []
    params: list[float] = []
    for col, scale, _weight, veto in _FEATURES:
        x = feats.get(col)
        if not veto or x is None:
            continue
        clauses.append(f"AND ({col} IS NULL OR ABS({col} - ?) <= ?)")
        params.extend((float(x), _VETO_DISTANCE * scale))
    return " ".join(clauses), params


# Signals that express a directional call. "hold" is deliberately in neither: it is the absence of a
# call, and scoring it would reward doing nothing in a rising market.
BUY_SIGNALS = ("buy", "strong_buy", "add")
SELL_SIGNALS = ("sell", "strong_sell", "trim", "reduce")


def _directional(
    pairs: Sequence[tuple[float, sqlite3.Row]], signals: Sequence[str], *, bullish: bool,
    k: int = 40,
) -> dict | None:
    """Score one side of the book. Returns None below the minimum sample.

    The SELL convention is inverted, and the inversion is the whole point: a sell is *right* when the
    name subsequently does WORSE than the benchmark, because the alternative to holding it was owning
    the index. So `median_excess_20d_pct` is negated for sells and reported as `avoided` — a positive
    number always means "this call was good", on both sides, which is the only way the two are
    comparable at a glance. Scoring sells on raw forward return would have marked every sell in a
    bull market as wrong regardless of judgement.
    """
    called = [
        (d, r) for d, r in pairs
        if r["origin"] == "model" and (r["signal"] or "").lower() in signals
    ]
    rows = called[:k]
    if len(rows) < _MIN_SAMPLES:
        return None
    fwd = [float(r["fwd_20d"]) for _, r in rows if r["fwd_20d"] is not None]
    if len(fwd) < _MIN_SAMPLES:
        return None
    out: dict[str, Any] = {
        "n": len(fwd),
        "n_symbols": _n_symbols(rows),
        "median_fwd_20d_pct": round(_median(fwd), 2),
    }
    ex = _excess(rows, bullish=bullish)
    if ex:
        out["vs_benchmark"] = ex
    for h in LONG_HORIZONS:
        ex_h = _excess(_having(called, h)[:k], horizon=h, bullish=bullish)
        if ex_h:
            out[f"vs_benchmark_{h}d"] = ex_h
    return out


def _excess(
    pairs: Sequence[tuple[float, sqlite3.Row]], *, horizon: int = 20, bullish: bool | None = None,
) -> dict | None:
    """Median `horizon`-bar return minus the benchmark's, over rows where both are known.

    With no direction attached (`bullish=None`) this is the whole cohort and the keys say so:
    `beat_rate` — "did this setup beat the index" is literally the question. For a directional call
    the keys are `correct_rate` and, on the sell side, `median_avoided`: a sell is *right* when the
    name subsequently does WORSE than the benchmark, because the alternative to holding it was
    owning the index, so the excess is negated and a positive number always means "this call was
    good", on both sides — the only way the two are comparable at a glance. Scoring sells on raw
    forward return would mark every sell in a bull market as wrong regardless of judgement.
    """
    col, bcol = f"fwd_{horizon}d", f"bench_fwd_{horizon}d"
    used = [(d, r) for d, r in pairs if r[col] is not None and r[bcol] is not None]
    diffs = [float(r[col]) - float(r[bcol]) for _, r in used]
    if len(diffs) < _MIN_SAMPLES:
        return None
    sign = -1.0 if bullish is False else 1.0
    if bullish is None:
        median_key, rate_key = f"median_excess_{horizon}d_pct", f"beat_rate_{horizon}d"
    else:
        median_key = f"median_excess_{horizon}d_pct" if bullish else f"median_avoided_{horizon}d_pct"
        rate_key = f"correct_rate_{horizon}d"
    return {
        "n": len(diffs),
        "n_symbols": _n_symbols(used),
        median_key: round(sign * _median(diffs), 2),
        rate_key: round(sum(1 for x in diffs if sign * x > 0) / len(diffs), 2),
    }


def recent_notes(*, kind: str | None = None, limit: int = 10) -> list[dict]:
    """Most recent prose memories, newest first. Returns [] on any failure."""
    try:
        sql = "SELECT ts, kind, symbol, body FROM notes"
        params: list[Any] = []
        if kind:
            sql += " WHERE kind = ?"
            params.append(kind)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(max(1, min(limit, 50)))
        with _lock:
            rows = _db().execute(sql, params).fetchall()
        return [{"ts": r["ts"], "kind": r["kind"], "symbol": r["symbol"], "body": r["body"]} for r in rows]
    except Exception:  # noqa: BLE001
        log.warning("memory: recent_notes failed", exc_info=True)
        return []


def blocked_summary(*, days: int = 30) -> dict | None:
    """Which risk rules actually bound recently, and how often.

    This is the question the blocked-trade log exists to answer. A cap that fires constantly is
    either mis-set or the strategy is repeatedly trying something the account's own rules forbid —
    and until this was aggregated, that pattern was only visible by reading raw JSONL by hand.
    """
    try:
        cutoff = time.time() - max(1, days) * 86_400
        with _lock:
            rows = _db().execute(
                "SELECT json_extract(meta, '$.skip_reason') reason, COUNT(*) c "
                "FROM notes WHERE kind = 'blocked' AND ts >= ? "
                "GROUP BY reason ORDER BY c DESC LIMIT 8",
                (cutoff,),
            ).fetchall()
        by_reason = {r["reason"]: r["c"] for r in rows if r["reason"]}
        if not by_reason:
            return None
        return {"days": days, "total": sum(by_reason.values()), "by_reason": by_reason}
    except Exception:  # noqa: BLE001
        log.warning("memory: blocked_summary failed", exc_info=True)
        return None


# -------------------------------------------------------------------------------------- scoring

def symbols_missing_baseline(candidates: Iterable[str], *, min_rows: int = 20) -> list[str]:
    """Of `candidates`, which have too little replayed history to produce a track record?

    A symbol added to the watchlist after the one-shot backfill would otherwise carry no base rate
    for ~20 trading days, and the UI would simply omit its track record with no indication why. The
    nightly scan uses this to seed newcomers automatically.
    """
    try:
        wanted = {(s or "").upper() for s in candidates if s}
        if not wanted:
            return []
        with _lock:
            rows = _db().execute(
                "SELECT symbol, COUNT(*) c FROM verdicts WHERE origin = 'backfill' GROUP BY symbol"
            ).fetchall()
        have = {r["symbol"]: r["c"] for r in rows}
        missing = [s for s in wanted if have.get(s, 0) < min_rows]
        # Deterministic ALPHABETICAL order let a permanently un-backfillable symbol (delisted, too
        # little history, a bad ticker) sit at the front of the queue forever, consuming one of the
        # four nightly slots on every run and starving everything after it. Order by how recently we
        # last tried, so a symbol that keeps failing drifts to the back instead of blocking the line.
        attempts = {
            r["symbol"]: r["ts"]
            for r in _db().execute(
                "SELECT symbol, MAX(ts) ts FROM notes WHERE kind = 'seed_attempt' GROUP BY symbol"
            )
        }
        return sorted(missing, key=lambda x: (attempts.get(x, 0.0), x))
    except Exception:  # noqa: BLE001
        log.warning("memory: symbols_missing_baseline failed", exc_info=True)
        return []


def _pending_where(now: float, min_age_days: int) -> tuple[str, tuple[float, ...]]:
    """Rows with at least one horizon that is both unwritten and old enough to be measurable.

    Each horizon has its own age gate, so a row graded at 20 days last month is not refetched every
    night for a quarter's mark it cannot have yet — and a row nothing can ever grade (a delisted
    name) stays pending exactly as it always did, which is the honest state for it.
    """
    now = float(now)
    clauses = ["(scored_at IS NULL AND ts < ?)"]
    params: list[float] = [now - min_age_days * 86_400]
    for h in LONG_HORIZONS:
        clauses.append(f"(fwd_{h}d IS NULL AND ts < ?)")
        params.append(now - _HORIZON_MIN_AGE_DAYS[h] * 86_400)
    return "(" + " OR ".join(clauses) + ")", tuple(params)


def pending_work(*, min_age_days: int = 5) -> dict[str, float]:
    """Symbols with something left to grade, each with the `ts` of its OLDEST such row.

    The timestamp is what the caller needs to choose a fetch range: a row from fourteen months ago
    cannot be graded from a one-year series because its anchor bar is not in it — which is exactly
    why no 252-bar mark could have been written before this existed, whatever the schema said.
    """
    try:
        where, params = _pending_where(time.time(), min_age_days)
        with _lock:
            rows = _db().execute(
                f"SELECT symbol, MIN(ts) oldest FROM verdicts WHERE {where} GROUP BY symbol", params
            ).fetchall()
        return {r["symbol"]: float(r["oldest"]) for r in rows}
    except Exception:  # noqa: BLE001
        log.warning("memory: pending_work failed", exc_info=True)
        return {}


def pending_symbols(*, min_age_days: int = 5) -> list[str]:
    """Symbols with verdicts old enough to have a measurable outcome at SOME horizon, not yet scored there."""
    return list(pending_work(min_age_days=min_age_days))


def score_symbol(
    symbol: str,
    dates: Sequence[str],
    closes: Sequence[float],
    *,
    bench_dates: Sequence[str] | None = None,
    bench_closes: Sequence[float] | None = None,
) -> int:
    """Fill in realized forward returns for one symbol from a fetched price series.

    Returns how many rows gained at least one mark. A horizon is only written once the FULL window
    exists, so a 20-day return is never computed from 11 bars and quietly treated as equivalent. The
    5- and 20-bar marks are written together and set `scored_at`; each long horizon is written on
    its own the first time the series reaches far enough, and a mark already present is never
    rewritten — a row visited for its one-year mark keeps the 20-day number it was graded with.

    `bench_*` is the benchmark series (^GSPC) indexed on its own dates — the same calendar window is
    located by date, not by reusing the symbol's index, because a symbol can be missing bars the
    benchmark has (halts, late listings) and an off-by-a-few-bars comparison would be worse than none.
    """
    try:
        idx = {d: i for i, d in enumerate(dates)}
        bidx = {d: i for i, d in enumerate(bench_dates or ())}
        long_cols = ", ".join(f"fwd_{h}d" for h in LONG_HORIZONS)
        long_missing = " OR ".join(f"fwd_{h}d IS NULL" for h in LONG_HORIZONS)
        with _lock:
            db = _db()
            rows = db.execute(
                f"SELECT id, asof_date, price, scored_at, {long_cols} FROM verdicts "
                f"WHERE symbol = ? AND (scored_at IS NULL OR {long_missing})",
                ((symbol or "").upper(),),
            ).fetchall()
            updates: list[tuple] = []
            now = time.time()
            for r in rows:
                i = idx.get(r["asof_date"])
                if i is None:
                    continue
                # Anchor on the SERIES' own bar, not the price stored at verdict time. Yahoo
                # returns SPLIT- AND DIVIDEND-ADJUSTED closes, and the whole series is re-adjusted
                # retroactively — so a stored raw price divided into a later adjusted close reports
                # a fabricated return across any split or dividend (a 4:1 split alone shows as -75%).
                # Both ends must come from the same adjusted series or the number is meaningless.
                base = float(closes[i]) if i < len(closes) and closes[i] else float(r["price"] or 0)
                if not base:
                    continue
                bj = bidx.get(r["asof_date"])
                bbase = float(bench_closes[bj]) if (
                    bj is not None and bench_closes and bench_closes[bj]) else None

                def _bench(h: int) -> float | None:
                    # BOTH ends located by date. Stepping `h` *benchmark* bars from the anchor was
                    # only half the promise in the docstring: when the symbol is missing bars the
                    # benchmark has (halt, late listing, thin crypto calendar), the two windows end
                    # on different days and the "excess" compares mismatched periods — precisely
                    # the case the date-anchoring was introduced to avoid.
                    if bbase is None:
                        return None
                    return _fwd_to_date(bench_closes, bidx, dates, i, h, bbase)

                fwd: dict[int, float | None] = {}
                bench: dict[int, float | None] = {}
                unscored = r["scored_at"] is None
                if unscored:
                    f20 = _fwd(closes, i, 20, base)
                    if f20 is None:
                        continue  # not enough history yet — leave unscored and revisit next run
                    fwd[5], fwd[20] = _fwd(closes, i, 5, base), f20
                    bench[5], bench[20] = _bench(5), _bench(20)
                for h in LONG_HORIZONS:
                    if r[f"fwd_{h}d"] is not None:
                        continue
                    f = _fwd(closes, i, h, base)
                    if f is None:
                        continue
                    fwd[h], bench[h] = f, _bench(h)
                if not fwd:
                    continue
                updates.append((
                    *(fwd.get(h) for h in HORIZONS),
                    *(bench.get(h) for h in HORIZONS),
                    now if unscored else None,
                    r["id"],
                ))
            if updates:
                # COALESCE: a horizon this visit did not compute is passed as NULL and keeps
                # whatever the row already had — so the one statement serves both the first
                # grading and every later fill-in without ever overwriting a written mark.
                sets = ", ".join(
                    f"{c}=COALESCE(?, {c})"
                    for c in (*(f"fwd_{h}d" for h in HORIZONS), *(f"bench_fwd_{h}d" for h in HORIZONS))
                )
                db.executemany(
                    f"UPDATE verdicts SET {sets}, scored_at=COALESCE(scored_at, ?) WHERE id=?",
                    updates,
                )
                db.commit()
            return len(updates)
    except Exception:  # noqa: BLE001
        log.warning("memory: score_symbol failed for %s", symbol, exc_info=True)
        return 0


def _fwd_to_date(
    bench_closes: Sequence[float], bidx: dict, dates: Sequence[str], i: int, horizon: int, base: float,
) -> float | None:
    """Benchmark return over the SAME calendar window the symbol's `horizon` bars covered.

    Finds the symbol's end date, then looks that date up in the benchmark's own series. Falls back to
    the nearest earlier benchmark date when the exact one is missing (a holiday the symbol trades but
    the index does not — crypto against the S&P), and gives up rather than guessing if nothing is
    close enough.
    """
    j = i + horizon
    if j >= len(dates):
        return None
    end = dates[j]
    bj = bidx.get(end)
    if bj is None:
        earlier = [d for d in bidx if d <= end]
        if not earlier:
            return None
        bj = bidx[max(earlier)]
    px = bench_closes[bj] if bj < len(bench_closes) else None
    if not px or not base:
        return None
    return round((float(px) - base) / base * 100.0, 3)


def _fwd(closes: Sequence[float], i: int, horizon: int, base: float) -> float | None:
    j = i + horizon
    if j >= len(closes) or not closes[j]:
        return None
    return round((float(closes[j]) - base) / base * 100.0, 3)


# ------------------------------------------------------------------------------------- seeding

# Findings this project measured on its own universe (see research/README.md). They live here too so
# recall has real content before any verdict has been scored, and so the numbers survive as text that
# can be searched rather than only as prose baked into a prompt.
_RESEARCH: tuple[tuple[str, str], ...] = (
    ("gap-fill-base-rates",
     "Overnight gap fill rates measured on this watchlist: 0.5-1% gaps fill 87.8% within 10 sessions, "
     "1-2% fill 81.5%, 2-5% fill 71.9%, but >5% gaps fill only 39.4%. High-volume gaps fill 69.5%. "
     "So a 2-5% down gap with no catalyst is constructive; a >5% gap is usually repricing, not noise."),
    ("gap-catalyst-caveat",
     "Gaps accompanied by a news catalyst behave differently from mechanical gaps: they are repricing "
     "to new information and mean-revert far less often. Classify catalyst gaps as 'avoid', not 'buy the dip'."),
    ("dividend-etf-horizon",
     "Dividend-focused ETFs (e.g. SCHD) are a WORSE fit than broad total-return funds when the goal is "
     "long-horizon accumulation: dividends are taxable events in a taxable account and force "
     "distribution rather than compounding. They fit better near or after retirement when income "
     "matters more than terminal value. The user's original hypothesis on this was correct."),
    ("short-interest-direction",
     "High short interest — especially high days-to-cover — predicts UNDERperformance, not squeezes "
     "(Boehmer/Jones/Zhang 2008). The effect concentrates in small/illiquid names and is weak-to-absent "
     "in mega-caps and broad ETFs. Squeeze folklore is not supported by the SEC's own GME 2021 report."),
)


def seed_research() -> int:
    """Insert the standing research findings once. Returns how many were added.

    Idempotent by slug so a restart (or a redeploy) doesn't pile up duplicates that would then
    dominate FTS results.
    """
    added = 0
    try:
        with _lock:
            db = _db()
            have = {
                json.loads(r["meta"] or "{}").get("slug")
                for r in db.execute("SELECT meta FROM notes WHERE kind = 'research'")
            }
            for slug, body in _RESEARCH:
                if slug in have:
                    continue
                db.execute(
                    "INSERT INTO notes (ts, kind, symbol, body, meta) VALUES (?,?,?,?,?)",
                    (time.time(), "research", None, body, json.dumps({"slug": slug})),
                )
                added += 1
            db.commit()
    except Exception:  # noqa: BLE001
        log.warning("memory: seed_research failed", exc_info=True)
    return added


# ------------------------------------------------------------------------------------ housekeeping

def retention_days() -> int:
    """How far back rows are kept — the ceiling a backfill range should respect."""
    return _RETAIN_DAYS


def prune(retain_days: int = _RETAIN_DAYS, *, notes_retain_days: int = _NOTES_RETAIN_DAYS) -> int:
    """Drop rows past the retention window. Returns rows removed."""
    try:
        now = time.time()
        with _lock:
            db = _db()
            n = db.execute("DELETE FROM verdicts WHERE ts < ?", (now - retain_days * 86_400,)).rowcount
            n += db.execute("DELETE FROM notes WHERE ts < ?", (now - notes_retain_days * 86_400,)).rowcount
            db.commit()
        return n
    except Exception:  # noqa: BLE001
        log.warning("memory: prune failed", exc_info=True)
        return 0


def stats() -> dict:
    """Shape of what memory holds — surfaced at /memory/stats so it's inspectable, not a black box."""
    try:
        with _lock:
            db = _db()
            v = db.execute(
                "SELECT COUNT(*) c, SUM(scored_at IS NOT NULL) scored, MIN(ts) oldest, "
                "COUNT(DISTINCT symbol) syms FROM verdicts"
            ).fetchone()
            n = db.execute("SELECT COUNT(*) c FROM notes").fetchone()
            by_kind = db.execute(
                "SELECT kind, COUNT(*) c FROM notes GROUP BY kind ORDER BY c DESC"
            ).fetchall()
            # Two scorecards, deliberately NOT merged: 'model' is what the analyst *said* on the
            # watchlist, 'sandbox' is what the paper trader actually *bought*. Different processes
            # answering different questions — averaging them would hide which one works.
            def _card(origin: str, signals: Sequence[str], bullish: bool, h: int) -> sqlite3.Row:
                # `beat` uses a comparison whose direction flips for sells: owning the index was the
                # alternative to holding, so a sell is right when the name UNDERperforms it.
                cmp_ = ">" if bullish else "<"
                marks = ",".join("?" for _ in signals)
                f, b = f"fwd_{h}d", f"bench_fwd_{h}d"
                return db.execute(
                    f"SELECT COUNT(*) n, COUNT(DISTINCT symbol) syms, AVG({f} > 0) rate, "
                    f"AVG({f}) avg, "
                    f"AVG(CASE WHEN {b} IS NOT NULL THEN {f} - {b} END) exc, "
                    f"AVG(CASE WHEN {b} IS NOT NULL THEN {f} {cmp_} {b} END) beat "
                    f"FROM verdicts WHERE {f} IS NOT NULL AND origin = ? "
                    f"AND LOWER(COALESCE(signal,'')) IN ({marks})",
                    (origin, *signals),
                ).fetchone()

            cards = {
                "buy_calls": ("model", BUY_SIGNALS, True),
                "sell_calls": ("model", SELL_SIGNALS, False),
                "sandbox_buys": ("sandbox", BUY_SIGNALS, True),
                "sandbox_sells": ("sandbox", SELL_SIGNALS, False),
            }
            rows_20 = {label: _card(*spec, 20) for label, spec in cards.items()}
            # Long horizons on the ANALYST's cards only. The sandbox's own decisions are a handful
            # of fills — the account can never learn a strategy lesson from its own outcomes at any
            # horizon, so those are reported as dollars in the ledger cost block, never as a rate.
            rows_long = {
                label: {h: _card(*spec, h) for h in LONG_HORIZONS}
                for label, spec in cards.items() if spec[0] == "model"
            }
            coverage = db.execute(
                "SELECT " + ", ".join(
                    f"SUM(fwd_{h}d IS NOT NULL) h{h}, MIN(CASE WHEN fwd_{h}d IS NULL THEN ts END) p{h}"
                    for h in HORIZONS
                ) + " FROM verdicts"
            ).fetchone()
            by_origin = {
                r["origin"]: r["c"]
                for r in db.execute("SELECT origin, COUNT(*) c FROM verdicts GROUP BY origin")
            }
        out: dict[str, Any] = {
            "verdicts": v["c"] or 0,
            "scored": v["scored"] or 0,
            "symbols": v["syms"] or 0,
            "oldest_ts": v["oldest"],
            "retention_days": _RETAIN_DAYS,
            # Per horizon: rows carrying the mark, and the age of the oldest row still without it.
            # An `oldest_pending_age_days` far past the horizon's own length is a row nothing can
            # grade (delisted, re-dated bar), not a backlog — visible here so it is never mistaken
            # for one.
            "horizons": {
                f"{h}d": {
                    "scored": coverage[f"h{h}"] or 0,
                    "oldest_pending_age_days": (
                        round((time.time() - float(coverage[f"p{h}"])) / 86_400, 1)
                        if coverage[f"p{h}"] is not None else None),
                }
                for h in HORIZONS
            },
            "notes": n["c"] or 0,
            "notes_by_kind": {r["kind"]: r["c"] for r in by_kind},
            "by_origin": by_origin,
            "db_bytes": _FILE.stat().st_size if _FILE.exists() else 0,
        }
        # The scorecards: of everything it called, how much better was that than simply owning the
        # index? The number that says whether any of this works — surfaced even when unflattering.
        # `correct_rate_20d` means the same thing on both sides (higher is better), so buys and sells
        # can be read side by side without mentally inverting one of them.
        for label, (_origin, _signals, bullish) in cards.items():
            row = rows_20[label]
            if (row["n"] or 0) < _MIN_SAMPLES:
                continue
            card = _card_block(row, bullish, 20)
            card["n_symbols"] = row["syms"]
            for h, hrow in rows_long.get(label, {}).items():
                if (hrow["n"] or 0) >= _MIN_SAMPLES:
                    block = _card_block(hrow, bullish, h)
                    block["n_symbols"] = hrow["syms"]
                    card[f"at_{h}d"] = block
            out[label] = card
        return out
    except Exception:  # noqa: BLE001
        log.warning("memory: stats failed", exc_info=True)
        return {"error": "unavailable"}


def _card_block(row: sqlite3.Row, bullish: bool, h: int) -> dict[str, Any]:
    """One scorecard at one horizon, from a `_card` aggregate row. Keys carry the horizon."""
    card: dict[str, Any] = {
        "n": row["n"],
        f"avg_fwd_{h}d_pct": round(float(row["avg"] or 0), 2),
    }
    if row["exc"] is not None:
        sign = 1.0 if bullish else -1.0
        key = f"avg_excess_{h}d_pct" if bullish else f"avg_avoided_{h}d_pct"
        card[key] = round(sign * float(row["exc"]), 2)
        card[f"correct_rate_{h}d"] = round(float(row["beat"] or 0), 3)
    return card


# --------------------------------------------------------------------------------------- helpers

def _signal_str(v: Any) -> str | None:
    """Normalise a verdict signal to its bare value ("buy"), whatever shape the caller passed.

    Pydantic's `model_dump()` (without mode="json") hands back the Enum MEMBER, and `str()` of that is
    `"Signal.buy"`, not `"buy"`. Stored that way it matched no buy filter, so `when_model_said_buy`
    and the /memory/stats scorecard would have sat empty forever while looking perfectly healthy —
    the worst kind of bug in a feature whose entire job is to report honestly. Normalised here at the
    boundary rather than at each call site so no future caller can reintroduce it.
    """
    v = getattr(v, "value", v)          # Enum member -> its value
    s = str(v or "").strip()
    if "." in s and s.split(".", 1)[0].lower().endswith("signal"):
        s = s.split(".", 1)[1]          # tolerate an already-stringified "Signal.buy"
    return s[:24].lower() or None


def _num(v: Any) -> float | None:
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


def _median(xs: Iterable[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if not n:
        return 0.0
    m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2.0


def _asof_date(summary: dict) -> str:
    """The date of the last bar the verdict saw — the anchor forward returns are measured from.

    Normalised to the bare `YYYYMMDD` that `Series.dates` uses, so `score_symbol` can index straight
    into the fetched series. Falling back to *today* would silently mis-anchor every verdict formed
    on a weekend or before the close, so the caller's real bar date is strongly preferred.
    """
    d = summary.get("as_of_date") or summary.get("last_date")
    if isinstance(d, str):
        digits = "".join(c for c in d if c.isdigit())
        if len(digits) >= 8:
            return digits[:8]
    return time.strftime("%Y%m%d", time.gmtime())
