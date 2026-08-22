"""SWT-4 — the percentile pass: where each name sits in TONIGHT'S cross-section, metric by metric.

"RSI 81.4" is a number the reader has to already know the distribution to use. "RSI at the 96th
percentile of tonight's 3,101-name scan" is a read. That is the whole feature: the cross-section is
already sitting in scan.db, one row per (night, symbol), so the rank costs one sweep of the night
and a sort — no fetches, no model call, nothing that can fail halfway and leave a number behind.

WHAT THIS PASS IS, MECHANICALLY
-------------------------------
It runs AFTER a night is stored, reads that night whole, ranks ten metrics across it, and writes the
ranks back into columns no other writer owns (`scan_store.write_percentiles`). It never touches a
measured column, and running it twice produces byte-identical results, because it derives everything
from the measurements and nothing from its own previous output.

A PERCENTILE HERE IS A RANK, NOT A SCORE — read this before using one
---------------------------------------------------------------------
`rsi14_pctile: 96.0` says "96% of the names measured tonight had a lower RSI". It does not say the
name is good, strong, cheap, or worth buying. High RSI is not "better" than low RSI; a name at the
99th percentile of `adr20_pct` is the most volatile in the market, which is a fact and not a
compliment; `pct_off_52w_high_pctile: 100` means closest to its high, and `rel_volume_pctile: 2`
means almost nobody traded it. Direction of desirability is the READER'S judgement and differs by
strategy, so this module deliberately refuses to encode one — no sign flips, no "good" end.

The consequence, which is the reason this paragraph exists at all: DO NOT AVERAGE THESE INTO A
COMPOSITE. app/screener.py carries the same warning for the value screen — "folding a mean-reversion
value tilt into a momentum score produces a number that means nothing in either direction" — and
averaging ranks is that mistake with the units hidden. The mean of a momentum percentile and a
volatility percentile is not a quality: a calm leader and a wild laggard land on the same number and
the composite cannot tell them apart. A composite needs a stated thesis about direction and weight
(which is what `screener.value_score` is, weights, clipping and all); ranks alone are not one.

THE ABSENT RULE, which outranks everything else here
-----------------------------------------------------
A metric that is None gets a None percentile. NEVER the 0th percentile. The 0th percentile is a
confident claim that this name is the worst in the market, and it would be made about precisely the
names we could not measure — a stock with no ADX yet, a name whose benchmark fetch failed. Same rule
one level up: if too little of the night could be measured on a metric, EVERY row's rank for that
metric is None rather than a rank over a sample presented as a census.

Threading: every function here is SYNCHRONOUS (it is SQLite underneath). Async callers wrap it —
`await asyncio.to_thread(percentiles.run)`, the pattern main.py uses for the rest of scan_store.
"""

from __future__ import annotations

import logging
import time
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from typing import Any

from . import scan_store
from .scan_store import PCT_COLUMNS, PCT_METRICS, PCT_SUFFIX

log = logging.getLogger("signals.percentiles")

# THE COVERAGE FLOOR, read from scan_store rather than restated so the two can never drift apart.
# It is the same threshold and the same argument breadth() already applies: a rank computed over 300
# measurable rows of a 3,100-row night is not a MARKET percentile, it is a percentile of the tenth
# of the market that happened to have an ADX, handed to a reader who will read the word "market".
# Below this share of the night's rows the metric is not ranked at all — for anyone, including the
# rows that were measurable, because a rank is a statement about the population and half the
# population is missing.
_MIN_COVERAGE = scan_store._MIN_COVERAGE

# THE DISTRIBUTION FLOOR, and it is a separate question from coverage. Coverage asks "did we measure
# enough OF the night"; this asks "is the night big enough to have a distribution at all". Ten
# measured names cannot produce a percentile in any meaningful sense — the finest step available is
# 100/(n-1) points, so at n=10 every rank is a multiple of 11.1 and calling one of them "the 89th
# percentile of the market" is fabricated precision on top of a sample.
#
# 20 is a judgement call, stated as one, in the same spirit as _MIN_COVERAGE: it is where the step
# size (about 5 points) stops being the dominant term in the number. Its most important job is the
# degenerate case — a night with ONE row must not report that name at the 100th percentile of
# everything, which is the most confident possible statement made from no information at all.
_MIN_MEASURED = 20


def _measurable(v: Any) -> float | None:
    """`v` as a finite float, or None if it is not a number we can rank.

    Bools are excluded explicitly: `True` is 1.0 to Python, and a boolean column that wandered in
    here would sort into the distribution as a real measurement.
    """
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    # NaN sorts unpredictably and would corrupt every rank around it; inf pins an end of the scale.
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def rank(values: Sequence[Any], *, min_measured: int = _MIN_MEASURED,
         min_coverage: float = _MIN_COVERAGE) -> list[float | None]:
    """Percentile ranks for ONE metric across ONE night, POSITIONALLY ALIGNED with `values`.

    Ascending, always: a higher percentile means a higher raw value, for every metric, with no
    exceptions and no sign flips. See the module docstring — this is a rank, not a score.

    The scale is pinned at both ends: the lowest measured value is exactly 0.0 and the highest is
    exactly 100.0, i.e. `(position) / (n - 1) * 100` over the n measured values. The alternative
    ("share of the night at or below this value") can never emit 0 and gives the single lowest name
    in the market a positive percentile, which reads as though something ranked below it.

    TIES SHARE ONE MIDRANK, and both halves of that are deliberate:
      * SHARE ONE. Two names with the same rel_volume must get the same percentile. Any convention
        that breaks the tie — by symbol, by row order — invents an ordering the data does not
        contain and prints it to one decimal place, which is precisely "fabricating precision".
      * MIDRANK, i.e. the centre of the block the tied values occupy, rather than its top or bottom.
        The alternatives are not symmetric: with "share at or below", a large tie block at the
        bottom of a metric (thousands of names with rel_volume exactly 0) absorbs its own mass and
        reports the WORST names in the market at a middling percentile. The midrank puts a block at
        its centre of mass whichever end it sits at, and is the only choice that gives the same
        answer if every value's sign is flipped.

    Rounded to one decimal, the precision the rest of the store reports in. On a 3,100-name night one
    step is 0.032 points, so the top two or three names can all print 100.0 — that is the DISPLAY
    precision agreeing with itself, not a claim that they are tied, and a percentile is not the place
    to go looking for a tie-break anyway. (Measured against the live 2026-08-21 cross-section: the
    highest `rel_volume` is 97.58 and the runner-up 13.65, both at 100.0. Which is the point — the
    rank says "nothing traded more heavily", and the raw number beside it says by how much.)

    Returns None for a value that was not measurable — never 0.0, which would be a claim that the
    name we could not measure is the worst in the market. Returns all-None when too few of the
    night's rows were measurable (`min_coverage`) or when the night is too small to have a
    distribution (`min_measured`).
    """
    total = len(values)
    if not total:
        return []
    measured = [f for f in (_measurable(v) for v in values) if f is not None]
    n = len(measured)
    # A distribution needs at least two points whatever the caller passes, or the (n - 1) below is a
    # division by zero dressed up as a percentile.
    if n < max(2, int(min_measured)) or n < total * float(min_coverage):
        return [None] * total

    ordered = sorted(measured)
    span = float(n - 1)
    # One sort, then two bisects per row: O(n log n) for the night rather than O(n) per row. At
    # ~3,100 rows x 10 metrics the difference is a second of wall clock against a minute of it.
    out: list[float | None] = []
    for v in values:
        f = _measurable(v)
        if f is None:
            out.append(None)          # THE ABSENT RULE. Not 0.0. See the module docstring.
            continue
        lo = bisect_left(ordered, f)   # how many measured values are strictly below this one
        hi = bisect_right(ordered, f)  # [lo, hi) is the block of values equal to it
        out.append(round((lo + (hi - lo - 1) / 2.0) / span * 100.0, 1))
    return out


def compute(rows: Sequence[dict]) -> tuple[dict[str, dict[str, float | None]], dict[str, dict]]:
    """Rank a night's cross-section. Returns ({symbol: {column: rank|None}}, report).

    The report is per metric — how many rows carried it, whether it was ranked, and if not, why —
    because "rel_volume is null for every name tonight" must be a line an operator can read in the
    run summary, not a silence they have to go diffing the database to notice.

    Every symbol appears in the mapping with every column present, None included. That is what makes
    the write idempotent AND self-clearing: a metric that stops being rankable has its stale rank
    nulled rather than left showing last night's standing under tonight's date.
    """
    symbols = [str(r.get("symbol") or "").strip().upper() for r in rows]
    out: dict[str, dict[str, float | None]] = {s: {} for s in symbols if s}
    report: dict[str, dict] = {}

    for metric in PCT_METRICS:
        col = PCT_COLUMNS[metric]
        values = [r.get(metric) for r in rows]
        measured = sum(1 for v in values if _measurable(v) is not None)
        ranks = rank(values)
        ranked = any(x is not None for x in ranks)
        report[metric] = {
            "column": col,
            "measured": measured,
            "of": len(rows),
            "ranked": ranked,
            # A stated reason, not a bare False. The two ways a metric goes unranked are different
            # facts about the night and an operator needs to be able to tell them apart.
            "reason": None if ranked else (
                f"only {measured} of {len(rows)} rows measurable — below the "
                f"{_MIN_COVERAGE:.0%} coverage floor"
                if measured < len(rows) * _MIN_COVERAGE
                else f"{measured} measurable — below the {_MIN_MEASURED}-row distribution floor"
            ),
        }
        for sym, value in zip(symbols, ranks):
            if sym:
                out[sym][col] = value
    return out, report


def run(d: str | None = None) -> dict:
    """Rank one stored night and write the ranks back. Never raises; always returns a summary.

    `d` defaults to the most recent night stored. Idempotent by construction — it reads only the
    measured columns and overwrites its own — so it is safe to run at the tail of the nightly job,
    again from a backfill, and again after that.

    The status is never inferred from a count. "refused" (the date was not a date), "no_scan"
    (nothing has ever been stored) and "empty" (that night holds no rows) are three different
    answers, and the counters stay None on all three rather than reporting a confident 0 rows ranked
    out of a night we never looked at.
    """
    t0 = time.monotonic()
    summary: dict[str, Any] = {
        "job": "percentiles",
        "status": None,
        "reason": None,
        "generated_at": time.time(),
        "as_of": None,
        "suffix": PCT_SUFFIX,
        "rows": None,
        "updated": None,
        "ranked": None,
        "metrics": None,
        "duration_s": None,
    }
    try:
        if d is None:
            night = scan_store.latest_date()
            if night is None:
                summary.update(status="no_scan",
                               reason="no market scan has ever been stored — nothing to rank")
                return summary
        else:
            try:
                night = scan_store._norm_d(d)
            except ValueError as e:
                summary.update(status="refused", reason=str(e))
                return summary

        summary["as_of"] = night
        rows = scan_store.cross_section(night)
        summary["rows"] = len(rows)
        if not rows:
            summary.update(status="empty", reason=f"night {night} holds no rows to rank")
            return summary

        values, report = compute(rows)
        summary["updated"] = scan_store.write_percentiles(night, values)
        summary["metrics"] = report
        summary["ranked"] = [m for m, r in report.items() if r["ranked"]]
        summary["status"] = "ok"
        return summary
    except Exception as e:  # noqa: BLE001 — the pass is decoration on a night whose measurements are
        # already committed and correct. A failure here must cost the ranks (which stay NULL, i.e.
        # "not ranked", which every reader already handles) and never the night, the nightly job's
        # summary, or a request.
        log.warning("percentiles: run failed for %s", d, exc_info=True)
        summary.update(status="failed", reason=f"{type(e).__name__}: {e}")
        return summary
    finally:
        summary["duration_s"] = round(time.monotonic() - t0, 2)
