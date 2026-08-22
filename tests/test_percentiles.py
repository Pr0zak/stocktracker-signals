"""SWT-4 — the percentile pass, and the one number it must never print.

A percentile is the cheapest readability win this service has: "RSI 81.4" asks the reader to already
know the distribution, "RSI at the 96th percentile of tonight's 3,101-name scan" hands it to them.
It is also, for exactly the same reason, the easiest place to ship the defect this codebase keeps
correcting. A name whose ADX could not be measured, written down as the 0th percentile, is not a
missing value — it is a confident published claim that this stock is the single worst in the market,
manufactured out of nothing. Half this file exists to pin that down: a None metric gets a None
percentile, the stored column is SQL NULL rather than 0.0, and the unmeasured row does not shift
anybody else's rank on the way past.

The other half guards the population. A rank over 300 measurable rows of a 3,100-row night is not a
market percentile, it is a percentile of the tenth of the market that happened to have the metric —
so below scan_store's own coverage floor the metric is not ranked for ANYONE, including the rows that
were measurable. And a night with one row in it must not report that name at the 100th percentile of
everything, which is the most confident possible statement made from no information at all.

Two structural guarantees are exercised end to end rather than asserted about in the abstract. The
pass must be idempotent, because it runs at the tail of every nightly job and again from any
backfill. And it must survive `insert_night` re-running the same night — the ON CONFLICT ... DO
UPDATE contract that `test_reinserting_a_night_preserves_a_column_written_after_the_original_insert`
has been protecting since before these columns existed.

Isolation is `SIGNALS_DATA_DIR` plus `importlib.reload()`, in dependency order: the modules bind
`_DATA_DIR` at import time, so setting the env var without reloading gets the previous test's
database.
"""
from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A fresh, isolated scan.db, plus the pass that writes into it."""
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    from app import scan_store as ss

    importlib.reload(ss)
    from app import percentiles as pc

    importlib.reload(pc)
    yield ss, pc
    if ss._conn is not None:
        ss._conn.close()
        ss._conn = None


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """A TestClient over an isolated data dir, plus the store and the pass behind it."""
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    import app.settings_store as st
    importlib.reload(st)
    import app.scan_store as ss
    importlib.reload(ss)
    import app.percentiles as pc
    importlib.reload(pc)
    import app.market_scan_job as job
    importlib.reload(job)
    import app.main as m
    importlib.reload(m)
    m._cache.clear()
    with TestClient(m.app) as c:
        yield c, ss, pc
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


def _ladder(n: int = 100, **fixed) -> list[dict]:
    """`n` names whose rsi14 is 1..n — a distribution whose right answer is known by construction."""
    return [_row(f"S{i:03d}", rsi14=float(i + 1), **fixed) for i in range(n)]


def _pct(store_mod, symbol: str, column: str, d: str = "20260821"):
    row = store_mod.symbol_row(symbol, d)
    assert row is not None, f"{symbol} is not in night {d}"
    return row[column]


# ---- the ranking itself

def test_a_known_distribution_ranks_correctly_with_the_top_and_bottom_pinned_exactly(store):
    """Ranks are ascending over the whole night, and the two ends are exact, not nearly-exact.

    The scale runs 0.0 at the lowest measured value to 100.0 at the highest. The obvious alternative
    — "the share of the night at or below this value" — can never emit 0, so the single weakest name
    in the market would report a positive percentile as though something ranked below it.
    """
    ss, pc = store
    assert ss.insert_night(_ladder(100), d="20260821") == 100

    out = pc.run()
    assert out["status"] == "ok"
    assert (out["rows"], out["updated"]) == (100, 100)
    assert "rsi14" in out["ranked"]

    assert _pct(ss, "S000", "rsi14_pctile") == 0.0      # rsi 1, the lowest measured
    assert _pct(ss, "S099", "rsi14_pctile") == 100.0    # rsi 100, the highest
    assert _pct(ss, "S050", "rsi14_pctile") == 50.5     # 50 of 99 steps up the ladder
    assert _pct(ss, "S001", "rsi14_pctile") == 1.0


def test_a_rank_is_ascending_for_every_metric_including_the_ones_whose_high_end_is_bad(store):
    """A percentile is a position, never a verdict — so nothing here is sign-flipped to be "good".

    `pct_off_52w_high` is negative or zero, so its 100th percentile is the name closest to its high;
    `adr20_pct`'s 100th percentile is the most volatile name in the market, which is a fact and not a
    compliment. Flipping either to make "high = good" would turn a rank into a score, and a score is
    a thesis this module deliberately does not hold.
    """
    ss, pc = store
    rows = [_row(f"S{i:03d}", pct_off_52w_high=float(-i), adr20_pct=float(i + 1)) for i in range(40)]
    assert ss.insert_night(rows, d="20260821") == 40
    assert pc.run()["status"] == "ok"

    assert _pct(ss, "S000", "pct_off_52w_high_pctile") == 100.0   # at its high
    assert _pct(ss, "S039", "pct_off_52w_high_pctile") == 0.0     # 39% below it
    assert _pct(ss, "S039", "adr20_pct_pctile") == 100.0          # the wildest, not the best


# ---- THE RULE: absent is never zero

def test_a_none_metric_yields_a_none_percentile_and_never_the_zeroth(store):
    """The defect this whole feature could have shipped: a name we could not measure, ranked worst.

    S000 has no RSI at all. The 0th percentile is not a placeholder for that — it is the claim that
    S000 has the lowest RSI in the market, made about the one name whose RSI nobody knows.
    """
    ss, pc = store
    rows = _ladder(100)
    rows[0]["rsi14"] = None
    assert ss.insert_night(rows, d="20260821") == 100
    assert pc.run()["status"] == "ok"

    got = _pct(ss, "S000", "rsi14_pctile")
    assert got is None
    assert got != 0.0 and got != 0

    # And it is a real SQL NULL in the column, not a 0.0 that only reads as None through the helper.
    raw = ss._db().execute(
        "SELECT rsi14_pctile IS NULL n FROM scan WHERE d = ? AND symbol = ?", ("20260821", "S000")
    ).fetchone()
    assert raw["n"] == 1


def test_an_unmeasured_row_does_not_take_a_rank_away_from_anybody_else(store):
    """The absent row leaves the distribution alone: it is not counted, and it is not ranked.

    A None that fell into the sort as a zero would push every real name up a place AND plant an
    imposter at the bottom of the ladder. Both halves are checked here, because either one on its own
    would still look plausible on a screen.
    """
    ss, pc = store
    rows = _ladder(100)
    rows[0]["rsi14"] = None          # the name that WOULD have been the bottom of the ladder
    assert ss.insert_night(rows, d="20260821") == 100
    out = pc.run()

    assert out["metrics"]["rsi14"]["measured"] == 99
    assert out["metrics"]["rsi14"]["of"] == 100
    # S001 (rsi 2) is now the lowest MEASURED name, so it — and only it — sits at 0.0.
    assert _pct(ss, "S001", "rsi14_pctile") == 0.0
    assert _pct(ss, "S099", "rsi14_pctile") == 100.0
    zeros = ss._db().execute(
        "SELECT COUNT(*) n FROM scan WHERE d = ? AND rsi14_pctile = 0.0", ("20260821",)
    ).fetchone()["n"]
    assert zeros == 1, "exactly one name is the lowest measured — an unmeasured row must not join it"


def test_a_row_missing_one_metric_still_gets_its_other_percentiles(store):
    """"Unmeasured" is per metric, not per row: one absent number must not blank the whole name."""
    ss, pc = store
    rows = _ladder(100)
    rows[7]["rsi14"] = None
    rows[7]["mom_20d"] = 999.0       # measured, and the highest in the night
    assert ss.insert_night(rows, d="20260821") == 100
    assert pc.run()["status"] == "ok"

    assert _pct(ss, "S007", "rsi14_pctile") is None
    assert _pct(ss, "S007", "mom_20d_pctile") == 100.0


def test_a_metric_no_row_carries_is_unranked_for_everyone_rather_than_shared_out(store):
    """A night where SPY never fetched: rel_strength_3mo is None on every row.

    Ranking that would hand all 100 names the same fabricated percentile of each other. The pass
    reports the metric unranked, with a reason an operator can read in the run summary, and every
    column stays NULL.
    """
    ss, pc = store
    assert ss.insert_night(_ladder(100, rel_strength_3mo=None), d="20260821") == 100
    out = pc.run()

    assert "rel_strength_3mo" not in out["ranked"]
    rep = out["metrics"]["rel_strength_3mo"]
    assert (rep["measured"], rep["ranked"]) == (0, False)
    assert "0 of 100" in rep["reason"]
    filled = ss._db().execute(
        "SELECT COUNT(*) n FROM scan WHERE d = ? AND rel_strength_3mo_pctile IS NOT NULL",
        ("20260821",),
    ).fetchone()["n"]
    assert filled == 0


# ---- THE OTHER RULE: a sample is not a census

def test_a_metric_measured_on_too_few_of_the_nights_rows_is_ranked_for_nobody(store):
    """Below the coverage floor the metric is unranked for EVERY row — including the measurable ones.

    This is scan_store._MIN_COVERAGE's argument, one level down. 40 measurable rows of a 100-row
    night can be ranked against each other perfectly well; what they cannot be called is a market
    percentile, and "market" is the word the reader will supply. The 40 do not get a private
    percentile of themselves dressed up as the night's.
    """
    ss, pc = store
    rows = _ladder(100)
    for i in range(60):
        rows[i]["adx14"] = None      # 40 measurable of 100 — under half the night
    assert ss.insert_night(rows, d="20260821") == 100
    out = pc.run()

    assert "adx14" not in out["ranked"]
    assert out["metrics"]["adx14"]["measured"] == 40
    assert "coverage floor" in out["metrics"]["adx14"]["reason"]
    for sym in ("S060", "S099"):     # names that DID carry an adx14
        assert _pct(ss, sym, "adx14_pctile") is None


def test_a_metric_just_over_the_coverage_floor_is_ranked(store):
    """The other side of the same threshold, so the test above cannot pass by ranking nothing ever."""
    ss, pc = store
    rows = [_row(f"S{i:03d}", adx14=(float(i) if i >= 40 else None)) for i in range(100)]
    assert ss.insert_night(rows, d="20260821") == 100
    out = pc.run()

    assert "adx14" in out["ranked"]
    assert out["metrics"]["adx14"]["measured"] == 60
    assert _pct(ss, "S040", "adx14_pctile") == 0.0
    assert _pct(ss, "S099", "adx14_pctile") == 100.0
    assert _pct(ss, "S039", "adx14_pctile") is None


def test_a_night_with_one_row_does_not_produce_a_meaningless_hundredth_percentile(store):
    """One name is not a distribution, and it is certainly not the top of the market.

    Coverage is a perfect 100% here — one row measured of one row — so the coverage floor cannot
    catch this. The distribution floor is what does: below it there is no population to hold a
    position in, and the honest answer is that we do not know where this name sits.
    """
    ss, pc = store
    assert ss.insert_night([_row("AAPL")], d="20260821") == 1
    out = pc.run()

    assert out["status"] == "ok" and out["rows"] == 1
    assert out["ranked"] == []
    row = ss.symbol_row("AAPL")
    for col in ss.PCT_COLUMNS.values():
        assert row[col] is None, f"{col} was ranked over a single row"
    assert row["rsi14"] == 58.0, "the measurement itself is untouched"


def test_a_night_just_under_the_distribution_floor_is_not_ranked(store):
    """19 names cannot be ranked to one decimal place; 20 is where this module says they can.

    The floor is a judgement call, stated as one — its job is that the number stops being dominated
    by its own step size — so it is pinned on both sides here rather than left to drift.
    """
    ss, pc = store
    assert ss.insert_night(_ladder(19), d="20260820") == 19
    assert pc.run("20260820")["ranked"] == []
    assert _pct(ss, "S000", "rsi14_pctile", d="20260820") is None

    assert ss.insert_night(_ladder(20), d="20260821") == 20
    assert "rsi14" in pc.run("20260821")["ranked"]
    assert _pct(ss, "S000", "rsi14_pctile") == 0.0


# ---- ties

def test_tied_values_share_one_midrank_rather_than_being_split_by_row_order(store):
    """Identical measurements must produce identical percentiles, to the last decimal place.

    Any convention that breaks the tie — by symbol, by insertion order — invents an ordering the data
    does not contain and then prints it to one decimal place, which is fabricated precision in its
    purest form. The shared value is the MIDRANK: the centre of the block the tied values occupy,
    which is the only choice that answers the same way if every value's sign is flipped.
    """
    ss, pc = store
    # 100 names: ten at 5.0, ninety strictly above. The tied block occupies positions 0..9 of the
    # sorted night, so its midrank is 4.5 of 99 steps.
    rows = [_row(f"S{i:03d}", mom_20d=(5.0 if i < 10 else float(10 + i))) for i in range(100)]
    assert ss.insert_night(rows, d="20260821") == 100
    assert pc.run()["status"] == "ok"

    tied = [_pct(ss, f"S{i:03d}", "mom_20d_pctile") for i in range(10)]
    assert len(set(tied)) == 1, f"tied values were split apart: {sorted(set(tied))}"
    assert tied[0] == round(4.5 / 99.0 * 100.0, 1) == 4.5
    assert tied[0] != 0.0, "a tie at the bottom is not the bottom — nine names share the position"
    assert _pct(ss, "S099", "mom_20d_pctile") == 100.0


def test_a_metric_that_is_identical_across_the_whole_night_puts_everyone_at_the_middle(store):
    """The degenerate tie: nothing separates the names, so nobody leads and nobody trails.

    50.0 for all of them is the true reading — every name IS at the median of a distribution with no
    spread. It is recorded here so that the day this changes, it changes deliberately.
    """
    ss, pc = store
    assert ss.insert_night(_ladder(100, adx14=25.0), d="20260821") == 100
    assert pc.run()["status"] == "ok"
    assert _pct(ss, "S000", "adx14_pctile") == 50.0
    assert _pct(ss, "S099", "adx14_pctile") == 50.0


# ---- idempotence, and not disturbing the measurements

def test_running_the_pass_twice_changes_nothing(store):
    """It runs at the tail of every nightly job and again from any backfill; twice must equal once.

    It is idempotent by construction rather than by luck: every rank is derived from the measured
    columns and none of them from a percentile column, so the pass cannot feed on its own output.
    """
    ss, pc = store
    rows = _ladder(100)
    rows[3]["rsi14"] = None
    assert ss.insert_night(rows, d="20260821") == 100

    first = pc.run()
    snap = {r["symbol"]: dict(r) for r in ss.cross_section("20260821")}
    second = pc.run()
    again = {r["symbol"]: dict(r) for r in ss.cross_section("20260821")}

    assert first["ranked"] == second["ranked"]
    assert first["updated"] == second["updated"] == 100
    assert snap == again


def test_the_pass_does_not_disturb_any_measured_column(store):
    """The write names the ten percentile columns and nothing else, ever.

    UPDATE with an explicit column list, not the upsert `insert_night` uses — so a measurement cannot
    be nulled by a pass that has no opinion about it, and a symbol the night does not contain cannot
    be conjured into existence as an all-null row (checked below in its own test).
    """
    ss, pc = store
    rows = _ladder(100)
    assert ss.insert_night(rows, d="20260821") == 100
    before = {r["symbol"]: {k: v for k, v in r.items() if not k.endswith(ss.PCT_SUFFIX)}
              for r in ss.cross_section("20260821")}

    assert pc.run()["status"] == "ok"

    after = {r["symbol"]: {k: v for k, v in r.items() if not k.endswith(ss.PCT_SUFFIX)}
             for r in ss.cross_section("20260821")}
    assert before == after
    assert ss.stats()["rows"] == 100


def test_reinserting_the_night_keeps_the_percentiles_and_updates_the_measurements(store):
    """The ON CONFLICT guarantee, end to end, with the real percentile columns rather than a stand-in.

    `test_reinserting_a_night_preserves_a_column_written_after_the_original_insert` pins this with a
    synthetic column; this is the version that breaks if anyone ever "simplifies" insert_night back
    to INSERT OR REPLACE. Under REPLACE a manual re-scan of tonight silently drops every rank
    computed after the first run, and the reader downstream sees NULLs it will read as "not ranked".
    """
    ss, pc = store
    assert ss.insert_night(_ladder(100), d="20260821") == 100
    assert pc.run()["status"] == "ok"
    assert _pct(ss, "S099", "rsi14_pctile") == 100.0

    # The night is re-run — a retry, a forced re-measure — and S099's RSI comes back different.
    rerun = _ladder(100)
    rerun[99]["rsi14"] = 42.0
    assert ss.insert_night(rerun, d="20260821") == 100

    row = ss.symbol_row("S099")
    assert row["rsi14"] == 42.0, "the re-run's own measurement should win"
    assert row["rsi14_pctile"] == 100.0, "the rank was wiped — this is the REPLACE bug"

    # And the pass re-run afterwards brings the stale rank back in line with the new measurement:
    # 42.0 now ties S041, so both sit at the midrank of the two-name block (41.5 of 99 steps).
    assert pc.run()["status"] == "ok"
    assert ss.symbol_row("S099")["rsi14_pctile"] == 41.9
    assert ss.symbol_row("S041")["rsi14_pctile"] == 41.9


def test_a_rank_is_cleared_when_its_metric_stops_being_rankable(store):
    """Yesterday's standing must not sit in a row with tonight's date on it.

    The writer names all ten columns on every row precisely so this can happen: when the benchmark
    fetch fails and rel_strength_3mo falls to unmeasured across the night, the honest state is NULL,
    not the rank that was true the last time we could measure it.
    """
    ss, pc = store
    assert ss.insert_night(_ladder(100), d="20260821") == 100
    assert pc.run()["status"] == "ok"
    assert _pct(ss, "S000", "rel_strength_3mo_pctile") is not None

    assert ss.insert_night(_ladder(100, rel_strength_3mo=None), d="20260821") == 100
    out = pc.run()

    assert "rel_strength_3mo" not in out["ranked"]
    assert _pct(ss, "S000", "rel_strength_3mo_pctile") is None
    assert _pct(ss, "S000", "rsi14_pctile") == 0.0, "the metrics still measurable stay ranked"


def test_writing_a_rank_for_a_symbol_the_night_does_not_hold_creates_nothing(store):
    """An UPDATE that matches no row is a no-op, and that is the point.

    An upsert here would INSERT a row carrying nothing but ranks: a phantom name with no price, no
    bars and no measurements, sitting in the cross-section and counting toward breadth's coverage
    denominator, indistinguishable afterwards from a stock we tried and failed to measure.
    """
    ss, _pc = store
    assert ss.insert_night(_ladder(20), d="20260821") == 20
    wrote = ss.write_percentiles("20260821", {"GHOST": {"rsi14_pctile": 99.9}})

    assert wrote == 0
    assert ss.stats()["rows"] == 20
    assert ss.symbol_row("GHOST") is None


# ---- the pass's own reporting

def test_the_pass_reports_a_night_it_could_not_rank_rather_than_inventing_zeros(store):
    """"Nothing has been scanned", "that night is empty" and "that is not a date" are three answers.

    None of them is `rows: 0, updated: 0`, which reads off an ops page as a pass that ran over a
    night and ranked nothing — a claim about the data rather than about our failure to look.
    """
    ss, pc = store
    empty = pc.run()
    assert empty["status"] == "no_scan"
    assert empty["rows"] is None and empty["updated"] is None

    assert ss.insert_night(_ladder(20), d="20260821") == 20
    assert pc.run("not-a-date")["status"] == "refused"
    assert pc.run("20260101")["status"] == "empty"      # a real date this store holds no rows for
    assert pc.run("20260101")["rows"] == 0              # a genuine count, and it really is zero


def test_the_rank_helper_is_positional_and_survives_junk_values(store):
    """The pure function, directly: alignment with its input, and no crash on what SQLite can hold.

    A NaN that reached the sort would corrupt every comparison around it, and a bool is 1.0 to
    Python — either one silently joins the distribution as a measurement if it is not excluded.
    """
    _ss, pc = store
    vals = [1.0, None, 2.0, float("nan"), 3.0, True, "x"]
    # Both gates are set aside deliberately — four of these seven are junk, which is exactly what the
    # coverage floor exists to refuse, and it has its own tests above. What is under test here is
    # alignment and the classification of each value.
    out = pc.rank(vals, min_measured=2, min_coverage=0.0)

    assert len(out) == len(vals)
    assert out[0] == 0.0 and out[2] == 50.0 and out[4] == 100.0
    assert out[1] is None and out[3] is None and out[5] is None and out[6] is None
    assert pc.rank([]) == []
    assert pc.rank([5.0], min_measured=1) == [None], "one point is not a distribution"


# ---- the routes

def test_the_symbol_route_returns_percentiles_alongside_the_raw_row(env):
    """Both halves, always: the rank makes the raw number legible and the raw number is what it ranks.

    Before the pass has run the keys are PRESENT and null — never absent. An absent key is what a
    client defaults to zero, and zero here is "the worst name in the market".
    """
    client, ss, _pc = env
    assert ss.insert_night(_ladder(100), d="20260821") == 100

    body = client.get("/market_scan/S099").json()
    assert body["percentiles"]["rsi14"] is None, "not ranked yet — and null, not 0"
    assert set(body["percentiles"]) == set(ss.PCT_COLUMNS)
    assert body["percentiles_over"] == 100
    assert body["row"]["rsi14"] == 100.0

    ran = client.post("/market_scan/percentiles").json()
    assert ran["status"] == "ok" and ran["updated"] == 100

    body = client.get("/market_scan/S099").json()
    assert body["percentiles"]["rsi14"] == 100.0
    assert body["row"]["rsi14_pctile"] == 100.0
    assert body["row"]["rsi14"] == 100.0, "the measurement is still there next to its rank"


def test_the_list_route_carries_the_percentiles_and_the_population_they_rank_against(env):
    """A percentile of the SLICE would be a different number, so the denominator is published."""
    client, ss, _pc = env
    assert ss.insert_night(_ladder(100), d="20260821") == 100
    client.post("/market_scan/percentiles")

    body = client.get("/market_scan?sort=rsi14&limit=5").json()
    assert body["percentiles_over"] == 100, "ranked against the night, not against the five shown"
    assert body["results"][0]["rsi14_pctile"] == 100.0
    assert body["percentile_columns"]["rsi14"] == "rsi14_pctile"

    # A filter narrows total_matching but must not narrow the population the ranks belong to.
    body = client.get("/market_scan?sort=rsi14&limit=5&min_rsi14=90").json()
    assert body["total_matching"] == 11 and body["percentiles_over"] == 100


def test_the_backfill_refuses_a_date_that_is_not_a_date_and_a_night_it_does_not_hold(env):
    """Expensive work is an explicit POST here, and a bad request is named rather than absorbed."""
    client, ss, _pc = env
    assert client.post("/market_scan/percentiles").status_code == 503   # nothing scanned at all

    assert ss.insert_night(_ladder(100), d="20260821") == 100
    assert client.post("/market_scan/percentiles?d=yesterday").status_code == 422
    assert client.post("/market_scan/percentiles?d=20260101").status_code == 404
    assert client.post("/market_scan/percentiles?d=2026-08-21").status_code == 200


def test_the_backfill_evicts_the_slice_it_already_served_without_ranks(env):
    """The cache keys on the night, which a backfill does not change — so it must evict explicitly.

    Otherwise the ranked rows sit in the database while the route serves the pre-backfill payload,
    with `cached: true` stamped on it, for the rest of a 15-minute TTL.
    """
    client, ss, _pc = env
    assert ss.insert_night(_ladder(100), d="20260821") == 100

    first = client.get("/market_scan?sort=rsi14&limit=3").json()
    assert first["results"][0]["rsi14_pctile"] is None
    assert client.get("/market_scan?sort=rsi14&limit=3").json()["cached"] is True

    client.post("/market_scan/percentiles")
    after = client.get("/market_scan?sort=rsi14&limit=3").json()
    assert after["cached"] is False
    assert after["results"][0]["rsi14_pctile"] == 100.0


# ---- the nightly wiring, and the schema on a database older than the columns

@pytest.fixture()
def job(tmp_path, monkeypatch):
    """The nightly job over a throwaway data dir, with the pass wired in behind it."""
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    from app import scan_store as ss
    importlib.reload(ss)
    from app import percentiles as pc
    importlib.reload(pc)
    from app import market_scan_job as mj
    importlib.reload(mj)
    yield mj, ss
    if ss._conn is not None:
        ss._conn.close()
        ss._conn = None


def _fake_market(monkeypatch, mj, symbols: list[str]) -> None:
    """A published universe and a fetch that always answers with a clean 300-bar ramp."""
    import datetime as dt
    import time as _time

    from app.market import Series

    monkeypatch.setattr(mj.universe, "load",
                        lambda: {"built_at": _time.time(), "symbols": list(symbols)})

    async def fake(client, symbol, rng="1y", *, fallback=True):
        # Slope taken from the symbol's own number, so every name in the night is distinct and the
        # ranks below are exact. A hash would collide, and a collision here is a tie whose midrank is
        # correct but not 0.0 — a confusing way to fail a test that is about the wiring.
        digits = "".join(c for c in symbol if c.isdigit())
        seed = (int(digits) if digits else 0) / 100.0
        closes = [100.0 + (0.25 + seed) * i for i in range(300)]
        base = dt.date(2025, 1, 1)
        return Series(
            symbol=symbol, closes=closes, opens=list(closes), volumes=[1_000_000.0] * 300,
            dates=[(base + dt.timedelta(days=i)).strftime("%Y%m%d") for i in range(300)],
            fifty_two_high=max(closes), fifty_two_low=min(closes), currency="USD",
            highs=[c * 1.01 for c in closes], lows=[c * 0.99 for c in closes],
        )

    monkeypatch.setattr(mj, "fetch_series", fake)


def test_the_nightly_job_ranks_the_night_it_just_stored(job, monkeypatch):
    """The pass belongs at the tail of the job, in the job's own process, and this proves it is.

    Placement is the design, not a convenience: scan_store's single-writer invariant is the entire
    basis for there being no cross-process lock on scan.db, so the pass that writes derived columns
    is the sole writer doing one more thing before it exits — not a second writer appearing.
    """
    import asyncio

    mj, ss = job
    _fake_market(monkeypatch, mj, [f"S{i:03d}" for i in range(25)])

    out = asyncio.run(mj.run_market_scan())

    assert out["status"] == "ok" and out["rows_written"] == 25
    pct = out["percentiles"]
    assert pct["status"] == "ok" and pct["rows"] == 25 and pct["updated"] == 25
    assert "mom_20d" in pct["ranked"] and "rel_strength_3mo" in pct["ranked"]

    ranked = [r["mom_20d_pctile"] for r in ss.cross_section(out["session"])]
    assert all(v is not None for v in ranked)
    assert min(ranked) == 0.0 and max(ranked) == 100.0
    assert "percentiles ok 25/25" in mj.summary_line(out)


def test_a_run_that_stores_nothing_reports_no_percentile_pass_rather_than_zero_ranked(job, monkeypatch):
    """A refusal must not carry a rank summary that reads as a pass which ran and found nothing.

    Same rule as every other counter in `_blank_summary`: None says we never got that far, and 0
    would be a claim about the night.
    """
    import asyncio

    mj, _ss = job
    monkeypatch.setattr(mj.universe, "load", lambda: None)

    out = asyncio.run(mj.run_market_scan())
    assert out["status"] == "refused"
    assert out["percentiles"] is None
    assert "percentiles" not in mj.summary_line(out)


def test_a_database_older_than_these_columns_gains_them_on_open(store):
    """The new columns must be in _SCHEMA *and* in _migrate(), or they exist only on fresh installs.

    `CREATE TABLE IF NOT EXISTS` is a no-op against an existing scan.db, so a box that has been
    running since before SWT-4 would open a table with no percentile columns at all and every write
    into them would raise — on that machine only, at whatever hour the nightly job next ran.
    """
    ss, pc = store
    assert ss.insert_night(_ladder(25), d="20260821") == 25

    db = ss._db()
    for col in ss.PCT_COLUMNS.values():           # rewind the database to its pre-SWT-4 shape
        db.execute(f"ALTER TABLE scan DROP COLUMN {col}")
    db.commit()
    have = {r["name"] for r in db.execute("PRAGMA table_info(scan)")}
    assert not (have & set(ss.PCT_COLUMNS.values()))

    ss._conn.close()
    ss._conn = None                               # next _db() re-opens and migrates

    assert pc.run("20260821")["updated"] == 25
    assert _pct(ss, "S024", "rsi14_pctile") == 100.0
