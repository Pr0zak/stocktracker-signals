"""The four HTTP routes over the nightly market-wide cross-section (SWT-1).

The engine underneath these was proven against live data on 2026-08-21 — 3,113 of 3,147 symbols
scored in 49.9s — and app/scan_store.py already has a test pinning the one thing that matters most
about it: with nothing scanned, `breadth()` reports `available: False` and None for every reading,
never 0.0. That guarantee is worth nothing if the ROUTE in front of it flattens the Nones on the way
out, defaults them, or wraps them in an envelope that a client reads as a market print. A breadth
reading of "0% of the market is above its 50-day average" is the most bearish single number this
service can emit, and "we did not scan" is not that number. Half of this file exists to keep the
route honest about it.

The rest guards the failures a ranked slice of ~3,100 rows invites. A `limit` that is not capped
hands a phone the whole night. A cache key that omits the scan's date serves last night's ranking
after tonight's scan lands — the same shape of near-miss GET /regime still has, where `count` is
accepted and never keyed on. A filter name with a typo that is silently ignored returns a LONGER
list than was asked for and presents it as filtered. A `total_matching` read off `len(results)` can
only ever report the cap back. And provenance copied from a run summary that describes a DIFFERENT
night attaches one night's counters to another night's numbers, which reads as confirmation.

Isolation is SIGNALS_DATA_DIR plus importlib.reload(), in dependency order — scan_store, then
market_scan_job which binds it, then main which binds both. Setting the env var without reloading
gets the previous test's database.
"""
from __future__ import annotations

import importlib
import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """A TestClient over an isolated data dir, plus the store module the test seeds rows with."""
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    import app.scan_store as ss
    import app.settings_store as st
    importlib.reload(st)
    importlib.reload(ss)
    import app.market_scan_job as job
    importlib.reload(job)
    import app.main as m
    importlib.reload(m)
    m._cache.clear()
    with TestClient(m.app) as c:
        yield c, ss, job, m
    if ss._conn is not None:
        ss._conn.close()
        ss._conn = None


def _row(symbol: str, **over) -> dict:
    """A row shaped the way swing.metrics() delivers one — every key present, None where unmeasured."""
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


def _seed(store, d: str, n: int = 5, **over) -> None:
    rows = [_row(f"S{i:02d}", rel_strength_3mo=float(i), **over) for i in range(n)]
    assert store.insert_night(rows, d=d) == n


def _summary(job, *, session: str, **over) -> None:
    """Write the run summary the routes read provenance out of."""
    blob = {
        "job": "market_scan", "status": "ok", "reason": None,
        "generated_at": 1_760_000_000.0, "session": session,
        "universe_built_at": 1_759_990_000.0, "universe_symbols": 3147,
        "scanned": 3113, "fetch_failed": 0, "too_short": 34,
    }
    blob.update(over)
    job.LATEST.parent.mkdir(parents=True, exist_ok=True)
    job.LATEST.write_text(json.dumps(blob))


# ------------------------------------------------------- the payload the Android client decodes

def test_the_ranked_slice_carries_every_envelope_key_the_client_decodes(env):
    """A client cannot treat provenance as optional, so every key ships on every 200."""
    client, store, job, _m = env
    _seed(store, "20260821", n=5)
    _summary(job, session="20260821")

    body = client.get("/market_scan?limit=2").json()
    assert set(body) >= {
        "as_of", "generated_at", "universe_size", "scanned", "fetch_failed", "too_short",
        "universe_stale", "universe_built_at", "sort", "limit", "total_matching", "results",
        "note", "cached", "cached_age_seconds",
    }
    assert body["as_of"] == "20260821"
    assert body["universe_size"] == 3147 and body["scanned"] == 3113 and body["too_short"] == 34
    assert body["fetch_failed"] == 0
    assert body["cached"] is False and body["cached_age_seconds"] == 0
    assert len(body["results"]) == 2
    assert {"symbol", "price", "rel_strength_3mo", "above_sma200", "d"} <= set(body["results"][0])


def test_the_slice_is_ranked_by_the_requested_sort_and_says_what_it_is_a_slice_of(env):
    """`total_matching` is the count of matching rows, not the count that fitted under `limit`."""
    client, store, job, _m = env
    _seed(store, "20260821", n=9)
    _summary(job, session="20260821")

    body = client.get("/market_scan?limit=3&sort=rel_strength").json()
    assert [r["symbol"] for r in body["results"]] == ["S08", "S07", "S06"]
    assert body["total_matching"] == 9
    assert "Top 3 of 9 matching" in body["note"]
    assert "3,113 scored of 3,147" in body["note"]
    # The caveat travels with the payload, not just the docstring.
    assert "NOT A BUY SIGNAL" in body["note"].upper()


def test_a_filter_narrows_the_total_as_well_as_the_results(env):
    """A filtered slice that reports the unfiltered total describes the wrong population."""
    client, store, job, _m = env
    store.insert_night(
        [_row("AAA", adx14=40.0), _row("BBB", adx14=10.0), _row("CCC", adx14=5.0)], d="20260821")
    _summary(job, session="20260821")

    body = client.get("/market_scan?min_adx14=20").json()
    assert [r["symbol"] for r in body["results"]] == ["AAA"]
    assert body["total_matching"] == 1


def test_the_limit_is_capped_so_the_whole_night_can_never_be_requested(env):
    """~3,100 rows is a bulk export; `limit` is the only thing standing between it and a phone."""
    client, store, job, _m = env
    _seed(store, "20260821", n=250)
    _summary(job, session="20260821")

    body = client.get("/market_scan?limit=100000").json()
    assert body["limit"] == 200
    assert len(body["results"]) == 200
    assert body["total_matching"] == 250      # the cap trims the payload, not the reported truth


# ------------------------------------------------------------------------------ caller mistakes

def test_an_unknown_filter_is_refused_rather_than_silently_ignored(env):
    """A typo'd filter that does nothing returns a longer list and calls it filtered."""
    client, store, job, _m = env
    _seed(store, "20260821")
    _summary(job, session="20260821")

    r = client.get("/market_scan?min_adx=20")
    assert r.status_code == 422
    assert "min_adx" in r.json()["detail"]


def test_an_unknown_sort_is_refused_rather_than_falling_back_to_an_empty_ranking(env):
    """scan_store.top() answers a bad sort with [], which on a screen reads as 'nothing matched'."""
    client, store, job, _m = env
    _seed(store, "20260821")
    _summary(job, session="20260821")

    assert client.get("/market_scan?sort=moonshot").status_code == 422


def test_a_non_finite_filter_bound_is_refused(env):
    """float('nan') parses, and a NaN bound makes every comparison false — an empty cross-section
    produced by a typo is shaped exactly like a real one."""
    client, store, job, _m = env
    _seed(store, "20260821")
    _summary(job, session="20260821")

    assert client.get("/market_scan?min_adx14=nan").status_code == 422
    assert client.get("/market_scan?above_sma200=maybe").status_code == 422


def test_a_scan_that_has_never_run_is_a_503_and_not_an_empty_cross_section(env):
    """An empty ranking rendered as a market reading is the defect this service keeps correcting."""
    client, _store, _job, _m = env
    r = client.get("/market_scan")
    assert r.status_code == 503
    assert "market_scan/run" in r.json()["detail"]


# ----------------------------------------------------------------------------------- provenance

def test_counters_from_a_run_that_describes_another_night_are_not_borrowed(env):
    """A refused run today writes today's summary while yesterday's rows are still served.

    Attaching today's counters to yesterday's numbers is provenance for the wrong data, which is
    worse than none at all because it reads as confirmation.
    """
    client, store, job, _m = env
    _seed(store, "20260820")
    _summary(job, session="20260821", status="refused", scanned=None, universe_symbols=None)

    body = client.get("/market_scan").json()
    assert body["as_of"] == "20260820"
    assert body["scanned"] is None and body["universe_size"] is None
    assert body["universe_built_at"] is None and body["universe_stale"] is None
    assert "run summary is unavailable" in body["note"]


# --------------------------------------------------------------------------------------- cache

def test_a_repeat_call_is_served_from_cache_and_says_how_old_it_is(env):
    """Say WHEN it is cached, not just that it is."""
    client, store, job, _m = env
    _seed(store, "20260821")
    _summary(job, session="20260821")

    assert client.get("/market_scan").json()["cached"] is False
    again = client.get("/market_scan").json()
    assert again["cached"] is True and isinstance(again["cached_age_seconds"], int)


def test_the_cache_key_changes_when_the_scan_date_changes(env):
    """A fresh scan must invalidate on its own: the key carries the night it describes."""
    client, store, job, _m = env
    _seed(store, "20260820", n=3)
    _summary(job, session="20260820")
    first = client.get("/market_scan").json()
    assert first["as_of"] == "20260820"

    _seed(store, "20260821", n=4)
    _summary(job, session="20260821")
    second = client.get("/market_scan").json()
    assert second["as_of"] == "20260821"
    assert second["cached"] is False
    assert second["total_matching"] == 4


def test_the_cache_key_separates_callers_asking_different_questions(env):
    """The near-miss to avoid: GET /regime accepts `count` and never keys on it, so two callers
    with different counts share one entry."""
    client, store, job, _m = env
    _seed(store, "20260821", n=9)
    _summary(job, session="20260821")

    assert len(client.get("/market_scan?limit=2").json()["results"]) == 2
    assert len(client.get("/market_scan?limit=5").json()["results"]) == 5
    assert client.get("/market_scan?sort=adx14").json()["cached"] is False
    assert client.get("/market_scan?min_adx14=1").json()["cached"] is False


# -------------------------------------------------------------------------------------- breadth

def test_breadth_with_no_scan_reports_unavailable_and_never_zero(env):
    """THE one that matters. 0% above the 50-day average is the most bearish print there is, and it
    is not what "we did not scan" means. scan_store pins this; the route must preserve it."""
    client, _store, _job, _m = env
    body = client.get("/market_scan/breadth").json()

    assert body["available"] is False
    for k in ("pct_above_sma50", "pct_above_sma200", "advancers", "decliners",
              "new_52w_highs", "near_52w_high", "new_52w_lows", "age_hours", "as_of"):
        assert body[k] is None, f"{k} must be null when there is no scan, got {body[k]!r}"
    assert body["n"] == 0        # a genuine count of the rows we hold, which really is zero
    # Provenance is null too, rather than borrowed from a summary describing a night we cannot serve.
    assert body["scanned"] is None and body["universe_size"] is None


def test_breadth_with_a_scan_reports_the_readings_and_its_provenance(env):
    """SWT-2 decodes this envelope; the keys are the contract."""
    client, store, job, _m = env
    store.insert_night(
        [_row("AAA"), _row("BBB", above_sma50=False, above_sma200=False)], d="20260821")
    _summary(job, session="20260821")

    body = client.get("/market_scan/breadth").json()
    assert body["available"] is True
    assert body["as_of"] == "20260821" and body["n"] == 2
    assert body["pct_above_sma50"] == 50.0 and body["pct_above_sma200"] == 50.0
    assert body["new_52w_lows"] is None        # no low-side metric exists — None, not 0
    assert body["scanned"] == 3113 and body["too_short"] == 34
    assert body["cached"] is False and body["cached_age_seconds"] == 0
    assert client.get("/market_scan/breadth").json()["cached"] is True


# ------------------------------------------------------------------------------- one symbol

def test_a_symbol_that_was_never_scanned_is_a_404_naming_it(env):
    """"We looked and there is no data for this symbol" is the documented 404 here."""
    client, store, job, _m = env
    _seed(store, "20260821")
    _summary(job, session="20260821")

    r = client.get("/market_scan/AAPL")
    assert r.status_code == 404
    assert "AAPL" in r.json()["detail"]


def test_a_symbol_row_says_which_night_it_is_from(env):
    """symbol_row() falls back to the name's most recent night, which may not be the latest scan —
    a three-day-old row must not be served as tonight's."""
    client, store, job, _m = env
    store.insert_night([_row("AAA")], d="20260818")
    _seed(store, "20260821", n=2)
    _summary(job, session="20260821")

    body = client.get("/market_scan/aaa").json()
    assert body["symbol"] == "AAA"
    assert body["as_of"] == "20260818"
    assert body["latest_scan_date"] == "20260821"
    assert body["is_latest_night"] is False
    # Provenance follows the ROW's night, so the counters for 20260821 are not lent to it.
    assert body["scanned"] is None
    assert body["row"]["price"] == 100.0

    fresh = client.get("/market_scan/S01").json()
    assert fresh["is_latest_night"] is True and fresh["scanned"] == 3113


# ------------------------------------------------------------------------------------ the run

def test_running_the_scan_is_awaited_inline_and_returns_the_summary(env):
    """Matching POST /scan/run and POST /macro/run: the response IS the run, not a promise."""
    client, _store, _job, m = env
    seen = {}

    async def fake_run(*, force=False, limit=None):
        seen["force"], seen["limit"] = force, limit
        return {"job": "market_scan", "status": "ok", "scanned": 7}

    m.market_scan_job.run_market_scan = fake_run
    try:
        body = client.post("/market_scan/run?force=true&limit=5").json()
    finally:
        importlib.reload(m.market_scan_job)
    assert body["status"] == "ok" and body["scanned"] == 7
    assert seen == {"force": True, "limit": 5}


def test_a_nonsense_limit_on_the_run_is_the_callers_mistake(env):
    client, _store, _job, _m = env
    assert client.post("/market_scan/run?limit=0").status_code == 422


def test_a_forced_rerun_invalidates_the_slice_cached_under_the_same_night(env):
    """The key carries the DATE, and force=true rewrites the same date with new numbers."""
    client, store, job, m = env
    _seed(store, "20260821", n=2)
    _summary(job, session="20260821")
    assert client.get("/market_scan").json()["total_matching"] == 2

    async def fake_run(*, force=False, limit=None):
        _seed(store, "20260821", n=6)
        return {"job": "market_scan", "status": "ok"}

    m.market_scan_job.run_market_scan = fake_run
    try:
        client.post("/market_scan/run?force=true")
    finally:
        importlib.reload(m.market_scan_job)
    after = client.get("/market_scan").json()
    assert after["cached"] is False and after["total_matching"] == 6
