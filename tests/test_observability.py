"""Offline unit tests for app.observability — the pure ops/transparency helpers.

Everything here is deterministic and network-free: the request ring, status_snapshot over a synthetic
scan_latest.json, cost_breakdown math over a synthetic usage.jsonl, iv_rank_progress counting, the
shorts-cache pruner, and next_scan_at. The live source probes get a single self-skipping smoke test.

    .venv/bin/python -m pytest tests/test_observability.py -v
    .venv/bin/python -m pytest tests/ -q -m "not live"   # offline only
"""
from __future__ import annotations

import asyncio
import calendar
import collections
import json
import os
import time

import httpx
import pytest

from app import observability as ob


# ======================================================================================
# Request ring buffer
# ======================================================================================

def test_ring_records_and_orders_newest_first(monkeypatch):
    monkeypatch.setattr(ob, "_requests", collections.deque(maxlen=200))
    ob.record_request("GET", "/a", 200, 1.2)
    ob.record_request("POST", "/b", 201, 3.4)
    ob.record_request("GET", "/c", 200, 5.6)

    r = ob.recent(10)
    assert [x["path"] for x in r] == ["/c", "/b", "/a"]      # newest first
    assert r[0]["method"] == "GET" and r[0]["status"] == 200
    assert all("ts" in x and "ms" in x for x in r)


def test_ring_honors_limit_and_maxlen(monkeypatch):
    monkeypatch.setattr(ob, "_requests", collections.deque(maxlen=3))
    for i in range(6):
        ob.record_request("GET", f"/p{i}", 200, float(i))
    # maxlen=3 keeps only the last three requests
    assert [x["path"] for x in ob.recent(10)] == ["/p5", "/p4", "/p3"]
    # limit trims further, still newest-first
    assert [x["path"] for x in ob.recent(2)] == ["/p5", "/p4"]


def test_recent_errors_filters_non_2xx_and_raised(monkeypatch):
    monkeypatch.setattr(ob, "_requests", collections.deque(maxlen=200))
    ob.record_request("GET", "/ok", 200, 1.0)
    ob.record_request("GET", "/missing", 404, 1.0)
    ob.record_request("GET", "/boom", 500, 1.0, error="kaboom")
    ob.record_request("GET", "/redir", 302, 1.0)          # 3xx is not an error

    errs = ob.recent_errors(10)
    assert [x["path"] for x in errs] == ["/boom", "/missing"]   # newest first, only >=400
    assert errs[0]["error"] == "kaboom"


def test_record_request_truncates_long_error(monkeypatch):
    monkeypatch.setattr(ob, "_requests", collections.deque(maxlen=5))
    ob.record_request("GET", "/x", 500, 1.0, error="e" * 999)
    assert len(ob.recent(1)[0]["error"]) == 200


def test_uptime_positive():
    assert ob.uptime_seconds() >= 0.0


def test_redact_strips_secret_query_params(monkeypatch):
    # Any diagnostic string that could reach the unauthenticated /api/logs must not carry a secret.
    monkeypatch.setattr(ob, "_requests", collections.deque(maxlen=5))
    ob.record_request("GET", "/x", 500, 1.0,
                      error="connect fail https://finnhub.io/q?token=SECRET123&crumb=ABC.def-42")
    err = ob.recent(1)[0]["error"]
    assert "SECRET123" not in err and "ABC.def-42" not in err
    assert "token=***" in err and "crumb=***" in err
    # _short (used for probe details) redacts too, and still respects the length cap.
    assert "apikey=***" in ob._short("boom apikey=zzz")


# ======================================================================================
# status_snapshot / _scan_summary
# ======================================================================================

def _write_scan(path, results, *, generated_at=1_700_000_000.0, total_cost=0.1234):
    path.write_text(json.dumps({"generated_at": generated_at, "results": results,
                                "total_cost_usd": total_cost}))


def test_scan_summary_counts_changed_and_errors(monkeypatch, tmp_path):
    scan = tmp_path / "scan_latest.json"
    _write_scan(scan, [
        {"symbol": "AAA", "signal": "strong_buy", "conviction": 8, "flipped": True},
        {"symbol": "BBB", "signal": "buy", "dip_new": True},
        {"symbol": "CCC", "signal": "hold"},
        {"symbol": "DDD", "signal": "sell", "crossed_below_200wma": True},
        {"symbol": "EEE", "signal": "strong_sell"},
        {"symbol": "FFF", "error": "not enough history"},
    ])
    monkeypatch.setattr(ob, "SCAN_LATEST", scan)
    monkeypatch.setattr(ob, "SHORTS_DIR", tmp_path / "shorts")   # absent -> zero footprint
    monkeypatch.setattr(ob, "IV_HISTORY", tmp_path / "iv.jsonl")

    snap = ob.status_snapshot()
    sc = snap["scan"]
    assert sc["counts"] == {"buy": 2, "hold": 1, "sell": 2}       # strong_* fold into buy/sell
    assert sc["count"] == 5                                        # error row excluded from count
    assert set(sc["changed"]) == {"AAA", "BBB", "DDD"}
    assert sc["errors"] == ["FFF"]
    assert "FFF" in sc["symbols"] and len(sc["symbols"]) == 6      # symbols include the error row
    assert sc["total_cost"] == 0.1234
    # snapshot shell
    assert snap["uptime_s"] >= 0 and snap["next_scan_at"] > snap["now"]
    assert set(snap["disk"]) == {"used", "total", "free", "pct"}
    assert snap["cache"] == {"shorts_bytes": 0, "shorts_files": 0, "iv_history_days_total": 0}


def test_scan_summary_tolerates_type_malformed_cost(monkeypatch, tmp_path):
    # No top-level total_cost_usd -> _scan_summary falls back to summing per-result cost_usd.
    # A row whose cost_usd is a string (valid JSON, wrong type) must degrade to 0, not 500.
    scan = tmp_path / "scan_latest.json"
    scan.write_text(json.dumps({"generated_at": 1_700_000_000.0, "results": [
        {"symbol": "AAA", "signal": "buy", "cost_usd": 0.02},
        {"symbol": "BBB", "signal": "hold", "cost_usd": "oops"},   # type-malformed -> counted as 0
    ]}))
    monkeypatch.setattr(ob, "SCAN_LATEST", scan)
    monkeypatch.setattr(ob, "SHORTS_DIR", tmp_path / "shorts")
    monkeypatch.setattr(ob, "IV_HISTORY", tmp_path / "iv.jsonl")
    snap = ob.status_snapshot()                                    # must not raise
    assert snap["scan"]["total_cost"] == 0.02


def test_status_snapshot_missing_scan_is_null_not_raise(monkeypatch, tmp_path):
    monkeypatch.setattr(ob, "SCAN_LATEST", tmp_path / "nope.json")
    monkeypatch.setattr(ob, "SHORTS_DIR", tmp_path / "shorts")
    monkeypatch.setattr(ob, "IV_HISTORY", tmp_path / "iv.jsonl")
    snap = ob.status_snapshot()
    assert snap["scan"]["generated_at"] is None
    assert snap["scan"]["counts"] == {"buy": 0, "hold": 0, "sell": 0}


def test_cache_footprint_sums_shorts_and_counts_iv(monkeypatch, tmp_path):
    shorts = tmp_path / "shorts"; shorts.mkdir()
    (shorts / "shvol_20260101.json").write_text("x" * 10)
    (shorts / "ftd_202512b.json").write_text("y" * 20)
    iv = tmp_path / "iv.jsonl"
    iv.write_text('{"symbol":"AAPL","date":"2026-07-01","atm_iv":0.4}\n'
                  '{"symbol":"MSFT","date":"2026-07-01","atm_iv":0.3}\n\nGARBAGE\n')
    monkeypatch.setattr(ob, "SHORTS_DIR", shorts)
    monkeypatch.setattr(ob, "IV_HISTORY", iv)
    fp = ob._cache_footprint()
    assert fp["shorts_files"] == 2 and fp["shorts_bytes"] == 30
    assert fp["iv_history_days_total"] == 2     # blank + corrupt lines skipped


# ======================================================================================
# cost_breakdown
# ======================================================================================

def _usage_row(kind, cost, ts, tin=100, tout=50):
    return json.dumps({"ts": ts, "kind": kind, "cost_usd": cost,
                       "input_tokens": tin, "output_tokens": tout})


def test_cost_breakdown_math(monkeypatch, tmp_path):
    now = time.time()
    lt = time.localtime(now)
    this_month = time.mktime((lt.tm_year, lt.tm_mon, 1, 12, 0, 0, 0, 0, -1))
    py, pm = (lt.tm_year, lt.tm_mon - 1) if lt.tm_mon > 1 else (lt.tm_year - 1, 12)
    last_month = time.mktime((py, pm, 15, 12, 0, 0, 0, 0, -1))

    usage = tmp_path / "usage.jsonl"
    usage.write_text("\n".join([
        _usage_row("scan", 0.01, this_month),
        _usage_row("scan", 0.02, this_month),
        _usage_row("deep", 0.50, this_month),
        _usage_row("signal", 0.03, last_month),   # counts to all-time, NOT month-to-date
        "not json",                               # tolerated
    ]) + "\n")
    monkeypatch.setattr(ob, "USAGE_FILE", usage)

    c = ob.cost_breakdown(now=now)
    assert c["by_kind"]["scan"] == {"calls": 2, "tokens": 300, "usd": 0.03}
    assert c["by_kind"]["deep"]["calls"] == 1 and c["by_kind"]["deep"]["usd"] == 0.5
    assert c["all_time_usd"] == 0.56
    assert c["month_to_date_usd"] == 0.53
    assert c["per_scan_avg_usd"] == 0.015
    assert c["per_deep_avg_usd"] == 0.5

    day = lt.tm_mday
    dim = calendar.monthrange(lt.tm_year, lt.tm_mon)[1]
    assert c["projected_month_usd"] == round(0.53 / day * dim, 6)


def test_cost_breakdown_empty_is_zero(monkeypatch, tmp_path):
    monkeypatch.setattr(ob, "USAGE_FILE", tmp_path / "absent.jsonl")
    c = ob.cost_breakdown()
    assert c["all_time_usd"] == 0.0 and c["month_to_date_usd"] == 0.0
    assert c["by_kind"] == {} and c["per_scan_avg_usd"] is None and c["per_deep_avg_usd"] is None


def test_cost_breakdown_tolerates_type_malformed_rows(monkeypatch, tmp_path):
    # Valid JSON, wrong types (string cost, string tokens, string ts) must be skipped, not 500.
    now = time.time()
    lt = time.localtime(now)
    this_month = time.mktime((lt.tm_year, lt.tm_mon, 1, 12, 0, 0, 0, 0, -1))
    usage = tmp_path / "usage.jsonl"
    usage.write_text("\n".join([
        _usage_row("scan", 0.01, this_month),
        json.dumps({"ts": this_month, "kind": "scan", "cost_usd": "N/A",         # bad cost
                    "input_tokens": 100, "output_tokens": 50}),
        json.dumps({"ts": "not-a-number", "kind": "deep", "cost_usd": 0.5,       # bad ts
                    "input_tokens": 10, "output_tokens": 5}),
        json.dumps({"ts": this_month, "kind": "signal", "cost_usd": 0.02,        # bad tokens
                    "input_tokens": "lots", "output_tokens": 5}),
    ]) + "\n")
    monkeypatch.setattr(ob, "USAGE_FILE", usage)
    c = ob.cost_breakdown(now=now)                                # must not raise
    # only the one clean row survives; the three malformed rows are dropped
    assert c["by_kind"] == {"scan": {"calls": 1, "tokens": 150, "usd": 0.01}}
    assert c["all_time_usd"] == 0.01


def test_cost_breakdown_splits_billed_vs_cli_notional(monkeypatch, tmp_path):
    now = time.time()
    lt = time.localtime(now)
    this_month = time.mktime((lt.tm_year, lt.tm_mon, 1, 12, 0, 0, 0, 0, -1))

    def row(kind, cost, provider):
        return json.dumps({"ts": this_month, "kind": kind, "cost_usd": cost,
                           "input_tokens": 10, "output_tokens": 5, "provider": provider})

    usage = tmp_path / "usage.jsonl"
    usage.write_text("\n".join([
        row("scan", 0.02, "api"),
        row("deep", 0.50, "api"),
        row("scan", 0.03, "cli"),
        row("scan", 0.04, "cli"),
    ]) + "\n")
    monkeypatch.setattr(ob, "USAGE_FILE", usage)

    c = ob.cost_breakdown(now=now)
    # headline money = real billed (API rows) only
    assert c["billed_usd"] == 0.52
    assert c["all_time_usd"] == 0.52
    assert c["month_to_date_usd"] == 0.52
    # CLI rows reported separately as notional, never folded into billed
    assert c["cli_notional_usd"] == 0.07
    assert c["by_provider"]["api"]["calls"] == 2 and c["by_provider"]["cli"]["calls"] == 2


def test_cost_breakdown_tolerates_unhashable_provider(monkeypatch, tmp_path):
    # A valid-JSON row with a non-string provider must be str-coerced, never 500 the card.
    usage = tmp_path / "usage.jsonl"
    usage.write_text(
        json.dumps({"kind": "scan", "cost_usd": 0.0, "provider": ["x"], "ts": 1_700_000_000}) + "\n"
        + json.dumps({"kind": "scan", "cost_usd": 0.01, "ts": 1_700_000_000}) + "\n"  # normal api row
    )
    monkeypatch.setattr(ob, "USAGE_FILE", usage)
    c = ob.cost_breakdown()                 # must not raise
    assert c["all_time_usd"] == 0.01        # the api row's real $; the list-provider row isn't billed


# ======================================================================================
# iv_rank_progress
# ======================================================================================

def test_iv_rank_progress_counts_per_symbol(monkeypatch, tmp_path):
    iv = tmp_path / "iv.jsonl"
    rows = [json.dumps({"symbol": "AAPL", "date": f"2026-06-{d:02d}", "atm_iv": 0.4}) for d in range(1, 26)]
    rows += [json.dumps({"symbol": "MSFT", "date": f"2026-06-{d:02d}", "atm_iv": 0.3}) for d in range(1, 6)]
    iv.write_text("\n".join(rows) + "\n")
    monkeypatch.setattr(ob, "IV_HISTORY", iv)

    p = ob.iv_rank_progress()
    assert p["target"] == 20
    assert p["symbols"] == {"AAPL": 25, "MSFT": 5}


def test_iv_rank_progress_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(ob, "IV_HISTORY", tmp_path / "absent.jsonl")
    assert ob.iv_rank_progress(target=15) == {"target": 15, "symbols": {}}


# ======================================================================================
# prune_shorts_cache
# ======================================================================================

def test_prune_removes_only_old_prunable_files(monkeypatch, tmp_path):
    shorts = tmp_path / "shorts"; shorts.mkdir()
    now = time.time()
    old = now - 100 * 86400      # older than the 90-day cutoff
    fresh = now - 3 * 86400

    def mk(name, size, mtime):
        p = shorts / name
        p.write_text("z" * size)
        os.utime(p, (mtime, mtime))
        return p

    mk("shvol_20250101.json", 100, old)     # prune
    mk("ftd_202501a.json", 50, old)         # prune
    mk("shvol_20260701.json", 40, fresh)    # keep (fresh)
    mk("si_AAPL.json", 10, old)             # keep (not a prunable prefix)
    mk("settings.json", 10, old)            # keep (protected name / wrong prefix)
    monkeypatch.setattr(ob, "SHORTS_DIR", shorts)

    res = ob.prune_shorts_cache(now=now)
    assert res["deleted_files"] == 2
    assert res["bytes_freed"] == 150
    assert res["max_age_days"] == 90
    remaining = sorted(p.name for p in shorts.iterdir())
    assert remaining == ["settings.json", "shvol_20260701.json", "si_AAPL.json"]


def test_prune_missing_dir_is_noop(monkeypatch, tmp_path):
    monkeypatch.setattr(ob, "SHORTS_DIR", tmp_path / "absent")
    assert ob.prune_shorts_cache() == {"deleted_files": 0, "bytes_freed": 0, "max_age_days": 90}


# ======================================================================================
# next_scan_at
# ======================================================================================

def test_next_scan_at_is_upcoming_0630():
    now = time.time()
    nxt = ob.next_scan_at(now)
    assert 0 < nxt - now <= 86400 + 3600     # within a day (allow DST slack)
    tz = ob._scan_tz()
    if tz is not None:
        import datetime as dt
        local = dt.datetime.fromtimestamp(nxt, tz)
        assert (local.hour, local.minute) == (ob.SCAN_HOUR, ob.SCAN_MINUTE)


# ======================================================================================
# Live source probes (network — self-skips)
# ======================================================================================

@pytest.mark.live
def test_probe_sources_live_shape():
    async def run():
        async with httpx.AsyncClient(follow_redirects=True) as client:
            return await ob.probe_sources(client)

    try:
        sources = asyncio.run(run())
    except Exception as e:  # noqa: BLE001 — never fail the suite on a network hiccup
        pytest.skip(f"live source probe unavailable: {e}")

    assert isinstance(sources, list) and sources
    names = {s["name"] for s in sources}
    assert "Yahoo options (crumb)" in names       # the no-network crumb source is always present
    for s in sources:
        assert set(s) >= {"name", "status", "latency_ms", "detail"}
        assert s["status"] in ("ok", "warn", "down")
