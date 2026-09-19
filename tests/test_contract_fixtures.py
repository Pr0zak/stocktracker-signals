"""CI-2 — the missing link between the app's JSON contract and the backend that serves it.

The app declares ~432 `@SerialName` fields and decodes with `coerceInputValues = true`, so a field
this backend renames does not fail on the phone — it silently decodes as a numeric default (0.0 for
most money fields) and renders as a confident, wrong number. On this side, most routes return a bare
`dict`, so nothing here pins the shape either. Nothing connected the two contracts.

This file is that link, for the four routes where a silent zero would mislead someone about real
money: /health, /sandbox/state, /portfolio/review, /scan/latest. Each test below drives the REAL
route (through FastAPI's TestClient, or — for /scan/latest — the endpoint function itself, matching
tests/test_dip_rejects.py's existing convention) with deterministic, mocked inputs, and writes its
actual JSON response to tests/fixtures/contract/<route>.json. The values are chosen so the fields
that matter are numerically distinct from one another (e.g. total_return_pct != vs_benchmark_pct !=
cash_pct), so a decoder that reads the wrong key produces a value the Android-side test can actually
catch, rather than two coincidentally-equal numbers passing by accident.

These fixtures are consumed on the OTHER side of the contract by the Android repo's
ContractFixtureDecodeTest, which decodes each one with the app's own `Http.json` instance and asserts
the specific values below. A field renamed on either side turns that Kotlin test red.

HOW TO REGENERATE — do this whenever a pinned route's response shape changes on purpose:

    cd /home/spider/stocktracker-signals
    .venv/bin/python -m pytest tests/test_contract_fixtures.py -q
    cp tests/fixtures/contract/*.json /home/spider/stocktracker/app/src/test/resources/contract/

(the second path is wherever the Android checkout/worktree actually lives on your machine — the
fixtures must land under app/src/test/resources/contract/ for ContractFixtureDecodeTest to find them
as classpath resources). Then update the expected values in
app/src/test/java/com/stocktracker/app/data/ContractFixtureDecodeTest.kt to match, and re-run both
suites.

Never hand-edit a fixture file — it must always be the literal output of one of these tests, or it
stops proving anything about what the backend actually sends.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "contract"


# Wall-clock values that a route stamps at response time. Frozen before writing, because a
# fixture that changes on every run is not a fixture: it rewrites itself during an ordinary test
# run, leaves the working tree dirty, and silently drifts away from the copy committed in the
# Android repo — so the contract these files exist to pin would quietly stop being pinned.
# Frozen to 2026-01-01T00:00:00Z. Add a field here only if a route genuinely stamps it with "now".
_FROZEN_EPOCH_SECONDS = 1767225600.0
# Only fields that genuinely vary run to run. `generated_at` is deliberately excluded: the route
# under test already stamps it with a fixed value, and freezing it here would overwrite a
# deterministic number with a different deterministic number for no reason.
_NON_DETERMINISTIC_FIELDS = ("created_at", "as_of", "as_of_ts", "ts")


def _freeze_timestamps(value):
    """Replace stamped-at-response-time fields with a fixed value, recursively."""
    if isinstance(value, dict):
        return {
            k: (_FROZEN_EPOCH_SECONDS if k in _NON_DETERMINISTIC_FIELDS and isinstance(v, (int, float))
                else _freeze_timestamps(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_freeze_timestamps(v) for v in value]
    return value


def _write_fixture(name: str, payload: dict) -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    frozen = _freeze_timestamps(payload)
    (FIXTURES_DIR / name).write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Isolated app instance, matching tests/test_endpoints_state.py. No quote mock baked in — each
    test wires the price double it needs, since the fixtures below deliberately use different marks
    at fund-time vs read-time to keep total_return_pct and vs_benchmark_pct from coinciding."""
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    import app.sandbox_store as ss
    import app.settings_store as st
    importlib.reload(st)
    importlib.reload(ss)
    import app.main as m
    importlib.reload(m)
    with TestClient(m.app) as c:
        yield c


# ---------------------------------------------------------------- GET /health

def test_write_health_fixture(tmp_path, monkeypatch):
    """No LLM key configured is the more interesting shape (key_configured: false is where an app
    would wrongly show AI features as available), but the fixture pins a KNOWN key present so
    `key_configured` is checkably `true` rather than depending on this host's real environment."""
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fixture-not-a-real-key")
    monkeypatch.setenv("DEEP_MODEL", "claude-opus-4-8")
    monkeypatch.setenv("SCAN_MODEL", "claude-haiku-4-5")
    import app.sandbox_store as ss
    import app.settings_store as st
    importlib.reload(st)
    importlib.reload(ss)
    import app.main as m
    importlib.reload(m)

    with TestClient(m.app) as c:
        r = c.get("/health")
    assert r.status_code == 200
    body = r.json()

    assert body == {
        "ok": True,
        "key_configured": True,
        "deep_model": "claude-opus-4-8",
        "scan_model": "claude-haiku-4-5",
        "settings_source": "env",
    }
    _write_fixture("health.json", body)


# ---------------------------------------------------------------- GET /sandbox/state

def test_write_sandbox_state_fixture(client, monkeypatch):
    """One funded, one-holding paper account, marked at a DIFFERENT price than the benchmark was
    bought at — so total_return_pct (20.0), vs_benchmark_pct (9.09) and cash_pct (83.3) are three
    genuinely different numbers, and unrealized_pct (33.33) is a fourth. A decoder that swapped any
    two of these fields would fail on a mismatched value instead of passing by coincidence."""
    import app.main as m
    import app.sandbox_store as ss

    # The benchmark leg is bought at fund-time from whatever this quote double returns for ^GSPC.
    prices = {"^GSPC": 400.0}

    async def fake_quotes(http, syms):
        return {s: ({"price": prices[s]} if prices.get(s) is not None else None) for s in syms}

    monkeypatch.setattr(m.market_now, "fetch_quotes", fake_quotes)

    r = client.post("/sandbox/fund", json={"amount": 10000.0})
    assert r.status_code == 200, r.text

    blob = ss.get("main")
    blob["positions"] = [{"symbol": "AAPL", "shares": 10.0, "avg_cost": 150.0}]
    ss.save(blob, arm="main")

    # The market moved between funding and today's mark: AAPL up to 200, the S&P up 10% to 440.
    prices["AAPL"] = 200.0
    prices["^GSPC"] = 440.0

    r = client.get("/sandbox/state")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["cash"] == 10000.0
    assert body["equity"] == 12000.0
    assert body["positions_value"] == 2000.0
    assert body["funded_total"] == 10000.0
    assert body["total_return_pct"] == 20.0
    assert body["cash_pct"] == 83.3
    assert body["benchmark_value"] == 11000.0
    assert body["vs_benchmark_pct"] == 9.09
    assert body["stale_marks"] == []
    pos = body["positions"][0]
    assert pos["symbol"] == "AAPL"
    assert pos["shares"] == 10.0
    assert pos["avg_cost"] == 150.0
    assert pos["price"] == 200.0
    assert pos["value"] == 2000.0
    assert pos["unrealized_pct"] == 33.33

    _write_fixture("sandbox_state.json", body)


# ---------------------------------------------------------------- POST /portfolio/review

def test_write_portfolio_review_fixture(client, monkeypatch):
    """A one-holding, cash-heavy book. total_value (3000.0), cash_pct (33.3), weight_pct (66.7) and
    unrealized_gain_pct (25.0) are four distinct numbers, chosen so a field-swap on either side of
    the contract lands on a value the Kotlin test can tell is wrong.

    The LLM call (`review_portfolio`) is replaced with a real `analyst.PortfolioReview` instance —
    the actual pydantic response model, never a hand-typed dict — so the `review` block in the
    fixture is exactly what `.model_dump()` on that model produces. Pricing
    (`fetch_series`/`summarize`) is stubbed the same way tests/test_portfolio_snapshot.py already
    does it, so `_build_portfolio_snapshot` — the real, unmodified snapshot builder — computes every
    number in `portfolio` itself."""
    import app.main as m
    from app.analyst import PortfolioAction, PortfolioReview

    class _Series:
        closes = [100.0] * 60

    async def fake_fetch(http, sym, *a, **k):
        return _Series()

    def fake_summarize(series, bench):
        return {"price": 200.0, "currency": "USD", "rsi14": 55.0, "pct_vs_sma50": 4.2}

    async def fake_crypto_context(client, symbol, daily_closes):
        # /portfolio/review builds its snapshot with include_trend=True, which calls this for the
        # 200-week long-term block. Un-mocked it hits Yahoo for real (app/cycle.py's _weekly_max) —
        # {} is what a genuinely-too-short series already degrades to, so this keeps the fixture
        # deterministic and offline without changing what a short series would produce anyway.
        return {}

    monkeypatch.setattr(m, "fetch_series", fake_fetch)
    monkeypatch.setattr(m, "summarize", fake_summarize)
    monkeypatch.setattr(m.cycle, "crypto_context", fake_crypto_context)

    mock_review = PortfolioReview(
        health="Concentrated in one holding but healthy overall — no leverage, meaningful cash buffer.",
        concentration=["AAPL is 66.7% of the book — a single-name concentration risk."],
        actions=[PortfolioAction(
            symbol="AAPL", action="hold",
            reason="Up 25% with constructive momentum; no clear reason to trim yet.",
        )],
        cash_note="Deploy the idle cash gradually into the next high-conviction setup rather than all at once.",
    )
    mock_usage = {
        "model": "claude-haiku-4-5", "input_tokens": 842, "output_tokens": 96,
        "cache_read_tokens": 0, "cache_write_tokens": 0, "cost_usd": 0.00284, "provider": "api",
    }

    async def fake_review_portfolio(portfolio, *, cash, deep=False):
        return mock_review, mock_usage

    monkeypatch.setattr(m, "review_portfolio", fake_review_portfolio)

    r = client.post("/portfolio/review", json={
        "cash": 1000.0, "deep": False, "refresh": True,
        "holdings": [{"symbol": "AAPL", "shares": 10.0, "avg_cost": 160.0}],
    })
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["portfolio"]["total_value"] == 3000.0
    assert body["portfolio"]["cash_pct"] == 33.3
    pos = body["portfolio"]["positions"][0]
    assert pos["symbol"] == "AAPL"
    assert pos["value"] == 2000.0
    assert pos["weight_pct"] == 66.7
    assert pos["unrealized_gain_pct"] == 25.0
    assert body["review"]["actions"][0]["symbol"] == "AAPL"
    assert body["review"]["actions"][0]["action"] == "hold"
    assert body["model"] == "claude-haiku-4-5"
    assert body["cached"] is False

    _write_fixture("portfolio_review.json", body)


# ---------------------------------------------------------------- GET /scan/latest

def test_write_scan_latest_fixture(client, monkeypatch, tmp_path):
    """One qualified dip, one near-miss reject, one clean hold — built from the REAL scan_job dip
    logic (`_dip_tier`, `dip_verdicts`), the same helpers tests/test_dip_rejects.py already pins,
    rather than hand-typed reject/count numbers. Only the bookkeeping fields a nightly run would set
    itself (symbol/signal/conviction/prev_signal/flipped) are supplied directly, matching the
    `_measured_row` convention tests/test_scan_job.py already uses for the same reason.

    The route itself (GET /scan/latest) is exercised for real over HTTP — its `scan_available: true`
    stamp and its null-defaulting of `dip_rejects`/`dip_counts` on an old file are both real
    behaviour of app/main.py's `scan_latest()`, not reproduced here."""
    from app import scan_job

    aapl_tier = scan_job._dip_tier(
        closes=[100.0] * 62 + [88.0], pct_off_52w=-14.0, below_200wma=False, weekly_oversold=False,
    )
    msft_tier = scan_job._dip_tier(
        closes=[100.0] * 62 + [95.8], pct_off_52w=-6.0, below_200wma=False, weekly_oversold=False,
    )
    googl_tier = scan_job._dip_tier(
        closes=[100.0] * 63, pct_off_52w=1.0, below_200wma=False, weekly_oversold=False,
    )

    results = [
        {"symbol": "AAPL", "signal": "buy", "conviction": 78, "squeeze": "fuel",
         "prev_signal": "hold", "flipped": True, "squeeze_changed": True,
         "below_200wma": False, "crossed_below_200wma": False, **aapl_tier},
        {"symbol": "MSFT", "signal": "hold", "conviction": 52, "squeeze": None,
         "prev_signal": "hold", "flipped": False, "squeeze_changed": False,
         "below_200wma": False, "crossed_below_200wma": False, **msft_tier},
        {"symbol": "GOOGL", "signal": "hold", "conviction": 50, "squeeze": None,
         "prev_signal": "hold", "flipped": False, "squeeze_changed": False,
         "below_200wma": False, "crossed_below_200wma": False, **googl_tier},
    ]
    dip_rejects, dip_counts = scan_job.dip_verdicts(results)
    assert aapl_tier["dip"] == "pullback_10", "premise: AAPL must actually qualify for this fixture"
    assert dip_counts["qualified"] == 1 and dip_counts["scanned"] == 3

    payload = {
        "generated_at": 1_800_000_000.0,
        "results": results,
        "flips": ["AAPL"],
        "crossed_below_200wma": [],
        "dip_alerts": [{"symbol": "AAPL", "dip": "pullback_10",
                         "pct_off_recent_high": aapl_tier["pct_off_recent_high"],
                         "pct_off_52w_high": -14.0}],
        "date_alerts": [],
        "dip_rejects": dip_rejects,
        "dip_counts": dip_counts,
        "total_cost_usd": 0.0142,
    }

    import app.main as m
    # `main.LATEST` is NOT under SIGNALS_DATA_DIR (see app/scan_job.py's docstring on why the scan
    # cross-section lives in its own file) — tests/test_dip_rejects.py and tests/test_scan_job.py
    # both point it at an isolated tmp_path file rather than the real repo data/ dir, so this does
    # the same instead of writing into this checkout's actual data/scan_latest.json.
    scan_path = tmp_path / "scan_latest_fixture_source.json"
    scan_path.write_text(json.dumps(payload))
    monkeypatch.setattr(m, "LATEST", scan_path)

    r = client.get("/scan/latest")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["scan_available"] is True
    assert body["dip_counts"] == {"scanned": 3, "qualified": 1, "near_miss": 1, "nowhere_near": 1, "unmeasured": 0}
    aapl_row = next(row for row in body["results"] if row["symbol"] == "AAPL")
    assert aapl_row["dip"] == "pullback_10"
    assert aapl_row["dip_measured"] is True
    msft_row = next(row for row in body["results"] if row["symbol"] == "MSFT")
    assert msft_row["dip"] is None
    assert msft_row["dip_near_miss"] is True
    near_miss_symbols = {row["symbol"] for row in body["dip_rejects"]["near_miss"]}
    assert near_miss_symbols == {"MSFT"}

    _write_fixture("scan_latest.json", body)
