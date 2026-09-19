"""MONEY-1 — /portfolio/rebalance must not be tax-blind.

`Holding` gained an optional `opened_at`: one ISO date per purchase lot behind a holding, synced from
the app's own dated lots. `RebalanceRequest.taxable_account` (default True) says whether the account
is even subject to capital-gains treatment at all. Together they drive
`sandbox_job.annotate_holding_period` over the priced snapshot's `positions`, exactly as the paper
sandbox already gets — this file pins that wiring end to end through the real endpoint, not just the
pure function tested in tests/test_skip_reasons_and_tax.py.

`rebalance_portfolio` (the LLM call) is replaced with a fake that captures the `portfolio` dict it was
handed, so each test can assert on the exact position row the analyst would have seen, without caring
about the model's own reply.
"""
from __future__ import annotations

import datetime as dt
import importlib

import pytest
from fastapi.testclient import TestClient

from app import sandbox_job

_NOW = dt.datetime.now(sandbox_job.ET)


def _iso_days_ago(days: int) -> str:
    return (_NOW - dt.timedelta(days=days)).date().isoformat()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Isolated app instance — same recipe as tests/test_contract_fixtures.py's `client` fixture."""
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    import app.sandbox_store as ss
    import app.settings_store as st
    importlib.reload(st)
    importlib.reload(ss)
    import app.main as m
    importlib.reload(m)

    class _Series:
        closes = [100.0] * 60

    async def fake_fetch(http, sym, *a, **k):
        return _Series()

    def fake_summarize(series, bench):
        return {"price": 200.0, "currency": "USD", "rsi14": 55.0, "pct_vs_sma50": 4.2}

    async def fake_crypto_context(client, symbol, daily_closes):
        return {}  # too-short series -> no long_term block, deterministic and offline

    monkeypatch.setattr(m, "fetch_series", fake_fetch)
    monkeypatch.setattr(m, "summarize", fake_summarize)
    monkeypatch.setattr(m.cycle, "crypto_context", fake_crypto_context)

    captured: dict = {}

    async def fake_rebalance_portfolio(portfolio, *, max_position_pct, deep=False):
        captured["portfolio"] = portfolio
        from app.analyst import RebalancePlan
        return RebalancePlan(summary="ok", moves=[]), {
            "model": "claude-haiku-4-5", "input_tokens": 1, "output_tokens": 1,
            "cache_read_tokens": 0, "cache_write_tokens": 0, "cost_usd": 0.0, "provider": "api",
        }

    monkeypatch.setattr(m, "rebalance_portfolio", fake_rebalance_portfolio)

    with TestClient(m.app) as c:
        c.captured = captured  # type: ignore[attr-defined]
        yield c


def _position(client, symbol: str) -> dict:
    return next(p for p in client.captured["portfolio"]["positions"] if p["symbol"] == symbol)


def test_a_dated_single_lot_holding_is_annotated_short_term(client):
    r = client.post("/portfolio/rebalance", json={
        "cash": 1000.0, "refresh": True, "max_position_pct": 25.0,
        "holdings": [{"symbol": "AAPL", "shares": 10.0, "avg_cost": 160.0,
                     "opened_at": [_iso_days_ago(4)]}],
    })
    assert r.status_code == 200, r.text
    pos = _position(client, "AAPL")
    assert pos["capital_gains"] == "short_term"
    assert pos["holding_days"] == 4
    assert pos["days_to_long_term"] == sandbox_job._LONG_TERM_DAYS - 4


def test_a_dated_holding_past_a_year_is_long_term_with_no_countdown(client):
    r = client.post("/portfolio/rebalance", json={
        "cash": 1000.0, "refresh": True, "max_position_pct": 25.0,
        "holdings": [{"symbol": "AAPL", "shares": 10.0, "avg_cost": 160.0,
                     "opened_at": [_iso_days_ago(400)]}],
    })
    assert r.status_code == 200, r.text
    pos = _position(client, "AAPL")
    assert pos["capital_gains"] == "long_term"
    assert "days_to_long_term" not in pos


def test_a_multi_lot_holding_spanning_the_boundary_is_mixed(client):
    r = client.post("/portfolio/rebalance", json={
        "cash": 1000.0, "refresh": True, "max_position_pct": 25.0,
        "holdings": [{"symbol": "AAPL", "shares": 10.0, "avg_cost": 160.0,
                     "opened_at": [_iso_days_ago(400), _iso_days_ago(10)]}],
    })
    assert r.status_code == 200, r.text
    pos = _position(client, "AAPL")
    assert pos["capital_gains"] == "mixed"
    assert pos["holding_days"] == 10  # the youngest lot leads the countdown


def test_one_unknown_lot_date_leaves_the_position_unannotated(client):
    """Never assume short- or long-term from silence: a migrated lot with no recorded date must not
    be silently ignored in favor of the one dated lot the code CAN see."""
    r = client.post("/portfolio/rebalance", json={
        "cash": 1000.0, "refresh": True, "max_position_pct": 25.0,
        "holdings": [{"symbol": "AAPL", "shares": 10.0, "avg_cost": 160.0,
                     "opened_at": [None, _iso_days_ago(4)]}],
    })
    assert r.status_code == 200, r.text
    pos = _position(client, "AAPL")
    assert "capital_gains" not in pos
    assert "holding_days" not in pos


def test_a_holding_with_no_opened_at_at_all_is_unannotated(client):
    """A legacy sync (pre-MONEY-1 app build) sends no dates at all -- must behave exactly like an
    unknown date, not crash and not fabricate a holding period."""
    r = client.post("/portfolio/rebalance", json={
        "cash": 1000.0, "refresh": True, "max_position_pct": 25.0,
        "holdings": [{"symbol": "AAPL", "shares": 10.0, "avg_cost": 160.0}],
    })
    assert r.status_code == 200, r.text
    pos = _position(client, "AAPL")
    assert "capital_gains" not in pos


def test_a_tax_advantaged_account_suppresses_annotation_even_with_known_dates(client):
    """The setting must actually gate the behavior, not just exist -- a dated lot that would
    otherwise be a clean 'short_term' must produce nothing at all once the account says it is
    tax-advantaged."""
    r = client.post("/portfolio/rebalance", json={
        "cash": 1000.0, "refresh": True, "max_position_pct": 25.0, "taxable_account": False,
        "holdings": [{"symbol": "AAPL", "shares": 10.0, "avg_cost": 160.0,
                     "opened_at": [_iso_days_ago(4)]}],
    })
    assert r.status_code == 200, r.text
    pos = _position(client, "AAPL")
    assert "capital_gains" not in pos
    assert "holding_days" not in pos


def test_taxable_account_defaults_true_when_the_field_is_omitted(client):
    """An app build that predates MONEY-1 sends no `taxable_account` at all -- it must keep getting
    the annotation, not silently lose it because the field is absent from the request body."""
    r = client.post("/portfolio/rebalance", json={
        "cash": 1000.0, "refresh": True, "max_position_pct": 25.0,
        "holdings": [{"symbol": "AAPL", "shares": 10.0, "avg_cost": 160.0,
                     "opened_at": [_iso_days_ago(4)]}],
    })
    assert r.status_code == 200, r.text
    assert _position(client, "AAPL")["capital_gains"] == "short_term"


def test_a_crypto_symbols_usd_suffix_is_stripped_to_match_the_priced_row(client):
    r = client.post("/portfolio/rebalance", json={
        "cash": 1000.0, "refresh": True, "max_position_pct": 25.0,
        "holdings": [{"symbol": "BTC-USD", "shares": 0.5, "avg_cost": 40000.0,
                     "opened_at": [_iso_days_ago(4)]}],
    })
    assert r.status_code == 200, r.text
    pos = _position(client, "BTC")
    assert pos["capital_gains"] == "short_term"


def test_raw_opened_at_never_reaches_the_prompt(client):
    """The paragraph in REBALANCE_SYSTEM only ever explains holding_days/capital_gains/
    days_to_long_term -- the raw per-lot date list is an implementation detail and must not leak
    into the payload the model reads."""
    r = client.post("/portfolio/rebalance", json={
        "cash": 1000.0, "refresh": True, "max_position_pct": 25.0,
        "holdings": [{"symbol": "AAPL", "shares": 10.0, "avg_cost": 160.0,
                     "opened_at": [_iso_days_ago(4)]}],
    })
    assert r.status_code == 200, r.text
    assert "opened_at" not in _position(client, "AAPL")
