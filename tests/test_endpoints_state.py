"""Endpoint-level tests for the routes that MUTATE persisted state.

Measured 2026-07-28: 51 routes declared, 4 exercised by any TestClient call. The pure logic under
most of them is well covered, but the orchestration layer — validation, locking, what actually lands
in the store, and the payload shape the app decodes — was not exercised at all. These are the routes
where a defect costs the user their paper ledger or silently ignores a setting they changed.

Every test runs against an ISOLATED SIGNALS_DATA_DIR (both stores honour it), so nothing here can
touch real data.
"""
import importlib
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    # Re-import the stores so they pick up the isolated dir, then main which binds them.
    import app.sandbox_store as ss
    import app.settings_store as st
    importlib.reload(st)
    importlib.reload(ss)
    import app.main as m
    importlib.reload(m)

    async def fake_quotes(http, syms):
        return {s: {"price": 100.0} for s in syms}
    monkeypatch.setattr(m.market_now, "fetch_quotes", fake_quotes)
    with TestClient(m.app) as c:
        yield c


# ---------------------------------------------------------------- destructive routes

def test_reset_without_confirm_is_refused(client):
    # The one that destroys the paper book. It must not fire on a bare POST.
    assert client.post("/sandbox/reset", json={}).status_code == 422
    assert client.post("/sandbox/reset", json={"confirm": False}).status_code == 422


def test_reset_with_confirm_zeroes_the_book(client):
    client.post("/sandbox/fund", json={"amount": 5000.0})
    r = client.post("/sandbox/reset", json={"confirm": True})
    assert r.status_code == 200
    assert r.json()["cash"] == 0.0 and r.json()["funded_total"] == 0.0
    assert client.get("/sandbox/state").json()["cash"] == 0.0


def test_funding_zero_is_refused_rather_than_a_silent_no_op(client):
    assert client.post("/sandbox/fund", json={"amount": 0}).status_code == 422


def test_funding_accumulates_and_is_readable_back(client):
    client.post("/sandbox/fund", json={"amount": 1000.0})
    client.post("/sandbox/fund", json={"amount": 500.0})
    st = client.get("/sandbox/state").json()
    assert st["cash"] == pytest.approx(1500.0)
    assert st["funded_total"] == pytest.approx(1500.0)


def test_a_withdrawal_reduces_cash(client):
    client.post("/sandbox/fund", json={"amount": 1000.0})
    client.post("/sandbox/fund", json={"amount": -400.0})
    assert client.get("/sandbox/state").json()["cash"] == pytest.approx(600.0)


# ---------------------------------------------------------------- settings round-trip

def test_a_sandbox_setting_actually_persists(client):
    # This is where the encodeDefaults bug lived: the POST returned 200 while changing nothing.
    # `X or True` is always true — that line asserted nothing. Assert the real precondition instead:
    # the flag must start off, or "turning it on" proves nothing.
    assert client.get("/sandbox/settings").json()["master_enabled"] is False
    client.post("/sandbox/settings", json={"master_enabled": True})
    assert client.get("/sandbox/settings").json()["master_enabled"] is True
    client.post("/sandbox/settings", json={"master_enabled": False})
    assert client.get("/sandbox/settings").json()["master_enabled"] is False


def test_switching_a_setting_back_is_not_swallowed(client):
    # Turning something OFF is the direction that silently failed before.
    client.post("/sandbox/settings", json={"avoid_wash_sales": True})
    assert client.get("/sandbox/settings").json()["avoid_wash_sales"] is True
    client.post("/sandbox/settings", json={"avoid_wash_sales": False})
    assert client.get("/sandbox/settings").json()["avoid_wash_sales"] is False


def test_an_unrelated_setting_survives_a_partial_patch(client):
    client.post("/sandbox/settings", json={"avoid_wash_sales": True})
    client.post("/sandbox/settings", json={"master_enabled": True})
    s = client.get("/sandbox/settings").json()
    assert s["avoid_wash_sales"] is True and s["master_enabled"] is True


# ---------------------------------------------------------------- read routes the app decodes

def test_health_reports_the_configured_models(client):
    b = client.get("/health").json()
    assert b["ok"] is True
    assert "deep_model" in b and "scan_model" in b


def test_state_nav_and_trades_are_shaped_as_the_app_expects(client):
    st = client.get("/sandbox/state").json()
    for k in ("cash", "equity", "positions", "funded_total"):
        assert k in st, f"/sandbox/state lost {k}, which the app decodes"
    assert isinstance(client.get("/sandbox/nav").json(), (list, dict))
    assert isinstance(client.get("/sandbox/trades").json(), (list, dict))


def test_no_secret_reaches_the_settings_payload(client):
    # Public repo, public endpoints: a token must never be echoed back in full.
    body = client.get("/api/settings").text
    for leak in ("sk-ant", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
        assert leak not in body, f"{leak} appeared in /api/settings"
