"""SEC-2: the API has a shared-secret bearer-token gate in front of every non-GET route plus the
handful of GET routes that disclose the watchlist or the paper book (GET /api/settings, GET
/sandbox/*). Loopback callers — the systemd timer units that drive scans/sandbox ticks via
`curl http://127.0.0.1:8000/...`, see deploy/*.service — are exempt on the connection itself, since
they have no mechanism to be handed the secret. Startup refuses to run at all if SIGNALS_API_TOKEN is
unset, rather than silently coming up unauthenticated.

tests/conftest.py defaults every TestClient's fake peer to a loopback address (127.0.0.1) so the rest
of the suite — none of which sends a token — keeps passing. These tests are the ones that deliberately
override that default with a real (non-loopback) peer to exercise the gate itself, plus the startup
refusal.
"""
from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

_REMOTE = ("203.0.113.7", 54321)  # TEST-NET-3 (RFC 5737) — never a real loopback address
_TOKEN = "pytest-signals-token-do-not-use-in-prod"  # matches tests/conftest.py's default


@pytest.fixture()
def app_module(tmp_path, monkeypatch):
    """Reload app.main against an isolated data dir, matching tests/test_endpoints_state.py."""
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    import app.sandbox_store as ss
    import app.settings_store as st
    importlib.reload(st)
    importlib.reload(ss)
    import app.main as m
    importlib.reload(m)
    return m


@pytest.fixture()
def loopback_client(app_module):
    """Default TestClient peer (loopback), from conftest — a stand-in for the systemd curl calls."""
    with TestClient(app_module.app) as c:
        yield c


@pytest.fixture()
def remote_client(app_module):
    """A TestClient whose peer is a real, non-loopback address — a stand-in for a LAN/tailnet caller."""
    with TestClient(app_module.app, client=_REMOTE) as c:
        yield c


# ---------------------------------------------------------------- the dependency itself

def test_remote_caller_with_no_token_is_rejected(remote_client):
    r = remote_client.get("/sandbox/state")
    assert r.status_code == 401


def test_remote_caller_with_wrong_token_is_rejected(remote_client):
    r = remote_client.get("/sandbox/state", headers={"Authorization": "Bearer not-the-token"})
    assert r.status_code == 401


@pytest.mark.parametrize("header", [
    "not-even-bearer-shaped",
    "Bearer",              # scheme with nothing after it
    "Basic " + _TOKEN,      # right token, wrong scheme
    "bearer" + _TOKEN,      # missing the separating space
])
def test_remote_caller_with_malformed_header_is_rejected(remote_client, header):
    r = remote_client.get("/sandbox/state", headers={"Authorization": header})
    assert r.status_code == 401


def test_remote_caller_with_correct_token_is_accepted(remote_client):
    r = remote_client.get("/sandbox/state", headers={"Authorization": f"Bearer {_TOKEN}"})
    assert r.status_code == 200


def test_loopback_caller_needs_no_token_at_all(loopback_client):
    # This is the property the systemd timer units (deploy/*.service) depend on.
    r = loopback_client.get("/sandbox/state")
    assert r.status_code == 200


# ---------------------------------------------------------------- coverage across protected routes

@pytest.mark.parametrize("method,path,body", [
    ("get", "/api/settings", None),
    ("post", "/api/settings", {}),
    ("get", "/sandbox/state", None),
    ("get", "/sandbox/nav", None),
    ("get", "/sandbox/trades", None),
    ("get", "/sandbox/arms", None),
    ("get", "/sandbox/arms/nav", None),
    ("get", "/sandbox/settings", None),
    ("get", "/sandbox/inputs", None),
    ("get", "/sandbox/changes", None),
    ("post", "/sandbox/fund", {"amount": 100.0}),
    ("post", "/sandbox/reset", {"confirm": True}),
    ("post", "/sandbox/tick", {}),
    ("post", "/scan/run", {}),
    ("post", "/market_scan/run", {}),
    ("post", "/portfolio/review", {"holdings": []}),
    ("post", "/journal/replay", {}),
])
def test_protected_routes_reject_remote_callers_without_a_token(remote_client, method, path, body):
    r = getattr(remote_client, method)(path, json=body) if method == "post" else remote_client.get(path)
    assert r.status_code == 401, f"{method.upper()} {path} should require a token for a remote caller"


@pytest.mark.parametrize("path", [
    "/health",
    "/api/version",
    "/movers",
])
def test_unlisted_public_routes_stay_open_to_remote_callers(remote_client, path):
    # Sanity check on the flip side: routes deliberately NOT on the SEC-2 list (plain market data,
    # ops health) must not have been swept up by the change.
    r = remote_client.get(path)
    assert r.status_code != 401


# ---------------------------------------------------------------- fail-closed startup

def test_startup_refuses_when_token_is_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("SIGNALS_API_TOKEN", raising=False)
    import app.main as m
    importlib.reload(m)  # module import itself doesn't check; the check is in lifespan's startup
    try:
        with pytest.raises(RuntimeError, match="SIGNALS_API_TOKEN"):
            with TestClient(m.app):
                pass
    finally:
        # Restore a working module for any test collected after this one in the same process —
        # monkeypatch will re-set SIGNALS_API_TOKEN on teardown, but app.main itself must be
        # reloaded again once it is back for the module-level state to be sane.
        monkeypatch.undo()
        importlib.reload(m)
