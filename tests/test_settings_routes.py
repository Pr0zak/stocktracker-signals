"""HTTP-layer tests for OPS-3 (watchlist sync guard) and OPS-5 (settings_source) on POST /api/settings
and GET /health.

app/test_settings_store.py already pins the store-level logic (WatchlistSyncRefused semantics, atomic
write, .bak recovery, source() states). These tests pin the thin route layer on top: that a refusal at
the store becomes an HTTP 409 with a body an API client can act on, that the same-client and
replace=true exemptions reach through the route unharmed, and that /health's settings_source field is
actually wired to the store rather than hardcoded.

Isolation is SIGNALS_DATA_DIR plus importlib.reload(), matching tests/test_endpoints_state.py and
tests/test_gate_routes.py.
"""
from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    import app.sandbox_store as ss
    import app.settings_store as st
    importlib.reload(st)
    importlib.reload(ss)
    import app.main as m
    importlib.reload(m)
    with TestClient(m.app) as c:
        yield c


def test_cross_client_mass_removal_is_refused_with_409(client):
    client.post("/api/settings", json={"watchlist": [f"S{i}" for i in range(54)], "client_id": "phone-1"})
    r = client.post("/api/settings", json={"watchlist": [f"S{i}" for i in range(14)], "client_id": "emulator-2"})
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["error"] == "watchlist_sync_refused"
    assert detail["n_before"] == 54
    assert detail["n_after"] == 14
    assert detail["removed_count"] == 40
    assert "replace=true" in detail["message"]
    # The refused sync must not have landed.
    assert client.get("/api/settings").json()["watchlist"] == [f"S{i}" for i in range(54)]


def test_same_client_pruning_its_own_watchlist_is_never_refused(client):
    client.post("/api/settings", json={"watchlist": [f"S{i}" for i in range(54)], "client_id": "phone-1"})
    r = client.post("/api/settings", json={"watchlist": ["AAPL"], "client_id": "phone-1"})
    assert r.status_code == 200
    assert r.json()["watchlist"] == ["AAPL"]


def test_replace_true_forces_a_refused_cross_client_sync_through(client):
    client.post("/api/settings", json={"watchlist": [f"S{i}" for i in range(54)], "client_id": "phone-1"})
    r = client.post("/api/settings", json={
        "watchlist": [f"S{i}" for i in range(14)], "client_id": "emulator-2", "replace": True,
    })
    assert r.status_code == 200
    assert len(r.json()["watchlist"]) == 14


def test_settings_response_reports_synced_by_and_removal_guard(client):
    client.post("/api/settings", json={"watchlist": ["AAPL"], "client_id": "phone-1"})
    body = client.get("/api/settings").json()
    assert body["watchlist_synced_by"] == "phone-1"
    assert isinstance(body["watchlist_removal_guard"], int)


def test_health_reports_settings_source_file_by_default(client):
    # A first save creates settings.json — /health must then report the normal "file" state.
    client.post("/api/settings", json={"deep_model": "x"})
    body = client.get("/health").json()
    assert body["settings_source"] == "file"


def test_health_reports_settings_source_env_when_settings_file_is_absent(tmp_path, monkeypatch):
    """A backend that has never had settings.json written (e.g. wiped data dir) must report the
    degraded 'env' state rather than silently looking like a normally configured one."""
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    import app.sandbox_store as ss
    import app.settings_store as st
    importlib.reload(st)
    importlib.reload(ss)
    assert not st._FILE.exists()
    import app.main as m
    importlib.reload(m)
    with TestClient(m.app) as c:
        assert c.get("/health").json()["settings_source"] == "env"


def test_health_reports_settings_source_backup_after_corruption(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    import app.sandbox_store as ss
    import app.settings_store as st
    importlib.reload(st)
    importlib.reload(ss)
    st.update({"deep_model": "good"})
    st.update({"deep_model": "good-again"})  # second write turns the first into a real .bak
    st._FILE.write_text("{ broken")
    importlib.reload(st)
    importlib.reload(ss)
    import app.main as m
    importlib.reload(m)
    with TestClient(m.app) as c:
        assert c.get("/health").json()["settings_source"] == "backup"
