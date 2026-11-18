import pytest

from app import station_store as st


def w(**k):
    base = {"id": "a1", "type": "quote", "x": 0, "y": 0, "w": 3, "h": 2, "cfg": {"sym": "SPY"}}
    base.update(k)
    return base


def test_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "_FILE", tmp_path / "station_layout.json")
    assert st.load() is None
    doc = st.save({"name": " Desk ", "widgets": [w(), w(id="b2", type="watchlist", cfg={"syms": ["AAPL", "MSFT"]})]})
    assert doc["name"] == "Desk" and len(doc["widgets"]) == 2
    assert st.load()["widgets"][1]["cfg"]["syms"] == ["AAPL", "MSFT"]


@pytest.mark.parametrize("bad", [
    {"widgets": [w(type="nope")]},
    {"widgets": [w(x=10, w=3)]},
    {"widgets": [w(), w()]},
    {"widgets": [w(h=0)]},
    {"widgets": [w(x=True)]},
    {"widgets": [w(cfg={"sym": "x" * 300})]},
    {"widgets": [w(cfg={"bad key": 1})]},
    {"widgets": "x"},
    {"name": "n" * 61, "widgets": []},
])
def test_refuses_bad_layouts(bad):
    with pytest.raises(ValueError):
        st.validate(bad)


def test_empty_station_is_allowed():
    assert st.validate({"widgets": []}) == {"name": "My station", "widgets": []}


# ---------------------------------------------------------------- the endpoints
import importlib

from fastapi.testclient import TestClient

_TOKEN = "pytest-signals-token-do-not-use-in-prod"  # tests/conftest.py's default
_AUTH = {"Authorization": "Bearer " + _TOKEN}


@pytest.fixture()
def remote(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    import app.station_store as s
    importlib.reload(s)
    import app.main as m
    importlib.reload(m)
    with TestClient(m.app, client=("203.0.113.7", 54321)) as c:
        yield c


def test_endpoints_need_the_token(remote):
    assert remote.get("/station/layout").status_code == 401
    assert remote.post("/station/layout", json={"widgets": []}).status_code == 401


def test_endpoint_round_trip_and_refusal(remote):
    assert remote.get("/station/layout", headers=_AUTH).json() == {"saved": False}
    r = remote.post("/station/layout", headers=_AUTH, json={"name": "Desk", "widgets": [w()]})
    assert r.status_code == 200 and r.json()["saved"] is True
    got = remote.get("/station/layout", headers=_AUTH).json()
    assert got["saved"] is True and got["name"] == "Desk" and got["widgets"][0]["cfg"] == {"sym": "SPY"}
    bad = remote.post("/station/layout", headers=_AUTH, json={"widgets": [w(type="nope")]})
    assert bad.status_code == 422 and "unknown widget type" in bad.json()["detail"]
    assert remote.get("/station/layout", headers=_AUTH).json()["name"] == "Desk"   # a refused POST changes nothing
