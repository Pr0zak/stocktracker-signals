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
