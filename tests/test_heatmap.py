"""Heat-map tile data.

The server returns VALUES, never rectangles — it has no idea how wide the phone is, so the
squarified layout belongs on the device. What it must get right is that the two colour scales stay
separate and that anything it could not price is named rather than dropped.
"""
from app import heatmap as h


UNI = {"detail": [
    {"symbol": "AAPL", "market_cap": 4.99e12, "name": "Apple Inc."},
    {"symbol": "NVDA", "market_cap": 4.77e12, "name": "NVIDIA Corporation"},
    {"symbol": "DEAD", "market_cap": 1.0e12, "name": "No Quote Co"},
]}


def test_market_tiles_size_by_cap_and_colour_by_move():
    q = {"AAPL": {"price": 300.0, "pct": -0.56, "name": "Apple Inc."},
         "NVDA": {"price": 190.0, "pct": -3.55, "name": "NVIDIA Corporation"},
         "DEAD": {}}
    tiles, unpriced = h.market_tiles(UNI, q, limit=10)
    assert [t["symbol"] for t in tiles] == ["AAPL", "NVDA"]
    assert tiles[0]["size"] == 4990.0 and tiles[0]["value"] == -0.56
    assert all(t["scale"] == "price" for t in tiles)


def test_a_name_we_could_not_price_is_named_not_dropped():
    """"We could not price it" is a fact about our fetch; "it did not move" is a fact about the
    stock. Collapsing them makes a broken feed look like a flat day."""
    tiles, unpriced = h.market_tiles(UNI, {"AAPL": {"price": 1.0, "pct": 0.0}}, limit=10)
    assert [t["symbol"] for t in tiles] == ["AAPL"]
    assert sorted(unpriced) == ["DEAD", "NVDA"]


def test_a_zero_or_missing_cap_cannot_produce_a_zero_area_tile():
    uni = {"detail": [{"symbol": "X", "market_cap": 0, "name": "X"}]}
    tiles, unpriced = h.market_tiles(uni, {"X": {"price": 5.0, "pct": 1.0}}, limit=10)
    assert tiles == [] and unpriced == ["X"]


def test_signal_tiles_use_the_signal_scale_and_rank_dips():
    scan = {"results": [
        {"symbol": "GME", "pct_off_52w_high": -21.1, "dip": "mega_dip", "signal": "hold",
         "conviction": 42, "below_200wma": False},
        {"symbol": "SPY", "pct_off_52w_high": -2.5, "dip": None, "signal": "hold",
         "conviction": 48, "below_200wma": False},
        {"symbol": "UNH", "pct_off_52w_high": -7.1, "dip": "below_line", "signal": "buy",
         "conviction": 68, "below_200wma": True},
    ]}
    tiles, skipped = h.signal_tiles(scan, limit=10)
    assert all(t["scale"] == "signal" for t in tiles), "a dip tier must never ride the price scale"
    by = {t["symbol"]: t for t in tiles}
    assert by["GME"]["value"] > by["UNH"]["value"] > by["SPY"]["value"]
    assert by["GME"]["size"] > by["SPY"]["size"]      # sorted by drawdown, biggest first
    assert by["UNH"]["below_200wma"] is True


def test_a_name_with_no_drawdown_still_gets_a_tappable_area():
    """A tile you cannot hit is the same as a tile that is absent."""
    scan = {"results": [{"symbol": "FLAT", "pct_off_52w_high": 0.0, "dip": None}]}
    tiles, _ = h.signal_tiles(scan, limit=10)
    assert tiles[0]["size"] >= 1.5


def test_a_row_with_no_drawdown_figure_is_reported_not_guessed():
    scan = {"results": [{"symbol": "NEW", "dip": None}, {"symbol": "OK", "pct_off_52w_high": -5.0}]}
    tiles, skipped = h.signal_tiles(scan, limit=10)
    assert [t["symbol"] for t in tiles] == ["OK"]
    assert skipped == ["NEW"]


def test_empty_and_missing_inputs_do_not_raise():
    assert h.market_tiles(None, {}, limit=5) == ([], [])
    assert h.signal_tiles(None, limit=5) == ([], [])
    assert h.signal_tiles({"results": []}, limit=5) == ([], [])
