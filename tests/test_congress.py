"""Offline tests for the congressional-trades block (app/congress.py). No network — the index is mocked."""
from __future__ import annotations

import asyncio
import datetime as dt

from app import congress as cg


def _recent(days_ago: int) -> str:
    return (dt.date.today() - dt.timedelta(days=days_ago)).isoformat()


def _t(filer, side, days_ago, party="R", chamber="senate", amt_low=1001, amt_high=15000, days_to_file=20):
    ttype = "Purchase" if side == "buy" else "Sale (Full)"
    return {
        "ticker": "AAPL", "filer_name": f"{filer},", "party": party, "chamber": chamber,
        "transaction_type": ttype, "transaction_date": _recent(days_ago), "filing_date": _recent(1),
        "amount_range_low": amt_low, "amount_range_high": amt_high,
        "amount_range_label": f"${amt_low:,} - ${amt_high:,}", "days_to_file": days_to_file, "is_late": 0,
    }


def test_classifiers_and_filer_cleanup():
    assert cg._is_buy({"transaction_type": "Purchase"}) is True
    assert cg._is_sell({"transaction_type": "Sale (Partial)"}) is True
    assert cg._is_buy({"transaction_type": "Sale (Full)"}) is False
    assert cg._filer({"filer_name": "Jerry Moran,"}) == "Jerry Moran"


def test_build_block_cluster_buying():
    rows = [
        _t("Alice", "buy", 20, party="R", amt_high=50000),
        _t("Bob", "buy", 10, party="D", amt_high=250000),
        _t("Carol", "buy", 2, party="R", amt_high=15000),
    ]
    b = cg._build_block(rows, months=12)
    assert b["trade_count"] == 3 and b["buy_count"] == 3 and b["sell_count"] == 0
    assert b["net_direction"] == "buying"
    assert b["distinct_filers"] == 3
    assert b["cluster_buy"] is True                 # 3 distinct filers within 30 days
    assert b["largest_buy_amount_high"] == 250000
    assert b["parties"] == {"R": 2, "D": 1}
    assert len(b["latest"]) == 3 and b["latest"][0]["side"] == "buy"


def test_build_block_mixed_and_no_cluster():
    rows = [
        _t("Alice", "buy", 5),
        _t("Alice", "buy", 4),                       # same filer twice -> not 3 DISTINCT
        _t("Bob", "sell", 3),
        _t("Carol", "sell", 2),
    ]
    b = cg._build_block(rows, months=12)
    assert b["buy_count"] == 2 and b["sell_count"] == 2
    assert b["net_direction"] == "mixed"
    assert b["cluster_buy"] is False                 # only 2 distinct buyers
    assert b["distinct_filers"] == 3


def test_build_block_none_when_all_stale():
    rows = [_t("Alice", "buy", 800)]                 # ~2+ years ago, outside the 12-month window
    assert cg._build_block(rows, months=12) is None


def test_has_cluster_threshold():
    buys3 = [_t("A", "buy", 20), _t("B", "buy", 10), _t("C", "buy", 5)]
    buys2 = [_t("A", "buy", 20), _t("B", "buy", 5)]
    assert cg._has_cluster(buys3) is True
    assert cg._has_cluster(buys2) is False


def test_compact_slims_and_drops_empty():
    assert cg.compact(None) is None
    assert cg.compact({"trade_count": 0}) is None
    full = cg._build_block([_t("A", "buy", 5), _t("B", "buy", 3), _t("C", "buy", 2)], months=12)
    slim = cg.compact(full)
    assert slim["trade_count"] == 3 and slim["cluster_buy"] is True
    assert len(slim["latest"]) == 2                  # compact trims latest to 2
    assert "latest_filing_date" not in slim          # compact drops the verbose fields


def test_congress_trades_uses_index(monkeypatch):
    idx = {"AAPL": [_t("A", "buy", 5), _t("B", "buy", 3), _t("C", "buy", 2)]}

    async def fake_index(_client):
        return idx
    monkeypatch.setattr(cg, "_ensure_index", fake_index)

    got = asyncio.run(cg.congress_trades(None, "aapl"))
    assert got["buy_count"] == 3 and got["cluster_buy"] is True
    assert asyncio.run(cg.congress_trades(None, "TSLA")) is None   # not in the index
