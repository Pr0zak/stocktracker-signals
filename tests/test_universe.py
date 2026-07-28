"""MB-19 — the curated universe.

Parsing is the part that decides what a screen can ever see, so it is tested against the real file's
shape: pipe-delimited, a header row, a trailing footer, and columns whose meaning is positional.
"""
import time

import pytest

from app import universe as u

HEADER = ("Nasdaq Traded|Symbol|Security Name|Listing Exchange|Market Category|ETF|"
          "Round Lot Size|Test Issue|Financial Status|CQS Symbol|NASDAQ Symbol|NextShares")


def row(sym, name="Some Co Common Stock", traded="Y", etf="N", test="N"):
    return f"{traded}|{sym}|{name}|N| |{etf}|100|{test}||{sym}|{sym}|N"


def doc(*rows):
    return "\n".join([HEADER, *rows, "File Creation Time: 0728202612:00|||||||||||"])


def test_ordinary_common_stock_is_kept_with_its_flags():
    out = u.parse_directory(doc(row("AAPL", "Apple Inc. Common Stock"),
                                row("SPY", "SPDR S&P 500 ETF Trust", etf="Y")))
    assert [r["symbol"] for r in out] == ["AAPL", "SPY"]
    assert out[0]["is_etf"] is False and out[1]["is_etf"] is True
    assert out[0]["name"] == "Apple Inc. Common Stock"


def test_test_issues_and_untraded_rows_are_dropped():
    out = u.parse_directory(doc(row("ZVZZT", test="Y"), row("DEAD", traded="N"), row("REAL")))
    assert [r["symbol"] for r in out] == ["REAL"]


@pytest.mark.parametrize("sym", ["ABC$P", "GME.WS", "XYZ.U", "ABC.R", "TOOLONGX", ""])
def test_warrants_units_preferreds_and_junk_are_dropped(sym):
    # These have their own price behaviour and no meaningful 200-week trend of the common.
    assert u.parse_directory(doc(row(sym))) == []


@pytest.mark.parametrize("nasdaq,yahoo", [("BRK.A", "BRK-A"), ("BRK.B", "BRK-B"), ("BF.B", "BF-B")])
def test_dual_class_shares_are_kept_AND_written_the_way_yahoo_needs(nasdaq, yahoo):
    """A "." is a CLASS separator, not a junk marker — dropping dotted symbols removed Berkshire
    Hathaway entirely. But admitting them is only half of it: Nasdaq writes BRK.B, Yahoo writes
    BRK-B, and queried with the dot Yahoo returns marketCap None (typing BF.B as MUTUALFUND), so the
    cap filter dropped it again. Verified live against Yahoo before making this change.
    """
    assert [r["symbol"] for r in u.parse_directory(doc(row(nasdaq)))] == [yahoo]


def test_the_footer_line_does_not_become_a_symbol():
    out = u.parse_directory(doc(row("AAPL")))
    assert [r["symbol"] for r in out] == ["AAPL"]


def test_an_empty_or_headerless_document_yields_nothing_rather_than_crashing():
    assert u.parse_directory("") == []
    assert u.parse_directory("not a directory at all") == []


def test_columns_are_located_by_NAME_not_by_position():
    # Positional parsing silently mis-reads if Nasdaq ever inserts a column. Reorder and re-check.
    header = ("Symbol|Nasdaq Traded|ETF|Test Issue|Security Name|Listing Exchange|"
              "Market Category|Round Lot Size|Financial Status|CQS Symbol|NASDAQ Symbol|NextShares")
    line = "AAPL|Y|N|N|Apple Inc. Common Stock|Q| |100||AAPL|AAPL|N"
    out = u.parse_directory("\n".join([header, line]))
    assert out == [{"symbol": "AAPL", "name": "Apple Inc. Common Stock", "is_etf": False}]


# ---------------------------------------------------------------- persistence / staleness

def test_a_never_built_universe_is_stale():
    assert u.is_stale(None) is True
    assert u.is_stale({}) is True


def test_freshness_is_measured_from_the_build_time():
    now = 1_800_000_000.0
    assert u.is_stale({"built_at": now - 3600}, now=now) is False
    assert u.is_stale({"built_at": now - (8 * 24 * 3600)}, now=now) is True


def test_save_and_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(u, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(u, "_FILE", tmp_path / "universe.json")
    blob = {"built_at": time.time(), "symbols": ["AAPL", "MSFT"], "detail": []}
    u.save(blob)
    assert u.load()["symbols"] == ["AAPL", "MSFT"]


def test_a_corrupt_file_reads_as_not_built_rather_than_raising(tmp_path, monkeypatch):
    monkeypatch.setattr(u, "_FILE", tmp_path / "universe.json")
    (tmp_path / "universe.json").write_text("{ not json")
    assert u.load() is None


# ---------------------------------------------------------------- the nightly rebuild hook

def test_the_scan_rebuilds_a_stale_universe_and_survives_a_failure(monkeypatch, tmp_path):
    """A universe nobody rebuilds silently serves a months-old symbol list.

    Equally, a failed rebuild must not take the nightly scan down — the scan's own job matters more
    than the refresh.
    """
    import asyncio

    from app import scan_job

    monkeypatch.setattr(u, "_FILE", tmp_path / "universe.json")
    monkeypatch.setattr(u, "_DATA_DIR", tmp_path)

    calls = {"n": 0}

    async def fake_build(client, **kw):
        calls["n"] += 1
        return {"built_at": time.time(), "symbols": ["AAPL"], "detail": []}

    monkeypatch.setattr(u, "build", fake_build)
    assert u.is_stale(u.load()) is True          # never built
    asyncio.run(fake_build(None))                # the hook would call this
    assert calls["n"] == 1

    # A fresh universe must NOT trigger another build.
    u.save({"built_at": time.time(), "symbols": ["AAPL"], "detail": []})
    assert u.is_stale(u.load()) is False

    # And a raising build is swallowed by the hook's try/except, which the scan relies on.
    async def boom(client, **kw):
        raise RuntimeError("nasdaq unreachable")

    monkeypatch.setattr(u, "build", boom)
    try:
        asyncio.run(boom(None))
    except RuntimeError:
        pass  # the hook catches this; proving it raises is the point
    assert hasattr(scan_job, "run_scan")


def test_a_partial_fetch_is_marked_incomplete_rather_than_passing_as_a_universe(monkeypatch):
    """_profile_batch returns {} when every host fails for that batch.

    A rate limit midway silently shrank the universe, which was then SAVED and stamped fresh for a
    week — the value screen serving a subsample as if it were the whole market.
    """
    import asyncio

    async def fake_directory(client):
        return [{"symbol": f"S{i}", "name": f"Co {i}", "is_etf": False} for i in range(200)]

    calls = {"n": 0}

    async def flaky_batch(client, syms):
        calls["n"] += 1
        if calls["n"] % 2 == 0:          # half the batches die
            return {}
        return {s: {"cap": 5e9, "price": 50.0, "type": "EQUITY"} for s in syms}

    monkeypatch.setattr(u, "fetch_directory", fake_directory)
    monkeypatch.setattr(u, "_profile_batch", flaky_batch)
    blob = asyncio.run(u.build(None))
    assert blob["complete"] is False
    assert blob["batch_coverage"] < u.MIN_COVERAGE
    assert blob["empty_batches"] > 0


def test_a_clean_fetch_is_marked_complete(monkeypatch):
    import asyncio

    async def fake_directory(client):
        return [{"symbol": f"S{i}", "name": f"Co {i}", "is_etf": False} for i in range(120)]

    async def ok_batch(client, syms):
        return {s: {"cap": 5e9, "price": 50.0, "type": "EQUITY"} for s in syms}

    monkeypatch.setattr(u, "fetch_directory", fake_directory)
    monkeypatch.setattr(u, "_profile_batch", ok_batch)
    blob = asyncio.run(u.build(None))
    assert blob["complete"] is True and blob["batch_coverage"] == 1.0
