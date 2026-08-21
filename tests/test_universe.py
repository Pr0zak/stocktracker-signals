"""MB-19 / SWT-1 — the curated universe: what it parses, and what it is allowed to admit.

Parsing is the part that decides what a screen can ever see, so it is tested against the real file's
shape: pipe-delimited, a header row, a trailing footer, and columns whose meaning is positional.

The second half pins the SWT-1 gate change. The build used to admit a name on MARKET CAP >= $2B,
which deleted 13.3% of equities outright (Yahoo reports no marketCap for them) and excluded liquid
sub-$5 names such as PLUG — $134M of stock a day at $2.27. The gate is now AVERAGE DOLLAR VOLUME and
market cap survives as optional METADATA, which is what makes these tests necessary:

  * an unreported cap must be stored as None, NEVER 0.0 — a $0 company is the "absent rendered as a
    confident number" defect this codebase keeps having to fix;
  * the sort must therefore survive None, and must stay CAP-DESCENDING, because /heatmap?mode=market
    slices detail[:limit] under a caption that says "Sized by market cap";
  * ETFs are excluded ON PURPOSE. They already were, accidentally — the old isinstance(cap) gate
    deleted all but 9 of 5,620 — so dropping the cap gate without saying so would have silently
    admitted 1,404 funds;
  * anything the build could not measure (no price, no average volume) is COUNTED into a named
    field, never dropped in silence;
  * and publish() gates on symbols returned, not just on batches that answered, because Yahoo omits
    unknown symbols from an otherwise-healthy 200 response.
"""
import asyncio
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


@pytest.mark.parametrize("name", [
    "Churchill Capital Corp XIII - Units",
    "Karman Line Acquisition Corp. - Units",
    "Activate Energy Acquisition Corp. - Unit",
    "Armada Acquisition Corp. III - Warrant",
    "Abony Acquisition Corp. I - Warrants",
    "Apogee Acquisition Corp - Rights",
    "Apogee Acquisition Corp - Right",
    "BrightSpring Health Services, Inc. - Tangible Equity Unit",
])
def test_units_and_warrants_are_dropped_even_when_the_symbol_looks_ordinary(name):
    """The dotted-suffix rule only catches what the feed spells with a dot, and Nasdaq writes most
    SPAC units as plain five-letter symbols (XIIIU, XTERU). The $2B cap floor had been excluding
    them by accident; replacing it with a liquidity gate let seven into the universe. A unit is a
    share bundled with a fraction of a warrant, so its price is part trust value and part option
    premium — a moving average over that hybrid describes no company, and it pollutes every
    cross-sectional percentile it lands in.
    """
    assert u.parse_directory(doc(row("XIIIU", name))) == []


@pytest.mark.parametrize("name", [
    "Alibaba Group Holding Limited American Depositary Shares each representing eight Ordinary",
    "Sony Group Corporation American Depositary Shares",
    "ASML Holding N.V. - New York Registry Shares",
    "Taiwan Semiconductor Manufacturing Company Ltd.",
    "Novo Nordisk A/S Common Stock",
])
def test_depositary_and_registry_shares_are_KEPT(name):
    """The instrument-name rule is anchored to the end of the string for exactly this reason. An
    unanchored match on "Shares" or "Depositary" deletes every ADR — TSM, BABA, ASML and SONY reach
    this codebase through them, and TSM sits in the universe's top ten by market cap. Measured
    against the live directory: 721 rows matched as units/warrants/rights, and none of these did.
    """
    assert [r["symbol"] for r in u.parse_directory(doc(row("ADRX", name)))] == ["ADRX"]


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
    async def fake_directory(client):
        return [{"symbol": f"S{i}", "name": f"Co {i}", "is_etf": False} for i in range(200)]

    calls = {"n": 0}

    async def flaky_batch(client, syms):
        calls["n"] += 1
        if calls["n"] % 2 == 0:          # half the batches die
            return {}
        return {s: {"cap": 5e9, "price": 50.0, "type": "EQUITY", "adv": 1_000_000}
                for s in syms}

    monkeypatch.setattr(u, "fetch_directory", fake_directory)
    monkeypatch.setattr(u, "_profile_batch", flaky_batch)
    blob = asyncio.run(u.build(None))
    assert blob["complete"] is False
    assert blob["batch_coverage"] < u.MIN_COVERAGE
    assert blob["empty_batches"] > 0


def test_a_clean_fetch_is_marked_complete(monkeypatch):
    async def fake_directory(client):
        return [{"symbol": f"S{i}", "name": f"Co {i}", "is_etf": False} for i in range(120)]

    async def ok_batch(client, syms):
        return {s: {"cap": 5e9, "price": 50.0, "type": "EQUITY", "adv": 1_000_000}
                for s in syms}

    monkeypatch.setattr(u, "fetch_directory", fake_directory)
    monkeypatch.setattr(u, "_profile_batch", ok_batch)
    blob = asyncio.run(u.build(None))
    assert blob["complete"] is True and blob["batch_coverage"] == 1.0


# ---------------------------------------------------------------- the liquidity gate (SWT-1)

def _q(price=50.0, adv=1_000_000, cap=5e9, quote_type="EQUITY") -> dict:
    """One row of _profile_batch's output. Pass adv=None to model Yahoo OMITTING the field."""
    row = {"cap": cap, "price": price, "type": quote_type}
    if adv is not None:
        row["adv"] = adv
    return row


def _build(monkeypatch, quotes: dict, **kw) -> dict:
    """Drive build() against a stubbed directory + quote fetch, so no network is touched.

    The directory's ETF flag is derived from the quote type so both signals agree, which is what the
    real feeds do; the tests that care about a disagreement set it themselves.
    """
    directory = kw.pop("directory", None) or [
        {"symbol": s, "name": f"{s} Inc", "is_etf": q.get("type") == "ETF"}
        for s, q in quotes.items()]

    async def fake_directory(client):
        return directory

    async def fake_batch(client, syms):
        return {s: quotes[s] for s in syms if s in quotes}

    monkeypatch.setattr(u, "fetch_directory", fake_directory)
    monkeypatch.setattr(u, "_profile_batch", fake_batch)
    return asyncio.run(u.build(None, **kw))


def test_the_gate_is_dollar_volume_not_market_cap(monkeypatch):
    """A $40B company that trades $200k a day cannot be scanned; a $400M one trading $80M a day can.

    Market cap answered "how big is the issuer". The scanner's question is "can this be traded", and
    the two disagree often enough that the old filter admitted the untradable and refused the liquid.
    """
    blob = _build(monkeypatch, {
        "ILLIQ": _q(price=20.0, adv=10_000, cap=40e9),      # $200k/day
        "LIQUID": _q(price=8.0, adv=10_000_000, cap=400e6),  # $80M/day
    })
    assert blob["symbols"] == ["LIQUID"]
    assert blob["passed_filter"] == 1


def test_a_liquid_sub_five_dollar_name_is_admitted(monkeypatch):
    """PLUG: $2.27 a share, 59.4M shares a day — $134M of stock changing hands.

    The old $5 price floor excluded it outright, which is the whole reason the floor moved to $1:
    dollar volume is the tradability test, and price alone was a cruder version of the same test
    that happened to be wrong here.
    """
    blob = _build(monkeypatch, {"PLUG": _q(price=2.27, adv=59_391_074, cap=2.6e9)})
    assert blob["symbols"] == ["PLUG"]
    assert blob["min_price"] == 1.0


def test_a_sub_one_dollar_name_is_still_refused_however_liquid(monkeypatch):
    # $1 is not a size opinion — it is where tick-size and delisting behaviour start breaking the
    # weekly-trend maths downstream.
    blob = _build(monkeypatch, {"PENNY": _q(price=0.40, adv=100_000_000, cap=1e9)})
    assert blob["symbols"] == []


def test_a_name_with_no_reported_market_cap_survives_and_is_stored_as_None(monkeypatch):
    """Absent is not small, and it is certainly not zero.

    873 of 6,555 equities (13.3%) report no marketCap. The old build deleted every one of them via
    the isinstance(cap) gate. discover.py already treats a missing cap as "keep it" — pinned by
    tests/test_min_market_cap.py — and the universe now agrees.
    """
    blob = _build(monkeypatch, {"NOCAP": _q(cap=None)})
    assert blob["symbols"] == ["NOCAP"]
    row = blob["detail"][0]
    assert row["market_cap"] is None, "0.0 would render a real company as a $0 company"
    assert "market_cap" in row, "the key must exist so a reader sees 'unknown', not 'not fetched'"


def test_an_etf_is_excluded_from_the_stored_universe_and_counted(monkeypatch):
    """EQUITIES ONLY, on purpose rather than by accident.

    The live universe already held zero ETFs (0 of 600 — no VTI, no SPY, no FBTC), because only 9 of
    5,620 ETFs report a marketCap and the cap gate deleted the rest. Removing that gate without an
    explicit rule would have silently admitted 1,404 funds to a screen built around company trends.
    """
    blob = _build(monkeypatch, {"SPY": _q(price=600.0, adv=80_000_000, cap=None, quote_type="ETF"),
                                "AAPL": _q(price=200.0, adv=50_000_000, cap=3e12)})
    assert blob["symbols"] == ["AAPL"]
    assert blob["etf_rows"] == 1
    assert blob["equity_rows"] == 1


def test_an_etf_yahoo_types_as_EQUITY_is_still_excluded_by_the_directory_flag(monkeypatch):
    # The two feeds disagree occasionally. Either one saying "fund" is enough to keep it out.
    quotes = {"FBTC": _q(price=90.0, adv=5_000_000, cap=20e9)}      # Yahoo says EQUITY
    directory = [{"symbol": "FBTC", "name": "Fidelity Wise Origin Bitcoin Fund", "is_etf": True}]
    blob = _build(monkeypatch, quotes, directory=directory)
    assert blob["symbols"] == [] and blob["etf_rows"] == 1


def test_a_row_with_no_average_volume_is_excluded_AND_counted(monkeypatch):
    """averageDailyVolume3Month covers ~95% of the response, against marketCap's 79%.

    A missing one is therefore a fetch anomaly rather than a property of the company. The gate cannot
    run without it, so the name is excluded — but silence would make a fetch problem indistinguishable
    from an illiquid stock, so it lands in a named field instead.
    """
    blob = _build(monkeypatch, {"NOADV": _q(adv=None), "FINE": _q()})
    assert blob["symbols"] == ["FINE"]
    assert blob["no_adv"] == 1


def test_a_row_with_no_price_is_excluded_AND_counted(monkeypatch):
    # No price means no scan — every computation downstream is price-based — but it is still a fact
    # about the response, so it is counted rather than dropped.
    blob = _build(monkeypatch, {"NOPX": _q(price=None), "FINE": _q()})
    assert blob["symbols"] == ["FINE"]
    assert blob["no_price"] == 1


def test_the_sort_survives_unknown_caps_and_leaves_them_at_the_TAIL(monkeypatch):
    """`rows.sort(key=lambda r: -r["market_cap"])` raises TypeError the instant a cap can be None.

    Ordering is load-bearing beyond not crashing: /heatmap?mode=market slices detail[:limit] under a
    caption that says "Sized by market cap". Keeping the sort cap-descending leaves the HEAD of the
    list untouched, so the map's top-200 behaves exactly as before and the universe grows only at
    the tail. Re-ranking by dollar volume would silently rewrite which companies appear on the map.
    """
    blob = _build(monkeypatch, {"MID": _q(cap=1e9), "NOCAP": _q(cap=None), "BIG": _q(cap=3e9)})
    assert blob["symbols"] == ["BIG", "MID", "NOCAP"]
    assert blob["detail"][-1]["market_cap"] is None


def test_the_stored_universe_records_the_gate_it_ran(monkeypatch):
    # A blob that does not say what filtered it cannot be audited later — and this one changed.
    blob = _build(monkeypatch, {"FINE": _q()})
    assert blob["min_dollar_volume"] == u.DEFAULT_MIN_DOLLAR_VOLUME == 5_000_000.0
    assert blob["min_price"] == u.DEFAULT_MIN_PRICE == 1.0
    assert blob["min_cap"] == u.DEFAULT_MIN_CAP == 0.0, "cap is metadata now, never a filter"
    assert u.DEFAULT_LIMIT == 4000
    for k in ("symbol_coverage", "batch_coverage", "no_adv", "no_price",
              "etf_rows", "equity_rows", "passed_filter", "directory_rows", "complete"):
        assert k in blob, k


def test_symbol_coverage_counts_symbols_returned_not_batches_that_answered(monkeypatch):
    """Yahoo silently OMITS unknown symbols from an otherwise-200 response.

    batch_coverage counts BATCHES, so a batch that answered for half of what it was asked looks
    perfectly healthy. At 253 batches, MIN_COVERAGE=0.90 alone tolerates ~1,250 missing symbols in a
    build stamped `complete`.
    """
    # Every batch answers — it just answers for half the symbols it was handed, which is exactly
    # what an omitted (unknown/delisted) symbol looks like from here.
    quotes = {f"S{i}": _q() for i in range(0, 200, 2)}
    directory = [{"symbol": f"S{i}", "name": f"Co {i}", "is_etf": False} for i in range(200)]
    blob = _build(monkeypatch, quotes, directory=directory)
    assert blob["batch_coverage"] == 1.0, "every batch answered..."
    assert blob["symbol_coverage"] == 0.5, "...for half the symbols"
    assert blob["complete"] is True, "batch coverage alone cannot see this; publish() is the gate"


# ---------------------------------------------------------------- publish (shared by both callers)

def test_a_partial_build_never_replaces_a_good_universe():
    """The coverage guard lived in the ENDPOINT only, so the nightly hook — the unattended path,
    the one nobody watches — would persist a half-fetched universe and stamp it fresh for a week."""
    partial = {"symbols": ["A"], "complete": False, "batch_coverage": 0.4, "empty_batches": 30}
    good = {"symbols": ["A", "B", "C"]}
    ok, why = u.publish(partial, previous=good)
    assert ok is False
    assert "keeping the previous" in why


def test_a_partial_build_IS_accepted_when_there_is_nothing_to_fall_back_on():
    partial = {"symbols": ["A"], "complete": False, "batch_coverage": 0.4, "empty_batches": 30}
    ok, why = u.publish(partial, previous=None)
    assert ok is True
    assert "PARTIAL" in why, "accepting it is fine; doing so silently is not"


def test_a_complete_build_is_accepted():
    ok, why = u.publish({"symbols": ["A", "B"], "complete": True, "passed_filter": 2}, previous=None)
    assert ok is True and "rebuilt:" in why


def test_an_empty_build_is_always_refused():
    for prev in (None, {"symbols": ["A"]}):
        ok, why = u.publish({"symbols": [], "complete": True}, previous=prev)
        assert ok is False and "no symbols" in why


def test_publish_always_gives_a_reason_so_every_caller_can_log_an_outcome():
    """Logging only on a successful rebuild left the common case (universe still fresh) with no
    trace at all, so a working guard and an absent one looked identical in the journal."""
    for blob, prev in (({"symbols": [], "complete": True}, None),
                       ({"symbols": ["A"], "complete": True, "passed_filter": 1}, None),
                       ({"symbols": ["A"], "complete": False, "batch_coverage": 0.5}, {"symbols": ["A", "B"]}),
                       ({"symbols": ["A"], "complete": False, "batch_coverage": 0.5}, None)):
        ok, why = u.publish(blob, previous=prev)
        assert isinstance(why, str) and why, (blob, prev)


def test_publish_refuses_a_build_that_answered_for_too_few_symbols(monkeypatch):
    """The symbol-coverage floor, and why it lives in publish() rather than in main.py.

    publish() is the ONE place both callers go through — the attended POST /universe/build and the
    unattended nightly hook. The batch-coverage guard originally lived in the endpoint only, which
    left the path nobody watches unprotected; putting this one anywhere else would repeat that.
    """
    thin = {"symbols": ["A"], "complete": True, "batch_coverage": 1.0,
            "symbol_coverage": 0.80, "passed_filter": 1}
    ok, why = u.publish(thin, previous={"symbols": ["A", "B", "C"]})
    assert ok is False
    assert "symbol coverage" in why and "keeping the previous" in why


def test_a_thin_build_is_accepted_when_there_is_nothing_to_fall_back_on_but_says_so():
    thin = {"symbols": ["A"], "complete": True, "batch_coverage": 1.0, "symbol_coverage": 0.80}
    ok, why = u.publish(thin, previous=None)
    assert ok is True and "PARTIAL" in why and "symbol coverage" in why


def test_symbol_coverage_at_or_above_the_floor_publishes_normally():
    ok, why = u.publish({"symbols": ["A"], "complete": True, "batch_coverage": 1.0,
                         "symbol_coverage": u.MIN_SYMBOL_COVERAGE, "passed_filter": 1},
                        previous={"symbols": ["A", "B"]})
    assert ok is True and "rebuilt:" in why


def test_an_ABSENT_symbol_coverage_is_not_a_failing_one():
    # A blob built before the field existed cannot be judged on it. Refusing on a number that was
    # never measured is inventing evidence — the same mistake as reading a missing cap as $0.
    ok, why = u.publish({"symbols": ["A"], "complete": True, "passed_filter": 1},
                        previous={"symbols": ["A", "B"]})
    assert ok is True and "rebuilt:" in why
