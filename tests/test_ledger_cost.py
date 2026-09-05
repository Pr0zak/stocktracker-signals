"""The sandbox's own ledger, in dollars — what its decisions cost against holding and against the index.

`recent_activity` tells the tick what it did in the last two weeks. Nothing told it what that cost.
A sale two months old is outside that window and still a fact about the account, because the shares
it let go have a price today. Measured on the live paper account on 2026-09-04: the four sales it had
made were worth $57.89 more than they fetched, $48.40 of that beyond what the proceeds would have
earned in the S&P — and the model deciding the next sale had no way to know.

COUNTS AND DOLLARS ONLY. The 2026-09-04 research panel's ruling: an account making a handful of
decisions a month cannot measure its own skill (a 2%/yr edge at 5% tracking error needs ~49 years),
so any rate over its own outcomes is a number that only looks like knowledge. The block carries none.
"""

from app.sandbox_job import (
    _COST_MAX_SALES, _LONG_TERM_DAYS, _SOLD_PRICE_MAX, ledger_cost, sold_symbols,
)

TODAY = "2026-09-04"


class _Bench:
    """A stand-in for market.Series: YYYYMMDD dates and index-aligned closes."""
    def __init__(self, points: dict[str, float]):
        self.dates = list(points)
        self.closes = [points[d] for d in self.dates]


# A flat-then-rising index: 100 through July, 110 from 2026-08-15 on.
BENCH = _Bench({"20260701": 100.0, "20260730": 100.0, "20260806": 100.0,
                "20260815": 110.0, "20260902": 110.0, "20260904": 110.0})


def _fill(date, symbol, side, shares, price, *, group=None, realized=None, ts=0.0):
    row = {"ts": ts, "date": date, "symbol": symbol, "side": side, "status": "filled",
           "shares": shares, "price": price, "gross": round(shares * price, 2)}
    if group:
        row["exposure_group"] = group
    if realized is not None:
        row["realized_pl"] = realized
    return row


def _prices(**px):
    return lambda s: px.get(s.upper())


# --- the sale, priced today ------------------------------------------------------------------------


def test_a_sale_is_priced_against_holding_and_against_the_index():
    trades = [
        _fill("2026-07-01", "AAA", "buy", 10, 10.0, group="AAA"),
        _fill("2026-07-30", "AAA", "sell", 10, 12.0, group="AAA", realized=20.0),
    ]
    out = ledger_cost(trades, [], price_of=_prices(AAA=15.0, **{"^GSPC": 110.0}), bench=BENCH, today=TODAY)
    s = out["sales"][0]
    assert s["proceeds_usd"] == 120.0 and s["realized_pl_usd"] == 20.0
    # Holding: 10 shares now worth $150 against $120 received — selling cost $30.
    assert s["held_instead_usd"] == 30.0
    # The index did +10% on $120 = $12 over the same window; the shares beat that by $18.
    assert s["vs_index_usd"] == 18.0
    assert s["holding_days"] == 29 and s["capital_gains"] == "short_term"


def test_a_sale_that_dodged_a_fall_reads_negative_on_both():
    trades = [
        _fill("2026-07-01", "AAA", "buy", 10, 10.0),
        _fill("2026-07-30", "AAA", "sell", 10, 12.0, realized=20.0),
    ]
    out = ledger_cost(trades, [], price_of=_prices(AAA=9.0, **{"^GSPC": 110.0}), bench=BENCH, today=TODAY)
    s = out["sales"][0]
    assert s["held_instead_usd"] == -30.0       # the sold shares are worth $30 LESS than the sale
    assert s["vs_index_usd"] == -42.0           # and the index would have made $12 on the proceeds


def test_totals_cover_every_sale_while_the_list_is_capped():
    trades = [_fill("2026-07-01", "AAA", "buy", 100, 10.0)]
    for i in range(_COST_MAX_SALES + 5):
        trades.append(_fill("2026-07-30", "AAA", "sell", 1, 12.0, realized=2.0, ts=float(i)))
    out = ledger_cost(trades, [], price_of=_prices(AAA=13.0, **{"^GSPC": 110.0}), bench=BENCH, today=TODAY)
    assert out["sales_count"] == _COST_MAX_SALES + 5
    assert len(out["sales"]) == _COST_MAX_SALES
    assert out["sales_summary"]["sales"] == _COST_MAX_SALES + 5
    assert out["sales_summary"]["held_instead_total_usd"] == round((13.0 - 12.0) * (_COST_MAX_SALES + 5), 2)
    assert any("showing the newest" in n for n in out["notes"])


def test_newest_sale_first():
    trades = [
        _fill("2026-07-01", "AAA", "buy", 10, 10.0),
        _fill("2026-07-30", "AAA", "sell", 5, 12.0, realized=10.0),
        _fill("2026-09-02", "AAA", "sell", 5, 12.0, realized=10.0),
    ]
    out = ledger_cost(trades, [], price_of=_prices(AAA=12.0), bench=BENCH, today=TODAY)
    assert [s["date"] for s in out["sales"]] == ["2026-09-02", "2026-07-30"]


# --- absence is a note, never a zero ---------------------------------------------------------------


def test_an_unpriced_sold_name_says_so_instead_of_reading_break_even():
    trades = [
        _fill("2026-07-01", "GONE", "buy", 10, 10.0),
        _fill("2026-07-30", "GONE", "sell", 10, 12.0, realized=20.0),
    ]
    out = ledger_cost(trades, [], price_of=_prices(**{"^GSPC": 110.0}), bench=BENCH, today=TODAY)
    s = out["sales"][0]
    assert s["held_instead_usd"] is None and s["vs_index_usd"] is None
    assert "no current price for GONE" in s["note"]
    assert "held_instead_total_usd" not in out["sales_summary"]
    assert any("could not be priced" in n for n in out["notes"])


def test_a_sale_older_than_the_benchmark_series_gets_no_index_comparison():
    trades = [
        _fill("2026-06-01", "AAA", "buy", 10, 10.0),
        _fill("2026-06-15", "AAA", "sell", 10, 12.0, realized=20.0),
    ]
    out = ledger_cost(trades, [], price_of=_prices(AAA=15.0, **{"^GSPC": 110.0}), bench=BENCH, today=TODAY)
    s = out["sales"][0]
    assert s["held_instead_usd"] == 30.0        # holding is still priceable
    assert s["vs_index_usd"] is None
    assert "benchmark level unavailable" in s["note"]


def test_no_benchmark_at_all_still_prices_holding():
    trades = [_fill("2026-07-01", "AAA", "buy", 10, 10.0),
              _fill("2026-07-30", "AAA", "sell", 10, 12.0, realized=20.0)]
    out = ledger_cost(trades, [], price_of=_prices(AAA=15.0), bench=None, today=TODAY)
    assert out["sales"][0]["held_instead_usd"] == 30.0
    assert out["sales"][0]["vs_index_usd"] is None


def test_an_empty_ledger_returns_none_so_the_caller_omits_the_block():
    """"You have sold nothing" is carried by sales_count == 0 on a ledger with buys; "no ledger was
    supplied" is the block's absence. They must not look alike."""
    assert ledger_cost([], [], price_of=_prices(), bench=BENCH, today=TODAY) is None
    cash_only = [{"date": "2026-09-01", "symbol": "CASH", "side": "deposit", "status": "filled",
                  "shares": 0, "price": None, "gross": 500.0}]
    assert ledger_cost(cash_only, [], price_of=_prices(), bench=BENCH, today=TODAY) is None


def test_a_ledger_with_buys_and_no_sales_says_zero_sales():
    trades = [_fill("2026-07-01", "AAA", "buy", 10, 10.0)]
    out = ledger_cost(trades, [{"symbol": "AAA", "shares": 10}], price_of=_prices(AAA=11.0, **{"^GSPC": 110.0}),
                      bench=BENCH, today=TODAY)
    assert out["sales_count"] == 0 and "sales" not in out and "realized" not in out


def test_skipped_orders_are_not_fills():
    trades = [_fill("2026-07-01", "AAA", "buy", 10, 10.0),
              {**_fill("2026-07-30", "AAA", "sell", 10, 12.0), "status": "skipped"}]
    out = ledger_cost(trades, [], price_of=_prices(AAA=15.0), bench=BENCH, today=TODAY)
    assert out["sales_count"] == 0


# --- tax lots: FIFO, and the short/long split in dollars -------------------------------------------


def test_realized_dollars_split_by_lot_age_at_the_sale():
    trades = [
        _fill("2025-06-01", "AAA", "buy", 10, 10.0),                       # 456 days old at the sale
        _fill("2026-07-01", "AAA", "buy", 10, 10.0),                       # 29 days old
        _fill("2026-07-30", "AAA", "sell", 15, 12.0, realized=30.0),       # takes all of lot 1, 5 of lot 2
    ]
    out = ledger_cost(trades, [], price_of=_prices(AAA=12.0), bench=BENCH, today=TODAY)
    s = out["sales"][0]
    assert s["capital_gains"] == "mixed"
    assert s["holding_days"] == 424                # FIFO: the oldest lot leads
    # $30 realised over 15 shares: 10 long-term shares = $20, 5 short-term = $10.
    assert out["realized"] == {"short_term_usd": 10.0, "long_term_usd": 20.0}


def test_a_long_held_lot_is_long_term():
    trades = [
        _fill("2025-06-01", "AAA", "buy", 10, 10.0),
        _fill("2026-07-30", "AAA", "sell", 10, 12.0, realized=20.0),
    ]
    out = ledger_cost(trades, [], price_of=_prices(AAA=12.0), bench=BENCH, today=TODAY)
    assert out["sales"][0]["holding_days"] >= _LONG_TERM_DAYS
    assert out["sales"][0]["capital_gains"] == "long_term"
    assert out["realized"] == {"short_term_usd": 0.0, "long_term_usd": 20.0}


def test_shares_no_buy_explains_are_reported_unmatched_not_classified_by_guess():
    trades = [_fill("2026-07-30", "AAA", "sell", 10, 12.0, realized=20.0)]     # ledger starts mid-position
    out = ledger_cost(trades, [], price_of=_prices(AAA=12.0), bench=BENCH, today=TODAY)
    s = out["sales"][0]
    assert s["capital_gains"] == "unknown" and s["unmatched_shares"] == 10.0
    assert "holding_days" not in s
    assert out["realized"]["unclassified_usd"] == 20.0
    assert out["realized"]["short_term_usd"] == 0.0


def test_fifo_leaves_the_newest_lots_open():
    trades = [
        _fill("2026-07-01", "AAA", "buy", 10, 10.0),
        _fill("2026-08-20", "AAA", "buy", 10, 20.0),
        _fill("2026-09-02", "AAA", "sell", 10, 25.0, realized=150.0),
    ]
    out = ledger_cost(trades, [{"symbol": "AAA", "shares": 10}], price_of=_prices(AAA=25.0, **{"^GSPC": 110.0}),
                      bench=BENCH, today=TODAY)
    pos = out["open_positions_vs_index"][0]
    assert pos["since"] == "2026-08-20"                 # the July lot is gone; the August lot remains
    assert pos["gain_usd"] == 50.0                      # 10 × (25 − 20)
    assert pos["vs_index_usd"] == 50.0                  # index flat since 2026-08-15's level


# --- round trips -----------------------------------------------------------------------------------


def test_a_sale_reversed_within_the_window_is_a_round_trip_with_what_it_paid():
    trades = [
        _fill("2026-08-13", "SCHD", "buy", 8, 34.5, group="SCHD"),
        _fill("2026-09-02", "SCHD", "sell", 8, 35.0275, group="SCHD", realized=4.16),
        _fill("2026-09-04", "SCHD", "buy", 10, 34.8224, group="SCHD"),
    ]
    out = ledger_cost(trades, [], price_of=_prices(SCHD=35.0), bench=BENCH, today=TODAY)
    assert out["round_trips_count"] == 1
    trip = out["round_trips"][0]
    assert trip["days_between"] == 2
    assert trip["paid_more_usd"] == round((34.8224 - 35.0275) * 8, 2)     # rebought cheaper, so negative


def test_a_different_vehicle_for_the_same_exposure_is_a_round_trip_without_a_price_comparison():
    trades = [
        _fill("2026-07-28", "IBIT", "buy", 4, 36.0, group="BTC"),
        _fill("2026-07-30", "IBIT", "sell", 4, 36.7014, group="BTC", realized=2.27),
        _fill("2026-08-06", "FBTC", "buy", 6, 56.023, group="BTC"),
    ]
    out = ledger_cost(trades, [], price_of=_prices(IBIT=40.0, FBTC=60.0), bench=BENCH, today=TODAY)
    trip = out["round_trips"][0]
    assert trip["sold"]["symbol"] == "IBIT" and trip["rebought"]["symbol"] == "FBTC"
    assert trip["paid_more_usd"] is None and "not comparable" in trip["note"]


def test_a_rebuy_after_the_window_is_not_a_round_trip():
    trades = [
        _fill("2026-04-01", "AAA", "buy", 1, 10.0, group="AAA"),
        _fill("2026-04-02", "AAA", "sell", 1, 10.0, group="AAA", realized=0.0),
        _fill("2026-09-01", "AAA", "buy", 1, 10.0, group="AAA"),     # 152 days later
    ]
    out = ledger_cost(trades, [], price_of=_prices(AAA=10.0), bench=BENCH, today=TODAY)
    assert out["round_trips_count"] == 0 and "round_trips" not in out


# --- open positions --------------------------------------------------------------------------------


def test_an_open_position_is_measured_against_the_index_on_the_same_dollars():
    trades = [_fill("2026-07-30", "AAA", "buy", 10, 10.0)]
    out = ledger_cost(trades, [{"symbol": "AAA", "shares": 10}], price_of=_prices(AAA=12.0, **{"^GSPC": 110.0}),
                      bench=BENCH, today=TODAY)
    pos = out["open_positions_vs_index"][0]
    assert pos["gain_usd"] == 20.0                # $100 → $120
    assert pos["vs_index_usd"] == 10.0            # the index turned $100 into $110


def test_a_position_the_lots_do_not_cover_says_how_much_they_cover():
    trades = [_fill("2026-07-30", "AAA", "buy", 4, 10.0)]
    out = ledger_cost(trades, [{"symbol": "AAA", "shares": 10}], price_of=_prices(AAA=12.0, **{"^GSPC": 110.0}),
                      bench=BENCH, today=TODAY)
    assert "cover 4 of 10 shares" in out["open_positions_vs_index"][0]["note"]


def test_an_unpriced_open_position_is_left_out_rather_than_marked_at_zero():
    trades = [_fill("2026-07-30", "AAA", "buy", 4, 10.0)]
    out = ledger_cost(trades, [{"symbol": "AAA", "shares": 4}], price_of=_prices(**{"^GSPC": 110.0}),
                      bench=BENCH, today=TODAY)
    assert "open_positions_vs_index" not in out


# --- the sold names the tick must price ------------------------------------------------------------


def test_sold_symbols_are_distinct_newest_first_across_arms_and_capped():
    main = [_fill("2026-09-02", "SCHD", "sell", 1, 1.0), _fill("2026-08-06", "SPY", "sell", 1, 1.0),
            _fill("2026-09-04", "SCHD", "buy", 1, 1.0)]
    fast = [_fill("2026-08-20", "VXUS", "sell", 1, 1.0), _fill("2026-08-06", "SPY", "sell", 1, 1.0),
            {**_fill("2026-09-03", "GLDM", "sell", 1, 1.0), "status": "skipped"}]
    assert sold_symbols([main, fast]) == ["SCHD", "VXUS", "SPY"]
    many = [_fill("2026-08-01", f"S{i}", "sell", 1, 1.0) for i in range(_SOLD_PRICE_MAX + 10)]
    assert len(sold_symbols([many])) == _SOLD_PRICE_MAX


def test_nothing_in_the_block_is_a_rate():
    """The panel's ruling, as a test: every leaf of the block is a count, a dollar figure, a date, a
    symbol, a share count, a price or a reason — never a ratio of own outcomes."""
    trades = [
        _fill("2026-07-01", "AAA", "buy", 10, 10.0, group="AAA"),
        _fill("2026-07-30", "AAA", "sell", 5, 12.0, group="AAA", realized=10.0),
        _fill("2026-08-20", "AAA", "buy", 5, 11.0, group="AAA"),
    ]
    out = ledger_cost(trades, [{"symbol": "AAA", "shares": 10}], price_of=_prices(AAA=12.0, **{"^GSPC": 110.0}),
                      bench=BENCH, today=TODAY)

    def keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield k
                yield from keys(v)
        elif isinstance(o, list):
            for v in o:
                yield from keys(v)
    for k in keys(out):
        assert not any(w in k for w in ("rate", "pct", "ratio", "beat", "win")), k
