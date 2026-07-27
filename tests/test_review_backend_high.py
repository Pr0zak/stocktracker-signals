"""Regression guards for the high-severity findings in the options/cycle review.

Every case here produced a confidently WRONG number that the user would have acted on — a fabricated
prior-cycle return, a bearish volume signal derived from a partial week, an assurance about earnings
risk that was never checked, or a premium quoted from a week-old print as though it were current.
"""
from __future__ import annotations

import datetime as dt
import time

from app import cycle, options


# ---------------------------------------------------------------- BTC cycle analog

def _weekly(start: dt.date, n: int, step: float = 1.0) -> tuple[list[str], list[float]]:
    dates = [(start + dt.timedelta(weeks=i)).isoformat() for i in range(n)]
    closes = [100.0 * (step ** i) for i in range(n)]
    return dates, closes


def test_cycle_analog_skips_a_cycle_the_series_cannot_cover():
    """The series is hard-coded to 10 years, so the earliest halving pair's target predates every
    bar. The lookup used to return bar 0 and report the 12 months from the START of the file as if
    it were that cycle's outcome — measured live, a fabricated 341.4% "best prior cycle"."""
    # A series starting well after the first halving pair, long enough to satisfy the length gate.
    dates, closes = _weekly(dt.date(2016, 7, 25), 520, step=1.004)
    out = cycle._cycle_analog(dates, closes, 56.9)
    assert out is not None
    # Three halving pairs exist, but only the two the series actually covers may be measured.
    assert out["prior_cycles_measured"] == 2, "a cycle outside the series was still counted"


def test_cycle_analog_still_measures_cycles_inside_the_series():
    dates, closes = _weekly(dt.date(2016, 7, 25), 520, step=1.004)
    out = cycle._cycle_analog(dates, closes, 56.9)
    assert out["median_fwd_12mo_pct"] > 0


# ---------------------------------------------------------------- partial weekly bar

def test_completed_weeks_drops_the_in_progress_bar():
    today = dt.date.today()
    dates = [(today - dt.timedelta(weeks=i)).isoformat() for i in range(4)][::-1]
    vols = [100.0, 100.0, 100.0, 5.0]          # last bar is this week, only partly filled
    (trimmed,) = cycle._completed_weeks(dates, vols)
    assert trimmed == [100.0, 100.0, 100.0], "the in-progress week was kept"


def test_volume_signal_is_not_skewed_by_the_partial_week():
    """rvol divided a PARTIAL week by an average of 14 FULL weeks, so rvol>2 was unreachable and
    rvol<0.5 fired almost every week. Measured live on AAPL: 0.19 and a false 'distribution'."""
    today = dt.date.today()
    n = 20
    dates = [(today - dt.timedelta(weeks=i)).isoformat() for i in range(n)][::-1]
    closes = [100.0 + i for i in range(n)]
    vols = [1000.0] * (n - 1) + [50.0]         # normal weeks, then a partial current week

    skewed = cycle._volume_signal(closes, vols)              # old behaviour (no dates)
    fixed = cycle._volume_signal(closes, vols, dates)        # completed weeks only
    assert skewed["rvol_14"] < 0.1, "premise: the partial week really does crush rvol"
    assert fixed["rvol_14"] == 1.0, "rvol should compare like with like once trimmed"


# ---------------------------------------------------------------- earnings claims

def test_unknown_earnings_is_never_reported_as_clear():
    """The live config has no Finnhub key, so earnings_date is None on every request. Reporting that
    as 'clear of the next earnings date' turns a missing input into a positive assurance about the
    one risk this strategy is most exposed to."""
    txt = options._wheel_rationale(35, 25, 45, earnings_in_window=False, earnings_known=False)
    assert "clear of the next earnings date" not in txt
    assert "unknown" in txt.lower()


def test_known_clear_earnings_still_says_so():
    txt = options._wheel_rationale(35, 25, 45, earnings_in_window=False, earnings_known=True)
    assert "clear of the next earnings date" in txt


def test_earnings_inside_the_window_is_flagged():
    txt = options._wheel_rationale(35, 25, 45, earnings_in_window=True, earnings_known=True)
    assert "earnings falls before it" in txt


# ---------------------------------------------------------------- stale last trade

def _contract(**kw) -> options.OptionContract:
    base = dict(type="call", contract_symbol="X", strike=100.0, expiration=0)
    base.update(kw)
    return options.OptionContract(**base)


def test_a_live_mid_is_not_treated_as_stale():
    # `mid` is a field populated by annotate_expiry, not derived on access — set it as that would.
    c = _contract(bid=1.0, ask=1.2, mid=1.1, last_price=0.5,
                  last_trade_epoch=int(time.time()) - 30 * 86400)
    price, src = options._limit_price_with_source(c)
    assert src == "mid" and price is not None
    assert options.stale_trade_age_days(c, time.time()) is None


def test_a_week_old_print_is_reported_as_stale():
    """With bid/ask at 0 the limit, yields, break-even and premium income all come from this print."""
    now = time.time()
    c = _contract(bid=0.0, ask=0.0, last_price=1.42, last_trade_epoch=int(now - 8 * 86400))
    price, src = options._limit_price_with_source(c)
    assert src == "last" and price == 1.42
    assert options.stale_trade_age_days(c, now) == 8.0


def test_a_print_from_over_the_weekend_is_not_flagged():
    now = time.time()
    c = _contract(bid=0.0, ask=0.0, last_price=1.42, last_trade_epoch=int(now - 2 * 86400))
    assert options.stale_trade_age_days(c, now) is None


def test_last_trade_date_is_actually_parsed():
    """It was dropped in _parse_contract, which made staleness unknowable at every call site."""
    raw = {"contractSymbol": "AAPL260727C00215000", "strike": 215.0, "expiration": 1785110400,
           "bid": 0.0, "ask": 0.0, "lastPrice": 2.5, "lastTradeDate": 1784500000}
    c = options._parse_contract(raw, "call")
    assert c.last_trade_epoch == 1784500000


# ---------------------------------------------------------------- IV history integrity

def test_implausible_iv_is_rejected(tmp_path, monkeypatch):
    """iv_rank is a min/max percentile with no eviction, so ONE bad point pins the scale forever."""
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    import importlib

    from app import options as o
    importlib.reload(o)
    assert o.append_iv_history("A", 0.00001) is False    # Yahoo's placeholder for an unquoted leg
    assert o.append_iv_history("A", 9.0) is False        # 900%
    assert o.append_iv_history("A", 0.28) is True        # a real reading


def test_iv_rank_refuses_to_compare_across_tenors():
    """The scan logs a ~37 DTE ATM IV; /options ranked a ~60 DTE reading against it. Implied vol is
    a term structure, so that percentile could land at the opposite end of the scale."""
    history = [0.20 + i * 0.001 for i in range(40)]
    same = options.iv_rank(history, current=0.30, current_dte=37, history_dte=37)
    across = options.iv_rank(history, current=0.30, current_dte=60, history_dte=37)
    assert same is not None, "a like-for-like comparison must still produce a rank"
    assert across is None, "a 23-day tenor gap must suppress the rank, not report one"


def test_iv_rank_still_works_when_tenor_is_unknown():
    """Rows written before the tenor was recorded have no dte — those must not break ranking."""
    history = [0.20 + i * 0.001 for i in range(40)]
    assert options.iv_rank(history, current=0.30, current_dte=60, history_dte=None) is not None
