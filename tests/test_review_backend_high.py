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


# ---------------------------------------------------------------- medium-severity batch

def test_dte_is_measured_to_the_real_expiry_moment():
    """Yahoo's `expiration` is MIDNIGHT UTC on the expiry date, but options expire at 16:00 ET —
    so `(ts - now)/86400` read a full calendar day short through the entire US session. Measured
    live, today's nearest expiry computed as -1 DTE while it still had ~7.5 hours to run."""
    import datetime as dt

    # An expiry stamped midnight UTC on a given date.
    d = dt.date(2026, 8, 21)
    ts = int(dt.datetime.combine(d, dt.time(0, 0), tzinfo=dt.timezone.utc).timestamp())
    true_moment = options.expiry_epoch(ts)
    # The real expiry is 16:00 ET that day — 20 or 21 hours after the stamped timestamp.
    gap_hours = (true_moment - ts) / 3600.0
    assert 19.5 <= gap_hours <= 21.5, gap_hours


def test_assignment_probability_is_not_delta_for_a_put():
    """|delta| = N(-d1) is the common shorthand, but the risk-neutral chance of assignment is
    N(-d2) — strictly larger for a put, so the shorthand made the strategy look safer than it is."""
    g = options.black_scholes_greeks(spot=100.0, strike=93.0, t=35 / 365, r=0.043,
                                     sigma=0.30, is_call=False)
    assert g["prob_itm"] > abs(g["delta"]), "N(-d2) must exceed |delta| for a put"


def test_assignment_probability_is_below_delta_for_a_call():
    g = options.black_scholes_greeks(spot=100.0, strike=107.0, t=35 / 365, r=0.043,
                                     sigma=0.30, is_call=True)
    assert g["prob_itm"] < g["delta"], "N(d2) must be below delta for a call"


def test_session_phase_respects_a_half_day():
    """market_now hard-coded a 16:00 close, so it reported REGULAR for hours after the market shut
    on the three 1pm NYSE half-days a year."""
    import datetime as dt
    from zoneinfo import ZoneInfo

    from app.market_now import session_phase
    et = ZoneInfo("America/New_York")
    assert session_phase(dt.datetime(2026, 11, 27, 14, 30, tzinfo=et)) == "AFTER"   # half-day
    assert session_phase(dt.datetime(2026, 11, 30, 14, 30, tzinfo=et)) == "REGULAR"  # normal day
    assert session_phase(dt.datetime(2026, 11, 27, 12, 0, tzinfo=et)) == "REGULAR"   # before 1pm


def test_debit_spread_is_sized_on_the_same_budget_as_the_long_call():
    """Both blocks appear in one body under identical key names (`cost`, `max_loss`), so quoting the
    spread as a single lot against a budget-scaled long call made the alternative look far cheaper
    than the thing it is an alternative to."""
    from app.options import ExpiryChain, OptionContract

    def call(strike, mid):
        return OptionContract(type="call", contract_symbol=f"C{strike}", strike=strike,
                              expiration=0, bid=mid - 0.05, ask=mid + 0.05, mid=mid)

    expiry = ExpiryChain(expiration=0, expiration_iso="2026-08-21",
                         calls=[call(100.0, 5.0), call(110.0, 4.0)], puts=[])
    ref = {"contract_symbol": "C100.0"}
    one = options._debit_spread(expiry, ref, target_price=None, budget=None)
    many = options._debit_spread(expiry, ref, target_price=None, budget=1000.0)
    assert one["spreads"] is None, "no budget -> quantity is the caller's problem, and stated as such"
    assert many["spreads"] == 10, "a $1000 budget buys ten $100 spreads"
    assert many["cost"] == one["cost"] * 10
    assert many["max_loss"] == one["max_loss"] * 10
    assert many["max_profit"] == one["max_profit"] * 10
    assert many["breakeven"] == one["breakeven"], "breakeven is per-share and must not scale"
