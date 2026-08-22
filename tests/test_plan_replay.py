"""SWT-8 — the plan replay: what a recorded plan actually did next, without flattering itself.

The gap this closes is that nothing in this service measured the USER's decisions. memory.py scores
what the ANALYST said, against a fixed 20-day forward return with no stop and no target; the sandbox
scores the PAPER TRADER in its own ledger with its own money; the options tracker scores real option
positions only. So "are these plans actually helping" had no data behind it. The journal's mechanical
leg — this module — replays the plan exactly as written against the bars that followed, and the
user's own fills are scored against the same curve.

THE INCIDENT THESE TESTS EXIST TO PREVENT HAS NOT HAPPENED HERE YET, WHICH IS THE POINT. A backtest's
characteristic failure is not a crash, it is a curve that is quietly 20% too good, and the two ways
it usually gets there are both exercised below:

  1. THE INTRABAR AMBIGUITY. A single daily bar whose low reached the stop and whose high reached the
     target cannot say which came first. Resolving that in the trade's favour turns every violent
     reversal into a winner and leaves no trace in the output. This replay takes the STOP and sets
     `ambiguous: true`, and the ambiguous case is tested hardest: the same bar is worth +2.333R read
     optimistically and -1.0R read honestly, and the test pins BOTH numbers so the difference between
     the two policies is written down rather than assumed.

  2. AN INVENTED ENTRY. A plan whose zone price never traded was never a plan anyone could have
     taken, and filling it at the zone anyway manufactures trades out of the ones that got away —
     which are exactly the ones that ran. That is SWT-3's chase problem seen after the fact, and it
     comes back as `never_filled`, with no R.

And the house absence rule, which this module can break in a specific way: R is a RATIO OVER THE
RISK THE PLAN DEFINED, so a plan with no usable stop has no R at all. Not 0.0 — 0R is the real claim
that a trade made exactly what it risked. The same holds for a still-open plan and for a stop of
0.0, which in this app is what a null decoded through a non-nullable Double looks like (see
tests/test_chase.py for the "Stop $0 - target $0" incident that taught us).

All bars here are synthetic and there is no network. The worked arithmetic is deliberate: every R
below can be checked by hand from the plan's own numbers, so a change in the fill or exit convention
shows up as a test that has to be edited on purpose.
"""
from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from app.market import Series
from app.plan_replay import (
    DEFAULT_FILL_WINDOW_DAYS,
    DEFAULT_HORIZON_DAYS,
    bars_from_series,
    norm_date,
    replay,
)

# One plan, used throughout, with arithmetic that divides cleanly: filled at 101 against a stop of
# 98 the risk is exactly 3.00, so an exit at 98 is -1R, at 104 is +1R and at 110 is +3R.
ZONE = (100.0, 102.0)
STOP = 98.0
TARGET = 108.0


def bar(d, o, h, l, c):
    return {"date": d, "open": o, "high": h, "low": l, "close": c}


# The session that fills the plan: opens at 101 inside the zone, so the fill is 101 and the risk is 3.
FILL_BAR = bar("20260601", 101.0, 102.5, 100.0, 102.0)


def run(bars, **kw):
    kw.setdefault("entry", ZONE)
    kw.setdefault("stop", STOP)
    kw.setdefault("target", TARGET)
    return replay(bars, **kw)


# ------------------------------------------------------------------ the three clean outcomes

def test_a_plan_that_reaches_its_target_reports_the_target_with_r_measured_off_its_own_risk():
    out = run([
        FILL_BAR,
        bar("20260602", 103.0, 105.0, 102.0, 104.0),
        bar("20260603", 105.0, 109.0, 104.0, 108.5),   # high 109 >= target 108
    ])
    assert out["outcome"] == "target"
    assert out["entry_price"] == 101.0 and out["entry_date"] == "20260601"
    assert out["exit_price"] == 108.0, "the exit is the LEVEL, not the bar's high"
    assert out["exit_date"] == "20260603"
    assert out["bars_held"] == 3
    # (108 - 101) / (101 - 98) = 7 / 3
    assert out["r"] == 2.333
    assert out["return_pct"] == round((108.0 / 101.0 - 1) * 100, 3)
    assert out["ambiguous"] is False


def test_a_plan_that_reaches_its_stop_reports_exactly_minus_one_r():
    out = run([
        FILL_BAR,
        bar("20260602", 100.0, 101.0, 97.0, 97.5),     # low 97 <= stop 98
    ])
    assert out["outcome"] == "stop"
    assert out["exit_price"] == 98.0
    assert out["exit_date"] == "20260602"
    assert out["bars_held"] == 2
    # (98 - 101) / (101 - 98) = -1, by construction: a stop-out is one unit of the defined risk.
    assert out["r"] == -1.0
    assert out["ambiguous"] is False


def test_a_plan_that_reaches_neither_level_times_out_at_the_last_in_horizon_close():
    out = run([
        FILL_BAR,
        bar("20260602", 101.0, 103.0, 100.0, 102.0),
        bar("20260603", 102.0, 104.0, 101.0, 103.0),
    ], horizon_days=3)
    assert out["outcome"] == "time"
    assert out["exit_price"] == 103.0, "a time exit marks out at the close, the only price left"
    assert out["exit_date"] == "20260603"
    assert out["bars_held"] == 3
    # (103 - 101) / 3
    assert out["r"] == 0.667
    # A time exit is a real, measured result — it is the plan going nowhere, not an absence.
    assert out["reason"] is None and out["refused"] is False


# ------------------------------------------------------------------ the crux: the ambiguous bar

def test_a_bar_that_touched_both_levels_resolves_to_the_stop_and_says_so():
    """THE test. One session, low through the stop AND high through the target, no way to know the
    order. Read optimistically this trade is +2.333R; read honestly it is -1R. A backtest that picks
    the first number and prints no flag is 3.3R per occurrence too good and gives the reader nothing
    to notice it by."""
    both = bar("20260602", 101.0, 109.0, 97.0, 105.0)  # touches 108 above AND 98 below
    out = run([FILL_BAR, both])

    assert out["outcome"] == "stop", "the pessimistic reading is the load-bearing convention"
    assert out["ambiguous"] is True, "and the reader must be able to see that it was assumed"
    assert out["exit_price"] == 98.0
    assert out["r"] == -1.0

    # The size of the assumption, written down: this is what the optimistic reading would have paid.
    optimistic_r = round((TARGET - 101.0) / (101.0 - STOP), 3)
    assert optimistic_r == 2.333
    assert out["r"] != optimistic_r

    # And an UNambiguous bar of either kind must never carry the flag, or the flag means nothing.
    assert run([FILL_BAR, bar("20260602", 101.0, 103.0, 97.0, 98.0)])["ambiguous"] is False
    assert run([FILL_BAR, bar("20260602", 101.0, 109.0, 100.0, 108.5)])["ambiguous"] is False


def test_a_session_that_opens_through_the_stop_is_not_ambiguous_because_the_open_settles_the_order():
    """The open is the first price of the session, so a gap below the stop happened BEFORE anything
    else in that bar could — including a later run at the target. That is knowable, so it is not
    flagged, and the exit is the OPEN rather than the stop: a stop becomes a market order once it
    triggers, and claiming the stop price on a gap-down is the same optimism in smaller clothes."""
    out = run([FILL_BAR, bar("20260602", 95.0, 109.0, 94.0, 100.0)])
    assert out["outcome"] == "stop"
    assert out["ambiguous"] is False
    assert out["exit_price"] == 95.0, "filled at the gap, not at the stop it gapped past"
    # (95 - 101) / 3 — a loss LARGER than 1R, which is the honest cost of a gap.
    assert out["r"] == -2.0


def test_a_session_that_opens_above_the_target_takes_the_target_at_the_open_and_is_not_ambiguous():
    out = run([FILL_BAR, bar("20260602", 110.0, 111.0, 97.0, 100.0)])
    assert out["outcome"] == "target"
    assert out["ambiguous"] is False
    assert out["exit_price"] == 110.0, "gapped past the target — the fill is the open, in our favour"
    # (110 - 101) / 3
    assert out["r"] == 3.0


def test_the_fill_bars_own_open_cannot_resolve_the_order_because_it_happened_before_the_fill():
    """A subtle one. On the session the limit fills, the open is already in the past — the entry
    happened somewhere after it — so the open-settles-the-order shortcut must NOT be applied there.
    Here the fill bar opens above the target; treating that as a target hit would report a plan that
    took profit before it had a position."""
    opened_above = bar("20260601", 109.0, 110.0, 97.0, 100.0)  # traded down into the zone AND to the stop
    out = run([opened_above])
    assert out["entry_price"] == 102.0, "a limit at the zone top, filled on the way down"
    assert out["outcome"] == "stop", "the stop, not the target it opened above before we were in"
    assert out["ambiguous"] is True, "within the bar, after the fill, the order is genuinely unknown"
    assert out["bars_held"] == 1, "filled and stopped in one session is 1 bar held, never 0"
    # (98 - 102) / (102 - 98) = -1
    assert out["r"] == -1.0


# ------------------------------------------------------------------ entries that never happened

def test_a_plan_whose_price_never_traded_into_the_zone_never_filled_and_has_no_r():
    """SWT-3's chase problem after the fact: the plan said buy 100-102, the thing opened at 104 and
    left. Filling it anyway at an invented price manufactures a trade out of the very setups that
    ran away, which is a systematic upward bias, not a rounding error."""
    out = run([
        bar("20260601", 104.0, 106.0, 103.0, 105.0),
        bar("20260602", 106.0, 108.0, 105.0, 107.0),
        bar("20260603", 108.0, 112.0, 107.0, 111.0),
    ], fill_window_days=3)
    assert out["outcome"] == "never_filled"
    assert out["entry_price"] is None and out["entry_date"] is None
    assert out["exit_price"] is None and out["exit_date"] is None
    assert out["r"] is None, "no fill, no risk taken, no R — and emphatically not 0R"
    assert out["bars_held"] is None, "not 0 bars held: there was never a position to hold"
    assert out["reason"] and "never traded into the entry zone" in out["reason"]
    assert out["refused"] is False, "the plan was fine; the market simply never offered it"


def test_a_plan_left_working_past_the_fill_window_is_never_filled_even_though_price_came_back():
    """The zone gets touched, but a week and a half later. The plan was drawn against a specific
    day's indicators, and counting a fill that far out both invents trades and biases them: the
    zones that eventually get revisited are disproportionately the ones price fell hard into."""
    away = [bar(f"2026060{i}", 104.0, 106.0, 103.0, 105.0) for i in range(1, 7)]
    late = bar("20260607", 103.0, 104.0, 101.0, 102.0)   # finally touches the zone, session 7
    assert run(away + [late], fill_window_days=5)["outcome"] == "never_filled"
    filled = run(away + [late], fill_window_days=7)
    assert filled["outcome"] == "open" and filled["entry_price"] == 102.0


def test_a_fill_window_that_has_not_closed_yet_cannot_claim_the_plan_never_filled():
    """`never_filled` is a CLAIM — that the market never offered this plan — and two sessions into a
    five-session window it is a claim about days that have not happened. The zone may well be touched
    tomorrow, and a journal that filed this as a missed trade would be wrong on its own timeline."""
    away = [bar("20260601", 104.0, 106.0, 103.0, 105.0), bar("20260602", 106.0, 108.0, 105.0, 107.0)]
    pending = run(away, fill_window_days=5)
    assert pending["outcome"] is None, "not never_filled: the order is still working"
    assert pending["refused"] is False and pending["r"] is None
    assert pending["reason"] and "still open" in pending["reason"]
    # Once the window HAS closed on the same tape, the claim becomes sayable.
    assert run(away, fill_window_days=2)["outcome"] == "never_filled"


def test_a_session_that_opens_through_the_stop_before_the_entry_voids_the_plan_rather_than_scoring_it_flat():
    """Price gapped below the whole plan overnight. A resting limit would technically have filled at
    the open and a stop-market would have exited at the same open — which computes to a tidy 0.0R,
    a number that says "this trade made exactly what it risked" about a setup that was destroyed
    before the bell. Say it was never entered, and say why."""
    out = run([bar("20260601", 96.0, 103.0, 95.0, 99.0)])
    assert out["outcome"] == "never_filled"
    assert out["r"] is None and out["r"] != 0.0
    assert out["entry_price"] is None
    assert out["reason"] and "stop" in out["reason"] and "20260601" in out["reason"]


# ------------------------------------------------------------------ absent risk, absent exits

def test_a_plan_with_no_stop_still_reports_its_outcome_but_has_no_r_at_all():
    out = run([FILL_BAR, bar("20260602", 105.0, 109.0, 104.0, 108.5)], stop=None)
    assert out["outcome"] == "target"
    assert out["exit_price"] == 108.0, "the outcome is knowable without a stop; only the R is not"
    assert out["return_pct"] == round((108.0 / 101.0 - 1) * 100, 3)
    assert out["r"] is None, "no risk defined means no R"
    assert out["r"] != 0.0


def test_a_stop_of_zero_is_read_as_absent_rather_than_as_a_level_at_zero():
    """0.0 is what a null looks like after a non-nullable-Double decoder, which is precisely how
    "Stop $0 - target $0" once shipped onto a money decision (tests/test_chase.py). A zero taken
    literally here is worse than cosmetic: it makes the risk the entire entry price, so every
    stop-out reads as a 3% loss instead of -1R, and nothing ever stops out."""
    out = run([FILL_BAR, bar("20260602", 100.0, 101.0, 97.0, 97.5)], stop=0.0)
    assert out["r"] is None
    assert out["outcome"] == "open", "with no stop, a bar through 98 is not an exit"
    for absent in (0.0, -5.0, None, float("nan"), "n/a"):
        assert run([FILL_BAR], stop=absent)["r"] is None


def test_a_still_open_plan_reports_open_with_no_exit_price_and_no_fabricated_mark_in_the_exit():
    out = run([FILL_BAR, bar("20260602", 102.0, 104.0, 101.0, 103.5)], horizon_days=10)
    assert out["outcome"] == "open"
    assert out["exit_price"] is None and out["exit_date"] is None
    assert out["r"] is None, "an unrealized R is not an R"
    assert out["return_pct"] is None
    assert out["bars_held"] == 2, "the clock is running and says how far it has run"
    # Where it stands IS reportable — under a name that cannot be mistaken for a fill.
    assert out["mark_price"] == 103.5 and out["mark_date"] == "20260602"
    assert out["reason"] and "still open" in out["reason"]


# ------------------------------------------------------------------ the horizon boundary

def test_the_last_in_horizon_bar_still_counts_and_the_one_after_it_does_not():
    """Off-by-one on the horizon is invisible in aggregate and changes every trade: an extra bar is
    an extra free look at the target. Pinned from both sides."""
    quiet = [FILL_BAR, bar("20260602", 101.0, 103.0, 100.0, 102.0)]
    hit = bar("20260603", 102.0, 109.0, 101.0, 108.5)      # target, on the 3rd session of the trade
    late = bar("20260604", 102.0, 109.0, 101.0, 108.5)     # the same session, one bar too late

    on_the_line = run(quiet + [hit], horizon_days=3)
    assert on_the_line["outcome"] == "target" and on_the_line["bars_held"] == 3

    over_the_line = run(quiet + [bar("20260603", 102.0, 103.0, 101.0, 102.5), late], horizon_days=3)
    assert over_the_line["outcome"] == "time", "bar 4 is outside a 3-session horizon and is not consulted"
    assert over_the_line["exit_price"] == 102.5 and over_the_line["exit_date"] == "20260603"
    assert over_the_line["bars_held"] == 3


def test_the_horizon_counts_from_the_fill_not_from_the_plan_date():
    # Two sessions above the zone, then the fill. A horizon measured from the plan date would give
    # this trade one session; measured from the fill it gets its full three.
    away = [bar("20260601", 104.0, 106.0, 103.0, 105.0), bar("20260602", 104.0, 106.0, 103.0, 105.0)]
    out = run(away + [FILL_BAR, bar("20260604", 101.0, 103.0, 100.0, 102.0),
                      bar("20260605", 102.0, 109.0, 101.0, 108.5)], horizon_days=3)
    assert out["outcome"] == "target" and out["entry_date"] == "20260601"
    assert out["bars_held"] == 3


# ------------------------------------------------------------------ plans that cannot be replayed

@pytest.mark.parametrize("kw, fragment", [
    ({"entry": (None, None)}, "no entry level"),
    ({"entry": (0.0, 0.0)}, "no entry level"),
    ({"entry": (102.0, 100.0)}, "inverted"),
    ({"stop": 100.0}, "not below the entry zone"),      # a stop INSIDE the zone is not a stop
    ({"stop": 105.0}, "not below the entry zone"),
    ({"target": 101.0}, "long-only"),                   # a target under the entry is a short, or a typo
    ({"horizon_days": 0}, "horizon_days"),
    ({"fill_window_days": 0}, "fill_window_days"),
])
def test_an_unusable_plan_is_refused_with_a_stated_reason_and_never_replayed_anyway(kw, fragment):
    out = run([FILL_BAR], **kw)
    assert out["refused"] is True
    assert out["outcome"] is None, "no outcome is a better answer than one measured off broken levels"
    assert out["r"] is None and out["exit_price"] is None
    assert fragment in (out["reason"] or "")


def test_no_bars_at_all_is_not_a_refusal_and_not_an_outcome():
    out = run([])
    assert out["outcome"] is None and out["refused"] is False
    assert out["reason"] and "no usable bars" in out["reason"]
    assert out["bars_seen"] == 0 and out["r"] is None


def test_bars_missing_prices_are_dropped_and_counted_rather_than_walked_around_silently():
    """Yahoo nulls the odd session. Skipping them quietly would let a replay claim it walked a window
    it only partly saw — and a hole on the wrong day is a stop that never triggered."""
    holed = [
        bar("20260601", 101.0, None, 100.0, 102.0),        # no high
        FILL_BAR,
        {"date": "20260603", "open": 101.0, "high": 99.0, "low": 103.0, "close": 100.0},  # high < low
        bar("20260604", 105.0, 109.0, 104.0, 108.5),
    ]
    out = run(holed)
    assert out["bars_skipped"] == 2 and out["bars_seen"] == 2
    assert out["outcome"] == "target" and out["entry_date"] == "20260601"

    allbad = replay([bar("20260601", None, None, None, None)], entry=ZONE, stop=STOP, target=TARGET)
    assert allbad["outcome"] is None and allbad["bars_skipped"] == 1


def test_the_defaults_are_present_on_every_result_so_a_reader_knows_what_clock_was_used():
    out = run([FILL_BAR])
    assert out["horizon_days"] == DEFAULT_HORIZON_DAYS
    assert out["fill_window_days"] == DEFAULT_FILL_WINDOW_DAYS
    # Guards on the constants themselves. The reference this came from closes at ten sessions; this
    # app's plans are ~5%-down / ~8%-up, which ten sessions rarely resolves, and a horizon that spans
    # more than one earnings cycle (~63 sessions) stops being a study of the plan.
    assert 15 <= DEFAULT_HORIZON_DAYS <= 63
    assert 1 <= DEFAULT_FILL_WINDOW_DAYS <= 10


def test_a_scalar_entry_is_a_zone_of_zero_width_and_a_dict_plan_is_read_as_written():
    at_limit = replay([bar("20260601", 103.0, 104.0, 101.0, 102.0)],
                      entry=102.0, stop=STOP, target=TARGET)
    assert at_limit["entry_price"] == 102.0, "a single limit price fills at the limit"
    as_dict = replay([FILL_BAR], entry={"entry_low": 100.0, "entry_high": 102.0},
                     stop=STOP, target=TARGET)
    assert as_dict["entry_price"] == 101.0


# ------------------------------------------------------------------ slicing a Series into bars

class _StubSeries:
    """Deliberately not a real Series — bars_from_series must read a partial object without raising,
    the same way swing.metrics does, because a stubbed series in an unrelated test should not take a
    route down with an AttributeError."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_bars_from_series_starts_strictly_after_the_plan_date_because_that_bar_was_already_known():
    """The lookahead guard. A plan is written from a session's closing snapshot, so letting the
    replay trade that same session's range hands every trade one free bar of hindsight — the analyst
    had already seen that high, that low and that close when it drew the levels."""
    s = _StubSeries(
        dates=["20260529", "20260601", "20260602"],
        opens=[10.0, 11.0, 12.0], highs=[10.5, 11.5, 12.5],
        lows=[9.5, 10.5, 11.5], closes=[10.0, 11.0, 12.0],
    )
    out = bars_from_series(s, "20260601")
    assert [b["date"] for b in out] == ["20260602"]
    assert out[0] == {"date": "20260602", "open": 12.0, "high": 12.5, "low": 11.5, "close": 12.0}
    # ISO in, same answer out — both forms turn up at call sites.
    assert bars_from_series(s, "2026-06-01") == out


def test_bars_from_series_survives_a_series_with_no_highs_or_lows_at_all():
    # market.Series defaults highs/lows to empty lists for callers that predate them (see the field
    # comment there). Those bars come back unusable and get counted, not silently walked.
    s = _StubSeries(dates=["20260601", "20260602"], opens=[1.0, 2.0], closes=[1.0, 2.0])
    bars = bars_from_series(s, "20260531")
    assert len(bars) == 2 and bars[0]["high"] is None
    assert replay(bars, entry=ZONE, stop=STOP, target=TARGET)["bars_skipped"] == 2
    assert bars_from_series(_StubSeries(), "20260601") == []


def test_norm_date_refuses_anything_that_is_not_a_date_instead_of_guessing():
    assert norm_date("2026-06-01") == "20260601"
    assert norm_date("20260601") == "20260601"
    for bad in ("", None, "yesterday", "2026"):
        with pytest.raises(ValueError):
            norm_date(bad)


# ------------------------------------------------------------------ the /journal/replay route

@pytest.fixture
def journal(tmp_path, monkeypatch):
    """A TestClient whose /journal/replay runs against a settable fake price series and a frozen
    'today', so the route's date arithmetic is deterministic."""
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    import app.settings_store as st
    importlib.reload(st)
    import app.main as m
    importlib.reload(m)

    state: dict = {"series": None, "error": None}

    async def fake_fetch(client, symbol, rng="1y", *, fallback=True):
        if state["error"]:
            raise RuntimeError("https://query1.finance.yahoo.com/v8/finance/chart/ABC — 429")
        return state["series"]

    monkeypatch.setattr(m, "fetch_series", fake_fetch)
    monkeypatch.setattr(m.sandbox_job, "today_et_str", lambda *a, **k: "2026-06-10")
    with TestClient(m.app) as c:
        yield c, state


def _series(bars):
    return Series(
        symbol="ABC",
        closes=[b["close"] for b in bars],
        opens=[b["open"] for b in bars],
        volumes=[None] * len(bars),
        dates=[b["date"] for b in bars],
        fifty_two_high=None, fifty_two_low=None, currency="USD",
        highs=[b["high"] for b in bars],
        lows=[b["low"] for b in bars],
    )


def _body(**kw):
    base = {"symbol": "ABC", "date": "20260531", "entry_low": 100.0, "entry_high": 102.0,
            "stop": 98.0, "target": 108.0}
    base.update(kw)
    return base


def test_the_route_replays_the_plan_against_the_sessions_after_the_date_it_was_recorded(journal):
    client, state = journal
    state["series"] = _series([
        bar("20260531", 99.0, 120.0, 90.0, 99.0),      # the plan's OWN session — must not be traded
        FILL_BAR,
        bar("20260602", 105.0, 109.0, 104.0, 108.5),
    ])
    r = client.post("/journal/replay", json=_body())
    assert r.status_code == 200
    body = r.json()
    assert body["outcome"] == "target" and body["r"] == 2.333
    assert body["entry_date"] == "20260601", "the plan-date bar would have filled at 102 and hit both levels"
    assert body["bars_from"] == "20260601" and body["bars_to"] == "20260602"
    assert body["symbol"] == "ABC" and body["as_of"] == "20260531"


def test_the_route_refuses_a_plan_it_cannot_replay_rather_than_returning_an_empty_result(journal):
    client, state = journal
    state["series"] = _series([FILL_BAR])
    assert client.post("/journal/replay", json=_body(date="not-a-date")).status_code == 422
    assert client.post("/journal/replay", json=_body(date="20260901")).status_code == 422  # future
    inverted = client.post("/journal/replay", json=_body(entry_low=102.0, entry_high=100.0))
    assert inverted.status_code == 422 and "invert" in inverted.json()["detail"]
    bad_stop = client.post("/journal/replay", json=_body(stop=101.0))
    assert bad_stop.status_code == 422 and "not below the entry zone" in bad_stop.json()["detail"]


def test_a_symbol_with_no_history_is_a_404_and_a_failed_fetch_is_a_502_that_leaks_no_url(journal):
    client, state = journal
    state["series"] = _series([])
    assert client.post("/journal/replay", json=_body()).status_code == 404

    state["error"] = True
    r = client.post("/journal/replay", json=_body())
    assert r.status_code == 502
    # The upstream message carries the Yahoo chart URL. 502 says the fetch failed and nothing else.
    assert "yahoo" not in r.json()["detail"].lower() and "http" not in r.json()["detail"].lower()


def test_a_plan_recorded_today_reports_nothing_to_replay_yet_rather_than_an_outcome(journal):
    client, state = journal
    state["series"] = _series([bar("20260610", 101.0, 102.0, 100.0, 101.0)])
    body = client.post("/journal/replay", json=_body(date="20260610")).json()
    assert body["outcome"] is None, "no session has traded since — that is not an outcome"
    assert body["refused"] is False
    assert "nothing to replay yet" in body["reason"]
    assert body["r"] is None and body["bars_from"] is None
