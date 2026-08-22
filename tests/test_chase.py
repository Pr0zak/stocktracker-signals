"""SWT-3 — the chase read: how far above its own entry zone the thing is actually trading.

The incident this closes is a rendering one, not a math one. The detail screen showed a plan with an
entry zone of $68.67-$70.73 and, a few millimetres away, a live price of $74.02. Both numbers were
correct and the screen said nothing, because saying something required the reader to divide one by
the other on a phone with the buy button in reach. 4.65% above the top of the zone is a chase; two
correct numbers side by side are not a warning.

The second thing under test is the absence rule, and it is the one that matters more. The levels are
nullable on purpose: an analyst that will not commit to a zone returns null, and the app's EntryPlan
models entry_low/entry_high as nullable Doubles because they were once non-nullable and defaulted to
0.0 — which rendered as a confident "Stop $0 · target $0" on a money decision. A chase read built on
a missing zone must come out as None, NOT as 0.0% and NOT as the reassuring "ok". Zero here reads as
"you are exactly at the top of the zone", which is a specific and false claim, and "ok" reads as
permission. The same holds for a missing quote and for a zone the model returned inverted.

The threshold itself is CHASE_OK_PCT and the boundary is pinned exactly here, so that widening the
tolerated overshoot is a deliberate edit to a test rather than a number that drifts.
"""
from __future__ import annotations

import importlib
import math

import pytest
from fastapi.testclient import TestClient

from app.analyst import EntryPlan
from app.chase import CHASE_OK_PCT, annotate, read

# The plan from the incident: a real zone, and the price it was actually trading at.
LOW, HIGH, CHASING_PRICE = 68.67, 70.73, 74.02


def test_a_price_inside_the_zone_is_in_zone_and_never_reads_as_a_chase():
    r = read(69.50, LOW, HIGH)
    assert r.status == "in_zone"
    # Measured against the TOP of the zone, so anything inside it is at or under zero. A positive
    # number on a price the plan is happy to pay would be the warning firing on a non-event.
    assert r.pct is not None and r.pct <= 0
    assert r.warning is None


def test_a_price_exactly_at_the_top_of_the_zone_is_still_in_the_zone():
    # The boundary belongs to the zone: the plan named this price as one it would pay.
    r = read(HIGH, LOW, HIGH)
    assert r.status == "in_zone"
    assert r.pct == 0.0


def test_a_price_below_the_entry_low_is_flagged_as_cheap_and_not_as_a_problem():
    r = read(65.00, LOW, HIGH)
    assert r.status == "below_zone"
    # Cheaper than the analyst planned to pay. There is nothing to warn about, and the sandbox agrees
    # — it fills buys below the zone happily. So: no warning, and a negative (not absent) distance.
    assert r.warning is None
    assert r.pct is not None and r.pct < 0


def test_a_small_overshoot_is_ok_because_the_band_is_quote_noise_not_permission():
    r = read(HIGH * 1.005, LOW, HIGH)   # half a percent over — inside the delayed-quote wobble
    assert r.status == "ok"
    assert r.pct == 0.5


def test_the_incident_price_is_chase_too_deep():
    r = read(CHASING_PRICE, LOW, HIGH)
    assert r.status == "chase_too_deep"
    assert r.pct == 4.65
    assert r.warning is None            # the read worked; it is the trade that is the problem


def test_the_ok_boundary_sits_exactly_at_chase_ok_pct_and_the_next_step_is_too_deep():
    # Pinned exactly, in both directions, so the tolerated overshoot cannot drift silently.
    at = read(HIGH * (1 + CHASE_OK_PCT / 100.0), LOW, HIGH)
    assert at.pct == CHASE_OK_PCT and at.status == "ok"

    just_past = read(HIGH * (1 + (CHASE_OK_PCT + 0.01) / 100.0), LOW, HIGH)
    assert just_past.pct == round(CHASE_OK_PCT + 0.01, 2) and just_past.status == "chase_too_deep"

    # And the status is derived from the SAME rounded number the screen shows: a card reading
    # "1.5% above the zone" must never be labelled chase_too_deep beside it.
    hair_over = read(HIGH * (1 + (CHASE_OK_PCT + 0.004) / 100.0), LOW, HIGH)
    assert hair_over.pct == CHASE_OK_PCT and hair_over.status == "ok"


def test_a_missing_entry_high_yields_no_percentage_and_no_status_at_all():
    # THE central case. The analyst declined to name a zone; every field of the read is unknowable.
    for low in (LOW, None):
        r = read(CHASING_PRICE, low, None)
        assert r.pct is None, "a missing zone top must not become a percentage"
        assert r.status is None, "a missing zone top must not become a status"
        # Spelled out, because these are the two specific wrong answers that shipped before:
        assert r.pct != 0.0 and r.status != "ok"
        assert r.warning is None   # not naming a zone is a legitimate answer, not a defect


def test_a_zero_or_negative_entry_high_is_absent_rather_than_a_division():
    # 0.0 is what a non-nullable-Double decoder produces from a null, so it arrives meaning "absent"
    # — and it is also the one value that would make the percentage a ZeroDivisionError.
    for hi in (0.0, -5.0):
        r = read(CHASING_PRICE, None, hi)
        assert (r.pct, r.status) == (None, None)


def test_a_non_finite_zone_or_price_is_absent_rather_than_a_nan_percentage():
    # NaN propagates through the arithmetic silently and then compares False against every
    # threshold, which would hand back a NaN pct with a status attached.
    assert read(CHASING_PRICE, LOW, float("nan")).pct is None
    assert read(float("nan"), LOW, HIGH).status is None
    assert read(CHASING_PRICE, LOW, float("inf")).pct is None
    r = read("not a number", LOW, HIGH)
    assert (r.pct, r.status) == (None, None)


def test_an_inverted_zone_is_refused_with_a_warning_instead_of_a_nonsense_percentage():
    # low above high means the model swapped its own two numbers, so neither one means what it says.
    # The arithmetic still "works" — it just measures against the wrong end of a broken zone.
    r = read(CHASING_PRICE, HIGH, LOW)
    assert (r.pct, r.status) == (None, None)
    assert r.warning is not None and "invert" in r.warning.lower()


def test_a_missing_price_yields_both_fields_none():
    for px in (None, 0.0, -1.0):
        r = read(px, LOW, HIGH)
        assert (r.pct, r.status) == (None, None)
        assert r.warning is None   # a quote we could not fetch is an absence, not a broken plan


def test_a_known_top_with_an_unknown_bottom_still_warns_about_paying_up():
    # Half a zone is enough to answer "am I above the top?", which is the question that costs money.
    assert read(CHASING_PRICE, None, HIGH).status == "chase_too_deep"
    # But under the top, "inside the zone" and "cheaper than the zone" are indistinguishable without
    # the floor, so the honest answer is the distance with no label — never a guessed one.
    under = read(69.50, None, HIGH)
    assert under.status is None
    assert under.pct == round((69.50 / HIGH - 1) * 100, 2)


def test_annotate_attaches_all_four_keys_to_a_plan_payload_even_when_the_read_is_absent():
    payload = {"symbol": "ABC", "plan": {"entry_low": LOW, "entry_high": HIGH}}
    out = annotate(payload, CHASING_PRICE)
    assert out["chase_status"] == "chase_too_deep"
    assert out["chase_pct"] == 4.65
    assert out["chase_price"] == CHASING_PRICE
    assert out["chase_warning"] is None
    assert payload.get("chase_pct") is None, "annotate must not mutate the cached payload"

    # Present-and-null, not missing: a client can tell "we looked and could not say" apart from an
    # older server that never had the field.
    blind = annotate({"symbol": "ABC", "plan": {"entry_low": None, "entry_high": None}}, CHASING_PRICE)
    assert blind["chase_pct"] is None and blind["chase_status"] is None
    assert "chase_pct" in blind and "chase_status" in blind
    assert blind["chase_price"] == CHASING_PRICE   # we had a price; it was the zone that was absent

    # A payload whose plan is missing entirely must not raise on the way out of the route.
    empty = annotate({"symbol": "ABC"}, None)
    assert (empty["chase_pct"], empty["chase_status"], empty["chase_price"]) == (None, None, None)


def test_the_ok_band_is_small_enough_that_it_cannot_swallow_a_real_chase():
    # A guard on the constant itself: the band exists to absorb quote noise (about a percent), and a
    # plan's whole risk:reward edge is spent within a few percent of the zone top. Anything wider
    # than a couple of percent would let a genuinely chased entry read as fine.
    assert 0.5 <= CHASE_OK_PCT <= 2.0
    assert math.isfinite(CHASE_OK_PCT)


# ------------------------------------------------------- the /plan route wiring

@pytest.fixture
def planned(tmp_path, monkeypatch):
    """A TestClient whose /plan route runs against a stub analyst and a settable quote.

    Yields (client, quote) where `quote` is a mutable dict holding the price the fake feed serves —
    so a test can move the market between two calls to the SAME cached plan — and the zone the stub
    analyst returns.
    """
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    import app.settings_store as st
    importlib.reload(st)
    import app.main as m
    importlib.reload(m)

    async def fake_snapshot(symbol, *, crypto=False, bench_closes=None):
        return {"symbol": symbol.upper(), "price": 69.00}

    quote: dict = {"price": 69.00, "zone": (LOW, HIGH)}

    async def fake_plan_entry(summary, *, cash, deep=False):
        lo, hi = quote["zone"]
        entry = EntryPlan(
            symbol=summary["symbol"], action="buy_on_pullback", conviction=60,
            entry_low=lo, entry_high=hi, suggested_shares=1.0, allocation_usd=100.0,
            stop=65.0, target=80.0, timing="on a pullback", thesis="t",
        )
        return entry, {"model": "stub", "input_tokens": 1, "output_tokens": 1, "cost_usd": 0.0}

    async def fake_quotes(http, syms):
        if quote.get("price") is None:
            raise RuntimeError("quote feed down")
        return {s.upper(): {"price": quote["price"], "state": "REGULAR"} for s in syms}

    monkeypatch.setattr(m, "_snapshot", fake_snapshot)
    monkeypatch.setattr(m, "plan_entry", fake_plan_entry)
    monkeypatch.setattr(m.market_now, "fetch_quotes", fake_quotes)
    with TestClient(m.app) as c:
        yield c, quote


def test_the_plan_route_carries_the_chase_read_for_the_price_it_is_serving(planned):
    client, quote = planned
    quote["price"] = CHASING_PRICE
    body = client.get("/plan/ABC", params={"cash": 1000}).json()
    assert body["chase_status"] == "chase_too_deep"
    assert body["chase_pct"] == 4.65
    assert body["chase_price"] == CHASING_PRICE


def test_a_cached_plan_re_reads_the_chase_against_todays_price_not_the_cached_one(planned):
    # The reason the read is attached on the way OUT of the route instead of being stored: plans
    # cache for verdict_ttl_seconds (4h by default), and a four-hour-old "you're inside the zone" is
    # the exact stale-confident-number defect this codebase keeps deleting. Same plan, moved market.
    client, quote = planned
    quote["price"] = 69.00
    first = client.get("/plan/ABC", params={"cash": 1000}).json()
    assert first["cached"] is False and first["chase_status"] == "in_zone"

    quote["price"] = CHASING_PRICE
    second = client.get("/plan/ABC", params={"cash": 1000}).json()
    assert second["cached"] is True, "the plan itself should still be the cached one"
    assert second["plan"] == first["plan"]
    assert second["chase_status"] == "chase_too_deep" and second["chase_pct"] == 4.65


def test_a_cached_plan_whose_quote_cannot_be_fetched_reports_no_read_rather_than_a_stale_one(planned):
    client, quote = planned
    quote["price"] = 69.00
    client.get("/plan/ABC", params={"cash": 1000})

    quote["price"] = None            # feed down on the second call
    body = client.get("/plan/ABC", params={"cash": 1000}).json()
    assert body["cached"] is True
    assert body["chase_pct"] is None and body["chase_status"] is None
    assert body["chase_price"] is None
    # And the plan itself still comes back — a missing quote costs the read, never the response.
    assert body["plan"]["entry_high"] == HIGH


def test_a_fresh_plan_falls_back_to_the_snapshot_price_when_the_quote_feed_is_down(planned):
    # Only on this path: the snapshot was fetched seconds ago and is the very print the analyst drew
    # the zone against, so it is a real answer rather than a stand-in.
    client, quote = planned
    quote["price"] = None
    body = client.get("/plan/ABC", params={"cash": 1000}).json()
    assert body["cached"] is False
    assert body["chase_price"] == 69.00 and body["chase_status"] == "in_zone"


def test_the_route_reads_the_zone_the_user_is_actually_shown_after_an_inversion_is_corrected(planned):
    # chase.read() refuses an inverted zone outright, but on THIS path it never sees one:
    # _sanitize_plan swaps low/high before the payload is built. That ordering is deliberate — the
    # read has to measure against the zone printed on the card, or the card and the warning disagree
    # and the user is back to doing arithmetic to find out which one lied.
    client, quote = planned
    quote["zone"] = (HIGH, LOW)      # analyst returned them backwards
    quote["price"] = CHASING_PRICE
    body = client.get("/plan/ABC", params={"cash": 1000}).json()
    assert (body["plan"]["entry_low"], body["plan"]["entry_high"]) == (LOW, HIGH)
    assert body["chase_status"] == "chase_too_deep" and body["chase_pct"] == 4.65
    assert body["chase_warning"] is None
