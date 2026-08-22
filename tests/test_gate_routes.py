"""The two HTTP routes over the five-leg market gate (SWT-2).

app/gate.py already has a test file pinning the thing that matters most about the engine: a leg that
could not be measured reports `ok: None`, its name lands in `unmeasured` and NOT in `failing`, and
`passed` is None rather than False. That guarantee is worth nothing if the ROUTE in front of it
flattens the third state on the way out. FastAPI/pydantic will happily coerce, a `dict.get` with a
default will happily substitute, and JSON has exactly one null but two ways to spell a false: the
literal, and a missing key a client reads as one. Any of those turns "the nightly scan has not run"
into "the market is bearish" — indistinguishable to every consumer downstream, and the sandbox arm
that blocks buys on a closed gate would then block them for a reason that never happened.

The same collapse has a second chance to happen on the way out of SQLite. `bool(0)` and `bool(None)`
are both False in Python, so a day the gate declined to decide reads back as a day it failed unless
the restore is written for three states. `test_history_keeps_passed_three_valued_...` pins that end,
and `test_a_null_passed_is_the_json_literal_null_...` pins the wire.

The rest guards what a live-priced, cached route invites. A cache that keys on nothing but the route
name serves a gate computed before last night's scan landed — with `cached: true` stamped on it, so
the staleness is visible only to someone who reads the timestamp. A failed evaluation cached for the
same minute as a real one turns a three-second upstream blip into a minute of "unknown". And an
evaluation that measured nothing at all must never be FILED: the history row is keyed on the day and
upserted, so writing one would let an outage overwrite a real morning reading with five nulls.

Isolation is SIGNALS_DATA_DIR plus importlib.reload(), in dependency order — scan_store, then
market_scan_job and main which bind it. app.gate is deliberately NOT reloaded: `importlib.reload`
re-executes into the SAME module object, so gate.py's `from . import scan_store` already points at
the isolated store, and reloading gate would give main a module object its own binding no longer
matches.
"""
from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from app import gate

# The contract, spelled out here rather than imported, so that a rename in gate.py has to be made
# twice on purpose instead of once by accident. Every consumer of this service decodes these names.
_CONTRACT_KEYS = {"passed", "available", "as_of", "evaluated_at", "market_score", "legs",
                  "failing", "unmeasured", "note"}
_LEG_KEYS = {"name", "key", "ok", "value", "threshold", "note"}
_LEG_ORDER = ["spy_above_ema50", "qqq_above_ema50", "breadth_55", "vix_under_20", "spy_mom_20d"]


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """A TestClient over an isolated data dir, plus the store the test seeds gate rows into."""
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    import app.scan_store as ss
    import app.settings_store as st
    importlib.reload(st)
    importlib.reload(ss)
    import app.market_scan_job as job
    importlib.reload(job)
    import app.main as m
    importlib.reload(m)
    m._cache.clear()
    with TestClient(m.app) as c:
        yield c, ss, m
    if ss._conn is not None:
        ss._conn.close()
        ss._conn = None


class _Bars:
    """A duck-typed stand-in for market.Series — gate reads `.closes` and `.dates`, nothing else."""

    def __init__(self, closes: list[float], dates: list[str] | None = None):
        self.closes = closes
        self.dates = dates or []


def _bars(last: float, level: float = 100.0, n: int = 260, day: str = "20260821") -> _Bars:
    """`n` flat bars at `level` with a final close of `last`, all dated `day`.

    Flat history makes both price legs analytically obvious: the 50-EMA of a constant series is that
    constant, and the 20-day return is `last` against `level`. The date matters because it is what
    `evaluate()` files the history row under.
    """
    closes = [level] * (n - 1) + [last]
    return _Bars(closes, [day] * n)


def _payload(**over) -> dict:
    """A canned evaluate() result — the contract shape, with every leg measured and passing."""
    legs = [{"name": k, "key": k, "ok": True, "value": 1.0, "threshold": 0.0, "note": "n"}
            for k in _LEG_ORDER]
    out = {
        "passed": True, "available": True, "as_of": "20260821", "evaluated_at": 1_787_353_715.3,
        "market_score": 88.9, "legs": legs, "failing": [], "unmeasured": [], "note": "n",
    }
    out.update(over)
    return out


def _stub_evaluate(monkeypatch, payload_for) -> list[int]:
    """Replace gate.evaluate with a counting stub. Returns a one-element call counter.

    main.py resolves `gate.evaluate` at call time off the module, so patching the module attribute
    is what a real request goes through — no import-order trickery.
    """
    calls = [0]

    async def _fake(client, *, d=None):
        calls[0] += 1
        return payload_for(calls[0])

    monkeypatch.setattr(gate, "evaluate", _fake)
    return calls


def _seed_scan(store, d: str, n: int = 10, above: int = 8) -> None:
    """One night of the cross-section: `above` of `n` names over their own 50-SMA."""
    rows = [{"symbol": f"S{i:02d}", "price": 100.0, "above_sma50": i < above, "above_sma200": True}
            for i in range(n)]
    assert store.insert_night(rows, d=d) == n


# ------------------------------------------------------------------------------------ the envelope

def test_the_gate_payload_is_exactly_the_contract_plus_the_cache_stamp(env, monkeypatch):
    """Every contract key present, nothing invented, and no LLM fields — this route is free.

    The key set is asserted as an equality rather than a subset. A route that grows an extra
    `confidence` or `recommendation` alongside five published legs is a route that has started
    making a claim the legs do not support, and a consumer that finds `usage`/`model` here would
    reasonably conclude an analyst had been paid for this answer.
    """
    client, _ss, _m = env
    _stub_evaluate(monkeypatch, lambda n: _payload())

    body = client.get("/gate").json()
    assert set(body) == _CONTRACT_KEYS | {"cached", "cached_age_seconds"}
    assert body["cached"] is False and body["cached_age_seconds"] == 0
    assert [leg["key"] for leg in body["legs"]] == _LEG_ORDER
    for leg in body["legs"]:
        assert set(leg) == _LEG_KEYS


@pytest.mark.parametrize("passed", [True, False, None])
def test_a_three_valued_passed_survives_the_route_unchanged(env, monkeypatch, passed):
    """True stays true, False stays false, and None stays None — not false, not absent.

    None is the case this exists for. It means "a leg could not be measured and nothing failed",
    which is a refusal to judge; false is an observation of a bad tape. A consumer standing aside
    needs to be able to say which of those happened.
    """
    client, _ss, _m = env
    _stub_evaluate(monkeypatch, lambda n: _payload(passed=passed, available=passed is not None))

    r = client.get("/gate")
    assert r.json()["passed"] is passed
    assert "passed" in r.json()


def test_a_null_passed_is_the_json_literal_null_and_not_a_dropped_key(env, monkeypatch):
    """Asserted against the raw bytes, because `.json()` cannot tell null from a missing key.

    Both decode to None in Python. On the wire they are different messages: a null is the gate
    saying "I could not decide", an absent key is a client's `body.get("passed", False)` deciding
    for it.
    """
    client, _ss, _m = env
    _stub_evaluate(monkeypatch, lambda n: _payload(passed=None, market_score=None,
                                                   unmeasured=["VIX < 20"]))

    text = client.get("/gate").text.replace(" ", "")
    assert '"passed":null' in text
    assert '"market_score":null' in text


# ------------------------------------------------------------------------------- the unmeasurable

def test_an_unevaluable_gate_answers_200_with_available_false_and_five_null_legs(env, monkeypatch):
    """The real evaluate(), with every upstream dead and no scan on disk — not a stub.

    This is the path a fresh install takes, and it is deliberately driven through the actual fetch
    code so that "unavailable is reachable" is proven rather than assumed. It must not 503: a
    consumer standing aside needs a decodable envelope more than it needs an error code, and it must
    certainly not fabricate a verdict — `passed: false` here would be a bearish claim made out of
    three failed sockets and an empty table.
    """
    client, _ss, _m = env

    async def _boom(*a, **k):
        raise RuntimeError("upstream down")

    monkeypatch.setattr(gate.market, "fetch_series", _boom)
    monkeypatch.setattr(gate.market_now, "fetch_quotes", _boom)

    r = client.get("/gate")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["passed"] is None
    assert body["market_score"] is None
    assert body["failing"] == []
    assert len(body["legs"]) == 5 and all(leg["ok"] is None for leg in body["legs"])
    # The names are still listed, so a checklist renders five dashes rather than an empty panel.
    assert len(body["unmeasured"]) == 5


def test_an_unevaluable_gate_is_never_filed_in_the_history(env, monkeypatch):
    """A total outage must not leave a row behind. The row is keyed on the day and upserted.

    Filing it would let thirty seconds of dead network overwrite a morning on which all five legs
    were measured, turning that day's record from `passed: true` into `passed: null` forever.
    """
    client, _ss, _m = env

    async def _boom(*a, **k):
        raise RuntimeError("upstream down")

    monkeypatch.setattr(gate.market, "fetch_series", _boom)
    monkeypatch.setattr(gate.market_now, "fetch_quotes", _boom)

    assert client.get("/gate").json()["available"] is False
    hist = client.get("/gate/history").json()
    assert hist["history"] == [] and hist["count"] == 0


def test_a_measurable_gate_is_evaluated_end_to_end_and_filed_under_its_day(env, monkeypatch):
    """The happy path through the real engine: five measured legs, a verdict, and one history row.

    Deliberately paired with the test above. The two together are what proves the recording rule is
    a rule about MEASURABILITY and not simply a write that never happens.
    """
    client, ss, _m = env
    _seed_scan(ss, "20260820")  # 80% above their 50-SMA — the breadth leg passes

    async def _series(client_, symbol, **kw):
        return _bars(110.0)  # 10% over a flat 50-EMA, +10% over 20 sessions

    async def _quotes(client_, symbols):
        return {gate.market_now.VIX: {"price": 15.0, "state": "REGULAR"}}

    monkeypatch.setattr(gate.market, "fetch_series", _series)
    monkeypatch.setattr(gate.market_now, "fetch_quotes", _quotes)

    body = client.get("/gate").json()
    assert body["available"] is True and body["passed"] is True
    assert body["failing"] == [] and body["unmeasured"] == []
    assert body["as_of"] == "20260821"  # SPY's last bar, not the scan's night

    row = client.get("/gate/history").json()["history"][0]
    assert row["d"] == "20260821" and row["passed"] is True


# ------------------------------------------------------------------------------------- the cache

def test_a_repeat_call_is_served_from_cache_and_stamps_its_age(env, monkeypatch):
    """The second call re-uses the evaluation, says so, and does not re-fetch three quotes."""
    client, _ss, _m = env
    calls = _stub_evaluate(monkeypatch, lambda n: _payload(market_score=float(n)))

    first = client.get("/gate").json()
    second = client.get("/gate").json()
    assert calls[0] == 1
    assert first["cached"] is False and second["cached"] is True
    assert second["cached_age_seconds"] >= 0
    # Same evaluation, not a re-run wearing a cached label.
    assert second["market_score"] == first["market_score"] == 1.0
    assert second["evaluated_at"] == first["evaluated_at"]


def test_refresh_forces_a_re_evaluation(env, monkeypatch):
    """`refresh=true` is the escape hatch for a caller that does not trust the last minute."""
    client, _ss, _m = env
    calls = _stub_evaluate(monkeypatch, lambda n: _payload(market_score=float(n)))

    client.get("/gate")
    body = client.get("/gate?refresh=true").json()
    assert calls[0] == 2
    assert body["cached"] is False and body["market_score"] == 2.0


def test_a_new_scan_landing_changes_the_cache_key(env, monkeypatch):
    """The key carries the nightly scan's date — the one input to the gate that changes discretely.

    Breadth is one of the five legs and it comes out of those rows. Without the date in the key, the
    first request after the nightly job finishes would be answered from an evaluation whose breadth
    leg was null — served with `cached: true` on it, so the only evidence would be a timestamp.
    """
    client, ss, _m = env
    calls = _stub_evaluate(monkeypatch, lambda n: _payload(market_score=float(n)))

    assert client.get("/gate").json()["market_score"] == 1.0
    assert client.get("/gate").json()["cached"] is True  # same night, still cached

    _seed_scan(ss, "20260821")
    fresh = client.get("/gate").json()
    assert calls[0] == 2
    assert fresh["cached"] is False and fresh["market_score"] == 2.0


def test_a_forced_rescan_of_the_same_night_drops_the_cached_gate(env, monkeypatch):
    """`force=true` re-measures a night already stored, so the DATE alone cannot invalidate.

    A re-run exists because someone did not trust the stored numbers; serving the gate computed off
    them for another minute is the wrong answer to that.
    """
    client, ss, m = env
    calls = _stub_evaluate(monkeypatch, lambda n: _payload(market_score=float(n)))
    _seed_scan(ss, "20260821")

    client.get("/gate")
    m._invalidate_market_scan_cache()
    assert client.get("/gate").json()["cached"] is False
    assert calls[0] == 2


def test_an_unavailable_evaluation_is_cached_far_more_briefly_than_a_real_one(env, monkeypatch):
    """A dead-upstream answer describes our plumbing, not the market, and expires in ~10s.

    Caching it for the full minute would turn a three-second blip into a minute of "unknown" —
    and, for a gated sandbox arm, a minute of blocked buys attributed to the market.
    """
    client, _ss, m = env
    calls = _stub_evaluate(monkeypatch, lambda n: _payload(passed=None, available=False,
                                                          market_score=None))

    client.get("/gate")
    key = ("gate", None)
    ts, payload = m._cache[key]
    # Age the entry past the failure window but nowhere near the 60s a real evaluation gets.
    m._cache[key] = (ts - (m._GATE_FAIL_TTL + 1), payload)
    assert m._GATE_FAIL_TTL + 1 < m._GATE_TTL

    assert client.get("/gate").json()["cached"] is False
    assert calls[0] == 2


# ----------------------------------------------------------------------------------- the history

def test_history_keeps_passed_three_valued_and_returns_newest_first(env):
    """Three days, three verdicts, through SQLite and back out as JSON.

    The null day is the point. SQLite has no boolean, and a generic restore collapses 0 and NULL to
    the same False — rewriting a day the gate declined to decide as a day the market failed.
    """
    client, ss, _m = env
    for d, passed in (("20260819", True), ("20260820", False), ("20260821", None)):
        assert ss.record_gate({"passed": passed, "evaluated_at": 1.0, "market_score": None,
                               "legs": [], "failing": [], "unmeasured": []}, d=d) is True

    r = client.get("/gate/history?limit=10")
    body = r.json()
    assert [row["d"] for row in body["history"]] == ["20260821", "20260820", "20260819"]
    assert [row["passed"] for row in body["history"]] == [None, False, True]
    assert '"passed":null' in r.text.replace(" ", "")
    assert body["count"] == 3 and body["limit"] == 10


def test_history_limit_takes_the_most_recent_rows_and_is_capped(env):
    """`limit` means "the most recent N" — an ascending order would hand back the oldest N."""
    client, ss, m = env
    for i in range(5):
        ss.record_gate({"passed": True, "evaluated_at": float(i)}, d=f"2026081{i}")

    body = client.get("/gate/history?limit=2").json()
    assert [row["d"] for row in body["history"]] == ["20260814", "20260813"]
    assert body["count"] == 2

    # The upper bound is silently clamped rather than 422'd: asking for more history than exists is
    # not a caller error, it is a caller with no way to know how much there is.
    assert client.get("/gate/history?limit=99999").json()["limit"] == m._GATE_HISTORY_MAX


def test_a_history_limit_below_one_is_a_422(env):
    """Zero rows is not a request anyone means, and it is the caller's input that is wrong."""
    client, _ss, _m = env
    assert client.get("/gate/history?limit=0").status_code == 422
    assert client.get("/gate/history?limit=-3").status_code == 422


def test_an_empty_history_is_an_empty_list_and_never_a_run_of_failing_days(env):
    """Nothing recorded yet reads as nothing recorded — no placeholder row, no false verdicts."""
    client, _ss, _m = env
    body = client.get("/gate/history").json()
    assert body["history"] == [] and body["count"] == 0
    assert "null" in body["note"]
