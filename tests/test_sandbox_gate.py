"""The per-arm regime gate: an unproven hypothesis, shipped switchable so it can be measured.

The gate says "do not open new risk while the tape is not confirming". It arrives from Swing
Terminal's published breakout plan, whose own live numbers do NOT establish it: 50.0% win / 1.53
profit factor over 514 backtested trades, against 73.9% / 5.17 over just 23 closed forward trades —
and their methodology page says samples under 20-30 are noise. A 23-trade record diverging that far
from a 514-trade one is the shape of noise, so switching the gate on for the five live arms (main,
fast, rejects, reviewed, rules) would bet every ledger the account has on a coin-flip.

So it ships OFF everywhere, as `gate_enabled` in the per-arm settings. The first test below is the
one that matters most: with the switch off, a gate that fails as hard as it can must not change a
single fill. That is the guarantee protecting the five arms already running, and it is also what
makes the experiment worth running at all — one arm turns it on, the others are the control, and the
difference between the curves is the gate rather than the weather.

The rest pin the semantics that make the results readable afterwards: buys are blocked and sells are
never (you can always exit; standing aside is about not opening risk), every refusal lands in the
blocked log under a fixed reason string, and the three-valued verdict keeps "the gate failed" and
"the gate could not be measured" as two different, separately countable facts. Both BLOCK — see
sandbox_job.gate_block_reason for why the unmeasurable case is not allowed through — but an arm
whose gate was simply never computable must never read back as an arm that saw bad tape.
"""
from __future__ import annotations

import importlib
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setenv("SIGNALS_DATA_DIR", tempfile.mkdtemp())


def _group(sym: str) -> str:
    # The REAL exposure map, not a stub: a hand-written one would encode the answer the grouping
    # code exists to decide, and the cap interactions below run through it.
    from app.main import _exposure_group

    return _exposure_group(sym)


def _settings(**over):
    s = {
        "master_enabled": True, "max_position_pct": 20.0, "cash_floor_pct": 10.0,
        "min_conviction_to_trade": 55, "max_trades_per_tick": 4, "max_new_positions_per_tick": 2,
        "max_turnover_pct": 0.0, "account_type": "margin", "respect_entry_zones": False,
        "avoid_wash_sales": False, "slippage_bps": 5,
    }
    s.update(over)
    return s


def _blob(**over):
    b = {
        "cash": 10_000.0,
        "positions": [{"symbol": "SPY", "shares": 10.0, "avg_cost": 400.0,
                       "opened_at": 0.0, "last_add_at": 0.0}],
        "settings": _settings(),
    }
    b.update(over)
    return b


_PRICES = {"SPY": 500.0, "MSFT": 400.0}


def _price_of(sym: str):
    return _PRICES.get(sym.upper())


_BUY = {"symbol": "MSFT", "side": "buy", "dollars": 2_000.0, "conviction": 90,
        "reason": "the analyst's case for the trade"}
_SELL = {"symbol": "SPY", "side": "sell", "shares": 5.0, "conviction": 80,
         "reason": "trimming a winner"}

# As hostile a verdict as the gate can produce, used to prove the OFF switch is really off.
_FAILED = {"passed": False, "reason": "SPY below its 50-day and breadth negative"}


def _run(blob, orders, gate):
    from app import sandbox_job

    return sandbox_job.validate_and_fill(
        blob, orders, _price_of, group_of=_group, source="test", now_ts=1_700_000_000.0, gate=gate)


# ------------------------------------------------- the guarantee that protects the five live arms

def test_an_arm_with_the_gate_switched_off_is_untouched_by_a_failing_gate():
    """The default is False, and False must mean the gate does not exist.

    main, fast, rejects, reviewed and rules are all mid-experiment. If a failing verdict could reach
    an arm that never opted in, every curve already on disk would break in the middle and the
    comparison the arms exist for would be lost — and it would be lost to a hypothesis whose entire
    forward record is 23 trades.
    """
    ungated = _blob()
    assert ungated["settings"].get("gate_enabled") in (None, False)

    _, filled_off, skipped_off = _run(ungated, [dict(_BUY)], gate=_FAILED)
    # Identical run with no verdict supplied at all — the pre-gate world.
    _, filled_none, skipped_none = _run(_blob(), [dict(_BUY)], gate=None)

    assert [r["symbol"] for r in filled_off] == ["MSFT"], "a failing gate reached an ungated arm"
    assert filled_off == filled_none
    assert skipped_off == skipped_none == []


def test_the_gate_is_off_for_a_newly_created_arm():
    """A new arm seeds its settings from another arm, so a default that leaked True once would
    propagate into every arm created afterwards."""
    d = tempfile.mkdtemp()
    from app import sandbox_store as store

    importlib.reload(store)
    store._DATA_DIR = Path(d)           # the module constant was bound before the reload
    store._cache = {store.MAIN_ARM: store._load(store.MAIN_ARM)}

    assert store.DEFAULT_SETTINGS["gate_enabled"] is False
    arm = store.create_arm("gatetest", engine="rules", label="Gate test")
    assert arm["settings"]["gate_enabled"] is False
    assert store.get("gatetest")["settings"]["gate_enabled"] is False


# ------------------------------------------------------------------ the gated arm, verdict by verdict

def test_a_gated_arm_buys_normally_when_the_gate_passes():
    """A passing gate must be a no-op, or the arm measures the gate's plumbing instead of the gate."""
    _, filled, skipped = _run(_blob(settings=_settings(gate_enabled=True)), [dict(_BUY)],
                              gate={"passed": True, "reason": "trend and breadth confirming"})
    assert [r["symbol"] for r in filled] == ["MSFT"]
    assert skipped == []


def test_a_failing_gate_blocks_the_buy_and_records_why():
    """A blocked buy that leaves no row is indistinguishable from a buy nobody proposed.

    The reason string reaches memory.blocked_summary() through the ordinary skip path, which GROUPs
    BY it — so it carries no per-event numbers, or one recurring cause becomes a column of counts
    of 1.
    """
    from app import sandbox_job

    _, filled, skipped = _run(_blob(settings=_settings(gate_enabled=True)), [dict(_BUY)],
                              gate=_FAILED)
    assert filled == []
    assert len(skipped) == 1
    row = skipped[0]
    assert row["skip_reason"] == sandbox_job.GATE_SKIP_FAILED
    assert row["status"] == "skipped" and row["side"] == "buy" and row["symbol"] == "MSFT"
    # The analyst's own case survives alongside the refusal: `reason` is why it wanted the trade,
    # `skip_reason` is why the ledger refused it.
    assert row["reason"] == _BUY["reason"]


def test_a_failing_gate_still_lets_the_account_sell():
    """You can always exit. A gate that could trap a position would be a risk control that
    manufactures risk — the refusal is about opening NEW exposure, not about holding what is open."""
    from app import sandbox_job

    blob, filled, skipped = _run(_blob(settings=_settings(gate_enabled=True)),
                                 [dict(_SELL), dict(_BUY)], gate=_FAILED)
    assert [(r["symbol"], r["side"]) for r in filled] == [("SPY", "sell")]
    assert filled[0]["shares"] == 5.0
    assert [r["symbol"] for r in skipped] == ["MSFT"]
    assert skipped[0]["skip_reason"] == sandbox_job.GATE_SKIP_FAILED
    # And the sale actually settled into the book rather than merely being logged.
    assert blob["cash"] > 10_000.0


def test_an_unmeasurable_gate_blocks_too_but_says_something_different():
    """`passed: None` means a leg could not be measured — a different claim from "the regime is bad".

    A gated arm stands aside on None (see gate_block_reason: buying through an unmeasurable gate
    turns the experiment into a blend of gate-on and gate-off ticks weighted by feed reliability,
    and leaves no record of which were which). But it must be logged as its own case: if the two
    reasons were one string, a month of broken feeds would read back as a month of bad tape, and the
    gate's own reliability would be unmeasurable.
    """
    from app import sandbox_job

    _, filled, skipped = _run(_blob(settings=_settings(gate_enabled=True)), [dict(_BUY)],
                              gate={"passed": None, "reason": "breadth feed returned nothing"})
    assert filled == []
    assert skipped[0]["skip_reason"] == sandbox_job.GATE_SKIP_UNKNOWN
    assert sandbox_job.GATE_SKIP_UNKNOWN != sandbox_job.GATE_SKIP_FAILED
    assert "could not be measured" in sandbox_job.GATE_SKIP_UNKNOWN
    assert "could not be measured" not in sandbox_job.GATE_SKIP_FAILED


def test_a_gated_arm_with_no_verdict_at_all_reads_as_unmeasured_not_as_approved():
    """Absent is never a pass. An arm with the gate on whose caller supplied nothing knows exactly
    as much about the regime as one whose legs failed to compute, and must trade like it."""
    from app import sandbox_job

    _, filled, skipped = _run(_blob(settings=_settings(gate_enabled=True)), [dict(_BUY)], gate=None)
    assert filled == []
    assert skipped[0]["skip_reason"] == sandbox_job.GATE_SKIP_UNKNOWN


@pytest.mark.parametrize("verdict", [{}, {"passed": 0}, {"passed": ""}, {"passed": "yes"},
                                     {"passed": 1}, "passed", 1])
def test_a_malformed_verdict_never_impersonates_a_pass(verdict):
    """Only the literal True opens the gate. A truthy string or a 1 from some future serialisation
    must not be able to wave a buy through, and a falsy 0 must not be reported as a hard failure it
    never was."""
    from app import sandbox_job

    reason = sandbox_job.gate_block_reason({"gate_enabled": True}, verdict)
    assert reason == sandbox_job.GATE_SKIP_UNKNOWN


def test_the_gate_reason_is_named_ahead_of_a_cap_that_would_also_have_bound():
    """When the gate is on and failing, NOTHING opens risk — so the gate is the honest headline
    reason even for an order a cap would have refused anyway. Scattering gated refusals across
    whichever cap was tested next would make "how often did the gate bind?" uncountable."""
    from app import sandbox_job

    # Adding to a group already over the 20% position cap: ungated, the cap refuses it outright.
    # (SPY is a third of this book, so its exposure group has negative room.)
    huge = {**_BUY, "symbol": "SPY", "dollars": 2_000.0}
    _, _, capped = _run(_blob(), [dict(huge)], gate=None)
    assert capped and "cap" in (capped[0]["skip_reason"] or "")

    _, filled, skipped = _run(_blob(settings=_settings(gate_enabled=True)), [dict(huge)],
                              gate=_FAILED)
    assert filled == []
    assert skipped[0]["skip_reason"] == sandbox_job.GATE_SKIP_FAILED
    assert "cap" not in skipped[0]["skip_reason"]


# ---- deciding whether the tick should evaluate a gate at all

def test_no_gate_is_evaluated_when_every_arm_has_the_setting_off():
    """The shipped state. All five live arms default to gate_enabled False, so a tick must not spend
    three HTTP fetches computing a verdict that `gate_block_reason` would discard unread."""
    from app import sandbox_job

    assert sandbox_job.any_arm_wants_gate([{"gate_enabled": False}] * 5) is False
    assert sandbox_job.any_arm_wants_gate([{}, {}, {}]) is False
    assert sandbox_job.any_arm_wants_gate([]) is False


def test_one_gated_arm_is_enough_to_make_the_tick_evaluate_a_verdict():
    """The failure mode this guards is silent and looks like a strategy, not a bug.

    `gate_block_reason` blocks on a None verdict deliberately. So an arm with the gate switched on
    that never RECEIVES a verdict blocks every buy it proposes as "could not be measured" — for
    ever, without raising, and giving the blocked log the same reason a real feed outage would. From
    the equity curve it reads as a strategy that quietly stopped trading.
    """
    from app import sandbox_job

    arms = [{"gate_enabled": False}, {"gate_enabled": False}, {"gate_enabled": True}]
    assert sandbox_job.any_arm_wants_gate(arms) is True


def test_the_check_survives_an_arm_with_no_settings_at_all():
    # list_arms() can name an arm whose blob is mid-write or unreadable; a None must not take down
    # the real account's tick on behalf of an experiment that is off everywhere.
    from app import sandbox_job

    assert sandbox_job.any_arm_wants_gate([None, {"gate_enabled": True}]) is True
    assert sandbox_job.any_arm_wants_gate([None, None]) is False


def test_a_generator_is_accepted_because_the_caller_passes_one():
    # main.py hands this a generator expression over list_arms() so an unreadable arm blob is not
    # materialised for every arm before the first True short-circuits.
    from app import sandbox_job

    assert sandbox_job.any_arm_wants_gate({"gate_enabled": i == 2} for i in range(4)) is True
