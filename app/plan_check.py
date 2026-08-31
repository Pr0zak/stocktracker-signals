"""Verify, server-side, the two things PLAN_SYSTEM already asks the analyst for.

The prompt instructs the model to sanity-check that risk:reward is at least about 1.5. Nothing ever
checked it. That is a stronger failure than an absent rule: a rule requested on every plan and
verified on none reads, to anyone looking at the output, exactly like a rule being followed.

The second check is newer. Until 2026-08-31 market.summarize() carried no volatility magnitude at
all, so a stop was placed with no reference to how far the name ordinarily travels in a session. A
stop closer than one average true range is not a level the thesis has failed at; it is a coin flip on
one day's noise. Now that `atr14` rides in the snapshot, the distance is measurable.

SHAPE. The result attaches to the DUMPED plan dict under `plan_check` — never to the pydantic model,
which rejects unknown fields. The key is always present and its value is always an object; a client
seeing no `plan_check` at all is talking to an old server, and that is the only thing its absence
means. Inside it, every field is always present and `null` means "not computed". Nothing is ever
substituted by 0, false, or "ok" — this module exists to make an unverifiable claim visible, so it
must not manufacture one of its own.

`messages` borrows its element shape verbatim from the TradingView Broker REST API spec (v1.18.0),
where a rejected or caveated object carries its reason on itself:

    "message": {"text": "This order has been rejected due to the closed market", "type": "error"}

It is a list here rather than the spec's singular field because a plan can fail both checks at once
and each caveat has to carry its own reason. Emission order is fixed and each check contributes at
most one message, so two plans with the same defects serialise identically.

OUT OF SCOPE, stated rather than implied: /signal and the nightly scan also emit price levels
(`invalidation_price`, `target`) that nothing verifies. They cannot be checked here because a Verdict
carries no entry zone, and R is undefined without one.

PRECISION. Money fields are stored at full precision and never rounded before being divided into.
Rounding a level to four decimal places is how a sub-penny asset ends up with a $0 midpoint and a
zero risk distance — the same class of defect tests/test_broken_quote_guards.py exists to pin. Only
the two ratios are rounded, and nothing divides by them.
"""

from __future__ import annotations

from typing import Any

from .plan_replay import _level, _num

# Both thresholds are inherited, not invented.
#
# 1.5 is the number PLAN_SYSTEM already states to the model; this module checks the prompt's own
# rule rather than a stricter one of its own devising.
_MIN_RR = 1.5
# 1.0 follows from what ATR is: the MEAN true range of one session. A stop inside that distance is
# inside an ordinary day's travel, so it is expected to be hit by noise alone. Any multiple above 1.0
# would be a risk preference; 1.0 is the definitional boundary.
_MIN_STOP_ATR = 1.0

_ACTIONABLE = ("buy_now", "buy_on_pullback")

ERROR = "error"
WARNING = "warning"
INFO = "info"


def _action(plan: dict) -> str:
    """The plan's action as a plain string.

    `model_dump()` without mode="json" hands back the ENUM MEMBER, and PlanAction subclasses str, so
    `str(PlanAction.buy_now)` is "PlanAction.buy_now" on Python 3.11+ — not "buy_now". Comparing the
    stringified member against the vocabulary therefore never matches, and every plan would be
    graded with the lenient severity. Read `.value` when it is there.
    """
    a = plan.get("action")
    return str(getattr(a, "value", a) or "")


def check(plan: dict, snapshot: dict | None) -> dict:
    """The `plan_check` block for one plan, measured against one symbol's snapshot."""
    out: dict[str, Any] = {
        "entry_mid": None, "r_usd": None, "reward_usd": None, "rr_ratio": None, "rr_ok": None,
        "atr14": None, "stop_atr": None, "stop_ok": None, "messages": [],
    }
    msgs: list[dict] = out["messages"]
    actionable = _action(plan) in _ACTIONABLE
    missing = ERROR if actionable else INFO

    # (1) The noise unit. Absent is reported with its reason and never filled from a percentage of
    # price or a close-to-close deviation: one key must mean one basis.
    atr = _num(snapshot.get("atr14")) if snapshot else None
    if snapshot is None:
        msgs.append({"text": "no technical snapshot reached this plan, so the stop could not be "
                             "measured against the symbol's ordinary daily range", "type": INFO})
    elif atr is None:
        msgs.append({"text": "this symbol's average true range could not be measured, so the stop "
                             "distance is unverified", "type": INFO})
    else:
        out["atr14"] = atr

    # (2) The entry zone. Everything downstream is measured from its midpoint, so a bad zone ends
    # the check rather than degrading it.
    lo, hi = _level(plan.get("entry_low")), _level(plan.get("entry_high"))
    if lo is None or hi is None:
        msgs.append({"text": "the entry zone is missing a bound, so risk and reward cannot be "
                             "measured", "type": missing})
        return out
    if lo > hi:
        msgs.append({"text": "the entry zone is inverted (low above high), so every distance drawn "
                             "from it would be nonsense", "type": ERROR})
        return out
    mid = (lo + hi) / 2.0
    out["entry_mid"] = mid

    # (3) Risk. EntryPlan.stop is a non-nullable float, so a model that cannot justify a stop emits
    # 0.0 — which _level maps to absence, exactly as the "Stop $0 - target $0" defect in the Android
    # client taught. The stop is tested against entry_low, not the midpoint: a fill at the bottom of
    # the zone is already under a stop that merely clears the middle.
    stop = _level(plan.get("stop"))
    if stop is None:
        msgs.append({"text": "this plan names no stop, so there is no risk to measure a reward "
                             "against", "type": missing})
    elif stop >= lo:
        msgs.append({"text": f"the stop at {stop:g} sits at or inside the entry zone "
                             f"({lo:g}-{hi:g}), so a fill would be stopped out immediately",
                     "type": ERROR})
    else:
        r = mid - stop
        if r > 0:
            out["r_usd"] = r

    # (4) Reward.
    target = _level(plan.get("target"))
    if target is None:
        msgs.append({"text": "this plan names no target, so its risk:reward cannot be checked "
                             "against the 1.5 the plan prompt asks for", "type": missing})
    elif target <= hi:
        msgs.append({"text": f"the target at {target:g} is at or below the top of the entry zone "
                             f"({lo:g}-{hi:g}), so the plan offers no upside from a fill",
                     "type": ERROR})
    else:
        out["reward_usd"] = target - mid

    # (5) The ratio the prompt asks for and nothing checked.
    if out["r_usd"] and out["reward_usd"] is not None:
        rr = out["reward_usd"] / out["r_usd"]
        out["rr_ratio"] = round(rr, 2)
        # Compare the rounded value: the prompt says "at least ~1.5", and 1.498 displayed as 1.5 and
        # flagged as a failure is a contradiction on the face of the same card.
        out["rr_ok"] = out["rr_ratio"] >= _MIN_RR
        if not out["rr_ok"]:
            msgs.append({"text": f"risk:reward is {out['rr_ratio']:.2f} against the {_MIN_RR} the "
                                 f"plan prompt asks for — risking {out['r_usd']:g} to make "
                                 f"{out['reward_usd']:g}", "type": WARNING})

    # (6) The stop against the noise. Guarded on the STORED r_usd, not on an intermediate, so there
    # is no path where a value that passed a guard is not the value divided.
    if out["r_usd"] and atr:
        out["stop_atr"] = round(out["r_usd"] / atr, 2)
        out["stop_ok"] = out["stop_atr"] >= _MIN_STOP_ATR
        if not out["stop_ok"]:
            msgs.append({"text": f"the stop is {out['stop_atr']:.2f} average true ranges below the "
                                 f"{mid:g} entry midpoint — inside one ordinary session's travel "
                                 f"(atr14 {atr:g}), so it is a coin flip on a single day rather "
                                 f"than a level the thesis has failed at", "type": WARNING})
    elif out["r_usd"] and atr == 0.0:
        msgs.append({"text": "this symbol's average true range measures zero over the window, so "
                             "there is no daily range to size the stop against", "type": INFO})

    return out


def _snap_for(symbol: Any, snaps: list[dict] | None) -> dict | None:
    """The snapshot belonging to `symbol`, or None.

    Exact match only, with one suffix retry for the crypto naming the app uses. A near match is how a
    plan ends up annotated with another symbol's volatility, which is worse than not annotating it:
    absence is visible and a wrong number is not.
    """
    if not snaps:
        return None
    want = str(symbol or "").strip().upper()
    if not want:
        return None
    for s in snaps:
        if str(s.get("symbol") or "").strip().upper() == want:
            return s
    for s in snaps:
        got = str(s.get("symbol") or "").strip().upper()
        if got and (got == f"{want}-USD" or want == f"{got}-USD"):
            return s
    return None


def annotate(plan: dict, snapshot: dict | None) -> dict:
    """Attach `plan_check` to a dumped plan dict, in place, and return it."""
    plan["plan_check"] = check(plan, snapshot)
    return plan


def annotate_picks(picks: list[dict], snaps: list[dict] | None) -> list[dict]:
    """Annotate each pick with ITS OWN symbol's snapshot.

    /recommendations ranks many symbols in one call, so the lookup is per-pick. A pick whose snapshot
    cannot be found is annotated with None — the ratio half is still measured from the plan's own
    levels, and the volatility half reports why it is absent.
    """
    for p in picks:
        annotate(p, _snap_for(p.get("symbol"), snaps))
    return picks
