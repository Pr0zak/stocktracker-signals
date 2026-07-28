"""Validate an LLM rebalance plan against the priced snapshot the server itself built.

The plan's numbers were passed through untouched: share counts, dollar amounts, `cash_after` and
`resulting_top_weight_pct` all came straight from the model and rendered as instructions the user
types into a broker. Nothing checked that a sell was of shares they actually hold, that the buys
were affordable, that `dollars` agreed with shares x price, or that the moves produced the resulting
book the plan claimed.

This module is PURE (no I/O) so it can be tested exhaustively. The rule throughout: the SNAPSHOT is
authoritative and the model is a proposal. Where they disagree, the snapshot wins and we say so.
"""
from __future__ import annotations

# A move's dollars may drift from shares x price by rounding; beyond this we recompute rather than
# print two numbers that contradict each other on the same row.
_DOLLAR_TOLERANCE_PCT = 2.0
_DOLLAR_TOLERANCE_ABS = 1.0


def _held(portfolio: dict) -> dict[str, dict]:
    return {str(p.get("symbol", "")).upper(): p for p in (portfolio.get("positions") or [])}


def _normalise(moves: list[dict]) -> tuple[list[dict], list[str]]:
    """Canonical symbol + action, and ONE move per symbol, before anything is validated.

    Ordering is the whole point. This used to run AFTER validation, which defeated it twice over:
    two sells of one symbol were each capped at shares-held and then SUMMED, producing an
    instruction to sell more than is owned (a short); and the model's raw casing was compared
    downstream, so a "SELL" passed the whitelist and then matched no branch at all, skipping every
    check. Both are reproducible. Merge first, then validate once, so each guarantee holds on the
    row that is actually emitted.
    """
    warnings: list[str] = []
    by_sym: dict[str, list[dict]] = {}
    for mv in moves:
        m = dict(mv)
        m["symbol"] = str(m.get("symbol", "")).strip().upper()
        m["action"] = str(m.get("action", "")).strip().lower()
        by_sym.setdefault(m["symbol"], []).append(m)

    out: list[dict] = []
    for sym, group in by_sym.items():
        acting = [m for m in group if m["action"] in ("buy", "sell")]
        if not acting:
            out.append(group[0])
            continue
        kinds = {m["action"] for m in acting}
        if kinds == {"buy", "sell"}:
            warnings.append(f"dropped {sym}: the plan both bought and sold it")
            continue
        if len(group) > len(acting):
            warnings.append(f"{sym}: dropped a redundant hold alongside a {acting[0]['action']}")
        if len(acting) == 1:
            out.append(acting[0])
            continue
        warnings.append(f"{sym}: merged {len(acting)} {acting[0]['action']} moves into one")
        merged = dict(acting[0])

        def _f(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0
        merged["shares"] = sum(_f(m.get("shares")) for m in acting)
        merged["dollars"] = sum(_f(m.get("dollars")) for m in acting)
        out.append(merged)
    return out, warnings


def validate_plan(
    portfolio: dict,
    moves: list[dict],
    *,
    cash: float,
    max_position_pct: float,
) -> tuple[list[dict], dict, list[str]]:
    """Return (kept_moves, derived, warnings).

    `derived` carries `cash_after` and `resulting_top_weight_pct` RECOMPUTED from the kept moves —
    these replace the model's own figures, which were unverified guesses. Any move that cannot be
    executed against the real book is dropped and named in `warnings`; the user is better served by
    a shorter honest plan than a longer one containing an instruction that cannot be filled.
    """
    held = _held(portfolio)
    # 1. Canonicalise and merge FIRST, so every check below sees the row that will be emitted.
    merged, warnings = _normalise(moves)
    kept: list[dict] = []

    # 2. Validate each final row.
    for m in merged:
        sym, action = m["symbol"], m["action"]
        pos = held.get(sym)

        if action not in ("buy", "sell", "hold"):
            warnings.append(f"dropped {sym or '(no symbol)'}: unknown action {m.get('action')!r}")
            continue
        if pos is None:
            warnings.append(f"dropped {sym or '(no symbol)'}: not a holding in this book")
            continue
        if action == "hold":
            m["shares"], m["dollars"] = 0.0, 0.0
            kept.append(m)
            continue

        try:
            shares = float(m.get("shares"))
        except (TypeError, ValueError):
            warnings.append(f"dropped {sym}: share count {m.get('shares')!r} is not a number")
            continue
        if shares != shares or shares in (float("inf"), float("-inf")):   # NaN / inf
            warnings.append(f"dropped {sym}: share count is not finite")
            continue
        if shares <= 0:
            warnings.append(f"dropped {sym}: {action} of {shares} shares")
            continue

        # You cannot sell what you do not hold. This is the one that reaches a broker as a short.
        if action == "sell":
            owned = float(pos.get("shares") or 0.0)
            if shares > owned + 1e-9:
                warnings.append(
                    f"{sym}: plan sold {shares:g} shares but only {owned:g} are held — capped")
                shares = owned
                if shares <= 0:
                    warnings.append(f"dropped {sym}: nothing held to sell")
                    continue
        m["shares"] = round(shares, 6)

        # dollars must agree with shares x price, or the row contradicts itself on its face.
        price = pos.get("price")
        if isinstance(price, (int, float)) and price > 0:
            expect = shares * float(price)
            got = m.get("dollars")
            got = float(got) if isinstance(got, (int, float)) else None
            if got is None or abs(got - expect) > max(_DOLLAR_TOLERANCE_ABS,
                                                     expect * _DOLLAR_TOLERANCE_PCT / 100.0):
                if got is not None:
                    warnings.append(
                        f"{sym}: plan said ${got:,.0f} for {shares:g} shares at ${price:,.2f} "
                        f"— corrected to ${expect:,.0f}")
                m["dollars"] = round(expect, 2)
            else:
                m["dollars"] = round(got, 2)
        kept.append(m)

    # 3. Affordability LAST, over the surviving set only. Computing it earlier meant a buy could be
    #    funded by proceeds from a sell that was afterwards dropped — reproduced as a $4,000 buy
    #    against $0 cash, with cash_after honestly reporting -$4,000 and no warning on the buy.
    proceeds = sum(float(m["dollars"]) for m in kept if m["action"] == "sell")
    budget = max(cash, 0.0) + proceeds
    spend = 0.0
    affordable: list[dict] = []
    for m in kept:
        if m["action"] != "buy":
            affordable.append(m)
            continue
        d = float(m["dollars"])
        if spend + d > budget + 1e-6:
            room = max(0.0, budget - spend)
            price = held[m["symbol"]].get("price")
            if room < 1.0 or not (isinstance(price, (int, float)) and price > 0):
                warnings.append(
                    f"dropped buy {m['symbol']}: ${d:,.0f} but only ${room:,.0f} would be left")
                continue
            warnings.append(
                f"{m['symbol']}: buy trimmed from ${d:,.0f} to ${room:,.0f} — the plan spent more "
                f"than the cash plus sale proceeds")
            m = {**m, "dollars": round(room, 2), "shares": round(room / float(price), 6)}
            d = room
        spend += d
        affordable.append(m)

    derived = _derive(portfolio, affordable, cash=cash)
    # The whole promise of the feature is "bring the largest weight to max_position_pct". If the
    # plan does not get there, say so — silently printing a 36.7% outcome under a 25% target reads
    # as success.
    top = derived.get("resulting_top_weight_pct")
    if isinstance(top, (int, float)) and top > max_position_pct + 0.5:
        warnings.append(
            f"this plan still leaves the largest position at {top:.1f}%, above the "
            f"{max_position_pct:.0f}% target — it does not fully rebalance the book")
    return affordable, derived, warnings


def _derive(portfolio: dict, moves: list[dict], *, cash: float) -> dict:
    """Recompute the after-state from the moves, rather than trusting the model's own claim."""
    held = _held(portfolio)
    values: dict[str, float] = {}
    for sym, p in held.items():
        v = p.get("value")
        values[sym] = float(v) if isinstance(v, (int, float)) else 0.0

    cash_after = max(cash, 0.0)
    for m in moves:
        sym = m["symbol"].upper()
        d = float(m.get("dollars") or 0.0)
        if m["action"] == "sell":
            values[sym] = max(0.0, values.get(sym, 0.0) - d)
            cash_after += d
        elif m["action"] == "buy":
            values[sym] = values.get(sym, 0.0) + d
            cash_after -= d

    # A book carrying an unvalued holding has no computable weights at all — say so rather than
    # quote a top weight off a denominator we know is missing a position.
    if portfolio.get("unvalued"):
        return {"cash_after": round(cash_after, 2), "resulting_top_weight_pct": None,
                "weights_computable": False}

    total = sum(values.values()) + max(cash_after, 0.0)
    if not values or total <= 0:
        return {"cash_after": round(cash_after, 2), "resulting_top_weight_pct": None,
                "weights_computable": True}

    # The cap is per EXPOSURE, so the top weight must be the top GROUP, not the top position. Two
    # vehicles of one underlying (IBIT + FBTC) each show ~33% while the real exposure is ~67%, and
    # judging per-position reports a 25% target as met at nearly 3x it.
    held = _held(portfolio)
    group_value: dict[str, float] = {}
    for sym, v in values.items():
        g = str((held.get(sym) or {}).get("exposure_group") or sym).upper()
        group_value[g] = group_value.get(g, 0.0) + v
    top_group = max(group_value.items(), key=lambda kv: kv[1])
    return {"cash_after": round(cash_after, 2),
            "resulting_top_weight_pct": round(100.0 * top_group[1] / total, 1),
            "resulting_top_exposure": top_group[0],
            "weights_computable": True}


_REVIEW_ACTIONS = ("trim", "hold", "add", "watch")


def validate_actions(portfolio: dict, actions: list[dict]) -> tuple[list[dict], list[str]]:
    """Reconcile /portfolio/review's per-holding action list against the book it was built from.

    Nothing checked these either: the model could name a symbol the user does not hold, or invent an
    action string outside trim/hold/add/watch, and it rendered as a per-holding instruction. Also
    drops a second action for a symbol that already has one — two contradictory rows for one position
    is worse than one.
    """
    held = _held(portfolio)
    unpriced = {u["symbol"].upper() for u in (portfolio.get("unpriced") or [])}
    kept, warnings, seen = [], [], set()
    for a in actions:
        d = dict(a)
        sym = str(d.get("symbol", "")).upper()
        act = str(d.get("action", "")).lower()
        if sym not in held:
            warnings.append(f"dropped {sym or '(no symbol)'}: not a holding in this book")
            continue
        if act not in _REVIEW_ACTIONS:
            warnings.append(f"dropped {sym}: unknown action {d.get('action')!r}")
            continue
        if sym in seen:
            warnings.append(f"dropped a second action for {sym}")
            continue
        # A holding with no live price has no setup to judge; only "hold"/"watch" are honest for it.
        if sym in unpriced and act in ("trim", "add"):
            warnings.append(f"{sym}: downgraded {act} to watch — it could not be priced")
            d["action"] = "watch"
        seen.add(sym)
        d["action"] = str(d["action"]).lower()
        kept.append(d)
    return kept, warnings
