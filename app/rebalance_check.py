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
    warnings: list[str] = []
    kept: list[dict] = []

    for mv in moves:
        m = dict(mv)
        sym = str(m.get("symbol", "")).upper()
        action = str(m.get("action", "")).lower()
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

        shares = m.get("shares")
        try:
            shares = float(shares)
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

    # Affordability: buys may only spend the cash on hand plus what the sells actually raise.
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
            price = held[m["symbol"].upper()].get("price")
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

    # One symbol, one instruction. A live plan came back with both "sell AAPL 4" and "hold AAPL" —
    # rendered as two rows telling the user opposite things about the same position. Keep the
    # actionable move and drop the redundant hold; genuinely conflicting buy+sell is dropped whole
    # because we cannot tell which the model meant.
    deduped: list[dict] = []
    by_sym: dict[str, list[dict]] = {}
    for m in affordable:
        by_sym.setdefault(m["symbol"].upper(), []).append(m)
    for sym, group in by_sym.items():
        if len(group) == 1:
            deduped.append(group[0])
            continue
        acting = [m for m in group if m["action"] in ("buy", "sell")]
        kinds = {m["action"] for m in acting}
        if len(acting) == 1:
            warnings.append(f"{sym}: dropped a redundant hold alongside a {acting[0]['action']}")
            deduped.append(acting[0])
        elif not acting:
            deduped.append(group[0])
        elif kinds == {"buy", "sell"}:
            warnings.append(f"dropped {sym}: the plan both bought and sold it")
        else:
            warnings.append(f"{sym}: merged {len(acting)} {acting[0]['action']} moves into one")
            merged = dict(acting[0])
            merged["shares"] = round(sum(float(m["shares"]) for m in acting), 6)
            merged["dollars"] = round(sum(float(m["dollars"]) for m in acting), 2)
            deduped.append(merged)

    derived = _derive(portfolio, deduped, cash=cash)
    # The whole promise of the feature is "bring the largest weight to max_position_pct". If the
    # plan does not get there, say so — silently printing a 36.7% outcome under a 25% target reads
    # as success.
    top = derived.get("resulting_top_weight_pct")
    if isinstance(top, (int, float)) and top > max_position_pct + 0.5:
        warnings.append(
            f"this plan still leaves the largest position at {top:.1f}%, above the "
            f"{max_position_pct:.0f}% target — it does not fully rebalance the book")
    return deduped, derived, warnings


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
    top = round(100.0 * max(values.values()) / total, 1) if values and total > 0 else None
    return {"cash_after": round(cash_after, 2), "resulting_top_weight_pct": top,
            "weights_computable": True}
