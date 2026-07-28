"""MB-17 — is a dislocated name a DISCOUNT or is it DETERIORATING?

The 200-week screen (`screener.py`) answers "what is unusually cheap versus its own trend". The
obvious next question, and the one that decides whether a dip is worth anything, is whether the
business behind it is still intact. mungbeans calls this the value-trap check.

Everything here is already fetched elsewhere: `fundamentals.quality()` (FCF trend, share count,
debt, ROE, margins, moat flags), `insider.summary()` (12-month insider buying), and
`cycle.long_term_trend()` (still falling, or turning up). This module only reasons over them, so it
is pure and can be tested exhaustively.

THE RULE THAT MATTERS: absent evidence is reported as absent. A name with no fundamentals available
must come back "can't assess", never "looks fine" — an unknown rendered as a clean bill of health is
the exact failure this codebase keeps having to fix. `verdict` is only ever "discount" or
"deteriorating" when there was enough evidence to say so, and `missing` always names what was not
available.
"""
from __future__ import annotations

# (key, weight, human sentence). Weight reflects how strongly the signal bears on "is the business
# still intact", not how loud it sounds.
_RED = "deteriorating"
_GREEN = "discount"

_HIGH_DEBT = 2.0        # debt/equity RATIO (not percent — a units mistake made here before)
_DILUTION_PCT = 2.0     # share count up >2% y/y
_BUYBACK_PCT = -1.0     # share count down >1% y/y


def assess(trend: dict | None, quality: dict | None, insider: dict | None) -> dict:
    """Weigh deterioration evidence against discount evidence for one name.

    Returns {verdict, confidence, red[], green[], missing[], assessable}. `verdict` is
    "deteriorating" | "discount" | "unclear"; "unclear" also covers "we could not see enough".
    """
    red: list[str] = []
    green: list[str] = []
    missing: list[str] = []
    r = g = 0.0

    # ---- cash generation: the single most direct read on whether the business still works
    if quality and quality.get("fcf_trend"):
        t = quality["fcf_trend"]
        if t == "falling":
            red.append("Free cash flow is falling"); r += 2.0
        elif t == "rising":
            green.append("Free cash flow is rising"); g += 2.0
        else:
            green.append("Free cash flow is flat, not falling"); g += 0.5
        pos, yrs = quality.get("fcf_positive_years"), quality.get("fcf_years")
        if isinstance(pos, int) and isinstance(yrs, int) and yrs:
            if pos < yrs:
                red.append(f"Free cash flow was negative in {yrs - pos} of the last {yrs} years")
                r += 1.0
            else:
                green.append(f"Free cash flow positive in all {yrs} reported years"); g += 1.0
    else:
        missing.append("free cash flow")

    # ---- share count: dilution to stay alive vs buying back from strength
    sc = quality.get("shares_change_pct") if quality else None
    if isinstance(sc, (int, float)):
        if sc > _DILUTION_PCT:
            red.append(f"Share count up {sc:.1f}% — dilution"); r += 1.5
        elif sc < _BUYBACK_PCT:
            green.append(f"Share count down {abs(sc):.1f}% — buybacks"); g += 1.5
    else:
        missing.append("share count")

    # ---- balance sheet and returns
    if quality:
        de = quality.get("debt_to_equity")
        if isinstance(de, (int, float)):
            # RATIO, not percent. low_debt is <0.5; treating this as a percentage was a real bug here.
            if de > _HIGH_DEBT:
                red.append(f"Debt/equity {de:.1f} — heavily levered"); r += 1.5
            elif quality.get("low_debt"):
                green.append("Low debt"); g += 1.0
        else:
            missing.append("debt/equity")
        roe = quality.get("roe")
        if isinstance(roe, (int, float)):     # PERCENT
            if roe < 0:
                red.append(f"Return on equity is negative ({roe:.0f}%)"); r += 1.5
            elif quality.get("high_roe"):
                green.append(f"High return on equity ({roe:.0f}%)"); g += 1.0
        nm = quality.get("net_margin")
        if isinstance(nm, (int, float)) and nm < 0:
            red.append(f"Net margin is negative ({nm:.0f}%)"); r += 1.5
        for flag, sentence, w in (("wide_moat", "Wide moat", 0.5),
                                  ("buffett_quality", "Passes the quality screen", 0.5),
                                  ("dividend_aristocrat", "Dividend aristocrat", 0.5)):
            if quality.get(flag):
                green.append(sentence); g += w
    else:
        missing.append("fundamentals")

    # ---- insider behaviour: the people with the most information voting with their own money
    if insider:
        if insider.get("has_conviction_buy"):
            green.append("An insider made a large conviction buy"); g += 1.5
        if insider.get("has_cluster_buy"):
            green.append("Multiple insiders bought together"); g += 1.0
        if not insider.get("buy_count_12m"):
            red.append("No insider buying in 12 months"); r += 0.5
    else:
        missing.append("insider activity")

    # ---- price behaviour: still falling is not the same as bottoming
    if trend and trend.get("direction"):
        d = trend["direction"]
        if d == "deepening":
            red.append("Still falling away from its 200-week line"); r += 1.0
        elif d == "recovering":
            green.append("Turning back up toward its 200-week line"); g += 1.0
    else:
        missing.append("trend direction")

    # An "unclear" from genuine balance is a different statement from "we could not see enough", so
    # the caller gets `assessable` rather than having to infer it from an empty-ish verdict.
    assessable = bool(quality) and (r + g) >= 3.0
    if not assessable:
        verdict, confidence = "unclear", "low"
    elif r >= g * 1.5:
        verdict, confidence = _RED, ("high" if r >= g * 2.5 else "medium")
    elif g >= r * 1.5:
        verdict, confidence = _GREEN, ("high" if g >= r * 2.5 else "medium")
    else:
        verdict, confidence = "unclear", "medium"

    return {
        "verdict": verdict,
        "confidence": confidence,
        "assessable": assessable,
        "red": red,
        "green": green,
        # Always present, never inferred from silence: what we could not see.
        "missing": missing,
        "weights": {"deterioration": round(r, 1), "discount": round(g, 1)},
        "note": ("Whether a name that is cheap versus its own trend still has an intact business. "
                 "Evidence, not a recommendation — and an 'unclear' with a non-empty `missing` "
                 "means the data was unavailable, not that the business looks fine."),
    }
