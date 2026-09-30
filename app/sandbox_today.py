"""
TODAY-1 — every arm's most recent run on one card.

The Sandbox tab showed one arm at a time, so "what did the sandbox do today" took nine arm switches.
This builds the whole answer from what the store already keeps: each arm's `last_tick_date` and its
trade log, where skipped orders sit beside fills.

Three states per arm, and they must never collapse into one another:

* ran and traded — fills (and possibly skips) dated the run day;
* ran and held — the arm's cursor says it ran that day, and it filled nothing;
* did not run — the cursor is on an earlier day (a disabled arm, or a failed tick). Shown as such,
  never as a quiet "held", because "decided to do nothing" and "never asked" are different facts.

Interest and deposits are cash movements, not decisions, so they are not trades here.

Pure: the endpoint passes the ledgers in.
"""
from __future__ import annotations

import re

_TRADE_SIDES = ("buy", "sell")

# Skip reasons are written for the log and run long (a reviewer's reason can be a paragraph). The card
# has one line per order, so the common ones get a short plain name and the rest their first clause.
_SHORT = (
    (re.compile(r"^turnover cap", re.I), "daily trade limit"),
    (re.compile(r"wash-sale", re.I), "wash-sale wait"),
    (re.compile(r"^cash floor|not enough cash|insufficient cash", re.I), "not enough cash"),
    (re.compile(r"review model dropped|review dropped|reviewer", re.I), "reviewer dropped it"),
    (re.compile(r"regime gate|gate (is )?shut|gate closed", re.I), "market gate shut"),
    (re.compile(r"entry zone|above (the )?entry", re.I), "price above entry zone"),
    (re.compile(r"excluded", re.I), "excluded fund"),
    (re.compile(r"less than one share|under one share", re.I), "too small for one share"),
)


def short_reason(reason: str | None) -> str:
    text = (reason or "").strip()
    if not text:
        return "skipped"
    for pat, name in _SHORT:
        if pat.search(text):
            return name
    first = re.split(r" — | · |; |\. ", text, maxsplit=1)[0].strip()
    return first if len(first) <= 48 else first[:47].rstrip() + "…"


def _order(row: dict) -> dict:
    out = {
        "symbol": str(row.get("symbol") or "").upper(),
        "side": str(row.get("side") or "").lower(),
        "shares": row.get("shares"),
        "price": row.get("price"),
        "gross": row.get("gross"),
    }
    if row.get("status") == "skipped":
        out["reason"] = short_reason(row.get("skip_reason"))
    return out


def summarize(arms: list[dict], trades: dict[str, list[dict]], ran_at: dict[str, float | None]) -> dict:
    """`arms`: [{arm, label, engine, universe, enabled, last_tick_date, last_posture}], `main` first.
    `trades`: arm -> trade rows (any order). `ran_at`: arm -> epoch seconds of that day's NAV row.

    The day reported is the most recent `last_tick_date` of any arm, so the card always describes one
    run rather than mixing days."""
    days = [a.get("last_tick_date") for a in arms if a.get("last_tick_date")]
    if not days:
        return {"date": None, "ran_at": None, "arms": [], "totals": {"buys": 0, "sells": 0, "skipped": 0,
                                                                     "bought": 0.0, "sold": 0.0}}
    day = max(days)
    rows_out = []
    buys = sells = skipped = 0
    bought = sold = 0.0
    for a in arms:
        arm = a["arm"]
        ran = a.get("last_tick_date") == day
        filled, skips = [], []
        if ran:
            for t in trades.get(arm, []):
                if str(t.get("date")) != day or str(t.get("side") or "").lower() not in _TRADE_SIDES:
                    continue
                if t.get("status") == "filled":
                    filled.append(_order(t))
                elif t.get("status") == "skipped":
                    skips.append(_order(t))
        # Biggest first: the card lists what mattered most at the top.
        filled.sort(key=lambda o: -(o.get("gross") or 0.0))
        for o in filled:
            g = float(o.get("gross") or 0.0)
            if o["side"] == "buy":
                buys += 1
                bought += g
            else:
                sells += 1
                sold += g
        skipped += len(skips)
        posture = str(a.get("last_posture") or "").strip()
        rows_out.append({
            "arm": arm,
            "label": a.get("label") or arm,
            "engine": a.get("engine", "llm"),
            "universe": a.get("universe", "all"),
            "ran": ran,
            "last_tick_date": a.get("last_tick_date"),
            "enabled": bool(a.get("enabled", True)),
            "filled": filled,
            "skipped": skips,
            "posture": posture,
        })
    stamps = [ran_at.get(r["arm"]) for r in rows_out if r["ran"] and ran_at.get(r["arm"])]
    return {
        "date": day,
        "ran_at": max(stamps) if stamps else None,
        "arms": rows_out,
        "totals": {"buys": buys, "sells": sells, "skipped": skipped,
                   "bought": round(bought, 2), "sold": round(sold, 2)},
    }
