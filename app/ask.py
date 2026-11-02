"""Ask AI — one free-form question from the app, answered from TODAY'S data on this service.

The questions this exists for are the ones the investor actually types: "which ETF should I buy
today?", "what's on sale in today's dip?", "should I add to my holdings?". Every one of them needs
numbers the model does not have — today's moves, how far each fund sits off its high, whether its
long trend is intact, what it costs — and a model asked without them answers from memory, which is
the failure this service keeps correcting (see the 2026-08-26 sandbox note: GLD priced at $170 from
memory against a market of $421).

So the endpoint assembles a context from data this service already holds and hands the question to
the analyst with it. PURE here: no clients, no files, no model. main.py fetches, this module
compacts and checks.

Two rules carry over from the rest of the app:
  * ABSENT IS NOT ZERO. A block that could not be built is left out and named in `missing`, never
    sent empty — an empty dip list would assert there are no dips.
  * The model may name only symbols that appear in the context. Anything else is dropped before the
    app sees it, and the reply says so, because a name with no numbers behind it is a guess.
"""
from __future__ import annotations

import re
from typing import Iterable

# The examples the app shows under the box. Kept here so the app and the prompt agree on what the
# feature is for — the app reads them from GET /ask/examples rather than hard-coding its own.
EXAMPLES: tuple[str, ...] = (
    "Find an ETF to invest in today",
    "Which ticker should I invest in today?",
    "Based on today's market dips, which ticker or ETF should I buy?",
    "Which ETF has a decent dip to shore up my holdings?",
)

MAX_QUESTION_CHARS = 500
MAX_IDEAS = 3
# A sound fund that has pulled back at least this far off its 52-week high is a "dip" for the
# board's `dip` flag. Below 3% is ordinary noise; the board still carries every fund either way.
DIP_MIN_PCT = 4.0

_TICKER = re.compile(r"^[A-Z0-9][A-Z0-9.\-^=]{0,11}$")


def clean_question(q: str | None) -> str:
    """The question as the model will see it: whitespace collapsed, length capped. '' if empty."""
    q = " ".join(str(q or "").split())
    return q[:MAX_QUESTION_CHARS]


def _r(x, n: int = 2):
    try:
        return None if x is None else round(float(x), n)
    except (TypeError, ValueError):
        return None


def etf_board(build: dict | None, rows: dict[str, dict | None], quotes: dict[str, dict],
              pool: Iterable[str]) -> list[dict]:
    """One row per index (the cheapest copy — `pool` is etf_arm.pool_symbols), with today's move and
    where the fund sits in its own longer trend.

    A fund with no history row keeps its catalogue facts and gets null trend fields — it is still
    a fund the investor can buy, and dropping it would make the board look like the whole market.
    """
    by_sym = {str(f.get("symbol") or "").upper(): f for f in (build or {}).get("funds") or []}
    out: list[dict] = []
    for sym in pool:
        sym = str(sym).upper()
        f = by_sym.get(sym) or {}
        r = rows.get(sym) or {}
        q = quotes.get(sym) or {}
        off = _r(r.get("pct_off_52w_high"))
        above = r.get("above_sma200")
        out.append({
            "symbol": sym,
            "name": f.get("name"),
            "holds": f.get("group_label") or f.get("long_name"),
            "category": f.get("category"),
            "fee_pct": f.get("expense_ratio_pct"),
            "price": _r(q.get("price") if q.get("price") is not None else r.get("price")),
            "today_pct": _r(q.get("pct")),
            "pct_off_52w_high": off,
            "pct_vs_50_day": _r(r.get("pct_vs_sma50")),
            "pct_vs_200_day": _r(r.get("pct_vs_sma200")),
            "above_200_day": above,
            "gain_6m_pct": _r(r.get("mom_126d")),
            "return_1y_pct": (f.get("returns") or {}).get("1y"),
            "worst_drop_3y_pct": (f.get("drops") or {}).get("3y"),
            # Sound AND pulled back: the long trend is intact and the price is meaningfully under its
            # high. Null when either half is unmeasured — "not a dip" would be a claim.
            "dip": (None if off is None or above is None
                    else bool(above and off <= -DIP_MIN_PCT)),
        })
    return out


def stock_dips(rows: list[dict], quotes: dict[str, dict]) -> list[dict]:
    """Compact market-scan rows (already filtered to liquid names above their 200-day and pulled
    back) with today's live move added."""
    out = []
    for r in rows:
        sym = str(r.get("symbol") or "").upper()
        q = quotes.get(sym) or {}
        out.append({
            "symbol": sym,
            "name": q.get("name"),
            "price": _r(q.get("price") if q.get("price") is not None else r.get("price")),
            "today_pct": _r(q.get("pct")),
            "pct_off_52w_high": _r(r.get("pct_off_52w_high")),
            "pct_vs_50_day": _r(r.get("pct_vs_sma50")),
            "pct_vs_200_day": _r(r.get("pct_vs_sma200")),
            "rel_strength_3mo": _r(r.get("rel_strength_3mo")),
            "rsi14": _r(r.get("rsi14"), 1),
            "scan_night": r.get("d"),
        })
    return out


def watchlist_rows(scan: dict | None, limit: int = 40) -> list[dict]:
    """The investor's own watchlist from the nightly scan: signal, conviction and dip tier."""
    out = []
    for r in ((scan or {}).get("results") or [])[:limit]:
        out.append({
            "symbol": r.get("symbol"),
            "signal": r.get("stable_signal") or r.get("signal"),
            "conviction": r.get("conviction"),
            "dip": r.get("dip"),
            "pct_off_52w_high": _r(r.get("pct_off_52w_high")),
            "below_200_week": r.get("below_200wma"),
            "thesis": (str(r.get("thesis") or "")[:220] or None),
        })
    return out


def pick_brief(card: dict | None) -> dict | None:
    """The daily stock/ETF pick card reduced to what a question needs. None when it never ran."""
    if not card or not card.get("available"):
        return None
    pick = card.get("pick") or {}
    return {
        "date": card.get("date"),
        "stale": card.get("stale"),
        "status": card.get("status"),
        "symbol": pick.get("symbol"),
        "headline": pick.get("headline"),
        "thesis": pick.get("thesis"),
        "conviction": pick.get("conviction"),
        "levels": pick.get("levels"),
        "none_reason": card.get("none_reason") or pick.get("none_reason"),
        "closest": (pick.get("closest") or {}).get("symbol"),
        "runners_up": [{"symbol": x.get("symbol"), "why_not": x.get("why_not")}
                       for x in pick.get("runners_up") or []],
    }


def holdings_rows(holdings: list[dict] | None) -> list[dict]:
    """What the app sent about the investor's real portfolio: symbol + weight, nothing else."""
    out = []
    for h in holdings or []:
        sym = str(h.get("symbol") or "").upper().strip()
        if not sym or not _TICKER.match(sym):
            continue
        out.append({"symbol": sym, "weight_pct": _r(h.get("weight_pct"), 1)})
    return out[:60]


def known_symbols(context: dict) -> set[str]:
    """Every symbol the context carries numbers for — the only names an answer may suggest."""
    syms: set[str] = set()
    for key in ("etfs", "stock_dips", "watchlist", "holdings"):
        for r in context.get(key) or []:
            if r.get("symbol"):
                syms.add(str(r["symbol"]).upper())
    for key in ("stock_pick", "etf_pick"):
        p = context.get(key) or {}
        if p.get("symbol"):
            syms.add(str(p["symbol"]).upper())
    return syms


def check_ideas(ideas: list[dict], context: dict) -> tuple[list[dict], list[str]]:
    """(kept, dropped symbols). Keeps at most MAX_IDEAS, only for symbols the context priced, and
    enriches each with the live numbers from the context so the app shows data, not the model's
    restatement of it."""
    known = known_symbols(context)
    rows: dict[str, dict] = {}
    for key in ("watchlist", "stock_dips", "etfs"):      # later keys win: ETF/dip rows carry prices
        for r in context.get(key) or []:
            if r.get("symbol"):
                rows[str(r["symbol"]).upper()] = r
    kept, dropped, seen = [], [], set()
    for i in ideas:
        sym = str(i.get("symbol") or "").upper().strip()
        if not sym or sym in seen:
            continue
        if sym not in known:
            dropped.append(sym)
            continue
        seen.add(sym)
        r = rows.get(sym) or {}
        kept.append({
            **i,
            "symbol": sym,
            "price": r.get("price"),
            "today_pct": r.get("today_pct"),
            "pct_off_52w_high": r.get("pct_off_52w_high"),
            "fee_pct": r.get("fee_pct"),
        })
        if len(kept) >= MAX_IDEAS:
            break
    return kept, dropped
