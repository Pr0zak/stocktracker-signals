"""DP-1/2/11/12/16 — the Daily Pick: one buy candidate a trading day, why, and what would make it wrong.

This module is PURE. It holds no clients, opens no files and makes no model call; `main.py` does the
fetching and `daily_pick_store.py` does the persistence. Everything that decides what the card says is
here, so all of it can be tested without a network.

THE SHAPE OF A RUN
------------------
1. `shortlist()` filters the night's market-scan cross-section and ranks the survivors with a STATED
   score (below). No model, no network.
2. main.py drops finalists that report earnings within `EARNINGS_BLACKOUT_SESSIONS`, fetches a full
   snapshot for the top `SHORTLIST_N`, and hands them to the analyst.
3. The analyst returns a `DailyPickChoice` (analyst.py). `reconcile()` treats that as a PROPOSAL:
   every number on the card is filled in here from the scan and the snapshot, never taken from the
   model's prose.
4. The shortlist's top name is the RULE pick (DP-11) and is recorded on every run, including the days
   the analyst declines, so "does the model beat sorting a list?" is answerable later.

WHY THE SCORE IS A STATED THESIS AND NOT AN AVERAGE OF PERCENTILES
----------------------------------------------------------------
percentiles.py forbids averaging its ranks into a composite, and it is right: the mean of a momentum
rank and a volatility rank means nothing. The score below instead states a direction and a weight for
each input, in the style of `screener.value_score`:

  * relative strength vs the S&P over ~3 months, and 60-session momentum — the best-evidenced factor
    (the analyst prompt weights it most for the same reason). HIGHER is better.
  * structural trend: price above its 50- and 200-day, and the averages stacked. MORE is better.
  * the 20-day EMA's slope — whether the trend is still rising. HIGHER is better.
  * penalties for extension (far above the 50-day, RSI very high) and for extreme daily volatility,
    because a stretched move is a reason for caution, not for chasing.

It is a shortlist, not a verdict. Its only job is to hand the analyst eight reasonable names out of
~3,000, and to be the mechanical baseline the analyst's choice is graded against.

ABSENT IS NOT ZERO
------------------
A metric the scan could not measure is left out of the score and the name is flagged `partial`; it is
never filled with 0, which would read as "the worst in the market". A factor the scan did not measure
has no row on the card, and a reason the analyst wrote about it is dropped.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Callable, Iterable

# ------------------------------------------------------------------------------------ constants

MIN_PRICE = 5.0
MIN_DOLLAR_VOLUME = 20_000_000.0      # 20-day average; below this a retail fill moves the price
MIN_BARS = 252                        # a year of bars, so the 200-day and 52-week reads are real
SHORTLIST_N = 8                       # what the analyst sees
PREFETCH_N = 16                       # ranked names checked for earnings before trimming to 8
EARNINGS_BLACKOUT_SESSIONS = 3        # a report inside this window is a coin flip, not a setup
CONVICTION_FLOOR = 60
CONVICTION_FLOOR_GATE_SHUT = 70       # a shut regime gate raises the bar rather than forbidding a pick
MAX_ZONE_DISTANCE_PCT = 10.0          # an entry zone further than this from price is not a plan for today
MAX_REASONS = 6
MAX_RUNNERS_UP = 3
REPEAT_WINDOW_SESSIONS = 20

# The stated score: weights over components that are each 0-100, higher = more attractive.
WEIGHTS: dict[str, float] = {
    "rel_strength": 0.35,
    "momentum": 0.20,
    "trend": 0.30,
    "slope": 0.15,
}

# Reject reasons, in the order they are tested. The order matters only for the counts: each rejected
# name is counted once, under the first rule it failed.
REJECT_NOT_EQUITY = "not_equity"
REJECT_PRICE = "price_under_min"
REJECT_ILLIQUID = "illiquid"
REJECT_HISTORY = "short_history"
REJECT_UNSCORABLE = "unscorable"

SUPPORTS = "supports"
AGAINST = "against"

STATUS_PICK = "pick"
STATUS_NONE = "none"
STATUS_FAILED = "failed"


# ------------------------------------------------------------------------------------ helpers

def _num(v: Any) -> float | None:
    """A finite float or None. NaN and inf are absent, not values."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _flag(v: Any) -> bool | None:
    """The scan stores booleans as 1 / 0 / NULL. NULL stays None: 'not measurable' is not False."""
    if v is None:
        return None
    try:
        return bool(int(v))
    except (TypeError, ValueError):
        return None


def _signed(v: float, digits: int = 1) -> str:
    return f"{v:+.{digits}f}"


# ------------------------------------------------------------------------------------ DP-1 shortlist

def reject_reason(row: dict) -> str | None:
    """The first hard filter this scan row fails, or None if it is eligible."""
    sym = str(row.get("symbol") or "").upper()
    if not sym or sym.endswith("-USD") or sym.startswith("^"):
        return REJECT_NOT_EQUITY
    price = _num(row.get("price"))
    if price is None or price < MIN_PRICE:
        return REJECT_PRICE
    dv = _num(row.get("dollar_volume_20d"))
    if dv is None or dv < MIN_DOLLAR_VOLUME:
        return REJECT_ILLIQUID
    bars = _num(row.get("bars"))
    if bars is None or bars < MIN_BARS:
        return REJECT_HISTORY
    return None


def trend_component(row: dict) -> float | None:
    """0-100 structural trend from the scan's three flags. None when the 200-day is unmeasured.

    100 = averages stacked (price > 50 > 150 > 200); 70 = above both the 50- and 200-day but not
    stacked; 40 = above the 200-day only; 20 = above the 50-day only; 0 = below both.
    """
    above200 = _flag(row.get("above_sma200"))
    if above200 is None:
        return None
    above50 = _flag(row.get("above_sma50"))
    if _flag(row.get("ma_stacked")):
        return 100.0
    if above200 and above50:
        return 70.0
    if above200:
        return 40.0
    if above50:
        return 20.0
    return 0.0


def penalty(row: dict) -> tuple[float, list[str]]:
    """Points subtracted for a stretched or wild setup, and the reasons, in words."""
    pts = 0.0
    why: list[str] = []
    ext = _num(row.get("pct_vs_sma50"))
    if ext is not None and ext > 20:
        pts += 15
        why.append(f"{ext:.0f}% above its 50-day")
    elif ext is not None and ext > 12:
        pts += 7
        why.append(f"{ext:.0f}% above its 50-day")
    rsi = _num(row.get("rsi14"))
    if rsi is not None and rsi > 78:
        pts += 10
        why.append(f"RSI {rsi:.0f}")
    vol = _num(row.get("adr20_pct_pctile"))
    if vol is not None and vol > 95:
        pts += 10
        why.append("among the most volatile 5% of the market")
    return pts, why


def score_row(row: dict) -> dict | None:
    """The stated score for one eligible row, or None when too little of it was measured.

    Needs the trend component AND at least one of relative strength / momentum; a name with neither
    of the best-evidenced inputs is not rankable on this thesis at all. Missing components are left
    out and the remaining weights renormalised, and the result says which were missing.
    """
    parts: dict[str, float | None] = {
        "rel_strength": _num(row.get("rel_strength_3mo_pctile")),
        "momentum": _num(row.get("mom_60d_pctile")),
        "trend": trend_component(row),
        "slope": _num(row.get("ema20_slope_pct_pctile")),
    }
    if parts["trend"] is None or (parts["rel_strength"] is None and parts["momentum"] is None):
        return None
    present = {k: v for k, v in parts.items() if v is not None}
    wsum = sum(WEIGHTS[k] for k in present)
    base = sum(WEIGHTS[k] * v for k, v in present.items()) / wsum
    pts, why = penalty(row)
    return {
        "score": round(max(0.0, base - pts), 2),
        "parts": {k: (round(v, 1) if v is not None else None) for k, v in parts.items()},
        "penalty": pts,
        "penalty_reasons": why,
        "missing": sorted(k for k, v in parts.items() if v is None),
        "partial": len(present) < len(parts),
    }


def shortlist(rows: Iterable[dict], *, only: set[str] | None = None, limit: int = PREFETCH_N) -> dict:
    """Filter + rank one night. Returns {ranked, rejects, scanned, eligible}.

    `only` restricts the universe (the "watchlist only" setting); names outside it are not counted as
    rejects, they were never candidates. `ranked` is best-first, ties broken on symbol so two runs over
    the same night agree.
    """
    rejects: dict[str, int] = {}
    scored: list[dict] = []
    scanned = 0
    for row in rows:
        sym = str(row.get("symbol") or "").upper()
        if only is not None and sym not in only:
            continue
        scanned += 1
        why = reject_reason(row)
        if why is None:
            s = score_row(row)
            if s is None:
                why = REJECT_UNSCORABLE
            else:
                scored.append({"symbol": sym, **s, "row": row})
        if why is not None:
            rejects[why] = rejects.get(why, 0) + 1
    scored.sort(key=lambda c: (-c["score"], c["symbol"]))
    return {
        "ranked": scored[:max(1, int(limit))],
        "rejects": rejects,
        "scanned": scanned,
        "eligible": len(scored),
    }


def sessions_until(today_iso: str, event_iso: str | None, is_trading_day: Callable[[Any], bool]) -> int | None:
    """Trading sessions from `today` (exclusive) up to and including `event`. None when no date.

    0 means the event is today. A date in the past returns None — it is not upcoming.
    """
    import datetime as _dt

    if not event_iso:
        return None
    try:
        t = _dt.date.fromisoformat(today_iso[:10])
        e = _dt.date.fromisoformat(str(event_iso)[:10])
    except ValueError:
        return None
    if e < t:
        return None
    n = 0
    d = t
    while d < e:
        d += _dt.timedelta(days=1)
        if is_trading_day(d):
            n += 1
    return n


# ------------------------------------------------------------------------------------ factors (DP-2)

# Every factor the analyst may cite. The key is what a reason points at; the card's row for it is
# built by the server from measured data. Order is the card's display order.
FACTOR_LABELS: dict[str, str] = {
    # Plain words, not trader shorthand: the card is read by someone who may not know what RSI or
    # relative strength means. Each has a tap-to-explain in the app for the exact definition.
    "trend": "Trend",
    "rel_strength": "Vs the S&P 500",
    "momentum": "Recent run",
    "rsi": "Overheated?",
    "extension": "How stretched",
    "range_52w": "Vs its yearly high",
    "long_cycle": "4-year trend",
    "volume": "Trading activity",
    "volatility": "Daily swings",
    "track_record": "Similar past setups",
    "insider": "Insider buying",
    "quality": "Business quality",
    "short_interest": "Bets against it",
    "seasonality": "Time of year",
    "macro": "News backdrop",
    "earnings": "Earnings",
    "regime": "Market checks",
    "today_move": "Today so far",
}
FACTOR_KEYS = tuple(FACTOR_LABELS)


def _factor(key: str, display: str, *, value: float | None = None, pctile: float | None = None,
            unit: str | None = None) -> dict:
    return {
        "key": key,
        "label": FACTOR_LABELS[key],
        "display": display,
        "value": value,
        "unit": unit,
        # A RANK in last night's cross-section, 0-100. Not a grade: a high RSI rank is not "better".
        "pctile": round(pctile, 1) if pctile is not None else None,
    }


def factors_for(row: dict | None, summary: dict | None, *, gate: dict | None = None,
                earnings: dict | None = None, today: dict | None = None) -> dict[str, dict]:
    """The measured factor rows for one candidate. A factor that could not be measured is ABSENT from
    the dict — never present with a zero — so a reason citing it can be dropped and the card has no
    row for it."""
    row = row or {}
    s = summary or {}
    out: dict[str, dict] = {}

    t = trend_component(row)
    if t is not None:
        if _flag(row.get("ma_stacked")):
            txt = "rising steadily: above its 50-, 150- and 200-day averages, in order"
        elif _flag(row.get("above_sma200")) and _flag(row.get("above_sma50")):
            txt = "above both its 50-day and 200-day averages"
        elif _flag(row.get("above_sma200")):
            txt = "above its 200-day, below its 50-day"
        elif _flag(row.get("above_sma50")):
            txt = "above its 50-day, below its 200-day"
        else:
            txt = "below its 50- and 200-day averages"
        out["trend"] = _factor("trend", txt, value=t, unit="trend_score")

    rs = _num(row.get("rel_strength_3mo"))
    if rs is not None:
        out["rel_strength"] = _factor(
            "rel_strength",
            (f"beat the S&P 500 by {abs(rs):.1f} points over 3 months" if rs >= 0
             else f"trailed the S&P 500 by {abs(rs):.1f} points over 3 months"),
            value=rs, unit="pp",
            pctile=_num(row.get("rel_strength_3mo_pctile")))

    m60 = _num(row.get("mom_60d"))
    if m60 is not None:
        out["momentum"] = _factor(
            "momentum", f"{_signed(m60)}% over the last 3 months", value=m60, unit="pct",
            pctile=_num(row.get("mom_60d_pctile")))

    rsi = _num(row.get("rsi14"))
    if rsi is not None:
        zone = ("overheated — it has risen fast" if rsi >= 70 else "washed out — it has fallen fast"
                if rsi <= 30 else "not overheated")
        out["rsi"] = _factor("rsi", f"{zone} (RSI {rsi:.0f})", value=rsi, unit="rsi",
                             pctile=_num(row.get("rsi14_pctile")))

    ext = _num(row.get("pct_vs_sma50"))
    if ext is not None:
        side = "above" if ext >= 0 else "below"
        out["extension"] = _factor("extension", f"{abs(ext):.1f}% {side} its 50-day average price",
                                   value=ext, unit="pct")

    off = _num(row.get("pct_off_52w_high"))
    if off is not None:
        txt = "at its 52-week high" if off >= -0.5 else f"{abs(off):.1f}% below its 52-week high"
        out["range_52w"] = _factor("range_52w", txt, value=off, unit="pct",
                                   pctile=_num(row.get("pct_off_52w_high_pctile")))

    lt = s.get("long_term_trend") if isinstance(s.get("long_term_trend"), dict) else None
    v200 = _num((lt or {}).get("price_vs_200w_sma_pct"))
    if v200 is not None:
        side = "above" if v200 >= 0 else "below"
        out["long_cycle"] = _factor("long_cycle", f"{abs(v200):.0f}% {side} its 4-year (200-week) average",
                                    value=v200, unit="pct")

    rv = _num(row.get("rel_volume"))
    if rv is not None:
        out["volume"] = _factor("volume", f"traded {rv:.1f}× its normal amount", value=rv, unit="x",
                                pctile=_num(row.get("rel_volume_pctile")))

    adr = _num(row.get("adr20_pct"))
    if adr is not None:
        out["volatility"] = _factor("volatility", f"moves {adr:.1f}% a day on average", value=adr,
                                    unit="pct", pctile=_num(row.get("adr20_pct_pctile")))

    tr = s.get("track_record") if isinstance(s.get("track_record"), dict) else None
    an = (tr or {}).get("analogues") if isinstance((tr or {}).get("analogues"), dict) else None
    vb = (an or {}).get("vs_benchmark") if isinstance((an or {}).get("vs_benchmark"), dict) else None
    beat = _num((vb or {}).get("beat_rate_20d"))
    n = (vb or {}).get("n")
    if beat is not None and isinstance(n, int) and n > 0:
        ns = vb.get("n_symbols")
        names = f" across {ns} names" if isinstance(ns, int) else ""
        out["track_record"] = _factor(
            "track_record",
            f"in {n} similar past cases{names}, it beat the S&P 500 {beat * 100:.0f}% of the time over the next month",
            value=round(beat * 100, 1), unit="pct")

    ins = s.get("insider") if isinstance(s.get("insider"), dict) else None
    if ins and isinstance(ins.get("buy_count_12m"), int):
        extra = ", including a conviction-size buy" if ins.get("has_conviction_buy") else ""
        out["insider"] = _factor("insider", f"{ins['buy_count_12m']} insider buys in 12 months{extra}",
                                 value=float(ins["buy_count_12m"]), unit="count")

    q = s.get("quality") if isinstance(s.get("quality"), dict) else None
    if q:
        bits = []
        roe = _num(q.get("roe"))
        if roe is not None:
            bits.append(f"ROE {roe:.0f}%")
        for k, label in (("low_debt", "low debt"), ("wide_moat", "wide moat"),
                         ("dividend_aristocrat", "dividend aristocrat")):
            if q.get(k) is True:
                bits.append(label)
        if bits:
            out["quality"] = _factor("quality", ", ".join(bits), value=roe, unit="pct")

    sp = s.get("short_pressure") if isinstance(s.get("short_pressure"), dict) else None
    if sp and sp.get("state"):
        dtc = _num(sp.get("days_to_cover"))
        tail = f" ({dtc:.1f} days of normal trading to buy back)" if dtc is not None else ""
        out["short_interest"] = _factor("short_interest", f"short sellers betting against it: {sp['state']}{tail}",
                                        value=dtc, unit="days")

    sea = s.get("seasonality") if isinstance(s.get("seasonality"), dict) else None
    cm = (sea or {}).get("current_month") if isinstance((sea or {}).get("current_month"), dict) else None
    avg = _num((cm or {}).get("avg_pct"))
    if avg is not None and cm.get("name"):
        hr = cm.get("hit_rate")
        yrs = cm.get("n")
        tail = f" (up {hr}% of {yrs} years)" if isinstance(hr, (int, float)) and isinstance(yrs, int) else ""
        out["seasonality"] = _factor("seasonality", f"{cm['name']} has averaged {_signed(avg)}%{tail}",
                                     value=avg, unit="pct")

    mac = s.get("macro") if isinstance(s.get("macro"), dict) else None
    if mac and mac.get("risk_level"):
        head = str(mac.get("headline") or "").strip()
        txt = f"news risk {mac['risk_level']}" + (f": {head}" if head else "")
        if mac.get("stale"):
            txt += " (stale read)"
        out["macro"] = _factor("macro", txt[:200])

    if earnings is not None:
        if earnings.get("ok") and earnings.get("date"):
            out["earnings"] = _factor("earnings", f"next earnings {earnings['date']}",
                                      value=_num(earnings.get("sessions")), unit="sessions")
        elif earnings.get("ok") and earnings.get("window_days"):
            out["earnings"] = _factor(
                "earnings", f"no earnings report in the next {earnings['window_days']} days")
        # A failed lookup is left ABSENT here; the card shows it as an unknown chip instead.

    # Intraday re-check only: the live quote. The daily history ends at yesterday's close during the
    # session, so without this block a re-check would reason about exactly what the morning did.
    if today is not None:
        px, chg = _num(today.get("price")), _num(today.get("change_pct"))
        if px is not None and chg is not None:
            word = "up" if chg >= 0 else "down"
            out["today_move"] = _factor("today_move", f"{word} {abs(chg):.1f}% today, at ${px:,.2f}",
                                        value=chg, unit="pct")

    if gate is not None and gate.get("available"):
        passed = gate.get("passed")
        if passed is True:
            txt = "all five market checks pass"
        elif passed is False:
            txt = f"market checks: {gate_failing_words(gate)}"
        else:
            txt = "a market check could not be measured"
        out["regime"] = _factor("regime", txt, value=_num(gate.get("market_score")), unit="score")
    return out


# Plain words for each market check the gate can fail, keyed like gate.py's legs.
_GATE_FAIL_WORDS = {
    "breadth_55": "the market is narrow — fewer than 55% of stocks are in uptrends",
    "spy_above_ema50": "the S&P 500 is below its 50-day average",
    "qqq_above_ema50": "the Nasdaq-100 is below its 50-day average",
    "vix_under_20": "the fear index (VIX) is above 20",
    "spy_mom_20d": "the S&P 500 is down over the last month",
}
_GATE_NAME_TO_KEY = {
    "Breadth > 55%": "breadth_55", "SPY > 50-EMA": "spy_above_ema50", "QQQ > 50-EMA": "qqq_above_ema50",
    "VIX < 20": "vix_under_20", "SPY 20-day momentum > 0": "spy_mom_20d",
}


def gate_failing_words(gate: dict | None) -> str:
    """Why the market checks failed, in a clause a non-trader can read."""
    names = (gate or {}).get("failing") or []
    words = [_GATE_FAIL_WORDS.get(_GATE_NAME_TO_KEY.get(n, n), n) for n in names]
    if not words:
        return "a market check failed"
    return words[0] if len(words) == 1 else f"{len(words)} market checks failed ({'; '.join(words)})"


# ------------------------------------------------------------------------------------ reconcile (DP-2)

def _level(v: Any) -> float | None:
    f = _num(v)
    return f if f is not None and f > 0 else None


def sanitize_levels(choice: dict, price: float | None) -> tuple[dict, list[str]]:
    """The analyst's four price levels, with every one that cannot be true removed.

    Returns (levels, notes). A removed level becomes None and the card draws no mark for it; each
    removal is explained in `notes` so the reader can see why the plan has a gap.
    """
    notes: list[str] = []
    lo, hi = _level(choice.get("entry_low")), _level(choice.get("entry_high"))
    stop, target = _level(choice.get("stop")), _level(choice.get("target"))
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo
    if price and (lo is not None or hi is not None):
        top = hi if hi is not None else lo
        bot = lo if lo is not None else hi
        # Distance from the NEAREST edge, so a zone that contains the price is always 0.
        if price > top:
            dist = (price / top - 1) * 100
        elif price < bot:
            dist = (1 - price / bot) * 100
        else:
            dist = 0.0
        if dist > MAX_ZONE_DISTANCE_PCT:
            notes.append(f"the entry zone was {dist:.0f}% away from the ${price:,.2f} price, so it was "
                         f"dropped rather than drawn")
            lo = hi = None
    floor = lo if lo is not None else hi
    if stop is not None and floor is not None and stop >= floor:
        notes.append("the stop sat inside or above the entry zone, so it was dropped")
        stop = None
    if stop is not None and price and stop >= price:
        notes.append("the stop was above the current price, so it was dropped")
        stop = None
    ceiling = hi if hi is not None else lo
    if target is not None and ceiling is not None and target <= ceiling:
        notes.append("the target was not above the entry zone, so it was dropped")
        target = None
    if target is not None and price and target <= price:
        notes.append("the target was not above the current price, so it was dropped")
        target = None
    return {"entry_low": lo, "entry_high": hi, "stop": stop, "target": target}, notes


def risk_reward(levels: dict) -> dict:
    """R and reward in dollars per share from the zone midpoint, and their ratio. Nulls when a level
    is missing — a ratio is never computed against an absent stop or target."""
    lo, hi = levels.get("entry_low"), levels.get("entry_high")
    stop, target = levels.get("stop"), levels.get("target")
    out: dict[str, float | None] = {"entry_mid": None, "risk_per_share": None,
                                    "reward_per_share": None, "rr_ratio": None}
    if lo is None or hi is None:
        return out
    mid = (lo + hi) / 2.0
    out["entry_mid"] = round(mid, 4)
    if stop is not None and stop < mid:
        out["risk_per_share"] = round(mid - stop, 4)
    if target is not None and target > mid:
        out["reward_per_share"] = round(target - mid, 4)
    if out["risk_per_share"] and out["reward_per_share"] is not None:
        out["rr_ratio"] = round(out["reward_per_share"] / out["risk_per_share"], 2)
    return out


def reconcile(choice: dict, *, candidates: dict[str, dict], gate: dict | None) -> dict:
    """Turn the analyst's proposal into the card's pick, or into a `none` with the reason.

    `candidates` is {SYMBOL: {"row", "summary", "factors", "price"}} for exactly the names the analyst
    was shown. The returned dict never carries a number the model wrote except its conviction (clamped)
    and the four price levels (sanitised); every factor value comes from `candidates`.
    """
    sym = str(choice.get("symbol") or "").strip().upper() or None
    conviction = int(max(0, min(100, _num(choice.get("conviction")) or 0)))
    gate_shut = bool(gate and gate.get("available") and gate.get("passed") is False)
    floor = CONVICTION_FLOOR_GATE_SHUT if gate_shut else CONVICTION_FLOOR

    runners = []
    seen: set[str] = set()
    for r in choice.get("runners_up") or []:
        rs = str((r or {}).get("symbol") or "").strip().upper()
        if rs and rs in candidates and rs != sym and rs not in seen:
            seen.add(rs)
            runners.append({"symbol": rs, "why_not": str(r.get("why_not") or "").strip()[:240]})
        if len(runners) >= MAX_RUNNERS_UP:
            break

    base = {"runners_up": runners, "conviction_floor": floor, "gate_shut": gate_shut}

    def none(reason: str, **extra) -> dict:
        # Shown as a sentence on the card and in the notification, so it starts with a capital.
        reason = reason[:1].upper() + reason[1:]
        return {"status": STATUS_NONE, "symbol": None, "none_reason": reason, **base, **extra}

    if sym is None:
        why = str(choice.get("none_reason") or "").strip() or "the AI found nothing worth buying today"
        return none(why[:400])
    if sym not in candidates:
        return none(f"the AI named {sym}, which was not one of today's candidates, so its answer was thrown out",
                    rejected_symbol=sym)

    cand = candidates[sym]
    factors = cand.get("factors") or {}
    reasons: list[dict] = []
    dropped = 0
    for r in choice.get("reasons") or []:
        key = str((r or {}).get("factor") or "").strip()
        stance = str((r or {}).get("stance") or "").strip().lower()
        text = str((r or {}).get("text") or "").strip()
        if key not in factors or stance not in (SUPPORTS, AGAINST) or not text:
            dropped += 1
            continue
        if any(x["factor"] == key and x["stance"] == stance for x in reasons):
            continue
        reasons.append({"factor": key, "stance": stance, "text": text[:240]})
    # A shut gate is always on the card as a reason against, whether or not the model said so.
    if gate_shut and "regime" in factors and not any(x["factor"] == "regime" for x in reasons):
        reasons.append({"factor": "regime", "stance": AGAINST, "text": factors["regime"]["display"]})
    reasons = reasons[:MAX_REASONS]

    if not any(r["stance"] == SUPPORTS for r in reasons):
        return none(f"the AI's case for {sym} rested only on data that could not be checked, so it was not shown", rejected_symbol=sym)
    if not any(r["stance"] == AGAINST for r in reasons):
        return none(f"the AI gave no reason against {sym}; a pick with no downside listed has not been "
                    f"thought through, so it was not shown", rejected_symbol=sym)
    if conviction < floor:
        tail = (f" today because {gate_failing_words(gate)}" if gate_shut else "")
        return none(f"the best candidate, {sym}, scored {conviction} out of 100 for confidence; it needed "
                    f"{floor}{tail}", rejected_symbol=sym, rejected_conviction=conviction)

    price = _num(cand.get("price"))
    levels, level_notes = sanitize_levels(choice, price)
    return {
        "status": STATUS_PICK,
        "symbol": sym,
        "conviction": conviction,
        "thesis": str(choice.get("thesis") or "").strip()[:300],
        "invalidation": str(choice.get("invalidation") or "").strip()[:300],
        "reasons": reasons,
        "reasons_dropped": dropped,
        "levels": levels,
        "level_notes": level_notes,
        "risk_reward": risk_reward(levels),
        "none_reason": None,
        **base,
    }


# ------------------------------------------------------------------------------------ DP-11 comparison

def paired_comparison(days: list[dict], horizon: int) -> dict:
    """AI pick vs rule pick, paired by date, at one horizon.

    `days` rows carry `ai_excess` / `rule_excess` (pct points vs the S&P over the same window, or None
    when that mark is not written yet) and `same` (both picked one name). Only days where BOTH marks
    exist are compared. Days the AI declined are reported separately, as what the rule pick did — the
    cost, or saving, of declining.
    """
    both = [d for d in days if d.get("ai_excess") is not None and d.get("rule_excess") is not None]
    ties = sum(1 for d in both if d.get("same"))
    diffs = [d["ai_excess"] - d["rule_excess"] for d in both if not d.get("same")]
    declined = [d["rule_excess"] for d in days
                if d.get("ai_status") == STATUS_NONE and d.get("rule_excess") is not None]
    return {
        "horizon_sessions": horizon,
        "n_days": len(both),
        "ai_better": sum(1 for x in diffs if x > 0),
        "rule_better": sum(1 for x in diffs if x < 0),
        "ties": ties + sum(1 for x in diffs if x == 0),
        "median_diff_pp": round(statistics.median(diffs), 2) if diffs else None,
        "declined_days": len(declined),
        "declined_rule_median_excess_pp": round(statistics.median(declined), 2) if declined else None,
    }


# ------------------------------------------------------------------------------------ DP-16 repeats

def repeat_counts(history: list[dict], symbol: str, *, group_of: Callable[[str], str],
                  sector_of: Callable[[str], str | None] | None = None,
                  window: int = REPEAT_WINDOW_SESSIONS) -> dict:
    """How often this symbol, its exposure group and its sector were the AI pick in the last `window`
    runs (history is newest-first and excludes today). Only actual picks count, not declined days."""
    recent = [h for h in history[:window] if h.get("status") == STATUS_PICK and h.get("symbol")]
    sym = symbol.upper()
    grp = group_of(sym)
    sec = sector_of(sym) if sector_of else None
    return {
        "window_runs": min(window, len(history)),
        "symbol": sum(1 for h in recent if h["symbol"].upper() == sym),
        "group": sum(1 for h in recent if group_of(h["symbol"].upper()) == grp),
        "sector": (sum(1 for h in recent if sector_of(h["symbol"].upper()) == sec)
                   if sec and sector_of else None),
        "recent_picks": [h["symbol"] for h in recent],
    }


# ------------------------------------------------------------------------------------ DP-12 fit

def portfolio_fit(symbol: str, price: float | None, holdings: list[dict], *,
                  group_of: Callable[[str], str], sectors: dict[str, str | None]) -> dict:
    """What the pick would do to the user's book. Never a share count or an order size.

    `holdings` = [{"symbol", "value"}] where value is the position's market value in dollars, or None
    when the app could not price it. Any unpriced holding makes every weight UNKNOWN, because a weight
    computed over a partial total inflates every other weight (the 2026-07-28 snapshot defect).
    The "after" figure assumes a buy the size of the MEDIAN position, so it describes a typical add
    rather than recommending an amount.
    """
    sym = symbol.upper()
    grp = group_of(sym)
    sec = sectors.get(sym)
    rows = []
    unpriced = []
    for h in holdings or []:
        s = str(h.get("symbol") or "").strip().upper()
        if not s:
            continue
        v = _num(h.get("value"))
        if v is None or v < 0:
            unpriced.append(s)
            continue
        rows.append((s, v))
    held = any(s == sym for s, _ in rows) or sym in unpriced
    out: dict[str, Any] = {
        "symbol": sym, "exposure_group": grp, "sector": sec,
        "already_held": held, "holdings_sent": len(rows) + len(unpriced),
        "unpriced": sorted(unpriced),
        "weight_pct": None, "group_weight_pct": None, "group_members_held": [],
        "sector_weight_pct": None, "sector_members_held": [], "unclassified": [],
        "reference_buy_usd": None, "group_weight_after_pct": None, "sector_weight_after_pct": None,
        "available": False, "note": None,
    }
    if not rows and not unpriced:
        out["note"] = "no holdings were sent, so fit with your portfolio is unknown"
        return out
    if unpriced:
        out["note"] = (f"{len(unpriced)} holding{'s' if len(unpriced) != 1 else ''} could not be priced, "
                       f"so portfolio weights are unknown")
        return out
    total = sum(v for _, v in rows)
    if total <= 0:
        out["note"] = "the holdings sent total $0, so weights cannot be computed"
        return out
    pos = sum(v for s, v in rows if s == sym)
    grp_rows = [(s, v) for s, v in rows if group_of(s) == grp]
    out["weight_pct"] = round(pos / total * 100, 1)
    out["group_weight_pct"] = round(sum(v for _, v in grp_rows) / total * 100, 1)
    out["group_members_held"] = sorted(s for s, _ in grp_rows)
    ref = statistics.median(v for _, v in rows)
    out["reference_buy_usd"] = round(ref, 2)
    new_total = total + ref
    out["group_weight_after_pct"] = round((sum(v for _, v in grp_rows) + ref) / new_total * 100, 1)
    if sec:
        unclassified = sorted(s for s, _ in rows if not sectors.get(s))
        sec_rows = [(s, v) for s, v in rows if sectors.get(s) == sec]
        out["unclassified"] = unclassified
        out["sector_members_held"] = sorted(s for s, _ in sec_rows)
        sv = sum(v for _, v in sec_rows)
        out["sector_weight_pct"] = round(sv / total * 100, 1)
        out["sector_weight_after_pct"] = round((sv + ref) / new_total * 100, 1)
    out["available"] = True
    return out


def fit_sentence(fit: dict) -> str:
    """The one line under the plan. Built here so the app and the notification say the same thing."""
    if not fit.get("available"):
        return str(fit.get("note") or "fit with your portfolio is unknown")
    sym = fit["symbol"]
    parts = []
    if fit.get("already_held"):
        parts.append(f"You own {sym}: {fit['weight_pct']:.1f}% of your account.")
    others = [s for s in fit.get("group_members_held") or [] if s != sym]
    if others:
        parts.append(f"You hold the same exposure through {', '.join(others)} "
                     f"({fit['group_weight_pct']:.1f}% together).")
    if fit.get("sector") and fit.get("sector_weight_pct") is not None:
        held = fit.get("sector_members_held") or []
        who = f" ({', '.join(held[:4])})" if held else ""
        parts.append(f"{fit['sector']} is {fit['sector_weight_pct']:.1f}% of your account{who}, about "
                     f"{fit['sector_weight_after_pct']:.1f}% after a typical-size buy.")
        if fit.get("unclassified"):
            parts.append(f"{len(fit['unclassified'])} holding(s) have no sector on file and are not "
                         f"counted in it.")
    elif not fit.get("already_held") and not others:
        parts.append(f"You hold nothing with the same exposure as {sym}.")
    return " ".join(parts)
