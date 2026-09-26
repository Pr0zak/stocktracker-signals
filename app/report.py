"""RPT-1 — the weekly and monthly report: the pure half.

One report covers three things with equal weight — the market, the user's own portfolio and the AI
sandbox — for one calendar week or one calendar month. This module holds every calculation that does
not touch the network or the disk, so each one can be tested against hand-made data. `report_job.py`
fetches the bars, calls these, and stores the result. The user's portfolio is NOT computed here: the
holdings never leave the phone, so the app prices that section itself.

The rules every number in here keeps:

* **A period is two closes.** "This week" is the close of the last session before the week began to
  the close of the week's last session. A symbol only gets a number when it has a bar on BOTH of those
  dates; a missing end bar (halted, delisted, data late) is a missing number, never a stale one passed
  off as the week's close.
* **Adjusted closes, one fetch per symbol.** Each symbol's change comes from a single freshly fetched,
  split-adjusted series. The nightly scan's stored prices must never be differenced for this: each
  night stores the basis Yahoo served that night, so a split between two nights reads as a crash
  (Amphenol showed -46.96% for September 2026 from stored rows; the real move was +6.07%).
* **Futures are not used for oil and gold.** A front-month contract rolls mid-period, and the roll
  lands in the change: in the week of 2026-09-21 `CL=F` read -7.87% across the roll while the oil
  fund USO fell 3.57%. The funds hold the roll inside their own price, so they are what is quoted.
* **Absent is None, never 0.** A section that could not be measured says so with a reason.
"""
from __future__ import annotations

import datetime as dt
import math
import re
from collections import Counter, deque

from . import market_calendar

KINDS = ("week", "month")
TOP_N = 5
BIG_COMPANY_CAP = 10e9
# A sale at a loss inside this many days of buying is flagged for a second look: the sandbox's own
# objective is long-horizon terminal value, and a quick loss-making exit is the pattern it forbids.
QUICK_LOSS_DAYS = 30

# (key, symbol, name, plain note, unit). `unit` "pct" quotes a percent change; "pts" quotes the change
# in the level itself (a yield or the VIX, where "the VIX rose 0.4%" means nothing to anyone).
INDEXES: list[tuple[str, str, str, str, str]] = [
    ("sp500", "^GSPC", "S&P 500", "Big US companies", "pct"),
    ("nasdaq", "^IXIC", "Nasdaq", "Tech-heavy", "pct"),
    ("dow", "^DJI", "Dow", "30 blue chips", "pct"),
    ("small_caps", "^RUT", "Small caps", "Russell 2000", "pct"),
    ("bitcoin", "BTC-USD", "Bitcoin", "Crypto", "pct"),
    ("ten_year", "^TNX", "10-year yield", "What the US pays to borrow", "pts"),
    ("oil", "USO", "Oil", "Oil fund (USO)", "pct"),
    ("gold", "GLD", "Gold", "Gold fund (GLD)", "pct"),
    ("vix", "^VIX", "Fear (VIX)", "How big a swing traders expect", "pts"),
]
# The equal-weight S&P: every member counted the same, so it moves like the typical big company
# rather than like the ten largest. The gap between it and the S&P is what "narrow" means.
TYPICAL_STOCK = "RSP"

SECTORS: list[tuple[str, str]] = [
    ("XLK", "Tech"), ("XLC", "Comms"), ("XLV", "Health"), ("XLI", "Industrial"),
    ("XLB", "Materials"), ("XLY", "Consumer"), ("XLP", "Staples"), ("XLF", "Financials"),
    ("XLRE", "Real estate"), ("XLE", "Energy"), ("XLU", "Utilities"),
]

# Popular funds with a plain label for what each one holds. No leveraged or inverse funds: a 3x fund
# tops or bottoms every list by construction and says nothing a plain fund would not. The sector
# funds above are left out here because they have their own tile. Funds holding the same thing share
# a label, and the movers list keeps one fund per label (see `movers`).
ETFS: dict[str, str] = {
    "SPY": "S&P 500", "VOO": "S&P 500", "VTI": "Whole US market", "QQQ": "Nasdaq 100",
    "DIA": "Dow 30", "IWM": "Small companies", "IJR": "Small companies", "MDY": "Mid-size companies",
    "IJH": "Mid-size companies", "RSP": "S&P 500, equal weight", "VUG": "Growth stocks",
    "IWF": "Growth stocks", "VTV": "Value stocks", "IWD": "Value stocks", "MTUM": "Momentum stocks",
    "QUAL": "Quality stocks", "USMV": "Low-swing stocks",
    "VXUS": "Stocks outside the US", "VEA": "Developed markets", "EFA": "Developed markets",
    "VWO": "Emerging markets", "EEM": "Emerging markets", "EWJ": "Japan", "INDA": "India",
    "FXI": "China large companies", "KWEB": "China internet", "EWZ": "Brazil",
    "SMH": "Chip makers", "SOXX": "Chip makers", "IGV": "Software", "CIBR": "Cybersecurity",
    "ARKK": "Disruptive tech", "BOTZ": "Robotics and AI", "KRE": "Regional banks", "XBI": "Biotech",
    "IBB": "Biotech", "ITB": "Home builders", "XHB": "Home builders", "XRT": "Retail",
    "ITA": "Defense and aerospace", "JETS": "Airlines", "PAVE": "Infrastructure", "TAN": "Solar",
    "ICLN": "Clean energy", "URA": "Uranium", "XOP": "Oil and gas drillers", "OIH": "Oil services",
    "XME": "Metals and mining", "COPX": "Copper miners", "GDX": "Gold miners",
    "GLD": "Gold", "GLDM": "Gold", "SLV": "Silver", "USO": "Crude oil", "UNG": "Natural gas",
    "TLT": "Long-term Treasuries", "IEF": "7-10 year Treasuries", "SHY": "Short-term Treasuries",
    "BND": "US bonds", "AGG": "US bonds", "LQD": "Corporate bonds", "HYG": "High-yield bonds",
    "SCHD": "Dividend stocks", "VYM": "High-dividend stocks", "DGRO": "Dividend growers",
    "VNQ": "Real estate trusts", "IBIT": "Bitcoin", "FBTC": "Bitcoin", "ETHA": "Ethereum",
}

# Daily Pick gate legs in words, keyed by the leg's stable `key` (its display name carries numbers).
_GATE_PLAIN = {
    "breadth_55": "Too few stocks were rising",
    "spy_above_ema50": "The S&P was below its 50-day average",
    "qqq_above_ema50": "The Nasdaq 100 was below its 50-day average",
    "vix_under_20": "Fear (VIX) was high",
    "spy_mom_20d": "The S&P was losing momentum",
}


# --- periods --------------------------------------------------------------------------------------

def _prev_trading_day(d: dt.date) -> dt.date:
    d -= dt.timedelta(days=1)
    while not market_calendar.is_trading_day(d):
        d -= dt.timedelta(days=1)
    return d


def _trading_days(first: dt.date, last: dt.date) -> list[dt.date]:
    out, d = [], first
    while d <= last:
        if market_calendar.is_trading_day(d):
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def _first_day(kind: str, d: dt.date) -> dt.date:
    return d - dt.timedelta(days=d.weekday()) if kind == "week" else d.replace(day=1)


def _last_day(kind: str, d: dt.date) -> dt.date:
    if kind == "week":
        return _first_day(kind, d) + dt.timedelta(days=4)
    nxt = (d.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    return nxt - dt.timedelta(days=1)


def bounds(kind: str, d: dt.date) -> tuple[dt.date, dt.date] | None:
    """(start_close, end) for the week (Mon-Fri) or calendar month containing `d`.

    `end` is the period's last trading session; `start_close` is the last session BEFORE the period,
    whose close is the starting line. None when the period holds no session at all (a week of
    holidays does not happen on the NYSE calendar, but the answer to it is still not a guess)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    first = _first_day(kind, d)
    sessions = _trading_days(first, _last_day(kind, d))
    if not sessions:
        return None
    return _prev_trading_day(first), sessions[-1]


def latest_completed(kind: str, last_session: dt.date) -> tuple[dt.date, dt.date]:
    """The newest period of `kind` whose last session has CLOSED, given the last closed session.

    On the period's final session after the close this is the period just ending; any earlier in the
    period it is the previous one, which is what a catch-up run on a Wednesday must rebuild."""
    b = bounds(kind, last_session)
    if b is not None and b[1] <= last_session:
        return b
    d = _first_day(kind, last_session) - dt.timedelta(days=1)
    while True:
        b = bounds(kind, d)
        if b is not None:
            return b
        d = _first_day(kind, d) - dt.timedelta(days=1)


def report_id(kind: str, end: dt.date) -> str:
    return f"{kind}-{end.isoformat()}"


def period_label(kind: str, start_close: dt.date, end: dt.date) -> str:
    """"Sep 21 – 25", "Sep 28 – Oct 2", or "September 2026" — the dates a reader thinks in."""
    if kind == "month":
        return end.strftime("%B %Y")
    first = _trading_days(start_close + dt.timedelta(days=1), end)[0]
    if first.month == end.month:
        return f"{first:%b} {first.day} – {end.day}"
    return f"{first:%b} {first.day} – {end:%b} {end.day}"


def sessions_in(start_close: dt.date, end: dt.date) -> int:
    return len(_trading_days(start_close + dt.timedelta(days=1), end))


# --- series arithmetic ----------------------------------------------------------------------------

def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and x > 0


# A split moves the raw-to-adjusted factor by a third or more (3-for-2) and usually far more; a
# dividend moves it by a percent or two a quarter. Anything past this is treated as a split.
_SPLIT_FACTOR_JUMP = 0.2


def close_on(dates: list[str], closes: list, day: dt.date) -> float | None:
    """The close dated exactly `day` (YYYYMMDD bars), or None. Exact, not "on or before": a bar
    missing on the day that bounds the period is a gap, and filling it from the day before would
    report a week that was not measured."""
    key = day.strftime("%Y%m%d")
    for d, c in zip(dates, closes):
        if d == key:
            return float(c) if _finite(c) else None
    return None


def _price_pair(dates, closes, raw, start_close, end):
    """(start, end, split) closes for the period. PRICE closes — the change everyone quotes — unless
    a split sits between the two dates, which the raw prices would report as a crash; then the
    split-adjusted pair, flagged."""
    a = close_on(dates, closes, start_close)
    b = close_on(dates, closes, end)
    if a is None or b is None:
        return None
    if raw:
        ra = close_on(dates, raw, start_close)
        rb = close_on(dates, raw, end)
        if ra is not None and rb is not None:
            factor_move = (b / rb) / (a / ra) - 1.0
            if abs(factor_move) <= _SPLIT_FACTOR_JUMP:
                return ra, rb, False
            return a, b, True
    return a, b, False


def change(dates: list[str], closes: list, start_close: dt.date, end: dt.date, unit: str = "pct",
           raw: list | None = None) -> dict | None:
    """{"close", "start", "pct" | "change"} over the period, or None when either bounding bar is
    missing. `unit` "pts" reports the level change (yields, the VIX) instead of a percent. With `raw`
    closes the change is the price change (dividends not added back), except across a split, where
    the adjusted series is used and `split: true` says so."""
    pair = _price_pair(dates, closes, raw, start_close, end)
    if pair is None:
        return None
    a, b, split = pair
    out = {"close": round(b, 4), "start": round(a, 4)}
    if split:
        out["split"] = True
    if unit == "pts":
        out["change"] = round(b - a, 4)
    else:
        out["pct"] = round((b / a - 1.0) * 100.0, 2)
    return out


def daily_moves(dates: list[str], closes: list, start_close: dt.date, end: dt.date) -> list[dict]:
    """Each session's percent move inside the period — the monthly calendar. A session without a
    usable bar on either side is left out rather than drawn as flat."""
    lo, hi = start_close.strftime("%Y%m%d"), end.strftime("%Y%m%d")
    out: list[dict] = []
    prev = None
    for d, c in zip(dates, closes):
        if d > hi:
            break
        if d > lo and prev is not None and _finite(c):
            out.append({"date": f"{d[:4]}-{d[4:6]}-{d[6:]}", "pct": round((c / prev - 1.0) * 100.0, 2)})
        prev = float(c) if _finite(c) else None
    return out


# --- movers ----------------------------------------------------------------------------------------

def movers(rows: list[dict], n: int = TOP_N, dedupe_key: str | None = None) -> dict:
    """{"best": [...], "worst": [...]} from rows carrying "pct". With `dedupe_key`, only the first row
    per value of that key is kept in each list (two funds holding the same chips are one fact). The
    lists never share a row, so a short input cannot show one name as both best and worst."""
    usable = [r for r in rows if isinstance(r.get("pct"), (int, float)) and math.isfinite(r["pct"])]
    ordered = sorted(usable, key=lambda r: r["pct"], reverse=True)

    def key(r: dict):
        return r.get(dedupe_key) if dedupe_key else r.get("symbol")

    def pick(seq: list[dict], taken: set) -> list[dict]:
        out, seen = [], set(taken)
        for r in seq:
            if key(r) in seen:
                continue
            seen.add(key(r))
            out.append(r)
            if len(out) == n:
                break
        return out

    best = pick(ordered, set())
    # Excluded by KEY, not symbol: a fund dropped from "best" because its twin was already there must
    # not then turn up as one of the worst.
    worst = pick(list(reversed(ordered)), {key(r) for r in best})
    return {"best": best, "worst": worst}


def breadth(rows: list[dict]) -> dict:
    """How many of the measured names rose, fell or sat flat. `measured` is the denominator the share
    is taken over — never the list the job tried to fetch."""
    pcts = [r["pct"] for r in rows if isinstance(r.get("pct"), (int, float)) and math.isfinite(r["pct"])]
    up = sum(1 for p in pcts if p > 0)
    down = sum(1 for p in pcts if p < 0)
    flat = len(pcts) - up - down
    return {"up": up, "down": down, "flat": flat, "measured": len(pcts),
            "up_share": round(up / len(pcts), 4) if pcts else None}


_NAME_TAILS = re.compile(
    r"(,?\s+(inc\.?|incorporated|corporation|corp\.?|company|co\.|holdings?|group|ltd\.?|limited|plc|"
    r"n\.v\.|s\.a\.|s\.p\.a\.|se|ag|l\.p\.|lp|class [a-c]|common stock|ordinary shares|"
    r"american depositary shares?|ads))+\s*$", re.IGNORECASE)


def clean_name(name: str | None) -> str | None:
    """"Moderna, Inc. - Common Stock" -> "Moderna". The exchange directory's legal names read as
    noise on a phone; the ticker is beside it for anyone who wants the precise one."""
    if not name:
        return None
    n = name.split(" - ")[0].strip()
    prev = None
    while prev != n:
        prev = n
        n = _NAME_TAILS.sub("", n).strip().rstrip(",").strip()
    return n or name.strip()


# --- the headline ----------------------------------------------------------------------------------

def headline(kind: str, sp_pct: float | None, up_share: float | None, *, tech_led: bool = False) -> str:
    """One plain sentence for the top of the report, from measured numbers only.

    Deterministic on purpose: a model-written line could state a number the report does not hold,
    and the facts a one-liner needs (did the S&P rise, did most stocks) are already computed."""
    period = "week" if kind == "week" else "month"
    if sp_pct is None:
        return f"The S&P could not be measured this {period}."
    most = None if up_share is None else ("rose" if up_share >= 0.5 else "fell")
    if abs(sp_pct) < 0.25:
        return f"A flat {period} for the S&P." + (f" Most stocks {most}." if most else "")
    if sp_pct > 0:
        if most == "fell":
            who = "Big tech" if tech_led else "The biggest companies"
            return f"{who} lifted the S&P. Most stocks fell."
        return f"A good {period}: the S&P rose" + (" and most stocks did too." if most else ".")
    if most == "rose":
        return "The S&P slipped, but most stocks rose."
    return f"A down {period} for the S&P" + (" and most stocks." if most else ".")


# --- the AI sandbox --------------------------------------------------------------------------------

def _row_on_or_before(rows: list[dict], day: dt.date) -> dict | None:
    key = day.isoformat()
    best = None
    for r in rows:
        if str(r.get("date")) <= key:
            best = r
        else:
            break
    return best


def sandbox_period(nav: list[dict], start_close: dt.date, end: dt.date) -> dict | None:
    """One book's result over the period from its own NAV record, deposits taken out.

    `change_pct` is time-weighted: the product of each session's return with that session's deposit
    removed, so a deposit is never counted as performance and a large one late in the month does not
    dilute the return. `bench_pct` is the book's own "same money in the S&P" shadow measured the same
    way at the same moments — the sandbox marks its books at its 3:35 PM ET check, so it is compared
    with a benchmark marked then too, not with the 4 PM close. None when the NAV does not cover both
    ends of the period."""
    rows = sorted((r for r in nav if r.get("date") and _finite(r.get("equity"))), key=lambda r: str(r["date"]))
    s = _row_on_or_before(rows, start_close)
    e = _row_on_or_before(rows, end)
    if s is None or e is None or str(e["date"]) <= str(s["date"]):
        return None
    seg = [r for r in rows if str(s["date"]) <= str(r["date"]) <= str(e["date"])]
    twr = btwr = 1.0
    bench_ok = True
    deposits = 0.0
    for prev, cur in zip(seg, seg[1:]):
        dep = float(cur.get("funded_total") or 0.0) - float(prev.get("funded_total") or 0.0)
        deposits += dep
        twr *= (float(cur["equity"]) - dep) / float(prev["equity"])
        pb, cb = prev.get("benchmark_value"), cur.get("benchmark_value")
        if _finite(pb) and _finite(cb):
            btwr *= (float(cb) - dep) / float(pb)
        else:
            bench_ok = False
    out = {
        "start_date": str(s["date"]), "end_date": str(e["date"]),
        "start_equity": round(float(s["equity"]), 2), "end_equity": round(float(e["equity"]), 2),
        "deposits": round(deposits, 2),
        "change_usd": round(float(e["equity"]) - float(s["equity"]) - deposits, 2),
        "change_pct": round((twr - 1.0) * 100.0, 2),
        "bench_pct": round((btwr - 1.0) * 100.0, 2) if bench_ok else None,
        "bench_change_usd": (round(float(e["benchmark_value"]) - float(s["benchmark_value"]) - deposits, 2)
                             if bench_ok else None),
        "cash": round(float(e.get("cash") or 0.0), 2) if e.get("cash") is not None else None,
    }
    out["vs_pts"] = round(out["change_pct"] - out["bench_pct"], 2) if out["bench_pct"] is not None else None
    out["cash_pct"] = (round(out["cash"] / out["end_equity"] * 100.0, 1)
                       if out["cash"] is not None and out["end_equity"] else None)
    # The period's last session had no check (the timer did not run, the service was down): say which
    # day the book was last measured instead of presenting an older number as the period's close.
    out["measured_through_end"] = out["end_date"] == end.isoformat()
    return out


def plain_skip_reason(reason: str | None) -> str:
    """A blocked order's reason in words. The ledger records the validator's precise wording, which is
    right for the log and wrong for a phone."""
    r = (reason or "").lower()
    if "under one share" in r:
        return "Not enough cash for one share"
    if "wash-sale" in r or "wash sale" in r:
        return "Waiting out the 30-day wash-sale rule"
    if "regime gate" in r or "market checks" in r:
        return "Market checks said stand aside"
    if "unsettled" in r:
        return "Waiting for sale cash to settle"
    if "conviction" in r:
        return "Not confident enough"
    if "cash floor" in r:
        return "Keeping its cash cushion"
    if "cap" in r and ("exposure" in r or "position" in r or "group" in r):
        return "Would go over its size limit"
    if "zone" in r:
        return "Price was outside its buy range"
    reason = (reason or "").strip()
    return (reason[:60] + "…") if len(reason) > 61 else (reason or "Blocked")


def period_trades(trades: list[dict], start_close: dt.date, end: dt.date,
                  *, quick_loss_days: int = QUICK_LOSS_DAYS) -> dict:
    """What the book did inside the period: its filled orders, its blocked ones, and its interest.

    Holding days on a sale come from first-in-first-out lots over the WHOLE ledger (the oldest shares
    go first, as a broker would report it), so a sale in this period is aged correctly even when its
    purchase was months earlier. A sale at a loss inside `quick_loss_days` is flagged."""
    lo, hi = start_close.isoformat(), end.isoformat()
    rows = sorted((t for t in trades if t.get("date")), key=lambda t: (str(t["date"]), float(t.get("ts") or 0.0)))
    lots: dict[str, deque] = {}
    fills: list[dict] = []
    blocked: dict[str, dict] = {}
    interest = 0.0
    for t in rows:
        date = str(t["date"])
        side = str(t.get("side") or "").lower()
        sym = str(t.get("symbol") or "").upper()
        inside = lo < date <= hi
        if side == "interest":
            if inside and t.get("status") == "filled":
                interest += float(t.get("gross") or 0.0)
            continue
        if side not in ("buy", "sell") or sym in ("", "CASH"):
            continue
        if t.get("status") == "filled" and t.get("shares") and t.get("price") is not None:
            shares = float(t["shares"])
            held_days = None
            if side == "buy":
                lots.setdefault(sym, deque()).append([date, shares])
            else:
                q, oldest, left = lots.setdefault(sym, deque()), None, shares
                while left > 1e-9 and q:
                    lot = q[0]
                    oldest = oldest or lot[0]
                    take = min(lot[1], left)
                    lot[1] -= take
                    left -= take
                    if lot[1] <= 1e-9:
                        q.popleft()
                if oldest:
                    held_days = (dt.date.fromisoformat(date) - dt.date.fromisoformat(oldest)).days
            if inside:
                pl = t.get("realized_pl")
                row = {"date": date, "side": side, "symbol": sym, "shares": shares,
                       "price": round(float(t["price"]), 4), "gross": round(float(t.get("gross") or 0.0), 2),
                       "realized_pl": round(float(pl), 2) if isinstance(pl, (int, float)) else None,
                       "held_days": held_days, "flag": None}
                if (side == "sell" and row["realized_pl"] is not None and row["realized_pl"] < 0
                        and held_days is not None and held_days < quick_loss_days):
                    row["flag"] = "quick_loss"
                fills.append(row)
        elif t.get("status") == "skipped" and inside:
            b = blocked.setdefault(sym, {"symbol": sym, "side": side, "count": 0, "dates": [], "reason": None})
            b["count"] += 1
            b["dates"].append(date)
            b["reason"] = plain_skip_reason(t.get("skip_reason"))
    return {"fills": fills, "blocked": sorted(blocked.values(), key=lambda b: -b["count"]),
            "buys": sum(1 for f in fills if f["side"] == "buy"),
            "sells": sum(1 for f in fills if f["side"] == "sell"),
            "blocked_count": sum(b["count"] for b in blocked.values()),
            "interest": round(interest, 2)}


# --- the Daily Pick --------------------------------------------------------------------------------

def daily_pick_summary(runs: list[dict], start_close: dt.date, end: dt.date) -> dict:
    """Picks made in the period, and when there were none, the most common reason in words."""
    lo, hi = start_close.isoformat(), end.isoformat()
    inside = [r for r in runs if lo < str(r.get("date") or "") <= hi]
    picks = [r for r in inside if r.get("status") == "pick" and (r.get("pick") or {}).get("symbol")]
    none_runs = [r for r in inside if r.get("status") == "none"]
    failed = [r for r in inside if r.get("status") == "failed"]
    reason = None
    if not picks and none_runs:
        keys: Counter = Counter()
        for r in none_runs:
            g = r.get("gate") or {}
            by_name = {str(l.get("name")): l.get("key") for l in (g.get("legs") or [])}
            for name in g.get("failing") or []:
                k = by_name.get(str(name))
                if k in _GATE_PLAIN:
                    keys[k] += 1
        reason = _GATE_PLAIN[keys.most_common(1)[0][0]] if keys else "No stock cleared the bar"
    return {"runs": len(inside), "picks": len(picks),
            "symbols": [str(r["pick"]["symbol"]) for r in sorted(picks, key=lambda r: str(r["date"]))],
            "failed": len(failed), "reason": reason}
