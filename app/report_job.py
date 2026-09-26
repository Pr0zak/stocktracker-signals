"""RPT-1 — the weekly and monthly report: fetch, assemble, store.

`report.py` holds the arithmetic; this file supplies it with bars and ledgers and keeps the results.
One report is one JSON file under `data/reports/`, named by its id (`week-2026-09-25`,
`month-2026-09-30`). A built report is never rebuilt unless forced, so the timer can call this every
weekday and only the day a period closes does any work.

What it fetches, all in one pass with the market scan's adaptive concurrency gate:
  * the index rows, the equal-weight S&P and the eleven sector funds (~21 symbols);
  * about seventy popular ETFs;
  * every company in the scan universe worth $10B or more (~930 names).
Each symbol is one fresh split-adjusted series, and each change is measured inside that one series
(see report.py for why the nightly scan's stored prices must not be differenced).

A report is REFUSED rather than stored when the S&P itself cannot be measured or most of the big
companies failed to load: a half-measured report saved under the week's id would be served as the
week's report forever. The refusal goes back to the caller, the timer's catch-up slot retries.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import os
import re
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

from . import daily_pick_store, market_calendar, report, sandbox_store, swing, universe
from .market import fetch_series
# The scan's gate is the only concurrency setting that has been measured against Yahoo from this box;
# reusing it beats inventing a second, unmeasured one.
from .market_scan_job import _AdaptiveGate, _is_throttle

log = logging.getLogger("signals.report")

_DATA_DIR = Path(os.environ.get("SIGNALS_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
_DIR = _DATA_DIR / "reports"
_ET = ZoneInfo("America/New_York")
VERSION = 1
# A month needs ~23 sessions before its last close; six months lets a catch-up build of LAST month
# still work late in this one without reaching for a second range.
_RANGE = "6mo"
# Below this share of the big companies measured, the movers and the "how many rose" count are too
# thin to publish as the market's week.
_MIN_STOCK_COVERAGE = 0.5
_ID_RE = re.compile(r"^(week|month)-\d{4}-\d{2}-\d{2}$")

_build_lock = asyncio.Lock()


class ReportError(RuntimeError):
    """The report could not be measured well enough to store."""


# --- storage ---------------------------------------------------------------------------------------

def _path(rid: str) -> Path:
    if not _ID_RE.fullmatch(rid or ""):
        raise ValueError("not a report id")
    return _DIR / f"{rid}.json"


def load(rid: str) -> dict | None:
    try:
        p = _path(rid)
    except ValueError:
        return None
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except ValueError:
        log.warning("report %s is unreadable", rid)
        return None


def save(rep: dict) -> None:
    p = _path(rep["id"])
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rep, separators=(",", ":")))
    os.replace(tmp, p)


def summary(rep: dict) -> dict:
    """The list row: enough for the Reports screen and the notification, nothing heavier."""
    idx = {i.get("key"): i for i in (rep.get("market") or {}).get("indexes") or []}
    main = ((rep.get("sandbox") or {}).get("main")) or {}
    return {"id": rep.get("id"), "kind": rep.get("kind"), "label": rep.get("label"),
            "start_close": rep.get("start_close"), "end": rep.get("end"), "made_at": rep.get("made_at"),
            "headline": rep.get("headline"),
            "sp500_pct": (idx.get("sp500") or {}).get("pct"),
            "sandbox_pct": main.get("change_pct"), "sandbox_change_usd": main.get("change_usd")}


def list_reports(kind: str | None = None, limit: int = 30) -> list[dict]:
    """Summaries, newest period first (a month and the week ending the same day: the month first).

    Ordered by the ids alone — the id carries the kind and the end date — and only the files that make
    the cut are opened, so the phone's half-hourly poll does not parse a year of reports each time."""
    if not _DIR.is_dir():
        return []
    ids = [p.stem for p in _DIR.glob("*.json")
           if _ID_RE.fullmatch(p.stem) and (not kind or p.stem.startswith(kind + "-"))]
    ids.sort(key=lambda rid: (rid.split("-", 1)[1], rid.startswith("month")), reverse=True)
    out = []
    for rid in ids:
        rep = load(rid)
        if rep:
            out.append(summary(rep))
        if len(out) >= max(1, int(limit)):
            break
    return out


# --- periods ---------------------------------------------------------------------------------------

def last_closed_session(now_et: dt.datetime) -> dt.date:
    """The most recent session whose regular close has passed, in ET (early closes included)."""
    d = now_et.date()
    if market_calendar.is_trading_day(d):
        close_s = market_calendar.session_end_seconds(d)[0]
        if now_et.hour * 3600 + now_et.minute * 60 >= close_s:
            return d
    d -= dt.timedelta(days=1)
    while not market_calendar.is_trading_day(d):
        d -= dt.timedelta(days=1)
    return d


# --- fetching --------------------------------------------------------------------------------------

async def _fetch_all(client: httpx.AsyncClient, symbols: list[str]) -> tuple[dict, list[str]]:
    gate = _AdaptiveGate()
    got: dict = {}
    failed: list[str] = []

    async def one(sym: str) -> None:
        async with gate:
            t0 = time.monotonic()
            try:
                got[sym] = await fetch_series(client, sym, rng=_RANGE, fallback=False)
                await gate.record(throttled=False)
            except Exception as e:  # noqa: BLE001 — a symbol that will not load is counted, not fatal
                failed.append(sym)
                await gate.record(throttled=_is_throttle(e, time.monotonic() - t0))

    await asyncio.gather(*[one(s) for s in dict.fromkeys(symbols)])
    return got, sorted(failed)


def _core_symbols() -> set[str]:
    return ({s for _, s, _, _, _ in report.INDEXES} | {report.TYPICAL_STOCK}
            | {s for s, _ in report.SECTORS} | set(report.ETFS))


def _big_companies(blob: dict | None) -> list[dict]:
    rows = (blob or {}).get("detail") or []
    return [r for r in rows if not r.get("is_etf") and (r.get("market_cap") or 0) >= report.BIG_COMPANY_CAP
            and r.get("symbol")]


# --- assembly --------------------------------------------------------------------------------------

def _change(series: dict, sym: str, start: dt.date, end: dt.date, unit: str = "pct") -> dict | None:
    s = series.get(sym)
    if s is None:
        return None
    return report.change(s.dates, s.closes, start, end, unit, raw=getattr(s, "raw_closes", None) or None)


def assemble(kind: str, start: dt.date, end: dt.date, series: dict, failed: list[str], big: list[dict],
             *, nav_by_arm: dict[str, list[dict]], labels: dict[str, str], main_trades: list[dict],
             pick_runs: list[dict], now: float | None = None) -> dict:
    """The report from already-fetched inputs. Pure apart from reading `time` for `made_at`."""
    indexes = []
    for key, sym, name, note, unit in report.INDEXES:
        c = _change(series, sym, start, end, unit)
        indexes.append({"key": key, "symbol": sym, "name": name, "note": note, "unit": unit,
                        "measured": c is not None, **(c or {})})
    sp = next(i for i in indexes if i["key"] == "sp500")
    if not sp["measured"]:
        raise ReportError("the S&P 500 could not be measured for this period")

    typical = _change(series, report.TYPICAL_STOCK, start, end)
    sectors = []
    for sym, name in report.SECTORS:
        c = _change(series, sym, start, end)
        sectors.append({"symbol": sym, "name": name, "pct": c["pct"] if c else None})
    sectors.sort(key=lambda s: (s["pct"] is None, -(s["pct"] or 0.0)))

    sector_syms = {s for s, _ in report.SECTORS}
    etf_rows = []
    for sym, label in report.ETFS.items():
        if sym in sector_syms:
            continue
        c = _change(series, sym, start, end)
        if c:
            etf_rows.append({"symbol": sym, "name": label, "pct": c["pct"], "close": c["close"]})

    stock_rows, suspect = [], []
    for d in big:
        sym = str(d["symbol"])
        s = series.get(sym)
        if s is None:
            continue
        # A reverse split can come back on a mixed basis even in a fresh adjusted series (the scan's
        # BYND lesson); such a name would top or bottom the list with a fictional move.
        if swing.implausible_jump(s.closes):
            suspect.append(sym)
            continue
        c = _change(series, sym, start, end)
        if c:
            stock_rows.append({"symbol": sym, "name": report.clean_name(d.get("name")) or sym,
                               "pct": c["pct"], "close": c["close"]})
    if big and len(stock_rows) < _MIN_STOCK_COVERAGE * len(big):
        raise ReportError(f"only {len(stock_rows)} of {len(big)} big companies could be measured")

    br = report.breadth(stock_rows)
    nas = next(i for i in indexes if i["key"] == "nasdaq")
    tech_led = bool(sectors and sectors[0]["name"] == "Tech" and sectors[0]["pct"] is not None
                    and nas.get("pct") is not None and nas["pct"] > sp["pct"])
    stocks = {**report.movers(stock_rows, dedupe_key="name"),
              "universe": "Companies worth $10B or more", "measured": len(stock_rows),
              "attempted": len(big), "suspect": sorted(suspect)}
    etfs = {**report.movers(etf_rows, dedupe_key="name"),
            "universe": "Popular funds, no leveraged or inverse funds", "measured": len(etf_rows),
            "attempted": sum(1 for s in report.ETFS if s not in sector_syms)}
    gspc = series.get("^GSPC")
    # The index has no dividends or splits: its adjusted and raw closes are the same numbers.
    daily = report.daily_moves(gspc.dates, gspc.closes, start, end) if (kind == "month" and gspc) else None

    arms = []
    for arm, nav in nav_by_arm.items():
        p = report.sandbox_period(nav, start, end)
        arms.append({"arm": arm, "label": labels.get(arm) or arm, "main": arm == sandbox_store.MAIN_ARM,
                     "measured": p is not None, **(p or {})})
    arms.sort(key=lambda a: (not a["measured"], -(a.get("change_pct") or 0.0)))
    main = next((a for a in arms if a["main"]), None)
    sandbox = {"available": bool(main and main["measured"]),
               "note": ("Measured at the sandbox's 3:35 PM ET check each day, like the Sandbox tab. "
                        "Its S&P line is the same money in the S&P, measured at the same moments."),
               "main": main if main and main["measured"] else None,
               "trades": report.period_trades(main_trades, start, end),
               "arms": arms}
    if not sandbox["available"]:
        sandbox["reason"] = "The sandbox has no record covering this period"

    return {
        "version": VERSION, "id": report.report_id(kind, end), "kind": kind,
        "start_close": start.isoformat(), "end": end.isoformat(),
        "label": report.period_label(kind, start, end), "sessions": report.sessions_in(start, end),
        "made_at": round(time.time() if now is None else now, 3),
        "headline": report.headline(kind, sp["pct"], br["up_share"], tech_led=tech_led),
        "market": {
            "indexes": indexes,
            "typical_stock": {"symbol": report.TYPICAL_STOCK, "name": "Typical stock",
                              "note": "The S&P 500 with every company counted the same",
                              "measured": typical is not None, **(typical or {})},
            "breadth": {**br, "universe": "Companies worth $10B or more"},
            "sectors": sectors, "stocks": stocks, "etfs": etfs, "daily": daily,
            # Only the named rows' failures; the big companies' shortfall is `stocks.measured` against
            # `stocks.attempted`, which is the number a reader needs, not 30 tickers.
            "failed": [s for s in failed if s in _core_symbols()],
        },
        "daily_pick": report.daily_pick_summary(pick_runs, start, end),
        "sandbox": sandbox,
    }


async def build(kind: str, *, client: httpx.AsyncClient, end: dt.date | None = None,
                force: bool = False, now_et: dt.datetime | None = None) -> dict:
    """Build (or return the stored) report for `kind`. `end` defaults to the newest closed period.

    Returns {"status": "built" | "exists", "report": {...}}; raises ReportError when the period cannot
    be measured well enough to store."""
    if kind not in report.KINDS:
        raise ValueError(f"kind must be one of {report.KINDS}")
    now_et = now_et or dt.datetime.now(_ET)
    last = last_closed_session(now_et)
    if end is None:
        start, end = report.latest_completed(kind, last)
    else:
        b = report.bounds(kind, end)
        if b is None or b[1] != end:
            raise ValueError(f"{end.isoformat()} is not the last session of its {kind}")
        if end > last:
            raise ValueError(f"the {kind} ending {end.isoformat()} has not closed yet")
        start = b[0]
    rid = report.report_id(kind, end)
    async with _build_lock:
        if not force and (existing := load(rid)) is not None:
            return {"status": "exists", "report": existing}
        blob = universe.load()
        big = _big_companies(blob)
        syms = ([s for _, s, _, _, _ in report.INDEXES] + [report.TYPICAL_STOCK]
                + [s for s, _ in report.SECTORS] + list(report.ETFS) + [str(r["symbol"]) for r in big])
        t0 = time.monotonic()
        series, failed = await _fetch_all(client, syms)
        log.info("report %s: fetched %d of %d symbols in %.1fs", rid, len(series), len(set(syms)),
                 time.monotonic() - t0)
        arms = sandbox_store.list_arms()
        rep = assemble(
            kind, start, end, series, failed, big,
            nav_by_arm={a: sandbox_store.read_nav(None, a) for a in arms},
            labels={a: (sandbox_store.get(a).get("label") or a) for a in arms},
            main_trades=list(reversed(sandbox_store.read_trades(10 ** 7, sandbox_store.MAIN_ARM))),
            pick_runs=daily_pick_store.runs(limit=400),
        )
        save(rep)
        return {"status": "built", "report": rep}


def _main() -> None:
    ap = argparse.ArgumentParser(description="Build the weekly or monthly report")
    ap.add_argument("--kind", choices=report.KINDS, required=True)
    ap.add_argument("--end", help="YYYY-MM-DD, the period's last session (default: newest closed)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    async def run() -> dict:
        async with httpx.AsyncClient() as client:
            return await build(a.kind, client=client, force=a.force,
                               end=dt.date.fromisoformat(a.end) if a.end else None)

    out = asyncio.run(run())
    rep = out["report"]
    print(out["status"], rep["id"], "·", rep["headline"])


if __name__ == "__main__":
    _main()
