"""
Nightly watchlist scan. Run standalone (`python -m app.scan_job`, wired to a systemd timer) or via
POST /scan/run. Scores each configured symbol with the cheap scan model, flags "flips" (the signal
changed vs. the previous run), and writes data/scan_latest.json for the app to poll.

Cost note: this runs the symbols concurrently through the same structured-output analyst path as
/signal. The Anthropic Batch API (~50% cheaper) is a future optimization — for a personal-size
watchlist the absolute nightly cost is a few cents either way, so correctness/simplicity wins.
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import date, timedelta
from pathlib import Path

import httpx

from . import cycle, settings_store, shorts, usage_store
from .analyst import analyze
from .market import fetch_series, summarize
from .news import fetch_context

LATEST = Path(__file__).resolve().parent.parent / "data" / "scan_latest.json"


async def _score(client: httpx.AsyncClient, symbol: str, crypto: bool, bench_closes: list[float] | None) -> dict:
    series = await fetch_series(client, symbol)
    if len(series.closes) < 30:
        raise ValueError("not enough history")
    summary = summarize(series, None if crypto else bench_closes)
    squeeze = None
    below_200wma = None
    if crypto:
        try:
            summary.update(await cycle.crypto_context(client, series.symbol, series.closes))
            below_200wma = summary.get("long_term_trend", {}).get("below_line")
        except Exception:  # noqa: BLE001
            pass
    if not crypto:
        summary.update(await fetch_context(client, series.symbol))
        try:  # short-pressure enrichment — best-effort, cached across the whole scan
            sp = await shorts.short_pressure(
                client, series.symbol, dates=series.dates, closes=series.closes, volumes=series.volumes,
            )
            if sp:
                summary["short_pressure"] = shorts.compact(sp)
                squeeze = sp["state"]
        except Exception:  # noqa: BLE001
            pass
        try:  # 200-week-line state for cross-below alerts — a neutral event flag, NOT fed to the analyst
            lt = (await cycle.crypto_context(client, series.symbol, series.closes)).get("long_term_trend")
            below_200wma = lt.get("below_line") if lt else None
        except Exception:  # noqa: BLE001
            pass
    verdict, usage = await analyze(summary, deep=False)
    usage_store.record(usage, symbol=series.symbol, kind="scan")
    return {
        "symbol": series.symbol,
        "signal": verdict.signal.value,
        "conviction": verdict.conviction,
        "thesis": verdict.thesis,
        "squeeze": squeeze,
        "below_200wma": below_200wma,
        "cost_usd": usage["cost_usd"],
    }


def _prev_state() -> dict[str, dict]:
    if not LATEST.exists():
        return {}
    try:
        return {
            r["symbol"]: {"signal": r.get("signal"), "squeeze": r.get("squeeze"),
                          "below_200wma": r.get("below_200wma")}
            for r in json.loads(LATEST.read_text()).get("results", [])
            if "signal" in r
        }
    except Exception:  # noqa: BLE001
        return {}


async def run_scan() -> dict:
    cfg = settings_store.get()
    stocks = cfg.get("watchlist", [])
    cryptos = cfg.get("crypto_watchlist", [])
    prev = _prev_state()

    async with httpx.AsyncClient() as client:
        bench: list[float] | None = None
        if stocks:
            try:
                bench = (await fetch_series(client, "^GSPC")).closes
            except Exception:  # noqa: BLE001 — relative strength just gets skipped
                bench = None

        async def one(sym: str, crypto: bool) -> dict:
            try:
                r = await _score(client, sym, crypto, bench)
            except Exception as e:  # noqa: BLE001
                return {"symbol": sym.upper(), "error": str(e)}
            p = prev.get(r["symbol"], {})
            r["prev_signal"] = p.get("signal")
            r["flipped"] = r["prev_signal"] is not None and r["prev_signal"] != r["signal"]
            # Squeeze-state transitions (quiet→fuel→ignition) are notification-worthy events too.
            r["prev_squeeze"] = p.get("squeeze")
            r["squeeze_changed"] = (
                r.get("squeeze") is not None
                and r["prev_squeeze"] is not None
                and r["squeeze"] != r["prev_squeeze"]
            )
            # Newly below the 200-week line this run — mungbeans' weekly signal, surfaced as a
            # neutral "heads up" event (a mirror of the flipped diff; first scan has prev=None → no alert).
            r["prev_below_200wma"] = p.get("below_200wma")
            r["crossed_below_200wma"] = r.get("below_200wma") is True and p.get("below_200wma") is False
            return r

        results = list(await asyncio.gather(
            *[one(s, False) for s in stocks],
            *[one(s, True) for s in cryptos],
        ))

    # Day-of / day-before key-date alerts (SI publication, OPEX, earnings, speculative T+35 echoes)
    # so the app can warn BEFORE the event, not after.
    date_alerts: list[str] = []
    try:
        async with httpx.AsyncClient() as c2:
            cal = await shorts.calendar(c2, stocks)
        today, tomorrow = date.today().isoformat(), (date.today() + timedelta(days=1)).isoformat()
        for e in cal:
            if e["date"] in (today, tomorrow):
                when = "Today" if e["date"] == today else "Tomorrow"
                sym = f"{e['symbol']} " if e.get("symbol") else ""
                date_alerts.append(f"{when}: {sym}{e['label']}")
    except Exception:  # noqa: BLE001 — alerts are enrichment
        pass

    payload = {
        "generated_at": time.time(),
        "results": results,
        "flips": [r["symbol"] for r in results if r.get("flipped")],
        "crossed_below_200wma": [r["symbol"] for r in results if r.get("crossed_below_200wma")],
        "date_alerts": date_alerts,
        "total_cost_usd": round(sum(r.get("cost_usd", 0.0) for r in results), 6),
    }
    LATEST.parent.mkdir(parents=True, exist_ok=True)
    LATEST.write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    out = asyncio.run(run_scan())
    print(f"scanned {len(out['results'])} · flips {out['flips']} · ${out['total_cost_usd']}")
