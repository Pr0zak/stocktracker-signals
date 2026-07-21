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

from . import cycle, options, settings_store, shorts, usage_store
from .analyst import analyze
from .market import fetch_series, summarize
from .news import fetch_context

LATEST = Path(__file__).resolve().parent.parent / "data" / "scan_latest.json"


def _dip_tier(
    closes: list[float], pct_off_52w: float | None, below_200wma: bool | None, weekly_oversold: bool | None,
) -> dict:
    """How much of a 'good time to add' this is, most-severe first: mega_dip (>=20% off the 52-week
    high) > below_line (below its 200-week line) > oversold (weekly RSI<30) > pullback_10 / pullback_5
    (off a ~3-month high). None = no dip. A layered 'add EXTRA on weakness' cue, never 'buy signal'."""
    price = closes[-1]
    recent_high = max(closes[-63:]) if len(closes) >= 5 else price   # ~3 months of daily bars
    pct_off_recent = round((price - recent_high) / recent_high * 100, 1) if recent_high else 0.0
    if pct_off_52w is not None and pct_off_52w <= -20:
        tier = "mega_dip"
    elif below_200wma:
        tier = "below_line"
    elif weekly_oversold:
        tier = "oversold"
    elif pct_off_recent <= -10:
        tier = "pullback_10"
    elif pct_off_recent <= -5:
        tier = "pullback_5"
    else:
        tier = None
    return {"dip": tier, "pct_off_recent_high": pct_off_recent, "pct_off_52w_high": pct_off_52w}


# ATM-IV logging window (OC-6a): a ~30-45 DTE expiry — cheaper/shorter-dated than the OC-1 45-90 band.
_IV_LOG_DTE_LOW, _IV_LOG_DTE_HIGH, _IV_LOG_DTE_TARGET = 30, 45, 37


async def _log_atm_iv(client: httpx.AsyncClient, symbol: str) -> None:
    """Best-effort (OC-6a): fetch this stock's option chain, pick a ~30-45 DTE expiry, annotate it, and
    append the ATM IV to data/iv_history.jsonl (one line/symbol/day). Skips crypto/no-chain symbols and
    swallows EVERY error — the nightly scan must never break on IV logging."""
    try:
        chain = await options.fetch_chain(client, symbol)
        if not chain.expirations or not chain.spot or chain.spot <= 0:
            return
        chosen, _ = options.select_wheel_expiry(
            chain, low=_IV_LOG_DTE_LOW, high=_IV_LOG_DTE_HIGH, target=_IV_LOG_DTE_TARGET,
        )
        if chosen is None:
            return
        if not (chain.expiry and chain.expiry.expiration == chosen["ts"]):
            chain = await options.fetch_chain(client, symbol, chosen["ts"])
        if chain.expiry is None:
            return
        options.annotate_expiry(chain)
        options.append_iv_history(chain.symbol, chain.expiry.atm_iv)
    except Exception:  # noqa: BLE001 — IV logging is best-effort; never break the scan
        pass


async def _score(client: httpx.AsyncClient, symbol: str, crypto: bool, bench_closes: list[float] | None) -> dict:
    series = await fetch_series(client, symbol)
    if len(series.closes) < 30:
        raise ValueError("not enough history")
    summary = summarize(series, None if crypto else bench_closes)
    squeeze = None
    below_200wma = None
    weekly_oversold = None
    if crypto:
        try:
            summary.update(await cycle.crypto_context(client, series.symbol, series.closes))
            lt = summary.get("long_term_trend", {})
            below_200wma = lt.get("below_line")
            weekly_oversold = lt.get("weekly_oversold")
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
            weekly_oversold = lt.get("weekly_oversold") if lt else None
        except Exception:  # noqa: BLE001
            pass
    verdict, usage = await analyze(summary, deep=False)
    usage_store.record(usage, symbol=series.symbol, kind="scan")
    if not crypto:  # OC-6a: log this stock's ~30-45 DTE ATM IV for the /options IV-rank read
        await _log_atm_iv(client, series.symbol)
    dip = _dip_tier(series.closes, summary.get("pct_off_52w_high"), below_200wma, weekly_oversold)
    return {
        "symbol": series.symbol,
        "signal": verdict.signal.value,
        "conviction": verdict.conviction,
        "thesis": verdict.thesis,
        "squeeze": squeeze,
        "below_200wma": below_200wma,
        **dip,  # dip tier + pct_off_recent_high + pct_off_52w_high — the "good time to add" read
        "cost_usd": usage["cost_usd"],
    }


def _prev_state() -> dict[str, dict]:
    if not LATEST.exists():
        return {}
    try:
        return {
            r["symbol"]: {"signal": r.get("signal"), "squeeze": r.get("squeeze"),
                          "below_200wma": r.get("below_200wma"), "dip": r.get("dip")}
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
            # Newly entered (or escalated to) a dip tier — the "good time to add" event.
            r["prev_dip"] = p.get("dip")
            r["dip_new"] = r.get("dip") is not None and p.get("dip") != r.get("dip")
            return r

        results = list(await asyncio.gather(
            *[one(s, False) for s in stocks],
            *[one(s, True) for s in cryptos],
        ))

    # Day-of / day-before key-date alerts (SI publication, OPEX, earnings)
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
        "dip_alerts": [
            {"symbol": r["symbol"], "dip": r["dip"],
             "pct_off_recent_high": r.get("pct_off_recent_high"), "pct_off_52w_high": r.get("pct_off_52w_high")}
            for r in results if r.get("dip_new")
        ],
        "date_alerts": date_alerts,
        "total_cost_usd": round(sum(r.get("cost_usd", 0.0) for r in results), 6),
    }
    LATEST.parent.mkdir(parents=True, exist_ok=True)
    LATEST.write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    out = asyncio.run(run_scan())
    print(f"scanned {len(out['results'])} · flips {out['flips']} · ${out['total_cost_usd']}")
