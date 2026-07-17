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
from pathlib import Path

import httpx

from . import settings_store
from .analyst import analyze
from .market import fetch_series, summarize
from .news import fetch_context

LATEST = Path(__file__).resolve().parent.parent / "data" / "scan_latest.json"


async def _score(client: httpx.AsyncClient, symbol: str, crypto: bool, bench_closes: list[float] | None) -> dict:
    series = await fetch_series(client, symbol)
    if len(series.closes) < 30:
        raise ValueError("not enough history")
    summary = summarize(series, None if crypto else bench_closes)
    if not crypto:
        summary.update(await fetch_context(client, series.symbol))
    verdict, usage = await analyze(summary, deep=False)
    return {
        "symbol": series.symbol,
        "signal": verdict.signal.value,
        "conviction": verdict.conviction,
        "thesis": verdict.thesis,
        "cost_usd": usage["cost_usd"],
    }


def _prev_signals() -> dict[str, str]:
    if not LATEST.exists():
        return {}
    try:
        return {r["symbol"]: r.get("signal") for r in json.loads(LATEST.read_text()).get("results", []) if "signal" in r}
    except Exception:  # noqa: BLE001
        return {}


async def run_scan() -> dict:
    cfg = settings_store.get()
    stocks = cfg.get("watchlist", [])
    cryptos = cfg.get("crypto_watchlist", [])
    prev = _prev_signals()

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
            r["prev_signal"] = prev.get(r["symbol"])
            r["flipped"] = r["prev_signal"] is not None and r["prev_signal"] != r["signal"]
            return r

        results = list(await asyncio.gather(
            *[one(s, False) for s in stocks],
            *[one(s, True) for s in cryptos],
        ))

    payload = {
        "generated_at": time.time(),
        "results": results,
        "flips": [r["symbol"] for r in results if r.get("flipped")],
        "total_cost_usd": round(sum(r.get("cost_usd", 0.0) for r in results), 6),
    }
    LATEST.parent.mkdir(parents=True, exist_ok=True)
    LATEST.write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    out = asyncio.run(run_scan())
    print(f"scanned {len(out['results'])} · flips {out['flips']} · ${out['total_cost_usd']}")
