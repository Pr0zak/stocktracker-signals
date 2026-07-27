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
import logging
import time
import datetime as dt
from datetime import date, timedelta
from pathlib import Path

import httpx

from . import cycle, market_calendar, memory, options, settings_store, shorts, usage_store
from .analyst import analyze
from .market import Series, fetch_series, summarize
from .news import fetch_context

log = logging.getLogger("signals.scan")

# How many newly-tracked symbols to seed base rates for per nightly run. Each costs one 2y Yahoo
# fetch plus a few hundred cheap inserts, so a handful a night is invisible; the cap only matters
# when a batch of names is added at once.
_BACKFILL_PER_RUN = 4

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
        options.append_iv_history(chain.symbol, chain.expiry.atm_iv, dte=int(chosen["dte"]))
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
    # The scan sees the same setup every trading day across the whole watchlist, which makes it by
    # far the richest source of "what did this setup actually do next" — so recall feeds in here and
    # the result is recorded, exactly as on the interactive /signal path.
    try:
        track = memory.similar_setups(series.symbol, summary)
        if track:
            summary["track_record"] = track
    except Exception:  # noqa: BLE001 — enrichment, never a blocker
        pass
    verdict, usage = await analyze(summary, deep=False)
    usage_store.record(usage, symbol=series.symbol, kind="scan")
    memory.record_verdict(
        symbol=series.symbol, summary=summary, verdict=verdict.model_dump(),
        model=settings_store.get()["scan_model"], deep=False,
    )
    if not crypto:  # OC-6a: log this stock's ~30-45 DTE ATM IV for the /options IV-rank read
        # Only on real trading days. The scan runs every calendar day, and append_iv_history de-dupes
        # per CALENDAR date, so weekends and holidays each added a row carrying Friday's stale IV —
        # inflating the history ~1.4x and letting the 20-point "still building" gate clear after ~14
        # sessions instead of 20, against a window documented in trading days.
        if market_calendar.is_trading_day(dt.date.today()):
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


async def score_memory() -> int:
    """Grade every verdict old enough to have a 20-day outcome. Returns rows scored.

    Fetches 1y of bars per symbol with pending verdicts, plus the benchmark once, and hands both to
    `memory.score_symbol`. Symbols are fetched with limited concurrency: this runs unattended
    alongside the scan and the point is never to hammer Yahoo hard enough to get rate-limited out of
    the data the user actually asked for.
    """
    syms = memory.pending_symbols()
    if not syms:
        return 0
    total = 0
    async with httpx.AsyncClient() as client:
        bench_dates: list[str] | None = None
        bench_closes: list[float] | None = None
        try:
            b = await fetch_series(client, "^GSPC")
            bench_dates, bench_closes = b.dates, b.closes
        except Exception:  # noqa: BLE001 — without it we still score raw return, just not excess
            pass

        sem = asyncio.Semaphore(4)

        async def one(sym: str) -> int:
            async with sem:
                try:
                    s = await fetch_series(client, sym)
                except Exception:  # noqa: BLE001 — a delisted/renamed ticker just stays unscored
                    return 0
                # Crypto trades weekends, so aligning it to the S&P's calendar would drop most rows;
                # score those on raw return only rather than against a mismatched window.
                same_calendar = bench_dates is not None and not _is_crypto_symbol(sym)
                return memory.score_symbol(
                    sym, s.dates, s.closes,
                    bench_dates=bench_dates if same_calendar else None,
                    bench_closes=bench_closes if same_calendar else None,
                )

        total = sum(await asyncio.gather(*[one(s) for s in syms]))
    if total:
        log.info("memory: scored %d verdict(s) across %d symbol(s)", total, len(syms))
    memory.prune()
    return total


async def backfill_memory(symbols: list[str], *, every: int = 3, rng: str = "2y") -> dict:
    """Seed memory by replaying real price history.

    Without this the track record is empty for ~20 trading days after deployment, since a verdict
    can't be graded until its horizon exists. Setup base rates, though, are a property of PRICE — they
    don't need a verdict to have been made — so every `every`-th historical bar is re-summarized from
    the truncated series and stored with `origin="backfill"` and NO signal attached.

    That distinction is load-bearing: backfilled rows answer "what does this pattern usually do?", and
    are excluded from `when_model_said_buy`, which must keep meaning "when the MODEL said buy".
    """
    out = {"symbols": 0, "recorded": 0, "scored": 0}
    async with httpx.AsyncClient() as client:
        bench = None
        try:
            bench = await fetch_series(client, "^GSPC", rng=rng)
        except Exception:  # noqa: BLE001
            pass
        bench_idx = {d: i for i, d in enumerate(bench.dates)} if bench else {}

        for sym in symbols:
            # Stamp the attempt BEFORE deciding whether it worked, so a symbol that can never be
            # backfilled rotates to the back of the queue instead of consuming a slot every night.
            memory.add_note("seed_attempt", f"backfill attempt for {sym}", symbol=sym)
            try:
                s = await fetch_series(client, sym, rng=rng)
            except Exception:  # noqa: BLE001
                continue
            if len(s.closes) < 90:
                continue
            out["symbols"] += 1
            crypto = _is_crypto_symbol(sym)
            # Start at 60 so SMA50/RSI are warm; stop 21 bars from the end so only rows that can
            # actually be scored get written.
            for i in range(60, len(s.closes) - 21, max(1, every)):
                sub = Series(
                    symbol=s.symbol, closes=s.closes[:i + 1], opens=s.opens[:i + 1],
                    volumes=s.volumes[:i + 1], dates=s.dates[:i + 1],
                    fifty_two_high=max(s.closes[max(0, i - 252):i + 1]),
                    fifty_two_low=min(s.closes[max(0, i - 252):i + 1]),
                    currency=s.currency,
                )
                bc = None
                if bench and not crypto:
                    bj = bench_idx.get(s.dates[i])
                    bc = bench.closes[:bj + 1] if bj else None
                if memory.record_verdict(
                    symbol=s.symbol, summary=summarize(sub, bc), verdict={},
                    origin="backfill", ts=_bar_ts(s.dates[i]),
                ):
                    out["recorded"] += 1
            out["scored"] += memory.score_symbol(
                sym, s.dates, s.closes,
                bench_dates=bench.dates if (bench and not crypto) else None,
                bench_closes=bench.closes if (bench and not crypto) else None,
            )
    log.info("memory backfill: %s", out)
    return out


def _bar_ts(yyyymmdd: str) -> float:
    """Epoch seconds for a YYYYMMDD bar, so backfilled rows age out on the real retention schedule
    instead of all looking like they were written the day the backfill ran."""
    try:
        return time.mktime(time.strptime(yyyymmdd, "%Y%m%d"))
    except ValueError:
        return time.time()


def _is_crypto_symbol(sym: str) -> bool:
    """Yahoo crypto tickers are `BTC-USD`-shaped; equities never contain a dash suffix like that."""
    return "-" in sym and sym.rsplit("-", 1)[-1].upper() in ("USD", "USDT", "EUR", "GBP")


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

    # Grade past verdicts against what price actually did. The scan is the natural home for this:
    # it already runs once a day and the horizons are measured in days, so nothing is gained by
    # scoring more often. Best-effort — a scoring failure must not cost the user their scan.
    scored = 0
    try:
        scored = await score_memory()
    except Exception:  # noqa: BLE001
        log.warning("memory scoring failed", exc_info=True)

    # Seed base rates for symbols added to the watchlist since the last backfill. Without this a
    # newly-tracked ticker silently has no track record for ~20 trading days and the UI just omits
    # it, which is indistinguishable from "this setup has no edge". Bounded per run so adding a
    # dozen names at once spreads the fetches over a few nights instead of hammering Yahoo.
    seeded: dict = {}
    try:
        fresh = memory.symbols_missing_baseline(stocks + cryptos)[:_BACKFILL_PER_RUN]
        if fresh:
            log.info("memory: seeding base rates for %s", fresh)
            seeded = await backfill_memory(fresh)
    except Exception:  # noqa: BLE001 — seeding is enrichment, never a blocker
        log.warning("memory seeding failed", exc_info=True)

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
        "dip_alerts": [
            {"symbol": r["symbol"], "dip": r["dip"],
             "pct_off_recent_high": r.get("pct_off_recent_high"), "pct_off_52w_high": r.get("pct_off_52w_high")}
            for r in results if r.get("dip_new")
        ],
        "date_alerts": date_alerts,
        "memory_scored": scored,
        "memory_seeded": seeded or None,
        "total_cost_usd": round(sum(r.get("cost_usd", 0.0) for r in results), 6),
    }
    LATEST.parent.mkdir(parents=True, exist_ok=True)
    LATEST.write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    out = asyncio.run(run_scan())
    print(f"scanned {len(out['results'])} · flips {out['flips']} · ${out['total_cost_usd']}")
