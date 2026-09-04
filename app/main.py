"""StockTracker Signals — Tier-2 Claude analyst service. Decision support only, not advice."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import copy
import time
from collections.abc import Iterable
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import observability, redact, selfupdate, settings_store, usage_store
from . import chase, congress, cycle, fundamentals, insider, market_now, options, seasonality, shorts, webull
from . import dashboard, gaps, gate, market_calendar, market_scan_job, memory, percentiles, plan_check, plan_replay, rebalance_check, heatmap, sandbox_job, sandbox_store, scan_store, screener, smartmoney, universe, valuetrap
from .analyst import (
    analyze,
    daily_brief,
    market_overview,
    market_regime,
    news_moves,
    options_note,
    plan_entry,
    rebalance_portfolio,
    recommend,
    review_decision,
    review_portfolio,
    sandbox_decision,
    strategy_review,
)
from .discover import WIDE_SCREENS, discover
from .market import fetch_series, summarize
from .news import earnings_on, fetch_context, fetch_dated_news, fetch_next_earnings
from . import macro, scan_job, sectors
from .macro_job import run_macro
from .scan_job import LATEST, run_scan

_http: httpx.AsyncClient | None = None
_cache: dict[tuple, tuple[float, dict]] = {}
_MARKET_NOW_TTL = 180  # market-now overview cached ~3 min so repeated taps don't re-run the model
# A FAILED news lookup is parked in the 1h cache with its timestamp back-dated by this much, so it
# expires after ~60s. It still needs to be cached briefly — otherwise a symbol whose news source is
# down re-hammers it on every poll — but freezing "we couldn't look" for a full hour turns a blip
# into an hour of confidently wrong "no news".
_NEWS_FAIL_TTL_OFFSET = 3600 - 60
_DAILY_BRIEF_TTL = 1800  # morning brief cached ~30 min — the app fires it once/day; this just guards retries
# ...unless its catalysts came back unknown, in which case it is held just long enough to coalesce a
# burst of retries rather than to settle the question for the morning.
_BRIEF_INCOMPLETE_TTL = 300
# How many catalyst rows /calendar returns. Generous — the payload is small and the lookups behind it
# are now one request each — but still bounded, and the response says when it bit.
_CALENDAR_MAX_EVENTS = 60
_log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http
    # SEC-1, and it goes FIRST: every log line emitted from here on is scrubbed of secret-bearing
    # query parameters. uvicorn has finished configuring logging by the time lifespan runs, so this
    # sees the handlers records will actually reach.
    redact.install_log_filter()
    _http = httpx.AsyncClient()
    try:
        memory.seed_research()  # idempotent by slug; safe on every restart
    except Exception:  # noqa: BLE001 — memory must never block startup
        _log.warning("memory: seeding skipped", exc_info=True)
    try:
        yield
    finally:
        await _http.aclose()


app = FastAPI(title="StockTracker Signals", version="0.2.0", lifespan=lifespan)


@app.middleware("http")
async def _timing_middleware(request: Request, call_next):
    """Time every request and record it in the in-memory ring (no disk I/O, never slows the path).
    A handler that raises is logged as a 500 before the exception propagates."""
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as e:  # noqa: BLE001 — record then re-raise so FastAPI still returns its 500
        observability.record_request(
            request.method, request.url.path, 500, (time.perf_counter() - t0) * 1000.0, error=str(e),
        )
        raise
    observability.record_request(
        request.method, request.url.path, response.status_code, (time.perf_counter() - t0) * 1000.0,
    )
    return response


# --- settings UI + API ---

@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    return dashboard.PAGE


def _cli_token() -> str:
    """The EFFECTIVE headless-CLI subscription token — the one saved via the settings UI, else the
    CLAUDE_CODE_OAUTH_TOKEN service env var. Used only for the masked settings-page status; the full
    value never leaves the server (only a last-4 hint does)."""
    import os
    return (settings_store.get().get("cli_oauth_token") or "").strip() or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")


@app.get("/api/cli-auth-test")
async def cli_auth_test() -> dict:
    """Live check that the headless claude CLI is installed + authenticated (subscription token) — one
    tiny call. Backs the settings page's 'Test CLI auth' button. Never 500s (returns {ok:false,...})."""
    from . import llm_cli
    return await llm_cli.auth_probe(settings_store.get().get("scan_model", "claude-haiku-4-5"))


@app.get("/api/settings")
async def get_settings() -> dict:
    cfg = settings_store.get()
    key = cfg["anthropic_api_key"]
    tok = _cli_token()
    return {
        "anthropic_api_key_set": bool(key),
        "anthropic_api_key_hint": ("…" + key[-4:]) if len(key) >= 4 else ("set" if key else ""),
        "finnhub_api_key_set": bool(cfg.get("finnhub_api_key", "")),
        "deep_model": cfg["deep_model"],
        "scan_model": cfg["scan_model"],
        "llm_provider": cfg.get("llm_provider", "api"),
        "cli_token_set": bool(tok),
        "cli_token_hint": ("…" + tok[-4:]) if len(tok) >= 4 else ("set" if tok else ""),
        "verdict_ttl_seconds": cfg["verdict_ttl_seconds"],
        "watchlist": cfg.get("watchlist", []),
        "crypto_watchlist": cfg.get("crypto_watchlist", []),
        "watchlist_synced_at": cfg.get("watchlist_synced_at"),
    }


class SettingsPatch(BaseModel):
    anthropic_api_key: str | None = None
    finnhub_api_key: str | None = None
    deep_model: str | None = None
    scan_model: str | None = None
    llm_provider: str | None = None
    cli_oauth_token: str | None = None
    verdict_ttl_seconds: int | None = None
    watchlist: str | list[str] | None = None
    crypto_watchlist: str | list[str] | None = None


@app.post("/api/settings")
async def post_settings(patch: SettingsPatch) -> dict:
    settings_store.update(patch.model_dump(exclude_none=True))
    return await get_settings()


@app.get("/api/usage")
async def api_usage(days: int = 30) -> dict:
    """All-time token/cost totals + a per-day series for the last `days` days."""
    return await asyncio.to_thread(usage_store.summary, max(1, min(365, days)))


@app.get("/api/version")
async def api_version() -> dict:
    return await asyncio.to_thread(selfupdate.status)


@app.post("/api/update")
async def api_update() -> dict:
    return await asyncio.to_thread(selfupdate.update)


# --- ops + transparency dashboard API ---

@app.get("/api/status")
async def api_status() -> dict:
    """Operations snapshot: uptime, disk, last scan (counts/cost/changed), next-scan time, cache
    footprint, and per-symbol IV-rank progress. Cheap read-only file reads; offloaded to a thread."""
    return await asyncio.to_thread(observability.status_snapshot)


@app.get("/api/sources")
async def api_sources() -> dict:
    """Concurrent short-timeout liveness probes of every upstream data source."""
    assert _http is not None
    return {"as_of": time.time(), "sources": await observability.probe_sources(_http)}


@app.get("/api/logs")
async def api_logs(limit: int = 50) -> dict:
    """Recent served requests + recent non-2xx errors from the in-memory ring."""
    limit = max(1, min(200, limit))
    return {
        "requests": observability.recent(limit),
        "errors": observability.recent_errors(min(limit, 50)),
    }


@app.get("/api/cost")
async def api_cost() -> dict:
    """Per-kind cost/token breakdown + month-to-date, projected month, and per-scan/per-deep averages."""
    return await asyncio.to_thread(observability.cost_breakdown)


@app.post("/api/prune-cache")
async def api_prune_cache() -> dict:
    """Delete stale whole-market shvol_/ftd_ caches under data/shorts/ (older than ~90 days) and
    report bytes freed. Never touches settings/scan/usage/iv-history files."""
    return await asyncio.to_thread(observability.prune_shorts_cache)


@app.get("/health")
async def health() -> dict:
    cfg = settings_store.get()
    return {
        "ok": True,
        "key_configured": bool(cfg["anthropic_api_key"]),
        "deep_model": cfg["deep_model"],
        "scan_model": cfg["scan_model"],
    }


# --- signals ---

def _position_block(summary: dict, shares: float | None, avg_cost: float | None) -> dict | None:
    """The user's holding in the snapshot's terms (value, unrealized P/L), or None if not held."""
    price = summary.get("price")
    if not (shares and avg_cost and shares > 0 and avg_cost > 0 and price):
        return None
    return {
        "shares": round(shares, 6),
        "avg_cost": round(avg_cost, 4),
        "position_value": round(shares * price, 2),
        "unrealized_gain_pct": round((price - avg_cost) / avg_cost * 100.0, 2),
        "unrealized_gain_abs": round(shares * (price - avg_cost), 2),
        "currency": summary.get("currency", "USD"),
    }


def _sanitize_plan(p, cash: float, crypto: bool) -> None:
    """Enforce the numeric contract the prompt only requests: entry zone ordered, allocation within
    the cash, shares consistent with allocation/entry (whole for stocks, 6dp for crypto)."""
    if p.entry_low > p.entry_high:
        p.entry_low, p.entry_high = p.entry_high, p.entry_low
    if p.action in ("wait", "avoid"):
        p.allocation_usd = 0.0
        p.suggested_shares = 0.0
        return
    p.allocation_usd = round(max(0.0, min(p.allocation_usd, cash)), 2)
    mid = (p.entry_low + p.entry_high) / 2
    if mid > 0:
        p.suggested_shares = (
            round(p.allocation_usd / mid, 6) if crypto else float(int(p.allocation_usd / mid))
        )


async def _chase_price(symbol: str) -> float | None:
    """The price a chase read has to be taken at: the CURRENT session's print (pre/post included),
    not the frozen 4pm close — "am I paying up?" is a question about right now.

    None on any failure, and the caller must keep it None rather than substituting a close. A chase
    read against a price we could not fetch is not a read.
    """
    assert _http is not None
    try:
        quotes = await market_now.fetch_quotes(_http, [symbol])
    except Exception:  # noqa: BLE001 — a missing quote costs the chase read, never the plan itself
        return None
    return market_now.session_price(quotes.get(symbol.upper()) or {})


async def _chase_prices(symbols: list[str]) -> dict[str, float | None]:
    """Session prices for many symbols in ONE batched quote, for /recommendations (SWT-15).

    The per-symbol `_chase_price` would be one Yahoo round trip per pick. Same rule as there: a
    symbol whose price could not be read is absent from the result, and the caller must let it stay
    absent — a chase read against a price we could not fetch is not a read.
    """
    assert _http is not None
    syms = sorted({s.strip().upper() for s in symbols if s and s.strip()})
    if not syms:
        return {}
    try:
        quotes = await market_now.fetch_quotes(_http, syms)
    except Exception:  # noqa: BLE001 — a missing quote costs the chase read, never the picks
        return {}
    out: dict[str, float | None] = {}
    for s in syms:
        px = market_now.session_price(quotes.get(s) or {})
        if px is not None:
            out[s] = px
    return out


async def _snapshot(symbol: str, *, crypto: bool, bench_closes: list[float] | None = None) -> dict:
    """Fetch + summarize one asset's daily technicals (plus news/earnings for stocks). Pass
    `bench_closes` to reuse an already-fetched S&P series (batch callers); stocks fetch it otherwise."""
    assert _http is not None
    try:
        series = await fetch_series(_http, symbol)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"data fetch failed: {redact.redact(e)}")
    if len(series.closes) < 30:
        raise HTTPException(status_code=422, detail="not enough history for a signal")

    if not crypto and bench_closes is None:  # relative strength vs the S&P is equity-only
        try:
            bench_closes = (await fetch_series(_http, "^GSPC")).closes
        except Exception:  # noqa: BLE001 — RS just gets skipped
            bench_closes = None

    summary = summarize(series, None if crypto else bench_closes)
    # Multi-year trend for EVERY asset (plus, for BTC, the halving-cycle position — weak-sample
    # context, flagged as such). This used to be crypto-only, which left an asymmetry with no
    # justification behind it: a stock's position against its own 200-week line is exactly as real as
    # bitcoin's, and its absence meant the per-symbol verdict and the entry plan could only ever see
    # a name through a ≤3-month window. `crypto_context` is asset-agnostic despite the name — it is
    # what already powers the equity /trend endpoint. Cached 6h per symbol; best-effort.
    try:
        summary.update(await cycle.crypto_context(_http, series.symbol, series.closes))
    except Exception:  # noqa: BLE001 — long-term context is enrichment, never a blocker
        pass
    if not crypto:  # optional news/earnings context (Finnhub, stocks only)
        summary.update(await fetch_context(_http, series.symbol))
        # Short-pressure context (FINRA SI + daily short volume + SEC FTDs) — best-effort; the
        # sources cache aggressively so a watchlist sweep costs one download per file, not per symbol.
        try:
            sp = await shorts.short_pressure(
                _http, series.symbol, dates=series.dates, closes=series.closes, volumes=series.volumes,
            )
            if sp:
                summary["short_pressure"] = shorts.compact(sp)
        except Exception:  # noqa: BLE001 — shorts data is enrichment, never a blocker
            pass
        try:  # insider buying (Finnhub Form 4) — the bullish informed-money mirror of short_pressure
            ins = insider.compact(await insider.insider_buying(_http, series.symbol))
            if ins:
                summary["insider"] = ins
        except Exception:  # noqa: BLE001 — enrichment, never a blocker
            pass
        try:  # congressional / political trades (kadoa dataset) — public-official smart-money, lagging
            cg = congress.compact(await congress.congress_trades(_http, series.symbol))
            if cg:
                summary["congress"] = cg
        except Exception:  # noqa: BLE001 — enrichment, never a blocker
            pass
        try:  # seasonality — typical per-month price action from ~10y of monthly bars (weak tilt)
            sea = seasonality.compact(await seasonality.seasonality(_http, series.symbol))
            if sea:
                summary["seasonality"] = sea
        except Exception:  # noqa: BLE001 — enrichment, never a blocker
            pass
        try:  # quality tags (Finnhub basic-financials) — stance-neutral business descriptors
            q = fundamentals.compact(await fundamentals.fetch_quality(_http, series.symbol))
            if q:
                summary["quality"] = q
        except Exception:  # noqa: BLE001 — enrichment, never a blocker
            pass
    return summary


async def _build_signal(
    symbol: str, *, deep: bool, crypto: bool,
    shares: float | None = None, avg_cost: float | None = None,
    rule_score: int | None = None, refresh: bool = False,
) -> dict:
    cfg = settings_store.get()
    # Position is part of the cache identity: a different holding must yield a fresh, re-personalized
    # verdict rather than a stale one keyed only on the symbol.
    key = (symbol.upper(), crypto, deep, shares, avg_cost, rule_score)
    now = time.time()
    hit = _cache.get(key)
    if hit and not refresh and now - hit[0] < cfg["verdict_ttl_seconds"]:
        return {**hit[1], "cached": True}

    summary = await _snapshot(symbol, crypto=crypto)
    # Personalize when the user holds this asset — the analyst frames the verdict as add/hold/trim.
    pos = _position_block(summary, shares, avg_cost)
    if pos:
        summary["position"] = pos
    if rule_score is not None:  # the app's mechanical composite — the analyst reconciles with it
        summary["rule_score"] = max(0, min(100, rule_score))
    # What happened the last time this exact setup showed up — the analyst's own measured track
    # record, not a prior. Absent until enough verdicts have been scored, and never fatal.
    try:
        track = memory.similar_setups(symbol, summary)
        if track:
            summary["track_record"] = track
    except Exception:  # noqa: BLE001 — recall is enrichment, never a blocker
        pass
    # Market-wide exogenous backdrop (NEWS-4) — same read the sandbox and the nightly scan see, so a
    # tap-through verdict and the overnight one reason from the same world.
    try:
        mac = macro.compact(macro.load_state(), limit=3)
        if mac:
            summary["macro"] = mac
    except Exception:  # noqa: BLE001 — enrichment, never a blocker
        pass
    try:
        verdict, usage = await analyze(summary, deep=deep)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"analyst failed: {redact.redact(e)}")
    usage_store.record(usage, symbol=summary.get("symbol", symbol.upper()), kind="deep" if deep else "signal")
    memory.record_verdict(
        symbol=summary.get("symbol", symbol.upper()), summary=summary, verdict=verdict.model_dump(),
        model=cfg["deep_model"] if deep else cfg["scan_model"], deep=deep,
    )

    payload = {
        "symbol": summary.get("symbol", symbol.upper()),
        "model": cfg["deep_model"] if deep else cfg["scan_model"],
        "as_of": now,
        "summary": summary,
        "verdict": verdict.model_dump(),
        "usage": usage,
        "cached": False,
    }
    _cache[key] = (now, payload)
    return payload


@app.get("/signal/{symbol}")
async def signal(
    symbol: str, deep: bool = False, crypto: bool = False,
    shares: float | None = None, avg_cost: float | None = None,
    rule_score: int | None = None, refresh: bool = False,
) -> dict:
    """One asset's analyst verdict. `deep=true` uses the deep model; crypto symbols use Yahoo's
    `BTC-USD` form with `crypto=true` (skips the S&P benchmark). Optional `shares` + `avg_cost`
    personalize the verdict as an add/hold/trim call on an existing position; optional `rule_score`
    (the app's mechanical 0-100 composite) makes the analyst reconcile a diverging read.
    `refresh=true` bypasses the cache to force a freshly generated verdict."""
    return await _build_signal(
        symbol, deep=deep, crypto=crypto, shares=shares, avg_cost=avg_cost,
        rule_score=rule_score, refresh=refresh,
    )


@app.get("/history/{symbol}")
async def history_endpoint(symbol: str) -> dict:
    """Daily close+volume bars for a symbol. fetch_series is Yahoo-primary with a Webull fallback
    (warrants/OTC), so this reports whichever source actually supplied the data."""
    assert _http is not None
    try:
        s = await fetch_series(_http, symbol)
    except Exception:  # noqa: BLE001
        raise HTTPException(status_code=404, detail="no historical data for this symbol")
    bars = [
        {"t": int(time.mktime(time.strptime(d, "%Y%m%d"))) * 1000, "c": c, "v": v or 0.0}
        for d, c, v in zip(s.dates, s.closes, s.volumes)
    ]
    return {"symbol": s.symbol, "source": s.source, "bars": bars}


@app.get("/shorts/{symbol}")
async def shorts_endpoint(symbol: str) -> dict:
    """Full short-pressure read for one stock (no LLM, free): state, days-to-cover, short-volume
    ratio, FTD series/trend, per-symbol event study after past FTD spikes, and upcoming key dates."""
    key = ("shorts", symbol.upper())
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < 900:  # 15 min — underlying sources cache far longer anyway
        return {**hit[1], "cached": True}
    assert _http is not None
    try:
        series = await fetch_series(_http, symbol)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"data fetch failed: {redact.redact(e)}")
    sp = await shorts.short_pressure(
        _http, series.symbol, dates=series.dates, closes=series.closes, volumes=series.volumes,
    )
    if sp is None:
        raise HTTPException(status_code=404, detail="no short data available for this symbol")
    payload = {"symbol": series.symbol, "as_of": now, **sp, "cached": False}
    _cache[key] = (now, payload)
    return payload


async def _movers_side(scr: str, count: int) -> list[dict]:
    """One side of the market-wide movers (a Yahoo predefined screener). Best-effort → [] on failure."""
    assert _http is not None
    from .discover import _raw, _screen  # reuse the screener fetch + raw-field unwrap
    try:
        rows = await _screen(_http, scr, count)
    except Exception as e:  # noqa: BLE001
        _log.warning("movers %s failed: %s", scr, e)
        return []
    out: list[dict] = []
    for q in rows:
        sym = q.get("symbol")
        if not sym:
            continue
        out.append({
            "symbol": sym,
            "change_percent": round(_raw(q.get("regularMarketChangePercent")), 2),
            "price": round(_raw(q.get("regularMarketPrice")), 2),
        })
    return out[:count]


@app.get("/movers")
async def movers_endpoint(count: int = 6) -> dict:
    """Market-wide top movers on the day — Yahoo's `day_gainers` / `day_losers` predefined screeners.
    Feeds the app's market-close summary when it's set to 'whole market' instead of the watchlist.
    Best-effort: returns empty lists rather than erroring."""
    return {"gainers": await _movers_side("day_gainers", count), "losers": await _movers_side("day_losers", count)}


@app.get("/market_now")
async def market_now_endpoint(deep: bool = False, count: int = 6) -> dict:
    """AIE-5 — an instant AI overview of what US markets are doing RIGHT NOW. Composes a live snapshot
    (session phase, indices, VIX, sector rotation, market-wide + watchlist movers) and has the analyst
    narrate it. Cached ~3 min so repeated taps don't re-run the model. deep=true uses Opus for a richer
    read (slower); default is the fast scan model."""
    assert _http is not None
    cfg = settings_store.get()
    key = ("market_now", deep)
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < _MARKET_NOW_TTL:
        return {**hit[1], "cached": True}

    # Crypto trades round the clock; map watchlist crypto to Yahoo's <SYM>-USD so it resolves in the quote.
    watchlist = list(cfg.get("watchlist") or []) + [f"{c}-USD" for c in (cfg.get("crypto_watchlist") or [])]
    try:
        snap = await market_now.build_snapshot(_http, watchlist)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"market snapshot failed: {redact.redact(e)}")
    # Market-wide movers are a bonus — never let them fail the overview.
    try:
        snap["market_movers"] = {
            "gainers": await _movers_side("day_gainers", count),
            "losers": await _movers_side("day_losers", count),
        }
    except Exception as e:  # noqa: BLE001
        _log.warning("market_now movers failed: %s", e)

    try:
        ov, usage = await market_overview(snap, deep=deep)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"analyst failed: {redact.redact(e)}")
    usage_store.record(usage, symbol="", kind="market_now")

    # `overview` stays a (now grouped, multi-line) string so existing app builds render it prettier
    # immediately; `overview_struct` carries the tone/headline/points for the app's richer rendering.
    overview_str = ov.headline + "\n\n" + "\n".join(f"• {p}" for p in ov.points)
    payload = {
        "overview": overview_str,
        "overview_struct": ov.model_dump(),
        "snapshot": snap,
        "session": snap["session"],
        "model": usage["model"],
        "as_of": now,
        "usage": usage,
        "cached": False,
    }
    _cache[key] = (now, payload)
    return payload


@app.get("/memory/stats")
async def memory_stats() -> dict:
    """What long-term memory holds, and the service's own scorecard.

    Four blocks, each absent until there are enough graded rows to mean anything: `buy_calls` /
    `sell_calls` (what the analyst said) and `sandbox_buys` / `sandbox_sells` (what the paper trader
    actually did). `correct_rate_20d` means the same thing on all four — higher is better — because
    the sell side is inverted: a sell is RIGHT when the name underperforms the index, since owning
    the index was the alternative to holding it.
    """
    return memory.stats()


@app.post("/memory/backfill")
async def memory_backfill(every: int = 3, rng: str = "2y") -> dict:
    """Seed memory from real price history so setup base rates exist immediately.

    A verdict can't be graded until its 20-session horizon elapses, so without this the track record
    is empty for about a month after deployment. Replayed rows carry `origin="backfill"` and no
    signal — they inform what a PATTERN does, never what the model's calls achieved.
    """
    # Ranges beyond the retention window record rows that `prune` deletes on the very next scan, so
    # the endpoint would cheerfully report writing thousands of rows that were already doomed.
    allowed = {"6mo", "1y", "2y"}
    if rng not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f"rng must be one of {sorted(allowed)} — retention is "
                   f"{memory.retention_days()} days, so anything longer is pruned immediately",
        )
    cfg = settings_store.get()
    syms = list(cfg.get("watchlist", [])) + list(cfg.get("crypto_watchlist", []))
    if not syms:
        raise HTTPException(status_code=422, detail="no watchlist configured")
    return await scan_job.backfill_memory(syms, every=max(1, every), rng=rng)


@app.get("/memory/notes")
async def memory_notes(kind: str | None = None, limit: int = 10) -> dict:
    """Recent prose memory (weekly strategy, blocked trades, research findings), newest first.

    Also returns `blocked` — which risk rules actually bound over the last 30 days. That aggregate is
    the reason this log exists; free-text search over it never had a caller and was removed.
    """
    return {
        "notes": memory.recent_notes(kind=kind, limit=limit),
        "blocked": memory.blocked_summary(),
    }


@app.get("/regime")
async def regime_endpoint(deep: bool = False, count: int = 6) -> dict:
    """Theme D — the current market REGIME (structural backdrop, not the intraday tape): a short label,
    trend (up/down/sideways), volatility bucket, and a positioning note. Composes the market_now snapshot
    plus the S&P's 50/200-day structural trend and has the analyst classify it. Cached ~30 min."""
    assert _http is not None
    cfg = settings_store.get()
    key = ("regime", deep)
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < _DAILY_BRIEF_TTL:
        return {**hit[1], "cached": True}

    watchlist = list(cfg.get("watchlist") or []) + [f"{c}-USD" for c in (cfg.get("crypto_watchlist") or [])]
    try:
        snap = await market_now.build_snapshot(_http, watchlist)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"market snapshot failed: {redact.redact(e)}")

    # The S&P's structural trend — what makes this a "regime" read rather than a snapshot. 200-day SMA is
    # computed here (summarize only carries the 50-day).
    try:
        spy = await fetch_series(_http, "^GSPC")
        ssum = summarize(spy, None)
        closes = spy.closes
        sma200 = sum(closes[-200:]) / len(closes[-200:]) if len(closes) >= 200 else None
        price = ssum["price"]
        snap["spy_trend"] = {
            "price": round(price, 2),
            "pct_vs_sma50": ssum.get("pct_vs_sma50"),
            "pct_vs_sma200": round((price / sma200 - 1) * 100, 2) if sma200 else None,
            "above_200d": (price > sma200) if sma200 else None,
            "rsi14": ssum.get("rsi14"),
            "macd_hist": ssum.get("macd_hist"),
            "golden_cross": ssum.get("golden_cross"),
        }
    except Exception as e:  # noqa: BLE001 — the snapshot alone still yields a usable regime read
        _log.warning("regime spy_trend failed: %s", e)

    try:
        reg, usage = await market_regime(snap, deep=deep)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"analyst failed: {redact.redact(e)}")
    usage_store.record(usage, symbol="", kind="regime")

    payload = {
        "regime": reg.model_dump(),
        "spy_trend": snap.get("spy_trend"),
        "session": snap["session"],
        "model": usage["model"],
        "as_of": now,
        "usage": usage,
        "cached": False,
    }
    _cache[key] = (now, payload)
    return payload


@app.get("/daily_brief")
async def daily_brief_endpoint(deep: bool = False, count: int = 6) -> dict:
    """AIE-3 — a once-a-morning push brief. Same live snapshot as /market_now (session, indices, VIX,
    sectors, market + watchlist movers) PLUS `catalysts_today` (watchlist names reporting earnings today,
    in ET), narrated by the analyst into a notification title + a couple of sentences. Cached ~30 min;
    the app's worker fires it once per trading day, so this mostly just coalesces retries."""
    assert _http is not None
    cfg = settings_store.get()
    key = ("daily_brief", deep)
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < _DAILY_BRIEF_TTL:
        return {**hit[1], "cached": True}

    watchlist = list(cfg.get("watchlist") or []) + [f"{c}-USD" for c in (cfg.get("crypto_watchlist") or [])]
    try:
        snap = await market_now.build_snapshot(_http, watchlist)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"market snapshot failed: {redact.redact(e)}")
    try:
        snap["market_movers"] = {
            "gainers": await _movers_side("day_gainers", count),
            "losers": await _movers_side("day_losers", count),
        }
    except Exception as e:  # noqa: BLE001
        _log.warning("daily_brief movers failed: %s", e)

    # Today's catalysts: which of the user's (equity) names report earnings today, in ET.
    #
    # NEWS-7. This used to call news.fetch_context() once per equity, and that helper makes TWO
    # Finnhub requests of which this reads one — 100 requests inside a second for a 50-name
    # watchlist, against a 60-a-minute free tier. About forty of them were refused with HTTP 429
    # every trading day, and because the per-symbol failure was swallowed and returned None, a name
    # whose lookup was refused was indistinguishable from a name that is not reporting. The brief
    # then narrated "no catalysts today" over an answer drawn from roughly 40% of the watchlist.
    #
    # One market-wide call for today's date answers the whole list, and it either succeeds or says
    # it did not.
    from datetime import datetime
    from zoneinfo import ZoneInfo
    today_et = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    equities = [s for s in (cfg.get("watchlist") or []) if not s.upper().endswith("-USD")]
    try:
        reporting, catalysts_ok = await earnings_on(_http, today_et, equities, wait=15.0)
    except Exception as e:  # noqa: BLE001
        _log.warning("daily_brief catalysts failed: %s", redact.redact(e))
        reporting, catalysts_ok = set(), False
    snap["catalysts_today"] = sorted(reporting)
    # Carried into the snapshot the analyst is handed, and out in the payload, so neither the model
    # nor the client can read an empty list as a checked one.
    snap["catalysts_complete"] = catalysts_ok
    if not catalysts_ok:
        snap["catalysts_note"] = ("the earnings calendar could not be read this morning, so it is "
                                  "unknown whether any watchlist name reports today")

    try:
        brief, usage = await daily_brief(snap, deep=deep)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"analyst failed: {redact.redact(e)}")
    usage_store.record(usage, symbol="", kind="daily_brief")

    payload = {
        "title": brief.title,
        "body": brief.body,
        "tone": brief.tone,
        "catalysts_today": snap.get("catalysts_today", []),
        # False = the calendar could not be read; an empty catalysts_today is then UNKNOWN, not "none".
        "catalysts_complete": snap.get("catalysts_complete", False),
        "session": snap["session"],
        "model": usage["model"],
        "as_of": now,
        "usage": usage,
        "cached": False,
    }
    # A brief whose catalysts could not be read is held only briefly, the same way a failed news
    # lookup is (see _NEWS_FAIL_TTL_OFFSET). Freezing "we could not check" for the full half hour
    # would mean a single rate-limit blip decides the morning, and every retry inside that window
    # is served the same unknown without ever asking again.
    age = 0.0 if snap.get("catalysts_complete") else (_DAILY_BRIEF_TTL - _BRIEF_INCOMPLETE_TTL)
    _cache[key] = (now - age, payload)
    return payload


@app.get("/calendar")
async def calendar_endpoint(symbol: str | None = None) -> dict:
    """Catalyst calendar: SI settlements/publications, OPEX, earnings, clearly-labeled speculative
    T+35 FTD-echo windows, and the next BTC halving. Whole watchlist by default; `symbol` narrows to
    one asset (a crypto symbol gets only crypto-relevant events — equity SI/OPEX dates are noise
    there). Cached 1h."""
    cfg = settings_store.get()
    is_crypto_symbol = bool(symbol and symbol.upper().endswith("-USD"))
    syms = [] if is_crypto_symbol else ([symbol.upper()] if symbol else cfg.get("watchlist", []))
    # Symbol set is part of the cache identity so a just-synced add/remove refreshes immediately.
    key = ("calendar", symbol.upper() if symbol else None, tuple(sorted(syms)))
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < 3600:
        return {**hit[1], "cached": True}
    assert _http is not None
    # Finnhub next-earnings per symbol, fetched CONCURRENTLY (was a sequential await-loop → ~30s for a
    # full watchlist). NEWS-7: this asks for the earnings date ALONE rather than through
    # fetch_context(), which also fetched headlines this route discards — halving the request count
    # for a full watchlist from 100 to 50, under the 60/min free tier. news.RATE paces what remains,
    # so the semaphore here is only about not opening fifty sockets at once.
    #
    # Unlike the brief, this cannot use one market-wide call: the window is 90 days, and Finnhub
    # truncates that at 1500 rows without saying so (see news._CALENDAR_ROW_CAP).
    _earn_sem = asyncio.Semaphore(8)
    unchecked: list[str] = []

    async def _next_earnings(s: str) -> tuple[str, str | None]:
        async with _earn_sem:
            try:
                got = await fetch_next_earnings(_http, s, wait=15.0)
            except Exception:  # noqa: BLE001
                unchecked.append(s.upper())
                return s, None
            if not got.ok:
                # A refused lookup drops this symbol's earnings event silently otherwise, which reads
                # on the calendar exactly like a company with nothing scheduled.
                unchecked.append(s.upper())
            return s, got.date

    earnings: dict[str, str] = {
        s: e for s, e in await asyncio.gather(*[_next_earnings(s) for s in syms]) if e
    }
    events = await shorts.calendar(_http, syms, earnings) if syms else []
    # Next Bitcoin halving (estimated): on the watchlist-wide view with BTC exposure, and on any
    # BTC-* symbol's own calendar.
    show_halving = (
        (symbol is None and any("BTC" in c.upper() for c in cfg.get("crypto_watchlist", [])))
        or (symbol is not None and "BTC" in symbol.upper())
    )
    if show_halving:
        events.append({
            "date": cycle.NEXT_HALVING_EST.isoformat(), "symbol": "BTC-USD",
            "label": "Bitcoin halving (~estimated from block schedule)", "kind": "btc_halving",
        })
        events.sort(key=lambda x: x["date"])
    # CAL-1. The row cap lives here, applied AFTER the halving joins the list, and it reports itself.
    # It used to sit inside shorts.calendar() at `events[:30]`, where nothing downstream could tell a
    # short calendar from a truncated one — and the halving was appended after the slice, so the
    # payload quietly carried 31 of a capped 30.
    total = len(events)
    shown = events[:_CALENDAR_MAX_EVENTS]
    payload = {"as_of": now, "symbol": symbol.upper() if symbol else None, "events": shown,
               # Present whether or not anything was cut, so a client never has to infer completeness
               # from a length it would have to know the cap to interpret.
               "events_total": total,
               "truncated_after": shown[-1]["date"] if total > len(shown) else None,
               # Symbols whose earnings lookup FAILED. Their events are missing, not absent.
               "earnings_unchecked": sorted(set(unchecked)), "cached": False}
    _cache[key] = (now, payload)
    return payload


@app.get("/cycle/{symbol}")
async def cycle_endpoint(symbol: str) -> dict:
    """Crypto long-term context for the app's cycle card: halving-cycle position (BTC), multi-year
    trend metrics, and past halving dates for chart markers. Free — no LLM."""
    assert _http is not None
    try:
        series = await fetch_series(_http, symbol)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"data fetch failed: {redact.redact(e)}")
    ctx = await cycle.crypto_context(_http, series.symbol, series.closes)
    if not ctx:
        raise HTTPException(status_code=404, detail="no long-term data for this symbol")
    return {
        "symbol": series.symbol,
        **ctx,
        "halving_dates": [h.isoformat() for h in cycle.HALVINGS],
        "next_halving_est": cycle.NEXT_HALVING_EST.isoformat(),
    }


@app.get("/trend/{symbol}")
async def trend(symbol: str) -> dict:
    """Below-the-200-week-line context for a STOCK (or any symbol) — the equity mirror of /cycle.
    200-week SMA, where price sits vs the line (below_line, 7-band zone, week-over-week
    recovering/deepening direction), a 14-week RSI oversold read, Mayer, ATH distance, 3y CAGR.
    Free — no LLM. 404 for names with under ~4 years of weekly history (no 200-week value)."""
    assert _http is not None
    try:
        series = await fetch_series(_http, symbol)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"data fetch failed: {redact.redact(e)}")
    ctx = await cycle.crypto_context(_http, series.symbol, series.closes)
    lt = ctx.get("long_term_trend")
    if not lt or "sma_200w" not in lt:
        raise HTTPException(status_code=404, detail="not enough weekly history for a 200-week trend")
    return {"symbol": series.symbol, "close": round(series.closes[-1], 4), **lt}


@app.post("/universe/build")
async def universe_build_endpoint() -> dict:
    """MB-19 — (re)build the curated ticker universe from the Nasdaq Trader symbol directory plus
    batched Yahoo market caps, and persist it. A few hundred HTTP calls, so this is deliberately an
    explicit POST rather than something a read path can trigger."""
    assert _http is not None
    try:
        blob = await universe.build(_http)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"universe build failed: {redact.redact(e)}")
    # Same decision as the nightly hook, made in one place so the two cannot drift apart — the
    # coverage guard used to live here only, leaving the unattended path unprotected.
    ok, why = universe.publish(blob, previous=universe.load())
    if not ok:
        raise HTTPException(status_code=502, detail=why)
    universe.save(blob)
    return {k: v for k, v in blob.items() if k != "detail"}


@app.get("/universe")
async def universe_endpoint(limit: int = 25) -> dict:
    """What the curated universe currently holds, and whether it needs rebuilding."""
    blob = universe.load()
    if not blob:
        raise HTTPException(status_code=404, detail="universe not built yet — POST /universe/build")
    return {
        "built_at": blob.get("built_at"),
        "stale": universe.is_stale(blob),
        "source": blob.get("source"),
        "directory_rows": blob.get("directory_rows"),
        "passed_filter": blob.get("passed_filter"),
        "size": len(blob.get("symbols") or []),
        "sample": (blob.get("detail") or [])[:max(0, min(100, limit))],
    }


@app.get("/sectors")
async def sectors_endpoint(symbols: str = "") -> dict:
    """{SYMBOL: {sector, industry}} for the requested tickers. Free — NO LLM.

    The same disk-cached classification the heat map groups by, exposed so the watchlist can section
    itself the same way. A company's sector changes approximately never, so this is a write-once cost
    per symbol and almost always a pure cache read.

    A symbol that cannot be classified comes back present with a null sector rather than missing from
    the map. The two mean different things to a caller — "we looked and it is unclassified" versus
    "we never looked" — and only the first one is safe to render as "Other".
    """
    assert _http is not None
    want = [s.strip().upper() for s in symbols.split(",") if s.strip()][:200]
    if not want:
        return {"sectors": {}}
    try:
        found = await sectors.lookup(_http, want)
    except Exception as e:  # noqa: BLE001
        # 502 rather than an empty map. An empty map is indistinguishable from "none of these could
        # be classified", which would file the caller's entire watchlist under Other and look like an
        # answer. A failure has to read as a failure.
        raise HTTPException(status_code=502, detail=f"sector lookup failed: {redact.redact(e)}")
    return {"sectors": {s: {"sector": (found.get(s) or {}).get("sector"),
                            "industry": (found.get(s) or {}).get("industry")} for s in want}}


@app.get("/heatmap")
async def heatmap_endpoint(mode: str = "market", limit: int = 80, refresh: bool = False) -> dict:
    """Tile data for the market heat map. Free — NO LLM.

    `mode=market` sizes by market cap and colours by today's move; `mode=signals` sizes by distance
    below the 52-week high and colours by the dip tier the nightly scan assigned. The two use
    DIFFERENT colour scales and each tile says which, so a dip tier can never render as a green day.

    No rectangles are computed here — the server does not know the viewport. The squarified layout
    runs on the device against the real width.
    """
    assert _http is not None
    if mode not in ("market", "signals"):
        raise HTTPException(status_code=422, detail="mode must be 'market' or 'signals'")
    limit = max(1, min(200, limit))
    cfg = settings_store.get()
    now = time.time()

    if mode == "signals":
        # Straight off the nightly scan — no fetching, so no cache needed.
        scan = json.loads(LATEST.read_text()) if LATEST.exists() else None
        tiles, skipped = heatmap.signal_tiles(scan, limit=limit)
        return {
            "mode": "signals", "as_of": (scan or {}).get("generated_at"), "tiles": tiles,
            "skipped": skipped, "scale": "signal",
            "note": ("Sized by how far below its 52-week high, coloured by this system's own dip "
                     "tier — not by price. Context, not a buy signal."),
            "cached": False,
        }

    key = ("heatmap_market", limit)
    hit = _cache.get(key)
    # Quotes move, so this TTL is short on purpose — a heat map showing a stale move is worse than
    # one that takes a second to load.
    if hit and not refresh and now - hit[0] < 60:
        return {**hit[1], "cached": True, "cached_age_seconds": int(now - hit[0])}

    uni = universe.load()
    if not uni or not uni.get("detail"):
        raise HTTPException(status_code=404, detail="universe not built yet — POST /universe/build")
    syms = [r["symbol"] for r in uni["detail"][:limit]]
    try:
        quotes = await market_now.fetch_quotes(_http, syms)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"quote fetch failed: {redact.redact(e)}")
    phase = None
    try:
        phase = market_now.session_phase()
    except Exception:  # noqa: BLE001 — no phase just means we fall back to the regular-session move
        phase = None
    # Sector per symbol, so the map can be drawn as labelled blocks instead of one flat wall of
    # rectangles. Disk-cached with a 90-day TTL — a company's sector changes approximately never, so
    # this is a write-once cost per symbol. Best-effort: an unclassified name still gets a tile.
    try:
        profiles = await sectors.lookup(
            _http, [r["symbol"] for r in ((uni or {}).get("detail") or [])[:limit]])
    except Exception:  # noqa: BLE001 — the map is more useful ungrouped than absent
        _log.warning("heatmap: sector lookup failed, drawing ungrouped", exc_info=True)
        profiles = {}
    tiles, unpriced = heatmap.market_tiles(
        uni, quotes, limit=limit, phase=phase, sectors=profiles)
    if not tiles:
        raise HTTPException(status_code=502, detail="no tiles could be priced")

    up = sum(1 for t in tiles if t["value"] > 0)
    payload = {
        "mode": "market", "as_of": now, "tiles": tiles,
        # Named, never silently dropped: unpriceable is a fact about our fetch, not about the stock.
        "unpriced": unpriced,
        "advancing": up, "declining": len(tiles) - up,
        "universe_built_at": uni.get("built_at"),
        "universe_stale": universe.is_stale(uni),
        "scale": "price",
        "session": phase,
        "note": ("Sized by market cap, coloured by the move for the session in progress"
                 + (f" ({phase})." if phase else ".")),
        "cached": False,
    }
    _cache[key] = (now, payload)
    return payload


@app.get("/screener/value")
async def value_screener(limit: int = 20, below_line_only: bool = True,
                         include_watchlist: bool = True, refresh: bool = False) -> dict:
    """MB-15/MB-18 — rank a value-tilted universe by how far below its own 200-week trend each name
    sits. Free: NO LLM, one weekly-bars call per symbol, price-based only.

    CONTEXT, NOT A BUY SIGNAL. The touch study on this codebase found below-the-line dips
    UNDERPERFORMED the S&P on 12/24-month forward horizons, so this answers "what is unusually
    dislocated", not "what should I buy". The score is deliberately separate from the 0-100 momentum
    score; do not add them together.
    """
    assert _http is not None
    cfg = settings_store.get()
    limit = max(1, min(50, limit))
    # The key must capture EVERY input that changes the answer. It held only the flags, so adding a
    # watchlist name or rebuilding the universe served the previous ranking as current. The universe
    # is identified by its build timestamp — cheap, and it changes exactly when the pool does.
    _u = universe.load()
    key = ("value_screen", limit, below_line_only, include_watchlist,
           tuple(sorted(settings_store.get().get("watchlist") or [])) if include_watchlist else (),
           (_u or {}).get("built_at"))
    now = time.time()
    hit = _cache.get(key)
    # Weekly bars change weekly; a long TTL is correct here and keeps the fan-out cheap.
    if hit and not refresh and now - hit[0] < max(cfg["verdict_ttl_seconds"], 3600):
        return {**hit[1], "cached": True, "cached_age_seconds": int(now - hit[0])}

    extra = sorted(settings_store.get().get("watchlist") or []) if include_watchlist else []
    # Prefer the CURATED universe (MB-19). The live Yahoo screens are rebuilt server-side on every
    # call, so two runs minutes apart returned different names off a ~58-wide pool — a sampler, not
    # a screen. Fall back to them only when the curated list has not been built yet.
    cur = _u
    universe_source = "curated"
    if cur and cur.get("symbols"):
        pool = list(cur["symbols"])[:max(limit * 10, 150)]
        syms = pool + [s for s in extra if s not in pool]
    else:
        universe_source = "yahoo_screens"
        try:
            syms = await screener.build_universe(_http, extra=extra)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"universe build failed: {redact.redact(e)}")
    if not syms:
        raise HTTPException(status_code=502, detail="no candidates could be screened")

    async def trend_of(client, sym: str):
        series = await fetch_series(client, sym)
        ctx = await cycle.crypto_context(client, series.symbol, series.closes)
        return ctx.get("long_term_trend")

    ranked, too_short, failed = await screener.screen(
        _http, trend_of, syms, limit=limit, below_line_only=below_line_only)
    payload = {
        "as_of": now,
        "universe_size": len(syms),
        "universe_source": universe_source,
        "universe_stale": bool(cur and universe.is_stale(cur)),
        "scored": len(syms) - len(too_short) - len(failed),
        # Named, not silently dropped — and split by CAUSE. "Too short" is a fact about the company;
        # "failed" is a fact about our fetch, and telling the user the first when the second happened
        # is a confident statement about the wrong thing.
        "skipped": too_short,
        "fetch_failed": failed,
        "below_line_only": below_line_only,
        "results": ranked,
        "note": ("How dislocated a name is below its own 200-week trend — context, not a buy signal. "
                 "The historical touch study on this codebase found below-the-line dips "
                 "underperformed the S&P over the following 12-24 months."),
        "cached": False,
    }
    _cache[key] = (now, payload)
    return payload


@app.get("/touches/{symbol}")
async def touches(symbol: str) -> dict:
    """Historical 200-week-line touch study: what happened the last N times this name traded below
    its 200-week line — median/avg forward 12- and 24-month return, % that resolved higher, and the
    S&P 500's average over the same windows. Evidence context, not a buy signal. Free — no LLM.
    404 for names with under ~4 years of weekly history."""
    assert _http is not None
    try:
        dates, weekly, _ = await cycle._weekly_max(_http, symbol.upper())
        spy_dates, spy_weekly = await cycle.spy_weekly(_http)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"data fetch failed: {redact.redact(e)}")
    study = cycle.wma_touch_study(dates, weekly, spy_dates, spy_weekly)
    if study is None:
        raise HTTPException(status_code=404, detail="not enough weekly history for a 200-week touch study")
    return {"symbol": symbol.upper(), **study}


@app.get("/insider/{symbol}")
async def insider_endpoint(symbol: str) -> dict:
    """Open-market insider PURCHASES (SEC Form 4 via Finnhub) over the last 12 months — the bullish
    informed-money read. Free; needs a Finnhub key configured. 404 without a key."""
    assert _http is not None
    data = await insider.insider_buying(_http, symbol.upper())
    if data is None:
        raise HTTPException(status_code=404, detail="no insider data (set a Finnhub key in settings)")
    return {"symbol": symbol.upper(), **data}


@app.get("/smart_money")
async def smart_money_endpoint(limit: int = 20, months: int = 12, refresh: bool = False) -> dict:
    """Theme C — where informed money has been BUYING across the watchlist, in one view.

    Composes the two existing per-symbol feeds (SEC Form 4 insider purchases, congressional STOCK Act
    disclosures) into a cross-sectional ranking. Free — NO LLM.

    Weighting is deliberate: an insider buy outranks a congressional one, because insiders disclose
    within two business days and spend their own money, while congressional filings lag up to ~45
    days and report AMOUNT RANGES rather than figures. Insider SALES are not scored at all — people
    sell for tax, diversification and houses.

    Corroborating context, never a reason on its own. A feed that could not be read is reported
    separately from one that came back empty.
    """
    assert _http is not None
    cfg = settings_store.get()
    limit = max(1, min(50, limit))
    watch = sorted({s.upper() for s in (cfg.get("watchlist") or [])})
    if not watch:
        raise HTTPException(status_code=422, detail="no watchlist symbols configured")

    key = ("smart_money", limit, months, tuple(watch))
    now = time.time()
    hit = _cache.get(key)
    if hit and not refresh and now - hit[0] < max(cfg["verdict_ttl_seconds"], 3600):
        return {**hit[1], "cached": True, "cached_age_seconds": int(now - hit[0])}

    async def insider_of(sym: str):
        return await insider.insider_buying(_http, sym)

    async def congress_of(sym: str):
        return await congress.congress_trades(_http, sym, months=months)

    ranked, no_evidence, failed = await smartmoney.sweep(
        watch, insider_of, congress_of, limit=limit)

    # insider_buying returns None (not raises) when no Finnhub key is set, which would otherwise
    # render as "nobody is buying" across the entire watchlist.
    key_set = bool(cfg.get("finnhub_api_key"))
    payload = {
        "as_of": now,
        "watchlist_size": len(watch),
        "results": ranked,
        # Names we looked at and found nothing for — distinct from ones we could not look at.
        "no_evidence": no_evidence,
        "fetch_failed": failed,
        "insider_feed_configured": key_set,
        "note": ("Where insiders and members of Congress have been buying names you follow. "
                 "Insider buys are disclosed within two business days; congressional filings lag up "
                 "to ~45 days and report amount ranges. Corroborating context, not a buy signal."),
        "cached": False,
    }
    if not key_set:
        payload["warning"] = ("No Finnhub key configured, so insider filings were not consulted — "
                              "this ranking is congressional disclosures only.")
    _cache[key] = (now, payload)
    return payload


@app.get("/congress/{symbol}")
async def congress_endpoint(symbol: str, months: int = 12) -> dict:
    """Congressional / political trades in a stock over the last `months` — House + Senate + cabinet
    disclosures (free kadoa dataset). Free, no LLM. Lagging (~45-day STOCK Act filing window). Returns
    {symbol, congress: {...}|null} — null when nobody disclosed a trade in the name."""
    assert _http is not None
    data = await congress.congress_trades(_http, symbol.upper(), months=months)
    return {"symbol": symbol.upper(), "congress": data}


@app.get("/seasonality/{symbol}")
async def seasonality_endpoint(symbol: str) -> dict:
    """Typical per-calendar-month price action from ~10y of monthly bars: avg return + hit rate per
    month, the current month's tendency, and the strongest/weakest months. Free, no LLM. WEAK, sample-
    limited context. Returns {symbol, seasonality: {...}|null} (null for names with under ~2y history)."""
    assert _http is not None
    data = await seasonality.seasonality(_http, symbol.upper())
    return {"symbol": symbol.upper(), "seasonality": data}


@app.get("/news_moves/{symbol}")
async def news_moves_endpoint(symbol: str, deep: bool = False, refresh: bool = False) -> dict:
    """AIE-4 — why the stock moved. Finds its notable daily moves over ~3 weeks, pulls dated company
    news, and has the analyst correlate them: which move was news-driven and which happened on flows/
    technicals. Equities only (Finnhub news is equities); crypto returns a friendly note. Cached ~1h
    (`refresh=true` bypasses it). Returns {symbol, news_moves: {summary, drivers[]}|null, note?}."""
    assert _http is not None
    sym = symbol.upper()
    if sym.endswith("-USD"):
        return {"symbol": sym, "news_moves": None, "note": "News correlation isn't available for crypto."}

    now = time.time()
    key = ("news_moves", sym, deep)
    hit = _cache.get(key)
    if hit and not refresh and now - hit[0] < 3600:
        return {**hit[1], "cached": True}

    try:
        series = await fetch_series(_http, sym)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"price fetch failed: {redact.redact(e)}")

    # Notable daily moves over the last ~15 trading days: |move| >= 3%, keep the 4 most extreme, newest
    # first. A YYYYMMDD date → YYYY-MM-DD so it lines up with the news dates.
    closes, dates = series.closes[-16:], series.dates[-16:]
    moves: list[dict] = []
    for i in range(1, len(closes)):
        prev, cur = closes[i - 1], closes[i]
        if not prev:
            continue
        pct = (cur - prev) / prev * 100.0
        if abs(pct) >= 3.0:
            d = dates[i]
            iso = f"{d[0:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 else d
            moves.append({"date": iso, "move_pct": round(pct, 2)})
    moves = sorted(moves, key=lambda m: abs(m["move_pct"]), reverse=True)[:4]
    moves.sort(key=lambda m: m["date"], reverse=True)

    if not moves:
        payload = {"symbol": sym, "news_moves": {
            "summary": "No outsized daily moves in the last few weeks — the stock's been calm.",
            "drivers": [],
        }, "model": "", "as_of": now, "cached": False}
        _cache[key] = (now, payload)
        return payload

    news = await fetch_dated_news(_http, sym)
    if news is None:
        # The LOOKUP FAILED — we do not know whether there was news. Say exactly that, and do NOT
        # cache it for the usual hour: a transient Finnhub error (a rate limit from the ~10-call
        # burst the app fires when a symbol is opened, or indexing lag) would otherwise be frozen
        # into a confident "no news" for the next 60 minutes. Measured on GME 2026-08-03: it fell
        # 10.4% on a $1.4B convertible-notes dilution, Finnhub had 26 articles including "Why
        # GameStop Stock Just Slumped", and the app said there were none.
        payload = {"symbol": sym, "news_moves": {
            "summary": "Couldn't load the news for these moves — this is a data-source problem, "
                       "not a sign that nothing happened.",
            "drivers": [{"date": m["date"], "move_pct": m["move_pct"], "headline": None,
                         "explanation": "News lookup failed — unknown, not absent."} for m in moves],
        }, "model": "", "as_of": now, "cached": False, "news_unavailable": True}
        _cache[key] = (now - _NEWS_FAIL_TTL_OFFSET, payload)  # expires in ~60s, not an hour
        return payload
    if not news:
        # Fetch SUCCEEDED and the symbol genuinely has no coverage in the window. This one is a real
        # claim, so it is safe to make and safe to cache.
        payload = {"symbol": sym, "news_moves": {
            "summary": "Notable moves, but no news coverage was available to correlate them.",
            "drivers": [{"date": m["date"], "move_pct": m["move_pct"], "headline": None,
                         "explanation": "No headlines available for this day."} for m in moves],
        }, "model": "", "as_of": now, "cached": False}
        _cache[key] = (now, payload)
        return payload

    try:
        nm, usage = await news_moves(sym, moves, news, deep=deep)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"analyst failed: {redact.redact(e)}")
    usage_store.record(usage, symbol=sym, kind="news_moves")

    payload = {"symbol": sym, "news_moves": nm.model_dump(), "model": usage["model"],
               "as_of": now, "usage": usage, "cached": False}
    _cache[key] = (now, payload)
    return payload


@app.get("/quality/{symbol}")
async def quality_endpoint(symbol: str) -> dict:
    """Quality tags (Finnhub basic-financials): ROE / margins / debt-to-equity + buffett_quality,
    wide_moat, dividend_aristocrat flags. Stance-neutral descriptors. Free (the aristocrat flag works
    without a key). 404 when nothing is available."""
    assert _http is not None
    sym = symbol.upper()
    data = await fundamentals.fetch_quality(_http, sym)
    # FCF-trend + share-count-trend (MB-13/14) from Finnhub's as-reported SEC financials — best-effort.
    funda = None
    try:
        funda = await fundamentals.fetch_financials(_http, sym)
    except Exception:  # noqa: BLE001
        funda = None
    if data is None and funda is None:
        raise HTTPException(status_code=404, detail="no quality data for this symbol")
    return {"symbol": sym, **(data or {}), **(funda or {})}


@app.get("/valuetrap/{symbol}")
async def valuetrap_endpoint(symbol: str) -> dict:
    """MB-17 — is this cheap name a DISCOUNT or is it DETERIORATING?

    The 200-week screen says what is unusually cheap versus its own trend; this says whether the
    business behind it is still intact. Composes three feeds already fetched elsewhere (quality/FCF,
    insider buying, trend direction) and reasons over them. Free — NO LLM.

    Evidence, not a recommendation. An "unclear" verdict with a non-empty `missing` means the data
    was unavailable, NOT that the business looks fine — the two are reported separately via
    `assessable` so a caller cannot confuse them.
    """
    assert _http is not None
    sym = symbol.upper()

    # Each feed is best-effort and independently optional: a missing one becomes an entry in
    # `missing`, which is materially different from it having come back clean.
    q = None
    try:
        base = await fundamentals.fetch_quality(_http, sym)
        fin = await fundamentals.fetch_financials(_http, sym)
        if base or fin:
            q = {**(base or {}), **(fin or {})}
    except Exception:  # noqa: BLE001
        q = None
    try:
        ins = await insider.insider_buying(_http, sym)
    except Exception:  # noqa: BLE001
        ins = None
    lt = None
    try:
        series = await fetch_series(_http, sym)
        ctx = await cycle.crypto_context(_http, series.symbol, series.closes)
        lt = ctx.get("long_term_trend")
    except Exception:  # noqa: BLE001
        lt = None

    out = valuetrap.assess(lt, q, ins)
    return {
        "symbol": sym,
        "price_vs_200w_sma_pct": (lt or {}).get("price_vs_200w_sma_pct"),
        "below_line": (lt or {}).get("below_line"),
        **out,
    }


@app.get("/options/{symbol}")
async def options_endpoint(
    symbol: str,
    budget: float | None = None,
    style: str = "balanced",
    target_date: str | None = None,
    crypto: bool = False,
    deep: bool = False,
) -> dict:
    """No-LLM long-CALL suggester (OC-1): a go/no-go light + up to 3 delta-picked call contracts, each
    with cost/max-loss/breakeven/greeks and a copy-pasteable order ticket. Pure math + the existing
    directional technicals. `budget` sizes the contract count (max loss you'll accept); `style` is
    safer|balanced|cheaper (the delta bucket surfaced first); `target_date` (YYYY-MM-DD) forces an
    expiry at/after your timeframe. Additive fields: `iv_rank` (from nightly ATM-IV logging, null while
    building), a nullable `alternative` debit-call-spread block + `recommend_alternative` bool (OC-6),
    and — when `deep=true` — a one-paragraph Opus `analyst` explanation (OC-7; null by default / on any
    failure, never a 500). Options aren't available for crypto — a 400, not a 500. Not investment advice."""
    # 0) Validate budget FIRST — before any network call — so inf/nan/negatives can't reach
    #    math.floor() and blow up as a 500. A non-positive or non-finite budget is a client error.
    if budget is not None and (not math.isfinite(budget) or budget <= 0):
        raise HTTPException(status_code=422, detail="budget must be a positive number")
    assert _http is not None
    if crypto or symbol.upper().endswith("-USD"):
        raise HTTPException(status_code=400, detail="options aren't available for crypto symbols")

    # 1) The chain (spot + all expirations + the default expiry's contracts).
    try:
        chain = await options.fetch_chain(_http, symbol)
    except Exception as e:  # noqa: BLE001 — no chain (crypto/ETN/illiquid/unknown) is a 400, never a 500
        # Keep the client-facing detail generic: httpx errors embed the request URL (which carries
        # the Yahoo crumb) — never leak that. Log the real exception server-side.
        _log.warning("options fetch_chain failed for %s: %s", symbol.upper(), e)
        raise HTTPException(status_code=400, detail=f"options aren't available for {symbol.upper()}")
    if not chain.expirations or not chain.spot or chain.spot <= 0:
        raise HTTPException(status_code=400, detail=f"no option chain available for {symbol.upper()}")

    # 2) Directional read + next earnings — REUSE market.summarize (the ChartMath the Signals card /
    #    /signal use) and news.fetch_context (Finnhub earnings). Best-effort: if the history fetch
    #    fails we still return contracts, but the light defaults to caution.
    summary: dict = {}
    earnings_date: str | None = None
    warnings: list[str] = []
    try:
        series = await fetch_series(_http, symbol)
        try:
            bench = (await fetch_series(_http, "^GSPC")).closes
        except Exception:  # noqa: BLE001 — relative strength just gets skipped
            bench = None
        summary = summarize(series, bench)
        earnings_date = (await fetch_context(_http, series.symbol)).get("next_earnings")
    except Exception:  # noqa: BLE001
        warnings.append("directional data unavailable — the go/no-go light defaults to caution")

    # 3) Pick the expiry (45-90 DTE, clears target_date, skips earnings straddles, ~60 DTE).
    try:
        chosen, exp_warnings = options.select_expiry(
            chain, target_date=target_date, earnings_date=earnings_date,
        )
    except Exception as e:  # noqa: BLE001 — malformed expiration data degrades to 400, not 500
        _log.warning("options select_expiry failed for %s: %s", chain.symbol, e)
        raise HTTPException(status_code=400, detail=f"options aren't available for {symbol.upper()}")
    if chosen is None:
        raise HTTPException(status_code=400, detail=f"no suitable expiration for {chain.symbol}")
    warnings.extend(exp_warnings)

    # 4) Load the chosen expiry's contracts (reuse the first fetch if it already loaded it) + annotate.
    if not (chain.expiry and chain.expiry.expiration == chosen["ts"]):
        try:
            chain = await options.fetch_chain(_http, symbol, chosen["ts"])
        except Exception as e:  # noqa: BLE001 — generic 400 (the httpx error embeds the crumb'd URL)
            _log.warning("options expiry reload failed for %s: %s", chain.symbol, e)
            raise HTTPException(status_code=400, detail=f"options aren't available for {symbol.upper()}")
    if chain.expiry is None:
        raise HTTPException(status_code=400, detail=f"no contracts for the chosen expiry of {chain.symbol}")

    # 5) Annotate + assemble. Malformed chain data here degrades to 400 (matching the fetch above),
    #    never a 500. IV rank (OC-6a) is computed from data/iv_history.jsonl and passed in — the
    #    assembler stays pure; a missing/short history yields a null rank (noted as "building").
    try:
        options.annotate_expiry(chain)
        # Only rank when TODAY's ATM IV is actually known. iv_rank defaults `current` to the last
        # historical point, so passing None silently ranked a past reading and labelled it today's.
        iv_rank = None
        if chain.expiry.atm_iv:
            iv_rank = options.iv_rank(
                options.load_iv_history(chain.symbol), current=chain.expiry.atm_iv,
                current_dte=int(chosen["dte"]), history_dte=options.load_iv_history_dte(chain.symbol),
            )
        body = options.assemble_suggestion(
            chain, chain.expiry, summary,
            chosen=chosen, style=style, budget=budget, earnings_date=earnings_date,
            extra_warnings=warnings, iv_rank=iv_rank,
        )
    except Exception as e:  # noqa: BLE001
        _log.warning("options assembly failed for %s: %s", chain.symbol, e)
        raise HTTPException(status_code=400, detail=f"options aren't available for {symbol.upper()}")

    # 6) OC-7 — optional Opus analyst paragraph. deep=false leaves `analyst` null with no LLM call
    #    (the free path is unchanged). On any failure the paragraph stays null — never a 500.
    if deep:
        try:
            ctx = {
                "symbol": body["symbol"], "spot": body["spot"],
                "light": body["light"], "light_reason": body["light_reason"],
                "expiry": body["expiry"], "structure": body["structure"],
                "suggested_contract": body["candidates"][0] if body["candidates"] else None,
                "expected_move": body["expected_move"], "iv_rank": body["iv_rank"],
                "earnings": body["earnings"], "alternative": body["alternative"],
                "recommend_alternative": body["recommend_alternative"],
                "directional": options.directional_read(summary) if summary else None,
            }
            paragraph, usage = await options_note(ctx, deep=True)
            body["analyst"] = paragraph
            usage_store.record(usage, symbol=body["symbol"], kind="options")
        except Exception as e:  # noqa: BLE001 — the paragraph is enrichment; never 500
            _log.warning("options analyst paragraph failed for %s: %s", chain.symbol, e)
    return body


@app.get("/option_quote/{symbol}")
async def option_quote(
    symbol: str,
    expiry_ts: int,
    strike: float,
    type: str = "call",
) -> dict:
    """Re-price ONE specific option contract (OC-3 call-position tracker): given the exact expiry +
    strike the user already bought, return its live bid/ask/mid/limit + greeks so the app can show
    running P/L. No LLM, pure data. `expiry_ts` is a unix ts from an earlier `/options` (or chain)
    call; `strike` is the contract's strike; `type` is call|put (default call). Options aren't
    available for crypto — that's a 400, not a 500. A strike that isn't in the chain is a 404.
    Decision support only, not investment advice."""
    assert _http is not None
    if symbol.upper().endswith("-USD"):
        raise HTTPException(status_code=400, detail="options aren't available for crypto symbols")
    kind = "put" if str(type).lower() == "put" else "call"
    now = time.time()

    # 1) The chain for this exact expiry (spot + the expiry's calls/puts).
    try:
        chain = await options.fetch_chain(_http, symbol, expiry_ts)
    except Exception as e:  # noqa: BLE001 — no chain (crypto/ETN/illiquid/unknown) is a 400, never a 500.
        # httpx errors embed the request URL (which carries the Yahoo crumb) — never leak that; keep
        # the client detail generic and log the real exception server-side.
        _log.warning("option_quote fetch_chain failed for %s: %s", symbol.upper(), e)
        raise HTTPException(status_code=400, detail=f"options aren't available for {symbol.upper()}")
    if chain.expiry is None or not chain.spot or chain.spot <= 0:
        raise HTTPException(status_code=400, detail=f"no option chain available for {symbol.upper()}")

    # 2) Annotate (mid / spread% / greeks) — malformed chain data degrades to 400, never a 500.
    try:
        options.annotate_expiry(chain, now_ts=now)
    except Exception as e:  # noqa: BLE001
        _log.warning("option_quote annotate failed for %s: %s", chain.symbol, e)
        raise HTTPException(status_code=400, detail=f"options aren't available for {symbol.upper()}")

    # 3) Find the one contract at this strike (small tolerance for float noise). Missing -> 404.
    pool = chain.expiry.puts if kind == "put" else chain.expiry.calls
    match = next((c for c in pool if abs(strike - c.strike) < 0.01), None)
    if match is None:
        raise HTTPException(
            status_code=404,
            detail=f"no {kind} at strike {strike} for {chain.symbol} {chain.expiry.expiration_iso}",
        )

    _m = lambda v: round(v, 2) if v is not None else None  # noqa: E731 — money to 2dp, nullable
    _mid_val = options.mid_price(match.bid, match.ask)
    _limit_val, _limit_src = options._limit_price_with_source(match)
    return {
        "symbol": chain.symbol,
        "spot": _m(chain.spot),
        "as_of": now,
        "quote_delayed": chain.quote_delayed,
        "dte": chain.expiry.dte_days,
        "contract": {
            "contract_symbol": match.contract_symbol,
            "type": match.type,
            "strike": match.strike,
            "expiration": match.expiration,
            "bid": _m(match.bid),
            "ask": _m(match.ask),
            "last_price": _m(match.last_price),
            # limit_price MUST equal mid when the quote rests on the mid. They were computed by two
            # different routes — mid fresh from bid/ask, limit_price via match.mid which is already
            # _round(.., 4) — so the second was DOUBLE-rounded and the two could render a cent apart
            # on the same row of an options ticket. Derive both from one value.
            "mid": _m(_mid_val),
            "limit_price": _m(_mid_val) if _limit_src == "mid" else _m(_limit_val),
            "implied_volatility": round(match.implied_volatility, 4) if match.implied_volatility is not None else None,
            "delta": match.delta,   # already 4dp from annotate_expiry
            "theta": match.theta,   # already 4dp from annotate_expiry
            "open_interest": match.open_interest,
            "in_the_money": match.in_the_money,
            "spread_pct": match.spread_pct,
        },
    }


@app.get("/puts/{symbol}")
async def puts_endpoint(
    symbol: str,
    cash: float,
    style: str = "balanced",
    crypto: bool = False,
) -> dict:
    """No-LLM cash-secured-PUT suggester (OC-8, the wheel's entry leg): sell a put to acquire shares
    BELOW today's price / get paid to wait. Returns up to 3 delta-picked put strikes (aggressive ~0.45
    |Δ| near-money → high assignment chance, balanced ~0.30, conservative ~0.20 deep-OTM), each with
    net cost/share, discount vs. spot, cash-to-reserve, static + annualized yield, assignment
    probability and a copy-pasteable order ticket. `cash` (required) is the reserve you can set aside
    and sizes the contract count; `style` is surfaced first. Short-dated (~25-50 DTE, target ~35) —
    theta favours the seller. Options aren't available for crypto — a 400, not a 500. Only sell puts
    on names you'd happily own at the strike. Decision support only, not investment advice."""
    # 0) Validate cash FIRST — before any network call — so inf/nan/negatives can't reach math.floor()
    #    and blow up as a 500. A non-positive or non-finite reserve is a client error.
    if not math.isfinite(cash) or cash <= 0:
        raise HTTPException(status_code=422, detail="cash must be a positive number")
    assert _http is not None
    if crypto or symbol.upper().endswith("-USD"):
        raise HTTPException(status_code=400, detail="options aren't available for crypto symbols")
    now = time.time()

    # 1) The chain (spot + all expirations + the default expiry's contracts).
    try:
        chain = await options.fetch_chain(_http, symbol)
    except Exception as e:  # noqa: BLE001 — no chain (crypto/ETN/illiquid/unknown) is a 400, never a 500.
        _log.warning("puts fetch_chain failed for %s: %s", symbol.upper(), e)
        raise HTTPException(status_code=400, detail=f"options aren't available for {symbol.upper()}")
    if not chain.expirations or not chain.spot or chain.spot <= 0:
        raise HTTPException(status_code=400, detail=f"no option chain available for {symbol.upper()}")

    # 2) Next earnings (best-effort Finnhub) — used to skip straddling expiries + warn. No LLM here.
    earnings_date: str | None = None
    warnings: list[str] = []
    try:
        earnings_date = (await fetch_context(_http, symbol)).get("next_earnings")
    except Exception:  # noqa: BLE001 — decorative context; never fail the suggestion on it
        pass

    # 3) Pick the expiry (~25-50 DTE, target ~35, skip earnings straddles).
    try:
        chosen, exp_warnings = options.select_wheel_expiry(
            chain, low=options.PUT_DTE_LOW, high=options.PUT_DTE_HIGH, target=options.PUT_DTE_TARGET,
            now=now, earnings_date=earnings_date,
        )
    except Exception as e:  # noqa: BLE001 — malformed expiration data degrades to 400, not 500
        _log.warning("puts select_wheel_expiry failed for %s: %s", chain.symbol, e)
        raise HTTPException(status_code=400, detail=f"options aren't available for {symbol.upper()}")
    if chosen is None:
        raise HTTPException(status_code=400, detail=f"no suitable expiration for {chain.symbol}")
    warnings.extend(exp_warnings)

    # 4) Load the chosen expiry's contracts (reuse the first fetch if it already loaded it) + annotate.
    if not (chain.expiry and chain.expiry.expiration == chosen["ts"]):
        try:
            chain = await options.fetch_chain(_http, symbol, chosen["ts"])
        except Exception as e:  # noqa: BLE001 — generic 400 (the httpx error embeds the crumb'd URL)
            _log.warning("puts expiry reload failed for %s: %s", chain.symbol, e)
            raise HTTPException(status_code=400, detail=f"options aren't available for {symbol.upper()}")
    if chain.expiry is None:
        raise HTTPException(status_code=400, detail=f"no contracts for the chosen expiry of {chain.symbol}")

    # 5) Annotate + assemble. Malformed chain data degrades to 400, never a 500.
    try:
        options.annotate_expiry(chain, now_ts=now)
        body = options.assemble_put_suggestion(
            chain, chain.expiry, chosen=chosen, cash=cash, style=style,
            earnings_date=earnings_date, now=now, extra_warnings=warnings,
        )
    except Exception as e:  # noqa: BLE001
        _log.warning("puts assembly failed for %s: %s", chain.symbol, e)
        raise HTTPException(status_code=400, detail=f"options aren't available for {symbol.upper()}")
    if not body["candidates"]:
        raise HTTPException(status_code=400, detail=f"no quotable put contracts for the chosen expiry of {chain.symbol}")
    return body


@app.get("/covered_call/{symbol}")
async def covered_call_endpoint(
    symbol: str,
    shares: int,
    target: float | None = None,
    crypto: bool = False,
) -> dict:
    """No-LLM COVERED-CALL suggester (OC-8, the wheel's income/exit leg): sell a call on shares you
    already hold for income, capping upside at the strike. Requires `shares >= 100`; sizes
    `contracts = shares // 100`. Picks ONE call — the nearest strike AT/ABOVE `target` if given, else
    ~0.30 delta OTM — and reports premium income, premium + annualized yield, assignment probability,
    the called-away gain from today's price, greeks and an order ticket. Short-dated (~25-45 DTE).
    Options aren't available for crypto — a 400, not a 500. Decision support only, not investment advice."""
    assert _http is not None
    if crypto or symbol.upper().endswith("-USD"):
        raise HTTPException(status_code=400, detail="options aren't available for crypto symbols")
    if shares < 100:
        raise HTTPException(status_code=400, detail="covered calls need at least 100 shares")
    if target is not None and (not math.isfinite(target) or target <= 0):
        raise HTTPException(status_code=422, detail="target must be a positive number")
    now = time.time()

    # 1) The chain (spot + all expirations + the default expiry's contracts).
    try:
        chain = await options.fetch_chain(_http, symbol)
    except Exception as e:  # noqa: BLE001 — no chain (crypto/ETN/illiquid/unknown) is a 400, never a 500.
        _log.warning("covered_call fetch_chain failed for %s: %s", symbol.upper(), e)
        raise HTTPException(status_code=400, detail=f"options aren't available for {symbol.upper()}")
    if not chain.expirations or not chain.spot or chain.spot <= 0:
        raise HTTPException(status_code=400, detail=f"no option chain available for {symbol.upper()}")

    # 2) Next earnings (best-effort Finnhub) — used to skip straddling expiries. No LLM here.
    earnings_date: str | None = None
    warnings: list[str] = []
    try:
        earnings_date = (await fetch_context(_http, symbol)).get("next_earnings")
    except Exception:  # noqa: BLE001
        pass

    # 3) Pick the expiry (~25-45 DTE, target ~35, skip earnings straddles).
    try:
        chosen, exp_warnings = options.select_wheel_expiry(
            chain, low=options.CALL_DTE_LOW, high=options.CALL_DTE_HIGH, target=options.CALL_DTE_TARGET,
            now=now, earnings_date=earnings_date,
        )
    except Exception as e:  # noqa: BLE001
        _log.warning("covered_call select_wheel_expiry failed for %s: %s", chain.symbol, e)
        raise HTTPException(status_code=400, detail=f"options aren't available for {symbol.upper()}")
    if chosen is None:
        raise HTTPException(status_code=400, detail=f"no suitable expiration for {chain.symbol}")
    warnings.extend(exp_warnings)

    # 4) Load the chosen expiry's contracts (reuse the first fetch if it already loaded it) + annotate.
    if not (chain.expiry and chain.expiry.expiration == chosen["ts"]):
        try:
            chain = await options.fetch_chain(_http, symbol, chosen["ts"])
        except Exception as e:  # noqa: BLE001
            _log.warning("covered_call expiry reload failed for %s: %s", chain.symbol, e)
            raise HTTPException(status_code=400, detail=f"options aren't available for {symbol.upper()}")
    if chain.expiry is None:
        raise HTTPException(status_code=400, detail=f"no contracts for the chosen expiry of {chain.symbol}")

    # 5) Annotate + assemble. Malformed chain data degrades to 400, never a 500.
    try:
        options.annotate_expiry(chain, now_ts=now)
        body = options.assemble_covered_call(
            chain, chain.expiry, shares=shares, chosen=chosen, target=target,
            now=now, earnings_date=earnings_date, extra_warnings=warnings,
        )
    except Exception as e:  # noqa: BLE001
        _log.warning("covered_call assembly failed for %s: %s", chain.symbol, e)
        raise HTTPException(status_code=400, detail=f"options aren't available for {symbol.upper()}")
    if body is None:
        raise HTTPException(status_code=400, detail=f"no quotable call contracts for the chosen expiry of {chain.symbol}")
    return body


@app.get("/plan/{symbol}")
async def plan(
    symbol: str, cash: float, crypto: bool = False, deep: bool = False,
    shares: float | None = None, avg_cost: float | None = None, refresh: bool = False,
) -> dict:
    """Scenario: "if I deployed $cash into this symbol" — one asset's entry plan (action, entry zone,
    share count, stop/target, timing). Optional shares+avg_cost tell the analyst it's already held.
    `refresh=true` bypasses the cache to force a fresh plan.

    Also carries the SWT-3 chase read (`chase_pct`/`chase_status`/`chase_warning`/`chase_price`) —
    server-side arithmetic putting the live price against the analyst's own entry zone, so the screen
    can say "you are 4.7% above the plan's zone" instead of printing two numbers and hoping."""
    if cash <= 0:
        raise HTTPException(status_code=422, detail="cash must be > 0")
    cfg = settings_store.get()
    key = ("plan", symbol.upper(), crypto, round(cash, 2), deep, shares, avg_cost)
    now = time.time()
    hit = _cache.get(key)
    if hit and not refresh and now - hit[0] < cfg["verdict_ttl_seconds"]:
        # The chase read is recomputed on the way out and deliberately NOT stored in the cache entry.
        # Plans cache for verdict_ttl_seconds (4h by default) and the whole point of the read is
        # where the price is NOW: "you're inside the zone" measured four hours ago is precisely the
        # stale-number-worn-as-a-fact defect this app keeps having to delete.
        return chase.annotate({**hit[1], "cached": True}, await _chase_price(hit[1]["symbol"]))

    summary = await _snapshot(symbol, crypto=crypto)
    pos = _position_block(summary, shares, avg_cost)
    if pos:
        summary["position"] = pos
    try:
        entry, usage = await plan_entry(summary, cash=cash, deep=deep)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"analyst failed: {redact.redact(e)}")
    _sanitize_plan(entry, cash, crypto)
    usage_store.record(usage, symbol=summary.get("symbol", symbol), kind="plan")

    payload = {
        "symbol": summary.get("symbol", symbol.upper()),
        "model": usage["model"],
        "as_of": now,
        "cash": cash,
        # `summary` IS the snapshot this plan was drawn against — same scope, no lookup, so the
        # volatility the check measures against is by construction this symbol's.
        "plan": plan_check.annotate(entry.model_dump(), summary),
        "usage": usage,
        "cached": False,
    }
    _cache[key] = (now, payload)
    # Live quote first; the snapshot's own last print is the fallback, and only here — on this path
    # it is seconds old and is the very number the analyst drew the zone against. (The cached path
    # above has no such fresh price and reports the read as absent rather than inventing one.)
    return chase.annotate(payload, await _chase_price(payload["symbol"]) or summary.get("price"))


class Holding(BaseModel):
    symbol: str
    shares: float
    avg_cost: float


# Holdings that are the SAME economic exposure map to a shared group key, so the portfolio review +
# rebalance can treat e.g. BTC and a spot-Bitcoin ETF (FBTC/IBIT/…) as one exposure instead of two
# independent positions with their own (near-identical) technicals. Keys are the un-suffixed symbol.
_EXPOSURE_GROUP: dict[str, str] = {}
for _s in ("BTC", "IBIT", "FBTC", "GBTC", "BITB", "ARKB", "BTCO", "HODL", "BRRR", "EZBC", "BTCW"):
    _EXPOSURE_GROUP[_s] = "BTC"          # bitcoin + US spot-bitcoin ETFs
for _s in ("ETH", "FETH", "ETHA", "ETHE", "ETHW", "CETH", "EZET", "ETHV"):
    _EXPOSURE_GROUP[_s] = "ETH"          # ether + US spot-ether ETFs
for _s in ("GLD", "IAU", "GLDM", "SGOL", "IAUM", "OUNZ", "BAR", "AAAU"):
    _EXPOSURE_GROUP[_s] = "GOLD"         # gold-bullion ETFs
# Broad international ex-US — one exposure, for the same reason US_EQUITY is. Measured on 2y of
# daily returns (2026-08-27) against VXUS: IXUS 0.999, VEU 0.998, SPDW 0.986, VEA 0.987, SCHF 0.985,
# IEFA 0.967, VWO 0.913. Every one clears the 0.90 bar this file already uses, and VXUS-VTI came
# back at 0.773 on the same run, confirming the deliberate split from US equity still holds.
#
# VWO is included despite being emerging-markets ONLY rather than a substitute for VXUS. Left
# ungrouped it reproduces the exact failure that created US_EQUITY: two funds at 0.913 each drawing
# their own cap, so a book could hold twice the intended international weight while its risk model
# reported two diversified positions. An EM tilt is still expressible inside the group; it just
# shares the one international ceiling, which is the conservative default and the point of a cap.
#
# "VXUS" maps to INTL as well, which does double duty: it is the ticker's group AND the alias for
# the group name standing plans have been using ("13% shortfall vs plan" resolved against "VXUS"),
# so notes written before this regrouping keep resolving. See SUPERSEDED GROUP NAMES below.
for _s in ("VXUS", "IXUS", "VEU",            # all-world ex-US
           "VEA", "SCHF", "SPDW", "IEFA",    # developed ex-US
           "VWO"):                           # emerging markets
    _EXPOSURE_GROUP[_s] = "INTL"
# Broad US equity — ONE exposure, not several. Measured on 2y of daily returns (2026-08-05):
# SPY-VTI 0.997, VOO-VTI 0.997, SPY-VOO 0.998, QQQM-SPY 0.951, and SPMO joins the same cluster at
# 0.90. Treating these as independent let the 25% per-group cap apply once EACH, so the sandbox held
# 46.3% of its book (VTI 24.8% + SPY 21.5%) in what moves as a single asset while its risk model
# believed it was diversified across two.
#
# VTI is the total US market and SPY is the S&P 500, which is genuinely a difference in holdings —
# the S&P is ~80-85% of US market cap, so VTI adds a mid/small tail. At 0.997 that tail is not
# diversification, and pretending otherwise defeats the cap. Deliberately NOT merged: VXUS (0.778 vs
# VTI) and SCHD (0.641) do real work and keep their own groups.
for _s in ("SPY", "VOO", "IVV", "SPLG",      # S&P 500
           "VTI", "ITOT", "SCHB",            # total US market
           "QQQ", "QQQM",                    # Nasdaq-100
           "SPMO"):                          # S&P 500 momentum
    _EXPOSURE_GROUP[_s] = "US_EQUITY"

# SUPERSEDED GROUP NAMES. Strategy notes name their targets by GROUP, not by ticker, so a note
# written before a regrouping carries the old label forever — the standing plan still said "SP500"
# days after that group ceased to exist, and every consumer that resolved it got "SP500" back
# unchanged. That let a plan asking for VTI 25% + SP500 21% read as two compliant targets when it was
# 46% of one capped group. Aliasing the retired names keeps old notes resolvable against today's map.
for _s in ("SP500",):
    _EXPOSURE_GROUP[_s] = "US_EQUITY"


# Fund expense ratios, % per year. Fetched live from Yahoo's fundProfile, refreshed 2026-08-27 —
# re-check occasionally, issuers do cut them. Every figure the 2026-08-05 pass had came back
# unchanged on the refresh, which is the reason to trust the ones it added.
#
# ETFs only. Yahoo's `annualReportExpenseRatio` is not reliable for MUTUAL FUNDS: the same call
# returned 0.590% for FZILX and 1.760% for FZROX, which are Fidelity ZERO funds priced at 0.00%.
# Whatever that field carries for a mutual fund, it is not the number a holder pays, so no mutual
# fund belongs in this map until the figure comes from somewhere that can be trusted.
#
# This matters BECAUSE of the grouping above: once two funds are 0.997 correlated, the exposure is
# identical and cost is the only durable difference left between them. SPY charges 0.095% for what
# VOO, IVV and VTI deliver at 0.030% and SPLG at 0.020% — a 3-5x fee for the same tape. Likewise QQQ
# 0.180% vs QQQM 0.150% on the same index. The sandbox currently holds the expensive one (SPY).
#
# Absent from this map = unknown, not free. Single stocks have no expense ratio at all and are simply
# omitted rather than recorded as 0.
_EXPENSE_RATIO_PCT: dict[str, float] = {
    "SPLG": 0.020, "VTI": 0.030, "VOO": 0.030, "ITOT": 0.030, "SCHB": 0.030, "SPTM": 0.030,
    "IVV": 0.030, "SPY": 0.095, "SPMO": 0.130, "QQQM": 0.150, "QQQ": 0.180,
    # International. VXUS is the cheapest TRUE total-international vehicle here: VEA/SCHF/SPDW look
    # cheaper only because they drop emerging markets, and IXUS buys the identical exposure for more.
    "VEA": 0.030, "SCHF": 0.030, "SPDW": 0.030, "VEU": 0.040, "VXUS": 0.050,
    "VWO": 0.060, "IXUS": 0.070, "IEFA": 0.070,
    # Dividend funds are NOT one exposure — measured against SCHD on 2026-08-27: DGRO 0.892,
    # VYM 0.857, FDVV 0.783, all under the 0.90 bar. Priced here, grouped separately on purpose.
    "VYM": 0.040, "SCHD": 0.060, "SPYD": 0.070, "DGRO": 0.080, "FDVV": 0.150,
    # Gold. The widest like-for-like gap in this map: GLD charges 4x GLDM for the same bullion from
    # the same issuer, and the group above means the ledger already refuses to churn one into the
    # other — so this number can only ever steer NEW money, which is the only place it belongs.
    "IAUM": 0.090, "GLDM": 0.100, "SGOL": 0.170, "IAU": 0.250, "OUNZ": 0.250, "GLD": 0.400,
    # Spot bitcoin. HODL and BRRR are 0.000 by PROMOTIONAL WAIVER, not by pricing — both revert on
    # expiry, so re-check these two before treating them as the cheap option.
    "HODL": 0.000, "BRRR": 0.000, "BITB": 0.200, "ARKB": 0.210,
    "FBTC": 0.250, "IBIT": 0.250, "BTCO": 0.250, "GBTC": 1.500,
}


def _exposure_group(symbol: str) -> str:
    """The shared-exposure key for a holding (its own symbol when it has no known equivalent)."""
    base = symbol.upper().removesuffix("-USD")
    return _EXPOSURE_GROUP.get(base, base)


def _expense_ratio(symbol: str) -> float | None:
    """Annual expense ratio in %, or None when unknown (which is NOT the same as zero)."""
    return _EXPENSE_RATIO_PCT.get(symbol.upper().removesuffix("-USD"))


async def _build_portfolio_snapshot(
    holdings: list[Holding], cash: float, *, include_trend: bool = False,
) -> dict:
    """Price a set of holdings into the shared portfolio structure used by the review + rebalance
    endpoints: cash / cash_pct / total_value plus a `positions` list (price, shares, value, weight_pct,
    unrealized gain, key technicals) sorted by value, priced concurrently against one S&P fetch for
    relative strength. Unpriceable holdings are dropped; raises 502 if none can be priced."""
    assert _http is not None
    # One position per symbol. The same ticker sent twice produced two independent rows (66.7% and
    # 33.3% of one book), an equivalent_exposures entry of {"AAPL": ["AAPL", "AAPL"]} telling the
    # analyst to consolidate AAPL into AAPL, and a cap judged against half the real position. Merge
    # on shares, cost-weighting the basis.
    if len({h.symbol.upper() for h in holdings}) != len(holdings):
        merged: dict[str, Holding] = {}
        for h in holdings:
            k = h.symbol.upper()
            if k in merged:
                prev = merged[k]
                tot = (prev.shares or 0.0) + (h.shares or 0.0)
                basis = (prev.avg_cost or 0.0) * (prev.shares or 0.0) + (h.avg_cost or 0.0) * (h.shares or 0.0)
                merged[k] = Holding(symbol=k, shares=tot,
                                    avg_cost=(basis / tot) if tot else 0.0)
            else:
                merged[k] = Holding(symbol=k, shares=h.shares, avg_cost=h.avg_cost)
        holdings = list(merged.values())

    try:  # fetch the S&P once for all equity relative-strength calcs
        bench = (await fetch_series(_http, "^GSPC")).closes
    except Exception:  # noqa: BLE001 — relative strength just gets skipped
        bench = None

    _sem = asyncio.Semaphore(6)

    async def _price(h: Holding) -> dict | None:
        async with _sem:
            sym = h.symbol.upper()
            crypto = sym.endswith("-USD")
            try:
                series = await fetch_series(_http, sym)
                summ = summarize(series, None if crypto else bench)
            except Exception:  # noqa: BLE001 — carried at cost below rather than dropped
                # avg_cost can legitimately be 0 (the user never entered a basis). Valuing at cost
                # would then be valuing at ZERO, which reproduces the very weight-inflation this
                # fallback exists to prevent — measured: 33.3% -> 50.0%. Such a holding is UNVALUED,
                # and a book containing one cannot yield trustworthy weights at all.
                basis = (h.avg_cost or 0.0) * h.shares
                return {"symbol": sym.removesuffix("-USD"), "_unpriced": True,
                        "shares": h.shares, "avg_cost": h.avg_cost,
                        "value": round(basis, 2) if basis > 0 else None}
            price = summ["price"]
            # The multi-year value lens on a HELD name — the only input a sell decision gets about
            # where the position sits in its own long cycle. Opt-in (`include_trend`) so the sandbox
            # gets it without silently changing /portfolio/review and /rebalance, which share this
            # builder and would otherwise pick up a fetch and ~10 prompt fields per position.
            long_term = None
            if include_trend:
                try:
                    long_term = await _long_term_block(sym, series.closes)
                except Exception:  # noqa: BLE001 — enrichment, never a blocker
                    long_term = None
            return {
                "symbol": sym.removesuffix("-USD"),
                "exposure_group": _exposure_group(sym),
                **({"expense_ratio_pct": _expense_ratio(sym)} if _expense_ratio(sym) is not None else {}),
                "currency": (summ.get("currency") or "USD").upper(),
                "shares": h.shares,
                "avg_cost": h.avg_cost,
                "price": round(price, 4),
                "value": round(price * h.shares, 2),
                **({"long_term": long_term} if long_term else {}),
                "unrealized_gain_pct": round((price / h.avg_cost - 1) * 100, 1) if h.avg_cost else None,
                # Same key set as candidates (_SANDBOX_TECH_KEYS). It was a divergent hardcoded
                # copy, so held positions — the only source for a SELL's setup — carried fewer
                # features than candidates did.
                "technicals": {k: summ.get(k) for k in _SANDBOX_TECH_KEYS if summ.get(k) is not None},
            }

    all_rows = [r for r in await asyncio.gather(*[_price(h) for h in holdings]) if r]
    # A holding we could not price used to be DROPPED. That silently shrank total_value, which
    # inflated every surviving weight_pct AND cash_pct — measured: one unpriceable holding of two
    # equal ones took the other from a true 33.3% to a reported 50.0%, and made a 33% cash account
    # look 50% cash. The analyst sizes against exactly these numbers, on the real /portfolio/review
    # and /rebalance as well as the sandbox. So carry it at cost so the denominator stays honest,
    # keep it OUT of `positions` (it has no live price or technicals to make a decision on), and
    # name it in `unpriced` so the caller and the prompt both know the book is partial.
    unpriced = [r for r in all_rows if r.get("_unpriced")]
    unvalued = [r for r in unpriced if r.get("value") is None]
    rows = [r for r in all_rows if not r.get("_unpriced")]
    if not rows:
        # Nothing priced at all: fail loudly rather than serve a book valued entirely at cost basis,
        # which would render as "0% gain on everything" and read as real.
        raise HTTPException(status_code=502, detail="couldn't price any holdings")
    total_value = sum(r["value"] or 0.0 for r in all_rows) + max(cash, 0.0)
    # An unvalued holding means the DENOMINATOR is missing a position outright, so no weight in this
    # book is computable. Emit none at all rather than a confident wrong one — measured, a single
    # unvalued holding of two equal ones reports the other at 50.0% when the truth is 33.3%, and a
    # caveat under a wrong number is still a wrong number on a screen that drives real sells.
    for r in rows:
        r["weight_pct"] = (None if unvalued else
                           round(100.0 * r["value"] / total_value, 1) if total_value else None)
    # Groups the user holds more than one vehicle of — the SAME underlying exposure via redundant
    # tickers (e.g. BTC + FBTC). Surfaced so the analyst combines their weight + gives a consistent
    # stance instead of independent per-ticker calls.
    _by_group: dict[str, list[str]] = {}
    for r in rows:
        _by_group.setdefault(r["exposure_group"], []).append(r["symbol"])
    equivalent = {g: syms for g, syms in _by_group.items() if len(syms) > 1}
    # The cap is defined per EXPOSURE, but every weight above is per POSITION. Holding IBIT + FBTC
    # shows two 33.3% rows while the real BTC exposure is 66.6% — so a 25% cap reads as satisfied at
    # nearly 3x. equivalent_exposures named the pair but gave no combined number, and the prompt
    # asked the model to sum it mentally. Compute it here instead; arithmetic is not the model's job.
    # No FX rates here, so `value = price * shares` was summed across currencies one-for-one: a GBP
    # holding entered the USD total at face value and every weight computed off that total was wrong.
    # The APP already refuses to present such a sum without a caveat; the backend fed the analyst the
    # bare number. Name the mismatch, as with `unpriced` — a total we know is meaningless must say so.
    currencies = {r.get("currency", "USD") for r in rows}
    mixed = sorted(c for c in currencies if c != "USD") if len(currencies) > 1 else []

    group_value: dict[str, float] = {}
    for r in rows:
        group_value[r["exposure_group"]] = group_value.get(r["exposure_group"], 0.0) + r["value"]
    exposure_weights = ({} if unvalued else
                        {g: round(100.0 * v / total_value, 1)
                         for g, v in sorted(group_value.items(), key=lambda kv: -kv[1])}
                        if total_value else {})
    out = {
        "cash": round(cash, 2),
        "cash_pct": (None if unvalued else
                     round(100.0 * max(cash, 0.0) / total_value, 1) if total_value else 0.0),
        "total_value": round(total_value, 2),
        "positions": sorted(rows, key=lambda r: r["value"], reverse=True),
        "equivalent_exposures": equivalent,
        "exposure_weights": exposure_weights,
    }
    if mixed:
        out["mixed_currencies"] = mixed
    if unpriced:
        out["unpriced"] = [{"symbol": r["symbol"], "shares": r["shares"],
                            "value_at_cost": r["value"]} for r in unpriced]
        # Weights are computed over a denominator we know is incomplete. Say so, at two strengths:
        # carried-at-cost is a rough mark; unvalued means the denominator is missing that holding
        # entirely and no weight in this book can be trusted.
        out["weights_approximate"] = True
        if unvalued:
            out["unvalued"] = [r["symbol"] for r in unvalued]
    return out


class PortfolioReviewRequest(BaseModel):
    cash: float = 0.0
    deep: bool = False
    refresh: bool = False         # bypass the cache — what the Refresh control must actually do
    holdings: list[Holding] = []  # transient — reviewed, never persisted


@app.post("/portfolio/review")
async def portfolio_review_endpoint(req: PortfolioReviewRequest) -> dict:
    """AI review of the WHOLE portfolio: overall health, concentration/diversification flags, a per-
    holding action list (trim/hold/add/watch), and a cash-deployment note. One structured LLM call over
    lightweight technical snapshots of each holding (fast — no per-name enrichment). Cached by the
    holdings+cash identity for the verdict TTL. Send crypto holdings as <SYM>-USD."""
    if not req.holdings:
        raise HTTPException(status_code=422, detail="no holdings to review")
    cfg = settings_store.get()
    key = ("portfolio_review", req.deep, round(req.cash, 2),
           tuple(sorted((h.symbol.upper(), round(h.shares, 6), round(h.avg_cost, 4)) for h in req.holdings)))
    now = time.time()
    hit = _cache.get(key)
    if hit and not req.refresh and now - hit[0] < cfg["verdict_ttl_seconds"]:
        # Say WHEN, not just that it is cached: these carry concrete share counts priced at the
        # moment of the original call, and a plan up to the full TTL old read as current.
        return {**hit[1], "cached": True, "cached_age_seconds": int(now - hit[0])}

    # Same multi-year value lens the sandbox gets — a whole-book review and a rebalance are
    # decisions about where a name sits in its cycle at least as much as its last three months.
    portfolio = await _build_portfolio_snapshot(req.holdings, req.cash, include_trend=True)
    try:
        review, usage = await review_portfolio(portfolio, cash=req.cash, deep=req.deep)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"analyst failed: {redact.redact(e)}")
    usage_store.record(usage, symbol="", kind="portfolio")
    # Same rule as the rebalance path: the book is authoritative, the model is a proposal. An action
    # naming a symbol the user does not hold, or an action string outside trim/hold/add/watch,
    # rendered as a per-holding instruction with nothing checking it.
    review_d = review.model_dump()
    review_d["actions"], review_warnings = rebalance_check.validate_actions(
        portfolio, review_d.get("actions") or [])
    payload = {
        "review": review_d,
        "review_warnings": review_warnings,
        "portfolio": portfolio,
        "model": usage["model"],
        "as_of": now,
        "usage": usage,
        "cached": False,
    }
    _cache[key] = (now, payload)
    return payload


class RebalanceRequest(BaseModel):
    cash: float = 0.0
    deep: bool = False
    refresh: bool = False           # bypass the cache — what the Refresh control must actually do
    max_position_pct: float = 25.0  # target largest single-position weight after rebalancing
    holdings: list[Holding] = []    # transient — never persisted


@app.post("/portfolio/rebalance")
async def portfolio_rebalance_endpoint(req: RebalanceRequest) -> dict:
    """Theme C — a CONCRETE rebalance plan: sell N shares of the over-weights, redeploy proceeds + idle
    cash into the best-setup existing holdings, targeting `max_position_pct` as the largest single weight.
    Real share/dollar amounts for manual trading. One structured LLM call over the same priced snapshot
    the review uses. Cached by holdings+cash+target for the verdict TTL. Send crypto as <SYM>-USD."""
    if not req.holdings:
        raise HTTPException(status_code=422, detail="no holdings to rebalance")
    cfg = settings_store.get()
    mpp = max(5.0, min(100.0, req.max_position_pct))
    key = ("portfolio_rebalance", req.deep, round(req.cash, 2), round(mpp, 1),
           tuple(sorted((h.symbol.upper(), round(h.shares, 6), round(h.avg_cost, 4)) for h in req.holdings)))
    now = time.time()
    hit = _cache.get(key)
    if hit and not req.refresh and now - hit[0] < cfg["verdict_ttl_seconds"]:
        # Say WHEN, not just that it is cached: these carry concrete share counts priced at the
        # moment of the original call, and a plan up to the full TTL old read as current.
        return {**hit[1], "cached": True, "cached_age_seconds": int(now - hit[0])}

    # Same multi-year value lens the sandbox gets — a whole-book review and a rebalance are
    # decisions about where a name sits in its cycle at least as much as its last three months.
    portfolio = await _build_portfolio_snapshot(req.holdings, req.cash, include_trend=True)
    try:
        plan, usage = await rebalance_portfolio(portfolio, max_position_pct=mpp, deep=req.deep)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"analyst failed: {redact.redact(e)}")
    usage_store.record(usage, symbol="", kind="rebalance")
    # The plan is a PROPOSAL; the priced snapshot is authoritative. Nothing used to check that a sell
    # was of shares actually held, that the buys were affordable, that `dollars` agreed with
    # shares x price, or that the moves produced the resulting book the plan claimed — and the user
    # types these straight into a broker. Anything unexecutable is dropped or capped and named.
    plan_d = plan.model_dump()
    kept, derived, plan_warnings = rebalance_check.validate_plan(
        portfolio, plan_d.get("moves") or [], cash=req.cash, max_position_pct=mpp)
    plan_d["moves"] = kept
    plan_d["resulting_top_weight_pct"] = derived["resulting_top_weight_pct"]
    plan_d["cash_after"] = derived["cash_after"]
    payload = {
        "plan": plan_d,
        "plan_warnings": plan_warnings,
        "portfolio": portfolio,
        "max_position_pct": mpp,
        "model": usage["model"],
        "as_of": now,
        "usage": usage,
        "cached": False,
    }
    _cache[key] = (now, payload)
    return payload


class RecommendRequest(BaseModel):
    cash: float
    deep: bool = False
    holdings: list[Holding] = []  # transient — informs concentration, never persisted
    scope: str = "watchlist"     # "watchlist" | "market" (adds live-screened candidates)


@app.post("/recommendations")
async def recommendations(req: RecommendRequest) -> dict:
    """Rank candidates for NEW money: the analyst sees every snapshot at once (cross-comparison),
    picks the top 2-4, and spreads the cash across them with share counts. scope="market" widens the
    pool beyond the watchlist with candidates from live Yahoo screeners."""
    if req.cash <= 0:
        raise HTTPException(status_code=422, detail="cash must be > 0")
    market = req.scope == "market"
    cfg = settings_store.get()
    stocks = cfg.get("watchlist", [])
    cryptos = cfg.get("crypto_watchlist", [])
    if not stocks and not cryptos and not market:
        raise HTTPException(status_code=422, detail="watchlist is empty — open the app to sync it")

    assert _http is not None
    discovered: list[str] = []
    if market:
        exclude = {s.upper() for s in stocks} | {c.upper() for c in cryptos}
        discovered = await discover(_http, exclude)

    holdings = {h.symbol.upper(): h for h in req.holdings}
    key = (
        "recs", round(req.cash, 2), req.deep, tuple(sorted(stocks)), tuple(sorted(cryptos)),
        tuple(sorted((s, h.shares, h.avg_cost) for s, h in holdings.items())),
        req.scope, tuple(discovered),
    )
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < cfg["verdict_ttl_seconds"]:
        # The chase read is recomputed on the way out and deliberately NOT stored in the cache entry,
        # exactly as /plan does it. The ranking is a judgement that keeps for the TTL; "am I paying up
        # right now?" is not, and serving an hour-old chase percentage as current would be worse than
        # serving none.
        cached = {**hit[1], "cached": True}
        picks = [dict(p) for p in (cached.get("picks") or [])]
        cached["picks"] = chase.annotate_picks(
            picks, await _chase_prices([p.get("symbol") or "" for p in picks]))
        return cached

    bench: list[float] | None = None
    if stocks or discovered:  # fetch the S&P once for all equity snapshots
        try:
            bench = (await fetch_series(_http, "^GSPC")).closes
        except Exception:  # noqa: BLE001 — relative strength just gets skipped
            bench = None

    async def snap(sym: str, crypto: bool, source: str) -> dict | None:
        try:
            s = await _snapshot(sym, crypto=crypto, bench_closes=bench)
        except HTTPException:
            return None  # skip unfetchable symbols rather than failing the whole ranking
        s["source"] = source
        h = holdings.get(str(s.get("symbol", sym)).upper())
        if h:
            pos = _position_block(s, h.shares, h.avg_cost)
            if pos:
                s["position"] = pos
        return s

    snaps = [
        s for s in await asyncio.gather(
            *[snap(x, False, "watchlist") for x in stocks],
            *[snap(x, True, "watchlist") for x in cryptos],
            *[snap(x, False, "market_screen") for x in discovered],
        ) if s
    ]
    if not snaps:
        raise HTTPException(status_code=502, detail="no watchlist symbols could be fetched")

    try:
        recs, usage = await recommend(snaps, cash=req.cash, deep=req.deep)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"analyst failed: {redact.redact(e)}")
    # Enforce what the prompt only asks for: actionable picks, allocations that sum within the cash.
    actionable = [p for p in recs.picks if p.action in ("buy_now", "buy_on_pullback")]
    dropped = [p.symbol for p in recs.picks if p.action not in ("buy_now", "buy_on_pullback")]
    total = sum(p.allocation_usd for p in actionable)
    if total > req.cash > 0:
        scale = req.cash / total
        for p in actionable:
            p.allocation_usd *= scale
    for p in actionable:
        _sanitize_plan(p, req.cash, p.symbol.upper().endswith("-USD"))
    recs.picks = actionable
    recs.passed = list(dict.fromkeys([*recs.passed, *dropped]))  # dropped picks show as passed
    usage_store.record(usage, symbol="WATCHLIST", kind="recommend")

    payload = {
        "model": usage["model"],
        "as_of": now,
        "cash": req.cash,
        "scope": req.scope,
        "discovered": discovered,
        "considered": len(snaps),
        "overview": recs.overview,
        # Per-pick lookup: one /recommendations call ranks many symbols, and annotating a plan with
        # a neighbour's volatility would be worse than not annotating it at all.
        "picks": chase.annotate_picks(
            plan_check.annotate_picks([p.model_dump() for p in recs.picks], snaps),
            await _chase_prices([p.symbol for p in recs.picks]),
        ),
        "passed": recs.passed,
        "usage": usage,
        "cached": False,
    }
    _cache[key] = (now, payload)
    return payload


class ScanRequest(BaseModel):
    symbols: list[str]
    crypto_symbols: list[str] = []


@app.post("/scan")
async def scan(req: ScanRequest) -> dict:
    """Score a watchlist with the cheap scan model. MVP runs concurrently; the nightly job
    (task #6) should move this to the Anthropic Batch API + prompt caching for ~50% cost."""
    async def one(sym: str, crypto: bool) -> dict:
        try:
            return await _build_signal(sym, deep=False, crypto=crypto)
        except HTTPException as e:
            return {"symbol": sym.upper(), "error": e.detail}

    results = await asyncio.gather(
        *[one(s, False) for s in req.symbols],
        *[one(s, True) for s in req.crypto_symbols],
    )
    return {"count": len(results), "results": results}


@app.get("/scan/latest")
async def scan_latest() -> dict:
    """The most recent nightly-scan result (what the app polls).

    `scan_available` is the field clients must branch on. When it is False there is NO scan on disk
    (none has ever run, or the stored one is unreadable) and every list comes back null rather than
    empty — because "we looked and the market is calm" and "we could not look" are different claims
    and only the first is ours to make. The app rendered a reassuring "No dips right now" for both,
    which is the reassuring one being the lie.

    On a scan that did run, `dip_rejects` carries the names the dip radar TURNED DOWN with the reason
    for each, and `dip_counts` partitions the scanned set (qualified + near_miss + nowhere_near +
    unmeasured == scanned). A symbol whose data failed to fetch is `unmeasured`, never "no dip".
    """
    reason = "no scan has run yet"
    if LATEST.exists():
        try:
            payload = json.loads(LATEST.read_text())
        except (OSError, ValueError) as e:
            # A corrupt/half-written file is a FAILURE to read the scan, not an empty scan. Fall
            # through to the unavailable payload rather than 500-ing or serving fabricated calm.
            _log.warning("scan/latest: stored scan unreadable (%s) — reporting unavailable, not empty", e)
            reason = "the stored scan could not be read"
        else:
            payload["scan_available"] = True
            # A scan written before the reject fields existed carries neither. Null them explicitly
            # rather than leaving the keys missing: a client reading an absent `dip_counts` as zeros
            # would report "0 near misses" about a scan that never measured any.
            payload.setdefault("dip_rejects", None)
            payload.setdefault("dip_counts", None)
            return payload
    return {
        "generated_at": None, "scan_available": False, "unavailable_reason": reason,
        "results": None, "flips": None, "crossed_below_200wma": None, "dip_alerts": None,
        "dip_rejects": None, "dip_counts": None, "date_alerts": None, "total_cost_usd": None,
    }


@app.post("/scan/run")
async def scan_run() -> dict:
    """Run the configured-watchlist scan now (also wired to a nightly systemd timer)."""
    return await run_scan()


# ======================================================================================
# Macro / geopolitical catalysts (NEWS) — the exogenous-risk layer. Everything else in this service
# is derived from price or fundamentals, so a war or a rate decision was previously invisible to it.
# ======================================================================================

@app.get("/macro/catalysts")
async def macro_catalysts() -> dict:
    """The current macro read: risk level, standing catalysts, and how old the read is.

    `available` is False when no read exists at all, and `degraded` is True when the last run failed
    but an older read survives. Consumers MUST distinguish those from an empty catalyst list — "we
    couldn't look" and "nothing is happening" are different claims and only one is ours to make.
    """
    return macro.load_state()


@app.post("/macro/run")
async def macro_run(force: bool = False) -> dict:
    """Run the macro research pass now (also wired to a systemd timer a few times a day)."""
    return await run_macro(force=force)


# ======================================================================================
# SWT-1 — the nightly MARKET-WIDE cross-section: ~3,100 liquid US equities, the same ~25 price
# measurements on each, stored one row per (night, symbol) in data/scan.db.
#
# Free. NO LLM anywhere in this section — daily bars and arithmetic, nothing else.
#
# CONTEXT, NOT A BUY SIGNAL, on every route below. A cross-section says where a name sits relative
# to the rest of the market on one night. That is a description of the tape, not a recommendation,
# and "top of the momentum ranking" is emphatically not "buy this".
#
# Threading: scan_store's functions are SYNCHRONOUS SQLite and are reached ONLY through
# asyncio.to_thread (the pattern at the usage_store/selfupdate calls above). GET /memory/stats calls
# memory.stats() straight from its async def and blocks this single-worker event loop for the
# duration; over a 288k-row table that is not survivable, so this section does not copy it.
#
# Writing: the API process is a READ-ONLY consumer of scan.db. app/market_scan_job.py is its sole
# writer, and POST /market_scan/run is the one door to it.
# ======================================================================================

_MARKET_SCAN_TTL = 900          # ~15 min. The underlying rows change once a night; this only
                                # coalesces a tab being reopened, and the key below carries the
                                # scan's own date so a fresh scan invalidates it outright.
_MARKET_SCAN_LIMIT_MAX = 200    # Hard cap. The night is ~3,100 rows; that is a bulk export, not
                                # something to hand a phone, and `limit` is the only thing between
                                # the two.
_MARKET_SCAN_DEFAULT_SORT = "rel_strength"

# Query parameters that are NOT filters. Everything else in the query string is looked up in
# scan_store.FILTER_NAMES and 422s if it is not there — a filter that silently does nothing returns
# a LONGER list than was asked for and presents it as filtered.
_MARKET_SCAN_RESERVED = frozenset({"limit", "sort", "refresh"})


def _market_scan_summary() -> dict | None:
    """The last run's summary JSON, or None if there isn't a readable one. Blocking file read."""
    try:
        return json.loads(market_scan_job.LATEST.read_text())
    except Exception:  # noqa: BLE001 — an absent or corrupt summary means "we cannot say how the
        # last run went". That is reported as null provenance below, never as zeros, and it must not
        # take down a read of rows that are sitting in the database perfectly intact.
        return None


def _market_scan_provenance(night: str | None) -> dict:
    """Run counters for the night being served — and nulls whenever they'd describe a DIFFERENT one.

    The summary file and the rows can legitimately disagree. A run that refuses today (stale
    universe) still writes a summary stamped with today's session and `scanned: null`, while
    scan.db keeps last night's rows and happily serves them. Pinning today's counters onto last
    night's numbers would be provenance for the wrong data — worse than no provenance, because it
    reads as confirmation.

    `scanned` / `fetch_failed` / `too_short` stay three separate facts: a name we could not fetch
    and a name with too little history are different claims, and neither is "we scanned it".
    """
    s = _market_scan_summary() or {}
    same = bool(night) and str(s.get("session") or "") == str(night)
    built = s.get("universe_built_at") if same else None
    stale: bool | None = None
    if built:
        try:
            stale = universe.is_stale({"built_at": built})
        except Exception:  # noqa: BLE001 — a corrupt built_at reaches is_stale as a string and
            # raises there; "we cannot judge its freshness" is None, and is not the same as "fresh".
            stale = None
    return {
        "generated_at": s.get("generated_at") if same else None,
        "universe_size": s.get("universe_symbols") if same else None,
        "scanned": s.get("scanned") if same else None,
        "fetch_failed": s.get("fetch_failed") if same else None,
        "too_short": s.get("too_short") if same else None,
        "universe_built_at": built,
        "universe_stale": stale,
    }


def _market_scan_note(shown: int, total: int | None, prov: dict) -> str:
    """The one human sentence that says what slice this is, and of what."""
    head = f"Top {shown} of {total:,} matching" if total is not None else f"Top {shown} matching"
    scanned, size = prov.get("scanned"), prov.get("universe_size")
    if scanned is not None and size is not None:
        mid = f", from {scanned:,} scored of {size:,} in the universe."
    elif scanned is not None:
        mid = f", from {scanned:,} scored."
    else:
        # Rows without a matching run summary. Saying "of 3,113 scanned" here would be inventing the
        # denominator; saying nothing at all would let the reader assume the whole market.
        mid = ", from a scan whose run summary is unavailable."
    return (head + mid + " Where these names sit relative to the market on this night — "
            "CONTEXT, NOT A BUY SIGNAL.")


def _market_scan_percentiles(row: dict | None) -> dict:
    """SWT-4 — the rank half of a row: {metric: percentile or None}, EVERY metric always present.

    Always present, and null when it is not there, because the alternative is a client reading an
    absent key as zero — "this name is at the bottom of the market" assembled out of a pass that has
    not run yet. A night stored before the pass existed, a night whose backfill has not been run, and
    a metric too little of the market could be measured on all report the same honest null.

    These are RANKS, not scores: 96 means 96% of the names measured that night were lower, for a
    metric whose "good" end is the reader's judgement and not this service's. See app/percentiles.py
    before averaging any of them together.
    """
    r = row or {}
    return {metric: r.get(col) for metric, col in scan_store.PCT_COLUMNS.items()}


def _market_scan_bool(name: str, raw: str) -> bool:
    """A query-string boolean, refused rather than guessed at.

    `above_sma200=maybe` silently read as False is a bearish filter the caller never asked for.
    """
    v = str(raw).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    raise HTTPException(status_code=422, detail=f"filter {name!r} must be true or false, got {raw!r}")


def _market_scan_filters(request: Request) -> dict:
    """The query string's filter kwargs, validated against scan_store's own published vocabulary."""
    out: dict = {}
    for key, raw in request.query_params.multi_items():
        if key in _MARKET_SCAN_RESERVED:
            continue
        if key not in scan_store.FILTER_NAMES:
            raise HTTPException(status_code=422, detail=(
                f"unknown filter {key!r} — filters are min_<metric> / max_<metric>, or a bare "
                f"boolean metric ({', '.join(sorted(scan_store.BOOL_FILTERS))})"))
        if key in scan_store.BOOL_FILTERS:
            out[key] = _market_scan_bool(key, raw)
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            raise HTTPException(status_code=422,
                                detail=f"filter {key!r} must be a number, got {raw!r}") from None
        # float("nan") parses. A NaN bound makes every comparison false, so the response would be a
        # legitimately-shaped empty cross-section produced by a typo.
        if not math.isfinite(val):
            raise HTTPException(status_code=422, detail=f"filter {key!r} must be a finite number, got {raw!r}")
        out[key] = val
    return out


def _invalidate_market_scan_cache() -> None:
    """Drop every cached market-scan slice and breadth reading — and the gate, which reads breadth.

    The cache keys on the scan's DATE, which is the right key for a job that runs once a night — but
    `force=true` re-measures the SAME night, so the date alone would keep serving the pre-run answer
    for up to 15 minutes with `cached: true` stamped on it.

    The gate is included because its breadth leg comes out of these same rows: a forced re-measure of
    tonight changes the gate's answer without changing its cache key either. Its TTL is only 60s, so
    this buys a minute — but a re-run exists precisely because someone did not trust the stored
    numbers, and serving them back for another minute is the wrong answer to that.
    """
    for k in [k for k in _cache
              if isinstance(k, tuple) and k and str(k[0]).startswith(("market_scan", "gate"))]:
        _cache.pop(k, None)


@app.get("/market_scan")
async def market_scan_endpoint(request: Request, limit: int = 50,
                               sort: str = _MARKET_SCAN_DEFAULT_SORT,
                               refresh: bool = False) -> dict:
    """SWT-1 — a ranked SLICE of last night's market-wide cross-section. Free: NO LLM, no fetches.

    CONTEXT, NOT A BUY SIGNAL. This ranks ~3,100 liquid names on one measurement and hands back the
    head of that ranking. Being top of a momentum sort is a statement about the last three months of
    price, not about what happens next — read it the way you read a leaderboard, not a shortlist.

    `sort` is any metric column, descending, NULLs last; prefix it with "-" for ascending
    ("-atr14_pct" for the calmest names rather than the wildest). Filters are `min_<metric>` /
    `max_<metric>` query params plus bare booleans (`above_sma200=true`); an unknown one is a 422
    rather than a filter that quietly does nothing. `limit` is capped at 200 — the whole night is a
    bulk export, not a payload.

    Every row also carries SWT-4's `<metric>_pctile` columns — where that name sat in the night's
    full cross-section, ranked ascending, `percentiles_over` names deep. They are RANKS, not scores
    (high RSI is not "good"), and a null is "not ranked": the pass has not run for that night, or too
    little of the market was measurable on that metric. A null is never a zero, and a client that
    defaults it to one is claiming the name is the worst in the market.

    503 when no scan has ever run: an empty cross-section rendered as a market reading is the exact
    defect this service keeps correcting.
    """
    if sort.lstrip("-") not in scan_store.SORT_COLUMNS:
        raise HTTPException(status_code=422, detail=(
            f"unknown sort {sort!r} — one of: {', '.join(sorted(scan_store.SORT_COLUMNS))} "
            f"(prefix with '-' for ascending)"))
    limit = max(1, min(_MARKET_SCAN_LIMIT_MAX, limit))
    filters = _market_scan_filters(request)

    night = await asyncio.to_thread(scan_store.latest_date)
    if night is None:
        raise HTTPException(status_code=503, detail="no market scan has run yet — POST /market_scan/run")

    # The key carries EVERY input that changes the answer, including the scan's own date — the same
    # rule /screener/value keys the universe's built_at on, and the one GET /regime's `count` misses.
    # The date is what makes this self-invalidating: a new night is a new key, so a fresh scan can
    # never be served from the previous night's entry.
    key = ("market_scan", night, sort, limit, tuple(sorted(filters.items())))
    now = time.time()
    hit = _cache.get(key)
    if hit and not refresh and now - hit[0] < _MARKET_SCAN_TTL:
        return {**hit[1], "cached": True, "cached_age_seconds": int(now - hit[0])}

    rows = await asyncio.to_thread(scan_store.top, night, sort=sort, limit=limit, **filters)
    # What the slice is a slice OF. len(rows) can only ever report `limit` back.
    total = await asyncio.to_thread(scan_store.count, night, **filters)
    # SWT-4 — the denominator the `*_pctile` columns on each row were ranked against. Unfiltered on
    # purpose: a percentile is a position in the whole night, not in the slice being served, and a
    # client that labelled "98th of 50 shown" would be describing the wrong population. Re-uses
    # `total` when no filter narrowed it, rather than asking the same COUNT twice.
    ranked_over = total if not filters else await asyncio.to_thread(scan_store.count, night)
    prov = await asyncio.to_thread(_market_scan_provenance, night)

    payload = {
        "as_of": night,
        **prov,
        "sort": sort,
        "limit": limit,
        "total_matching": total,
        "percentiles_over": ranked_over,
        "percentile_columns": dict(scan_store.PCT_COLUMNS),
        "results": rows,
        "note": _market_scan_note(len(rows), total, prov),
        "cached": False,
        # Present even when fresh, so a client decoding this never has to treat the key as optional.
        "cached_age_seconds": 0,
    }
    _cache[key] = (now, payload)
    return payload


@app.get("/market_scan/breadth")
async def market_scan_breadth_endpoint(refresh: bool = False) -> dict:
    """SWT-1 — market participation for the last night scanned. Free: NO LLM, no fetches.

    CONTEXT, NOT A BUY SIGNAL. Breadth describes how much of the market is participating; it is a
    backdrop reading, and on its own it says nothing about any individual name.

    READ `available` FIRST. When it is False every reading here is null and the caller must render
    "no scan" — a dash, not a number. This route NEVER substitutes 0 for a missing scan:
    `pct_above_sma50: 0.0` is not "we did not scan", it is "none of the market is above its 50-day
    average", which is the single most bearish breadth print that exists. Unlike the other routes in
    this section it therefore does not 503 on a missing scan either — SWT-2 depends on always
    getting a decodable, honestly-null envelope back.
    """
    night = await asyncio.to_thread(scan_store.latest_date)
    key = ("market_scan_breadth", night)
    now = time.time()
    hit = _cache.get(key)
    if hit and not refresh and now - hit[0] < _MARKET_SCAN_TTL:
        return {**hit[1], "cached": True, "cached_age_seconds": int(now - hit[0])}

    # scan_store.breadth() already returns the all-None shape on a missing night or an unreadable
    # database, and it never raises. Passing `night` (rather than None) keeps the reading and the
    # cache key describing the same night even if a scan lands between the two calls.
    reading = await asyncio.to_thread(scan_store.breadth, night)
    prov = await asyncio.to_thread(_market_scan_provenance, night)
    payload = {**reading, **prov, "cached": False, "cached_age_seconds": 0}
    _cache[key] = (now, payload)
    return payload


@app.get("/market_scan/{symbol}")
async def market_scan_symbol_endpoint(symbol: str) -> dict:
    """SWT-1 — one name's row from the market scan. Free: NO LLM, no fetches.

    CONTEXT, NOT A BUY SIGNAL: these are measurements of this name's own price history plus where it
    sits against the market, and nothing here is a view on it.

    `row` is the night's raw measurements; `percentiles` is where each of ten of them sat in that
    night's whole cross-section (`percentiles_over` names deep). Read them together — the rank is
    what makes the raw number legible, and the raw number is what the rank is a rank OF. A null
    percentile means the pass has not run for that night or too little of the market could be
    measured on that metric; it is never a zero, and never "worst in the market".

    404 — "we looked and there is no data for this symbol" — when the name is in no night we hold.
    The row returned is that symbol's MOST RECENT one, which is not necessarily the most recent
    night overall: a name that was halted or failed to fetch last night still has an older
    observation, and `as_of` / `is_latest_night` / `latest_scan_date` say plainly which night it is
    rather than passing a three-day-old row off as tonight's.
    """
    sym = str(symbol or "").strip().upper()
    if not sym:
        raise HTTPException(status_code=422, detail="symbol is required")
    row = await asyncio.to_thread(scan_store.symbol_row, sym)
    if row is None:
        raise HTTPException(status_code=404, detail=f"{sym} was not in the last market scan")

    night = await asyncio.to_thread(scan_store.latest_date)
    # Provenance for the night THIS ROW came from, not for the latest one — see
    # _market_scan_provenance. An older row gets null counters, which is the truth about it.
    prov = await asyncio.to_thread(_market_scan_provenance, row.get("d"))
    # The population THIS ROW's ranks were computed against — its own night, not the latest one, for
    # the same reason its provenance is. A three-day-old row's percentile is a position in a
    # three-day-old market and saying so is the only way the number stays true.
    ranked_over = await asyncio.to_thread(scan_store.count, row.get("d"))
    return {
        "symbol": sym,
        "as_of": row.get("d"),
        "latest_scan_date": night,
        "is_latest_night": bool(night) and row.get("d") == night,
        "row": row,
        # Alongside the raw row, never instead of it: the reader needs both "RSI 81.4" and "higher
        # than 96% of the names measured that night", and the second is meaningless without the
        # first. Nulls here mean "not ranked", never the 0th percentile.
        "percentiles": _market_scan_percentiles(row),
        "percentiles_over": ranked_over,
        **prov,
        "note": ("One name's measurements from the nightly market-wide scan, and where each sits in "
                 "that night's cross-section — a RANK, not a score. CONTEXT, NOT A BUY SIGNAL."),
    }


@app.post("/market_scan/run")
async def market_scan_run_endpoint(force: bool = False, limit: int | None = None) -> dict:
    """SWT-1 — run the market-wide scan now (also wired to a nightly systemd timer). No LLM.

    EXPENSIVE, and therefore a POST that no read path can trigger: ~3,100 symbols of daily bars,
    ~50s wall on this box. Awaited inline like POST /scan/run and POST /macro/run, so the response
    IS the run summary rather than a promise that something started.

    `force` re-measures a session already stored. `limit` takes the head of the cap-sorted universe
    for a smoke test — a limited run is a SAMPLE, and the job deliberately skips retiring absent rows
    on one so a partial run can never replace a night's cross-section with a fraction of itself.
    """
    if limit is not None and limit < 1:
        raise HTTPException(status_code=422,
                            detail="limit must be >= 1 — omit it to scan the whole universe")
    out = await market_scan_job.run_market_scan(force=force, limit=limit)
    # A forced re-run rewrites tonight's rows under tonight's cache key; without this the previous
    # answer would keep being served for the rest of the TTL.
    _invalidate_market_scan_cache()
    return out


@app.post("/market_scan/percentiles")
async def market_scan_percentiles_endpoint(d: str | None = None) -> dict:
    """SWT-4 — rank an ALREADY-STORED night and write the percentiles onto it. No LLM, no fetches.

    The pass normally runs at the tail of the nightly job, in that job's own process. This is the
    door for the nights that predate it, or one whose pass was interrupted: `d` is a YYYYMMDD (the
    latest stored night by default), and it is idempotent — it reads only the measured columns, so
    running it twice writes the same ranks twice.

    A POST because it is bulk work over ~3,100 rows, in keeping with every other expensive operation
    here, and because it WRITES — the API process is otherwise a read-only consumer of scan.db.
    `scan_store.write_percentiles` commits in small chunks specifically so this cannot hold the
    single write lock for the length of a whole pass; see the third-writer note in that module.

    422 on a date that is not a date, 404 on a night that holds no rows, 503 when nothing has ever
    been scanned. None of the three invents an empty result to hand back: a rank pass that ran over
    no rows and one that never ran are different facts.
    """
    out = await asyncio.to_thread(percentiles.run, d)
    status = out.get("status")
    if status == "refused":
        raise HTTPException(status_code=422, detail=out.get("reason") or "unusable date")
    if status == "no_scan":
        raise HTTPException(status_code=503, detail="no market scan has run yet — POST /market_scan/run")
    if status == "empty":
        raise HTTPException(status_code=404, detail=out.get("reason") or f"night {d!r} holds no rows")
    # The ranks are columns on rows a cached slice already served without them. The cache keys on the
    # night, which a backfill does not change, so nothing else would ever evict them.
    _invalidate_market_scan_cache()
    return {**out, "note": ("Where each name sits in that night's cross-section, metric by metric — "
                            "a RANK, not a score, and not a buy signal.")}



# ======================================================================================
# SWT-2 — the market GATE: the checkable half of the market read. Five mechanical legs (SPY and QQQ
# over their 50-day EMAs, breadth over 55%, VIX under 20, SPY's 20-day return positive) evaluated
# together, so a buy-side consumer can stand aside for a reason it can NAME and a reader can check
# the claim against any chart package.
#
# GET /regime stays, and is the other half. It is an analyst narrative: what kind of market this is,
# in sentences, for a human. This is arithmetic: whether five stated conditions hold right now, with
# every number and threshold published beside the verdict. Neither replaces the other, and when they
# disagree that disagreement is information — a bullish narrative over a closed gate is exactly the
# moment to look at which leg failed.
#
# THE INVARIANT THESE ROUTES CARRY (see app/gate.py): a leg that could not be measured is NEVER a
# silent pass and NEVER a silent fail. Its `ok` is null and its name is in `unmeasured`; `passed` is
# true only when all five are explicitly true, and null — not false — when a leg is unknown and none
# of the measurable ones failed. These routes must not flatten that on the way out: `passed: false`
# manufactured out of a missing nightly scan is indistinguishable from a bearish market, and every
# consumer that sees it once learns to ignore a real closure.
# ======================================================================================

_GATE_TTL = 60          # The price legs come off the live tape and the app refreshes its own quotes
                        # every 60s; a gate lagging the prices printed next to it on screen would
                        # contradict them. This only coalesces a burst of taps, it is not a stale
                        # window anyone should be reading through.
_GATE_FAIL_TTL = 10     # An evaluation that measured NOTHING is cached far more briefly than a real
                        # one: it is a statement about our upstreams, not about the market, and
                        # freezing it for a full minute turns a three-second blip into a minute of
                        # "unknown". Long enough to absorb a retry storm, short enough that a
                        # recovered feed shows up on the very next poll.
_GATE_HISTORY_MAX = 365  # ~a year of rows at ~400 bytes each. Gate rows are deliberately not pruned
                         # with the cross-section, so without a cap this grows into a bulk export.


@app.get("/gate")
async def gate_endpoint(refresh: bool = False) -> dict:
    """SWT-2 — the five-leg market gate, evaluated now. FREE: no LLM, no analyst, no tokens.

    Cost is three quote fetches (SPY, QQQ, ^VIX) plus one local SQLite read for breadth. This is a
    MECHANICAL, CHECKABLE gate — every leg publishes its `value`, its `threshold` and a sentence, so
    "breadth 54.1%, gate closed" can be verified against any chart package. It is deliberately
    distinct from GET /regime, which pays an analyst to narrate the structural backdrop in prose;
    that route answers "what kind of market is this", this one answers "do five stated conditions
    hold right now".

    READ `available` FIRST, THEN `passed`, WHICH IS THREE-VALUED.
      * `available: false` — nothing could be measured. There is no verdict here; render "unknown".
      * `passed: true`  — all five legs explicitly passed.
      * `passed: false` — at least one leg explicitly failed; `failing` names them.
      * `passed: null`  — a leg could not be measured (`unmeasured` names it) and nothing failed.

    Null is not false. A gate that could not read last night's scan has not observed a bearish
    market, and this route never fabricates one: it returns `available: false` with all five legs
    null rather than 503ing or inventing a verdict, because a consumer standing aside needs a
    decodable envelope more than it needs an error code. `market_score` (0-100) is a plotting aid
    over the same legs and is null whenever any leg is — it is NOT a probability, a forecast, or a
    position size.

    Cached ~60s (~10s when unavailable); `refresh=true` forces a re-evaluation. The cache key carries
    the nightly scan's date, so a scan landing between two calls re-evaluates instead of serving a
    gate that was computed without it.
    """
    assert _http is not None
    # The scan's date is the one input to the gate that changes discretely — the price legs change
    # continuously and are covered by the 60s TTL instead. Keying on it means the first request after
    # a scan lands re-evaluates rather than serving an evaluation whose breadth leg was null.
    night = await asyncio.to_thread(scan_store.latest_date)
    key = ("gate", night)
    now = time.time()
    hit = _cache.get(key)
    if hit and not refresh:
        # An unavailable evaluation gets its own, much shorter window — see _GATE_FAIL_TTL. The TTL
        # is read off the CACHED payload, not the live one, because it is that payload's honesty
        # about itself that decides how long it may be served.
        ttl = _GATE_TTL if hit[1].get("available") else _GATE_FAIL_TTL
        if now - hit[0] < ttl:
            return {**hit[1], "cached": True, "cached_age_seconds": int(now - hit[0])}

    # gate.evaluate() never raises and never fabricates: a dead upstream nulls the leg it feeds and
    # nothing else, and a totally failed evaluation comes back as the unavailable shape with all five
    # legs present and null. So there is no try/except here to write — adding one would only be able
    # to turn a precise "we could not measure X" into a vaguer 502.
    #
    # `d` is deliberately not passed and not exposed as a query param: the price legs are always
    # live, so a historical `d` would file today's tape under a past date. Backfilling real gate
    # history needs stored index bars, which this service does not keep.
    out = await gate.evaluate(_http)
    payload = {**out, "cached": False, "cached_age_seconds": 0}
    _cache[key] = (now, payload)
    return payload


@app.get("/gate/history")
async def gate_history_endpoint(limit: int = 30) -> dict:
    """SWT-2 — what the gate reported, day by day. FREE: no LLM, no fetches, one SQLite read.

    One row per day, NEWEST FIRST, as filed by GET /gate (`evaluate()` records every evaluation that
    measured at least one leg; one that measured nothing is never filed, so an outage cannot
    overwrite a real morning reading with five nulls). Within a day the row is last-write-wins — it
    is what the gate most recently reported, not what it first reported.

    `passed` STAYS THREE-VALUED THROUGH THE DATABASE. SQLite has no boolean and `bool(0)` and
    `bool(None)` are both False in Python, so a generic restore would collapse "the gate declined to
    decide" into "the gate failed" — a bearish claim about a day on which no such claim was made.
    A null here means the gate ran and could not decide; a day that is simply MISSING from this list
    means no evaluation was ever filed for it. Those are different facts and neither is a false.

    An empty list means no gate evaluation has ever been recorded — call GET /gate to file one. It is
    not a run of failing days.
    """
    if limit < 1:
        raise HTTPException(status_code=422, detail="limit must be >= 1 — it is a row count")
    limit = min(_GATE_HISTORY_MAX, limit)
    rows = await asyncio.to_thread(scan_store.gate_history, limit)
    return {
        "limit": limit,
        # Rows RETURNED, which is bounded by `limit` and is not a count of days on record. There is
        # no cheap total to report here and inventing one out of len(rows) would be that same
        # len()-as-a-total mistake /market_scan's `total_matching` exists to avoid.
        "count": len(rows),
        "history": rows,
        "note": ("One row per day, newest first. `passed` is three-valued: true, false, or null — "
                 "null means the gate ran and could not measure every leg, never that it failed."),
    }


# ======================================================================================
# AI Sandbox — an autonomous paper-trading agent (fictional money only; never touches the real
# Portfolio/watchlist). The systemd timer curls POST /sandbox/tick, so this single uvicorn worker is
# the SOLE ledger writer; _sandbox_lock serializes the read-modify-write endpoints. The LLM only
# proposes; sandbox_job.validate_and_fill is the authority on what the ledger does.
# ======================================================================================

_sandbox_lock = asyncio.Lock()
_SANDBOX_CAND_SEM = asyncio.Semaphore(6)

# The analyst sees these on every position and candidate. pct_vs_sma20 / bollinger_pct_b /
# stochastic_k were added because memory records sandbox fills from exactly this dict: without them
# every origin='sandbox' row was missing 3 of 8 features INCLUDING pct_vs_sma20, which is a veto
# dimension — so sandbox rows could never be excluded from a match on it, and matched more loosely
# than any other row in the table.
_SANDBOX_TECH_KEYS = ("rsi14", "macd_hist", "pct_vs_sma20", "pct_vs_sma50", "golden_cross",
                      "bollinger_pct_b", "stochastic_k",
                      "rel_strength_3mo_vs_benchmark", "pct_off_52w_high")

# The multi-year value lens, projected down to what a trading decision can use.
#
# Every key in _SANDBOX_TECH_KEYS above is momentum with a lookback of three months or less, so the
# sandbox could only ever see a name through a short window. Measured consequence: on 2026-07-30 it
# sold IBIT citing "-17.64% 3-month relative strength" while bitcoin sat 1.2% BELOW its 200-week line
# and 49% off its 10-year high — i.e. squarely in the accumulation zone this app's own value screener
# is built around. It wasn't weighing value against momentum and choosing momentum; it had no value
# input at all. This is that input.
_SANDBOX_TREND_KEYS = ("price_vs_200w_sma_pct", "below_line", "zone", "direction", "rsi_14w",
                       "weekly_oversold", "mayer_multiple", "pct_off_10y_high", "drawdown_z",
                       "cagr_3y_pct")


def _compact_trend(lt: dict | None) -> dict | None:
    """The long-term block trimmed to the decision-relevant keys (the full one carries raw SMA levels
    and history bookkeeping the model doesn't need and shouldn't pay tokens for)."""
    if not lt:
        return None
    out = {k: lt[k] for k in _SANDBOX_TREND_KEYS if lt.get(k) is not None}
    return out or None


# A spot-crypto ETF's long-cycle position is its UNDERLYING's, not its own.
#
# The 200-week fields need ~200 weekly bars (≈3.85 years) and are omitted below that. The US spot
# bitcoin ETFs launched in January 2024, so measured on the live feed IBIT and FBTC return only
# mayer_multiple / pct_off_10y_high / drawdown_z — no `below_line`, no `zone`, precisely the fields
# this wiring exists to supply. Worse, what they DO return is distorted: IBIT's drawdown_z reads
# -2.01 against bitcoin's own -0.55, because the fund's short history contains little but the
# drawdown, so its "own" distribution is not a meaningful baseline.
#
# So for these, measure the cycle on the coin (10+ years of history) and LABEL it — `proxy_for` says
# the block describes the underlying, not the fund. Silently swapping would be worse than the gap.
_TREND_PROXY = {"BTC": "BTC-USD", "ETH": "ETH-USD"}


async def _long_term_block(sym: str, closes: list[float]) -> dict | None:
    """The multi-year value lens for a symbol, via its underlying when it is a spot-crypto ETF."""
    proxy = _TREND_PROXY.get(_exposure_group(sym))
    if proxy and sym.upper() != proxy:
        try:
            ref = await fetch_series(_http, proxy)
            lt = _compact_trend((await cycle.crypto_context(_http, proxy, ref.closes)).get("long_term_trend"))
            if lt:
                return {**lt, "proxy_for": proxy,
                        "note": f"long-cycle position measured on {proxy}; this fund is too young for a 200-week read"}
        except Exception:  # noqa: BLE001 — fall through to the fund's own (partial) numbers
            pass
    return _compact_trend((await cycle.crypto_context(_http, sym, closes)).get("long_term_trend"))


# The sandbox picks its own universe. It does NOT read the user's watchlist.
#
# The account is meant to be an independent test of whether this system can invest, and a universe
# borrowed from the user's watchlist quietly made it a test of something else: the sandbox could only
# ever choose among names the user had already chosen, so a good result was partly the user's stock
# picking and partly the model's, with no way to separate them. Both channels below are market data.
#
# _SANDBOX_CORE is the broad-market shelf a long-horizon plan is mostly built from, and it is curated
# rather than screened because no screener returns it. Yahoo's `top_etfs_us` was tried on 2026-08-14
# and ranks by recent performance — it returned GDX, XME, ESPO, RING: thematic and sector funds, the
# opposite of a core holding. Without this shelf the daily screen (actives, gainers, growth tech,
# undervalued large caps) surfaces essentially no broad index funds, which would leave the standing
# plan's US_EQUITY 25% / VXUS 12% / SCHD 12% targets — 49% of the book — permanently unfillable. That
# exact failure is on record: on 2026-08-07 an unreachable VTI target fired a blocked buy every day
# and let cash drift to 58.1%.
#
# So: a fixed shelf of vehicles, and a screen for individual names. The shelf is deliberately vehicles
# only, never single stocks — picking which COMPANIES to consider is the model's job, and a curated
# name here would be a thumb on that scale.
_SANDBOX_CORE = [
    "VTI", "VOO", "SPY", "QQQM", "IWM",   # US broad market / large cap / small cap
    "VXUS", "VEA", "VWO",                 # international developed + emerging
    "SCHD", "VYM",                        # dividend / quality tilt
    "GLD", "GLDM",                        # gold bullion — the preference filter below keeps one
    "FBTC", "IBIT",                       # spot-bitcoin ETFs (still gated by allow_crypto_etf)
]

# One budget per channel; they do not compete.
#
# 80 screened, from the WIDE screen set — the four momentum/value angles plus two contrarian ones and
# all eleven sector screens, so every sector is represented rather than only whichever ones moved.
# Neither obvious cost is what bounds this: candidate rows measure ~230 tokens, so a full pool is
# ~22k tokens on ONE Haiku call per day, and fetching 95 candidates was measured at ~7s against a
# 25-minute window before the close. What plausibly does degrade at this size is the model's ranking
# across a long list, and nothing here measures that — so this is a deliberate bet on coverage over
# an unmeasured concentration effect, and it is worth revisiting with the arms once they have data.
_SANDBOX_MAX_SCREENED = 80


async def _sandbox_candidate(sym: str, bench_closes, core_set: set[str]) -> dict | None:
    """A LIGHT candidate snapshot (core technicals only — no news/shorts/insider enrichment) so the
    daily tick stays fast. Returns None for unfetchable symbols."""
    async with _SANDBOX_CAND_SEM:
        crypto = sym.endswith("-USD")
        try:
            series = await fetch_series(_http, sym)
            summ = summarize(series, None if crypto else bench_closes)
        except Exception:  # noqa: BLE001 — an unfetchable candidate is just dropped
            return None
        row = {
            "symbol": sym, "source": "core" if sym in core_set else "market_screen",
            "price": round(summ["price"], 4), "exposure_group": _exposure_group(sym),
            **({"expense_ratio_pct": _expense_ratio(sym)} if _expense_ratio(sym) is not None else {}),
            "technicals": {k: summ.get(k) for k in _SANDBOX_TECH_KEYS if summ.get(k) is not None},
        }
        # Overnight gap + its MEASURED fill/edge base rate (app/gaps.py) — a mild tilt, not a trigger.
        g = gaps.compact(gaps.detect(series.closes, series.opens, series.volumes))
        if g:
            row["gap"] = g
        # The multi-year value lens (200-week line, Mayer, drawdown-z). Cached 6h per symbol inside
        # cycle.crypto_context, and best-effort — a name without enough weekly history simply has no
        # long_term block rather than blocking the tick.
        try:
            lt = await _long_term_block(sym, series.closes)
            if lt:
                row["long_term"] = lt
        except Exception:  # noqa: BLE001 — enrichment, never a blocker
            pass
        # What this setup has actually done before, benchmark-relative. The sandbox commits capital on
        # these decisions, so it is the one caller with real consequences riding on the base rate.
        try:
            track = memory.similar_setups(sym, summ)
            if track:
                row["track_record"] = track
        except Exception:  # noqa: BLE001 — enrichment, never a blocker
            pass
        return row


async def _sandbox_prices(held: list[str], candidate_syms: list[str]) -> dict[str, float | None]:
    """One batched live quote for everything, with a last-close fallback for HELD names missing a live
    price (we must be able to mark + trade positions even if the quote endpoint drops one)."""
    syms = list({*held, *candidate_syms, "^GSPC"})
    prices: dict[str, float | None] = {}
    try:
        quotes = await market_now.fetch_quotes(_http, syms)
    except Exception:  # noqa: BLE001
        quotes = {}
    for s in syms:
        # The price for the CURRENT session, not the 4pm close. With allow_after_hours enabled a tick
        # in the 16:00-20:00 window used to fill against regularMarketPrice, which is frozen at the
        # close — so an order would book at a price that was never available. On BLZE (2026-08-03)
        # that gap was 17%, and the ledger would have recorded the difference as a real gain.
        prices[s] = market_now.session_price(quotes.get(s) or {})
    for s in held:
        if not prices.get(s):
            try:
                prices[s] = summarize(await fetch_series(_http, s), None)["price"]
            except Exception:  # noqa: BLE001
                pass
    return prices


def _crypto_symbol(entry: str) -> str:
    """A crypto watchlist entry as a Yahoo symbol, whichever shape it was stored in.

    The stored list already carries the suffix (`BTC-USD`), so the tick's `f"{c.upper()}-USD"` built
    `BTC-USD-USD` — a symbol nothing can price. It stayed invisible because `allow_crypto` is off and
    the candidate filter drops everything ending in `-USD`, malformed or not: the same filter that hid
    the ghosts also swept them up. Turning that one setting on would have made every crypto candidate
    unfillable, with no error anywhere, because a dropped candidate looks exactly like a name the
    model chose not to buy.
    """
    return f"{str(entry or '').upper().removesuffix('-USD')}-USD"


def _exposure_vocabulary(symbols: Iterable[str]) -> dict[str, list[str]]:
    """The exposure groups reachable from a symbol list, each with the tickers that map into it.

    The strategist names its targets by group, and until this existed it had to INFER the group
    vocabulary from whatever labels happened to appear in the book. That is how the 2026-08-10 plan
    came to ask for `US_EQUITY` and `SP500` separately: it saw `US_EQUITY` on the VTI holding, knew
    the universe contained S&P funds it did not own, and coined a second label for them. Showing the
    groups and their members outright removes the guess — and makes the "these tickers are ONE
    exposure" rule concrete rather than an instruction to be taken on faith.
    """
    vocab: dict[str, list[str]] = {}
    for s in symbols:
        sym = str(s or "").upper().removesuffix("-USD")
        if not sym or sym.startswith("^"):
            continue
        vocab.setdefault(_exposure_group(sym), []).append(sym)
    return {g: sorted(set(members)) for g, members in sorted(vocab.items())}


async def _maybe_weekly_review(
    blob: dict, book: dict, settings: dict, *, tradable: Iterable[str] = (),
) -> bool:
    """Run the Opus weekly strategy review if it's due (>=7 days). Mutates blob's strategy note/date on
    success; on failure keeps the prior note and does NOT advance the cursor (retries next trading day)."""
    from datetime import date as _date
    last = blob.get("last_weekly_review_date")
    today = sandbox_job.now_et().date()
    due = last is None
    if not due:
        try:
            due = (today - _date.fromisoformat(last)).days >= 7
        except ValueError:
            due = True
    if not due:
        return False
    context = {
        "book": book,
        # The ONLY legal target labels, with their members. See _exposure_vocabulary.
        "exposure_groups": _exposure_vocabulary(
            [p.get("symbol") for p in (book.get("positions") or [])] + list(tradable)),
        "performance": {
            "funded_total": blob.get("funded_total"), "cash": blob.get("cash"),
            "realized_pl_total": blob.get("realized_pl_total"),
        },
    }
    # What the idle cash actually cost, in dollars, against the benchmark the account is measured on.
    # The stance discussion is otherwise abstract: "22% cash feels prudent" reads very differently
    # next to "the cash you held lost $193 while the picks made $6".
    try:
        funded = float(blob.get("funded_total") or 0.0)
        cash = float(blob.get("cash") or 0.0)
        equity = float(book.get("total_value") or 0.0)
        bench = float((blob.get("benchmark") or {}).get("shares") or 0.0) * (
            (await _sandbox_prices([], []))["^GSPC"] or 0.0)
        if funded > 0 and bench > 0 and equity > 0:
            bench_ret = bench / funded - 1.0
            context["cash_drag"] = {
                "cash_pct": round(cash / equity * 100, 1),
                "benchmark_return_pct": round(bench_ret * 100, 2),
                "shortfall_usd": round(bench - equity, 2),
                "idle_cash_opportunity_cost_usd": round(cash * bench_ret, 2),
                "note": "opportunity cost is the cash balance times the benchmark's return since "
                        "inception — when it accounts for most of the shortfall, the picks are not "
                        "the problem and a lower cash target is the lever",
            }
    except Exception:  # noqa: BLE001 — enrichment, never a blocker
        pass
    # The weekly review is the right altitude to react to "the way I've been picking isn't working" —
    # the daily tick is too close to the trade to reconsider its own method.
    try:
        st = memory.stats()
        card = {k: st[k] for k in ("buy_calls", "sandbox_buys") if k in st}
        if card:
            context["track_record"] = card
        # Which rules actually bound. A cap firing constantly means the plan keeps asking for
        # something the account forbids — the strategy is the right level to resolve that, not the
        # daily tick, which can only keep getting refused.
        blocked = memory.blocked_summary()
        if blocked:
            context["blocked_trades"] = blocked
        # The previous few weeks' stances, so the plan has continuity instead of re-deciding from
        # scratch every Monday with no memory of what it already tried.
        prior = memory.recent_notes(kind="strategy", limit=4)
        if prior:
            context["prior_strategy_notes"] = [n["body"][:400] for n in prior]
        # Your own last plans that did not add up. Without this the strategist repeats the same short
        # allocation every week and never learns that the remainder silently became cash.
        gaps = memory.recent_notes(kind="strategy_gap", limit=3)
        if gaps:
            context["prior_allocation_gaps"] = [n["body"][:400] for n in gaps]
    except Exception:  # noqa: BLE001 — enrichment, never a blocker
        pass
    # The exogenous backdrop (NEWS-4). The weekly review is the right altitude for it: a changed
    # world should move the STANCE and the cash target, while the daily tick only applies it to
    # individual names. Omitted entirely when unavailable — see macro.compact.
    try:
        mac = macro.compact(macro.load_state(), limit=5)
        if mac:
            context["macro"] = mac
    except Exception:  # noqa: BLE001 — enrichment, never a blocker
        pass
    try:
        note, usage = await strategy_review(
            context, settings=sandbox_job.settings_for_prompt(settings), deep=True)
        usage_store.record(usage, symbol="SANDBOX", kind="sandbox_strategy")
        # Resolve the targets onto today's exposure groups BEFORE anything stores or reads them, so a
        # plan can never carry two labels for one group past this line. One dict from here on: the
        # note that gets stored must be the same object that gets audited and remembered.
        d = note.model_dump()
        renamed = sandbox_job.canonicalize_targets(d, group_of=_exposure_group)
        if renamed:
            _log.info("sandbox strategy targets canonicalised: %s", ", ".join(renamed))
        blob["last_strategy_note"] = d
        blob["last_weekly_review_date"] = today.isoformat()
        # Keep the weekly reads searchable. Without this each review overwrites the last and the
        # strategy's own history — what it believed and when — is lost.
        memory.add_note(
            "strategy",
            f"[{today.isoformat()}] stance={d.get('stance')} cash_target={d.get('cash_target_pct')}%\n"
            f"{d.get('note') or d.get('summary') or ''}",
            meta={"date": today.isoformat(), "stance": d.get("stance")},
        )
        # Audit the plan the strategist just wrote. A short plan cannot be fixed here — normalising
        # the targets upward would push them through the per-group cap and produce orders the tick
        # can never fill — so it is recorded instead, and fed to the NEXT review, which is the only
        # stage that can add groups or honestly raise the cash target.
        gap = sandbox_job.allocation_gap(
            d, max_position_pct=float(settings.get("max_position_pct", 25.0)),
            group_of=_exposure_group)
        if gap:
            _log.warning("sandbox strategy plan is short: %s", gap)
            memory.add_note(
                "strategy_gap",
                f"[{today.isoformat()}] targets sum to {gap['targets_sum_pct']}% against an investable "
                f"{gap['investable_pct']}% ({gap['cash_target_pct']}% cash target) — "
                f"{gap['unallocated_pct']}% left with no owner, which becomes idle cash. "
                f"{gap['groups']} group(s) named, at least {gap['groups_needed']} needed to cover the "
                f"invested share under the {settings.get('max_position_pct')}% per-group cap."
                + (f" Targets above the cap (unreachable): {gap['targets_over_cap']}."
                   if gap["targets_over_cap"] else ""),
                meta={"date": today.isoformat(), **gap},
            )
        return True
    except Exception as e:  # noqa: BLE001 — a failed Opus review must not break the daily tick
        _log.warning("sandbox weekly review failed (keeping prior note): %s", e)
        return False


def _extension_lookup(book: dict) -> "Callable[[str], float | None]":
    """symbol -> mayer_multiple, read off the book snapshot the tick already built.

    Injected into validate_and_fill rather than fetched there, so that module stays pure and
    network-free. Returns None for anything unmeasured -- a spot-crypto ETF too young for a 200-week
    read, or a name whose long-term block failed to build -- and the guard treats None as "do not
    judge" rather than as "not extended". Blocking a sell on missing data would be the wrong
    direction: absent is not a measurement.
    """
    ext = {}
    for p in (book or {}).get("positions") or []:
        lt = p.get("long_term") or {}
        m = lt.get("mayer_multiple")
        if m is not None:
            try:
                ext[str(p.get("symbol", "")).upper()] = float(m)
            except (TypeError, ValueError):
                pass
    return lambda sym: ext.get(str(sym).upper())


async def _run_extra_arm(
    arm: str, *, now, price_of, spy_price: float | None, shared_plan: dict | None,
    candidates: list[dict], macro_block, force: bool, held_rows: list[dict] | None = None,
    rejected: dict[str, list[dict]] | None = None,
    gate: dict | None = None,
) -> dict:
    """One decision cycle for a NON-main arm, against the market snapshot main already fetched.

    Same quotes, same tick, same day — that identity is the whole reason the arms are comparable, and
    it is why this takes the price map as an argument instead of fetching its own. Two arms priced
    from two fetches minutes apart would differ by the market as well as by the strategy, and the
    experiment would be measuring the weather.

    Extra arms INHERIT main's standing plan rather than running their own weekly review. The question
    these are here to answer is about execution — does the daily analyst beat mechanically filling the
    plan, does a higher per-tick cap deploy faster — so the plan has to be the constant. It is also
    the difference between one Opus call a week and one per arm per week.

    Failures are contained: an arm that raises is reported and skipped, never allowed to take down
    main's tick, which is the account that actually matters."""
    blob = sandbox_store.get(arm)
    engine = blob.get("engine", "llm")
    label = blob.get("label") or arm
    proceed, status = sandbox_job.tick_gate(blob, now=now, force=force)
    if not proceed:
        return {"arm": arm, "label": label, "engine": engine, "status": status}

    settings = blob["settings"]
    warnings: list[str] = []
    exclude = {s.upper() for s in (settings.get("exclusions") or [])}

    earned = sandbox_job.accrue_cash_interest(blob, now=now)
    if earned > 0:
        sandbox_store.append_trade({
            "ts": time.time(), "date": sandbox_job.today_et_str(now), "symbol": "CASH",
            "side": "interest", "status": "filled", "shares": 0.0, "price": None,
            "gross": earned, "cash_after": round(blob["cash"], 2), "source": "cash_yield",
            "reason": f"Interest on idle cash at {settings.get('cash_apy_pct')}% APY"}, arm)

    dep = float(settings.get("monthly_deposit") or 0.0)
    month = now.strftime("%Y-%m")
    if dep > 0 and blob.get("last_deposit_month") != month:
        if not spy_price:
            warnings.append("monthly deposit deferred — no benchmark quote; will retry next tick")
        else:
            blob["benchmark"]["shares"] = round(blob["benchmark"]["shares"] + dep / spy_price, 6)
            blob["benchmark"]["cost_basis"] = round(blob["benchmark"]["cost_basis"] + dep, 2)
            blob["cash"] = round(blob["cash"] + dep, 2)
            blob["funded_total"] = round(blob["funded_total"] + dep, 2)
            blob["last_deposit_month"] = month
            sandbox_store.append_trade({
                "ts": time.time(), "date": sandbox_job.today_et_str(now), "symbol": "CASH",
                "side": "deposit", "status": "filled", "shares": 0.0, "price": None,
                "gross": round(dep, 2), "cash_after": blob["cash"], "source": "recurring",
                "reason": f"Recurring monthly deposit ${dep:,.0f}"}, arm)

    # Bound before the engine fork: only the llm branch can populate it, but every branch reaches
    # the trade-log write below, and a name defined in one arm of a fork is not defined in the others.
    _review_skips: list[dict] = []
    # Same reason. Also stays empty for the rules and rejects engines by design: they consult no
    # candidates at all, so recording main's pool under their name would log an input they never got.
    arm_candidates: list[dict] = []
    flat = sandbox_job.exit_date_flatten_orders(blob, price_of)
    if flat is not None:
        orders, source, posture = flat, "exit_date", "Exit date reached — flattening to cash."
    elif engine == "rules":
        d = sandbox_job.rules_decision(
            blob, plan=shared_plan, group_of=_exposure_group, price_of=price_of)
        orders, source, posture = d["orders"], "rules_tick", d["posture"]
    elif engine == "rejects":
        # The review model's control group. Takes exactly what the reviewer refused on its source
        # arm, so the two books differ by the review verdict and nothing else. No model call: the
        # orders were written by the source arm's analyst and this arm only executes them.
        src = str(settings.get("rejects_from") or "").strip().lower()
        orders = list((rejected or {}).get(src) or [])
        source = "rejects_tick"
        posture = (f"Taking the {len(orders)} order(s) the review model rejected on '{src}'."
                   if orders else
                   f"Nothing rejected on '{src}' today." if src else
                   "No source arm configured (settings.rejects_from is unset).")
    else:
        holdings = [Holding(symbol=p["symbol"], shares=p["shares"], avg_cost=p["avg_cost"])
                    for p in blob["positions"]]
        # This arm's universe: the pool main assembled, plus the names main holds that this arm does
        # not, minus whatever this arm holds itself (already in its positions block) and its own
        # exclusions. Without this the arm inherits main's blind spots — see
        # sandbox_job.candidates_for_arm.
        arm_candidates = sandbox_job.candidates_for_arm(
            candidates, held_rows or [],
            held=[p["symbol"] for p in blob["positions"]], exclusions=exclude)
        try:
            book = (await _build_portfolio_snapshot(holdings, blob["cash"], include_trend=True)
                    if holdings else
                    {"total_value": blob["cash"], "cash_pct": 100.0, "positions": []})
            if holdings and settings.get("taxable_account", True):
                sandbox_job.annotate_holding_period(book.get("positions", []), blob["positions"])
            # Per-arm backbone. None = the service's configured scan model, so an arm that does not
            # set one is a true copy of main rather than a second variable.
            decision, usage = await sandbox_decision(
                book, arm_candidates, cash=blob["cash"],
                settings=sandbox_job.settings_for_prompt(settings),
                strategy_note=shared_plan, macro=macro_block, deep=False,
                model=(str(settings.get("model")).strip() or None) if settings.get("model") else None,
                gaps=sandbox_job.target_gaps(
                    blob["positions"], equity=book.get("total_value") or 0.0,
                    plan=shared_plan, group_of=_exposure_group, price_of=price_of),
                # This arm's own ledger. Read per-arm, never shared: an arm that is a control for
                # another must not be told what that other one did.
                recent_activity=sandbox_job.recent_activity(
                    sandbox_store.read_trades(120, arm), today=sandbox_job.today_et_str()))
            usage_store.record(usage, symbol=f"SANDBOX:{arm}", kind="sandbox_tick")
            _arm_orders = [o.model_dump() for o in decision.orders]
            # Review on the arm path too. This was implemented on main only, so review_enabled on an
            # arm silently did nothing — a setting that reads as on and changes no behaviour is worse
            # than one that is off, because the experiment it configures looks like it ran.
            if settings.get("review_enabled") and _arm_orders:
                try:
                    verdict, rusage = await review_decision(
                        decision, book, arm_candidates, cash=blob["cash"],
                        settings=sandbox_job.settings_for_prompt(settings),
                        strategy_note=shared_plan, deep=True)
                    usage_store.record(rusage, symbol=f"SANDBOX:{arm}", kind="sandbox_review")
                    _v = verdict.model_dump()
                    kept, dropped_syms = sandbox_job.apply_review(_arm_orders, _v)
                    blob["last_review"] = {
                        "date": sandbox_job.today_et_str(now), "approve": _v.get("approve", True),
                        "dropped": dropped_syms, "concerns": _v.get("concerns") or [],
                        "note": _v.get("note") or "",
                    }
                    if dropped_syms:
                        _review_skips = sandbox_job.review_skip_rows(
                            [o for o in _arm_orders
                             if str(o.get("symbol", "")).upper() in set(dropped_syms)],
                            _v, now_ts=time.time())
                    if dropped_syms and rejected is not None:
                        # Handed to any arm configured to take them. The rejected ORDERS, not just
                        # their symbols: the control arm has to execute the same trade that was
                        # refused, at the same size and zone, or it is testing something else.
                        rejected[arm] = [o for o in _arm_orders
                                         if str(o.get("symbol", "")).upper() in set(dropped_syms)]
                    if dropped_syms:
                        warnings.append(f"review dropped {len(dropped_syms)} order(s): "
                                        f"{', '.join(dropped_syms)}")
                    _arm_orders = kept
                except Exception as e:  # noqa: BLE001
                    warnings.append(f"review model unavailable, orders NOT reviewed: {e}")
            # _arm_orders, not decision.orders: the reviewed list is the one that trades. Reading
            # decision.orders here would run the review, log its verdict, and then send the original
            # orders anyway -- the most expensive possible way to change nothing.
            orders, posture = _arm_orders, decision.posture
            source = "haiku_tick"
            blob["last_decision_date"] = sandbox_job.today_et_str(now)
        except Exception as e:  # noqa: BLE001 — a failed decision = no trades, still mark NAV
            warnings.append(f"decision failed: {e}")
            orders, source, posture = [], "haiku_tick", "No decision (analyst unavailable) — held."

    try:
        new_blob, filled, skipped = sandbox_job.validate_and_fill(
            blob, orders, price_of, group_of=_exposure_group, source=source, exclude=exclude,
            liquidation=(source == "exit_date"),
            # Every arm on the tick sees the SAME verdict, for the same reason they all see the same
            # quotes: an arm that gated on its own evaluation seconds later would differ from its
            # control by the feed as well as by the setting.
            gate=gate,
            extension_of=_extension_lookup(locals().get("book")))
    except AssertionError as e:
        # Main is mid-tick and already persisted. Aborting the whole request over a side arm would
        # throw away a completed real tick, so this arm alone is skipped and says why.
        _log.error("sandbox arm %s ABORTED (cash not conserved): %s", arm, e)
        return {"arm": arm, "label": label, "engine": engine, "status": "aborted",
                "warnings": [f"cash not conserved: {e}"]}

    stale = sandbox_job.stale_marks(new_blob["positions"], price_of)
    if stale:
        warnings.append(f"no fresh quote for {', '.join(stale)} — valued at last known mark")

    pv = sandbox_job.positions_value(new_blob["positions"], price_of)
    nav = sandbox_job.nav_row(new_blob, positions_val=pv, spy_price=spy_price)
    new_blob["last_tick_date"] = sandbox_job.today_et_str(now)
    new_blob["last_posture"] = posture or ""     # see the main path; arms need it for the same reason
    sandbox_store.save(new_blob, arm)
    for r in filled + skipped + _review_skips:
        sandbox_store.append_trade(r, arm)
    # What THIS arm was shown, not what main was shown.
    if arm_candidates:
        sandbox_store.append_inputs({
            "ts": time.time(), "date": sandbox_job.today_et_str(now), "arm": arm,
            "candidates": sandbox_job.candidate_fingerprint(arm_candidates),
        }, arm)
    sandbox_store.append_nav(nav, arm)
    return {"arm": arm, "label": label, "engine": engine, "status": "ok", "posture": posture,
            "orders_filled": filled, "orders_skipped": skipped, "nav": nav, "warnings": warnings}


async def run_sandbox_tick(*, force: bool = False, manual: bool = False) -> dict:
    """One decision cycle: gate → price → (weekly review) → decide → validate+fill → log → NAV → persist."""
    assert _http is not None
    async with _sandbox_lock:
        blob = sandbox_store.get()
        now = sandbox_job.now_et()

        # Heal a standing plan that names one exposure group twice — ABOVE the gate, because this
        # does not depend on market state and the gate is exactly what would delay it. Notes written
        # from now on are canonicalised at store time, but a plan lasts a WEEK: today's already-stored
        # `US_EQUITY 22% + SP500 18%` would otherwise steer seven days of ticks toward a 40% target it
        # can never fill, and the first tick that could fix it is the one the gate turns away.
        # Idempotent, so a clean note costs nothing and writes nothing.
        _renamed = sandbox_job.canonicalize_targets(
            blob.get("last_strategy_note"), group_of=_exposure_group)
        if _renamed:
            _log.info("standing strategy plan canonicalised: %s", ", ".join(_renamed))
            sandbox_store.save(blob)

        proceed, status = sandbox_job.tick_gate(blob, now=now, force=force)
        if not proceed:
            return {"status": status, "date": sandbox_job.today_et_str(now)}

        # SWT-2's market regime gate, evaluated ONCE for the whole tick and handed to every arm.
        #
        # Evaluated only when some arm actually wants it. `gate_block_reason` returns None the moment
        # it sees `gate_enabled` falsy, so with the setting off everywhere — which is the shipped
        # default on all five live arms — this verdict would be read by nobody, and three HTTP
        # fetches a tick to compute an answer no arm consults is a cost with no reader.
        #
        # It is evaluated ONCE rather than per arm for the same reason `price_of` is fetched once:
        # arms are only comparable if the tick is identical across them. An arm that ran its own
        # evaluation a few seconds later could stand aside while its control bought, and the
        # difference between their curves would be the feed rather than the setting under test.
        #
        # A failure here leaves the verdict None, which a gated arm reads as "could not be measured"
        # and blocks on — deliberately, and distinguishably from "the market failed". It must never
        # raise: the gate is off everywhere by default, and an unproven experiment has no business
        # being able to take down the real account's tick.
        gate_verdict: dict | None = None
        if sandbox_job.any_arm_wants_gate(
                sandbox_store.get(a).get("settings", {}) for a in sandbox_store.list_arms()):
            try:
                gate_verdict = await gate.evaluate(_http)
            except Exception:  # noqa: BLE001 — a gated arm blocks on the resulting None and says
                # "could not be measured", which is the honest reading of a failed evaluation.
                _log.warning("sandbox tick: regime gate evaluation failed", exc_info=True)

        settings = blob["settings"]
        cfg = settings_store.get()

        warnings: list[str] = []

        # Interest on idle cash, BEFORE the deposit and the decision so the day's balance is right.
        # No benchmark leg: interest is earned by cash the benchmark never holds (it is 100% invested
        # by construction), so crediting it here is what makes the two comparable rather than
        # penalising the sandbox twice for the same choice.
        earned = sandbox_job.accrue_cash_interest(blob, now=now)
        if earned > 0:
            # One row per accrual keeps it auditable — the trade log is the ledger's only human-
            # readable history, and silently growing cash is exactly the kind of thing that should
            # never just appear.
            sandbox_store.append_trade({
                "ts": time.time(), "date": sandbox_job.today_et_str(now), "symbol": "CASH",
                "side": "interest", "status": "filled", "shares": 0.0, "price": None,
                "gross": earned, "cash_after": round(blob["cash"], 2), "source": "cash_yield",
                "reason": f"Interest on idle cash at {settings.get('cash_apy_pct')}% APY",
            })

        # Recurring monthly deposit (DCA) — added once per ET calendar month, BEFORE the decision so the
        # AI can deploy it; the benchmark shadow gets the same cash on the same day.
        dep = float(settings.get("monthly_deposit") or 0.0)
        month = now.strftime("%Y-%m")
        if dep > 0 and blob.get("last_deposit_month") != month:
            try:
                dspy = (await market_now.fetch_quotes(_http, ["^GSPC"])).get("^GSPC", {}).get("price")
            except Exception:  # noqa: BLE001
                dspy = None
            if not dspy:
                # Skip the whole deposit rather than credit cash with no benchmark leg. Crucially the
                # month cursor is NOT advanced, so it retries tomorrow instead of silently losing the
                # month — the old code burned the cursor either way and left the shadow permanently
                # short by one deposit, flattering every return figure measured against it.
                warnings.append("monthly deposit deferred — no benchmark quote; will retry next tick")
            else:
                blob["benchmark"]["shares"] = round(blob["benchmark"]["shares"] + dep / dspy, 6)
                blob["benchmark"]["cost_basis"] = round(blob["benchmark"]["cost_basis"] + dep, 2)
                blob["cash"] = round(blob["cash"] + dep, 2)
                blob["funded_total"] = round(blob["funded_total"] + dep, 2)
                blob["last_deposit_month"] = month
                sandbox_store.append_trade({
                    "ts": time.time(), "date": sandbox_job.today_et_str(now), "symbol": "CASH",
                    "side": "deposit", "status": "filled", "shares": 0.0, "price": None,
                    "gross": round(dep, 2), "cash_after": blob["cash"], "source": "recurring",
                    "reason": f"Recurring monthly deposit ${dep:,.0f}"})

        # Cadence: "weekly" decides at most every 7 days (NAV still marks daily); "daily" always decides.
        from datetime import date as _date
        decide = True
        if (settings.get("cadence") or "daily").lower() == "weekly":
            ld = blob.get("last_decision_date")
            if ld:
                try:
                    decide = (now.date() - _date.fromisoformat(ld)).days >= 7
                except ValueError:
                    decide = True
        exclude = {s.upper() for s in (settings.get("exclusions") or [])}

        held = [p["symbol"].upper() for p in blob["positions"]]
        # NOT cfg["watchlist"]. The sandbox's universe is its own — see _SANDBOX_CORE.
        #
        # The shelf is entirely ETFs, so `allow_etf` empties it outright. That setting had been DEAD
        # since it was added — sandbox_job read it into a local and never used it — which mattered
        # little while the universe was the user's watchlist and matters a lot now that switching it
        # off would otherwise still serve nine index funds a day.
        core = list(_SANDBOX_CORE) if settings.get("allow_etf", True) else []
        # One bitcoin ETF, the one the user picked. Every vehicle in the BTC group holds the same
        # asset, so offering all of them is not a choice between exposures — it is an invitation to
        # fragment one position across three tickers. prefer_btc_etf already reroutes a stray BUY,
        # but a candidate slot spent on a vehicle that will be redirected anyway is a wasted slot
        # and a confusing line of reasoning in the trade log.
        # Same for gold: GLD and GLDM hold the same bullion at 0.400% and 0.100%, so offering both
        # is not a choice between exposures, it is an invitation to fragment one position across two
        # tickers — and a candidate slot spent on a vehicle prefer_gold_etf will redirect anyway.
        for _family, _key in ((sandbox_job.BTC_ETFS, "preferred_btc_etf"),
                              (sandbox_job.GOLD_ETFS, "preferred_gold_etf")):
            _pref = str(settings.get(_key) or "").strip().upper()
            if _pref in _family:
                core = [s for s in core if s not in _family or s == _pref]
        try:
            # Ask for more than the budget: the vehicle and exclusion filters below run AFTER this
            # cap, so requesting exactly the budget lets a few filtered names shrink the screen's
            # slice below its size. The screeners are already fetched in full, so asking wide is free.
            discovered = [s.upper() for s in await discover(
                _http, set(held) | set(_SANDBOX_CORE),
                cap=_SANDBOX_MAX_SCREENED + 10, screens=WIDE_SCREENS,
                allow_etf=bool(settings.get("allow_etf", True)),
                min_market_cap=float(settings.get("min_market_cap", 2_000_000_000.0) or 0.0))]
        except Exception:  # noqa: BLE001
            discovered = []
        core_set = set(core)
        candidate_syms, dropped_syms = sandbox_job.select_candidates(
            watchlist=core, discovered=discovered, held=held, exclusions=exclude,
            allow_crypto=bool(settings.get("allow_crypto", False)),
            allow_crypto_etf=bool(settings.get("allow_crypto_etf", True)),
            group_of=_exposure_group,
            max_watchlist=len(core), max_discovered=_SANDBOX_MAX_SCREENED)
        if dropped_syms:
            # Never truncate in silence. A shortened list looks exactly like a short list from the
            # inside, and the names that fall off the end are the ones nobody is watching for.
            warnings.append(
                f"universe truncated — {len(dropped_syms)} symbol(s) were not considered: "
                f"{', '.join(dropped_syms)}")

        # A preferred vehicle must always be priced, even when it didn't make the candidate cut —
        # prefer_vehicle can only route a buy onto something it can fill, so without a price the
        # preference silently does nothing.
        pref_etf = str(settings.get("preferred_btc_etf") or "").strip().upper()
        pref_gold = str(settings.get("preferred_gold_etf") or "").strip().upper()
        price_syms = list(candidate_syms)
        if (pref_etf in sandbox_job.BTC_ETFS and settings.get("allow_crypto_etf", True)
                and pref_etf not in price_syms and pref_etf not in held):
            price_syms.append(pref_etf)
        # No allow_* gate on gold: it is an ordinary ETF, covered by allow_etf like the rest of the
        # shelf, and `core` is already empty when that is off.
        if (pref_gold in sandbox_job.GOLD_ETFS and settings.get("allow_etf", True)
                and pref_gold not in price_syms and pref_gold not in held):
            price_syms.append(pref_gold)

        # Every group the STANDING PLAN names must be priceable, whether or not it made the candidate
        # cut. A target is an instruction to hold something, and a group that cannot be priced cannot
        # be bought — so an unpriced target is a plan line that silently does nothing, forever. This
        # also feeds the extra arms, which have no candidate pipeline of their own and must be able to
        # act on the same plan against the same quotes.
        plan_syms: list[str] = []
        for _t in ((blob.get("last_strategy_note") or {}).get("targets") or []):
            _g = _exposure_group(str(_t.get("exposure_group") or ""))
            # "assume priceable" — we are deciding WHAT to fetch, so the real price map does not
            # exist yet and a truthful price_of here would reject every candidate representative.
            _rep = sandbox_job.group_representative(
                _g, positions=blob["positions"], price_of=lambda _s: 1.0, group_of=_exposure_group,
                preferred_btc_etf=pref_etf or "FBTC", preferred_gold_etf=pref_gold or "GLDM")
            if _rep and _rep not in plan_syms:
                plan_syms.append(_rep)
        # Arms other than main hold their own book; those symbols need marks too or their NAV is
        # computed from stale prices and the curves stop being comparable.
        arm_held: list[str] = []
        for _a in sandbox_store.list_arms():
            if _a == sandbox_store.MAIN_ARM:
                continue
            for _p in sandbox_store.get(_a).get("positions") or []:
                _s = _p["symbol"].upper()
                if _s not in arm_held:
                    arm_held.append(_s)
        for _s in plan_syms + arm_held:
            if _s not in price_syms and _s not in held:
                price_syms.append(_s)

        prices = await _sandbox_prices(held, price_syms)
        def price_of(sym: str):
            return prices.get(sym.upper())
        spy_price = prices.get("^GSPC")

        # Drop candidates one share of which costs more than the position cap could ever buy. This
        # runs HERE rather than at selection because it needs live prices, which only exist now.
        # Logged, not warned: unlike a truncation this removes nothing the account could have used,
        # and a daily warning naming the same structurally-impossible ticker is noise that would
        # teach the reader to skim the warnings that do matter.
        _equity = blob["cash"] + sandbox_job.positions_value(blob["positions"], price_of)
        _too_dear = sandbox_job.unaffordable(
            candidate_syms, price_of=price_of, equity=_equity,
            max_position_pct=float(settings.get("max_position_pct", 20.0)))
        if _too_dear:
            _log.info("sandbox: dropping unaffordable candidates %s (one share exceeds the %.0f%% "
                      "cap on $%.0f equity)", ", ".join(_too_dear),
                      float(settings.get("max_position_pct", 20.0)), _equity)
            candidate_syms = [s for s in candidate_syms if s not in set(_too_dear)]

        weekly_ran = False
        posture = ""
        # Bound on every path — the exit-date and weekly-cadence branches below skip the decision
        # entirely, and the fill recorder after validate_and_fill still reads this. `macro_block` is
        # bound here for the same reason: the extra arms read it after this block on every path.
        candidates: list[dict] = []
        # Same reason as `candidates` and `macro_block`: the extra arms read it after this block on
        # every path, including the ones that skip the decision entirely.
        held_rows: list[dict] = []
        macro_block = None
        # Orders the review model refused, per arm, for this tick only. Consumed by any engine
        # "rejects" arm further down. Never persisted: a rejected order is a statement about today's
        # decision, and carrying it forward would execute a trade nobody re-proposed.
        rejected_by_arm: dict[str, list[dict]] = {}

        flat = sandbox_job.exit_date_flatten_orders(blob, price_of)
        if flat is not None:
            orders, source = flat, "exit_date"
            posture = "Exit date reached — flattening to cash."
        elif not decide:
            orders, source = [], "haiku_tick"
            posture = "Weekly cadence — holding until the next scheduled decision."
        else:
            holdings = [Holding(symbol=p["symbol"], shares=p["shares"], avg_cost=p["avg_cost"])
                        for p in blob["positions"]]
            if holdings:
                try:
                    book = await _build_portfolio_snapshot(
                        holdings, blob["cash"], include_trend=True)
                    # Holding period per position, so a trim can weigh short- vs long-term capital
                    # gains. Skipped entirely in a tax-advantaged account, where it is noise.
                    if settings.get("taxable_account", True):
                        sandbox_job.annotate_holding_period(
                            book.get("positions", []), blob["positions"])
                except Exception as e:  # noqa: BLE001
                    # NEVER claim 100% cash while the ledger holds shares. That substitution told the
                    # weekly strategy review the account was entirely in cash — and that note then
                    # persists for SEVEN DAYS, steering every daily tick under it. Mark the real
                    # positions at cost instead: a rough denominator beats a false one.
                    warnings.append(f"book snapshot failed, positions marked at cost: {e}")
                    at_cost = [{"symbol": h.symbol.upper(), "exposure_group": _exposure_group(h.symbol),
                                "shares": h.shares, "avg_cost": h.avg_cost,
                                "value": round((h.avg_cost or 0.0) * h.shares, 2),
                                "technicals": {}} for h in holdings]
                    tv = sum(r["value"] for r in at_cost) + max(blob["cash"], 0.0)
                    for r in at_cost:
                        r["weight_pct"] = round(100.0 * r["value"] / tv, 1) if tv else None
                    book = {"total_value": round(tv, 2), "cash": round(blob["cash"], 2),
                            "cash_pct": round(100.0 * max(blob["cash"], 0.0) / tv, 1) if tv else 0.0,
                            "positions": at_cost, "priced_at_cost": True}
            else:
                book = {"total_value": blob["cash"], "cash_pct": 100.0, "positions": []}
            weekly_ran = await _maybe_weekly_review(
                blob, book, settings, tradable=candidate_syms)
            try:
                bench_closes = (await fetch_series(_http, "^GSPC")).closes
            except Exception:  # noqa: BLE001
                bench_closes = None
            candidates = [c for c in await asyncio.gather(
                *[_sandbox_candidate(s, bench_closes, core_set) for s in candidate_syms]) if c]
            # Rows for the names MAIN holds. `select_candidates` strips them from the pool above,
            # correctly — main already sees them in its own positions block — but the pool is then
            # handed to every comparison arm, and an arm that does not hold them has no row and no
            # price for them anywhere. See sandbox_job.candidates_for_arm for what that cost.
            # Fetched only when a comparison arm exists, and only for what is missing.
            _arm_ids = [a for a in sandbox_store.list_arms() if a != sandbox_store.MAIN_ARM]
            _missing = [s for s in held if s not in set(candidate_syms)] if _arm_ids else []
            held_rows = [c for c in await asyncio.gather(
                *[_sandbox_candidate(s, bench_closes, core_set) for s in _missing]) if c]
            source = "haiku_tick"
            # Exogenous risk overlay (NEWS-4). None when there is no usable read, which the prompt
            # is told to treat as "backdrop unknown" rather than "backdrop clear".
            try:
                macro_block = macro.compact(macro.load_state())
            except Exception:  # noqa: BLE001 — enrichment, never a blocker
                macro_block = None
            try:
                decision, usage = await sandbox_decision(
                    book, candidates, cash=blob["cash"],
                    settings=sandbox_job.settings_for_prompt(settings),
                    strategy_note=blob.get("last_strategy_note"), macro=macro_block, deep=False,
                    gaps=sandbox_job.target_gaps(
                        blob["positions"], equity=book.get("total_value") or 0.0,
                        plan=blob.get("last_strategy_note"), group_of=_exposure_group,
                        price_of=price_of),
                    recent_activity=sandbox_job.recent_activity(
                        sandbox_store.read_trades(120, sandbox_store.MAIN_ARM),
                        today=sandbox_job.today_et_str()))
                usage_store.record(usage, symbol="SANDBOX", kind="sandbox_tick")
                orders = [o.model_dump() for o in decision.orders]
                posture = decision.posture

                # Second opinion, when enabled and when there is something to review. A deeper model
                # reads the ORDERS before the ledger does. Skipped entirely on an empty list: there
                # is no argument to check in a decision to hold, and paying for a deep call to be
                # told so is the over-verification Anthropic's own Opus 5 guidance warns against.
                if settings.get("review_enabled") and orders:
                    try:
                        verdict, rusage = await review_decision(
                            decision, book, candidates, cash=blob["cash"],
                            settings=sandbox_job.settings_for_prompt(settings),
                            strategy_note=blob.get("last_strategy_note"), deep=True)
                        usage_store.record(rusage, symbol="SANDBOX", kind="sandbox_review")
                        _pre = list(orders)
                        _v = verdict.model_dump()
                        orders, dropped = sandbox_job.apply_review(orders, _v)
                        # Persisted whatever the outcome, so an APPROVED review is distinguishable
                        # from one that never ran. Both currently look like a tick with no rejection.
                        new_blob["last_review"] = {
                            "date": sandbox_job.today_et_str(now), "approve": _v.get("approve", True),
                            "dropped": dropped, "concerns": _v.get("concerns") or [],
                            "note": _v.get("note") or "",
                        }
                        if dropped:
                            _dropped_orders = [o for o in _pre
                                               if str(o.get("symbol", "")).upper() in set(dropped)]
                            rejected_by_arm[sandbox_store.MAIN_ARM] = _dropped_orders
                            # A dropped order never reaches validate_and_fill, so nothing else would
                            # write it a row -- and a rejected order that leaves no trace is
                            # indistinguishable in the log from one that was never proposed.
                            skipped.extend(sandbox_job.review_skip_rows(
                                _dropped_orders, _v, now_ts=time.time()))
                        if dropped:
                            warnings.append(
                                f"review dropped {len(dropped)} order(s): {', '.join(dropped)}"
                                + (f" — {'; '.join(verdict.concerns[:2])}" if verdict.concerns else ""))
                            posture = f"{posture} [reviewed: {verdict.note or 'orders dropped'}]"
                    except Exception as e:  # noqa: BLE001
                        # A failed review must not become a failed tick. It is an ADDITIONAL check,
                        # so its absence returns the account to the behaviour it had yesterday --
                        # whereas blocking here would let a rate limit halt trading entirely. Said
                        # out loud, because a silently unreviewed tick looks exactly like a reviewed
                        # one that found nothing.
                        warnings.append(f"review model unavailable, orders NOT reviewed: {e}")
                # Advanced only once a decision actually happened. Setting it at the TOP of this
                # branch meant a single transient LLM failure consumed the cadence cursor: on
                # "weekly" the account then held for another seven days having made no decision at
                # all, and the log said "holding until the next scheduled decision" as though that
                # were intentional.
                blob["last_decision_date"] = sandbox_job.today_et_str(now)
            except Exception as e:  # noqa: BLE001 — a failed decision = no trades, still mark NAV
                warnings.append(f"decision failed: {e}")
                orders, posture = [], "No decision (analyst unavailable) — held."

        try:
            new_blob, filled, skipped = sandbox_job.validate_and_fill(
                blob, orders, price_of, group_of=_exposure_group, source=source, exclude=exclude,
                gate=gate_verdict,
                extension_of=_extension_lookup(locals().get("book")),
                # An exit-date flatten is the user's scheduled instruction, not churn, so the
                # anti-churn caps must not silently leave the account still invested on that date.
                liquidation=(source == "exit_date"))
        except AssertionError as e:
            _log.error("sandbox tick ABORTED (cash not conserved): %s", e)
            raise HTTPException(status_code=500, detail=f"sandbox tick aborted (cash not conserved): {redact.redact(e)}")

        if source == "exit_date":
            left = [p["symbol"] for p in new_blob["positions"] if float(p.get("shares") or 0) > 0]
            if left:
                # The posture says "flattening to cash" — say so plainly when it did not finish
                # rather than letting the claim stand unqualified.
                warnings.append(f"exit-date flatten incomplete — still holding {', '.join(left)}")
                posture = f"Exit date reached — flattened what could be sold; still holding {', '.join(left)}."
        # A holding valued from a stale mark is a real caveat about the NAV point about to be
        # written, so surface it instead of letting an approximation look like a measurement.
        stale = sandbox_job.stale_marks(new_blob["positions"], price_of)
        if stale:
            warnings.append(f"no fresh quote for {', '.join(stale)} — valued at last known mark")

        # The LEDGER is the source of truth and it is saved FIRST. The append-only logs used to be
        # written before it, so a crash (or the process being killed) between them left
        # sandbox_trades.jsonl and sandbox_nav.jsonl asserting trades and an equity point that the
        # ledger never recorded — and being append-only, nothing can retract them. Saving first means
        # the worst case is a fill that happened but wasn't logged, which is recoverable from the
        # ledger; the reverse is not.
        pv = sandbox_job.positions_value(new_blob["positions"], price_of)
        nav = sandbox_job.nav_row(new_blob, positions_val=pv, spy_price=spy_price)
        new_blob["last_tick_date"] = sandbox_job.today_et_str(now)
        # The posture is the only record of WHY a tick did what it did, and until now it was returned
        # to the caller and then dropped. That made a no-trade day unexplainable after the fact: the
        # trade log shows the orders that existed, so a tick that proposed nothing left nothing at
        # all behind, and "the model declined" was indistinguishable from "the model was blocked"
        # without re-running it. Saved with last_tick_date in the same write, so the two cannot drift.
        new_blob["last_posture"] = posture or ""
        sandbox_store.save(new_blob)
        # What the model was SHOWN, recorded next to what it decided. Written only when a decision
        # actually happened -- a weekly-cadence hold or an exit-date flatten consults no candidates,
        # and a row of zero candidates would read as "it saw nothing" rather than "it was not asked".
        if candidates:
            sandbox_store.append_inputs({
                "ts": time.time(), "date": sandbox_job.today_et_str(now), "arm": "main",
                "candidates": sandbox_job.candidate_fingerprint(candidates),
            })

        for r in filled + skipped:
            sandbox_store.append_trade(r)
        sandbox_store.append_nav(nav)
        # Why a trade did NOT happen is the harder thing to reconstruct later — the JSONL has it, but
        # only searchable by hand. Recording blocks here (rather than inside sandbox_job, which is
        # deliberately pure) makes "how often does the position cap bind?" a query.
        for r in skipped:
            memory.add_note(
                "blocked",
                f"{r.get('side','?')} {r.get('symbol','?')} blocked: {r.get('skip_reason','?')}"
                + (f" — intended: {r['reason']}" if r.get("reason") else ""),
                symbol=r.get("symbol"),
                meta={"skip_reason": r.get("skip_reason"), "date": r.get("date")},
            )
        # Fills are the tighter loop: a verdict is an opinion, but a BUY is capital committed, and it
        # gets graded on exactly the same 20-day horizon as everything else. Recorded with the setup
        # that justified it, so "when this thing actually bought this pattern, did it beat the index?"
        # becomes answerable — separately from what it merely said.
        # Held positions AND candidates. `candidate_syms` deliberately excludes anything already
        # held, so keying only on candidates meant a SELL — which is by definition of a held name —
        # never matched, and neither did an ADD to an existing position. The sandbox_sells scorecard
        # could therefore never populate at all, and sandbox_buys silently covered only brand-new
        # positions. `book["positions"]` carries the same technicals shape for held names.
        by_symbol: dict[str, dict] = {}
        for row in list(book.get("positions") or []) + list(candidates):
            sym = (row.get("symbol") or "").upper()
            if sym and row.get("technicals"):
                by_symbol.setdefault(sym, row)
        today_str = sandbox_job.today_et_str(now)
        for r in filled:
            sym = (r.get("symbol") or "").upper()
            # The book snapshot strips the "-USD" suffix for display, so a BTC-USD fill has to fall
            # back to the bare ticker or crypto would silently never be recorded.
            cand = by_symbol.get(sym) or by_symbol.get(sym.removesuffix("-USD"))
            if not cand or r.get("side") not in ("buy", "sell"):
                continue
            summ = {"price": r.get("price"), "as_of_date": today_str, **(cand.get("technicals") or {})}
            memory.record_verdict(
                symbol=r["symbol"], summary=summ,
                verdict={"signal": r["side"], "conviction": r.get("conviction"),
                         "thesis": r.get("reason")},
                model=settings_store.get()["scan_model"], origin="sandbox",
            )
        # ---- comparison arms, on the snapshot main just used ----
        # Only once main has proceeded, because that is where the prices come from. A day main sat
        # out (closed, already run, disabled) is a day no arm advances either — which keeps the
        # curves aligned rather than giving one arm an extra observation the others never got.
        arms: list[dict] = []
        # A "rejects" arm consumes what an earlier arm's review refused, so it has to run after its
        # source. Ordered explicitly rather than relying on the alphabet: `fast` sorts before
        # `rejects` today by luck, and a rename would silently empty the control group.
        _ids = [a for a in sandbox_store.list_arms() if a != sandbox_store.MAIN_ARM]
        _ids.sort(key=lambda a: sandbox_store.get(a).get("engine") == "rejects")
        for _arm in _ids:
            try:
                arms.append(await _run_extra_arm(
                    _arm, now=now, price_of=price_of, spy_price=spy_price,
                    shared_plan=new_blob.get("last_strategy_note"), candidates=candidates,
                    held_rows=held_rows,
                    macro_block=macro_block, force=force, rejected=rejected_by_arm,
                    gate=gate_verdict))
            except Exception as e:  # noqa: BLE001 — a side arm must never break the real account
                _log.exception("sandbox arm %s failed", _arm)
                arms.append({"arm": _arm, "status": "error", "warnings": [str(e)]})

        out = {"status": "ok", "date": nav["date"], "posture": posture,
               "orders_filled": filled, "orders_skipped": skipped, "nav": nav,
               "weekly_review_ran": weekly_ran, "warnings": warnings}
        if arms:
            out["arms"] = arms
        return out


class SandboxTickRequest(BaseModel):
    force: bool = False
    manual: bool = False


class SandboxFundRequest(BaseModel):
    amount: float


class SandboxResetRequest(BaseModel):
    confirm: bool = False


class SandboxSettingsPatch(BaseModel):
    master_enabled: bool | None = None
    risk_tolerance: str | None = None
    retirement_date: str | None = None
    birth_date: str | None = None
    current_age: int | None = None
    retirement_age: int | None = None
    account_type: str | None = None
    avoid_wash_sales: bool | None = None
    exit_date: str | None = None
    goal_amount: float | None = None
    goal_date: str | None = None
    monthly_deposit: float | None = None
    max_position_pct: float | None = None
    cash_floor_pct: float | None = None
    allow_crypto: bool | None = None
    allow_crypto_etf: bool | None = None
    preferred_btc_etf: str | None = None
    preferred_gold_etf: str | None = None
    allow_etf: bool | None = None
    min_market_cap: float | None = None
    exclusions: list[str] | None = None
    cadence: str | None = None
    allow_after_hours: bool | None = None
    max_turnover_pct: float | None = None
    notify_on_trade: bool | None = None
    max_trades_per_tick: int | None = None
    max_new_positions_per_tick: int | None = None
    min_conviction_to_trade: int | None = None
    review_enabled: bool | None = None
    # Per-arm regime gate. Off everywhere by default and settable only per arm, because the gate is
    # an unproven hypothesis (see sandbox_store.DEFAULT_SETTINGS) — it is here so ONE arm can turn
    # it on and be compared against the arms that did not. While no verdict is supplied to
    # validate_and_fill, a gated arm blocks buys as "could not be measured", which is the intended
    # reading: the gate is on and nothing has measured the regime.
    gate_enabled: bool | None = None
    rejects_from: str | None = None
    respect_entry_zones: bool | None = None
    slippage_bps: int | None = None
    # Arm-scoped. `model` pins the LLM backbone for this arm ("" / null = the service default);
    # label and engine identify the experiment rather than configure it.
    model: str | None = None
    label: str | None = None
    engine: str | None = None


@app.post("/sandbox/tick")
async def sandbox_tick_endpoint(req: SandboxTickRequest = SandboxTickRequest()) -> dict:
    """Run one paper-trading decision cycle (the systemd timer curls this near the close each trading
    day). `force` bypasses the once-a-day + intraday-phase gates for a manual "run now"."""
    return await run_sandbox_tick(force=req.force, manual=req.manual)


def _arm_or_400(arm: str) -> str:
    """Validate an arm id from the query string. A bad id is the caller's error, not a 500 — and it
    must never reach the filesystem, since the id becomes a directory name."""
    try:
        return sandbox_store.validate_arm(arm)
    except sandbox_store.ArmError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/sandbox/state")
async def sandbox_state_endpoint(arm: str = sandbox_store.MAIN_ARM) -> dict:
    """Live-marked snapshot: cash, positions (with unrealized P/L), equity, return vs the S&P shadow,
    settings, cursors, and the latest strategy note.

    `arm` defaults to `main`, so every existing caller (the app, the timer) is unaffected."""
    assert _http is not None
    arm = _arm_or_400(arm)
    blob = sandbox_store.get(arm)
    held = [p["symbol"].upper() for p in blob["positions"]]
    prices = await _sandbox_prices(held, []) if (held or blob["benchmark"]["shares"]) else {}
    def price_of(sym: str):
        return prices.get(sym.upper())
    positions = []
    pv = 0.0
    for p in blob["positions"]:
        px = price_of(p["symbol"]) or p["avg_cost"]
        val = p["shares"] * px
        pv += val
        positions.append({
            **p,
            # Recompute the group rather than echoing the label stored when the position was opened.
            # The cap logic already looks it up fresh, so a stale stored label meant the API (and the
            # app) reported a grouping the risk engine was no longer using — after VTI and SPY were
            # merged into US_EQUITY they still displayed as two separate groups while being capped as
            # one. Same class of lie as any other stale field.
            "exposure_group": _exposure_group(p["symbol"]),
            **({"expense_ratio_pct": _expense_ratio(p["symbol"])}
               if _expense_ratio(p["symbol"]) is not None else {}),
            "price": round(px, 4), "value": round(val, 2),
            "unrealized_pct": round((px / p["avg_cost"] - 1) * 100, 2) if p["avg_cost"] else None,
        })
    cash = round(blob["cash"], 2)
    equity = round(cash + pv, 2)
    spy = price_of("^GSPC")
    bench = blob["benchmark"]
    bench_val = round(bench["shares"] * spy, 2) if spy and bench["shares"] else None
    funded = blob.get("funded_total") or 0.0
    return {
        "arm": arm, "label": blob.get("label") or arm, "engine": blob.get("engine", "llm"),
        "cash": cash, "equity": equity, "positions_value": round(pv, 2),
        "funded_total": round(funded, 2), "realized_pl_total": blob.get("realized_pl_total", 0.0),
        "interest_total": round(float(blob.get("interest_total") or 0.0), 2),
        "total_return_pct": round((equity / funded - 1) * 100, 2) if funded else None,
        "cash_pct": round(cash / equity * 100, 1) if equity else None,
        "benchmark_value": bench_val,
        "vs_benchmark_pct": round((equity - bench_val) / bench_val * 100, 2) if bench_val else None,
        "positions": sorted(positions, key=lambda x: -x["value"]),
        # `current_age` is derived here so a client renders today's age rather than whatever was
        # stored — the stored key is None once a birth_date exists. birth_date itself is kept in the
        # response (unlike in the prompt) so a settings UI can show and edit the field it owns.
        "settings": {**blob["settings"],
                     "current_age": sandbox_job.effective_age(blob["settings"])},
        "enabled": blob["settings"]["master_enabled"],
        "last_tick_date": blob.get("last_tick_date"),
        # Paired with last_tick_date so a client can tell whether the posture describes TODAY or the
        # last day the account traded — a posture with no date attached reads as current whatever
        # its age, which is the stale-as-fresh problem this codebase keeps having to fix.
        "last_posture": blob.get("last_posture") or None,
        # The review model's last verdict, dated. Present even when it APPROVED — otherwise a tick
        # the reviewer passed and a tick where it never ran look identical, which is the same
        # absent-vs-negative confusion the posture field was added to fix.
        "last_review": blob.get("last_review") or None,
        # Return without its risk is half a sentence. Computed from the NAV log rather than tracked
        # in the ledger: the log IS the record, and a running counter can drift from it with nothing
        # to reconcile against. Null until the series has two points -- "no drawdown yet" and
        # "measured, zero" are different claims.
        **sandbox_job.drawdown_stats(sandbox_store.read_nav(arm=arm)),
        "last_weekly_review_date": blob.get("last_weekly_review_date"),
        "last_strategy_note": blob.get("last_strategy_note"), "created_at": blob.get("created_at"),
    }


@app.get("/sandbox/nav")
async def sandbox_nav_endpoint(days: int = 120, arm: str = sandbox_store.MAIN_ARM) -> dict:
    return {"series": sandbox_store.read_nav(days, _arm_or_400(arm))}


@app.get("/sandbox/trades")
async def sandbox_trades_endpoint(limit: int = 100, arm: str = sandbox_store.MAIN_ARM) -> dict:
    return {"trades": sandbox_store.read_trades(limit, _arm_or_400(arm))}


class SandboxArmCreate(BaseModel):
    arm: str
    engine: str = "rules"
    label: str | None = None
    fund: float = 0.0
    enabled: bool = True
    settings: dict | None = None
    # Start from another arm's book (usually "main") so the two share a starting line and everything
    # after it is attributable to the strategy rather than to a head start.
    clone_from: str | None = None


@app.get("/sandbox/arms")
async def sandbox_arms_endpoint() -> dict:
    """Every arm with a comparable scoreboard: equity, return, and return vs its OWN benchmark shadow.

    Comparing arms to each other on raw equity would be meaningless when they were funded with
    different amounts on different days, so the honest cross-arm number is each arm's excess over the
    same-money-in-the-S&P shadow it carries."""
    assert _http is not None
    ids = sandbox_store.list_arms()
    blobs = {a: sandbox_store.get(a) for a in ids}
    held = sorted({p["symbol"].upper() for b in blobs.values() for p in b.get("positions") or []})
    prices = await _sandbox_prices(held, [])
    def price_of(sym: str):
        return prices.get(sym.upper())
    spy = price_of("^GSPC")

    out = []
    for a in ids:
        b = blobs[a]
        pv = sandbox_job.positions_value(b.get("positions") or [], price_of)
        cash = round(float(b.get("cash") or 0.0), 2)
        equity = round(cash + pv, 2)
        funded = float(b.get("funded_total") or 0.0)
        bench = b.get("benchmark") or {}
        bval = round(float(bench.get("shares") or 0.0) * spy, 2) if spy and bench.get("shares") else None
        out.append({
            "arm": a, "label": b.get("label") or a, "engine": b.get("engine", "llm"),
            "enabled": bool((b.get("settings") or {}).get("master_enabled")),
            "cash": cash, "equity": equity, "positions_value": round(pv, 2),
            "funded_total": round(funded, 2), "positions": len(b.get("positions") or []),
            "cash_pct": round(cash / equity * 100, 1) if equity else None,
            "total_return_pct": round((equity / funded - 1) * 100, 2) if funded else None,
            "benchmark_value": bval,
            "vs_benchmark_pct": round((equity - bval) / bval * 100, 2) if bval else None,
            "last_tick_date": b.get("last_tick_date"), "created_at": b.get("created_at"),
        })
    return {"arms": out}


@app.get("/sandbox/arms/nav")
async def sandbox_arms_nav_endpoint(days: int = 180) -> dict:
    """Every arm's equity curve on ONE shared date axis, for charting them against each other.

    Aligned here rather than on the client because the arms do not share a history: `main` has weeks
    the others do not, an arm created today has one point, and a day any arm sat out is a gap in that
    arm alone. Charting those raw would draw curves that step through different dates at the same x
    position — a picture that looks like a comparison and isn't. Each arm's `equity` is padded to the
    union of all dates with nulls where that arm has no observation, so index i is the same day in
    every series.

    `common_start` is the first date on which EVERY arm has a value. That is the only honest place to
    base an indexed comparison: before it, at least one arm did not exist, and normalising there would
    credit or blame it for a period it never traded."""
    ids = sandbox_store.list_arms()
    series = {a: sandbox_store.read_nav(days, a) for a in ids}
    dates = sorted({str(r.get("date")) for rows in series.values() for r in rows if r.get("date")})
    out = []
    for a in ids:
        blob = sandbox_store.get(a)
        by_date = {str(r.get("date")): r for r in series[a] if r.get("date")}
        eq, bench = [], []
        for d in dates:
            r = by_date.get(d)
            eq.append(round(float(r["equity"]), 2) if r and r.get("equity") is not None else None)
            bv = r.get("benchmark_value") if r else None
            bench.append(round(float(bv), 2) if bv is not None else None)
        out.append({"arm": a, "label": blob.get("label") or a, "engine": blob.get("engine", "llm"),
                    "equity": eq, "benchmark_value": bench})
    common = next((i for i, _ in enumerate(dates)
                   if all(s["equity"][i] is not None for s in out)), None)
    return {"dates": dates, "common_start": dates[common] if common is not None else None,
            "common_start_index": common, "arms": out}


@app.post("/sandbox/arms")
async def sandbox_create_arm_endpoint(req: SandboxArmCreate) -> dict:
    """Create a comparison arm, optionally funding and enabling it in the same call.

    Funding buys the benchmark shadow at today's price exactly as /sandbox/fund does, so the arm's
    "same money in the S&P" line starts on the day the arm did."""
    assert _http is not None
    async with _sandbox_lock:
        try:
            blob = sandbox_store.create_arm(
                req.arm, engine=req.engine, label=req.label, settings=req.settings,
                clone_from=req.clone_from)
        except sandbox_store.ArmError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        arm = blob["arm"]
        if req.fund > 0:
            spy = (await market_now.fetch_quotes(_http, ["^GSPC"])).get("^GSPC", {}).get("price")
            if not spy:
                # Refuse rather than fund a book whose benchmark leg silently never started — that
                # would flatter every comparison this arm exists to produce.
                sandbox_store.delete_arm(arm)
                raise HTTPException(status_code=503, detail="no benchmark quote — arm not created")
            blob["cash"] = round(float(blob.get("cash") or 0.0) + req.fund, 2)
            blob["funded_total"] = round(float(blob.get("funded_total") or 0.0) + req.fund, 2)
            blob["benchmark"]["shares"] = round(req.fund / spy, 6)
            blob["benchmark"]["cost_basis"] = round(req.fund, 2)
            sandbox_store.append_trade({
                "ts": time.time(), "date": sandbox_job.today_et_str(), "symbol": "CASH",
                "side": "deposit", "status": "filled", "shares": 0.0, "price": None,
                "gross": round(req.fund, 2), "cash_after": blob["cash"], "source": "fund",
                "reason": f"Arm funded with ${req.fund:,.0f}"}, arm)
        blob["settings"]["master_enabled"] = bool(req.enabled)
        return sandbox_store.save(blob, arm)


@app.delete("/sandbox/arms/{arm}")
async def sandbox_delete_arm_endpoint(arm: str) -> dict:
    async with _sandbox_lock:
        try:
            sandbox_store.delete_arm(_arm_or_400(arm))
        except sandbox_store.ArmError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"status": "deleted", "arm": arm}


@app.get("/sandbox/settings")
async def sandbox_get_settings_endpoint(arm: str = sandbox_store.MAIN_ARM) -> dict:
    return sandbox_store.get(_arm_or_400(arm))["settings"]


@app.post("/sandbox/fill_parked")
async def sandbox_fill_parked_endpoint(arm: str = sandbox_store.MAIN_ARM) -> dict:
    """Re-check today's parked buys against the live price and fill any whose zone is now met.

    NO model call. These orders were already decided — the analyst approved them this morning and
    the ledger refused them on price alone — so this is execution, not a second decision. That is
    also why it cannot churn: it can only ever fill something already proposed, at a price the
    proposer named, and every cap in validate_and_fill still applies on the way through.

    The gap it closes: with one look per trading day, an entry zone had to contain the market at
    15:35 ET or the buy waited a full session and was re-derived from scratch the next afternoon. A
    zone is a limit order in everything but name, and this is the part that was missing.

    Parked orders are same-day only. A zone describes today's setup, so anything left from an
    earlier date is dropped rather than executed against a thesis nobody re-examined.
    """
    assert _http is not None
    # `arm=all` sweeps every arm. The timer calls it that way because the original never did:
    # ExecStart carried no arm parameter, so five runs a day all re-checked `main` while the arms
    # that actually had parked orders were never looked at. An endpoint whose default is one arm is
    # right for a hand call and wrong for a sweep, so the sweep says so explicitly.
    if str(arm).lower() == "all":
        out = []
        for a in sandbox_store.list_arms():
            try:
                out.append(await sandbox_fill_parked_endpoint(arm=a))
            except HTTPException as e:  # noqa: PERF203 — one arm's quote failure must not stop the rest
                out.append({"arm": a, "status": "error", "detail": e.detail})
        return {"arm": "all", "arms": out}
    arm = _arm_or_400(arm)
    async with _sandbox_lock:
        blob = sandbox_store.get(arm)
        parked = list(blob.get("parked_orders") or [])
        today = sandbox_job.today_et_str()
        fresh = [o for o in parked if o.get("parked_date") == today]
        stale = len(parked) - len(fresh)
        if not fresh:
            return {"arm": arm, "status": "nothing_parked", "filled": [], "still_parked": 0,
                    "dropped_stale": stale}

        # Only the parked symbols need quoting; this runs on a short timer and should stay cheap.
        syms = sorted({str(o["symbol"]).upper() for o in fresh})
        try:
            prices = await _sandbox_prices([p["symbol"].upper() for p in blob["positions"]], syms)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"quote fetch failed: {redact.redact(e)}")

        def price_of(sym: str):
            return prices.get(sym.upper())

        # `exclude` is passed for the same reason the daily tick passes it, and its absence here was
        # a real hole: validate_and_fill reads every OTHER live setting off the blob itself — a
        # conviction floor, a trade cap or the gate changed after an order parked all still block the
        # sweep — but the exclusion set is the one constraint the caller has to hand in. Without it a
        # ticker added to `exclusions` at 14:38 was still bought by the 14:40 parked sweep, which
        # contradicts what that setting says it does: tickers the AI must never buy.
        new_blob, filled, skipped = sandbox_job.validate_and_fill(
            blob, fresh, price_of, group_of=_exposure_group, source="parked_fill",
            exclude={str(x).upper() for x in (blob.get("settings", {}).get("exclusions") or [])})
        # No extension_of here: parked orders are buys by construction, and the guard only gates sells.
        pv = sandbox_job.positions_value(new_blob["positions"], price_of)
        # No NAV row: this is an intraday execution, not a valuation point. Writing one would put a
        # second observation on some days and not others, and the equity series is compared
        # point-for-point against the benchmark's.
        sandbox_store.save(new_blob, arm)
        for r in filled + skipped:
            sandbox_store.append_trade(r, arm)
        return {"arm": arm, "status": "ok", "filled": filled, "skipped": skipped,
                "still_parked": len(new_blob.get("parked_orders") or []),
                "dropped_stale": stale, "positions_value": round(pv, 2)}


@app.get("/sandbox/inputs")
async def sandbox_inputs_endpoint(limit: int = 3, arm: str = sandbox_store.MAIN_ARM) -> dict:
    """What the model was shown on recent ticks, newest first. Free — NO LLM.

    The companion to /sandbox/trades: that answers what the account did, this answers what it was
    looking at when it decided. A reason quoting a number can be checked against this; before it
    existed, such a claim could not be settled against anything on disk.

    Small default limit: each row carries every candidate from one tick.
    """
    arm = _arm_or_400(arm)
    return {"arm": arm, "inputs": sandbox_store.read_inputs(limit=limit, arm=arm)}


@app.get("/sandbox/changes")
async def sandbox_changes_endpoint(limit: int = 50, arm: str = sandbox_store.MAIN_ARM) -> dict:
    """Settings changes for an arm, newest first. Free — NO LLM.

    The companion to /sandbox/trades: that log says what the account DID, this one says what it was
    told to do it with. Reading a performance change without it means guessing whether the strategy
    moved or the configuration did.
    """
    arm = _arm_or_400(arm)
    return {"arm": arm, "changes": sandbox_store.read_changes(limit=limit, arm=arm)}


@app.post("/sandbox/settings")
async def sandbox_set_settings_endpoint(
    patch: SandboxSettingsPatch, arm: str = sandbox_store.MAIN_ARM,
) -> dict:
    """Patch one arm's settings. `arm` defaults to main, so every existing caller is unaffected."""
    arm = _arm_or_400(arm)
    async with _sandbox_lock:
        blob = sandbox_store.get(arm)
        s = blob["settings"]
        # Snapshot BEFORE the patch runs, so the changelog can record what actually changed rather
        # than what was requested. The two differ constantly: the coercion below clamps, validates
        # and ignores, so a request to set max_position_pct to 500 becomes 100 and a request to set
        # it to its current value changes nothing at all. A log of requests would show edits that
        # never happened.
        _before = copy.deepcopy(s)
        d = patch.model_dump(exclude_none=True)
        if d.get("risk_tolerance") in ("conservative", "balanced", "aggressive"):
            s["risk_tolerance"] = d["risk_tolerance"]
        for k in ("master_enabled", "allow_crypto", "allow_crypto_etf", "allow_etf", "allow_after_hours",
                  "respect_entry_zones", "avoid_wash_sales", "review_enabled", "gate_enabled"):
            if k in d:
                s[k] = bool(d[k])
        for k in ("retirement_date", "exit_date", "goal_date"):
            if k in d:
                s[k] = d[k] or None
        if "birth_date" in d:
            # Validated, unlike the date fields above, because this one is ARITHMETIC rather than a
            # bias. An unparseable retirement_date is inert; an unparseable birth_date makes age_on()
            # return None, the account falls back to the stored `current_age`, and the exact drift
            # this field exists to remove comes back silently. So refuse the write instead.
            raw = str(d["birth_date"] or "").strip()
            if not raw:
                s["birth_date"] = None
            else:
                age = sandbox_job.age_on(raw)
                if age is None or age > 120:
                    raise HTTPException(
                        status_code=400,
                        detail=f"birth_date {raw!r} must be an ISO yyyy-mm-dd date in the past "
                               f"implying an age of 120 or under")
                s["birth_date"] = raw
                # One source of truth. The stored number is what went stale; keeping a copy of it
                # next to the date invites some later reader to pick the wrong one. Absent beats
                # stale — every consumer goes through sandbox_job.effective_age() now.
                s["current_age"] = None
        if "rejects_from" in d:
            # Validated as an arm id, not free text: it names a directory-backed arm, and a typo
            # would leave the control group silently empty with nothing to say why.
            v = str(d["rejects_from"] or "").strip().lower()
            if not v:
                s["rejects_from"] = ""
            elif v in sandbox_store.list_arms():
                s["rejects_from"] = v
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"rejects_from {v!r} is not an existing arm "
                           f"(have: {', '.join(sandbox_store.list_arms())})")
        if "account_type" in d and str(d["account_type"]).lower() in ("cash", "margin"):
            s["account_type"] = str(d["account_type"]).lower()
        # Validated against the known vehicles, so a typo can't silently disable the preference
        # (prefer_vehicle ignores an unrecognised value). Empty string = no preference.
        for _key, _family, _what in (("preferred_btc_etf", sandbox_job.BTC_ETFS, "bitcoin ETF"),
                                     ("preferred_gold_etf", sandbox_job.GOLD_ETFS, "gold ETF")):
            if _key in d:
                v = str(d[_key] or "").strip().upper()
                if not v or v in _family:
                    s[_key] = v
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"unknown {_what} {v!r} — expected one of "
                               f"{', '.join(sorted(_family))} (or empty for no preference)")
        for k in ("current_age", "retirement_age"):
            if k in d:
                v = d[k]
                s[k] = max(0, min(120, int(v))) if v else None
        if "min_market_cap" in d:
            # Clamped, not rejected. This is a coarse dial the user drags on a chip row, and refusing
            # a value there would surface as a failed save with nothing on screen explaining which
            # bound was crossed. 0 means no floor at all, which is a legitimate choice.
            s["min_market_cap"] = max(0.0, min(5e12, float(d["min_market_cap"])))
        if "goal_amount" in d:
            v = float(d["goal_amount"])
            s["goal_amount"] = v if v > 0 else None
        if "monthly_deposit" in d:
            s["monthly_deposit"] = max(0.0, float(d["monthly_deposit"]))
        if "cadence" in d and str(d["cadence"]).lower() in ("daily", "weekly"):
            s["cadence"] = str(d["cadence"]).lower()
        if "max_turnover_pct" in d:
            s["max_turnover_pct"] = max(0.0, min(100.0, float(d["max_turnover_pct"])))
        if "notify_on_trade" in d:
            s["notify_on_trade"] = bool(d["notify_on_trade"])
        if "exclusions" in d:
            s["exclusions"] = sorted({str(x).strip().upper() for x in (d["exclusions"] or []) if str(x).strip()})
        if "max_position_pct" in d:
            s["max_position_pct"] = max(5.0, min(100.0, float(d["max_position_pct"])))
        if "cash_floor_pct" in d:
            s["cash_floor_pct"] = max(0.0, min(90.0, float(d["cash_floor_pct"])))
        if "min_conviction_to_trade" in d:
            s["min_conviction_to_trade"] = max(0, min(100, int(d["min_conviction_to_trade"])))
        if "max_trades_per_tick" in d:
            s["max_trades_per_tick"] = max(0, min(20, int(d["max_trades_per_tick"])))
        if "max_new_positions_per_tick" in d:
            s["max_new_positions_per_tick"] = max(0, min(20, int(d["max_new_positions_per_tick"])))
        if "slippage_bps" in d:
            s["slippage_bps"] = max(0, min(200, int(d["slippage_bps"])))
        if "model" in d:
            # Free text on purpose — model ids ship faster than any allow-list here could track, and
            # an allow-list that lags is how a new frontier model becomes unusable. Empty = inherit
            # the service's configured scan model, which is the right default for every arm that is
            # not specifically testing a backbone.
            v = str(d["model"] or "").strip()
            s["model"] = v or None
        blob["settings"] = s
        # Label and engine live on the BLOB, not in settings — they identify the experiment rather
        # than configure it. Engine is validated here because an unknown value would silently fall
        # through to the analyst path and quietly turn a control arm into another copy of main.
        if d.get("label"):
            blob["label"] = str(d["label"]).strip()[:48]
        if d.get("engine"):
            e = str(d["engine"]).strip().lower()
            if e not in sandbox_store.ENGINES:
                raise HTTPException(
                    status_code=400,
                    detail=f"engine must be one of {', '.join(sandbox_store.ENGINES)}")
            blob["engine"] = e
        # Diff the real before/after. Only keys that MOVED are recorded; an unchanged key is not an
        # edit, and a changelog full of no-ops is one nobody reads.
        _changed = {k: {"from": _before.get(k), "to": s[k]}
                    for k in s if _before.get(k) != s[k]}
        if _changed:
            sandbox_store.append_change({
                "ts": time.time(), "date": sandbox_job.today_et_str(),
                "arm": arm, "source": "api", "changed": _changed,
            }, arm)
        sandbox_store.save(blob, arm)
        return {**s, "arm": arm, "label": blob.get("label"), "engine": blob.get("engine")}


@app.post("/sandbox/fund")
async def sandbox_fund_endpoint(req: SandboxFundRequest) -> dict:
    """Add (or withdraw, negative) fictional cash. A positive deposit also buys shadow ^GSPC shares at
    today's price so the benchmark tracks the same money on the same schedule."""
    assert _http is not None
    if req.amount == 0:
        raise HTTPException(status_code=422, detail="amount must be non-zero")
    async with _sandbox_lock:
        blob = sandbox_store.get()
        spy = None
        try:
            spy = (await market_now.fetch_quotes(_http, ["^GSPC"])).get("^GSPC", {}).get("price")
        except Exception:  # noqa: BLE001
            pass
        # A deposit must not be credited without its benchmark leg. The shadow ^GSPC position is the
        # only thing total_return_pct is measured against, so crediting cash while skipping the
        # shares makes the sandbox look permanently ahead of a benchmark that was never given the
        # money — and nothing later reconciles it.
        if req.amount > 0 and not spy:
            raise HTTPException(
                status_code=503,
                detail="can't price the benchmark right now — deposit not applied, try again shortly",
            )
        if req.amount < 0 and round(blob["cash"] + req.amount, 2) < 0:
            raise HTTPException(
                status_code=422,
                detail=f"withdrawal exceeds cash on hand (${blob['cash']:,.2f}) — sell positions first",
            )
        if req.amount > 0:
            blob["benchmark"]["shares"] = round(blob["benchmark"]["shares"] + req.amount / spy, 6)
            blob["benchmark"]["cost_basis"] = round(blob["benchmark"]["cost_basis"] + req.amount, 2)
        else:
            # Symmetry matters: a withdrawal that left funded_total and the shadow untouched made
            # total_return_pct (= equity/funded - 1) permanently understate, and the benchmark kept
            # compounding money the account no longer had. Withdraw proportionally from both.
            take = min(-req.amount, blob["funded_total"])
            if blob["funded_total"] > 0 and blob["benchmark"]["shares"] > 0:
                frac = take / blob["funded_total"]
                blob["benchmark"]["shares"] = round(blob["benchmark"]["shares"] * (1 - frac), 6)
                blob["benchmark"]["cost_basis"] = round(blob["benchmark"]["cost_basis"] * (1 - frac), 2)
            blob["funded_total"] = round(blob["funded_total"] - take, 2)
        blob["cash"] = round(blob["cash"] + req.amount, 2)
        if req.amount > 0:
            blob["funded_total"] = round(blob["funded_total"] + req.amount, 2)
        if blob.get("created_at") is None:
            blob["created_at"] = time.time()
        sandbox_store.save(blob)
        sandbox_store.append_trade({
            "ts": time.time(), "date": sandbox_job.today_et_str(), "symbol": "CASH",
            "side": "deposit" if req.amount > 0 else "withdraw", "status": "filled", "shares": 0.0,
            "price": None, "gross": round(req.amount, 2), "cash_after": blob["cash"], "source": "funding",
            "reason": f"{'Added' if req.amount > 0 else 'Withdrew'} ${abs(req.amount):,.0f}",
        })
        return {"cash": blob["cash"], "funded_total": blob["funded_total"], "benchmark": blob["benchmark"]}


@app.post("/sandbox/reset")
async def sandbox_reset_endpoint(req: SandboxResetRequest) -> dict:
    if not req.confirm:
        raise HTTPException(status_code=422, detail="reset requires confirm=true")
    async with _sandbox_lock:
        fresh = sandbox_store.reset()
        return {"status": "reset", "cash": fresh["cash"], "funded_total": fresh["funded_total"]}


# ======================================================================================
# SWT-8 — the JOURNAL's mechanical half: what a recorded plan would have done next.
#
# memory.py scores the ANALYST (a fixed 20-day forward return, no stop, no target), the sandbox
# scores the PAPER TRADER in its own ledger, and the options tracker scores real option positions.
# Nothing scored what the USER did with a verdict on a stock. This route computes the plan-as-written
# leg of that comparison; app/plan_replay.py holds the logic and every judgement call behind it.
# ======================================================================================

class PlanReplayRequest(BaseModel):
    symbol: str
    date: str                                # the session the plan was RECORDED (YYYYMMDD or ISO)
    entry_low: float | None = None
    entry_high: float | None = None
    stop: float | None = None
    target: float | None = None
    horizon_days: int = plan_replay.DEFAULT_HORIZON_DAYS
    fill_window_days: int = plan_replay.DEFAULT_FILL_WINDOW_DAYS


# Yahoo range strings, smallest first — a replay only ever needs history back to the plan date, and
# asking for "5y" to answer a three-week-old plan costs ~1,250 bars of parsing per call for nothing.
_REPLAY_RANGES: tuple[tuple[int, str], ...] = ((300, "1y"), (650, "2y"), (1700, "5y"), (3500, "10y"))


def _replay_range(plan_date: str, today: str) -> str:
    """The shortest Yahoo range that certainly reaches back to `plan_date` (both YYYYMMDD).

    The thresholds sit well under each nominal window (300 days for "1y") on purpose: a range that
    lands one session short of the plan date comes back as an EMPTY bar list, which this route would
    then report as "nothing has traded since" — a confident, wrong answer where the real fault was
    not asking for enough history. Erring long costs some parsing; erring short costs the truth.
    """
    from datetime import date
    try:
        a = date(int(plan_date[:4]), int(plan_date[4:6]), int(plan_date[6:8]))
        b = date(int(today[:4]), int(today[4:6]), int(today[6:8]))
        days_back = max(0, (b - a).days)
    except ValueError:  # a syntactically valid YYYYMMDD that is not a real day (e.g. 20260231)
        return "max"
    for limit, rng in _REPLAY_RANGES:
        if days_back <= limit:
            return rng
    return "max"


@app.post("/journal/replay")
async def journal_replay_endpoint(req: PlanReplayRequest) -> dict:
    """SWT-8 — replay one recorded plan against the daily bars that actually followed it. No LLM.

    Give it the plan as it stood (`entry_low`/`entry_high`, `stop`, `target`) and the date it was
    recorded, and it walks the sessions AFTER that date to say whether the plan hit its target,
    stopped out, ran out of clock, never filled, or is still running — with the outcome expressed in
    R, the units of the risk the plan itself defined.

    THREE THINGS TO READ BEFORE THE NUMBER:
      * `ambiguous: true` means a single daily bar touched BOTH the stop and the target and the bars
        cannot say which came first. This replay assumes the STOP — the pessimistic reading — so an
        ambiguous row is a result that rests on that assumption rather than on the tape.
      * `r: null` is not 0R. It means the plan named no usable stop (so there is no risk to divide
        by), or the trade has not exited. 0R would claim the trade made exactly what it risked.
      * `outcome: "never_filled"` means price never came into the entry zone inside the fill window,
        or gapped through the stop before the entry could fill. The plan was never tradeable — this
        is SWT-3's chase warning, seen after the fact, and it is not a losing trade.

    `outcome: null` with `refused: false` means the plan is fine but nothing has traded since the
    date given — a plan recorded today has no replay yet, which is not an error.

    422 for a plan that cannot be replayed (a date that is not a date, a date in the future, an
    inverted zone, a stop that is not below the entry, a target that is not above it). 404 when the
    symbol has no price history at all. 502 when the price fetch fails.
    """
    assert _http is not None
    if not (req.symbol or "").strip():
        raise HTTPException(status_code=422, detail="a plan needs a symbol")
    try:
        d0 = plan_replay.norm_date(req.date)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    today = sandbox_job.today_et_str().replace("-", "")
    if d0 > today:
        # A future plan date is not an empty result — every bar "after" it is one that has not
        # happened. Refusing is the only answer that cannot be mistaken for "the plan did nothing".
        raise HTTPException(status_code=422, detail=f"plan date {d0} is in the future — nothing to replay")

    try:
        series = await fetch_series(_http, req.symbol, rng=_replay_range(d0, today))
    except Exception:  # noqa: BLE001 — the upstream error text carries the Yahoo URL; log it, never ship it
        _log.warning("journal replay: price fetch failed for %s", req.symbol, exc_info=True)
        raise HTTPException(status_code=502, detail="price history fetch failed")
    if not series.closes:
        raise HTTPException(status_code=404, detail="no price history for this symbol")

    bars = plan_replay.bars_from_series(series, d0)
    out = plan_replay.replay(
        bars,
        entry=(req.entry_low, req.entry_high),
        stop=req.stop,
        target=req.target,
        horizon_days=req.horizon_days,
        fill_window_days=req.fill_window_days,
    )
    if out.get("refused"):
        raise HTTPException(status_code=422, detail=out.get("reason") or "the plan cannot be replayed")
    if not bars:
        # The symbol HAS history — none of it is after the plan date. That is a 200 saying "not
        # yet", not a 404: we looked, and what we found was that no session has traded since.
        out = {**out, "reason": f"no session has traded since {d0} — nothing to replay yet"}
    return {
        "symbol": series.symbol,
        "as_of": d0,
        "source": series.source,
        # The window actually walked, so a caller can see the replay was fed what it thinks it was.
        "bars_from": bars[0]["date"] if bars else None,
        "bars_to": bars[-1]["date"] if bars else None,
        **out,
        "note": ("What the plan as written would have done — the MECHANICAL leg. It is not what you "
                 "did, and an ambiguous bar resolves against the trade."),
    }
