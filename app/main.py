"""StockTracker Signals — Tier-2 Claude analyst service. Decision support only, not advice."""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import selfupdate, settings_store, usage_store
from .analyst import analyze, plan_entry, recommend
from .discover import discover
from .market import fetch_series, summarize
from .news import fetch_context
from .scan_job import LATEST, run_scan

_http: httpx.AsyncClient | None = None
_cache: dict[tuple[str, bool], tuple[float, dict]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http
    _http = httpx.AsyncClient()
    try:
        yield
    finally:
        await _http.aclose()


app = FastAPI(title="StockTracker Signals", version="0.2.0", lifespan=lifespan)


# --- settings UI + API ---

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StockTracker Signals</title>
<style>
  :root { color-scheme: light dark; --accent:#2563eb; --ok:#16a34a; --err:#dc2626; --muted:#888; }
  * { box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, sans-serif; max-width: 36rem; margin: 1.5rem auto;
         padding: 0 1rem; line-height: 1.5; }
  h1 { font-size: 1.4rem; margin: 0 0 .2rem; }
  .sub { color: var(--muted); margin: 0 0 1.4rem; }
  .card { border: 1px solid #8883; border-radius: .9rem; padding: 1rem 1.1rem; margin-bottom: 1.1rem; }
  .card h2 { font-size: 1rem; margin: 0 0 .8rem; }
  label { display: block; margin: .85rem 0 .3rem; font-weight: 600; font-size: .88rem; }
  .card label:first-of-type { margin-top: 0; }
  input { width: 100%; padding: .55rem .65rem; font-size: 1rem; border: 1px solid #8886;
          border-radius: .5rem; background: transparent; color: inherit; }
  .hint { font-size: .8rem; color: var(--muted); font-weight: 400; }
  .row { display: flex; gap: .5rem; align-items: center; }
  .row input { flex: 1; }
  button { padding: .55rem 1.1rem; font-size: .95rem; border: 0; border-radius: .5rem;
           background: var(--accent); color: #fff; cursor: pointer; white-space: nowrap; }
  button.secondary { background: #8883; color: inherit; }
  button.ok { background: var(--ok); }
  .chips { margin-top: .6rem; min-height: 1.1rem; }
  .chip { display: inline-flex; align-items: center; background: rgba(37,99,235,.15);
          border-radius: 1rem; padding: .22rem .7rem; margin: .15rem .25rem .15rem 0; font-size: .85rem; }
  .empty { color: var(--muted); font-size: .82rem; }
  .synced { font-size: .85rem; margin: .1rem 0 .5rem; color: var(--muted); }
  .synced.fresh { color: var(--ok); }
  .synced.stale { color: #d97706; }
  .usage-totals { font-size: 1rem; margin: .1rem 0 .3rem; }
  .usage-totals b { color: var(--accent); }
  #usage-chart svg { display: block; width: 100%; height: auto; margin: .5rem 0 .2rem; }
  #usage-chart .bar { fill: var(--accent); }
  #usage-chart .bar.zero { fill: rgba(136,136,136,.28); }
  #usage-chart .axis { fill: var(--muted); font-size: 9px; }
  .save { margin-top: .4rem; }
  #status, #upstatus { font-size: .88rem; min-height: 1.1rem; margin-top: .6rem; }
  .ok-t { color: var(--ok); } .err-t { color: var(--err); }
  code { background: #8882; padding: .1rem .3rem; border-radius: .3rem; font-size: .85em; }
</style></head>
<body>
<h1>StockTracker Signals</h1>
<p class="sub">Tier-2 Claude analyst — configuration</p>

<div class="card">
  <h2>AI usage</h2>
  <div id="usage-totals" class="usage-totals">loading…</div>
  <div id="usage-chart"></div>
  <div class="hint">Daily tokens over the last 30 days (hover a bar for the day's detail).</div>
  <div id="usage-models" class="hint"></div>
</div>

<form id="f">
  <div class="card">
    <h2>Connection &amp; models</h2>
    <label for="key">Anthropic API key</label>
    <input id="key" type="password" autocomplete="off" placeholder="leave blank to keep current">
    <div class="hint" id="keyhint"></div>
    <label for="fkey">Finnhub API key <span class="hint">— optional, adds news + earnings context</span></label>
    <input id="fkey" type="password" autocomplete="off" placeholder="leave blank to keep current">
    <div class="hint" id="fkeyhint"></div>
    <label for="deep">Deep model <span class="hint">— on-demand deep dives</span></label>
    <input id="deep" autocomplete="off">
    <label for="scan">Scan model <span class="hint">— cheap watchlist scans</span></label>
    <input id="scan" autocomplete="off">
    <label for="ttl">Verdict cache TTL <span class="hint">— seconds</span></label>
    <input id="ttl" type="number" min="0" autocomplete="off">
  </div>

  <div class="card">
    <h2>Nightly watchlist <span class="hint">— synced from the app</span></h2>
    <div id="synced" class="synced">checking sync…</div>
    <label>Stocks</label>
    <div class="chips" id="watch-chips"></div>
    <label style="margin-top:.7rem">Crypto</label>
    <div class="chips" id="cwatch-chips"></div>
    <div class="hint" style="margin-top:.7rem">The app is the source of truth — it pushes your watchlist
    here automatically, so add/remove symbols in the app, not on this page. Scanned nightly at 06:30;
    the app notifies you when a signal flips.</div>
  </div>

  <button class="save" type="submit">Save settings</button>
  <div id="status"></div>
</form>

<div class="card">
  <h2>Service</h2>
  <div id="version" class="hint">version …</div>
  <div class="row" style="margin-top:.8rem">
    <button type="button" class="secondary" id="check">Check for updates</button>
    <button type="button" class="ok" id="update" style="display:none">Update &amp; restart</button>
  </div>
  <div id="upstatus"></div>
  <div class="hint" style="margin-top:.8rem">
    <a href="https://github.com/Pr0zak/stocktracker-signals" target="_blank" rel="noopener">github.com/Pr0zak/stocktracker-signals</a>
  </div>
</div>

<p class="hint">API: <code>GET /signal/{symbol}</code> · <code>GET /plan/{symbol}?cash=</code> · <code>POST /recommendations</code> ·
<code>POST /scan/run</code> · <code>GET /scan/latest</code> · <code>GET /health</code>.
Decision support only — not investment advice.</p>

<script>
  const $ = (id) => document.getElementById(id);

  // Read-only chips — the watchlist is owned by the app and synced up via POST /api/settings.
  function renderChips(kind, syms) {
    const box = $(kind + "-chips");
    box.innerHTML = "";
    if (!syms.length) { box.innerHTML = '<span class="empty">none yet — connect the app to sync</span>'; return; }
    syms.forEach((sym) => {
      const chip = document.createElement("span");
      chip.className = "chip";
      const b = document.createElement("b"); b.textContent = sym; chip.appendChild(b);
      box.appendChild(chip);
    });
  }

  function agoText(sec) {
    const d = Math.max(0, Date.now() / 1000 - sec);
    if (d < 90) return "just now";
    if (d < 3600) return Math.round(d / 60) + " min ago";
    if (d < 86400) return Math.round(d / 3600) + " hr ago";
    const days = Math.round(d / 86400); return days + " day" + (days > 1 ? "s" : "") + " ago";
  }
  function renderSynced(ts) {
    const el = $("synced");
    if (!ts) {
      el.textContent = "Last synced: never — set this service's URL in the app's Settings to connect.";
      el.className = "synced stale"; return;
    }
    const fresh = (Date.now() / 1000 - ts) < 1800; // the app re-syncs every ~15 min
    el.textContent = (fresh ? "● " : "○ ") + "Last synced from the app: " + agoText(ts);
    el.className = "synced " + (fresh ? "fresh" : "stale");
  }
  // Refresh just the heartbeat line (never the form inputs — the user may be mid-edit).
  async function refreshSynced() {
    try { renderSynced((await (await fetch("/api/settings")).json()).watchlist_synced_at); } catch (e) {}
  }

  const fmt = (n) => Number(n).toLocaleString();
  function drawUsageChart(series) {
    const box = $("usage-chart");
    const W = 520, H = 130, padL = 6, padT = 8, padB = 18;
    const n = series.length;
    const max = Math.max(1, ...series.map((d) => d.tokens));
    const bw = (W - padL) / n;
    const bars = series.map((d, i) => {
      const h = (d.tokens / max) * (H - padT - padB);
      const x = padL + i * bw, y = H - padB - h;
      const t = d.date + ": " + fmt(d.tokens) + " tokens · $" + d.cost_usd.toFixed(4) +
        " · " + d.calls + " call" + (d.calls === 1 ? "" : "s");
      return '<rect class="bar' + (d.tokens ? '' : ' zero') + '" x="' + x.toFixed(1) +
        '" y="' + y.toFixed(1) + '" width="' + Math.max(1, bw - 1.5).toFixed(1) +
        '" height="' + Math.max(1, h).toFixed(1) + '" rx="1"><title>' + t + '</title></rect>';
    }).join("");
    const md = (s) => s.slice(5);
    const lbl = '<text class="axis" x="' + padL + '" y="' + (H - 5) + '">' + md(series[0].date) + '</text>' +
      '<text class="axis" x="' + (W / 2) + '" y="' + (H - 5) + '" text-anchor="middle">' + md(series[Math.floor(n / 2)].date) + '</text>' +
      '<text class="axis" x="' + W + '" y="' + (H - 5) + '" text-anchor="end">' + md(series[n - 1].date) + '</text>';
    box.innerHTML = '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="daily AI token usage">' + bars + lbl + '</svg>';
  }
  async function loadUsage() {
    try {
      const u = await (await fetch("/api/usage?days=30")).json();
      $("usage-totals").innerHTML = "<b>" + fmt(u.total_tokens) + "</b> tokens · <b>$" +
        u.total_cost_usd.toFixed(4) + "</b> · " + fmt(u.total_calls) + " calls" +
        ' <span class="hint">(' + fmt(u.total_input_tokens) + " in / " + fmt(u.total_output_tokens) + " out, all-time)</span>";
      drawUsageChart(u.series);
      const models = Object.entries(u.by_model).sort((a, b) => b[1].cost_usd - a[1].cost_usd)
        .map(([m, v]) => m + " — " + fmt(v.calls) + " calls · $" + v.cost_usd.toFixed(4)).join("<br>");
      $("usage-models").innerHTML = models || "No calls recorded yet.";
    } catch (e) { $("usage-totals").textContent = "usage unavailable"; }
  }

  async function load() {
    const s = await (await fetch("/api/settings")).json();
    $("deep").value = s.deep_model; $("scan").value = s.scan_model; $("ttl").value = s.verdict_ttl_seconds;
    renderChips("watch", s.watchlist || []); renderChips("cwatch", s.crypto_watchlist || []);
    renderSynced(s.watchlist_synced_at);
    $("keyhint").textContent = s.anthropic_api_key_set
      ? "Key is set (" + s.anthropic_api_key_hint + "). Leave blank to keep it."
      : "No key set — the analyst can't run until you add one.";
    $("fkeyhint").textContent = s.finnhub_api_key_set
      ? "Key is set. Leave blank to keep it." : "No Finnhub key — news/earnings context is off.";
  }
  $("f").onsubmit = async (e) => {
    e.preventDefault();
    const body = { deep_model: $("deep").value, scan_model: $("scan").value,
                   verdict_ttl_seconds: Number($("ttl").value) };
    if ($("key").value) body.anthropic_api_key = $("key").value;
    if ($("fkey").value) body.finnhub_api_key = $("fkey").value;
    const r = await fetch("/api/settings", { method: "POST",
      headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    const st = $("status");
    st.textContent = r.ok ? "Saved ✓" : "Save failed"; st.className = r.ok ? "ok-t" : "err-t";
    $("key").value = ""; $("fkey").value = ""; load();
  };

  async function checkVersion() {
    $("version").textContent = "checking…";
    const v = await (await fetch("/api/version")).json();
    let label = "version " + v.version;
    if (!v.git) label += " · (not a git checkout — updates disabled)";
    else if (v.update_available) label += " · " + v.behind + " update" + (v.behind > 1 ? "s" : "") + " available";
    else label += " · up to date";
    $("version").textContent = label;
    $("update").style.display = v.update_available ? "inline-block" : "none";
  }
  $("check").onclick = checkVersion;
  $("update").onclick = async () => {
    $("upstatus").textContent = "Updating — the service will restart…"; $("upstatus").className = "";
    try { await fetch("/api/update", { method: "POST" }); } catch (e) {}
    setTimeout(() => { $("upstatus").textContent = "Restarted. Reloading…"; location.reload(); }, 6000);
  };

  load(); checkVersion(); loadUsage();
  setInterval(() => { refreshSynced(); loadUsage(); }, 60000); // keep heartbeat + usage live
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    return _PAGE


@app.get("/api/settings")
async def get_settings() -> dict:
    cfg = settings_store.get()
    key = cfg["anthropic_api_key"]
    return {
        "anthropic_api_key_set": bool(key),
        "anthropic_api_key_hint": ("…" + key[-4:]) if len(key) >= 4 else ("set" if key else ""),
        "finnhub_api_key_set": bool(cfg.get("finnhub_api_key", "")),
        "deep_model": cfg["deep_model"],
        "scan_model": cfg["scan_model"],
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


async def _snapshot(symbol: str, *, crypto: bool, bench_closes: list[float] | None = None) -> dict:
    """Fetch + summarize one asset's daily technicals (plus news/earnings for stocks). Pass
    `bench_closes` to reuse an already-fetched S&P series (batch callers); stocks fetch it otherwise."""
    assert _http is not None
    try:
        series = await fetch_series(_http, symbol)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"data fetch failed: {e}")
    if len(series.closes) < 30:
        raise HTTPException(status_code=422, detail="not enough history for a signal")

    if not crypto and bench_closes is None:  # relative strength vs the S&P is equity-only
        try:
            bench_closes = (await fetch_series(_http, "^GSPC")).closes
        except Exception:  # noqa: BLE001 — RS just gets skipped
            bench_closes = None

    summary = summarize(series, None if crypto else bench_closes)
    if not crypto:  # optional news/earnings context (Finnhub, stocks only)
        summary.update(await fetch_context(_http, series.symbol))
    return summary


async def _build_signal(
    symbol: str, *, deep: bool, crypto: bool,
    shares: float | None = None, avg_cost: float | None = None,
) -> dict:
    cfg = settings_store.get()
    # Position is part of the cache identity: a different holding must yield a fresh, re-personalized
    # verdict rather than a stale one keyed only on the symbol.
    key = (symbol.upper(), crypto, deep, shares, avg_cost)
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < cfg["verdict_ttl_seconds"]:
        return {**hit[1], "cached": True}

    summary = await _snapshot(symbol, crypto=crypto)
    # Personalize when the user holds this asset — the analyst frames the verdict as add/hold/trim.
    pos = _position_block(summary, shares, avg_cost)
    if pos:
        summary["position"] = pos
    try:
        verdict, usage = await analyze(summary, deep=deep)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"analyst failed: {e}")
    usage_store.record(usage, symbol=summary.get("symbol", symbol.upper()), kind="deep" if deep else "signal")

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
) -> dict:
    """One asset's analyst verdict. `deep=true` uses the deep model; crypto symbols use Yahoo's
    `BTC-USD` form with `crypto=true` (skips the S&P benchmark). Optional `shares` + `avg_cost`
    personalize the verdict as an add/hold/trim call on an existing position."""
    return await _build_signal(symbol, deep=deep, crypto=crypto, shares=shares, avg_cost=avg_cost)


@app.get("/plan/{symbol}")
async def plan(
    symbol: str, cash: float, crypto: bool = False, deep: bool = False,
    shares: float | None = None, avg_cost: float | None = None,
) -> dict:
    """Scenario: "if I deployed $cash into this symbol" — one asset's entry plan (action, entry zone,
    share count, stop/target, timing). Optional shares+avg_cost tell the analyst it's already held."""
    if cash <= 0:
        raise HTTPException(status_code=422, detail="cash must be > 0")
    cfg = settings_store.get()
    key = ("plan", symbol.upper(), crypto, round(cash, 2), deep, shares, avg_cost)
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < cfg["verdict_ttl_seconds"]:
        return {**hit[1], "cached": True}

    summary = await _snapshot(symbol, crypto=crypto)
    pos = _position_block(summary, shares, avg_cost)
    if pos:
        summary["position"] = pos
    try:
        entry, usage = await plan_entry(summary, cash=cash, deep=deep)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"analyst failed: {e}")
    _sanitize_plan(entry, cash, crypto)
    usage_store.record(usage, symbol=summary.get("symbol", symbol), kind="plan")

    payload = {
        "symbol": summary.get("symbol", symbol.upper()),
        "model": usage["model"],
        "as_of": now,
        "cash": cash,
        "plan": entry.model_dump(),
        "usage": usage,
        "cached": False,
    }
    _cache[key] = (now, payload)
    return payload


class Holding(BaseModel):
    symbol: str
    shares: float
    avg_cost: float


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
        return {**hit[1], "cached": True}

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
        raise HTTPException(status_code=502, detail=f"analyst failed: {e}")
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
        "picks": [p.model_dump() for p in recs.picks],
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
    """The most recent nightly-scan result (what the app polls). Empty until the first scan runs."""
    if LATEST.exists():
        return json.loads(LATEST.read_text())
    return {"generated_at": None, "results": [], "flips": [], "total_cost_usd": 0.0}


@app.post("/scan/run")
async def scan_run() -> dict:
    """Run the configured-watchlist scan now (also wired to a nightly systemd timer)."""
    return await run_scan()
