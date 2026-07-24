"""StockTracker Signals — Tier-2 Claude analyst service. Decision support only, not advice."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import observability, selfupdate, settings_store, usage_store
from . import congress, cycle, fundamentals, insider, market_now, options, seasonality, shorts, webull
from .analyst import (
    analyze,
    daily_brief,
    market_overview,
    news_moves,
    options_note,
    plan_entry,
    recommend,
    review_portfolio,
)
from .discover import discover
from .market import fetch_series, summarize
from .news import fetch_context, fetch_dated_news
from .scan_job import LATEST, run_scan

_http: httpx.AsyncClient | None = None
_cache: dict[tuple, tuple[float, dict]] = {}
_MARKET_NOW_TTL = 180  # market-now overview cached ~3 min so repeated taps don't re-run the model
_DAILY_BRIEF_TTL = 1800  # morning brief cached ~30 min — the app fires it once/day; this just guards retries
_log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http
    _http = httpx.AsyncClient()
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

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StockTracker Signals</title>
<style>
  :root { color-scheme: light dark; --accent:#2563eb; --ok:#16a34a; --err:#dc2626; --muted:#888; }
  * { box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, sans-serif; max-width: 60rem; margin: 1.5rem auto;
         padding: 0 1rem; line-height: 1.5; }
  /* dashboard layout: cards flow into responsive columns (masonry) — 2-up on desktop, 1 on mobile.
     column width drives the breakpoint; the 60rem body cap keeps it at 2 columns max. */
  .masonry { columns: 27rem; column-gap: 1.2rem; }
  .masonry > * { -webkit-column-break-inside: avoid; page-break-inside: avoid; break-inside: avoid; }
  form#f { margin-bottom: 1.1rem; }
  h1 { font-size: 1.4rem; margin: 0 0 .2rem; }
  .sub { color: var(--muted); margin: 0 0 1.4rem; }
  .card { border: 1px solid #8883; border-radius: .9rem; padding: 1rem 1.1rem; margin-bottom: 1.1rem;
          background: rgba(127,127,127,.045); }
  .card h2 { font-size: 1rem; margin: 0 0 .8rem; }
  label { display: block; margin: .85rem 0 .3rem; font-weight: 600; font-size: .88rem; }
  .card label:first-of-type { margin-top: 0; }
  input, select { width: 100%; padding: .55rem .65rem; font-size: 1rem; border: 1px solid #8886;
          border-radius: .5rem; background: transparent; color: inherit; }
  select { appearance: auto; }
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
  .empty { color: var(--muted); font-size: .82rem; }
  .loading { color: var(--muted); font-size: .85rem; }
  /* status header */
  .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(8rem, 1fr)); gap: .8rem .9rem; }
  .stat .k { font-size: .7rem; text-transform: uppercase; letter-spacing: .03em; color: var(--muted); }
  .stat .v { font-size: 1.05rem; font-weight: 600; margin-top: .1rem; }
  .stat .d { font-size: .78rem; color: var(--muted); margin-top: .1rem; }
  .countdown { font-variant-numeric: tabular-nums; }
  .meter { height: .45rem; background: #8883; border-radius: .3rem; overflow: hidden; margin-top: .35rem; }
  .meter > span { display: block; height: 100%; width: 0; background: var(--accent); }
  .meter.warn > span { background: #d97706; }
  .meter.err > span { background: var(--err); }
  /* signal colors + change badges */
  .sig-buy { color: var(--ok); font-weight: 600; }
  .sig-sell { color: var(--err); font-weight: 600; }
  .sig-hold { color: var(--muted); font-weight: 600; }
  .badge { display: inline-block; font-size: .66rem; padding: .04rem .38rem; border-radius: .8rem;
           margin-left: .22rem; font-weight: 700; vertical-align: middle; }
  .badge.flip { background: rgba(217,119,6,.18); color: #d97706; }
  .badge.dip { background: rgba(22,163,74,.16); color: var(--ok); }
  .badge.cross { background: rgba(220,38,38,.16); color: var(--err); }
  /* compact tables */
  .scroll { overflow-x: auto; margin: .3rem -.2rem 0; }
  table.tbl { width: 100%; border-collapse: collapse; font-size: .82rem; }
  table.tbl th, table.tbl td { text-align: left; padding: .3rem .45rem; border-bottom: 1px solid #8882;
                               white-space: nowrap; }
  table.tbl th { color: var(--muted); font-weight: 600; font-size: .7rem; text-transform: uppercase; }
  table.tbl td.num { text-align: right; font-variant-numeric: tabular-nums; }
  table.tbl td.thesis { max-width: 13rem; overflow: hidden; text-overflow: ellipsis; }
  table.tbl tr.changed td:first-child { border-left: 3px solid #d97706; padding-left: calc(.45rem - 3px); }
  /* data sources */
  .src { display: flex; align-items: center; gap: .5rem; padding: .3rem 0; border-bottom: 1px solid #8882; }
  .dot { width: .6rem; height: .6rem; border-radius: 50%; flex: none; background: var(--muted); }
  .dot.ok { background: var(--ok); } .dot.warn { background: #d97706; } .dot.down { background: var(--err); }
  .src .nm { font-weight: 600; font-size: .85rem; }
  .src .lat { color: var(--muted); font-size: .76rem; }
  .src .dt { color: var(--muted); font-size: .76rem; flex: 1; text-align: right;
             overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .ivp { display: flex; flex-wrap: wrap; gap: .3rem; margin-top: .4rem; }
  .ivp .pill { font-size: .73rem; background: #8882; border-radius: .8rem; padding: .12rem .5rem; }
  .ivp .pill.done { background: rgba(22,163,74,.16); color: var(--ok); }
  /* cost */
  .cost-head { font-size: 1rem; margin: .1rem 0 .4rem; }
  .cost-head b { color: var(--accent); }
  /* activity log */
  .logrow { display: flex; gap: .5rem; font-size: .77rem; font-family: ui-monospace, SFMono-Regular, monospace;
            padding: .13rem 0; border-bottom: 1px solid #8881; }
  .logrow.bad { color: var(--err); }
  .logrow .m { width: 3.2rem; color: var(--muted); }
  .logrow .p { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .logrow .st { width: 2.4rem; text-align: right; }
  .logrow .ms { width: 4rem; text-align: right; color: var(--muted); }
</style></head>
<body>
<h1>StockTracker Signals</h1>
<p class="sub">Tier-2 Claude analyst — operations &amp; configuration</p>

<div class="masonry">
<div class="card" id="status-card">
  <h2>Status <span class="hint" id="uptime"></span></h2>
  <div class="stat-grid">
    <div class="stat"><div class="k">Last scan</div><div class="v" id="scan-when">…</div>
      <div class="d" id="scan-detail"></div></div>
    <div class="stat"><div class="k">Next scan</div><div class="v countdown" id="next-scan">…</div>
      <div class="d">06:30 America/Chicago</div></div>
    <div class="stat"><div class="k">Disk</div><div class="v" id="disk-v">…</div>
      <div class="meter" id="disk-meter"><span></span></div></div>
  </div>
  <div class="row" style="margin-top:.8rem">
    <button type="button" class="secondary" id="prune">Prune cache</button>
    <span class="hint" id="prune-status" style="flex:1"></span>
  </div>
  <div class="hint" id="cache-detail" style="margin-top:.5rem"></div>
</div>

<div class="card">
  <h2>AI usage</h2>
  <div id="usage-totals" class="usage-totals">loading…</div>
  <div id="usage-chart"></div>
  <div class="hint">Daily tokens over the last 30 days (hover a bar for the day's detail).</div>
  <div id="usage-models" class="hint"></div>
</div>

<div class="card">
  <h2>Latest scan <span class="hint" id="scan-count"></span></h2>
  <div class="scroll"><table class="tbl" id="scan-tbl">
    <thead><tr><th>Sym</th><th>Signal</th><th>Conv</th><th>Dip</th><th>Sqz</th><th>&lt;200w</th><th>Thesis</th></tr></thead>
    <tbody id="scan-body"><tr><td colspan="7" class="loading">loading…</td></tr></tbody>
  </table></div>
  <div class="hint" style="margin-top:.5rem">Changed rows are badged: <span class="badge flip">flip</span>
  signal flipped · <span class="badge dip">dip+</span> new/deeper dip · <span class="badge cross">×200w</span>
  crossed below the 200-week line.</div>
</div>

<div class="card">
  <h2>Data sources <span class="hint" id="src-as-of"></span></h2>
  <div id="sources"><div class="loading">probing…</div></div>
  <div class="hint" style="margin-top:.8rem">IV-rank progress — days of ATM-IV logged toward the
  <span id="iv-target">20</span>-day rank window (nightly, per stock):</div>
  <div class="ivp" id="iv-progress"><span class="empty">no IV history yet</span></div>
</div>

<div class="card">
  <h2>Cost breakdown</h2>
  <div class="cost-head" id="cost-head">loading…</div>
  <div class="scroll"><table class="tbl" id="cost-tbl">
    <thead><tr><th>Kind</th><th class="num">Calls</th><th class="num">Tokens</th><th class="num">Cost</th></tr></thead>
    <tbody id="cost-body"></tbody>
  </table></div>
  <div class="hint" id="cost-avg" style="margin-top:.6rem"></div>
</div>

<form id="f">
  <div class="card">
    <h2>Connection &amp; models</h2>
    <label for="provider">LLM backend</label>
    <select id="provider">
      <option value="api">Anthropic API — per-token billing</option>
      <option value="cli">Claude CLI — this machine's subscription (no per-token cost)</option>
    </select>
    <div class="hint">CLI mode shells out to the local <code>claude</code> CLI signed in to your
    subscription — no per-token billing, but it draws on your Max rate-limit budget and needs the CLI
    + OAuth present on the server. Keep a key set too, so API mode still works.</div>
    <div class="hint" id="cli-auth" style="margin-top:.5rem"></div>
    <div class="row" style="margin-top:.3rem">
      <button type="button" class="secondary" id="cli-test">Test CLI auth</button>
      <span class="hint" id="cli-test-status" style="flex:1"></span>
    </div>
    <div class="hint" style="margin-top:.35rem">Set up: run <code>claude setup-token</code> on any machine
    (subscription login) and paste the token below — a dedicated token that won't rotate or get logged out.
    Stored server-side and used immediately (no restart, no <code>.env</code> edit).</div>
    <label for="clitoken">CLI subscription token</label>
    <input id="clitoken" type="password" autocomplete="off" placeholder="paste to set/replace — leave blank to keep current">
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

<div class="card">
  <h2>Recent activity <span class="hint">— last requests, in-memory since restart</span></h2>
  <div id="logs"><div class="loading">loading…</div></div>
  <div class="hint" id="errs-head" style="margin-top:.7rem"></div>
  <div id="errs"></div>
</div>
</div><!-- /masonry -->

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
      const bp = u.by_provider || {};
      const billed = (bp.api && bp.api.cost_usd) || 0;
      const notional = (bp.cli && bp.cli.cost_usd) || 0;
      let cost = "<b>$" + billed.toFixed(4) + "</b> billed";
      if (notional > 0) cost += ' · <span class="hint">$' + notional.toFixed(4) + " notional (subscription)</span>";
      $("usage-totals").innerHTML = "<b>" + fmt(u.total_tokens) + "</b> tokens · " + cost + " · " +
        fmt(u.total_calls) + " calls" +
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
    $("provider").value = s.llm_provider || "api";
    $("cli-auth").innerHTML = s.cli_token_set
      ? 'CLI subscription token: <b class="ok-t">set</b> (' + esc(s.cli_token_hint) + ') — used when LLM backend is CLI.'
      : 'CLI subscription token: <b class="err-t">not set</b> — CLI mode will fail until you add one.';
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
                   verdict_ttl_seconds: Number($("ttl").value), llm_provider: $("provider").value };
    if ($("key").value) body.anthropic_api_key = $("key").value;
    if ($("fkey").value) body.finnhub_api_key = $("fkey").value;
    if ($("clitoken").value) body.cli_oauth_token = $("clitoken").value;
    const r = await fetch("/api/settings", { method: "POST",
      headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    const st = $("status");
    st.textContent = r.ok ? "Saved ✓" : "Save failed"; st.className = r.ok ? "ok-t" : "err-t";
    $("key").value = ""; $("fkey").value = ""; $("clitoken").value = ""; load();
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

  // ---- ops dashboard ----
  const pad2 = (n) => String(n).padStart(2, "0");
  function fmtDur(s) {
    s = Math.max(0, Math.floor(s));
    const d = Math.floor(s / 86400); s -= d * 86400;
    const h = Math.floor(s / 3600); s -= h * 3600;
    const m = Math.floor(s / 60); const sec = s - m * 60;
    if (d > 0) return d + "d " + h + "h " + m + "m";
    if (h > 0) return h + "h " + m + "m " + pad2(sec) + "s";
    return m + "m " + pad2(sec) + "s";
  }
  function fmtBytes(n) {
    if (n == null) return "–";
    const u = ["B", "KB", "MB", "GB", "TB"]; let i = 0; n = Number(n);
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return (i === 0 ? n : n.toFixed(1)) + " " + u[i];
  }
  const money = (n) => "$" + Number(n || 0).toFixed(4);
  const esc = (s) => { const d = document.createElement("div"); d.textContent = s == null ? "" : String(s); return d.innerHTML.replace(/"/g, "&quot;"); };

  let nextScanTs = null;
  function tickCountdown() {
    const el = $("next-scan"); if (!el || !nextScanTs) return;
    const left = nextScanTs - Date.now() / 1000;
    el.textContent = left <= 0 ? "due now" : "in " + fmtDur(left);
  }
  function renderIvProgress(ivp) {
    const box = $("iv-progress"); const target = ivp.target || 20;
    $("iv-target").textContent = target;
    const syms = ivp.symbols || {}; const keys = Object.keys(syms);
    if (!keys.length) { box.innerHTML = '<span class="empty">no IV history yet — the nightly scan logs one point per stock</span>'; return; }
    box.innerHTML = keys.map((k) => {
      const n = syms[k]; const done = n >= target;
      return '<span class="pill' + (done ? " done" : "") + '">' + esc(k) + " " + (done ? "✓ " + n : n + "/" + target) + "</span>";
    }).join("");
  }
  function renderStatus(s) {
    $("uptime").textContent = "· up " + fmtDur(s.uptime_s);
    const sc = s.scan || {};
    if (sc.generated_at) {
      $("scan-when").textContent = agoText(sc.generated_at);
      const c = sc.counts || {};
      let d = (c.buy || 0) + " buy · " + (c.hold || 0) + " hold · " + (c.sell || 0) + " sell";
      if (sc.total_cost != null) d += " · " + money(sc.total_cost);
      if (sc.changed && sc.changed.length) d += " · " + sc.changed.length + " changed";
      $("scan-detail").textContent = d;
    } else { $("scan-when").textContent = "never"; $("scan-detail").textContent = "no scan yet"; }
    nextScanTs = s.next_scan_at || null; tickCountdown();
    const dk = s.disk || {};
    $("disk-v").textContent = dk.pct != null ? dk.pct + "% · " + fmtBytes(dk.used) + " / " + fmtBytes(dk.total) : "–";
    const meter = $("disk-meter"); const bar = meter.firstElementChild;
    bar.style.width = (dk.pct != null ? Math.min(100, dk.pct) : 0) + "%";
    meter.className = "meter" + (dk.pct >= 90 ? " err" : dk.pct >= 75 ? " warn" : "");
    const ca = s.cache || {};
    $("cache-detail").textContent = "Cache: " + fmtBytes(ca.shorts_bytes) + " in " + (ca.shorts_files || 0) +
      " shorts files · " + (ca.iv_history_days_total || 0) + " IV-history rows";
    renderIvProgress(s.iv_progress || {});
  }
  async function loadStatus() {
    try { renderStatus(await (await fetch("/api/status")).json()); }
    catch (e) { $("uptime").textContent = "· status unavailable"; }
  }

  const sigClass = (sig) => (sig && sig.indexOf("buy") >= 0) ? "sig-buy" : (sig && sig.indexOf("sell") >= 0) ? "sig-sell" : "sig-hold";
  function scanBadges(r) {
    let b = "";
    if (r.flipped) b += '<span class="badge flip">flip</span>';
    if (r.dip_new) b += '<span class="badge dip">dip+</span>';
    if (r.crossed_below_200wma) b += '<span class="badge cross">×200w</span>';
    return b;
  }
  async function loadScan() {
    let data; try { data = await (await fetch("/scan/latest")).json(); } catch (e) { $("scan-body").innerHTML = '<tr><td colspan="7" class="empty">scan unavailable</td></tr>'; return; }
    const rows = (data.results || []).filter((r) => !r.error);
    const errs = (data.results || []).filter((r) => r.error);
    $("scan-count").textContent = data.generated_at
      ? "· " + rows.length + " scored" + (errs.length ? " · " + errs.length + " err" : "") : "· none yet";
    const body = $("scan-body");
    if (!rows.length) { body.innerHTML = '<tr><td colspan="7" class="empty">no scan results yet — runs nightly at 06:30, or POST /scan/run</td></tr>'; return; }
    body.innerHTML = rows.map((r) => {
      const changed = r.flipped || r.dip_new || r.crossed_below_200wma;
      const th = esc(r.thesis || "");
      return '<tr class="' + (changed ? "changed" : "") + '">' +
        "<td><b>" + esc(r.symbol) + "</b>" + scanBadges(r) + "</td>" +
        '<td class="' + sigClass(r.signal) + '">' + esc(r.signal) + "</td>" +
        '<td class="num">' + (r.conviction != null ? esc(r.conviction) : "–") + "</td>" +
        "<td>" + esc(r.dip || "–") + "</td>" +
        "<td>" + esc(r.squeeze || "–") + "</td>" +
        "<td>" + (r.below_200wma ? "yes" : "–") + "</td>" +
        '<td class="thesis" title="' + th + '">' + th + "</td></tr>";
    }).join("");
  }

  async function loadSources() {
    let data;
    try { data = await (await fetch("/api/sources")).json(); }
    catch (e) { $("sources").innerHTML = '<div class="empty">sources unavailable</div>'; return; }
    $("src-as-of").textContent = "· checked " + agoText(data.as_of);
    $("sources").innerHTML = (data.sources || []).map((s) =>
      '<div class="src"><span class="dot ' + esc(s.status) + '"></span>' +
      '<span class="nm">' + esc(s.name) + "</span>" +
      '<span class="lat">' + (s.latency_ms != null ? Math.round(s.latency_ms) + " ms" : "") + "</span>" +
      '<span class="dt" title="' + esc(s.detail) + '">' + esc(s.detail) + "</span></div>"
    ).join("") || '<div class="empty">no sources</div>';
  }

  async function loadCost() {
    let c; try { c = await (await fetch("/api/cost")).json(); } catch (e) { $("cost-head").textContent = "cost unavailable"; return; }
    let head = "Billed (API) — MTD <b>" + money(c.month_to_date_usd) + "</b> · projected <b>" +
      money(c.projected_month_usd) + "</b> · all-time <b>" + money(c.all_time_usd) + "</b>";
    if (c.cli_notional_usd > 0)
      head += '<br><span class="hint">Subscription (CLI): <b>$0 billed</b> · ' +
        money(c.cli_notional_usd) + " notional all-time (what it would’ve cost on the API)</span>";
    $("cost-head").innerHTML = head;
    const kinds = Object.entries(c.by_kind || {}).sort((a, b) => b[1].usd - a[1].usd);
    $("cost-body").innerHTML = kinds.length ? kinds.map(([k, v]) =>
      "<tr><td>" + esc(k) + '</td><td class="num">' + fmt(v.calls) + '</td><td class="num">' +
      fmt(v.tokens) + '</td><td class="num">' + money(v.usd) + "</td></tr>").join("")
      : '<tr><td colspan="4" class="empty">no calls recorded yet</td></tr>';
    const bits = [];
    if (c.per_scan_avg_usd != null) bits.push("per scan-call " + money(c.per_scan_avg_usd));
    if (c.per_deep_avg_usd != null) bits.push("per deep-call " + money(c.per_deep_avg_usd));
    $("cost-avg").textContent = bits.join(" · ");
  }

  async function loadLogs() {
    let data; try { data = await (await fetch("/api/logs?limit=30")).json(); } catch (e) { $("logs").innerHTML = '<div class="empty">logs unavailable</div>'; return; }
    const reqs = data.requests || [];
    $("logs").innerHTML = reqs.length ? reqs.map((r) => {
      const bad = r.status >= 400 || r.error;
      return '<div class="logrow' + (bad ? " bad" : "") + '"><span class="m">' + esc(r.method) +
        '</span><span class="p" title="' + esc(r.path) + '">' + esc(r.path) + '</span><span class="st">' +
        r.status + '</span><span class="ms">' + Math.round(r.ms) + " ms</span></div>";
    }).join("") : '<div class="empty">no requests recorded yet</div>';
    const errs = data.errors || [];
    $("errs-head").textContent = errs.length ? "Recent errors (" + errs.length + "):" : "No errors since restart.";
    $("errs").innerHTML = errs.map((r) =>
      '<div class="logrow bad"><span class="p">' + esc(r.method) + " " + esc(r.path) +
      '</span><span class="st">' + r.status + "</span></div>").join("");
  }

  $("prune").onclick = async () => {
    const ps = $("prune-status"); ps.textContent = "pruning…";
    try {
      const r = await (await fetch("/api/prune-cache", { method: "POST" })).json();
      ps.textContent = "freed " + fmtBytes(r.bytes_freed) + " (" + r.deleted_files + " file" + (r.deleted_files === 1 ? "" : "s") + ")";
    } catch (e) { ps.textContent = "prune failed"; }
    loadStatus();
  };

  $("cli-test").onclick = async () => {
    const st = $("cli-test-status"); st.textContent = "testing…"; st.className = "hint";
    try {
      const r = await (await fetch("/api/cli-auth-test")).json();
      st.textContent = r.ok ? "✓ authenticated" : "✗ " + (r.detail || "failed");
      st.className = r.ok ? "ok-t" : "err-t";
    } catch (e) { st.textContent = "✗ request failed"; st.className = "err-t"; }
  };

  load(); checkVersion(); loadUsage();
  loadStatus(); loadScan(); loadSources(); loadCost(); loadLogs();
  setInterval(() => { refreshSynced(); loadUsage(); loadCost(); }, 60000); // heartbeat + usage/cost
  setInterval(() => { loadStatus(); loadScan(); loadLogs(); }, 30000);     // live ops cards
  setInterval(loadSources, 60000);                                          // source probes (heavier)
  setInterval(tickCountdown, 1000);                                         // next-scan countdown
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    return _PAGE


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
    if crypto:  # multi-year trend + (BTC) halving-cycle position — weak-sample context, flagged as such
        summary.update(await cycle.crypto_context(_http, series.symbol, series.closes))
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
    rule_score: int | None = None,
) -> dict:
    cfg = settings_store.get()
    # Position is part of the cache identity: a different holding must yield a fresh, re-personalized
    # verdict rather than a stale one keyed only on the symbol.
    key = (symbol.upper(), crypto, deep, shares, avg_cost, rule_score)
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < cfg["verdict_ttl_seconds"]:
        return {**hit[1], "cached": True}

    summary = await _snapshot(symbol, crypto=crypto)
    # Personalize when the user holds this asset — the analyst frames the verdict as add/hold/trim.
    pos = _position_block(summary, shares, avg_cost)
    if pos:
        summary["position"] = pos
    if rule_score is not None:  # the app's mechanical composite — the analyst reconciles with it
        summary["rule_score"] = max(0, min(100, rule_score))
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
    rule_score: int | None = None,
) -> dict:
    """One asset's analyst verdict. `deep=true` uses the deep model; crypto symbols use Yahoo's
    `BTC-USD` form with `crypto=true` (skips the S&P benchmark). Optional `shares` + `avg_cost`
    personalize the verdict as an add/hold/trim call on an existing position; optional `rule_score`
    (the app's mechanical 0-100 composite) makes the analyst reconcile a diverging read."""
    return await _build_signal(
        symbol, deep=deep, crypto=crypto, shares=shares, avg_cost=avg_cost, rule_score=rule_score,
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
        raise HTTPException(status_code=502, detail=f"data fetch failed: {e}")
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
        raise HTTPException(status_code=502, detail=f"market snapshot failed: {e}")
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
        raise HTTPException(status_code=502, detail=f"analyst failed: {e}")
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
        raise HTTPException(status_code=502, detail=f"market snapshot failed: {e}")
    try:
        snap["market_movers"] = {
            "gainers": await _movers_side("day_gainers", count),
            "losers": await _movers_side("day_losers", count),
        }
    except Exception as e:  # noqa: BLE001
        _log.warning("daily_brief movers failed: %s", e)

    # Today's catalysts: which of the user's (equity) names report earnings today, in ET. Best-effort —
    # a Finnhub hiccup just drops the catalysts line, it never fails the brief.
    from datetime import datetime
    from zoneinfo import ZoneInfo
    today_et = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    equities = [s for s in (cfg.get("watchlist") or []) if not s.upper().endswith("-USD")]
    _earn_sem = asyncio.Semaphore(8)

    async def _reports_today(s: str) -> str | None:
        async with _earn_sem:
            try:
                if (await fetch_context(_http, s)).get("next_earnings") == today_et:
                    return s.upper()
            except Exception:  # noqa: BLE001
                pass
            return None

    try:
        snap["catalysts_today"] = [
            s for s in await asyncio.gather(*[_reports_today(s) for s in equities]) if s
        ]
    except Exception as e:  # noqa: BLE001
        _log.warning("daily_brief catalysts failed: %s", e)
        snap["catalysts_today"] = []

    try:
        brief, usage = await daily_brief(snap, deep=deep)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"analyst failed: {e}")
    usage_store.record(usage, symbol="", kind="daily_brief")

    payload = {
        "title": brief.title,
        "body": brief.body,
        "tone": brief.tone,
        "catalysts_today": snap.get("catalysts_today", []),
        "session": snap["session"],
        "model": usage["model"],
        "as_of": now,
        "usage": usage,
        "cached": False,
    }
    _cache[key] = (now, payload)
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
    # full watchlist). A small semaphore caps the burst so we don't trip Finnhub's 60/min rate limit.
    _earn_sem = asyncio.Semaphore(8)

    async def _next_earnings(s: str) -> tuple[str, str | None]:
        async with _earn_sem:
            try:
                ctx = await fetch_context(_http, s)
                return s, ctx.get("next_earnings")
            except Exception:  # noqa: BLE001
                return s, None

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
    payload = {"as_of": now, "symbol": symbol.upper() if symbol else None, "events": events, "cached": False}
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
        raise HTTPException(status_code=502, detail=f"data fetch failed: {e}")
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
        raise HTTPException(status_code=502, detail=f"data fetch failed: {e}")
    ctx = await cycle.crypto_context(_http, series.symbol, series.closes)
    lt = ctx.get("long_term_trend")
    if not lt or "sma_200w" not in lt:
        raise HTTPException(status_code=404, detail="not enough weekly history for a 200-week trend")
    return {"symbol": series.symbol, "close": round(series.closes[-1], 4), **lt}


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
        raise HTTPException(status_code=502, detail=f"data fetch failed: {e}")
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
async def news_moves_endpoint(symbol: str, deep: bool = False) -> dict:
    """AIE-4 — why the stock moved. Finds its notable daily moves over ~3 weeks, pulls dated company
    news, and has the analyst correlate them: which move was news-driven and which happened on flows/
    technicals. Equities only (Finnhub news is equities); crypto returns a friendly note. Cached ~1h.
    Returns {symbol, news_moves: {summary, drivers[]}|null, note?}."""
    assert _http is not None
    sym = symbol.upper()
    if sym.endswith("-USD"):
        return {"symbol": sym, "news_moves": None, "note": "News correlation isn't available for crypto."}

    now = time.time()
    key = ("news_moves", sym, deep)
    hit = _cache.get(key)
    if hit and now - hit[0] < 3600:
        return {**hit[1], "cached": True}

    try:
        series = await fetch_series(_http, sym)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"price fetch failed: {e}")

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
    if not news:
        # Moves but no news coverage to correlate — report the moves honestly without spending an LLM call.
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
        raise HTTPException(status_code=502, detail=f"analyst failed: {e}")
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
        iv_rank = options.iv_rank(options.load_iv_history(chain.symbol), current=chain.expiry.atm_iv)
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
            "mid": _m(options.mid_price(match.bid, match.ask)),
            "limit_price": _m(options._limit_price(match)),  # re-price: mid, else last trade
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
            now=now, extra_warnings=warnings,
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


class PortfolioReviewRequest(BaseModel):
    cash: float = 0.0
    deep: bool = False
    holdings: list[Holding] = []  # transient — reviewed, never persisted


@app.post("/portfolio/review")
async def portfolio_review_endpoint(req: PortfolioReviewRequest) -> dict:
    """AI review of the WHOLE portfolio: overall health, concentration/diversification flags, a per-
    holding action list (trim/hold/add/watch), and a cash-deployment note. One structured LLM call over
    lightweight technical snapshots of each holding (fast — no per-name enrichment). Cached by the
    holdings+cash identity for the verdict TTL. Send crypto holdings as <SYM>-USD."""
    if not req.holdings:
        raise HTTPException(status_code=422, detail="no holdings to review")
    assert _http is not None
    cfg = settings_store.get()
    key = ("portfolio_review", req.deep, round(req.cash, 2),
           tuple(sorted((h.symbol.upper(), round(h.shares, 6), round(h.avg_cost, 4)) for h in req.holdings)))
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < cfg["verdict_ttl_seconds"]:
        return {**hit[1], "cached": True}

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
            except Exception:  # noqa: BLE001 — an unpriceable holding is dropped, not fatal
                return None
            price = summ["price"]
            return {
                "symbol": sym.removesuffix("-USD"),
                "shares": h.shares,
                "avg_cost": h.avg_cost,
                "price": round(price, 4),
                "value": round(price * h.shares, 2),
                "unrealized_gain_pct": round((price / h.avg_cost - 1) * 100, 1) if h.avg_cost else None,
                "technicals": {k: summ.get(k) for k in (
                    "rsi14", "macd_hist", "pct_vs_sma50", "golden_cross",
                    "rel_strength_3mo_vs_benchmark", "pct_off_52w_high") if summ.get(k) is not None},
            }

    rows = [r for r in await asyncio.gather(*[_price(h) for h in req.holdings]) if r]
    if not rows:
        raise HTTPException(status_code=502, detail="couldn't price any holdings")
    total_value = sum(r["value"] for r in rows) + max(req.cash, 0.0)
    for r in rows:
        r["weight_pct"] = round(100.0 * r["value"] / total_value, 1) if total_value else None
    portfolio = {
        "cash": round(req.cash, 2),
        "cash_pct": round(100.0 * max(req.cash, 0.0) / total_value, 1) if total_value else 0.0,
        "total_value": round(total_value, 2),
        "positions": sorted(rows, key=lambda r: r["value"], reverse=True),
    }
    try:
        review, usage = await review_portfolio(portfolio, cash=req.cash, deep=req.deep)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"analyst failed: {e}")
    usage_store.record(usage, symbol="", kind="portfolio")
    payload = {
        "review": review.model_dump(),
        "portfolio": portfolio,
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
