"""The Station tab (Zote ytik5yyk, 2026-11-18): a market-watching board the user builds from the page.

Widgets sit on a 12-column grid. In edit mode the grid's cells show as a dot lattice, and widgets can
be added from a tray, dragged to move, dragged by the corner to resize, configured and removed; the
keyboard does the same (arrows move, Shift+arrows resize, Delete removes, Enter configures). The
layout is saved on the server (/station/layout, token-gated) so every browser shows the same
station, and in this browser's storage as well, so an edit is never lost to a missing token.

Spliced into dashboard.PAGE at markers like the Sandbox tab, and shares that script's scope: it uses
`$`, `esc`, `afetch`, `apiToken`, `showTokenPrompt`, `openTickerDrawer` from dashboard.py and `usd`,
`sgn`, `vsShown`, `shortLabel`, `vsRow`, `growBars`, `groupOf`, `pickGroup`, `pickArm` from the
Sandbox tab. Every class is prefixed `st-`: unprefixed names have collided on this page before.

Data is fetched once per URL across all widgets (a shared cache with a per-kind lifetime) and only
refreshed while the Station tab is showing and the browser tab is visible. Nothing here calls a
model: the watchlist's buy/hold/sell comes from the nightly scan, not a fresh /signal read.
"""

HTML = r"""
<section class="panel" id="p-station" hidden>
  <div class="st-top">
    <div class="st-title">
      <h2 id="st-name">My station</h2>
      <input id="st-name-in" class="st-name-in" maxlength="60" aria-label="Station name" hidden>
      <span class="st-saved" id="st-saved" aria-live="polite"></span>
    </div>
    <div class="st-actions">
      <button type="button" class="secondary sm" id="st-reset" hidden>Use the starter layout</button>
      <button type="button" class="secondary sm" id="st-add" hidden>Add widget</button>
      <button type="button" class="sm" id="st-edit" aria-pressed="false">Edit layout</button>
    </div>
  </div>
  <p class="st-help" id="st-help" hidden><span class="st-wide-only">Drag a widget to move it and drag its corner to resize it.
    With a widget selected, the arrow keys move it, Shift + arrows resize it, Enter opens its settings and Delete removes it.</span>
    <span class="st-narrow-only">Use ⚙ to change a widget and ✕ to remove it. To move or resize widgets, open this page on a wider screen.</span></p>
  <div class="st-tray" id="st-tray" hidden></div>
  <div class="st-grid" id="st-grid"></div>
  <div class="st-empty" id="st-empty" hidden>
    <p>Your station is empty.</p>
    <div class="st-empty-btns">
      <button type="button" class="sm" id="st-empty-add">Add a widget</button>
      <button type="button" class="secondary sm" id="st-empty-starter">Use the starter layout</button>
    </div>
  </div>
  <div class="st-sr" id="st-live" aria-live="polite"></div>
</section>
<dialog class="st-dlg" id="st-dlg">
  <form method="dialog" id="st-form">
    <h3 id="st-dlg-title">Widget settings</h3>
    <div id="st-dlg-body"></div>
    <div class="st-dlg-btns">
      <button type="button" class="secondary sm" id="st-dlg-cancel">Cancel</button>
      <button type="submit" class="sm" id="st-dlg-save">Save</button>
    </div>
  </form>
</dialog>
"""

CSS = r"""
  /* ================================================================ station */
  body[data-tab="p-station"] .wrap, body[data-tab="p-station"] .head-in { max-width: 120rem; }
  .st-tray[hidden], .st-grid[hidden], .st-empty[hidden], .st-help[hidden] { display: none; }
  .st-race .vs-row { grid-template-columns: minmax(0, 1fr) minmax(3rem, 5rem) 3.6rem; gap: .45rem; }
  .st-top { display: flex; align-items: center; gap: .8rem; flex-wrap: wrap; margin: .2rem 0 .9rem; }
  .st-title { display: flex; align-items: baseline; gap: .7rem; flex: 1; min-width: 12rem; }
  .st-title h2 { margin: 0; font-size: 1.35rem; font-weight: 400; letter-spacing: -.01em; }
  .st-name-in { font: inherit; font-size: 1.2rem; padding: .2rem .5rem; max-width: 20rem; }
  .st-saved { color: var(--muted); font-size: .8rem; }
  .st-actions { display: flex; gap: .45rem; flex-wrap: wrap; }
  #st-edit[aria-pressed="true"] { background: var(--chip-sel); color: var(--ink); }
  .st-help { color: var(--muted); font-size: .84rem; margin: -.3rem 0 .8rem; max-width: 62rem; }

  /* the tray of widget types, shown while editing */
  .st-tray { display: grid; gap: .5rem; grid-template-columns: repeat(auto-fill, minmax(13rem, 1fr)); margin-bottom: 1rem; }
  .st-type { text-align: left; background: var(--card2); border: 1px dashed var(--outline); color: var(--ink);
             border-radius: .9rem; padding: .6rem .75rem; cursor: pointer; font: inherit; }
  .st-type:hover, .st-type:focus-visible { border-style: solid; border-color: var(--accent); }
  .st-type b { display: block; font-weight: 500; font-size: .9rem; }
  .st-type span { color: var(--muted); font-size: .78rem; line-height: 1.35; display: block; margin-top: .1rem; }
  .st-type .st-lock { color: var(--amber); }
  .st-type, .st-type span { white-space: normal; }
  .st-editing .st-seg { visibility: hidden; }

  /* the grid. --pitch-x / --pitch-y are one cell plus one gap: the dot lattice in edit mode sits on
     the cells' corners, so the dots are where a widget's corner can land. */
  .st-grid { --cols: 12; --rowh: 4.6rem; display: grid; grid-template-columns: repeat(var(--cols), minmax(0, 1fr));
             grid-auto-rows: var(--rowh); gap: var(--gap); position: relative; min-height: 8rem;
             padding: 0; border-radius: 1.4rem; transition: background-color .2s; }
  .st-grid.st-editing { background-color: #ffffff04;
    background-image: radial-gradient(circle at 2px 2px, #8A94A2 0 1.4px, transparent 1.8px);
    background-size: calc((100% + var(--gap)) / var(--cols)) calc(var(--rowh) + var(--gap));
    background-position: calc(var(--gap) / -2 - 2px) calc(var(--gap) / -2 - 2px); }

  .st-w { position: relative; min-width: 0; min-height: 0; display: flex; flex-direction: column;
          border-radius: 1.25rem; padding: .8rem .95rem .75rem; border: 1px solid #ffffff0d; overflow: hidden;
          background: radial-gradient(120% 140% at 100% 0%, var(--tint, transparent) 0%, transparent 60%), var(--card); }
  .st-w.up { --tint: rgba(94,221,156,.12); } .st-w.down { --tint: rgba(232,102,63,.12); }
  .st-editing .st-w { outline: 1px dashed #6B7787; outline-offset: -1px; }
  .st-narrow-only { display: none; }
  @media (min-width: 62.01rem) { .st-editing .st-w { cursor: grab; user-select: none; touch-action: none; } }
  @media (max-width: 62rem) { .st-wide-only { display: none; } .st-narrow-only { display: inline; } }
  .st-editing .st-w:focus-visible, .st-w.st-sel { outline: 2px solid var(--accent); outline-offset: -2px; }
  .st-w.st-drag { cursor: grabbing; z-index: 5; box-shadow: 0 18px 40px #000a; opacity: .92; transition: none; }
  .st-ghost { border-radius: 1.25rem; border: 2px dashed var(--accent); background: rgba(180,160,255,.08);
              pointer-events: none; z-index: 1; }
  .st-h { display: flex; align-items: baseline; gap: .5rem; min-width: 0; margin-bottom: .35rem; }
  .st-h .t { font-weight: 500; font-size: .92rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .st-h .s { color: var(--muted); font-size: .78rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex: 1; min-width: 0; }
  .st-b { flex: 1; min-height: 0; display: flex; flex-direction: column; }
  .st-scroll { overflow-y: auto; scrollbar-width: thin; min-height: 0; flex: 1; margin-right: -.4rem; padding-right: .4rem; }
  .st-msg { color: var(--muted); font-size: .84rem; margin: auto 0; }
  .st-msg button { margin-top: .4rem; }

  /* the edit bar on each widget and its resize corner */
  .st-wbar { position: absolute; top: .45rem; right: .45rem; display: none; gap: .25rem; z-index: 3; }
  .st-editing .st-wbar { display: flex; }
  .st-wbar button { width: 1.9rem; height: 1.9rem; padding: 0; border-radius: 50%; background: var(--chip-sel);
                    color: var(--ink); font-size: .9rem; line-height: 1; }
  .st-wbar button:hover { background: var(--accent); color: var(--on-accent); }
  .st-grip { position: absolute; right: 0; bottom: 0; width: 1.6rem; height: 1.6rem; display: none; cursor: nwse-resize;
             z-index: 3; touch-action: none; }
  .st-grip::after { content: ""; position: absolute; right: .45rem; bottom: .45rem; width: .6rem; height: .6rem;
                    border-right: 2px solid var(--accent); border-bottom: 2px solid var(--accent); border-radius: 0 0 .2rem 0; }
  .st-editing .st-grip { display: block; }

  /* figures */
  .st-big { font-size: 1.85rem; font-weight: 300; letter-spacing: -.02em; line-height: 1.1; font-variant-numeric: tabular-nums; }
  .st-mid { font-size: 1.3rem; font-weight: 300; font-variant-numeric: tabular-nums; }
  .st-row1 { display: flex; align-items: center; gap: .55rem; flex-wrap: wrap; }
  .st-spark { flex: 1; min-height: 1.8rem; margin-top: .4rem; position: relative; }
  .st-spark svg { position: absolute; inset: 0; width: 100%; height: 100%; overflow: visible; }
  .st-sym { background: none; border: 0; padding: 0; color: var(--ink); font: inherit; font-weight: 600; cursor: pointer;
            text-align: left; justify-self: start; border-radius: .3rem; }
  .st-sym:hover { color: var(--accent); }
  .st-editing .st-sym { pointer-events: none; }

  /* chart */
  .st-seg { display: inline-flex; gap: .15rem; background: #ffffff0a; padding: .12rem; border-radius: .55rem; margin-left: auto; }
  .st-seg button { background: none; border: 0; color: var(--muted); padding: .12rem .5rem; font-size: .74rem; border-radius: .4rem; cursor: pointer; }
  .st-seg button[aria-pressed="true"] { background: var(--chip-sel); color: var(--ink); }
  .st-plot { position: relative; flex: 1; min-height: 3rem; margin-top: .35rem; }
  .st-plot svg { position: absolute; inset: 0 3.6rem 1.1rem 0; width: calc(100% - 3.6rem); height: calc(100% - 1.1rem); overflow: visible; }
  .st-ax { position: absolute; right: 0; font-size: .72rem; color: var(--muted); font-variant-numeric: tabular-nums; }
  .st-ax.lo { bottom: 1.1rem; } .st-ax.hi { top: -.3rem; }
  .st-ax.d0, .st-ax.d1 { bottom: 0; } .st-ax.d0 { left: 0; right: auto; } .st-ax.d1 { right: 3.6rem; }
  .st-cross { position: absolute; top: 0; bottom: 1.1rem; width: 1px; background: var(--outline); pointer-events: none; display: none; }
  .st-read { font-size: .8rem; color: var(--muted); font-variant-numeric: tabular-nums; min-height: 1.2em; }

  /* lists: watchlist, movers */
  .st-list { display: grid; }
  .st-li { display: grid; grid-template-columns: 4.6rem minmax(2rem, 1fr) auto auto; align-items: center; gap: .55rem;
           padding: .32rem 0; border-top: 1px solid var(--line); font-size: .86rem; }
  .st-li:first-child { border-top: 0; }
  .st-li .p { text-align: right; font-variant-numeric: tabular-nums; }
  .st-li .c { text-align: right; min-width: 3.9rem; font-variant-numeric: tabular-nums; }
  .st-li .sp { height: 1.3rem; position: relative; }
  .st-li .sp svg { position: absolute; inset: 0; width: 100%; height: 100%; overflow: visible; }
  .st-li.wl { grid-template-columns: 4.6rem minmax(2rem, 1fr) auto auto 3.6rem; }
  .st-sig { font-size: .72rem; padding: .08rem .45rem; border-radius: 999px; text-align: center; white-space: nowrap;
            background: color-mix(in srgb, var(--c, var(--muted)) 15%, transparent); color: var(--c, var(--muted)); }
  .st-sig.buy { --c: var(--ok); } .st-sig.sell { --c: var(--err); }
  .st-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; min-height: 0; flex: 1; }
  .st-cols .st-li { grid-template-columns: 1fr auto; }
  .st-sub { color: var(--muted); font-size: .78rem; margin-bottom: .1rem; }

  /* market check, pick, brief */
  .st-head2 { font-size: 1.05rem; font-weight: 500; margin: .1rem 0 .45rem; line-height: 1.3; }
  .st-leg { display: grid; grid-template-columns: 1.1rem 1fr; gap: .1rem .45rem; padding: .28rem 0; font-size: .84rem;
            border-top: 1px solid var(--line); }
  .st-leg:first-child { border-top: 0; }
  .st-leg i { font-style: normal; } .st-leg small { grid-column: 2; color: var(--muted); font-size: .76rem; }
  .st-para { font-size: .86rem; line-height: 1.45; margin: 0 0 .45rem; }
  .st-para b { font-weight: 600; }
  .st-bul { margin: 0; padding-left: 1rem; font-size: .84rem; line-height: 1.45; }
  .st-bul li { margin-bottom: .3rem; }

  .st-opt { display: grid; grid-template-columns: auto minmax(0, 1fr) auto; gap: .15rem .5rem; align-items: center;
            padding: .4rem 0; border-top: 1px solid var(--line); font-size: .86rem; }
  .st-opt:first-child { border-top: 0; }
  .st-opt .st-sym { font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .st-opt .v { font-variant-numeric: tabular-nums; }
  .st-opt .d { grid-column: 2 / -1; font-size: .78rem; }
  .st-foot { color: var(--muted); font-size: .78rem; padding-top: .4rem; border-top: 1px solid var(--line); margin-top: .2rem; }
  .st-tarm { font-size: .84rem; padding: .3rem 0; border-top: 1px solid var(--line); line-height: 1.4; }
  .st-tarm:first-child { border-top: 0; }
  .st-tarm b { font-weight: 500; }

  /* settings dialog */
  .st-dlg { border: 1px solid #ffffff14; border-radius: 1.25rem; background: var(--card); color: var(--ink);
            padding: 1.1rem 1.2rem; width: min(26rem, calc(100vw - 2rem)); }
  .st-dlg::backdrop { background: #000a; }
  .st-dlg h3 { margin: 0 0 .8rem; font-weight: 500; font-size: 1.05rem; }
  .st-dlg label { display: block; margin: .7rem 0 .25rem; font-size: .84rem; color: var(--muted); }
  .st-dlg input[type=text], .st-dlg select, .st-dlg textarea { width: 100%; font: inherit; }
  .st-dlg textarea { min-height: 5rem; }
  .st-dlg .st-radio { display: flex; gap: .45rem; align-items: center; color: var(--ink); margin: .35rem 0; }
  .st-dlg-btns { display: flex; justify-content: flex-end; gap: .5rem; margin-top: 1.1rem; }
  .st-err { color: var(--err); font-size: .82rem; min-height: 1em; margin-top: .4rem; }

  .st-empty { text-align: center; padding: 3rem 1rem; color: var(--muted); border: 1px dashed var(--outline); border-radius: 1.25rem; }
  .st-empty p { margin: 0 0 .9rem; font-size: 1rem; color: var(--ink); }
  .st-empty-btns { display: flex; gap: .5rem; justify-content: center; flex-wrap: wrap; }
  .st-sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); }

  /* Narrower screens: six columns, widgets keep their order but not their exact place, and moving
     is by keyboard or not at all. Phones: one column. */
  @media (max-width: 62rem) {
    .st-grid { --cols: 6; grid-auto-flow: row dense; }
    .st-grid .st-w { grid-column: span var(--wn) !important; grid-row: span var(--h) !important; }
    .st-grid.st-editing { background-image: none; }
    .st-editing .st-grip { display: none; }
  }
  @media (max-width: 40rem) {
    .st-grid { --cols: 1; --rowh: 4.2rem; }
    .st-grid .st-w { grid-column: 1 / -1 !important; }
  }
  @media (prefers-reduced-motion: reduce) { .st-grid { transition: none; } }
"""

JS = r"""
  // ================================================================ station
  const ST_COLS = 12, ST_LS = "station_layout_v1";
  const ST_TYPES = {
    quote:     { title: "Quote", desc: "Price, today's move and today's line", w: 3, h: 2, cfg: { sym: "SPY" } },
    chart:     { title: "Chart", desc: "One symbol over a day, a month, three months or a year", w: 6, h: 4, cfg: { sym: "SPY", range: "3mo" } },
    watchlist: { title: "Watchlist", desc: "Your phone's watchlist or your own list, with the nightly scan's call", w: 4, h: 6, cfg: { src: "synced", syms: [] } },
    gate:      { title: "Market check", desc: "The five checks that decide whether today's pick can buy", w: 3, h: 4, cfg: {} },
    pick:      { title: "Today's pick", desc: "The stock and ETF picks, in the app's words", w: 3, h: 3, cfg: {} },
    movers:    { title: "Movers", desc: "Today's biggest gainers and losers", w: 4, h: 4, cfg: {} },
    brief:     { title: "Market brief", desc: "What the market is doing, in a few lines", w: 4, h: 4, cfg: {} },
    arm:       { title: "Sandbox account", desc: "One paper account: balance, against the S&P, its curve", w: 4, h: 3, cfg: { arm: "main" }, tok: true },
    race:      { title: "Sandbox race", desc: "Which accounts are beating the S&P", w: 4, h: 5, cfg: { group: "all" }, tok: true },
    options:   { title: "Options contracts", desc: "An options account's contracts: value, gain or loss, days left, break-even", w: 4, h: 5, cfg: { arm: "options-rules" }, tok: true },
    today:     { title: "Sandbox today", desc: "What every paper account bought, sold or skipped on the last run", w: 4, h: 4, cfg: {}, tok: true },
    macro:     { title: "World news risk", desc: "The news the AI accounts weigh: wars, rates, trade", w: 4, h: 4, cfg: {} },
  };
  const ST_STARTER = [
    { type: "quote", x: 0, y: 0, cfg: { sym: "SPY" } }, { type: "quote", x: 3, y: 0, cfg: { sym: "QQQ" } },
    { type: "quote", x: 6, y: 0, cfg: { sym: "BTC-USD" } }, { type: "gate", x: 9, y: 0 },
    { type: "watchlist", x: 0, y: 2 }, { type: "chart", x: 4, y: 2, w: 5 },
    { type: "pick", x: 9, y: 4 }, { type: "arm", x: 4, y: 6, w: 5 },
    { type: "race", x: 9, y: 7, w: 3, h: 5 }, { type: "movers", x: 0, y: 8 }, { type: "brief", x: 4, y: 9, w: 5, h: 3 },
  ];
  const ST = { name: "My station", widgets: [], editing: false, sel: null, shown: false, loaded: false };
  const stId = () => Math.random().toString(36).slice(2, 10);
  const stMake = (s) => { const T = ST_TYPES[s.type];
    return { id: stId(), type: s.type, x: s.x || 0, y: s.y || 0, w: s.w || T.w, h: s.h || T.h,
             cfg: Object.assign({}, T.cfg, s.cfg || {}) }; };
  const stStarter = () => ST_STARTER.map(stMake);

  // ---------------------------------------------------------------- data: one fetch per URL, shared
  const ST_CACHE = new Map();
  function stData(url, ttlSec, auth) {
    const c = ST_CACHE.get(url), now = Date.now();
    if (c && (c.pending || now - c.at < ttlSec * 1000)) return c.p;
    const p = (async () => {
      const r = auth ? await afetch(url) : await fetch(url);
      if (!r.ok) { const e = new Error("HTTP " + r.status); e.status = r.status; throw e; }
      return r.json();
    })();
    const entry = { p, at: now, pending: true };
    ST_CACHE.set(url, entry);
    p.then(() => { entry.pending = false; entry.at = Date.now(); },
           () => { entry.pending = false; ST_CACHE.delete(url); });
    return p;
  }
  const ST_RANGES = { "1d": ["1d", "5m", 30], "1mo": ["1mo", "1d", 600], "3mo": ["3mo", "1d", 600], "1y": ["1y", "1d", 1800] };
  function stChart(sym, range, ttlSec) {
    const [r, i, ttl] = ST_RANGES[range] || ST_RANGES["1d"];
    return stData("/prices/yahoo/v8/finance/chart/" + encodeURIComponent(sym) + "?range=" + r + "&interval=" + i, ttlSec || ttl, false)
      .then(j => {
        const res = j && j.chart && j.chart.result && j.chart.result[0];
        if (!res) throw new Error("no data for " + sym);
        const m = res.meta || {}, ts = res.timestamp || [], cl = ((res.indicators || {}).quote || [{}])[0].close || [];
        const pts = []; ts.forEach((t, k) => { if (cl[k] != null) pts.push([t, cl[k]]); });
        const price = m.regularMarketPrice != null ? m.regularMarketPrice : (pts.length ? pts[pts.length - 1][1] : null);
        const prev = m.chartPreviousClose != null ? m.chartPreviousClose : m.previousClose;
        const chg = m.fulldayChangePercent != null ? m.fulldayChangePercent
          : (m.regularMarketChangePercent != null ? m.regularMarketChangePercent : (price != null && prev ? (price / prev - 1) * 100 : null));
        return { sym: m.symbol || sym, name: m.shortName || m.longName || "", price, prev, chg, pts, tz: m.exchangeTimezoneName };
      });
  }
  const stScanMap = () => stData("/scan/latest", 600, false).then(j => {
    const m = {}; ((j && j.results) || []).forEach(r => { m[r.symbol] = r; }); return m; }).catch(() => ({}));

  // ---------------------------------------------------------------- formatting
  const stPx = (v) => v == null ? "—" : v.toLocaleString("en-US", { minimumFractionDigits: v < 1 ? 4 : 2, maximumFractionDigits: v < 1 ? 4 : 2 });
  const stPct = (v) => v == null ? "—" : sgn(v, 2) + "%";
  const stDir = (v) => v == null ? "" : v >= 0 ? "up" : "down";
  // A line: the series scaled into a 100×40 box, with an optional dashed base (yesterday's close,
  // or the S&P for an account) and a faint fill under it.
  function stLine(vals, base, dir, bench) {
    const all = vals.concat(base != null ? [base] : [], bench || []).filter(v => v != null);
    if (vals.length < 2 || !all.length) return "";
    const lo = Math.min(...all), hi = Math.max(...all), sp = (hi - lo) || 1, n = vals.length;
    const X = (k, len) => (k / ((len || n) - 1) * 100).toFixed(2), Y = (v) => (38 - (v - lo) / sp * 36).toFixed(2);
    const pts = vals.map((v, k) => X(k) + "," + Y(v)).join(" ");
    const col = dir === "down" ? "var(--err)" : dir === "up" ? "var(--ok)" : "var(--accent)";
    let s = '<svg viewBox="0 0 100 40" preserveAspectRatio="none" aria-hidden="true">';
    s += '<polygon points="0,40 ' + pts + ' 100,40" fill="' + col + '" fill-opacity=".09"/>';
    if (base != null) s += '<line x1="0" x2="100" y1="' + Y(base) + '" y2="' + Y(base) + '" stroke="var(--bench)" stroke-dasharray="2 3" stroke-width="1" vector-effect="non-scaling-stroke"/>';
    if (bench && bench.length > 1) s += '<polyline points="' + bench.map((v, k) => X(k, bench.length) + "," + Y(v)).join(" ") +
      '" fill="none" stroke="var(--bench)" stroke-dasharray="3 3" stroke-width="1.2" vector-effect="non-scaling-stroke"/>';
    s += '<polyline points="' + pts + '" fill="none" stroke="' + col + '" stroke-width="1.8" stroke-linejoin="round" vector-effect="non-scaling-stroke"/></svg>';
    return s;
  }
  const stLocked = () => '<p class="st-msg">This needs the access token.<br><button type="button" class="secondary sm st-tok">Add the token</button></p>';
  const stFail = (e) => e && (e.status === 401 || e.status === 403) ? stLocked()
    : '<p class="st-msg">Couldn\'t load this' + (e && e.status ? " (HTTP " + e.status + ")" : "") + '. It will try again in a minute.</p>';

  // ---------------------------------------------------------------- widget renderers
  // Each returns { head, sub, body, tint } for its widget; head/sub are plain text, body is HTML.
  const ST_RENDER = {
    async quote(w) {
      const q = await stChart(w.cfg.sym, "1d");
      const d = stDir(q.chg);
      return { head: q.sym, symHead: true, sub: q.name, tint: d,
        body: '<div class="st-row1"><span class="st-big">' + stPx(q.price) + '</span><span class="pill ' + d + '">' + stPct(q.chg) + '</span></div>' +
              '<div class="st-spark">' + stLine(q.pts.map(p => p[1]), q.prev, d) + '</div>' };
    },
    async chart(w) {
      const range = w.cfg.range || "3mo", q = await stChart(w.cfg.sym, range);
      const vals = q.pts.map(p => p[1]);
      const first = vals.length ? vals[0] : null, last = vals.length ? vals[vals.length - 1] : null;
      const move = first && last != null ? (last / first - 1) * 100 : null, d = stDir(range === "1d" ? q.chg : move);
      const lo = vals.length ? Math.min(...vals) : null, hi = vals.length ? Math.max(...vals) : null;
      const fmtT = (t) => new Date(t * 1000).toLocaleString("en-US", range === "1d" ? { hour: "numeric", minute: "2-digit" } : { month: "short", day: "numeric" });
      const seg = Object.keys(ST_RANGES).map(r => '<button type="button" data-range="' + r + '" aria-pressed="' + (r === range) + '">' +
        ({ "1d": "1D", "1mo": "1M", "3mo": "3M", "1y": "1Y" })[r] + '</button>').join("");
      w._pts = q.pts; w._fmtT = fmtT;
      return { head: q.sym, symHead: true, sub: q.name, tint: d, extra: '<div class="st-seg">' + seg + '</div>',
        body: '<div class="st-row1"><span class="st-mid">' + stPx(last) + '</span><span class="pill ' + d + '">' + stPct(range === "1d" ? q.chg : move) +
              '</span><span class="st-read"></span></div>' +
              '<div class="st-plot">' + stLine(vals, range === "1d" ? q.prev : null, d) +
              '<span class="st-ax hi">' + stPx(hi) + '</span><span class="st-ax lo">' + stPx(lo) + '</span>' +
              (q.pts.length ? '<span class="st-ax d0">' + esc(fmtT(q.pts[0][0])) + '</span><span class="st-ax d1">' + esc(fmtT(q.pts[q.pts.length - 1][0])) + '</span>' : '') +
              '<div class="st-cross"></div></div>' };
    },
    async watchlist(w) {
      let syms = (w.cfg.syms || []).slice(), sub = "your list";
      if (w.cfg.src !== "custom") {
        const s = await stData("/api/settings", 300, true);
        syms = (s.watchlist || []).slice(); sub = "from your phone";
      }
      if (!syms.length) return { head: "Watchlist", sub, body: '<p class="st-msg">' + (w.cfg.src === "custom"
        ? "No symbols yet. Open this widget's settings to add some." : "Your phone hasn't synced a watchlist to the server yet.") + '</p>' };
      // Every row is a price fetch, so a long list refreshes each 2 minutes rather than each 30 s.
      const ttl = syms.length > 12 ? 120 : 30;
      const [scan, rows] = await Promise.all([stScanMap(), Promise.all(syms.map(s => stChart(s, "1d", ttl).catch(() => null)))]);
      const html = syms.map((s, k) => {
        const q = rows[k], r = scan[s], d = stDir(q && q.chg);
        const sig = r && r.signal ? '<span class="st-sig ' + esc(r.signal) + '" title="' + esc(r.thesis || "") + '">' +
          esc(r.signal[0].toUpperCase() + r.signal.slice(1)) + (r.conviction != null ? " " + r.conviction : "") + '</span>' : '<span class="st-sig">—</span>';
        return '<div class="st-li wl"><button type="button" class="st-sym" data-sym="' + esc(s) + '">' + esc(s) + '</button>' +
          '<span class="sp">' + (q ? stLine(q.pts.map(p => p[1]), q.prev, d) : "") + '</span>' +
          '<span class="p">' + (q ? stPx(q.price) : "—") + '</span><span class="c ' + d + '-t">' + (q ? stPct(q.chg) : "—") + '</span>' + sig + '</div>';
      }).join("");
      return { head: "Watchlist", sub: sub + " · " + syms.length, body: '<div class="st-scroll"><div class="st-list">' + html + '</div></div>' };
    },
    async gate() {
      const g = await stData("/gate", 300, false);
      if (!g.available) return { head: "Market check", body: '<p class="st-msg">No market check has run yet today.</p>' };
      const legs = g.legs || [], bad = legs.filter(l => !l.ok).length;
      return { head: "Market check", sub: g.as_of ? "as of " + g.as_of.slice(4, 6) + "/" + g.as_of.slice(6) : "", tint: g.passed ? "up" : "down",
        body: '<div class="st-head2">' + (g.passed ? "Clear for today's pick to buy" : "Too risky for new buys: " + bad + " of " + legs.length + " checks fail") + '</div>' +
              '<div class="st-scroll">' + legs.map(l => '<div class="st-leg"><i class="' + (l.ok ? "up-t" : "down-t") + '">' + (l.ok ? "✓" : "✕") + '</i><span>' +
                esc(l.name) + '</span>' + (l.note ? '<small>' + esc(l.note) + '</small>' : '') + '</div>').join("") + '</div>' };
    },
    async pick() {
      const [s, e] = await Promise.all([stData("/daily_pick", 300, false).catch(() => null), stData("/daily_pick?kind=etf", 300, false).catch(() => null)]);
      // A run from an earlier day is never presented as today's pick: it gets no headline and says its date.
      const fresh = (r) => r && r.available && !r.stale && (!r.today || r.date === r.today);
      const line = (r, kind) => {
        if (!r || !r.available) return '<p class="st-para"><b>' + kind + ':</b> no run yet.</p>';
        if (!fresh(r)) return '<p class="st-para"><b>' + kind + ':</b> no run yet today. <span class="amb-t">The last one was ' + esc(r.date) + '.</span></p>';
        const old = '';
        if (r.status === "failed") return '<p class="st-para"><b>' + kind + ':</b> the run failed.' + old + '</p>';
        if (r.status === "pick" && r.pick) {
          const p = r.pick;
          return '<p class="st-para"><b>' + kind + ': <button type="button" class="st-sym" data-sym="' + esc(p.symbol) + '">' + esc(p.symbol) + '</button></b>' +
            (p.conviction != null ? ' <span class="hint">confidence ' + esc(p.conviction) + '</span>' : '') + old + '<br>' + esc(p.headline || p.thesis || "") + '</p>';
        }
        const m = /^(\S+) scored (\d+)\/100; it needed (\d+)/.exec(r.none_reason || "");
        const why = m ? m[1] + " came close (" + m[2] + ", needs " + m[3] + ")." :
          (r.gate && r.gate.passed === false ? "The market looks too risky today." : "Nothing scored high enough.");
        return '<p class="st-para"><b>' + kind + ':</b> none. ' + esc(why) + old + '</p>';
      };
      const sp = fresh(s) && s.status === "pick" && s.pick ? s.pick.symbol : null, ep = fresh(e) && e.status === "pick" && e.pick ? e.pick.symbol : null;
      const anyToday = fresh(s) || fresh(e);
      const head = sp && ep ? "Picks: " + sp + " and " + ep : sp ? "Stock pick: " + sp : ep ? "ETF pick: " + ep : anyToday ? "No picks today" : "Today's run hasn't happened yet";
      return { head: "Today's pick", tint: sp || ep ? "up" : "", body: '<div class="st-head2">' + esc(head) + '</div><div class="st-scroll">' + line(s, "Stock") + line(e, "ETF") + '</div>' };
    },
    async movers() {
      const m = await stData("/movers", 120, false);
      const col = (xs, t) => '<div class="st-scroll"><div class="st-sub">' + t + '</div><div class="st-list">' + (xs || []).slice(0, 10).map(x =>
        '<div class="st-li"><button type="button" class="st-sym" data-sym="' + esc(x.symbol) + '">' + esc(x.symbol) + '</button><span class="c ' +
        stDir(x.change_percent) + '-t">' + stPct(x.change_percent) + '</span></div>').join("") + '</div></div>';
      return { head: "Movers", sub: "today", body: '<div class="st-cols">' + col(m.gainers, "Up most") + col(m.losers, "Down most") + '</div>' };
    },
    async brief() {
      const b = await stData("/market_now", 600, false);
      const st = b.overview_struct || {}, lines = String(b.overview || "").split("\n").map(x => x.trim()).filter(Boolean);
      const head = st.headline || lines[0] || "";
      const bul = lines.filter(x => x.startsWith("•")).map(x => x.replace(/^•\s*/, ""));
      const when = b.snapshot && b.snapshot.as_of_et ? b.snapshot.as_of_et.replace(/^\d{4}-\d\d-\d\d /, "") : "";
      return { head: "Market brief", sub: when,
        body: '<div class="st-head2">' + esc(head) + '</div><div class="st-scroll"><ul class="st-bul">' + bul.map(x => '<li>' + esc(x) + '</li>').join("") + '</ul></div>' };
    },
    async arm(w) {
      const list = await stData("/sandbox/arms", 60, true);
      const a = ((list && list.arms) || []).find(x => x.arm === w.cfg.arm);
      if (!a) return { head: "Sandbox account", body: '<p class="st-msg">The account “' + esc(w.cfg.arm) + '” no longer exists. Pick another in this widget\'s settings.</p>' };
      const nav = await stData("/sandbox/nav?arm=" + encodeURIComponent(a.arm), 600, true).catch(() => ({ series: [] }));
      const ser = (nav.series || []).slice(-66), v = vsShown(a), d = stDir(v);
      return { head: shortLabel(a), sub: a.engine === "rules" ? "Rules" : a.engine === "rejects" ? "Cut trades" : "AI", tint: d, open: a.arm,
        body: '<div class="st-row1"><span class="st-big">' + usd(a.equity) + '</span>' +
              '<span class="pill ' + d + '">' + (v == null ? "not run yet" : sgn(v) + " pts vs S&amp;P") + '</span>' +
              (a.total_return_pct != null ? '<span class="pill ' + stDir(a.total_return_pct) + '">' + stPct(a.total_return_pct) + ' on ' + usd(a.funded_total, 0) + '</span>' : '') + '</div>' +
              '<div class="st-spark">' + stLine(ser.map(p => p.equity), null, d, ser.map(p => p.benchmark_value)) + '</div>' };
    },
    async race(w) {
      const list = await stData("/sandbox/arms", 60, true);
      const g = w.cfg.group || "all", xs = ((list && list.arms) || []).filter(a => groupOf(a.universe) === g).sort((a, b) => {
        const x = vsShown(a), y = vsShown(b); if ((x == null) !== (y == null)) return x == null ? 1 : -1; return (y || 0) - (x || 0); });
      const scale = Math.max(1, ...xs.map(a => Math.abs(vsShown(a) || 0)));
      const name = { all: "Stocks", etf: "ETFs", options: "Options" }[g] || g;
      return { head: "Sandbox race", sub: name + ", points vs the S&P", after: (box) => growBars(box, false),
        body: '<div class="st-scroll st-race">' + xs.map(a => vsRow(shortLabel(a), vsShown(a), scale, a.arm, false, a.enabled ? "" : "paused")).join("") + '</div>' };
    },
  };

  Object.assign(ST_RENDER, {
    async options(w) {
      // /sandbox/state prices live option chains on the server, so it is read every 5 minutes at most.
      const st = await stData("/sandbox/state?arm=" + encodeURIComponent(w.cfg.arm), 300, true);
      if (st.universe && st.universe !== "options")
        return { head: "Options contracts", body: '<p class="st-msg">“' + esc(w.cfg.arm) + '” is not an options account. Pick one in this widget\'s settings.</p>' };
      const cs = st.contracts || [], wk = st.weekly_budget;
      const rows = cs.map(c => {
        const stale = !!c.mark_stale, pl = stale ? null : c.pl_pct;
        const be = c.premium_paid == null ? null : c.right === "put" ? c.strike - c.premium_paid : c.strike + c.premium_paid;
        const sub = [pl == null ? "no live quote" + (c.mark_date ? " since " + shortDate(c.mark_date) : "") : sgn(pl, 1) + "%",
          Math.floor(c.dte || 0) + " days left",
          be != null ? "needs " + (c.right === "put" ? "below " : "above ") + usd(be) : null].filter(Boolean).join(" · ");
        return '<div class="st-opt"><span class="pill ' + (c.right === "put" ? "down" : "up") + '">' + esc(String(c.right).toUpperCase()) + '</span>' +
          '<button type="button" class="st-sym" data-sym="' + esc(c.underlying) + '">' + esc(c.count + " × " + c.display.replace(/ exp .*/, "")) + '</button>' +
          '<span class="v' + (stale ? " amb-t" : "") + '">' + (c.value == null ? "—" : (stale ? "≈" : "") + usd(c.value, 0)) + '</span>' +
          '<span class="d ' + (pl == null ? "amb-t" : pl >= 0 ? "up-t" : "down-t") + '">' + esc(sub) + '</span></div>';
      }).join("");
      return { head: st.label || w.cfg.arm, sub: cs.length + " held", open: w.cfg.arm,
        body: '<div class="st-scroll">' + (rows || '<p class="st-msg">No contracts held.</p>') + '</div>' +
              (wk ? '<div class="st-foot">' + esc(weekLine(wk)) + '</div>' : '') };
    },
    async today() {
      const t = await stData("/sandbox/today?show_options=true", 120, true);
      if (!t || !t.arms) return { head: "Sandbox today", body: '<p class="st-msg">No run yet. Accounts trade at 2:35 PM CT on market days.</p>' };
      const tot = t.totals || {};
      const held = t.arms.filter(a => a.ran && !(a.filled || []).length && !(a.skipped || []).length);
      const acted = t.arms.filter(a => (a.filled || []).length || (a.skipped || []).length);
      const when = (t.date ? shortDate(t.date) : "") + (t.ran_at ? ", " + new Date(t.ran_at * 1000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) : "");
      const pills = (tot.buys ? '<span class="pill up">' + tot.buys + ' bought, ' + usd(tot.bought, 0) + '</span>' : '') +
        (tot.sells ? '<span class="pill down">' + tot.sells + ' sold, ' + usd(tot.sold, 0) + '</span>' : '') +
        (tot.skipped ? '<span class="pill amb">' + tot.skipped + ' skipped</span>' : '') + '<span class="pill">' + held.length + ' held</span>';
      const list = acted.map(a => '<div class="st-tarm"><b>' + esc(shortLabel(a)) + '</b> ' +
        (a.filled || []).map(f => '<span class="' + (f.side === "sell" ? "down-t" : "up-t") + '">' + esc((f.side === "sell" ? "sold " : "") + String(f.symbol || "")) + '</span>').join(", ") +
        ((a.skipped || []).length ? ' <span class="amb-t">skipped ' + esc((a.skipped || []).map(s => String(s.symbol || "").split(" ")[0]).join(", ")) + '</span>' : '') + '</div>').join("");
      return { head: "Sandbox today", sub: when, body: '<div class="pills" style="margin:0 0 .5rem">' + pills + '</div><div class="st-scroll">' + list + '</div>' };
    },
    async macro() {
      const m = await stData("/macro/catalysts", 900, false);
      if (!m.available) return { head: "World news risk", body: '<p class="st-msg">No news read yet.</p>' };
      const lvl = String(m.risk_level || "");
      const tint = /high|severe|elevated/.test(lvl) ? "down" : /low|calm/.test(lvl) ? "up" : "";
      return { head: "World news risk", sub: (m.stale ? "out of date, " : "") + (m.age_seconds == null ? "" : m.age_seconds < 3600 ? Math.max(1, Math.round(m.age_seconds / 60)) + " min old" : Math.round(m.age_seconds / 3600) + " h old"), tint,
        body: '<div class="st-row1" style="margin-bottom:.35rem"><span class="pill ' + (tint === "down" ? "amb" : tint) + '">' + esc(lvl ? lvl[0].toUpperCase() + lvl.slice(1) : "—") + ' risk</span></div>' +
              '<div class="st-head2">' + esc(m.headline || "") + '</div>' +
              '<div class="st-scroll"><ul class="st-bul">' + (m.bullets || []).map(x => '<li>' + esc(x) + '</li>').join("") + '</ul></div>' };
    },
  });

  // ---------------------------------------------------------------- render
  function stPlace(el, w) {
    el.style.gridColumn = (w.x + 1) + " / span " + w.w;
    el.style.gridRow = (w.y + 1) + " / span " + w.h;
    el.style.setProperty("--h", w.h);
    el.style.setProperty("--wn", w.w >= 6 ? 6 : 3);
  }
  function stShell(w) {
    const el = document.createElement("div");
    el.className = "st-w"; el.id = "stw-" + w.id; el.dataset.id = w.id;
    el.innerHTML = '<div class="st-h"><span class="t"></span><span class="s"></span></div><div class="st-b"><p class="st-msg">Loading…</p></div>' +
      '<div class="st-wbar"><button type="button" class="st-cfg" title="Settings" aria-label="Settings">⚙</button>' +
      '<button type="button" class="st-del" title="Remove" aria-label="Remove">✕</button></div><div class="st-grip" aria-hidden="true"></div>';
    stPlace(el, w);
    return el;
  }
  function stLabel(w) { const T = ST_TYPES[w.type]; return T.title + (w.cfg.sym ? " " + w.cfg.sym : w.cfg.arm ? " " + w.cfg.arm : ""); }
  function stRender() {
    const grid = $("st-grid");
    grid.classList.toggle("st-editing", ST.editing);
    const keep = new Set(ST.widgets.map(w => "stw-" + w.id));
    [...grid.children].forEach(c => { if (!keep.has(c.id)) c.remove(); });
    ST.widgets.slice().sort((a, b) => a.y - b.y || a.x - b.x).forEach(w => {
      let el = $("stw-" + w.id);
      if (!el) { el = stShell(w); grid.appendChild(el); stFill(w); }
      stPlace(el, w);
      el.tabIndex = ST.editing ? 0 : -1;
      el.setAttribute("aria-label", stLabel(w));
      el.classList.toggle("st-sel", ST.editing && ST.sel === w.id);
      grid.appendChild(el);            // DOM order follows reading order, for the narrow layouts and screen readers
    });
    $("st-empty").hidden = ST.widgets.length > 0;
    grid.hidden = ST.widgets.length === 0 && !ST.editing;
    $("st-name").textContent = ST.name;
  }
  async function stFill(w) {
    const el = $("stw-" + w.id); if (!el) return;
    const T = ST_TYPES[w.type];
    let out;
    try { out = await ST_RENDER[w.type](w); }
    catch (e) { out = { head: (w.cfg.title || T.title) + (w.cfg.sym ? " " + w.cfg.sym : ""), body: stFail(e) }; }
    if (!$("stw-" + w.id)) return;
    const h = el.querySelector(".st-h");
    const title = w.cfg.title || out.head || T.title;   // out.head null: the widget's own name
    h.innerHTML = (out.symHead && !w.cfg.title ? '<button type="button" class="st-sym t" data-sym="' + esc(out.head) + '">' + esc(out.head) + '</button>'
      : '<span class="t">' + esc(title) + '</span>') + '<span class="s">' + esc(out.sub || "") + '</span>' + (out.extra || "");
    const body = el.querySelector(".st-b");
    if (body._html !== out.body) {
      const sc = body.querySelector(".st-scroll"), top = sc ? sc.scrollTop : 0;
      body.innerHTML = out.body; body._html = out.body;
      const sc2 = body.querySelector(".st-scroll"); if (sc2 && top) sc2.scrollTop = top;
    }
    el.classList.remove("up", "down"); if (out.tint) el.classList.add(out.tint);
    el.dataset.open = out.open || "";
    if (out.after) out.after(el);
  }
  function stFillAll() { ST.widgets.forEach(stFill); }

  // ---------------------------------------------------------------- clicks inside widgets
  $("st-grid").addEventListener("click", (ev) => {
    const t = ev.target, el = t.closest(".st-w"); if (!el) return;
    const w = ST.widgets.find(x => x.id === el.dataset.id); if (!w) return;
    if (t.closest(".st-del")) { stRemove(w); return; }
    if (t.closest(".st-cfg")) { stConfigure(w); return; }
    if (t.closest(".st-tok")) { showTokenPrompt(false); return; }
    if (ST.editing) { ST.sel = w.id; stRender(); el.focus(); return; }
    const sym = t.closest(".st-sym"); if (sym && sym.dataset.sym) { openTickerDrawer(sym.dataset.sym); return; }
    const rng = t.closest(".st-seg button"); if (rng) { w.cfg.range = rng.dataset.range; stSave(); stFill(w); return; }
    const row = t.closest(".vs-row"); if (row) { stOpenArm(row.dataset.arm); return; }
    if (el.dataset.open) stOpenArm(el.dataset.open);
  });
  function stOpenArm(id) {
    const a = (typeof ARMS !== "undefined" ? ARMS : []).find(x => x.arm === id);
    showTab("p-sandbox");
    if (a) { pickGroup(groupOf(a.universe)); pickArm(id); }
    window.scrollTo(0, 0);
  }
  // the chart's pointer readout
  $("st-grid").addEventListener("pointermove", (ev) => {
    if (ST.editing) return;
    const plot = ev.target.closest && ev.target.closest(".st-plot"); if (!plot) return;
    const el = plot.closest(".st-w"), w = ST.widgets.find(x => x.id === el.dataset.id); if (!w || !w._pts || !w._pts.length) return;
    const svg = plot.querySelector("svg"); if (!svg) return;
    const r = svg.getBoundingClientRect(), f = Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width));
    const k = Math.round(f * (w._pts.length - 1)), p = w._pts[k];
    const cross = plot.querySelector(".st-cross"); cross.style.display = "block"; cross.style.left = (f * r.width) + "px";
    el.querySelector(".st-read").textContent = stPx(p[1]) + " at " + w._fmtT(p[0]);
  });
  $("st-grid").addEventListener("pointerout", (ev) => {
    const plot = ev.target.closest && ev.target.closest(".st-plot"); if (!plot || plot.contains(ev.relatedTarget)) return;
    const c = plot.querySelector(".st-cross"); if (c) c.style.display = "none";
    const rd = plot.closest(".st-w").querySelector(".st-read"); if (rd) rd.textContent = "";
  });

  // ---------------------------------------------------------------- layout maths
  const stHit = (a, b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
  // Put `moved` where it is, push anything it overlaps downward, then let everything float up into
  // free space, so the station never has holes nobody made on purpose.
  function stResolve(moved) {
    const placed = moved ? [moved] : [];
    ST.widgets.filter(w => w !== moved).sort((a, b) => a.y - b.y || a.x - b.x).forEach(o => {
      while (placed.some(p => stHit(p, o))) o.y++;
      placed.push(o);
    });
    const done = [];
    ST.widgets.slice().sort((a, b) => a.y - b.y || a.x - b.x).forEach(o => {
      while (o.y > 0 && !done.some(p => stHit(p, { x: o.x, y: o.y - 1, w: o.w, h: o.h }))) o.y--;
      done.push(o);
    });
  }
  function stFreeSpot(w, h) {
    for (let y = 0; y < 200; y++) for (let x = 0; x + w <= ST_COLS; x++) {
      const c = { x, y, w, h }; if (!ST.widgets.some(o => stHit(o, c))) return { x, y };
    }
    return { x: 0, y: 0 };
  }
  const stWide = () => window.matchMedia("(min-width: 62.01rem)").matches;
  function stMetrics() {
    const g = $("st-grid"), cs = getComputedStyle(g), gap = parseFloat(cs.columnGap) || 16;
    const r = g.getBoundingClientRect(), cw = (r.width - gap * (ST_COLS - 1)) / ST_COLS;
    return { r, gap, px: cw + gap, py: parseFloat(cs.gridAutoRows) + gap };
  }

  // ---------------------------------------------------------------- drag to move, corner to resize
  let stDrag = null;
  $("st-grid").addEventListener("pointerdown", (ev) => {
    if (!ST.editing || !stWide() || ev.button !== 0) return;
    const el = ev.target.closest(".st-w"); if (!el || ev.target.closest(".st-wbar")) return;
    const w = ST.widgets.find(x => x.id === el.dataset.id); if (!w) return;
    ev.preventDefault();
    const ghost = document.createElement("div"); ghost.className = "st-ghost"; $("st-grid").appendChild(ghost);
    stDrag = { w, el, ghost, resize: !!ev.target.closest(".st-grip"), sx: ev.clientX, sy: ev.clientY,
               x0: w.x, y0: w.y, w0: w.w, h0: w.h, m: stMetrics(), nx: w.x, ny: w.y, nw: w.w, nh: w.h };
    ST.sel = w.id; el.classList.add("st-sel", "st-drag"); el.setPointerCapture(ev.pointerId);
    stPlace(ghost, w);
  });
  $("st-grid").addEventListener("pointermove", (ev) => {
    if (!stDrag) return;
    const d = stDrag, dx = ev.clientX - d.sx, dy = ev.clientY - d.sy;
    if (d.resize) {
      d.nw = Math.max(1, Math.min(ST_COLS - d.x0, d.w0 + Math.round(dx / d.m.px)));
      d.nh = Math.max(1, Math.min(12, d.h0 + Math.round(dy / d.m.py)));
      stPlace(d.ghost, { x: d.x0, y: d.y0, w: d.nw, h: d.nh });
    } else {
      d.nx = Math.max(0, Math.min(ST_COLS - d.w0, d.x0 + Math.round(dx / d.m.px)));
      d.ny = Math.max(0, d.y0 + Math.round(dy / d.m.py));
      d.el.style.transform = "translate(" + dx + "px," + dy + "px)";
      stPlace(d.ghost, { x: d.nx, y: d.ny, w: d.w0, h: d.h0 });
    }
  });
  function stDragEnd() {
    if (!stDrag) return;
    const d = stDrag; stDrag = null;
    d.ghost.remove(); d.el.classList.remove("st-drag"); d.el.style.transform = "";
    if (d.resize) { d.w.w = d.nw; d.w.h = d.nh; } else { d.w.x = d.nx; d.w.y = d.ny; }
    const changed = d.w.x !== d.x0 || d.w.y !== d.y0 || d.w.w !== d.w0 || d.w.h !== d.h0;
    stResolve(d.w); stRender();
    if (changed) { stSave(); stSay(stLabel(d.w) + (d.resize ? " resized to " + d.w.w + " by " + d.w.h : " moved to column " + (d.w.x + 1) + ", row " + (d.w.y + 1))); }
  }
  $("st-grid").addEventListener("pointerup", stDragEnd);
  $("st-grid").addEventListener("pointercancel", stDragEnd);

  // ---------------------------------------------------------------- keyboard
  $("st-grid").addEventListener("keydown", (ev) => {
    if (!ST.editing) return;
    const el = ev.target.closest(".st-w"); if (!el || ev.target !== el) return;
    const w = ST.widgets.find(x => x.id === el.dataset.id); if (!w) return;
    const k = ev.key, step = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[k];
    if (step) {
      ev.preventDefault();
      if (ev.shiftKey) { w.w = Math.max(1, Math.min(ST_COLS - w.x, w.w + step[0])); w.h = Math.max(1, Math.min(12, w.h + step[1])); }
      else { w.x = Math.max(0, Math.min(ST_COLS - w.w, w.x + step[0])); w.y = Math.max(0, w.y + step[1]);
             if (step[1] > 0) { const under = ST.widgets.filter(o => o !== w && stHit(o, w)); if (under.length) w.y = Math.max(...under.map(o => o.y + o.h)); } }
      stResolve(w); ST.sel = w.id; stRender(); $("stw-" + w.id).focus(); stSave();
      stSay(stLabel(w) + ": column " + (w.x + 1) + ", row " + (w.y + 1) + ", " + w.w + " wide, " + w.h + " tall");
    } else if (k === "Delete" || k === "Backspace") { ev.preventDefault(); stRemove(w); }
    else if (k === "Enter") { ev.preventDefault(); stConfigure(w); }
  });
  $("st-grid").addEventListener("focusin", (ev) => {
    const el = ev.target.classList && ev.target.classList.contains("st-w") ? ev.target : null;
    if (el && ST.editing && ST.sel !== el.dataset.id) { ST.sel = el.dataset.id;
      document.querySelectorAll(".st-w.st-sel").forEach(x => x.classList.remove("st-sel")); el.classList.add("st-sel"); }
  });
  function stSay(t) { $("st-live").textContent = t; }

  // ---------------------------------------------------------------- add, remove, configure
  function stAdd(type) {
    if (ST.widgets.length >= 40) { $("st-saved").textContent = "A station holds at most 40 widgets. Remove one first."; return; }
    const T = ST_TYPES[type], spot = stFreeSpot(T.w, T.h);
    const w = stMake({ type, x: spot.x, y: spot.y });
    ST.widgets.push(w); ST.sel = w.id; stResolve(w); stRender(); stSave();
    const el = $("stw-" + w.id); el.scrollIntoView({ block: "nearest", behavior: "smooth" });
    if (["quote", "chart", "watchlist", "arm", "race", "options"].includes(type)) stConfigure(w, true); else el.focus();
  }
  function stRemove(w) {
    ST.widgets = ST.widgets.filter(x => x !== w); if (ST.sel === w.id) ST.sel = null;
    stResolve(null); stRender(); stSave(); stSay(stLabel(w) + " removed");
  }
  let stDlgW = null, stDlgNew = false;
  async function stConfigure(w, isNew) {
    stDlgW = w; stDlgNew = !!isNew;
    const T = ST_TYPES[w.type], c = w.cfg;
    $("st-dlg-title").textContent = (isNew ? "New " : "") + T.title.toLowerCase().replace(/^./, s => s.toUpperCase());
    let h = "";
    if (w.type === "quote" || w.type === "chart") {
      h += '<label for="st-f-sym">Symbol</label><input type="text" id="st-f-sym" value="' + esc(c.sym || "") + '" autocomplete="off" spellcheck="false" placeholder="AAPL, SPY, BTC-USD">';
    }
    if (w.type === "chart") {
      h += '<label for="st-f-range">Range</label><select id="st-f-range">' + [["1d", "Today"], ["1mo", "1 month"], ["3mo", "3 months"], ["1y", "1 year"]]
        .map(([v, t]) => '<option value="' + v + '"' + (c.range === v ? " selected" : "") + '>' + t + '</option>').join("") + '</select>';
    }
    if (w.type === "watchlist") {
      h += '<div class="st-radio"><input type="radio" name="st-src" id="st-src-s" value="synced"' + (c.src !== "custom" ? " checked" : "") +
           '><label for="st-src-s" style="margin:0">The watchlist your phone syncs</label></div>' +
           '<div class="st-radio"><input type="radio" name="st-src" id="st-src-c" value="custom"' + (c.src === "custom" ? " checked" : "") +
           '><label for="st-src-c" style="margin:0">These symbols</label></div>' +
           '<textarea id="st-f-syms" placeholder="AAPL MSFT NVDA">' + esc((c.syms || []).join(" ")) + '</textarea>';
    }
    if (w.type === "arm" || w.type === "race" || w.type === "options") {
      let arms = [];
      try { arms = ((await stData("/sandbox/arms", 60, true)).arms) || []; } catch (e) {}
      if (w.type === "options") arms = arms.filter(a => a.universe === "options");
      if (w.type === "arm" || w.type === "options") {
        h += '<label for="st-f-arm">Account</label><select id="st-f-arm">' + (arms.length ? arms.map(a => '<option value="' + esc(a.arm) + '"' +
          (a.arm === c.arm ? " selected" : "") + '>' + esc(shortLabel(a)) + '</option>').join("") : '<option value="' + esc(c.arm) + '">' + esc(c.arm) + '</option>') + '</select>';
      } else {
        h += '<label for="st-f-group">Group</label><select id="st-f-group">' + [["all", "Stocks"], ["etf", "ETFs"], ["options", "Options"]]
          .map(([v, t]) => '<option value="' + v + '"' + (c.group === v ? " selected" : "") + '>' + t + '</option>').join("") + '</select>';
      }
    }
    h += '<label for="st-f-title">Name on the widget <span class="hint">(optional)</span></label><input type="text" id="st-f-title" maxlength="40" value="' + esc(c.title || "") + '">';
    h += '<div class="st-err" id="st-f-err"></div>';
    $("st-dlg-body").innerHTML = h;
    $("st-dlg").showModal();
    const first = $("st-dlg-body").querySelector("input[type=text], select"); if (first) first.focus();
  }
  const stCleanSym = (s) => String(s || "").trim().toUpperCase().replace(/[^A-Z0-9.^=\-]/g, "").slice(0, 15);
  $("st-form").addEventListener("submit", (ev) => {
    const w = stDlgW; if (!w) return;
    const c = Object.assign({}, w.cfg), err = $("st-f-err");
    if ($("st-f-sym")) { const s = stCleanSym($("st-f-sym").value); if (!s) { ev.preventDefault(); err.textContent = "Type a symbol, like AAPL."; return; } c.sym = s; }
    if ($("st-f-range")) c.range = $("st-f-range").value;
    if ($("st-f-syms")) {
      c.src = $("st-src-c").checked ? "custom" : "synced";
      c.syms = [...new Set($("st-f-syms").value.split(/[\s,;]+/).map(stCleanSym).filter(Boolean))].slice(0, 60);
      if (c.src === "custom" && !c.syms.length) { ev.preventDefault(); err.textContent = "Add at least one symbol, or use the synced watchlist."; return; }
    }
    if ($("st-f-arm")) c.arm = $("st-f-arm").value;
    if ($("st-f-group")) c.group = $("st-f-group").value;
    const t = $("st-f-title").value.trim(); if (t) c.title = t; else delete c.title;
    w.cfg = c; stDlgW = null; stSave(); stFill(w); stRender();
    setTimeout(() => { const el = $("stw-" + w.id); if (el && ST.editing) el.focus(); }, 0);
  });
  $("st-dlg-cancel").onclick = () => { $("st-dlg").close(); stDlgW = null; };

  // ---------------------------------------------------------------- edit mode and the tray
  $("st-tray").innerHTML = Object.entries(ST_TYPES).map(([k, T]) =>
    '<button type="button" class="st-type" data-type="' + k + '"><b>' + esc(T.title) + '</b><span>' + esc(T.desc) +
    (T.tok ? ' <span class="st-lock">Needs the token.</span>' : '') + '</span></button>').join("");
  $("st-tray").onclick = (ev) => { const b = ev.target.closest(".st-type"); if (b) stAdd(b.dataset.type); };
  function stSetEditing(on) {
    ST.editing = on; if (!on) ST.sel = null;
    $("st-edit").setAttribute("aria-pressed", String(on)); $("st-edit").textContent = on ? "Done" : "Edit layout";
    ["st-add", "st-reset", "st-help"].forEach(id => $(id).hidden = !on);
    $("st-tray").hidden = true;
    $("st-name").hidden = on; $("st-name-in").hidden = !on; $("st-name-in").value = ST.name;
    stRender();
  }
  $("st-edit").onclick = () => stSetEditing(!ST.editing);
  $("st-add").onclick = () => { $("st-tray").hidden = !$("st-tray").hidden; };
  $("st-reset").onclick = () => {
    if (ST.widgets.length && !confirm("Replace this station with the starter layout? Your widgets will be removed.")) return;
    ST.widgets = stStarter(); stRender(); stSave();
  };
  $("st-name-in").onchange = () => { ST.name = $("st-name-in").value.trim() || "My station"; stSave(); };
  $("st-empty-add").onclick = () => { stSetEditing(true); $("st-tray").hidden = false; };
  $("st-empty-starter").onclick = () => { ST.widgets = stStarter(); stRender(); stSave(); };

  // ---------------------------------------------------------------- saving and loading
  let stSaveT = null;
  function stDoc() { return { name: ST.name, widgets: ST.widgets.map(w => ({ id: w.id, type: w.type, x: w.x, y: Math.min(w.y, 60), w: w.w, h: w.h, cfg: w.cfg })) }; }
  function stSave() {
    const doc = stDoc();
    try { localStorage.setItem(ST_LS, JSON.stringify(doc)); } catch (e) {}
    clearTimeout(stSaveT);
    stSaveT = setTimeout(async () => {
      if (!apiToken()) { $("st-saved").textContent = "Saved in this browser only. Add the access token to save it on the server."; return; }
      $("st-saved").textContent = "Saving…";
      try {
        const r = await afetch("/station/layout", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(doc) });
        if (r.ok) $("st-saved").textContent = "Saved";
        else { let m = "HTTP " + r.status; try { m = (await r.json()).detail || m; } catch (e) {} $("st-saved").textContent = "Not saved on the server: " + m; }
      } catch (e) { $("st-saved").textContent = "Saved in this browser only: the server didn't answer."; }
    }, 700);
  }
  async function stLoad() {
    let doc = null, local = null;
    try { local = JSON.parse(localStorage.getItem(ST_LS) || "null"); } catch (e) {}
    let pushLocal = false;
    if (apiToken()) {
      try { const r = await afetch("/station/layout"); if (r.ok) { const j = await r.json(); if (j.saved) doc = j; else pushLocal = !!local; } } catch (e) {}
    }
    doc = doc || local;
    ST.name = (doc && doc.name) || "My station";
    ST.widgets = doc && Array.isArray(doc.widgets) ? doc.widgets.filter(w => ST_TYPES[w.type])
      .map(w => ({ id: w.id || stId(), type: w.type, x: w.x | 0, y: w.y | 0, w: w.w || ST_TYPES[w.type].w, h: w.h || ST_TYPES[w.type].h,
                   cfg: Object.assign({}, ST_TYPES[w.type].cfg, w.cfg || {}) })) : stStarter();
    stResolve(null); ST.loaded = true; stRender();
    // A layout made before the token was added lives only in this browser: put it on the server now.
    if (pushLocal) stSave();
  }

  // ---------------------------------------------------------------- start and refresh
  document.addEventListener("tabshown", (ev) => {
    if (ev.detail !== "p-station") return;
    if (!ST.loaded) stLoad(); else if (!stDrag) stFillAll();
  });
  if (!$("p-station").hidden) stLoad();
  setInterval(() => {
    if (!ST.loaded || $("p-station").hidden || document.hidden || stDrag || $("st-dlg").open) return;
    stFillAll();
  }, 30000);
"""
