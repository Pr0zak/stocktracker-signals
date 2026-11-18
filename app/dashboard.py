"""The operations dashboard served at `/`.

Lifted out of main.py, where it sat as a 500-line string literal in the middle of the route
definitions. It is a page, not a route, and having it inline meant every read of main.py scrolled
through markup to get to the API.

Layout notes, since the previous version's problems were structural rather than cosmetic:

* It used CSS `columns` for a masonry effect. Multi-column flows content DOWN column one and then
  wraps into column two, so the reading order was "Status, AI usage, Latest scan" on the left and
  whatever happened to spill on the right — an order nobody chose, that changed with card height,
  and that broke differently at every window width. The settings <form> spanned two of those cards,
  so its Save button could land in a different column from the fields it saved. This is a plain
  grid with an explicit order instead.
* Nine cards at identical visual weight, all on screen at once, with no grouping. Now four TABS —
  Overview, Sandbox, Scan, Settings — so each view answers one question.
* The sandbox was absent entirely: the arms, their scoreboard and their settings existed only over
  the API. That was the actual functionality gap, and it is why the model could be set per arm but
  not seen or changed anywhere.

Element ids are deliberately unchanged from the previous version wherever the behaviour is
unchanged, so the existing script keeps driving them.
"""
from __future__ import annotations

from . import dashboard_sandbox as _sbx

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StockTracker Signals</title>
<style>
  /* Colour roles, mobile-first spacing, one breakpoint. Two of the old values were failing contrast
     outright and most of this page's actual information is rendered in them:
       --muted #8a8a8a is 3.5:1 on white — under the 4.5:1 floor — and it carries every hint, table
         header, stat label and log row, i.e. most of the words on the page.
       --accent #2563eb is ~3.4:1 against a dark ground and is used as TEXT (the selected tab, the
         headline figures on the usage and cost cards), so dark mode was the failing case.
     The replacements are 6.0:1 / 5.9:1 and 6.9:1 / 7.4:1 against their own grounds. Names are kept
     so the ~40 existing references inherit the fix with no call-site edits. */
  /* The phone app's dark palette (app ui/theme/Color.kt), dark only since 2026-11-18: the user reads
     StockTracker in dark mode and asked for this page to look like the app. Contrast against the
     card #1C232C: ink 13.2:1, muted 7.0:1, indigo 7.8:1, gain 10.2:1, loss 5.3:1, amber 5.3:1. */
  :root {
    color-scheme: dark;
    --accent:#B4A0FF; --ok:#5EDD9C; --err:#E8663F; --warn:#D3A23F; --amber:#C99A35;
    --muted:#A3ADBA; --line:#2C343F; --outline:#6B7787; --card:#1C232C; --card2:#151A21; --bg:#0D1116; --ink:#E7EBF0;
    --on-accent:#2A1E52; --chip-sel:#4A4458; --bench:#8A94A2;
    --sans: Roboto, "Segoe UI", system-ui, -apple-system, sans-serif;
    --mono: "Roboto Mono", "Noto Sans Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    --radius:1rem;
    /* Spacing and density scale with the viewport rather than being fixed at desktop values. */
    --pad:.8rem; --gap:.7rem; --card-pad:.85rem .9rem; --cell-pad:.34rem .45rem;
  }
  @media (min-width: 46rem) {
    :root { --pad:1rem; --gap:1rem; --card-pad:1rem 1.1rem; --cell-pad:.42rem .6rem; }
  }
  * { box-sizing: border-box; }
  /* The header is sticky, so any anchor jump — every tab switch writes a #hash — lands the target
     UNDER it. Without this the first card's stat labels sit behind the tab bar, which is exactly
     where "Last scan" was hiding. 8rem clears the title, subtitle and tabs at phone widths, where
     the header is tallest because the tab strip wraps closest to the text above it. */
  html { scroll-padding-top: 8rem; }
  body { font-family: var(--sans); margin: 0; line-height: 1.5;
         background: var(--bg); color: var(--ink); }
  .wrap { max-width: 82rem; margin: 0 auto; padding: 0 var(--pad) 3rem; }

  /* header + tabs: sticky, so switching views never needs a scroll back up */
  header { position: sticky; top: 0; z-index: 20; background: var(--bg);
           border-bottom: 1px solid var(--line); }
  .head-in { max-width: 82rem; margin: 0 auto; padding: .9rem var(--pad) .1rem; }
  h1 { font-size: 1.15rem; margin: 0; letter-spacing: -.01em; }
  .sub { color: var(--muted); margin: .1rem 0 .7rem; font-size: .82rem; }
  .tabs { display: flex; gap: .15rem; overflow-x: auto; scrollbar-width: none; }
  .tabs::-webkit-scrollbar { display: none; }
  .tab { padding: .5rem .9rem; font-size: .88rem; font-weight: 600; color: var(--muted);
         background: none; border: 0; border-bottom: 2px solid transparent; cursor: pointer;
         white-space: nowrap; border-radius: 0; }
  .tab:hover { color: var(--ink); }
  .tab[aria-selected="true"] { color: var(--accent); border-bottom-color: var(--accent); }
  .panel[hidden] { display: none; }

  /* a real grid — explicit order, predictable wrapping */
  /* `min(24rem, 100%)` rather than a bare 24rem. auto-fit honours the minimum literally, so a hard
     24rem (384px) floor against a phone's ~379px content box makes the COLUMN wider than the screen
     and the whole page scrolls sideways — measured on a 411dp viewport, where it pushed the arms
     scoreboard's equity and return columns off the right edge entirely. Wrapping the minimum in
     min(..., 100%) lets the track collapse to the viewport when there is not room for the ideal. */
  /* TWO columns at width, never three. auto-fit fitted three 384px tracks into the 82rem wrap, and
     the cards placed there have wildly unequal heights — a four-row Status beside a fifteen-row Cost
     breakdown — so the third column sat empty from the first card's bottom edge to the end of the
     page. The Scan tab was worse: `Latest scan` is `.span2`, so auto-placement could not use the
     leftover third track at all and left a hole the full height of the card beside it. Two wider
     columns halve the height disparity, give every table room, and leave no track that nothing can
     be placed in. */
  .grid { display: grid; gap: var(--gap); grid-template-columns: 1fr;
          align-items: start; margin-top: 1rem; }
  @media (min-width: 62rem) { .grid { grid-template-columns: 1fr 1fr; } }
  .grid.one { grid-template-columns: 1fr; }
  .span2 { grid-column: 1 / -1; }
  /* min-width:0 is load-bearing, not defensive. A grid item defaults to `min-width:auto`, which means
     it refuses to shrink below its content's intrinsic width — so a wide table inside a card made the
     CARD wide, and the `.scroll { overflow-x:auto }` wrapper around every table never engaged. The
     wrappers were correct all along and could not fire. */
  .card { min-width: 0; border: 1px solid #ffffff0a; border-radius: var(--radius); padding: var(--card-pad);
          background: var(--card); }
  .card h2 { font-size: 1rem; font-weight: 500; margin: 0 0 .8rem; display: flex; align-items: baseline;
             gap: .5rem; flex-wrap: wrap; }
  .card h2 .hint { font-weight: 400; }

  label { display: block; margin: .85rem 0 .3rem; font-weight: 600; font-size: .85rem; }
  .card label:first-of-type { margin-top: 0; }
  input, select { width: 100%; padding: .5rem .6rem; font-size: .95rem; border: 1px solid #8886;
          border-radius: .45rem; background: transparent; color: inherit; font-family: inherit; }
  select { appearance: auto; }
  .hint { font-size: .78rem; color: var(--muted); font-weight: 400; }
  .row { display: flex; gap: .5rem; align-items: center; flex-wrap: wrap; }
  .row.tight { gap: .35rem; }
  .row input, .row select { flex: 1; min-width: 6rem; }
  .fields { display: grid; grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr)); gap: .1rem .8rem; }
  button { padding: .5rem 1rem; font-size: .9rem; border: 0; border-radius: .45rem;
           background: var(--accent); color: var(--on-accent); font-weight: 600; cursor: pointer; white-space: nowrap; font-family: inherit; }
  button:disabled { opacity: .5; cursor: default; }
  button.secondary { background: #8883; color: inherit; }
  button.ok { background: var(--ok); }
  button.danger { background: transparent; color: var(--err); border: 1px solid var(--err); }
  button.sm { padding: .3rem .6rem; font-size: .78rem; }
  .chips { margin-top: .5rem; min-height: 1.1rem; }
  .chip { display: inline-flex; align-items: center; background: rgba(180,160,255,.14);
          border-radius: 1rem; padding: .2rem .65rem; margin: .12rem .22rem .12rem 0; font-size: .82rem; }
  .empty { color: var(--muted); font-size: .82rem; }
  .synced { font-size: .82rem; margin: .1rem 0 .5rem; color: var(--muted); }
  .synced.fresh { color: var(--ok); }
  .synced.stale { color: var(--warn); }
  .usage-totals { font-size: 1rem; margin: .1rem 0 .3rem; }
  .usage-totals b { color: var(--accent); }
  #usage-chart svg { display: block; width: 100%; height: auto; margin: .5rem 0 .2rem; }
  #usage-chart .bar { fill: var(--accent); }
  #usage-chart .bar.zero { fill: rgba(136,136,136,.28); }
  #usage-chart .axis { fill: var(--muted); font-size: 9px; }
  .save { margin-top: .9rem; }
  #status { font-size: .85rem; min-height: 1.1rem; margin-top: .6rem; }
  .ok-t { color: var(--ok); } .err-t { color: var(--err); }
  code { background: #8882; padding: .1rem .3rem; border-radius: .3rem; font-size: .85em; }
  .loading { color: var(--muted); font-size: .85rem; }

  .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(7.5rem, 1fr)); gap: .8rem .9rem; }
  .stat .k { font-size: .68rem; text-transform: uppercase; letter-spacing: .03em; color: var(--muted); }
  .stat .v { font-size: 1.05rem; font-weight: 600; margin-top: .1rem; }
  .stat .d { font-size: .76rem; color: var(--muted); margin-top: .1rem; }
  .countdown { font-variant-numeric: tabular-nums; }
  .meter { height: .4rem; background: #8883; border-radius: .3rem; overflow: hidden; margin-top: .35rem; }
  .meter > span { display: block; height: 100%; width: 0; background: var(--accent); }
  .meter.warn > span { background: var(--warn); }
  .meter.err > span { background: var(--err); }

  .sig-buy { color: var(--ok); font-weight: 600; }
  .sig-sell { color: var(--err); font-weight: 600; }
  .sig-hold { color: var(--muted); font-weight: 600; }
  .badge { display: inline-block; font-size: .64rem; padding: .04rem .36rem; border-radius: .8rem;
           margin-left: .2rem; font-weight: 700; vertical-align: middle; }
  .badge.flip { background: rgba(217,119,6,.18); color: var(--warn); }
  .badge.dip { background: rgba(22,163,74,.16); color: var(--ok); }
  .badge.cross { background: rgba(220,38,38,.16); color: var(--err); }
  .badge.eng { background: #8883; color: inherit; margin-left: 0; }
  .badge.on { background: rgba(22,163,74,.16); color: var(--ok); margin-left: 0; }
  .badge.off { background: #8883; color: var(--muted); margin-left: 0; }

  .scroll { overflow-x: auto; margin: .3rem -.2rem 0; }
  /* A long table is a scroll region, not a reason for its card to stand three times its neighbour's
     height. The cost breakdown is fifteen rows and was doing exactly that. */
  .scroll.capped { max-height: 22rem; overflow-y: auto; }
  .scroll.capped thead th { position: sticky; top: 0; background: var(--card); z-index: 1; }
  table.tbl { width: 100%; border-collapse: collapse; font-size: .82rem; }
  /* `white-space: nowrap` is the third cause of the sideways scroll, and the one that survived the
     grid fixes. It makes every cell contribute its full unwrapped width to the table's min-content
     size, so a nine-column scoreboard is ~1200px wide no matter how narrow its container gets — the
     `.scroll` wrapper then hands the reader a swipe to reach the equity and return columns, which on
     the tab where those numbers ARE the content is not a fix. It stays at width, where the table is
     genuinely the right shape and wrapping would only make it ragged; below the breakpoint the cells
     wrap and the label column pins so the arm's name stays on screen while the numbers scroll. */
  table.tbl th, table.tbl td { text-align: left; padding: var(--cell-pad); border-bottom: 1px solid var(--line);
                               white-space: nowrap; }
  @media (max-width: 46rem) {
    /* Tables that stay tabular pin their label column, so the row you are reading stays identified
       while the numbers scroll under your thumb. Letting the cells WRAP instead was tried and is
       worse than the swipe: nine columns squeezed into 411dp broke "$11,428" across four lines as
       "$1 / 1, / 42 / 8". A table too wide for the screen needs fewer columns, not narrower ones. */
    table.tbl:not(.stack) th:first-child, table.tbl:not(.stack) td:first-child {
      position: sticky; left: 0; z-index: 1; background: var(--card); box-shadow: 1px 0 0 var(--line); }
    table.tbl td.thesis { max-width: min(15rem, 42vw); }

    /* ...and the scoreboard, where the numbers ARE the content, stops being a table altogether and
       becomes one card per arm. Every field stays visible — this hides nothing, it re-flows the same
       nine cells down the screen instead of across it — and the header row is dropped because each
       cell now carries its own label from `data-l`. */
    table.tbl.stack, table.tbl.stack tbody, table.tbl.stack tr, table.tbl.stack td { display: block; width: auto; }
    table.tbl.stack thead { display: none; }
    table.tbl.stack tr { border: 1px solid var(--line); border-radius: var(--radius);
                         padding: .55rem .7rem; margin: 0 0 .55rem; background: var(--card); }
    table.tbl.stack tr.sel { outline: 2px solid var(--accent); outline-offset: -2px; }
    table.tbl.stack td { border: 0; padding: .16rem 0; display: flex; align-items: baseline;
                         justify-content: space-between; gap: 1rem; white-space: normal; }
    table.tbl.stack td::before { content: attr(data-l); color: var(--muted); font-size: .68rem;
                                 font-weight: 600; text-transform: uppercase; letter-spacing: .02em; flex: none; }
    table.tbl.stack td:first-child { display: block; padding-bottom: .35rem; }
    table.tbl.stack td:first-child::before, table.tbl.stack td:empty::before { content: none; }
    table.tbl.stack td[colspan] { display: block; text-align: center; }
  }
  table.tbl th { color: var(--muted); font-weight: 600; font-size: .68rem; text-transform: uppercase; }
  table.tbl td.num { text-align: right; font-variant-numeric: tabular-nums; }
  table.tbl td.thesis { max-width: 15rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  table.tbl tr.changed td:first-child { border-left: 3px solid var(--warn); padding-left: calc(.45rem - 3px); }
  table.tbl tr.sel td { background: rgba(180,160,255,.1); }
  .pos { color: var(--ok); } .neg { color: var(--err); }

  .src { display: flex; align-items: center; gap: .5rem; padding: .3rem 0; border-bottom: 1px solid #8882; }
  .dot { width: .55rem; height: .55rem; border-radius: 50%; flex: none; background: var(--muted); }
  .dot.ok { background: var(--ok); } .dot.warn { background: var(--warn); } .dot.down { background: var(--err); }
  .src .nm { font-weight: 600; font-size: .84rem; }
  .src .lat { color: var(--muted); font-size: .75rem; }
  .src .dt { color: var(--muted); font-size: .75rem; flex: 1; text-align: right;
             overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .ivp { display: flex; flex-wrap: wrap; gap: .3rem; margin-top: .4rem; }
  .ivp .pill { font-size: .72rem; background: #8882; border-radius: .8rem; padding: .1rem .48rem; }
  .ivp .pill.done { background: rgba(22,163,74,.16); color: var(--ok); }

  .cost-head { font-size: 1rem; margin: .1rem 0 .4rem; }
  .cost-head b { color: var(--accent); }
  .logrow { display: flex; gap: .5rem; font-size: .76rem; font-family: ui-monospace, SFMono-Regular, monospace;
            padding: .12rem 0; border-bottom: 1px solid #8881; }
  .logrow.bad { color: var(--err); }
  .logrow .m { width: 3.2rem; color: var(--muted); }
  .logrow .p { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .logrow .st { width: 2.4rem; text-align: right; }
  .logrow .ms { width: 4rem; text-align: right; color: var(--muted); }

  /* A run/verdict banner. Loud on purpose: the whole point of the market-scan and gate cards is
     that a job which stopped running, or a gate that could not decide, must be visible at a glance
     rather than inferred from a grid of dashes. `.unknown` is its own colour precisely because
     "we could not measure" must not look like "closed". */
  .banner { font-size: .85rem; font-weight: 600; border-radius: .5rem; padding: .45rem .7rem;
            margin: .1rem 0 .8rem; }
  .banner .sub2 { display: block; font-weight: 400; font-size: .78rem; opacity: .85; margin-top: .1rem; }
  .banner.ok { background: rgba(22,163,74,.14); color: var(--ok); }
  .banner.warn { background: rgba(217,119,6,.16); color: var(--warn); }
  .banner.err { background: rgba(220,38,38,.16); color: var(--err); }
  .banner.unknown { background: #8883; color: var(--muted); }
  .stat.absent .v { color: var(--muted); font-weight: 500; }
  .leg-ok { color: var(--ok); font-weight: 600; }
  .leg-fail { color: var(--err); font-weight: 600; }
  .leg-unk { color: var(--muted); font-weight: 600; }

  /* arms */
  .arm-edit { border-top: 1px solid var(--line); margin-top: .8rem; padding-top: .8rem; }
  .arm-edit[hidden] { display: none; }
  .note { font-size: .78rem; color: var(--muted); border-left: 2px solid var(--line);
          padding-left: .6rem; margin: .6rem 0 0; }

  /* ticker detail drawer */
  .drawer-backdrop { position: fixed; inset: 0; background: rgba(0,0,0,.45); opacity: 0;
                     pointer-events: none; transition: opacity .22s ease; z-index: 900; }
  .drawer-backdrop.open { opacity: 1; pointer-events: auto; }
  .drawer { position: fixed; top: 0; right: 0; bottom: 0; width: min(440px, 100vw);
            background: var(--card); border-left: 1px solid var(--line);
            box-shadow: -4px 0 24px rgba(0,0,0,.25); transform: translateX(100%);
            transition: transform .22s cubic-bezier(0.16, 1, 0.3, 1); z-index: 1000;
            display: flex; flex-direction: column; overflow: hidden; }
  .drawer.open { transform: translateX(0); }
  .drawer-header { padding: 1rem 1.1rem .8rem; border-bottom: 1px solid var(--line); background: var(--card); flex-shrink: 0; }
  .dr-head-top { display: flex; justify-content: space-between; align-items: flex-start; gap: .5rem; }
  .dr-sym { font-size: 1.4rem; font-weight: 700; margin: 0; letter-spacing: -.02em; display: flex; align-items: baseline; gap: .4rem; }
  .dr-name { font-size: .84rem; color: var(--muted); margin-top: .1rem; }
  .dr-close { background: none; border: 0; color: var(--muted); font-size: 1.3rem; padding: .1rem .4rem;
              cursor: pointer; border-radius: .3rem; line-height: 1; }
  .dr-close:hover { color: var(--ink); background: #8882; }
  .dr-price-row { display: flex; align-items: baseline; gap: .6rem; margin-top: .6rem; flex-wrap: wrap; }
  .dr-price { font-size: 1.35rem; font-weight: 700; font-variant-numeric: tabular-nums; }
  .dr-change { font-size: .9rem; font-weight: 600; font-variant-numeric: tabular-nums; }
  .drawer-body { padding: 1rem 1.1rem 2rem; overflow-y: auto; flex: 1; display: flex; flex-direction: column; gap: 1.1rem; }
  .dr-sec-title { font-size: .72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .04em;
                  color: var(--muted); margin-bottom: .5rem; display: flex; align-items: center; justify-content: space-between; }
  .dr-pill-row { display: flex; flex-wrap: wrap; gap: .35rem; margin-bottom: .5rem; }
  .dr-pill { font-size: .72rem; padding: .12rem .5rem; border-radius: .8rem; background: #8882; color: var(--muted); font-weight: 500; }
  .dr-lead { font-size: .88rem; line-height: 1.45; margin: 0 0 .4rem; }
  .dr-range-selector { display: flex; gap: .2rem; background: #8882; padding: .15rem; border-radius: .4rem; }
  .dr-range-btn { padding: .18rem .55rem; font-size: .75rem; font-weight: 600; border-radius: .3rem;
                  background: none; border: 0; color: var(--muted); cursor: pointer; }
  .dr-range-btn.sel { background: var(--card); color: var(--accent); box-shadow: 0 1px 3px rgba(0,0,0,.15); }
  .ticker-link { cursor: pointer; color: var(--accent); text-decoration: none; font-weight: 600; }
  .ticker-link:hover { text-decoration: underline; }
/*SANDBOX_CSS*/
</style></head>
<body>
<header>
  <div class="head-in">
    <h1>StockTracker Signals</h1>
    <p class="sub">Tier-2 Claude analyst — operations &amp; configuration</p>
    <div class="tabs" role="tablist">
      <button class="tab" role="tab" data-panel="p-overview" aria-selected="true">Overview</button>
      <button class="tab" role="tab" data-panel="p-sandbox" aria-selected="false">Sandbox</button>
      <button class="tab" role="tab" data-panel="p-scan" aria-selected="false">Scan</button>
      <button class="tab" role="tab" data-panel="p-settings" aria-selected="false">Settings</button>
    </div>
  </div>
</header>

<div class="wrap">

<!-- ============================== OVERVIEW ============================== -->
<section class="panel" id="p-overview">
  <div class="grid">
    <div class="card span2" id="status-card">
      <h2>Status <span class="hint" id="uptime"></span></h2>
      <div class="stat-grid">
        <div class="stat"><div class="k">Last scan</div><div class="v" id="scan-when">…</div>
          <div class="d" id="scan-detail"></div></div>
        <div class="stat"><div class="k">Next scan</div><div class="v countdown" id="next-scan">…</div>
          <div class="d">06:30 America/Chicago</div></div>
        <!-- The nightly MARKET scan (the whole-universe cross-section), not the watchlist scan on
             the left. It is here on Overview only as a freshness headline — a job that stopped
             running has to be wrong on the first screen, not three tabs away. The counters live on
             the Scan tab. -->
        <div class="stat"><div class="k">Market scan</div><div class="v" id="mscan-when">…</div>
          <div class="d" id="mscan-when-detail"></div></div>
        <div class="stat"><div class="k">Disk</div><div class="v" id="disk-v">…</div>
          <div class="meter" id="disk-meter"><span></span></div></div>
      </div>
      <div class="row" style="margin-top:.8rem">
        <button type="button" class="secondary sm" id="prune">Prune cache</button>
        <span class="hint" id="prune-status" style="flex:1"></span>
      </div>
      <div class="hint" id="cache-detail" style="margin-top:.5rem"></div>
    </div>

    <div class="card">
      <h2>AI usage <span class="hint">last 30 days</span></h2>
      <div id="usage-totals" class="usage-totals">loading…</div>
      <div id="usage-chart"></div>
      <div class="hint" id="usage-readout">&nbsp;</div>
      <div id="usage-models" class="hint"></div>
    </div>

    <div class="card">
      <h2>Cost breakdown</h2>
      <div class="cost-head" id="cost-head">loading…</div>
      <div class="scroll capped"><table class="tbl" id="cost-tbl">
        <thead><tr><th>Kind</th><th class="num">Calls</th><th class="num">Tokens</th><th class="num">Cost</th></tr></thead>
        <tbody id="cost-body"></tbody>
      </table></div>
      <div class="hint" id="cost-avg" style="margin-top:.6rem"></div>
    </div>

    <div class="card span2">
      <h2>Recent activity <span class="hint">last requests, in-memory since restart</span></h2>
      <div id="logs"><div class="loading">loading…</div></div>
      <div class="hint" id="errs-head" style="margin-top:.6rem"></div>
      <div id="errs"></div>
    </div>
  </div>
</section>

<!-- ============================== SANDBOX =============================== -->
<!--SANDBOX_HTML-->

<!-- =============================== SCAN ================================= -->
<section class="panel" id="p-scan" hidden>
  <div class="grid">
    <div class="card">
      <h2>Market scan <span class="hint" id="mscan-age"></span></h2>
      <p class="hint" style="margin:.1rem 0 .7rem">The nightly, LLM-free cross-section of the whole
      liquid universe — a different job and a different artifact from the watchlist scan below.
      Every counter is reported separately: a symbol that never answered and one with too little
      history are different facts about the market, and a blank is "not measured", never zero.</p>
      <div class="banner unknown" id="mscan-state">loading…</div>
      <div class="stat-grid" id="mscan-counters"></div>
      <div class="hint" id="mscan-reason" style="margin-top:.7rem"></div>
      <div class="hint" id="mscan-extra" style="margin-top:.35rem"></div>
    </div>

    <div class="card">
      <h2>Market gate <span class="hint" id="gate-age"></span></h2>
      <p class="hint" style="margin:.1rem 0 .7rem">Five mechanical legs, no model. Read the banner
      before the legs: <b>undecided</b> is not <b>closed</b> — a leg nobody could measure is not a
      bearish market, and the two are listed separately below for that reason.</p>
      <div class="banner unknown" id="gate-state">loading…</div>
      <div class="scroll"><table class="tbl" id="gate-tbl">
        <thead><tr><th>Leg</th><th></th><th class="num">Value</th><th class="num">Threshold</th></tr></thead>
        <tbody id="gate-body"><tr><td colspan="4" class="loading">loading…</td></tr></tbody>
      </table></div>
      <div class="hint" id="gate-failing" style="margin-top:.6rem"></div>
      <div class="hint" id="gate-unmeasured" style="margin-top:.25rem"></div>
      <div class="hint" id="gate-note" style="margin-top:.5rem"></div>
    </div>

    <div class="card span2">
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
  </div>
</section>

<!-- ============================= SETTINGS =============================== -->
<section class="panel" id="p-settings" hidden>
  <div class="grid">
    <form id="f" class="card">
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
        <button type="button" class="secondary sm" id="cli-test">Test CLI auth</button>
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
      <label for="scan">Scan model <span class="hint">— cheap watchlist scans; the default for every sandbox arm</span></label>
      <input id="scan" autocomplete="off">
      <label for="ttl">Verdict cache TTL <span class="hint">— seconds</span></label>
      <input id="ttl" type="number" min="0" autocomplete="off">
      <button class="save" type="submit">Save settings</button>
      <div id="status"></div>
    </form>

    <div class="card">
      <h2>Service</h2>
      <div id="version" class="hint">version …</div>
      <div class="row" style="margin-top:.8rem">
        <button type="button" class="secondary sm" id="check">Refresh</button>
      </div>
    </div>
  </div>
</section>

</div>

<!-- =========================== TICKER DRAWER =========================== -->
<div id="ticker-drawer-backdrop" class="drawer-backdrop" onclick="closeTickerDrawer()"></div>
<aside id="ticker-drawer" class="drawer" aria-hidden="true">
  <div class="drawer-header">
    <div class="dr-head-top">
      <div>
        <h2 id="dr-symbol" class="dr-sym">---</h2>
        <div id="dr-name" class="dr-name">---</div>
      </div>
      <button type="button" class="dr-close" onclick="closeTickerDrawer()" aria-label="Close">✕</button>
    </div>
    <div class="dr-price-row">
      <span id="dr-price" class="dr-price">---</span>
      <span id="dr-change" class="dr-change">---</span>
      <span id="dr-signal-badge" class="badge">---</span>
      <span id="dr-etf-badge" class="badge" style="display:none">ETF</span>
    </div>
  </div>

  <div class="drawer-body">
    <!-- What it is -->
    <div class="card" id="dr-what-card" style="padding:.85rem">
      <div class="dr-sec-title">
        <span>WHAT IT IS</span>
        <span id="dr-ai-badge" class="badge ok" style="display:none">AI SUMMARY</span>
      </div>
      <div class="dr-pill-row" id="dr-tags"></div>
      <p id="dr-lead" class="dr-lead"></p>
      <div id="dr-cust" class="hint" style="margin-top:.4rem"></div>
      <details id="dr-summary-details" style="margin-top:.5rem">
        <summary class="hint" style="cursor:pointer;color:var(--accent)">Read full description</summary>
        <p id="dr-full-summary" class="hint" style="margin-top:.4rem;line-height:1.45;white-space:pre-line"></p>
      </details>
    </div>

    <!-- Price History -->
    <div class="card" id="dr-chart-card" style="padding:.85rem">
      <div class="dr-sec-title">
        <span>PRICE HISTORY</span>
        <div class="dr-range-selector">
          <button type="button" class="dr-range-btn" data-r="1D" onclick="setDrawerRange('1D')">1D</button>
          <button type="button" class="dr-range-btn sel" data-r="1W" onclick="setDrawerRange('1W')">1W</button>
          <button type="button" class="dr-range-btn" data-r="1M" onclick="setDrawerRange('1M')">1M</button>
          <button type="button" class="dr-range-btn" data-r="1Y" onclick="setDrawerRange('1Y')">1Y</button>
        </div>
      </div>
      <div id="dr-period-change" class="hint" style="margin:.1rem 0 .4rem;font-weight:600"></div>
      <div id="dr-chart-box" style="width:100%;height:150px;position:relative;">
        <svg id="dr-chart-svg" viewBox="0 0 380 150" style="width:100%;height:100%;display:block;"></svg>
      </div>
    </div>

    <!-- Details -->
    <div class="card" id="dr-details-card" style="padding:.85rem">
      <div class="dr-sec-title">DETAILS</div>
      <table class="tbl" id="dr-details-tbl" style="margin-top:0">
        <tbody>
          <tr><td>Market cap</td><td class="num" id="dr-mcap">—</td></tr>
          <tr><td>P/E ratio</td><td class="num" id="dr-pe">—</td></tr>
          <tr><td>52-week range</td><td class="num" id="dr-52w">—</td></tr>
          <tr><td>Day range</td><td class="num" id="dr-dayrange">—</td></tr>
          <tr><td>Volume</td><td class="num" id="dr-vol">—</td></tr>
          <tr><td>Dividend yield</td><td class="num" id="dr-div">—</td></tr>
        </tbody>
      </table>
    </div>

    <!-- Sandbox Position / Posture -->
    <div class="card" id="dr-sandbox-card" style="padding:.85rem;display:none">
      <div class="dr-sec-title">SANDBOX POSITION</div>
      <div id="dr-sandbox-content"></div>
    </div>
  </div>
</aside>

<script>


  // SEC-2. This page is served unauthenticated, but the routes it reads are not: /api/settings and
  // the sandbox routes disclose the watchlist, the paper book and whether a key is configured. The
  // dashboard's JS runs in YOUR browser, so it has no access to the container's .env and has to be
  // told the token once, the same way the phone is.
  //
  // Kept in localStorage rather than a cookie: a cookie would be sent automatically on every
  // request including cross-site ones, and this page has no CSRF protection to go with it.
  // The token is kept in localStorage; if the browser refuses storage (a private window, blocked
  // site data) it falls back to sessionStorage, then to memory, and says so instead of looping.
  let MEM_TOKEN = "";
  function apiToken() {
    try { const t = localStorage.getItem("signals_token"); if (t) return t; } catch (e) {}
    try { const t = sessionStorage.getItem("signals_token"); if (t) return t; } catch (e) {}
    return MEM_TOKEN;
  }
  function setApiToken(t) {
    MEM_TOKEN = t;
    try { localStorage.setItem("signals_token", t); if (localStorage.getItem("signals_token") === t) return "kept"; } catch (e) {}
    try { sessionStorage.setItem("signals_token", t); if (sessionStorage.getItem("signals_token") === t) return "session"; } catch (e) {}
    return "memory";
  }
  // Accept what people actually paste: the whole .env line, quotes, spaces or a line break.
  function cleanToken(raw) {
    let t = String(raw || "").trim();
    const eq = t.indexOf("=");
    if (t.toUpperCase().startsWith("SIGNALS_API_TOKEN") && eq > 0) t = t.slice(eq + 1);
    if (t.toLowerCase().startsWith("bearer ")) t = t.slice(7);
    return t.split("").filter(c => c > " " && c !== '"' && c !== "'").join("");
  }
  async function afetch(url, opts) {
    const o = Object.assign({}, opts || {});
    const t = apiToken();
    if (t) { o.headers = Object.assign({}, o.headers || {}, { "Authorization": "Bearer " + t }); }
    const r = await fetch(url, o);
    if (r.status === 401) { showTokenPrompt(!!t); }
    return r;
  }
  function showTokenPrompt(hadToken) {
    if (document.getElementById("tokbar")) return;
    const bar = document.createElement("div");
    bar.id = "tokbar";
    bar.style.cssText = "position:fixed;top:0;left:0;right:0;z-index:9999;background:#C99A35;color:#0D1116;padding:10px 14px;font:14px system-ui;display:flex;gap:8px;align-items:center;flex-wrap:wrap";
    bar.innerHTML = '<span id="tokmsg"></span>'
      + '<input id="tokin" type="text" name="signals-access-token" autocomplete="off" autocapitalize="off" spellcheck="false" data-lpignore="true" data-1p-ignore placeholder="paste the token" style="flex:0 1 320px;padding:4px 8px;color:#0D1116;background:#fff;border:0;border-radius:4px;-webkit-text-security:disc">'
      + '<button id="toksave" style="padding:4px 12px">Save token</button>';
    document.body.appendChild(bar);
    // A refused saved token is dropped, so the next paste starts clean. The field is a masked TEXT
    // input, not type=password: a browser's password manager otherwise refills the first (wrong)
    // value on every retry, which is exactly how one bad paste became a loop on 2026-11-18.
    if (hadToken) { MEM_TOKEN = ""; try { localStorage.removeItem("signals_token"); } catch (e) {} try { sessionStorage.removeItem("signals_token"); } catch (e) {} }
    const msg = document.getElementById("tokmsg");
    msg.textContent = hadToken
      ? "The saved access token was refused. Paste it again (SIGNALS_API_TOKEN in /opt/signals/.env on the container)."
      : "This dashboard needs the access token (SIGNALS_API_TOKEN in /opt/signals/.env on the container).";
    const save = async function () {
      const t = cleanToken(document.getElementById("tokin").value);
      if (!t) { msg.textContent = "Paste the token first."; return; }
      msg.textContent = "Checking…";
      let ok = false;
      try { ok = (await fetch("/api/settings", { headers: { "Authorization": "Bearer " + t } })).ok; } catch (e) {}
      if (!ok) { msg.textContent = "That token was refused (" + t.length + " characters). Copy only the part after SIGNALS_API_TOKEN= and try again."; return; }
      const where = setApiToken(t);
      if (where === "kept") { location.reload(); return; }
      msg.textContent = where === "session"
        ? "Token accepted, but this browser only keeps it until the tab closes (site data is restricted)."
        : "Token accepted, but this browser won't store it (private window or blocked site data). It works until you reload.";
      document.getElementById("tokin").remove(); document.getElementById("toksave").remove();
      setTimeout(() => bar.remove(), 6000);
      if (typeof loadStatus === "function") { try { loadStatus(); } catch (e) {} }
      if (typeof bootSandbox === "function") { try { bootSandbox(); } catch (e) {} }
    };
    document.getElementById("toksave").onclick = save;
    document.getElementById("tokin").onkeydown = (e) => { if (e.key === "Enter") save(); };
  }
  const $ = (id) => document.getElementById(id);

  // Read-only chips — the watchlist is owned by the app and synced up via POST /api/settings.
  function renderChips(kind, syms) {
    const box = $(kind + "-chips");
    box.innerHTML = "";
    if (!syms.length) { box.innerHTML = '<span class="empty">none yet — connect the app to sync</span>'; return; }
    syms.forEach((sym) => {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.style.cursor = "pointer";
      chip.onclick = () => openTickerDrawer(sym);
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
    try { renderSynced((await (await afetch("/api/settings")).json()).watchlist_synced_at); } catch (e) {}
  }

  // Counts, with absence preserved. `Number(n).toLocaleString()` renders null as "0" and undefined
  // as "NaN" — the identical substitution money() was just fixed for, sitting one line away and
  // formatting every token and call count on the page. A token count nobody measured must not
  // render as a confident zero.
  const fmt = (n) => (n === null || n === undefined || !isFinite(Number(n)))
    ? "\u2014" : Number(n).toLocaleString();
  function drawUsageChart(series) {
    const box = $("usage-chart");
    const W = 520, H = 130, padL = 6, padT = 8, padB = 18;
    const n = series.length;
    const max = Math.max(1, ...series.map((d) => d.tokens));
    const bw = (W - padL) / n;
    const bars = series.map((d, i) => {
      const h = (d.tokens / max) * (H - padT - padB);
      const x = padL + i * bw, y = H - padB - h;
      const t = d.date + ": " + fmt(d.tokens) + " tokens · " + money(d.cost_usd) +
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
      const u = await (await afetch("/api/usage?days=30")).json();
      const bp = u.by_provider || {};
      const billed = (bp.api && bp.api.cost_usd) || 0;
      const notional = (bp.cli && bp.cli.cost_usd) || 0;
      let cost = "<b>" + money(billed) + "</b> billed";
      if (notional > 0) cost += ' · <span class="hint">' + money(notional) + " notional (subscription)</span>";
      $("usage-totals").innerHTML = "<b>" + fmt(u.total_tokens) + "</b> tokens · " + cost + " · " +
        fmt(u.total_calls) + " calls" +
        ' <span class="hint">(' + fmt(u.total_input_tokens) + " in / " + fmt(u.total_output_tokens) + " out, all-time)</span>";
      drawUsageChart(u.series);
      const models = Object.entries(u.by_model).sort((a, b) => b[1].cost_usd - a[1].cost_usd)
        .map(([m, v]) => m + " — " + fmt(v.calls) + " calls · " + money(v.cost_usd)).join("<br>");
      $("usage-models").innerHTML = models || "No calls recorded yet.";
    } catch (e) { $("usage-totals").textContent = "usage unavailable"; }
  }

  async function load() {
    const s = await (await afetch("/api/settings")).json();
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
    const r = await afetch("/api/settings", { method: "POST",
      headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    const st = $("status");
    st.textContent = r.ok ? "Saved ✓" : "Save failed"; st.className = r.ok ? "ok-t" : "err-t";
    $("key").value = ""; $("fkey").value = ""; $("clitoken").value = ""; load();
  };

  async function checkVersion() {
    // OPS-6: this is read-only status now — POST /api/update (git fetch + reset --hard + restart)
    // was removed. It could never have worked against an rsync deploy anyway, and it ran with no
    // authentication at all. Deploy with the rsync skill/README instead of from this page.
    //
    // The button says "Refresh", not "Check for updates". On this container git status is always
    // unavailable, so an update check can only ever return the same sentence — a control that
    // cannot do the thing its label promises is the exact defect the 2026-09-17 review was about.
    // What it genuinely does is re-read the version, which is worth a button after a deploy.
    $("version").textContent = "checking…";
    const v = await (await afetch("/api/version")).json();
    let label = "version " + v.version;
    if (!v.git) label += " · (not a git checkout — deploys are by rsync, see deploy/README.md)";
    else if (v.update_available) label += " · " + v.behind + " commit" + (v.behind > 1 ? "s" : "") + " behind origin/main (informational only)";
    else label += " · up to date with origin/main";
    $("version").textContent = label;
  }
  $("check").onclick = checkVersion;

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
  // Money, at a precision that matches the magnitude — and ABSENT stays absent.
  //
  // This was `"$" + Number(n || 0).toFixed(4)`, which got both halves wrong. Four decimal places on
  // every figure printed "$1.7143 billed" and "$55.5482 notional", where the extra digits are noise
  // that makes a page of costs harder to scan, not more accurate. And `n || 0` turned null into
  // "$0.0000" — a precise-looking zero standing in for a number nobody measured, which is the one
  // substitution this codebase does not allow anywhere else. The cost card was showing
  // "MTD $0.0000 · projected $0.0000" for figures that were simply absent.
  //
  // Sub-cent values keep their digits, because a per-call cost of $0.0134 is the whole point of
  // showing it and rounding it to $0.01 would flatten the range the reader is comparing.
  const money = (n) => {
    if (n === null || n === undefined || !isFinite(Number(n))) return "—";
    const v = Number(n);
    if (v === 0) return "$0";
    const abs = Math.abs(v);
    return "$" + v.toFixed(abs >= 1 ? 2 : abs >= 0.01 ? 3 : 4);
  };
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
      return '<span class="pill' + (done ? " done" : "") + '" style="cursor:pointer" onclick="openTickerDrawer(\\'' + esc(k) + '\\')">' + esc(k) + " " + (done ? "✓ " + n : n + "/" + target) + "</span>";
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

  // The market scan's own run summary.
  //
  // `present:false` and a status of "skipped" are DIFFERENT facts and must not render alike: the
  // first is a job that has never completed, the second is a job that ran and correctly declined
  // (Monday resolves to Friday's already-stored session, so it skips by design). A dashboard that
  // showed both as blank is exactly what let this job fail invisibly before the card existed.
  function renderMarketScan(m) {
    m = m || {};
    const banner = $("mscan-state"), age = $("mscan-age");
    const when = $("mscan-when"), whenD = $("mscan-when-detail");
    // null is UNKNOWN and renders as a dash. Coercing to 0 would turn "we did not measure" into
    // "we measured nothing", which is a claim about the market.
    const n = (v) => (v == null ? "&mdash;" : esc(v));

    if (!m.present) {
      banner.className = "banner err";
      banner.textContent = "No run recorded — the job has never written a summary";
      $("mscan-counters").innerHTML = "";
      $("mscan-reason").textContent = "";
      $("mscan-extra").textContent = "Runs nightly at 05:45 America/Chicago.";
      age.textContent = "";
      if (when) { when.textContent = "never"; when.className = "v err"; }
      if (whenD) whenD.textContent = "no summary on disk";
      return;
    }

    const st = (m.status || "unknown").toLowerCase();
    // `stale` is three-valued upstream: null means we could not tell how old the run is, which is
    // not the same claim as fresh, so it gets its own wording instead of folding into either.
    const staleTxt = m.stale === true ? " · STALE" : m.stale == null ? " · age unknown" : "";
    banner.className = "banner " + (st === "ok" ? "ok" : st === "skipped" ? "unknown" : "err");
    banner.textContent =
      (st === "ok" ? "Scan advanced" : st === "skipped" ? "Skipped — that session was already stored"
                                                        : "Run " + st)
      + " · session " + (m.session || "—") + staleTxt;

    const ageTxt = m.age_hours != null ? m.age_hours.toFixed(1) + "h ago" : "age unknown";
    age.textContent = "· " + ageTxt;
    if (when) {
      when.textContent = ageTxt;
      when.className = "v" + (m.stale === true ? " err" : "");
    }
    if (whenD) whenD.textContent = (m.session ? "session " + m.session : "") + (st ? " · " + st : "");

    const tile = (k, v, d) =>
      '<div class="stat"><div class="k">' + k + '</div><div class="v">' + v + "</div>" +
      (d ? '<div class="d">' + d + "</div>" : "") + "</div>";
    $("mscan-counters").innerHTML =
      tile("Scanned", n(m.scanned), m.universe_symbols != null ? "of " + esc(m.universe_symbols) : "") +
      tile("Fetch failed", n(m.fetch_failed), "never answered") +
      tile("Too short", n(m.too_short), "not enough history") +
      tile("Suspect", n(m.suspect_series), "mixed split basis") +
      tile("Stored", n(m.rows_written), m.retired != null ? esc(m.retired) + " retired" : "") +
      tile("Duration", m.duration_s != null ? esc(m.duration_s) + "s" : "&mdash;",
           m.pruned != null ? esc(m.pruned) + " pruned" : "");

    $("mscan-reason").textContent = m.reason ? "Reason: " + m.reason : "";
    const ranked = (m.percentiles && m.percentiles.ranked) ? m.percentiles.ranked.length : null;
    $("mscan-extra").textContent = ranked != null
      ? ranked + " metric(s) ranked across the night's cross-section."
      : "Percentile pass has not reported for this run.";
  }

  async function loadStatus() {
    try {
      const s = await (await afetch("/api/status")).json();
      renderStatus(s);
      renderMarketScan(s.market_scan);
    }
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
    let data; try { data = await (await afetch("/scan/latest")).json(); } catch (e) { $("scan-body").innerHTML = '<tr><td colspan="7" class="empty">scan unavailable</td></tr>'; return; }
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
        '<td><b class="ticker-link" onclick="openTickerDrawer(\\'' + esc(r.symbol) + '\\')">' + esc(r.symbol) + "</b>" + scanBadges(r) + "</td>" +
        '<td class="' + sigClass(r.signal) + '">' + esc(r.signal) + "</td>" +
        '<td class="num">' + (r.conviction != null ? esc(r.conviction) : "–") + "</td>" +
        "<td>" + esc(r.dip || "–") + "</td>" +
        "<td>" + esc(r.squeeze || "–") + "</td>" +
        "<td>" + (r.below_200wma ? "yes" : "–") + "</td>" +
        '<td class="thesis" title="' + th + '">' + linkifyTickers(r.thesis || "") + "</td></tr>";
    }).join("");
  }

  async function loadSources() {
    let data;
    try { data = await (await afetch("/api/sources")).json(); }
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
    let c; try { c = await (await afetch("/api/cost")).json(); } catch (e) { $("cost-head").textContent = "cost unavailable"; return; }
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
    let data; try { data = await (await afetch("/api/logs?limit=30")).json(); } catch (e) { $("logs").innerHTML = '<div class="empty">logs unavailable</div>'; return; }
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
      const r = await (await afetch("/api/prune-cache", { method: "POST" })).json();
      ps.textContent = "freed " + fmtBytes(r.bytes_freed) + " (" + r.deleted_files + " file" + (r.deleted_files === 1 ? "" : "s") + ")";
    } catch (e) { ps.textContent = "prune failed"; }
    loadStatus();
  };

  $("cli-test").onclick = async () => {
    const st = $("cli-test-status"); st.textContent = "testing…"; st.className = "hint";
    try {
      const r = await (await afetch("/api/cli-auth-test")).json();
      st.textContent = r.ok ? "✓ authenticated" : "✗ " + (r.detail || "failed");
      st.className = r.ok ? "ok-t" : "err-t";
    } catch (e) { st.textContent = "✗ request failed"; st.className = "err-t"; }
  };

  load(); checkVersion(); loadUsage();
  // The market gate. This card shipped as markup with no loader at all — six element ids that
  // nothing ever wrote to, so it read "loading…" forever on a page whose whole job is telling you
  // whether things are running. The CSS for its three leg states was already here, which is how it
  // is clear the function was designed and simply never written.
  //
  // The card's own copy states the distinction this has to honour: UNDECIDED IS NOT CLOSED. A leg
  // nobody could measure is not a bearish market, so a gate that failed to evaluate must never
  // render in the same colour as one that evaluated and shut.
  async function loadGate() {
    let g;
    try { g = await (await fetch("/gate")).json(); }
    catch (e) {
      $("gate-state").className = "banner unknown";
      $("gate-state").textContent = "gate unavailable — could not be read";
      $("gate-body").innerHTML = '<tr><td colspan="4" class="empty">gate unavailable</td></tr>';
      $("gate-age").textContent = ""; $("gate-failing").textContent = "";
      $("gate-unmeasured").textContent = ""; $("gate-note").textContent = "";
      return;
    }
    const legs = g.legs || [], failing = g.failing || [], unmeasured = g.unmeasured || [];
    const score = g.market_score;

    $("gate-age").textContent = g.evaluated_at ? "· " + agoText(g.evaluated_at) : "";

    const b = $("gate-state");
    if (g.available === false) {
      b.className = "banner unknown";
      b.textContent = "Gate could not be evaluated";
    } else if (g.passed === true) {
      b.className = "banner ok";
      b.textContent = "Gate OPEN — all measured legs pass";
    } else if (failing.length) {
      b.className = "banner err";
      b.textContent = "Gate CLOSED — " + failing.length + " leg" + (failing.length > 1 ? "s" : "") + " failing";
    } else {
      // Nothing failed, so this is not a closed gate; it is a gate that could not finish.
      b.className = "banner warn";
      b.textContent = "Gate UNDECIDED — nothing failed, but " + unmeasured.length + " leg"
        + (unmeasured.length === 1 ? "" : "s") + " could not be measured";
    }
    if (score !== null && score !== undefined) {
      const sub = document.createElement("span");
      sub.className = "sub2";
      sub.textContent = "market score " + Number(score).toFixed(1) + " / 100"
        + (g.as_of ? " · session " + g.as_of : "");
      b.appendChild(sub);
    }

    const num = (v) => (v === null || v === undefined || !isFinite(Number(v)))
      ? "\u2014" : Number(v).toLocaleString(undefined, {maximumFractionDigits: 2});
    $("gate-body").innerHTML = legs.length ? legs.map((l) => {
      // ok === null is a leg that could not be measured, and it must not look like a failure.
      const cls = l.ok === true ? "leg-ok" : l.ok === false ? "leg-fail" : "leg-unk";
      const mark = l.ok === true ? "pass" : l.ok === false ? "fail" : "unmeasured";
      return '<tr><td>' + esc(l.name || l.key || "\u2014") +
        (l.note ? '<div class="hint">' + esc(l.note) + "</div>" : "") + "</td>" +
        '<td class="' + cls + '">' + mark + "</td>" +
        '<td class="num">' + num(l.value) + "</td>" +
        '<td class="num">' + num(l.threshold) + "</td></tr>";
    }).join("") : '<tr><td colspan="4" class="empty">no legs reported</td></tr>';

    $("gate-failing").textContent = failing.length ? "Failing: " + failing.join(", ") : "";
    $("gate-unmeasured").textContent = unmeasured.length
      ? "Unmeasured (not counted as failures): " + unmeasured.join(", ") : "";
    $("gate-note").textContent = g.note || "";
  }

  loadStatus(); loadScan(); loadSources(); loadCost(); loadLogs(); loadGate();
  setInterval(() => { refreshSynced(); loadUsage(); loadCost(); }, 60000); // heartbeat + usage/cost
  setInterval(() => { loadStatus(); loadScan(); loadLogs(); loadGate(); }, 30000);     // live ops cards
  setInterval(loadSources, 60000);                                          // source probes (heavier)
  setInterval(tickCountdown, 1000);                                         // next-scan countdown

  // ---------------------------------------------------------------- tabs
  // Panels stay in the DOM and every loader keeps running regardless of which tab is showing: the
  // page is an ops dashboard people leave open, and a card that only refreshes while visible is a
  // card that is stale the moment you look at it.
  const tabs = [...document.querySelectorAll(".tab")];
  function showTab(id) {
    tabs.forEach(t => {
      const on = t.dataset.panel === id;
      t.setAttribute("aria-selected", on ? "true" : "false");
      $(t.dataset.panel).hidden = !on;
    });
    if (location.hash.slice(1).split("/")[0] !== id) history.replaceState(null, "", "#" + id);
  }
  tabs.forEach(t => t.onclick = () => showTab(t.dataset.panel));
  // Deep-linkable, so "the sandbox page" can be bookmarked and a reload does not bounce to Overview.
  const hash0 = location.hash.slice(1).split("/")[0];
  showTab(tabs.some(t => t.dataset.panel === hash0) ? hash0 : "p-overview");

  // ---------------------------------------------------------------- ticker drawer & links
  const COMMON_WORDS = new Set(["A", "I", "AND", "OR", "THE", "NOT", "FOR", "IN", "ON", "AT", "TO", "IS", "ARE", "OF", "BY", "BUT", "BE", "SO", "DO", "IF", "NO", "WE", "MY", "HE", "UP", "AS", "IT", "AN", "ALL", "NEW", "MAX", "MIN", "NOW", "OFF", "SET", "GET", "PUT", "RUN", "DAY", "TOP", "LOW", "DIP", "BUY", "OUT"]);
  function linkifyTickers(text) {
    if (!text) return "";
    return esc(text).replace(/\\b([A-Z]{1,5})\\b/g, (match) => {
      if (COMMON_WORDS.has(match)) return match;
      return '<span class="ticker-link" onclick="openTickerDrawer(\\'' + match + '\\')">' + match + '</span>';
    });
  }

  let currentDrawerSym = null;
  let currentDrawerBars = [];
  let currentDrawerRange = "1W";

  function openTickerDrawer(sym) {
    if (!sym) return;
    sym = sym.toUpperCase();
    currentDrawerSym = sym;
    $("ticker-drawer-backdrop").classList.add("open");
    $("ticker-drawer").classList.add("open");
    $("ticker-drawer").setAttribute("aria-hidden", "false");

    $("dr-symbol").textContent = sym;
    $("dr-name").textContent = "loading profile…";
    $("dr-price").textContent = "—";
    $("dr-change").textContent = "";
    $("dr-change").className = "dr-change";
    $("dr-signal-badge").style.display = "none";
    $("dr-etf-badge").style.display = "none";
    $("dr-tags").innerHTML = "";
    $("dr-lead").textContent = "";
    $("dr-cust").textContent = "";
    $("dr-ai-badge").style.display = "none";
    $("dr-full-summary").textContent = "";
    $("dr-summary-details").open = false;
    $("dr-mcap").textContent = "—";
    $("dr-pe").textContent = "—";
    $("dr-52w").textContent = "—";
    $("dr-dayrange").textContent = "—";
    $("dr-vol").textContent = "—";
    $("dr-div").textContent = "—";
    $("dr-sandbox-card").style.display = "none";
    $("dr-period-change").textContent = "loading price history…";
    $("dr-chart-svg").innerHTML = "";
    currentDrawerBars = [];

    loadDrawerProfile(sym);
    loadDrawerHistory(sym);
    loadDrawerSandboxHolding(sym);
  }

  function closeTickerDrawer() {
    $("ticker-drawer-backdrop").classList.remove("open");
    $("ticker-drawer").classList.remove("open");
    $("ticker-drawer").setAttribute("aria-hidden", "true");
    currentDrawerSym = null;
  }

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && currentDrawerSym) closeTickerDrawer();
  });

  async function loadDrawerProfile(sym) {
    try {
      const p = await (await afetch("/profile/" + encodeURIComponent(sym) + "?plain=true")).json();
      if (currentDrawerSym !== sym) return;
      $("dr-name").textContent = p.name || p.long_name || "";
      if (p.is_etf) $("dr-etf-badge").style.display = "inline-block";

      const tags = [];
      if (p.sector) tags.push(p.sector);
      if (p.industry) tags.push(p.industry);
      $("dr-tags").innerHTML = tags.map(t => '<span class="dr-pill">' + esc(t) + '</span>').join("");

      if (p.what_it_does) {
        $("dr-lead").textContent = p.what_it_does;
        $("dr-ai-badge").style.display = "inline-block";
      } else if (p.summary) {
        const dotIdx = p.summary.indexOf(". ");
        $("dr-lead").textContent = dotIdx > 0 ? p.summary.slice(0, dotIdx + 1) : p.summary;
      }
      if (p.customers) {
        $("dr-cust").textContent = "Customers: " + p.customers;
      }
      if (p.summary) {
        $("dr-full-summary").textContent = p.summary;
      }

      if (p.market_cap) {
        const mc = p.market_cap;
        $("dr-mcap").textContent = "$" + (mc >= 1e12 ? (mc/1e12).toFixed(2) + "T" : mc >= 1e9 ? (mc/1e9).toFixed(2) + "B" : (mc/1e6).toFixed(1) + "M");
      }
      $("dr-pe").textContent = p.pe_ratio != null ? p.pe_ratio.toFixed(1) : (p.forward_pe != null ? p.forward_pe.toFixed(1) + " (fwd)" : "—");
      if (p.week52_low != null && p.week52_high != null) {
        $("dr-52w").textContent = money(p.week52_low) + " – " + money(p.week52_high);
      }
      if (p.day_low != null && p.day_high != null) {
        $("dr-dayrange").textContent = money(p.day_low) + " – " + money(p.day_high);
      }
      if (p.volume) $("dr-vol").textContent = Number(p.volume).toLocaleString();
      if (p.dividend_yield_pct != null) $("dr-div").textContent = p.dividend_yield_pct.toFixed(2) + "%";

    } catch (e) {
      if (currentDrawerSym === sym) $("dr-name").textContent = "profile unavailable";
    }
  }

  async function loadDrawerHistory(sym) {
    try {
      const [hData, sData] = await Promise.all([
        (await afetch("/history/" + encodeURIComponent(sym))).json().catch(() => null),
        (await afetch("/signal/" + encodeURIComponent(sym))).json().catch(() => null),
      ]);
      if (currentDrawerSym !== sym) return;

      if (sData) {
        if (sData.price != null) $("dr-price").textContent = money(sData.price);
        if (sData.change != null) {
          const chg = sData.change;
          const pct = sData.change_pct;
          const up = chg >= 0;
          $("dr-change").textContent = (up ? "+" : "") + money(chg) + (pct != null ? " (" + (up ? "+" : "") + pct.toFixed(2) + "%)" : "");
          $("dr-change").className = "dr-change " + (up ? "pos" : "neg");
        }
        if (sData.signal) {
          $("dr-signal-badge").textContent = sData.signal.toUpperCase() + (sData.conviction != null ? " " + sData.conviction + "%" : "");
          $("dr-signal-badge").className = "badge " + (sData.signal.indexOf("buy") >= 0 ? "on" : sData.signal.indexOf("sell") >= 0 ? "cross" : "eng");
          $("dr-signal-badge").style.display = "inline-block";
        }
      }

      if (hData && hData.bars && hData.bars.length) {
        currentDrawerBars = hData.bars;
        renderDrawerChart();
      } else {
        $("dr-period-change").textContent = "no price history available";
      }
    } catch (e) {
      if (currentDrawerSym === sym) $("dr-period-change").textContent = "history unavailable";
    }
  }

  function setDrawerRange(r) {
    currentDrawerRange = r;
    document.querySelectorAll(".dr-range-btn").forEach(b => {
      b.classList.toggle("sel", b.dataset.r === r);
    });
    renderDrawerChart();
  }

  function renderDrawerChart() {
    if (!currentDrawerBars || currentDrawerBars.length < 2) return;
    const now = Date.now();
    let cutoff = 0;
    if (currentDrawerRange === "1D") cutoff = now - 2 * 86400 * 1000;
    else if (currentDrawerRange === "1W") cutoff = now - 7 * 86400 * 1000;
    else if (currentDrawerRange === "1M") cutoff = now - 32 * 86400 * 1000;
    else if (currentDrawerRange === "1Y") cutoff = now - 366 * 86400 * 1000;

    let pts = currentDrawerBars.filter(b => b.t >= cutoff);
    if (pts.length < 2) pts = currentDrawerBars.slice(-10);
    if (pts.length < 2) return;

    const first = pts[0].c;
    const last = pts[pts.length - 1].c;
    const up = last >= first;
    const pct = first > 0 ? ((last / first - 1) * 100) : 0;
    $("dr-period-change").textContent = (up ? "+" : "") + pct.toFixed(2) + "% over " + currentDrawerRange;
    $("dr-period-change").className = "hint " + (up ? "pos" : "neg");

    const W = 380, H = 150, padT = 12, padB = 20, padL = 10, padR = 10;
    const minPrice = Math.min(...pts.map(p => p.c));
    const maxPrice = Math.max(...pts.map(p => p.c));
    const priceSpan = (maxPrice - minPrice) || 1;

    const coords = pts.map((p, i) => {
      const x = padL + (i / (pts.length - 1)) * (W - padL - padR);
      const y = padT + (1 - (p.c - minPrice) / priceSpan) * (H - padT - padB);
      return {x, y, p};
    });

    const pathD = coords.reduce((acc, pt, i) => acc + (i === 0 ? "M " : " L ") + pt.x.toFixed(1) + " " + pt.y.toFixed(1), "");
    const areaD = pathD + " L " + coords[coords.length - 1].x.toFixed(1) + " " + (H - padB) + " L " + coords[0].x.toFixed(1) + " " + (H - padB) + " Z";
    const strokeColor = up ? "var(--ok)" : "var(--err)";
    const fillColor = up ? "rgba(22,163,74,.15)" : "rgba(220,38,38,.15)";

    const firstDate = new Date(pts[0].t).toLocaleDateString(undefined, {month: "short", day: "numeric"});
    const lastDate = new Date(pts[pts.length - 1].t).toLocaleDateString(undefined, {month: "short", day: "numeric"});

    let svg = '<path d="' + areaD + '" fill="' + fillColor + '" />' +
              '<path d="' + pathD + '" fill="none" stroke="' + strokeColor + '" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />' +
              '<text x="' + padL + '" y="' + (H - 4) + '" fill="var(--muted)" font-size="10">' + firstDate + '</text>' +
              '<text x="' + (W - padR) + '" y="' + (H - 4) + '" fill="var(--muted)" font-size="10" text-anchor="end">' + lastDate + '</text>' +
              '<text x="' + (W - padR) + '" y="' + (padT + 10) + '" fill="var(--muted)" font-size="10" text-anchor="end">High ' + money(maxPrice) + '</text>' +
              '<text x="' + (W - padR) + '" y="' + (H - padB - 4) + '" fill="var(--muted)" font-size="10" text-anchor="end">Low ' + money(minPrice) + '</text>';

    $("dr-chart-svg").innerHTML = svg;
  }

  async function loadDrawerSandboxHolding(sym) {
    try {
      const holdings = [];
      await Promise.all(ARMS.filter(a => a.universe !== "options").map(async a => {
        try {
          const st = await (await afetch("/sandbox/state?arm=" + encodeURIComponent(a.arm))).json();
          const pos = (st.positions || []).find(p => p.symbol.toUpperCase() === sym);
          if (pos) {
            holdings.push({arm: a.arm, label: a.label || a.arm, ...pos});
          }
        } catch (e) {}
      }));
      if (currentDrawerSym !== sym) return;

      if (holdings.length) {
        $("dr-sandbox-card").style.display = "block";
        $("dr-sandbox-content").innerHTML = holdings.map(h => {
          const basis = h.shares * h.avg_cost;
          const val = h.value || (h.price ? h.shares * h.price : basis);
          const pnl = val - basis;
          const up = pnl >= 0;
          return '<div style="margin-bottom:.5rem;padding-bottom:.5rem;border-bottom:1px solid var(--line)">' +
            '<div style="font-weight:600;font-size:.85rem;margin-bottom:.2rem">' + esc(h.label) + '</div>' +
            '<div style="display:flex;justify-content:space-between;font-size:.8rem">' +
              '<span>' + h.shares + ' sh @ ' + money(h.avg_cost) + '</span>' +
              '<span class="' + (up ? 'pos' : 'neg') + '">' + (up ? '+' : '') + money(pnl) + (h.unrealized_pct != null ? ' (' + (up ? '+' : '') + h.unrealized_pct.toFixed(2) + '%)' : '') + '</span>' +
            '</div>' +
            '<div class="hint" style="font-size:.75rem">Value: ' + money(val) + (h.price ? ' · Last: ' + money(h.price) : '') + '</div>' +
          '</div>';
        }).join("");
      } else {
        $("dr-sandbox-card").style.display = "none";
      }
    } catch (e) {
      $("dr-sandbox-card").style.display = "none";
    }
  }

/*SANDBOX_JS*/

</script>
</body></html>
"""

# The Sandbox tab lives in dashboard_sandbox.py (raw strings, so its JS needs no Python escaping).
PAGE = (PAGE.replace("<!--SANDBOX_HTML-->", _sbx.HTML)
            .replace("/*SANDBOX_CSS*/", _sbx.CSS)
            .replace("/*SANDBOX_JS*/", _sbx.JS))
