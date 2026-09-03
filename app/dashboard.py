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
  :root {
    color-scheme: light dark;
    --accent:#1d4ed8; --ok:#0f7a34; --err:#c02626; --warn:#96590d;
    --muted:#5b6270; --line:#d9dce3; --card:#ffffff; --bg:#f6f7f9; --ink:#16181d;
    --radius:.85rem;
    /* Spacing and density scale with the viewport rather than being fixed at desktop values. */
    --pad:.8rem; --gap:.7rem; --card-pad:.85rem .9rem; --cell-pad:.34rem .45rem;
  }
  @media (prefers-color-scheme: dark) {
    :root { --accent:#7ea6ff; --ok:#4cc172; --err:#ff6f6f; --warn:#e0a63f;
            --muted:#9aa2b1; --line:#2a2f38; --card:#171a20; --bg:#101216; --ink:#e7eaf0; }
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
  body { font-family: system-ui, -apple-system, sans-serif; margin: 0; line-height: 1.5;
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
  .tab:hover { color: CanvasText; }
  .tab[aria-selected="true"] { color: var(--accent); border-bottom-color: var(--accent); }
  .panel[hidden] { display: none; }

  /* a real grid — explicit order, predictable wrapping */
  /* `min(24rem, 100%)` rather than a bare 24rem. auto-fit honours the minimum literally, so a hard
     24rem (384px) floor against a phone's ~379px content box makes the COLUMN wider than the screen
     and the whole page scrolls sideways — measured on a 411dp viewport, where it pushed the arms
     scoreboard's equity and return columns off the right edge entirely. Wrapping the minimum in
     min(..., 100%) lets the track collapse to the viewport when there is not room for the ideal. */
  .grid { display: grid; gap: var(--gap); grid-template-columns: repeat(auto-fit, minmax(min(24rem, 100%), 1fr));
          align-items: start; margin-top: 1rem; }
  .grid.one { grid-template-columns: 1fr; }
  .span2 { grid-column: 1 / -1; }
  /* min-width:0 is load-bearing, not defensive. A grid item defaults to `min-width:auto`, which means
     it refuses to shrink below its content's intrinsic width — so a wide table inside a card made the
     CARD wide, and the `.scroll { overflow-x:auto }` wrapper around every table never engaged. The
     wrappers were correct all along and could not fire. */
  .card { min-width: 0; border: 1px solid var(--line); border-radius: var(--radius); padding: var(--card-pad);
          background: var(--card); }
  .card h2 { font-size: .95rem; margin: 0 0 .8rem; display: flex; align-items: baseline;
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
           background: var(--accent); color: #fff; cursor: pointer; white-space: nowrap; font-family: inherit; }
  button:disabled { opacity: .5; cursor: default; }
  button.secondary { background: #8883; color: inherit; }
  button.ok { background: var(--ok); }
  button.danger { background: transparent; color: var(--err); border: 1px solid var(--err); }
  button.sm { padding: .3rem .6rem; font-size: .78rem; }
  .chips { margin-top: .5rem; min-height: 1.1rem; }
  .chip { display: inline-flex; align-items: center; background: rgba(37,99,235,.15);
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
  #status, #upstatus { font-size: .85rem; min-height: 1.1rem; margin-top: .6rem; }
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
  table.tbl tr.sel td { background: rgba(37,99,235,.09); }
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
    <div class="card" id="status-card">
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
      <div class="hint">Hover a bar for that day's detail.</div>
      <div id="usage-models" class="hint"></div>
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

    <div class="card span2">
      <h2>Recent activity <span class="hint">last requests, in-memory since restart</span></h2>
      <div id="logs"><div class="loading">loading…</div></div>
      <div class="hint" id="errs-head" style="margin-top:.6rem"></div>
      <div id="errs"></div>
    </div>
  </div>
</section>

<!-- ============================== SANDBOX =============================== -->
<section class="panel" id="p-sandbox" hidden>
  <div class="grid one">
    <div class="card">
      <h2>Arms <span class="hint" id="arms-when"></span></h2>
      <p class="hint" style="margin:.1rem 0 .7rem">
        Independent paper ledgers run against the same market snapshot on the same tick, so the
        difference between them is the strategy rather than the weather. Compared on excess over each
        arm's own S&amp;P shadow — raw equity is not comparable when arms are funded with different
        amounts on different days.
      </p>
      <div class="scroll"><table class="tbl stack" id="arms-tbl">
        <thead><tr>
          <th>Arm</th><th>Engine</th><th>Model</th><th class="num">Equity</th><th class="num">Cash</th>
          <th class="num">Return</th><th class="num">vs S&amp;P</th><th>State</th><th></th>
        </tr></thead>
        <tbody id="arms-body"><tr><td colspan="9" class="loading">loading…</td></tr></tbody>
      </table></div>
      <div class="hint" id="arms-spread" style="margin-top:.6rem"></div>

      <div class="arm-edit" id="arm-edit" hidden>
        <h2 style="margin-bottom:.5rem">Edit <span id="arm-edit-name"></span></h2>
        <div class="fields">
          <div>
            <label for="a-label">Label</label>
            <input id="a-label" autocomplete="off">
          </div>
          <div>
            <label for="a-engine">Engine</label>
            <select id="a-engine">
              <option value="llm">llm — the analyst decides</option>
              <option value="rules">rules — mechanical, no model</option>
            </select>
          </div>
          <div>
            <label for="a-model">Model <span class="hint">— blank = service default</span></label>
            <input id="a-model" autocomplete="off" placeholder="e.g. claude-opus-5">
          </div>
          <div>
            <label for="a-risk">Risk tolerance</label>
            <select id="a-risk">
              <option value="conservative">conservative</option>
              <option value="balanced">balanced</option>
              <option value="aggressive">aggressive</option>
            </select>
          </div>
          <div>
            <label for="a-maxnew">Max new positions / tick</label>
            <input id="a-maxnew" type="number" min="0" max="20">
          </div>
          <div>
            <label for="a-maxtrades">Max trades / tick</label>
            <input id="a-maxtrades" type="number" min="0" max="20">
          </div>
          <div>
            <label for="a-maxpos">Max position %</label>
            <input id="a-maxpos" type="number" min="5" max="100" step="0.5">
          </div>
          <div>
            <label for="a-floor">Cash floor %</label>
            <input id="a-floor" type="number" min="0" max="90" step="0.5">
          </div>
          <div>
            <label for="a-turnover">Max turnover %</label>
            <input id="a-turnover" type="number" min="0" max="100" step="0.5">
          </div>
          <div>
            <label for="a-conv">Min conviction</label>
            <input id="a-conv" type="number" min="0" max="100">
          </div>
        </div>
        <div class="row" style="margin-top:.9rem">
          <button type="button" id="a-save">Save arm</button>
          <button type="button" class="secondary" id="a-toggle">Pause</button>
          <button type="button" class="secondary" id="a-cancel">Close</button>
          <span style="flex:1"></span>
          <button type="button" class="danger sm" id="a-delete">Delete arm</button>
        </div>
        <div id="a-status" class="hint" style="margin-top:.5rem"></div>
        <p class="note" id="a-mainnote" hidden>
          This is the real account. Its engine cannot be changed to <code>rules</code> here and it
          cannot be deleted — create a separate arm for that.
        </p>
      </div>
    </div>

    <div class="card">
      <h2>New arm</h2>
      <p class="hint" style="margin:.1rem 0 .7rem">
        A clone starts from another arm's book — cash, positions and benchmark shadow — so the two
        share a starting line and everything after it is attributable to the strategy rather than to
        a head start. Change ONE thing per arm; an arm with two variables measures neither.
      </p>
      <div class="fields">
        <div><label for="n-id">Id <span class="hint">— a-z 0-9 _ -</span></label>
          <input id="n-id" autocomplete="off" placeholder="conservative"></div>
        <div><label for="n-label">Label</label>
          <input id="n-label" autocomplete="off" placeholder="Conservative analyst"></div>
        <div><label for="n-engine">Engine</label>
          <select id="n-engine">
            <option value="llm">llm — the analyst decides</option>
            <option value="rules">rules — mechanical, no model</option>
          </select></div>
        <div><label for="n-clone">Clone book from</label>
          <select id="n-clone"></select></div>
        <div><label for="n-model">Model <span class="hint">— blank = service default</span></label>
          <input id="n-model" autocomplete="off"></div>
        <div><label for="n-fund">Fund <span class="hint">— only if not cloning</span></label>
          <input id="n-fund" type="number" min="0" step="100" value="0"></div>
      </div>
      <div class="row" style="margin-top:.9rem">
        <button type="button" id="n-create">Create arm</button>
        <span class="hint" id="n-status" style="flex:1"></span>
      </div>
    </div>
  </div>
</section>

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
        <button type="button" class="secondary sm" id="check">Check for updates</button>
        <button type="button" class="ok sm" id="update" style="display:none">Update &amp; restart</button>
      </div>
      <div id="upstatus"></div>
    </div>
  </div>
</section>

</div>
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
      const u = await (await fetch("/api/usage?days=30")).json();
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
      const s = await (await fetch("/api/status")).json();
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
    if (location.hash.slice(1) !== id) history.replaceState(null, "", "#" + id);
  }
  tabs.forEach(t => t.onclick = () => showTab(t.dataset.panel));
  // Deep-linkable, so "the sandbox page" can be bookmarked and a reload does not bounce to Overview.
  showTab(tabs.some(t => t.dataset.panel === location.hash.slice(1))
          ? location.hash.slice(1) : "p-overview");

  // ---------------------------------------------------------------- arms
  let ARMS = [], selArm = null;

  const armPct = (v, dp) => (v === null || v === undefined) ? "—"
        : (v >= 0 ? "+" : "") + v.toFixed(dp === undefined ? 2 : dp) + "%";
  const armMoney = (v) => (v === null || v === undefined) ? "—"
        : "$" + v.toLocaleString(undefined, {maximumFractionDigits: 0});

  function armRow(a) {
    const tr = document.createElement("tr");
    if (a.arm === selArm) tr.className = "sel";
    const vs = a.vs_benchmark_pct;
    // An arm with no benchmark shadow yet has no comparable number. "—", never a confident 0.00%.
    const vsCls = vs === null || vs === undefined ? "" : (vs >= 0 ? "pos" : "neg");
    // `data-l` is what lets the phone layout drop the header row and still label every value — see
    // `table.tbl.stack td::before`. The labels must match the <thead> text or the two layouts would
    // be naming the same number differently.
    tr.innerHTML =
      '<td data-l="Arm"><b></b><div class="hint"></div></td>' +
      '<td data-l="Engine"><span class="badge eng"></span></td>' +
      '<td data-l="Model" class="hint"></td>' +
      '<td data-l="Equity" class="num"></td><td data-l="Cash" class="num"></td>' +
      '<td data-l="Return" class="num"></td><td data-l="vs S&amp;P" class="num ' + vsCls + '"></td>' +
      '<td data-l="State"></td><td></td>';
    const td = tr.children;
    td[0].querySelector("b").textContent = a.label || a.arm;
    td[0].querySelector(".hint").textContent = a.arm;
    td[1].querySelector(".eng").textContent = a.engine;
    td[2].textContent = a.model || "default";
    td[3].textContent = armMoney(a.equity);
    td[4].textContent = a.cash_pct === null || a.cash_pct === undefined ? "—" : a.cash_pct.toFixed(1) + "%";
    td[5].textContent = armPct(a.total_return_pct);
    td[6].textContent = vs === null || vs === undefined ? "—" : armPct(vs);
    const st = document.createElement("span");
    st.className = "badge " + (a.enabled ? "on" : "off");
    st.textContent = a.enabled ? "live" : "paused";
    td[7].appendChild(st);
    const b = document.createElement("button");
    b.className = "secondary sm"; b.textContent = "Edit";
    b.onclick = () => selectArm(a.arm);
    td[8].appendChild(b);
    return tr;
  }

  async function loadArms() {
    try {
      const r = await (await fetch("/sandbox/arms")).json();
      ARMS = r.arms || [];
    } catch (e) {
      $("arms-body").innerHTML = '<tr><td colspan="9" class="err-t">couldn’t load arms</td></tr>';
      return;
    }
    // The per-arm model lives in settings, which /sandbox/arms does not carry. One extra fetch per
    // arm keeps the column honest rather than showing "default" for an arm that has pinned one.
    await Promise.all(ARMS.map(async a => {
      try {
        const s = await (await fetch("/sandbox/settings?arm=" + encodeURIComponent(a.arm))).json();
        a._settings = s; a.model = s.model || null;
      } catch (e) { a._settings = null; }
    }));
    const body = $("arms-body"); body.textContent = "";
    if (!ARMS.length) {
      body.innerHTML = '<tr><td colspan="9" class="empty">no arms</td></tr>';
    } else {
      ARMS.forEach(a => body.appendChild(armRow(a)));
    }
    const measured = ARMS.map(a => a.vs_benchmark_pct).filter(v => v !== null && v !== undefined);
    $("arms-spread").textContent = measured.length >= 2
      ? "Spread: " + (Math.max(...measured) - Math.min(...measured)).toFixed(2) +
        " points between best and worst. Weeks, not ticks — this means nothing yet."
      : "";
    $("arms-when").textContent = ARMS.length
      ? "last tick " + (ARMS[0].last_tick_date || "never") : "";
    // Clone-source options, refreshed with the list so a deleted arm cannot linger as a choice.
    const sel = $("n-clone"); const keep = sel.value;
    sel.textContent = "";
    const none = document.createElement("option"); none.value = ""; none.textContent = "— none (start empty) —";
    sel.appendChild(none);
    ARMS.forEach(a => {
      const o = document.createElement("option");
      o.value = a.arm; o.textContent = a.label || a.arm;
      sel.appendChild(o);
    });
    sel.value = keep || "main";
    if (selArm && !ARMS.some(a => a.arm === selArm)) closeArmEditor();
  }

  function closeArmEditor() { selArm = null; $("arm-edit").hidden = true; loadArms(); }

  function selectArm(id) {
    const a = ARMS.find(x => x.arm === id); if (!a) return;
    selArm = id;
    const s = a._settings || {};
    $("arm-edit").hidden = false;
    $("arm-edit-name").textContent = (a.label || a.arm) + " (" + a.arm + ")";
    $("a-label").value = a.label || "";
    $("a-engine").value = a.engine || "llm";
    $("a-model").value = s.model || "";
    $("a-risk").value = s.risk_tolerance || "balanced";
    $("a-maxnew").value = s.max_new_positions_per_tick ?? "";
    $("a-maxtrades").value = s.max_trades_per_tick ?? "";
    $("a-maxpos").value = s.max_position_pct ?? "";
    $("a-floor").value = s.cash_floor_pct ?? "";
    $("a-turnover").value = s.max_turnover_pct ?? "";
    $("a-conv").value = s.min_conviction_to_trade ?? "";
    $("a-toggle").textContent = a.enabled ? "Pause" : "Resume";
    // main is the real account: it is not deletable, and flipping it to the mechanical engine would
    // silently retire the analyst on the book that matters rather than on a control arm.
    const isMain = a.arm === "main";
    $("a-delete").style.display = isMain ? "none" : "";
    $("a-engine").disabled = isMain;
    $("a-mainnote").hidden = !isMain;
    $("a-status").textContent = "";
    loadArms();
    $("arm-edit").scrollIntoView({behavior: "smooth", block: "nearest"});
  }

  async function saveArm() {
    if (!selArm) return;
    const st = $("a-status"); st.textContent = "saving…"; st.className = "hint";
    const num = (id) => { const v = $(id).value.trim(); return v === "" ? null : Number(v); };
    const patch = {
      label: $("a-label").value.trim() || null,
      model: $("a-model").value.trim(),          // "" clears the pin back to the service default
      risk_tolerance: $("a-risk").value,
      max_new_positions_per_tick: num("a-maxnew"),
      max_trades_per_tick: num("a-maxtrades"),
      max_position_pct: num("a-maxpos"),
      cash_floor_pct: num("a-floor"),
      max_turnover_pct: num("a-turnover"),
      min_conviction_to_trade: num("a-conv"),
    };
    if (!$("a-engine").disabled) patch.engine = $("a-engine").value;
    try {
      const r = await fetch("/sandbox/settings?arm=" + encodeURIComponent(selArm),
        {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(patch)});
      if (!r.ok) throw new Error(((await r.json().catch(() => ({}))).detail) || r.status);
      st.textContent = "✓ saved"; st.className = "ok-t";
      await loadArms();
    } catch (e) { st.textContent = "✗ " + e.message; st.className = "err-t"; }
  }

  async function toggleArm() {
    if (!selArm) return;
    const a = ARMS.find(x => x.arm === selArm); if (!a) return;
    const st = $("a-status"); st.textContent = "…"; st.className = "hint";
    try {
      const r = await fetch("/sandbox/settings?arm=" + encodeURIComponent(selArm),
        {method: "POST", headers: {"Content-Type": "application/json"},
         body: JSON.stringify({master_enabled: !a.enabled})});
      if (!r.ok) throw new Error(r.status);
      st.textContent = a.enabled ? "paused" : "resumed"; st.className = "ok-t";
      await loadArms();
      const nx = ARMS.find(x => x.arm === selArm);
      if (nx) $("a-toggle").textContent = nx.enabled ? "Pause" : "Resume";
    } catch (e) { st.textContent = "✗ failed"; st.className = "err-t"; }
  }

  async function deleteArm() {
    if (!selArm || selArm === "main") return;
    // Irreversible and it takes the arm's whole history with it, so it asks — and names the arm,
    // because "are you sure?" on the wrong row is how the wrong book gets deleted.
    if (!confirm("Delete arm '" + selArm + "' and its entire trade history? This cannot be undone."))
      return;
    const st = $("a-status"); st.textContent = "deleting…"; st.className = "hint";
    try {
      const r = await fetch("/sandbox/arms/" + encodeURIComponent(selArm), {method: "DELETE"});
      if (!r.ok) throw new Error(((await r.json().catch(() => ({}))).detail) || r.status);
      closeArmEditor();
    } catch (e) { st.textContent = "✗ " + e.message; st.className = "err-t"; }
  }

  async function createArm() {
    const st = $("n-status"); const btn = $("n-create");
    const id = $("n-id").value.trim().toLowerCase();
    if (!id) { st.textContent = "an id is required"; st.className = "err-t"; return; }
    btn.disabled = true; st.textContent = "creating…"; st.className = "hint";
    const clone = $("n-clone").value || null;
    const body = {
      arm: id,
      engine: $("n-engine").value,
      label: $("n-label").value.trim() || null,
      clone_from: clone,
      // Funding a clone on top of the copied book would double-count it — the clone already carries
      // the source's cash and its benchmark shadow.
      fund: clone ? 0 : Number($("n-fund").value || 0),
      enabled: true,
    };
    const model = $("n-model").value.trim();
    try {
      const r = await fetch("/sandbox/arms",
        {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
      if (!r.ok) throw new Error(((await r.json().catch(() => ({}))).detail) || r.status);
      if (model) {
        await fetch("/sandbox/settings?arm=" + encodeURIComponent(id),
          {method: "POST", headers: {"Content-Type": "application/json"},
           body: JSON.stringify({model: model})});
      }
      st.textContent = "✓ created"; st.className = "ok-t";
      $("n-id").value = ""; $("n-label").value = ""; $("n-model").value = "";
      await loadArms();
    } catch (e) { st.textContent = "✗ " + e.message; st.className = "err-t"; }
    finally { btn.disabled = false; }
  }

  $("a-save").onclick = saveArm;
  $("a-toggle").onclick = toggleArm;
  $("a-cancel").onclick = closeArmEditor;
  $("a-delete").onclick = deleteArm;
  $("n-create").onclick = createArm;
  // Funding is meaningless for a clone; grey it out rather than silently ignoring what was typed.
  $("n-clone").onchange = () => {
    const cloning = !!$("n-clone").value;
    $("n-fund").disabled = cloning;
    $("n-fund").title = cloning ? "a clone already carries the source arm's cash" : "";
  };

  loadArms();
  setInterval(loadArms, 60000);

</script>
</body></html>
"""
