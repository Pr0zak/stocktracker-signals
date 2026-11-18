"""The dashboard's Sandbox tab, mirroring the phone app's Sandbox screen (2026-11-18).

Spliced into dashboard.PAGE at three markers. Kept apart, and in RAW strings, because PAGE is a
plain triple-quoted string: every backslash in its JS is a Python escape first, which is how the old
tab ended up with `\\'` in its onclick handlers. Nothing here needs that.

What it shows, in the app's own vocabulary and colours (ui/sandbox/*.kt in the app repo):

* The Today card: what every account did on the last run, folded to one line until opened.
* Group chips: Stocks, ETFs, Options, and Compare.
* The race board: each account's points ahead of or behind its own S&P shadow, bars from a centre
  line. An account that has not run shows "—" and ranks last (the app's `vsShown`), never +0.00.
* The open account: balance, cash, gap to the S&P; an equity curve against the S&P with a hover
  readout; holdings, or for an options account one card per contract (stop/target gauge, days to
  the days-left sell, break-even) and the weekly budget; its run today; the trade log; settings.
* Compare: AI vs rules per pair, the reviewer's kept vs cut trades, each tweak vs the Baseline
  (the app's `Matchups`).

Honesty rules carried over from the app: a stale option mark is drawn as "≈" in amber with no
profit or loss and no marker on the gauge; a null weekly budget draws nothing; every string from the
API (postures, reasons, notes) goes through `esc()` before it touches innerHTML, because postures are
model text and this page holds the access token.

Only the OPEN account's /sandbox/state is fetched. An options account's state prices live chains,
and fetching all of them on every refresh is how the old table cost fourteen requests a minute.
"""
from __future__ import annotations

HTML = r"""
<section class="panel" id="p-sandbox" hidden>
  <div class="sbx">
    <div class="sbx-rail">
      <div class="glow" id="sbx-today"><div class="loading">Loading the last run…</div></div>
      <div class="sbx-chips" id="sbx-chips" role="tablist" aria-label="Account group"></div>
      <div class="card sbx-board" id="sbx-board"><div class="loading">Loading accounts…</div></div>
      <details class="card sbx-fold" id="sbx-macro" hidden>
        <summary><span id="sbx-macro-sum">Macro backdrop</span></summary>
        <div id="sbx-macro-body"></div>
      </details>
      <details class="card sbx-fold" id="sbx-new">
        <summary>New account</summary>
        <p class="hint">A copy starts from another account's book, so the two share a starting
          line. Change one thing per account; an account with two changes measures neither.</p>
        <div class="fields">
          <div><label for="n-id">Id <span class="hint">a-z 0-9 _ -</span></label>
            <input id="n-id" autocomplete="off" placeholder="conservative"></div>
          <div><label for="n-label">Name</label>
            <input id="n-label" autocomplete="off" placeholder="Conservative analyst"></div>
          <div><label for="n-engine">Engine</label>
            <select id="n-engine">
              <option value="llm">AI analyst decides</option>
              <option value="rules">Fixed rules, no AI</option>
            </select></div>
          <div><label for="n-clone">Copy book from</label>
            <select id="n-clone"></select></div>
          <div><label for="n-model">Model <span class="hint">blank uses the default</span></label>
            <input id="n-model" autocomplete="off"></div>
          <div><label for="n-fund">Fund <span class="hint">only when not copying</span></label>
            <input id="n-fund" type="number" min="0" step="100" value="0"></div>
        </div>
        <div class="row" style="margin-top:.9rem">
          <button type="button" id="n-create">Create account</button>
          <span class="hint" id="n-status" style="flex:1"></span>
        </div>
      </details>
    </div>

    <div class="sbx-main" id="sbx-main" aria-live="polite">
      <div class="glow" id="sbx-head"><div class="loading">Pick an account.</div></div>
      <div class="card" id="sbx-curve" hidden></div>
      <div id="sbx-contracts" class="sbx-stack" hidden></div>
      <div class="card" id="sbx-holdings" hidden></div>
      <div class="card" id="sbx-day" hidden></div>
      <details class="card sbx-fold" id="sbx-log">
        <summary><span>Trade log</span> <span class="hint" id="sbx-log-n"></span></summary>
        <div class="scroll"><table class="tbl" id="sbx-log-tbl"><tbody></tbody></table></div>
      </details>
      <details class="card sbx-fold" id="arm-edit">
        <summary>Settings <span class="hint" id="arm-edit-name"></span></summary>
        <div class="fields">
          <div><label for="a-label">Name</label><input id="a-label" autocomplete="off"></div>
          <div><label for="a-engine">Engine</label>
            <select id="a-engine">
              <option value="llm">AI analyst decides</option>
              <option value="rules">Fixed rules, no AI</option>
            </select></div>
          <div><label for="a-model">Model <span class="hint">blank uses the default</span></label>
            <input id="a-model" autocomplete="off" placeholder="e.g. claude-opus-5"></div>
          <div><label for="a-risk">Risk</label>
            <select id="a-risk">
              <option value="conservative">Conservative</option>
              <option value="balanced">Balanced</option>
              <option value="aggressive">Aggressive</option>
            </select></div>
          <div class="share-only"><label for="a-maxnew">New positions per run</label>
            <input id="a-maxnew" type="number" min="0" max="20"></div>
          <div class="share-only"><label for="a-maxtrades">Trades per run</label>
            <input id="a-maxtrades" type="number" min="0" max="20"></div>
          <div class="share-only"><label for="a-maxpos">Largest position %</label>
            <input id="a-maxpos" type="number" min="5" max="100" step="0.5"></div>
          <div><label for="a-floor">Cash floor %</label>
            <input id="a-floor" type="number" min="0" max="90" step="0.5"></div>
          <div class="share-only"><label for="a-turnover">Turnover limit %</label>
            <input id="a-turnover" type="number" min="0" max="100" step="0.5"></div>
          <div><label for="a-conv">Lowest conviction to trade</label>
            <input id="a-conv" type="number" min="0" max="100"></div>
        </div>
        <div class="opt-limits" id="a-optlimits" hidden></div>
        <div class="row" style="margin-top:.9rem">
          <button type="button" id="a-save">Save settings</button>
          <button type="button" class="secondary" id="a-toggle">Pause</button>
          <button type="button" class="secondary" id="a-cancel">Close</button>
          <span style="flex:1"></span>
          <button type="button" class="danger sm" id="a-delete">Delete account</button>
        </div>
        <div id="a-status" class="hint" style="margin-top:.5rem"></div>
        <p class="note" id="a-mainnote" hidden>
          This is the Baseline. Its engine can't be switched to rules and it can't be deleted;
          make a separate account for that.
        </p>
      </details>
    </div>
  </div>
</section>
"""

CSS = r"""
  /* ---------------------------------------------------------------- sandbox (mirrors the app) */
  .sbx { display: grid; gap: var(--gap); margin-top: 1rem; align-items: start; grid-template-columns: 1fr; }
  @media (min-width: 62rem) { .sbx { grid-template-columns: minmax(22rem, 27rem) 1fr; } }
  .sbx-rail, .sbx-main, .sbx-stack { display: grid; gap: var(--gap); min-width: 0; }
  @media (min-width: 80rem) { .sbx-stack { grid-template-columns: 1fr 1fr; } .sbx-stack > .card { grid-column: 1 / -1; } }
  @media (min-width: 62rem) { .sbx-rail { position: sticky; top: 7.2rem; max-height: calc(100vh - 8rem);
                                          overflow-y: auto; scrollbar-width: thin; } }
  .num, .fig { font-family: var(--mono); font-variant-numeric: tabular-nums; }

  /* the app's GlowCard: 20dp corners and a tinted radial wash, tint from the account's direction */
  .glow { border-radius: 1.25rem; padding: 1rem 1.1rem; min-width: 0;
          background: radial-gradient(120% 140% at 100% 0%, var(--tint, rgba(180,160,255,.16)) 0%, transparent 60%),
                      var(--card); border: 1px solid #ffffff0d; }
  .glow.up { --tint: rgba(94,221,156,.16); } .glow.down { --tint: rgba(232,102,63,.16); }
  .pill { display: inline-flex; align-items: center; gap: .3rem; border-radius: 999px; padding: .2rem .65rem;
          font-size: .82rem; font-weight: 500; background: color-mix(in srgb, var(--c, var(--muted)) 14%, transparent);
          color: var(--c, var(--muted)); white-space: nowrap; }
  .pill.up { --c: var(--ok); } .pill.down { --c: var(--err); } .pill.acc { --c: var(--accent); }
  .pill.amb { --c: var(--amber); }
  .pills { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .6rem; }
  .up-t { color: var(--ok); } .down-t { color: var(--err); } .amb-t { color: var(--amber); }

  /* Today */
  .today-sum { display: flex; align-items: center; gap: .7rem; width: 100%; background: none; color: inherit;
               padding: 0; border-radius: 0; font-size: .92rem; text-align: left; flex-wrap: wrap; }
  .today-sum .when { color: var(--muted); font-size: .8rem; letter-spacing: .04em; }
  .today-sum .tdot::before { content: "●"; margin-right: .3rem; font-size: .7em; vertical-align: .15em; }
  .today-sum .chev { margin-left: auto; color: var(--muted); transition: transform .2s; }
  .today-sum[aria-expanded="true"] .chev { transform: rotate(180deg); }
  .today-list { margin-top: .6rem; }
  .today-arm { padding: .6rem 0; border-top: 1px solid var(--line); }
  .today-arm h3 { margin: 0 0 .25rem; font-size: .92rem; font-weight: 500; display: flex; gap: .45rem; align-items: center; }
  .today-arm h3::before { content: ""; width: .5rem; height: .5rem; border-radius: 50%; background: var(--dotc, var(--accent)); }
  .today-arm.rules h3 { --dotc: #14B8A6; }
  .t-row { display: grid; grid-template-columns: 1fr auto; gap: .6rem; font-size: .88rem; padding: .12rem 0; }
  .t-row b { color: var(--accent); font-weight: 600; cursor: pointer; }
  .t-row.skip b { color: var(--amber); }
  .t-row .why { color: var(--muted); grid-column: 1 / -1; font-size: .8rem; margin-top: -.1rem; }
  .today-held { color: var(--muted); font-size: .82rem; padding-top: .6rem; border-top: 1px solid var(--line); }

  /* chips: the app's FilterChips */
  .sbx-chips { display: flex; flex-wrap: wrap; gap: .45rem; }
  .sbx-chip { background: transparent; color: var(--ink); border: 1px solid var(--outline); border-radius: .55rem;
              padding: .4rem .8rem; font-size: .9rem; font-weight: 500; }
  .sbx-chip[aria-selected="true"] { background: var(--chip-sel); border-color: transparent; }
  .sbx-chip .n { color: var(--muted); margin-left: .2rem; }

  /* the race board */
  .board-head { display: flex; justify-content: space-between; align-items: baseline; gap: .6rem;
                color: var(--muted); font-size: .78rem; letter-spacing: .06em; margin-bottom: .35rem; }
  .board-head b { font-weight: 500; text-transform: uppercase; }
  .vs-row { display: grid; grid-template-columns: minmax(0, 1fr) 7.5rem 4rem; align-items: center; gap: .6rem;
            width: 100%; background: none; color: inherit; border-radius: .5rem; padding: .48rem .4rem;
            margin: 0 -.4rem; font-size: .93rem; text-align: left; width: calc(100% + .8rem); }
  .vs-row:hover { background: #ffffff08; }
  .vs-row[aria-current="true"] .lbl { font-weight: 600; }
  .vs-row[aria-current="true"] .lbl::after { content: " ●"; font-size: .7em; vertical-align: .15em; }
  .vs-row .lbl { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .vs-row .val { text-align: right; font-family: var(--mono); font-size: .92rem; }
  .vbar { position: relative; height: .62rem; }
  .vbar::before { content: ""; position: absolute; left: 50%; top: -.25rem; bottom: -.25rem; width: 1px; background: var(--muted); opacity: .7; }
  .vbar i { position: absolute; top: 0; bottom: 0; border-radius: .31rem; width: var(--w, 0%);
           transition: width .45s cubic-bezier(.2,.7,.2,1); }
  .vbar i.up { left: 50%; background: var(--ok); } .vbar i.down { right: 50%; background: var(--err); }
  .board-foot { color: var(--muted); font-size: .8rem; margin: .5rem 0 0; }
  .cmp-sec { color: var(--muted); font-size: .78rem; letter-spacing: .06em; text-transform: uppercase;
             display: flex; justify-content: space-between; margin: 1rem 0 .2rem; }
  .cmp-sec:first-child { margin-top: 0; }
  .cmp-pair { display: flex; justify-content: space-between; align-items: center; margin-top: .5rem; font-size: .95rem; }
  .cmp-pair + .vs-row { margin-top: .1rem; }

  /* account head */
  .acct-name { display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; color: var(--muted); font-size: .88rem; }
  .acct-name b { color: var(--ink); font-size: 1rem; font-weight: 500; }
  .acct-bal { display: flex; align-items: center; gap: .7rem; margin-top: .35rem; }
  .acct-bal .fig { font-size: 2.2rem; font-weight: 700; font-family: var(--sans); letter-spacing: -.01em; }
  .acct-note { color: var(--muted); font-size: .8rem; margin-top: .55rem; }
  .meter2 { height: .5rem; border-radius: .25rem; background: #ffffff1a; overflow: hidden; margin-top: .4rem; }
  .meter2 i { display: block; height: 100%; background: var(--accent); }

  /* equity curve */
  .curve-head { display: flex; justify-content: space-between; align-items: center; gap: .6rem; flex-wrap: wrap; }
  .curve-head h2 { margin: 0; }
  .seg { display: inline-flex; background: #ffffff0f; border-radius: .5rem; padding: .15rem; }
  .seg button { background: none; color: var(--muted); padding: .25rem .65rem; font-size: .8rem; border-radius: .4rem; }
  .seg button[aria-pressed="true"] { background: var(--chip-sel); color: var(--ink); }
  .curve-wrap { position: relative; margin-top: .6rem; touch-action: pan-y; }
  .curve-wrap svg { display: block; width: 100%; height: 13rem; overflow: visible; }
  .cv-hi, .cv-lo { position: absolute; right: 0; font-size: .75rem; color: var(--muted); font-family: var(--mono);
                  background: color-mix(in srgb, var(--card) 70%, transparent); padding: 0 .2rem; pointer-events: none; }
  .cv-hi { top: -.2rem; } .cv-lo { bottom: 1rem; }
  .cv-dates { display: flex; justify-content: space-between; color: var(--muted); font-size: .75rem; margin-top: -.9rem; }
  .curve-tip { position: absolute; top: 0; pointer-events: none; background: var(--card2); border: 1px solid var(--line);
               border-radius: .5rem; padding: .35rem .55rem; font-size: .78rem; white-space: nowrap; transform: translateX(-50%); }
  .curve-tip[hidden] { display: none; }
  .legend { display: flex; gap: 1rem; color: var(--muted); font-size: .8rem; margin-top: .3rem; }
  .legend span::before { content: ""; display: inline-block; width: .9rem; height: 2px; margin-right: .35rem;
                         vertical-align: .25em; background: var(--lc); }
  .legend .bm::before { background: repeating-linear-gradient(90deg, var(--lc) 0 4px, transparent 4px 7px); }

  /* holdings */
  .hold { display: grid; grid-template-columns: minmax(4.5rem, 1fr) auto auto; gap: .2rem .8rem; align-items: baseline;
          padding: .5rem 0; border-top: 1px solid var(--line); font-size: .9rem; }
  .hold-grid { display: grid; grid-template-columns: 1fr; column-gap: 1.6rem; }
  @media (min-width: 80rem) { .hold-grid { grid-template-columns: 1fr 1fr; } }
  .hold b { color: var(--accent); font-weight: 600; cursor: pointer; }
  .hold .sub2 { color: var(--muted); font-size: .78rem; grid-column: 1 / -1; }
  .hold .w { grid-column: 1 / -1; height: 3px; border-radius: 2px; background: #ffffff12; }
  .hold .w i { display: block; height: 100%; border-radius: 2px; background: var(--accent); opacity: .55; }

  /* contract cards (the app's ContractCard) */
  .contract { background: var(--card); border-radius: 1rem; padding: .85rem 1rem; display: grid; gap: .32rem; cursor: pointer; }
  .contract:hover { outline: 1px solid var(--line); }
  .c-top { display: grid; grid-template-columns: auto minmax(0, 1fr) auto; gap: .55rem; align-items: center; }
  .c-top .t { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .c-sub { font-size: .8rem; }
  .gauge { position: relative; height: .3rem; border-radius: .15rem; margin: .55rem 0 .15rem;
           background: linear-gradient(90deg, color-mix(in srgb, var(--err) 75%, transparent) 0 var(--cost),
                                      color-mix(in srgb, var(--ok) 70%, transparent) var(--cost) 100%); }
  .gauge .cost { position: absolute; left: var(--cost); top: -.35rem; bottom: -.35rem; width: 2px; background: var(--ink); opacity: .75; }
  .gauge .now { position: absolute; left: var(--at); top: 50%; width: .85rem; height: .85rem; border-radius: 50%;
                transform: translate(-50%, -50%); background: var(--dotc); box-shadow: 0 0 0 3px var(--card); }
  .g-lbl { display: flex; justify-content: space-between; color: var(--muted); font-size: .78rem; }
  .days { position: relative; height: .3rem; border-radius: .15rem; background: #ffffff1a; margin: .45rem 0 .15rem; }
  .days i { position: absolute; left: 0; top: 0; bottom: 0; border-radius: .15rem; background: var(--accent); width: var(--held); min-width: .3rem; }
  .days .exit { position: absolute; left: var(--exit); top: -.3rem; bottom: -.3rem; width: 2px; background: var(--amber); }
  .c-line { font-size: .86rem; color: var(--muted); }
  .c-line.need { color: var(--ink); font-size: .93rem; }

  .day-posture { font-size: .9rem; line-height: 1.55; margin: 0; display: -webkit-box; -webkit-line-clamp: 3;
                 -webkit-box-orient: vertical; overflow: hidden; }
  .day-posture.open { -webkit-line-clamp: unset; display: block; }
  .linkish { background: none; color: var(--accent); padding: .2rem 0; font-size: .82rem; }

  .sbx-fold > summary { cursor: pointer; font-weight: 500; font-size: .95rem; list-style: none; display: flex;
                        align-items: center; gap: .5rem; }
  .sbx-fold > summary::-webkit-details-marker { display: none; }
  .sbx-fold > summary::after { content: "⌄"; margin-left: auto; color: var(--muted); transition: transform .2s; }
  .sbx-fold[open] > summary::after { transform: rotate(180deg); }
  .sbx-fold[open] > summary { margin-bottom: .7rem; }
  .opt-limits { display: grid; grid-template-columns: repeat(auto-fit, minmax(10rem, 1fr)); gap: .4rem .9rem;
                margin-top: .9rem; font-size: .85rem; color: var(--muted); }
  .opt-limits b { color: var(--ink); font-weight: 500; font-family: var(--mono); }
  #sbx-log-tbl td { font-size: .82rem; vertical-align: top; }
  #sbx-log-tbl td.why { color: var(--muted); max-width: 26rem; }
  .err-box { color: var(--err); font-size: .88rem; }
  @media (prefers-reduced-motion: reduce) { .vbar i, .today-sum .chev, .sbx-fold > summary::after { transition: none; } }
"""

JS = r"""
  // ================================================================ sandbox (mirrors the app)
  let ARMS = [], selArm = null, GROUP = "all", TODAY = null, ACCT = null, RANGE = "3m";
  const GROUPS = [["all", "Stocks"], ["etf", "ETFs"], ["options", "Options"]];
  const groupOf = (u) => (u === "etf" ? "etf" : u === "options" ? "options" : "all");
  const armsIn = (g) => ARMS.filter(a => groupOf(a.universe) === g);
  // The app's vsShown: an account that has never run has no figure, not a confident +0.00.
  const vsShown = (a) => (a && a.last_tick_date && a.vs_benchmark_pct != null) ? a.vs_benchmark_pct : null;
  const shortLabel = (a) => {
    const full = a.label || a.arm; const m = full.match(/\s*\(([^)]*)\)\s*$/);
    return m && m[1].length > 8 ? (full.slice(0, m.index).trim() || full) : full;
  };
  const MINUS = "−";
  const sgn = (v, dp) => (v >= 0 ? "+" : MINUS) + Math.abs(v).toFixed(dp == null ? 2 : dp);
  const usd = (v, dp) => v == null ? "—" : (v < 0 ? MINUS : "") + "$" + Math.abs(v).toLocaleString("en-US",
    {minimumFractionDigits: dp == null ? 2 : dp, maximumFractionDigits: dp == null ? 2 : dp});
  const shortDate = (iso) => { if (!iso) return ""; const d = new Date(iso + "T12:00:00");
    return isNaN(d) ? iso : d.toLocaleDateString("en-US", {month: "short", day: "numeric"}); };
  const engineName = (a) => a.engine === "rules" ? "Rules" : a.engine === "rejects" ? "Cut trades" : "AI";
  const defaultArm = (g) => {
    if (g === "all") return ARMS.some(a => a.arm === "main") ? "main" : (armsIn("all")[0] || {}).arm;
    const xs = armsIn(g); return ((xs.find(a => a.engine === "llm") || xs[0]) || {}).arm;
  };

  // ---------------------------------------------------------------- route: #p-sandbox/<group>/<arm>
  function readRoute() {
    const p = location.hash.slice(1).split("/");
    if (p[0] !== "p-sandbox") return null;
    return {group: p[1] || null, arm: p[2] ? decodeURIComponent(p[2]) : null};
  }
  function writeRoute(push) {
    const h = "#p-sandbox/" + GROUP + (selArm ? "/" + encodeURIComponent(selArm) : "");
    if (location.hash !== h) history[push ? "pushState" : "replaceState"](null, "", h);
  }
  window.addEventListener("popstate", () => {
    const r = readRoute(); if (!r) return;
    if (r.group) GROUP = r.group;
    if (r.arm && r.arm !== selArm) { selArm = r.arm; loadAccount(); }
    renderChips(); renderBoard();
  });

  // ---------------------------------------------------------------- Today card
  function renderToday() {
    const box = $("sbx-today");
    if (!TODAY || !TODAY.arms) { box.innerHTML = '<div class="hint">No run recorded yet. Accounts trade at 2:35 PM CT on market days.</div>'; return; }
    const t = TODAY.totals || {};
    const when = new Date((TODAY.ran_at || 0) * 1000);
    const whenTxt = (TODAY.date ? shortDate(TODAY.date) : "") +
      (TODAY.ran_at ? " · " + when.toLocaleTimeString([], {hour: "numeric", minute: "2-digit"}) : "");
    const held = TODAY.arms.filter(a => a.ran && !(a.filled || []).length && !(a.skipped || []).length);
    const open = box.dataset.open === "1";
    let h = '<button class="today-sum" type="button" aria-expanded="' + open + '" id="sbx-today-btn">' +
      '<span class="when">' + esc(whenTxt.toUpperCase()) + '</span>' +
      (t.buys ? '<span class="tdot up-t">' + t.buys + ' bought · ' + usd(t.bought, 0) + '</span>' : '') +
      (t.sells ? '<span class="tdot down-t">' + t.sells + ' sold · ' + usd(t.sold, 0) + '</span>' : '') +
      (t.skipped ? '<span class="tdot amb-t">' + t.skipped + ' skipped</span>' : '') +
      '<span class="tdot" style="color:var(--muted)">' + held.length + ' held</span>' +
      '<span class="chev" aria-hidden="true">⌄</span></button>';
    if (open) {
      h += '<div class="today-list">';
      TODAY.arms.filter(a => (a.filled || []).length || (a.skipped || []).length).forEach(a => {
        h += '<div class="today-arm ' + (a.engine === "rules" ? "rules" : "") + '"><h3>' + esc(a.label || a.arm) + '</h3>';
        (a.filled || []).forEach(f => {
          const sym = String(f.symbol || "");
          h += '<div class="t-row"><span><b data-sym="' + esc(f.instrument === "option" ? sym.split(" ")[0] : sym) + '">' +
            esc(sym) + '</b> <span class="hint">' + esc((f.side === "sell" ? "sold " : "") + "× " + (f.shares ?? "")) + '</span></span>' +
            '<span class="num">' + usd(f.gross) + '</span></div>';
        });
        (a.skipped || []).forEach(s => {
          h += '<div class="t-row skip"><span><b data-sym="' + esc(String(s.symbol || "").split(" ")[0]) + '">' + esc(s.symbol) +
            '</b> <span class="hint">skipped</span></span><span></span>' +
            (s.reason ? '<span class="why">' + esc(s.reason) + '</span>' : '') + '</div>';
        });
        h += '</div>';
      });
      if (held.length) h += '<div class="today-held">Held: ' + held.map(a => esc(shortLabel(a))).join(", ") + '</div>';
      h += '</div>';
    }
    box.innerHTML = h;
    $("sbx-today-btn").onclick = () => { box.dataset.open = open ? "0" : "1"; renderToday(); };
    box.querySelectorAll("b[data-sym]").forEach(b => b.onclick = () => openTickerDrawer(b.dataset.sym));
  }

  // ---------------------------------------------------------------- chips
  function renderChips() {
    const box = $("sbx-chips");
    const chips = GROUPS.filter(([g]) => g === "all" || armsIn(g).length)
      .map(([g, name]) => [g, name, armsIn(g).length]).concat([["compare", "Compare", null]]);
    box.innerHTML = chips.map(([g, name, n]) =>
      '<button type="button" role="tab" class="sbx-chip" data-g="' + g + '" aria-selected="' + (GROUP === g) + '">' +
      esc(name) + (n != null ? '<span class="n">' + n + '</span>' : '') + '</button>').join("");
    box.querySelectorAll(".sbx-chip").forEach(c => c.onclick = () => pickGroup(c.dataset.g));
  }
  function pickGroup(g) {
    GROUP = g;
    if (g !== "compare") {
      const keep = ARMS.find(a => a.arm === selArm);
      if (!keep || groupOf(keep.universe) !== g) { selArm = defaultArm(g); loadAccount(); }
    }
    writeRoute(true); renderChips(); renderBoard(true);
  }

  // ---------------------------------------------------------------- the race board
  function vsRow(label, v, scale, armId, sel, sub) {
    const w = v == null ? 0 : Math.min(50, Math.abs(v) / scale * 50);
    return '<button type="button" class="vs-row" data-arm="' + esc(armId) + '" aria-current="' + (!!sel) + '"' +
      ' aria-label="' + esc(label + ": " + (v == null ? "not run yet" : sgn(v) + " points vs the S&P")) + '">' +
      '<span class="lbl">' + esc(label) + (sub ? ' <span class="hint">' + esc(sub) + '</span>' : '') + '</span>' +
      '<span class="vbar">' + (v == null ? '' : '<i class="' + (v >= 0 ? "up" : "down") + '" data-w="' + w.toFixed(1) + '%"></i>') + '</span>' +
      '<span class="val ' + (v == null ? "" : v >= 0 ? "up-t" : "down-t") + '">' + (v == null ? "—" : sgn(v)) + '</span></button>';
  }
  function growBars(box, animate) {
    const bars = [...box.querySelectorAll(".vbar i")];
    const set = () => bars.forEach(i => i.style.setProperty("--w", i.dataset.w));
    if (animate) requestAnimationFrame(() => requestAnimationFrame(set)); else { bars.forEach(i => i.style.transition = "none"); set(); }
  }
  function renderBoard(animate) {
    const box = $("sbx-board");
    if (!ARMS.length) { box.innerHTML = '<div class="hint">No accounts yet. Create one below.</div>'; return; }
    if (GROUP === "compare") { renderCompare(box); growBars(box, animate); return; }
    const xs = armsIn(GROUP).slice().sort((a, b) => {
      const x = vsShown(a), y = vsShown(b);
      if ((x == null) !== (y == null)) return x == null ? 1 : -1;
      return (y || 0) - (x || 0);
    });
    const scale = Math.max(1, ...xs.map(a => Math.abs(vsShown(a) || 0)));
    const name = (GROUPS.find(g => g[0] === GROUP) || [0, "Accounts"])[1];
    box.innerHTML = '<div class="board-head"><b>' + esc(name) + ': who\'s beating the S&amp;P</b><span>pts</span></div>' +
      xs.map(a => vsRow(shortLabel(a), vsShown(a), scale, a.arm, a.arm === selArm, a.enabled ? "" : "paused")).join("") +
      '<p class="board-foot">Points ahead of or behind the same money in the S&amp;P. The centre line is the S&amp;P.</p>';
    box.querySelectorAll(".vs-row").forEach(r => r.onclick = () => pickArm(r.dataset.arm));
    growBars(box, animate);
  }
  function pickArm(id) {
    if (id === selArm) { $("sbx-head").scrollIntoView({behavior: "smooth", block: "nearest"}); return; }
    selArm = id; writeRoute(true); renderBoard(); loadAccount();
    if (window.innerWidth < 992) $("sbx-head").scrollIntoView({behavior: "smooth", block: "start"});
  }

  // ---------------------------------------------------------------- Compare (the app's Matchups)
  function renderCompare(box) {
    const by = Object.fromEntries(ARMS.map(a => [a.arm, a]));
    const gap = (a, b) => { const x = vsShown(a), y = vsShown(b); return x == null || y == null ? null : x - y; };
    const title = (a) => a.universe === "options" ? "Calls + puts" : a.universe === "etf" ? (a.single_share ? "One ETF a day" : "ETFs")
      : a.single_share ? "One share a day" : "Stocks";
    const order = ["all", "etf", "options"];
    const twins = ARMS.filter(a => a.engine === "rules").map(r => {
      const ai = by[r.arm === "rules" ? "main" : r.arm.endsWith("-rules") ? r.arm.slice(0, -6) : ""];
      return ai ? {title: title(ai), ai, rules: r} : null;
    }).filter(Boolean).sort((x, y) => order.indexOf(groupOf(x.ai.universe)) - order.indexOf(groupOf(y.ai.universe)) ||
      (x.ai.single_share ? 1 : 0) - (y.ai.single_share ? 1 : 0));
    const cut = ARMS.find(a => a.engine === "rejects"), kept = by.reviewed;
    const base = by.main;
    const tweaks = ARMS.filter(a => a.engine === "llm" && groupOf(a.universe) === "all" && !a.single_share && a.arm !== "main")
      .map(a => ({a, v: base ? gap(a, base) : null})).sort((x, y) => (y.v ?? -1e9) - (x.v ?? -1e9));
    const scale = Math.max(1, ...ARMS.map(a => Math.abs(vsShown(a) || 0)), ...tweaks.map(t => Math.abs(t.v || 0)));
    const pair = (t, g) => '<div class="cmp-pair"><span>' + esc(t) + '</span><span class="pill">' +
      (g == null ? "not run yet" : "gap " + Math.abs(g).toFixed(2) + " pts") + '</span></div>';
    let h = "";
    if (twins.length) {
      h += '<div class="cmp-sec"><span>Does the AI beat plain rules?</span><span>pts vs S&amp;P</span></div>';
      twins.forEach(t => {
        h += pair(t.title, gap(t.ai, t.rules)) + vsRow("AI", vsShown(t.ai), scale, t.ai.arm, t.ai.arm === selArm) +
          vsRow("Rules", vsShown(t.rules), scale, t.rules.arm, t.rules.arm === selArm);
      });
    }
    if (cut && kept) {
      h += '<div class="cmp-sec"><span>Does the reviewer help?</span><span>pts vs S&amp;P</span></div>' + pair("Kept vs cut", gap(kept, cut)) +
        vsRow("Trades it kept", vsShown(kept), scale, kept.arm, kept.arm === selArm) +
        vsRow("Trades it cut", vsShown(cut), scale, cut.arm, cut.arm === selArm);
    }
    if (tweaks.length) {
      h += '<div class="cmp-sec"><span>Which tweak helps?</span><span>pts vs Baseline</span></div>' +
        tweaks.map(t => vsRow(shortLabel(t.a), t.v, scale, t.a.arm, t.a.arm === selArm)).join("");
    }
    box.innerHTML = h + '<p class="board-foot">Click a row to open that account. Most gaps here are fractions of a point.</p>';
    box.querySelectorAll(".vs-row").forEach(r => r.onclick = () => pickArm(r.dataset.arm));
  }

  // ---------------------------------------------------------------- the open account
  let acctSeq = 0;
  async function loadAccount() {
    const id = selArm; if (!id) return;
    const seq = ++acctSeq;
    const a = ARMS.find(x => x.arm === id);
    if (a) renderHead(a, null);
    try {
      const [st, nav, tr] = await Promise.all([
        afetch("/sandbox/state?arm=" + encodeURIComponent(id)).then(r => { if (!r.ok) throw new Error("state " + r.status); return r.json(); }),
        afetch("/sandbox/nav?days=400&arm=" + encodeURIComponent(id)).then(r => r.ok ? r.json() : {series: []}).catch(() => ({series: []})),
        afetch("/sandbox/trades?limit=120&arm=" + encodeURIComponent(id)).then(r => r.ok ? r.json() : {trades: []}).catch(() => ({trades: []})),
      ]);
      if (seq !== acctSeq) return;           // a newer pick won the race
      ACCT = {id, st, nav: nav.series || [], trades: tr.trades || []};
      renderAccount();
    } catch (e) {
      if (seq !== acctSeq) return;
      $("sbx-head").innerHTML = '<div class="err-box">Couldn\'t load this account: ' + esc(e.message) + '. It refreshes every minute.</div>';
    }
  }
  function renderAccount() {
    if (!ACCT) return;
    const a = ARMS.find(x => x.arm === ACCT.id) || {arm: ACCT.id, label: ACCT.st.label};
    const isOpt = ACCT.st.universe === "options" || a.universe === "options";
    renderHead(a, ACCT.st, isOpt);
    renderCurve();
    if (isOpt) { $("sbx-holdings").hidden = true; renderContracts(ACCT.st); }
    else { $("sbx-contracts").hidden = true; renderHoldings(ACCT.st); }
    renderDay(a);
    renderLog();
    fillSettings(a, ACCT.st, isOpt);
  }

  function renderHead(a, st, isOpt) {
    const box = $("sbx-head");
    const ran = !!(a.last_tick_date || (st && st.last_tick_date));
    const vs = vsShown(st ? Object.assign({}, a, {vs_benchmark_pct: st.vs_benchmark_pct, last_tick_date: st.last_tick_date || a.last_tick_date}) : a);
    box.className = "glow " + (vs == null ? "" : vs >= 0 ? "up" : "down");
    const eq = st ? st.equity : a.equity, cash = st ? st.cash : a.cash;
    const model = st && st.settings && st.settings.model;
    let h = '<div class="acct-name"><b>' + esc(a.label || a.arm) + '</b><span class="pill acc">' + engineName(a) + '</span>' +
      (model ? '<span class="pill">' + esc(model) + '</span>' : '') + (a.enabled === false ? '<span class="pill amb">Paused</span>' : '') + '</div>' +
      '<div class="acct-bal"><span class="fig">' + usd(eq) + '</span><span class="pill">PAPER</span></div>';
    if (isOpt && st) {
      const atRisk = st.premium_at_risk_usd || 0, n = (st.contracts || []).length;
      const share = eq > 0 ? Math.min(1, atRisk / eq) : 0;
      h += '<div class="hint" style="margin-top:.5rem">Premium in play</div><div class="meter2"><i style="width:' + (share * 100).toFixed(1) + '%"></i></div>';
      h += '<div class="pills"><span class="pill acc">' + (n ? usd(atRisk) + " in " + n + " contract" + (n === 1 ? "" : "s") + " · " + Math.round(share * 100) + "%" : "No contracts held") + '</span>';
    } else {
      h += '<div class="pills">';
      const pos = st ? (st.positions || []).length : a.positions;
      if (pos != null) h += '<span class="pill acc">' + pos + ' holding' + (pos === 1 ? "" : "s") + '</span>';
    }
    h += '<span class="pill">Cash ' + usd(cash) + '</span>';
    h += ran && vs != null ? '<span class="pill ' + (vs >= 0 ? "up" : "down") + '">' + (vs >= 0 ? "▲ " : "▼ ") + Math.abs(vs).toFixed(1) +
      ' pts ' + (vs >= 0 ? "ahead of" : "behind") + ' the S&amp;P</span>' : '<span class="pill">Not run yet</span>';
    const ret = st ? st.total_return_pct : a.total_return_pct;
    if (ran && ret != null) h += '<span class="pill ' + (ret >= 0 ? "up" : "down") + '">' + sgn(ret, 1) + '% on ' + usd(st ? st.funded_total : a.funded_total, 0) + ' put in</span>';
    h += '</div>';
    if (isOpt && st && st.weekly_budget) h += '<div class="acct-note">' + esc(weekLine(st.weekly_budget)) + '</div>';
    if (st && st.new_entries_paused_runs_left) h += '<div class="acct-note amb-t">New contracts paused after a drawdown: ' + st.new_entries_paused_runs_left + ' run(s) left</div>';
    if (st && st.max_drawdown_pct) h += '<div class="acct-note">Worst drop from a peak: ' + MINUS + Math.abs(st.max_drawdown_pct).toFixed(1) + '%' +
      (st.current_drawdown_pct ? ' · now ' + MINUS + Math.abs(st.current_drawdown_pct).toFixed(1) + '% under its peak' : '') + '</div>';
    const last = (st && st.last_tick_date) || a.last_tick_date;
    h += '<div class="acct-note">' + (last ? "Last ran " + esc(shortDate(last)) : "Hasn't run yet. Accounts trade at 2:35 PM CT on market days.") + '</div>';
    box.innerHTML = h;
  }
  function weekLine(w) {
    const m = (v) => usd(v, 0);
    const head = w.spent_usd <= 0 ? "This week: " + m(w.room_usd) + " left to spend"
      : w.room_usd <= 0 ? "This week: budget used (" + m(w.spent_usd) + " of " + m(w.budget_usd) + ")"
      : "This week: " + m(w.spent_usd) + " of " + m(w.budget_usd) + " spent";
    return head + (w.spent_usd > 0 && w.frees_on ? " · more from " + shortDate(w.frees_on) : "");
  }

  // ---------------------------------------------------------------- equity curve with a hover readout
  function renderCurve() {
    const box = $("sbx-curve");
    const all = (ACCT.nav || []).filter(p => p.equity != null);
    box.hidden = false;
    if (all.length < 2) {
      box.innerHTML = '<h2>Balance vs the S&amp;P</h2><div class="hint">The curve appears after the second daily run.</div>';
      return;
    }
    const n = {"1m": 22, "3m": 66, "all": all.length}[RANGE] || all.length;
    const pts = all.slice(-n);
    const W = 640, H = 208, padL = 4, padR = 4, padT = 10, padB = 18;
    const vals = pts.flatMap(p => [p.equity, p.benchmark_value]).filter(v => v != null);
    let lo = Math.min(...vals), hi = Math.max(...vals); const pad = (hi - lo) * .08 || hi * .01; lo -= pad; hi += pad;
    const X = (i) => padL + i / (pts.length - 1) * (W - padL - padR);
    const Y = (v) => padT + (1 - (v - lo) / (hi - lo)) * (H - padT - padB);
    const line = (k) => pts.map((p, i) => p[k] == null ? null : (i ? "L" : "M") + X(i).toFixed(1) + " " + Y(p[k]).toFixed(1)).filter(Boolean).join(" ").replace(/^L/, "M");
    const last = pts[pts.length - 1], up = last.benchmark_value == null || last.equity >= last.benchmark_value;
    const col = up ? "var(--ok)" : "var(--err)";
    const area = line("equity") + " L" + X(pts.length - 1).toFixed(1) + " " + (H - padB) + " L" + X(0).toFixed(1) + " " + (H - padB) + " Z";
    const seg = ["1m", "3m", "all"].map(r => '<button type="button" data-r="' + r + '" aria-pressed="' + (RANGE === r) + '">' + r.toUpperCase() + '</button>').join("");
    box.innerHTML = '<div class="curve-head"><h2>Balance vs the S&amp;P</h2><div class="seg" role="group" aria-label="Range">' + seg + '</div></div>' +
      '<div class="curve-wrap"><svg viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="none" role="img" aria-label="Balance against the same money in the S&P">' +
      '<defs><linearGradient id="cg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="' + col + '" stop-opacity=".22"/><stop offset="1" stop-color="' + col + '" stop-opacity="0"/></linearGradient></defs>' +
      '<path d="' + area + '" fill="url(#cg)"/>' +
      '<path d="' + line("benchmark_value") + '" fill="none" stroke="var(--bench)" stroke-width="1.6" stroke-dasharray="5 4" vector-effect="non-scaling-stroke"/>' +
      '<path d="' + line("equity") + '" fill="none" stroke="' + col + '" stroke-width="2.2" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>' +
      '<line id="cv-x" y1="' + padT + '" y2="' + (H - padB) + '" stroke="var(--muted)" stroke-width="1" vector-effect="non-scaling-stroke" visibility="hidden"/>' +
      '</svg><span class="cv-hi">' + usd(hi - pad, 0) + '</span><span class="cv-lo">' + usd(lo + pad, 0) + '</span>' +
      '<div class="curve-tip" id="cv-tip" hidden></div></div>' +
      '<div class="cv-dates"><span>' + esc(shortDate(pts[0].date)) + '</span><span>' + esc(shortDate(last.date)) + '</span></div>' +
      '<div class="legend"><span style="--lc:' + col + '">This account</span><span class="bm" style="--lc:var(--bench)">Same money in the S&amp;P</span></div>';
    box.querySelectorAll(".seg button").forEach(b => b.onclick = () => { RANGE = b.dataset.r; renderCurve(); });
    const wrap = box.querySelector(".curve-wrap"), svg = wrap.querySelector("svg"), tip = $("cv-tip"), xl = $("cv-x");
    const show = (clientX) => {
      const r = svg.getBoundingClientRect();
      const i = Math.max(0, Math.min(pts.length - 1, Math.round((clientX - r.left) / r.width * (pts.length - 1))));
      const p = pts[i], x = X(i);
      xl.setAttribute("x1", x); xl.setAttribute("x2", x); xl.setAttribute("visibility", "visible");
      const g = p.benchmark_value ? (p.equity / p.benchmark_value - 1) * 100 : null;
      tip.innerHTML = '<b>' + esc(shortDate(p.date)) + '</b> · ' + usd(p.equity) +
        (p.benchmark_value != null ? '<br><span class="hint">S&amp;P ' + usd(p.benchmark_value) + '</span> · <span class="' +
          (g >= 0 ? "up-t" : "down-t") + '">' + sgn(g) + ' pts</span>' : '');
      tip.hidden = false;
      tip.style.left = Math.max(70, Math.min(r.width - 70, x / W * r.width)) + "px";
    };
    wrap.onpointermove = (e) => show(e.clientX);
    wrap.onpointerdown = (e) => show(e.clientX);
    wrap.onpointerleave = () => { tip.hidden = true; xl.setAttribute("visibility", "hidden"); };
  }

  // ---------------------------------------------------------------- holdings
  function renderHoldings(st) {
    const box = $("sbx-holdings"); box.hidden = false;
    const pos = (st.positions || []).slice().sort((x, y) => (y.value || 0) - (x.value || 0));
    if (!pos.length) { box.innerHTML = '<h2>Holdings</h2><div class="hint">Nothing held. The account is all cash.</div>'; return; }
    const eq = st.equity || 1;
    box.innerHTML = '<h2>Holdings <span class="hint">' + pos.length + '</span></h2><div class="hold-grid">' + pos.map(p => {
      const u = p.unrealized_pct;
      const stale = p.price == null;
      return '<div class="hold"><span><b data-sym="' + esc(p.symbol) + '">' + esc(p.symbol) + '</b></span>' +
        '<span class="num">' + (stale ? "≈" : "") + usd(p.value) + '</span>' +
        '<span class="num ' + (u == null || stale ? "" : u >= 0 ? "up-t" : "down-t") + '" style="min-width:4.2rem;text-align:right">' +
        (u == null || stale ? "—" : sgn(u, 1) + "%") + '</span>' +
        '<span class="sub2">' + (+p.shares).toLocaleString("en-US", {maximumFractionDigits: 4}) + ' sh at ' + usd(p.avg_cost) +
        (p.price != null ? ' · now ' + usd(p.price) : ' · no price today') + (p.exposure_group && p.exposure_group !== p.symbol ? ' · ' + esc(p.exposure_group) : '') + '</span>' +
        '<span class="w"><i style="width:' + Math.min(100, (p.value || 0) / eq * 100).toFixed(1) + '%"></i></span></div>';
    }).join("") + '</div>';
    box.querySelectorAll("b[data-sym]").forEach(b => b.onclick = () => openTickerDrawer(b.dataset.sym));
  }

  // ---------------------------------------------------------------- contracts (the app's ContractMath)
  const optSet = (s, k, d) => (s && s[k] != null ? +s[k] : d);
  function renderContracts(st) {
    const box = $("sbx-contracts"); box.hidden = false;
    const s = st.settings || {};
    const ex = {dte: optSet(s, "opt_exit_dte", 21), stop: optSet(s, "opt_stop_pct", -50), target: optSet(s, "opt_target_pct", 100)};
    const span = ex.target - ex.stop, cost = (0 - ex.stop) / span;
    const cs = st.contracts || [];
    const sc = st.options_scorecard;
    let h = cs.length ? "" : '<div class="card"><h2>Contracts</h2><div class="hint">No contracts held.</div></div>';
    cs.forEach(c => {
      const stale = !!c.mark_stale;
      const pl = stale ? null : c.pl_pct;
      const at = pl == null ? null : Math.max(0, Math.min(1, (pl - ex.stop) / span));
      const be = c.premium_paid == null ? null : c.right === "put" ? c.strike - c.premium_paid : c.strike + c.premium_paid;
      const dteDays = Math.floor(c.dte || 0), toExit = Math.max(0, dteDays - ex.dte);
      const life = (c.held_days || 0) + (c.dte || 0);
      const heldPct = life > 0 ? (c.held_days || 0) / life * 100 : 0, exitPct = life > 0 ? ((c.held_days || 0) + toExit) / life * 100 : 0;
      let now = "";
      if (be != null && c.spot != null) {
        const past = c.right === "put" ? c.spot <= be : c.spot >= be;
        now = "now " + usd(c.spot) + " · " + (past ? "past break-even" : sgn((be / c.spot - 1) * 100, 1) + "% to go");
      }
      const plTxt = [pl == null ? null : sgn(pl, 1) + "%", c.cost_total != null ? "paid " + usd(c.cost_total) : null,
        stale ? "no live quote" + (c.mark_date ? " since " + shortDate(c.mark_date) : "") : null].filter(Boolean).join(" · ");
      h += '<div class="contract" tabindex="0" role="button" data-sym="' + esc(c.underlying) + '" aria-label="' + esc(c.display) + ', open ' + esc(c.underlying) + '">' +
        '<div class="c-top"><span class="pill ' + (c.right === "put" ? "down" : "up") + '">' + esc(String(c.right).toUpperCase()) + '</span>' +
        '<span class="t">' + esc(c.count + " × " + c.display) + '</span>' +
        '<span class="num ' + (stale ? "amb-t" : "") + '">' + (c.value == null ? "—" : (stale ? "≈" : "") + usd(c.value)) + '</span></div>' +
        '<div class="c-sub ' + (stale ? "amb-t" : pl > 0 ? "up-t" : pl < 0 ? "down-t" : "hint") + '">' + esc(plTxt) + '</div>' +
        '<div class="gauge" style="--cost:' + (cost * 100).toFixed(1) + '%"><span class="cost"></span>' +
          (at == null ? '' : '<span class="now" style="--at:' + (at * 100).toFixed(1) + '%;--dotc:' + (pl >= 0 ? "var(--ok)" : "var(--err)") + '"></span>') + '</div>' +
        '<div class="g-lbl"><span>' + ex.stop + '% sell</span><span>cost</span><span>+' + ex.target + '% sell</span></div>' +
        '<div class="days" style="--held:' + heldPct.toFixed(1) + '%;--exit:' + exitPct.toFixed(1) + '%"><i></i><span class="exit"></span></div>' +
        '<div class="c-line">' + dteDays + ' days left · sold in ' + toExit + ' days at ' + ex.dte + ' left</div>' +
        (be != null ? '<div class="c-line need">needs ' + esc(c.underlying) + ' ' + (c.right === "put" ? "below" : "above") + ' ' + usd(be) + ' by ' + esc(shortDate(c.expiry)) + '</div>' : '') +
        (now ? '<div class="c-line">' + esc(now) + '</div>' : '') +
        (c.reason ? '<div class="c-line" title="' + esc(c.reason) + '" style="font-size:.8rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">Why: ' + esc(c.reason) + '</div>' : '') +
      '</div>';
    });
    if (sc) {
      h += '<div class="card"><h2>Options record</h2>' + (sc.closed ? '<div class="pills">' +
        '<span class="pill">' + sc.closed + ' closed</span>' +
        '<span class="pill ' + (sc.pl_usd >= 0 ? "up" : "down") + '">' + (sc.pl_usd >= 0 ? "+" : "") + usd(sc.pl_usd) + ' on ' + usd(sc.premium_deployed_usd) + '</span>' +
        (sc.index_alt_usd != null ? '<span class="pill">S&amp;P would have made ' + usd(sc.index_alt_usd) + '</span>' : '') +
        '<span class="pill">Right direction ' + sc.direction_right + ' of ' + sc.closed + '</span></div>' : '<div class="hint">No contract closed yet.</div>') +
        (sc.open ? '<div class="acct-note">Open: ' + sc.open + ' · ' + usd(sc.premium_at_risk_usd) + ' at risk</div>' : '') + '</div>';
    }
    box.innerHTML = h;
    box.querySelectorAll(".contract").forEach(el => {
      el.onclick = () => openTickerDrawer(el.dataset.sym);
      el.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openTickerDrawer(el.dataset.sym); } };
    });
  }

  // ---------------------------------------------------------------- this account's last run
  function renderDay(a) {
    const box = $("sbx-day");
    const t = TODAY && (TODAY.arms || []).find(x => x.arm === ACCT.id);
    const posture = (t && t.posture) || ACCT.st.last_posture;
    if (!t && !posture) { box.hidden = true; return; }
    box.hidden = false;
    let h = '<h2>Last run <span class="hint">' + esc(shortDate((t && t.last_tick_date) || ACCT.st.last_tick_date)) + '</span></h2>';
    const fills = (t && t.filled) || [], skips = (t && t.skipped) || [];
    fills.forEach(f => {
      h += '<div class="t-row"><span><b data-sym="' + esc(String(f.symbol).split(" ")[0]) + '">' + esc(f.symbol) + '</b> <span class="hint">' +
        esc((f.side === "sell" ? "sold " : "bought ") + "× " + f.shares) + '</span></span><span class="num">' + usd(f.gross) + '</span></div>';
    });
    skips.forEach(s => {
      h += '<div class="t-row skip"><span><b data-sym="' + esc(String(s.symbol).split(" ")[0]) + '">' + esc(s.symbol) + '</b> <span class="hint">skipped</span></span><span></span>' +
        (s.reason ? '<span class="why">' + esc(s.reason) + '</span>' : '') + '</div>';
    });
    if (!fills.length && !skips.length) h += '<div class="hint" style="margin-bottom:.4rem">No trades. It held.</div>';
    if (posture) h += '<p class="day-posture" id="sbx-posture" style="margin-top:.6rem">' + esc(posture) + '</p>' +
      '<button type="button" class="linkish" id="sbx-posture-more">Show all</button>';
    const note = ACCT.st.last_strategy_note;
    const noteTxt = note && (typeof note === "string" ? note : (note.notes || note.text || ""));
    if (noteTxt) h += '<details class="sbx-fold" style="margin-top:.6rem"><summary class="hint">Strategy note' +
      (note.date ? " · " + esc(shortDate(note.date)) : "") + '</summary><p class="day-posture open">' + esc(noteTxt) + '</p></details>';
    box.innerHTML = h;
    box.querySelectorAll("b[data-sym]").forEach(b => b.onclick = () => openTickerDrawer(b.dataset.sym));
    const p = $("sbx-posture"), more = $("sbx-posture-more");
    if (p && more) {
      if (p.scrollHeight <= p.clientHeight + 2) more.hidden = true;
      more.onclick = () => { const o = p.classList.toggle("open"); more.textContent = o ? "Show less" : "Show all"; };
    }
  }

  // ---------------------------------------------------------------- trade log
  function renderLog() {
    const rows = (ACCT.trades || []).filter(t => t.side !== "interest").slice().sort((x, y) => (y.ts || 0) - (x.ts || 0));
    $("sbx-log-n").textContent = rows.length ? String(rows.length) : "";
    const tb = $("sbx-log-tbl").querySelector("tbody");
    if (!rows.length) { tb.innerHTML = '<tr><td class="empty">No trades yet.</td></tr>'; return; }
    const sideTxt = {buy: "Bought", sell: "Sold", buy_to_open: "Bought", sell_to_close: "Sold", expired: "Expired", settled: "Settled", deposit: "Deposit"};
    tb.innerHTML = rows.map(t => {
      const s = sideTxt[t.side] || t.side, isSell = /sell|expired|settled/.test(t.side || "");
      const skipped = t.status && t.status !== "filled";
      return '<tr><td class="hint">' + esc(shortDate(t.date)) + '</td>' +
        '<td class="' + (skipped ? "amb-t" : isSell ? "down-t" : t.side === "deposit" ? "" : "up-t") + '">' + esc(skipped ? "Skipped" : s) + '</td>' +
        '<td><b>' + esc(t.display || t.symbol) + '</b>' + (t.shares ? ' <span class="hint">× ' + esc(t.shares) + '</span>' : '') + '</td>' +
        '<td class="num">' + (t.gross != null ? usd(t.gross) : "") + '</td>' +
        '<td class="why">' + esc(t.skip_reason || t.reason || "") + '</td></tr>';
    }).join("");
  }

  // ---------------------------------------------------------------- settings
  function fillSettings(a, st, isOpt) {
    const s = (st && st.settings) || {};
    $("arm-edit-name").textContent = a.label || a.arm;
    $("a-label").value = a.label || "";
    $("a-engine").value = a.engine === "rules" ? "rules" : "llm";
    $("a-model").value = s.model || "";
    $("a-risk").value = s.risk_tolerance || "balanced";
    $("a-maxnew").value = s.max_new_positions_per_tick ?? "";
    $("a-maxtrades").value = s.max_trades_per_tick ?? "";
    $("a-maxpos").value = s.max_position_pct ?? "";
    $("a-floor").value = s.cash_floor_pct ?? "";
    $("a-turnover").value = s.max_turnover_pct ?? "";
    $("a-conv").value = s.min_conviction_to_trade ?? "";
    $("a-toggle").textContent = a.enabled === false ? "Resume" : "Pause";
    document.querySelectorAll("#arm-edit .share-only").forEach(el => el.hidden = isOpt);
    const ol = $("a-optlimits");
    ol.hidden = !isOpt;
    if (isOpt) {
      const v = (k, d, f) => { const x = s[k] != null ? s[k] : d; return f ? f(x) : x; };
      ol.innerHTML = [
        ["One contract", v("opt_max_premium_pct", 15) + "% of the account"],
        ["All contracts", v("opt_max_at_risk_pct", 30) + "% of the account"],
        ["New per week", v("opt_max_new_week_pct", 15) + "% of the account"],
        ["New per run", v("opt_max_new_per_tick", 1)],
        ["Open at once", v("opt_max_open", 6)],
        ["Sells at", v("opt_stop_pct", -50) + "% / +" + v("opt_target_pct", 100) + "%"],
        ["Sells with", v("opt_exit_dte", 21) + " days left"],
      ].map(([k, x]) => '<div>' + esc(k) + '<br><b>' + esc(x) + '</b></div>').join("") +
        '<div style="grid-column:1/-1" class="hint">Option limits are set on the server and are the same for both options accounts.</div>';
    }
    const isMain = a.arm === "main";
    $("a-delete").style.display = isMain ? "none" : "";
    $("a-engine").disabled = isMain;
    $("a-mainnote").hidden = !isMain;
  }

  async function saveArm() {
    if (!selArm) return;
    const st = $("a-status"); st.textContent = "Saving…"; st.className = "hint";
    const num = (id) => { const el = $(id); if (el.closest("[hidden]")) return null; const v = el.value.trim(); return v === "" ? null : Number(v); };
    const patch = {
      label: $("a-label").value.trim() || null,
      model: $("a-model").value.trim(),
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
      const r = await afetch("/sandbox/settings?arm=" + encodeURIComponent(selArm),
        {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(patch)});
      if (!r.ok) throw new Error(((await r.json().catch(() => ({}))).detail) || r.status);
      st.textContent = "Saved"; st.className = "ok-t";
      await loadArms(); loadAccount();
    } catch (e) { st.textContent = "Not saved: " + e.message; st.className = "err-t"; }
  }
  async function toggleArm() {
    if (!selArm) return;
    const a = ARMS.find(x => x.arm === selArm); if (!a) return;
    const st = $("a-status"); st.textContent = "…"; st.className = "hint";
    try {
      const r = await afetch("/sandbox/settings?arm=" + encodeURIComponent(selArm),
        {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({master_enabled: !a.enabled})});
      if (!r.ok) throw new Error(r.status);
      st.textContent = a.enabled ? "Paused" : "Resumed"; st.className = "ok-t";
      await loadArms(); loadAccount();
    } catch (e) { st.textContent = "Couldn't change it: " + e.message; st.className = "err-t"; }
  }
  async function deleteArm() {
    if (!selArm || selArm === "main") return;
    // Irreversible and it takes the account's whole history with it, so it asks and names the account.
    if (!confirm("Delete the account '" + selArm + "' and its whole trade history? This can't be undone.")) return;
    const st = $("a-status"); st.textContent = "Deleting…"; st.className = "hint";
    try {
      const r = await afetch("/sandbox/arms/" + encodeURIComponent(selArm), {method: "DELETE"});
      if (!r.ok) throw new Error(((await r.json().catch(() => ({}))).detail) || r.status);
      selArm = null; $("arm-edit").open = false;
      await loadArms(); selArm = defaultArm(GROUP === "compare" ? "all" : GROUP); writeRoute(false); renderBoard(); loadAccount();
    } catch (e) { st.textContent = "Not deleted: " + e.message; st.className = "err-t"; }
  }
  async function createArm() {
    const st = $("n-status"); const btn = $("n-create");
    const id = $("n-id").value.trim().toLowerCase();
    if (!id) { st.textContent = "Give it an id first."; st.className = "err-t"; return; }
    btn.disabled = true; st.textContent = "Creating…"; st.className = "hint";
    const clone = $("n-clone").value || null;
    // Funding a copy on top of the copied book would count its cash twice.
    const body = {arm: id, engine: $("n-engine").value, label: $("n-label").value.trim() || null,
                  clone_from: clone, fund: clone ? 0 : Number($("n-fund").value || 0), enabled: true};
    const model = $("n-model").value.trim();
    try {
      const r = await afetch("/sandbox/arms", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
      if (!r.ok) throw new Error(((await r.json().catch(() => ({}))).detail) || r.status);
      if (model) await afetch("/sandbox/settings?arm=" + encodeURIComponent(id),
        {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({model})});
      st.textContent = "Created"; st.className = "ok-t";
      $("n-id").value = ""; $("n-label").value = ""; $("n-model").value = "";
      await loadArms(); pickArm(id);
    } catch (e) { st.textContent = "Not created: " + e.message; st.className = "err-t"; }
    finally { btn.disabled = false; }
  }
  $("a-save").onclick = saveArm;
  $("a-toggle").onclick = toggleArm;
  $("a-cancel").onclick = () => { $("arm-edit").open = false; };
  $("a-delete").onclick = deleteArm;
  $("n-create").onclick = createArm;
  $("n-clone").onchange = () => { const c = !!$("n-clone").value; $("n-fund").disabled = c;
    $("n-fund").title = c ? "A copy already carries the source account's cash" : ""; };

  // ---------------------------------------------------------------- macro
  async function loadMacro() {
    let m; try { m = await (await afetch("/macro/catalysts")).json(); } catch (e) { return; }
    if (!m || !m.available) return;
    const box = $("sbx-macro"); box.hidden = false;
    const lvl = String(m.risk_level || "").toLowerCase();
    const c = lvl === "elevated" || lvl === "high" ? "amb" : lvl === "calm" || lvl === "low" ? "up" : "";
    $("sbx-macro-sum").innerHTML = 'Macro backdrop <span class="pill ' + c + '">' + esc(lvl.toUpperCase() || "UNKNOWN") + '</span>' +
      (m.stale ? ' <span class="hint amb-t">stale</span>' : '');
    $("sbx-macro-body").innerHTML = '<p style="margin:0 0 .5rem;font-weight:500">' + esc(m.headline || "") + '</p>' +
      (m.bullets || []).map(b => '<div class="hint" style="padding:.15rem 0">' + esc(b) + '</div>').join("") +
      (m.age_seconds != null ? '<div class="acct-note">' + esc(agoText(Date.now() / 1000 - m.age_seconds)) + '</div>' : '');
  }

  // ---------------------------------------------------------------- load + refresh
  async function loadArms() {
    try {
      const r = await afetch("/sandbox/arms?show_options=true");
      if (!r.ok) throw new Error(r.status);
      ARMS = (await r.json()).arms || [];
    } catch (e) {
      $("sbx-board").innerHTML = '<div class="err-box">Couldn\'t load the accounts (' + esc(e.message) + '). Check the access token, then reload.</div>';
      return;
    }
    const sel = $("n-clone"), keep = sel.value;
    sel.innerHTML = '<option value="">Start empty</option>' + ARMS.map(a => '<option value="' + esc(a.arm) + '">' + esc(a.label || a.arm) + '</option>').join("");
    sel.value = keep || "main";
    if (selArm && !ARMS.some(a => a.arm === selArm)) selArm = null;
    renderChips(); renderBoard();
  }
  async function loadToday() {
    try { const r = await afetch("/sandbox/today?show_options=true"); TODAY = r.ok ? await r.json() : null; } catch (e) { TODAY = null; }
    renderToday();
  }
  async function bootSandbox() {
    const r = readRoute();
    if (r && r.group && (r.group === "compare" || GROUPS.some(g => g[0] === r.group))) GROUP = r.group;
    await Promise.all([loadArms(), loadToday()]);
    selArm = (r && r.arm && ARMS.some(a => a.arm === r.arm)) ? r.arm : defaultArm(GROUP === "compare" ? "all" : GROUP);
    if (r) writeRoute(false);
    renderBoard(true); loadAccount(); loadMacro();
  }
  bootSandbox();
  // The board and today refresh each minute; the open account too, but only while the tab is showing,
  // because an options account's state prices live chains.
  setInterval(async () => {
    await Promise.all([loadArms(), loadToday()]);
    if (!$("p-sandbox").hidden && !document.hidden) loadAccount();
  }, 60000);
"""
