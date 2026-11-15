"""Dry run of the long-options arms against REAL chains and the REAL analyst, on a throwaway data
directory. Never touches /opt/signals/data. Run on the signals container:

    cd /opt/signals-dryrun && set -a && . /opt/signals/.env && set +a && \
    SIGNALS_DATA_DIR=/tmp/signals-dryrun-data /opt/signals/.venv/bin/python tools/options_dryrun.py [--live-quotes]

`--live-quotes` pretends the chains' market_state is REGULAR so the fill path is exercised on a
closed day (fills then happen at the last session's asks). Without it a closed-day run shows the
"quotes not live" refusals, which is the honest result on a weekend.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time

import httpx

sys.path.insert(0, os.getcwd())
from app import main, options_arm, sandbox_job, sandbox_store  # noqa: E402

LIVE = "--live-quotes" in sys.argv
POOL = ["AAPL", "NVDA", "AMD", "SOFI", "F", "MSFT", "TSLA", "GOOGL", "META", "AMZN", "PLTR", "INTC"]


def _mk(arm: str, engine: str, spy: float) -> None:
    if arm in sandbox_store.list_arms():
        return
    sandbox_store.create_arm(arm, engine=engine, label=f"Options ({engine})", settings={
        "universe": "options", "master_enabled": True, "monthly_deposit": 500.0,
        "deposit_frequency": "monthly", "cash_floor_pct": 50.0, "min_conviction_to_trade": 55,
        "review_enabled": False, "gate_enabled": False, "cash_apy_pct": 4.3})
    b = sandbox_store.get(arm)
    b["cash"] = 10_000.0
    b["funded_total"] = 10_000.0
    b["benchmark"] = {"symbol": "^GSPC", "shares": round(10_000 / spy, 6), "cost_basis": 10_000.0}
    b["last_deposit_period"] = sandbox_job.due_deposit_periods(b)[-1] if sandbox_job.due_deposit_periods(b) else None
    sandbox_store.save(b, arm)


async def run() -> None:
    async with httpx.AsyncClient(timeout=30) as client:
        main._http = client
        spy0 = (await main.market_now.fetch_quotes(client, ["^GSPC"])).get("^GSPC", {}).get("price")
        _mk("options", "llm", spy0)
        _mk("options-rules", "rules", spy0)
        ids = ["options", "options-rules"]
        t0 = time.monotonic()
        rows = [c for c in await asyncio.gather(*[main._sandbox_candidate(s, None, set()) for s in POOL]) if c]
        print(f"pool rows: {len(rows)} of {len(POOL)} in {time.monotonic() - t0:.1f}s")
        t0 = time.monotonic()
        ctx = await main._options_context(ids, candidates=rows, held_rows=[], bench_closes=None)
        print(f"context: {len(ctx['candidates'])} candidates, {len(ctx['marks'])} marks, "
              f"{len(ctx['errors'])} errors in {time.monotonic() - t0:.1f}s")
        for e in ctx["errors"][:8]:
            print("  error:", e)
        for c in ctx["candidates"]:
            call = (c["calls"].get("atm") or c["calls"].get("otm") or {})
            put = (c["puts"].get("atm") or c["puts"].get("otm") or {})
            print(f"  {c['symbol']:6} spot={c['price']:<9} state={c['market_state']:<8} score={c['directional']['score']:<3} "
                  f"earn={c['earnings']} dte={c['dte']} ivr={c['iv_rank']} em%={c['expected_move_pct']} "
                  f"call=${call.get('cost_one')} d={call.get('delta')} put=${put.get('cost_one')} d={put.get('delta')}")
        if LIVE:
            for c in ctx["candidates"]:
                c["market_state"] = "REGULAR"
        # the tick gate would refuse a closed day; this is a dry run of the arm cycle, not of the gate
        main.sandbox_job.tick_gate = lambda blob, **kw: (True, "ok")
        now = sandbox_job.now_et()
        spy = (await main.market_now.fetch_quotes(client, ["^GSPC"])).get("^GSPC", {}).get("price")
        print("spy:", spy)
        for arm in ids:
            t0 = time.monotonic()
            out = await main._run_options_arm(arm, now=now, spy_price=spy, macro_block=None, force=True, opt_ctx=ctx)
            print(f"\n=== {arm} ({time.monotonic() - t0:.1f}s) status={out.get('status')}")
            print("posture:", out.get("posture"))
            for w in out.get("warnings") or []:
                print("warning:", w)
            for r in out.get("orders_filled") or []:
                print("FILLED:", r["side"], r["display"], "x", r["shares"], "@", r["price"], "=", r["gross"],
                      "| conviction", r.get("conviction"), "|", (r.get("reason") or "")[:140])
            for r in out.get("orders_skipped") or []:
                print("skipped:", r.get("display"), "|", r.get("skip_reason"), "|", (r.get("reason") or "")[:100])
            print("nav:", json.dumps(out.get("nav")))
            st = await main._options_state(arm, sandbox_store.get(arm))
            print("state: equity", st["equity"], "cash", st["cash"], "contracts", len(st["contracts"]),
                  "at risk", st["premium_at_risk_usd"], "stale", st["stale_marks"], "score", st["options_scorecard"])
            for c in st["contracts"]:
                print("   held:", c["display"], "x", c["count"], "paid", c["premium_paid"], "mark", c["mark"],
                      "stale" if c["mark_stale"] else "fresh", "pl%", c["pl_pct"], "dte", c["dte"])
        arms = await main.sandbox_arms_endpoint(show_options=True)
        print("\n/sandbox/arms (shown):", [(a["arm"], a["universe"], a["positions"], a["equity"]) for a in arms["arms"]])
        arms = await main.sandbox_arms_endpoint(show_options=False)
        print("/sandbox/arms (default):", [a["arm"] for a in arms["arms"]])
        print("usage rows:", sum(1 for _ in open(os.path.join(os.environ["SIGNALS_DATA_DIR"], "usage.jsonl")))
              if os.path.exists(os.path.join(os.environ["SIGNALS_DATA_DIR"], "usage.jsonl")) else "none")


asyncio.run(run())
