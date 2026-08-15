"""Adversarial invariant fuzzer for the sandbox ledger.

Run:  .venv/bin/python tools/fuzz_ledger.py          (4000 scenarios, seed 7)

Not part of the pytest suite -- it is a HUNT, not a regression check. Anything it finds should be
converted into a named test in tests/ and fixed there; this file exists to keep finding the next one.
It has already produced tests/test_broken_quote_guards.py.

Deliberately NOT uniform random. The 2026-08-13 GOOGL bug survived 4000 random-price trials because
random floats never land on the half-cent boundary that triggers it; real quotes are round numbers,
and round numbers are where the arithmetic breaks. So prices, order sizes and settings are drawn from
pools of ADVERSARIAL values: exact cap boundaries, one-cent-either-side, whole dollars, zeros.

Every scenario asserts invariants that must hold whatever the model proposed.
"""
from __future__ import annotations

import itertools
import math
import random
import sys
import traceback
from collections import Counter

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from app import sandbox_job as SJ  # noqa: E402

SYMS = ["AAPL", "MSFT", "VTI", "VOO", "SPY", "FBTC", "IBIT", "BRK-A", "BTC-USD", "SCHD"]

GROUPS = {
    "VTI": "US_EQUITY", "VOO": "US_EQUITY", "SPY": "US_EQUITY",
    "FBTC": "BTC", "IBIT": "BTC", "BTC-USD": "BTC",
}
def group_of(s: str) -> str:
    return GROUPS.get(s.upper(), s.upper())

# Prices chosen to sit ON boundaries, not near them.
PRICES = [0.01, 1.0, 4.99, 5.0, 34.44, 100.0, 100.005, 250.0, 346.39, 500.0,
          712_000.0, 0.0, None, 1e-9, 99_999.99]

def adversarial_price_map(rnd):
    return {s: rnd.choice(PRICES) for s in SYMS}

def make_blob(rnd):
    n = rnd.randint(0, 4)
    positions = []
    for s in rnd.sample(SYMS, n):
        sh = rnd.choice([0.0, 1.0, 3.0, 0.5, 1e-9, 1000.0])
        positions.append({
            "symbol": s, "shares": sh,
            "avg_cost": rnd.choice([0.0, 1.0, 100.0, 346.39]),
            "exposure_group": group_of(s),
            "opened_at": 1.0, "last_add_at": 1.0,
        })
    return {
        "cash": rnd.choice([0.0, 0.01, 100.0, 10_000.0, 1e7, -5.0]),
        "positions": positions,
        "realized_pl_total": 0.0,
        "funded_total": 10_000.0,
        "benchmark": {"symbol": "^GSPC", "shares": 1.0, "cost_basis": 1.0},
        "settings": {
            "max_position_pct": rnd.choice([0.0, 5.0, 25.0, 100.0]),
            "cash_floor_pct": rnd.choice([0.0, 5.0, 50.0, 100.0]),
            "slippage_bps": rnd.choice([0, 5, 200]),   # API clamps to 0..200; above that is unreachable
            "min_conviction_to_trade": rnd.choice([0, 55, 100]),
            "max_trades_per_tick": rnd.choice([0, 1, 4, 99]),
            "max_new_positions_per_tick": rnd.choice([0, 2, 99]),
            "max_turnover_pct": rnd.choice([0.0, 25.0, 100.0]),
            "respect_entry_zones": rnd.choice([True, False]),
            "allow_crypto": rnd.choice([True, False]),
            "allow_crypto_etf": True,
            "account_type": rnd.choice(["cash", "margin"]),
            "avoid_wash_sales": rnd.choice([True, False]),
            "preferred_btc_etf": "FBTC",
        },
        "unsettled": [],
        "recent_loss_sales": {},
    }

def make_orders(rnd):
    out = []
    for _ in range(rnd.randint(0, 5)):
        s = rnd.choice(SYMS)
        out.append({
            "symbol": s,
            "side": rnd.choice(["buy", "sell"]),
            "shares": rnd.choice([0.0, 1.0, 2.0, -1.0, 1e9, 0.5]),
            "dollars": rnd.choice([0.0, 100.0, 346.0, 1e9, -50.0]),
            "conviction": rnd.choice([0, 55, 60, 100]),
            "reason": "fuzz",
        })
    return out


def check(blob, orders, new_blob, filled, skipped, price_of):
    """Return a list of invariant-violation strings."""
    bad = []
    s = blob["settings"]

    for p in new_blob.get("positions", []):
        sh = float(p.get("shares") or 0)
        if sh < 0:
            bad.append(f"NEGATIVE SHARES {p['symbol']}={sh}")
        if math.isnan(sh) or math.isinf(sh):
            bad.append(f"NON-FINITE SHARES {p['symbol']}={sh}")
        ac = float(p.get("avg_cost") or 0)
        if math.isnan(ac) or math.isinf(ac):
            bad.append(f"NON-FINITE AVG_COST {p['symbol']}={ac}")
        if sh > 0 and ac < 0:
            bad.append(f"NEGATIVE COST BASIS {p['symbol']}={ac}")

    c = float(new_blob.get("cash", 0))
    if math.isnan(c) or math.isinf(c):
        bad.append(f"NON-FINITE CASH {c}")
    # Starting cash can be negative in the fuzz pool; only flag cash made MORE negative.
    if c < min(0.0, float(blob["cash"])) - 1e-6:
        bad.append(f"CASH DRIVEN NEGATIVE {blob['cash']} -> {c}")

    for r in filled:
        if float(r.get("shares") or 0) <= 0:
            bad.append(f"FILLED WITH NO SHARES {r.get('symbol')} {r.get('shares')}")
        px = r.get("price")
        if px is not None and (px <= 0 or math.isnan(px) or math.isinf(px)):
            bad.append(f"FILLED AT BAD PRICE {r.get('symbol')} {px}")
    for r in skipped:
        if not r.get("skip_reason"):
            bad.append(f"SKIPPED WITH NO REASON {r.get('symbol')}")

    # Nothing may vanish: every input order ends up filled or skipped. Counted in TOTAL rather than
    # per (symbol, side) -- prefer_btc_etf legitimately REWRITES an order's symbol, so a per-key
    # comparison reports a rewrite as a disappearance.
    if len(filled) + len(skipped) < len(orders):
        bad.append(f"ORDERS UNACCOUNTED in={len(orders)} out={len(filled)+len(skipped)}")
    return bad


def main(trials=4000, seed=7):
    rnd = random.Random(seed)
    findings = Counter()
    examples = {}
    aborts = 0
    crashes = Counter()
    crash_examples = {}

    for i in range(trials):
        blob = make_blob(rnd)
        orders = make_orders(rnd)
        pm = adversarial_price_map(rnd)
        price_of = lambda s: pm.get(s.upper())
        try:
            nb, filled, skipped = SJ.validate_and_fill(
                blob, orders, price_of, group_of=group_of, now_ts=1_786_000_000.0)
        except AssertionError:
            aborts += 1
            continue
        except Exception as e:  # noqa: BLE001
            key = f"{type(e).__name__}: {e}"
            crashes[key] += 1
            crash_examples.setdefault(key, (blob, orders, pm, traceback.format_exc()))
            continue
        for msg in check(blob, orders, nb, filled, skipped, price_of):
            head = msg.split(" ")[0] + " " + msg.split(" ")[1] if len(msg.split(" ")) > 1 else msg
            findings[head.strip()] += 1
            examples.setdefault(head.strip(), (blob, orders, pm, msg))

    print(f"trials={trials}  cash-conservation aborts={aborts}  "
          f"crashes={sum(crashes.values())}  invariant hits={sum(findings.values())}")
    print()
    if crashes:
        print("== UNCAUGHT EXCEPTIONS ==")
        for k, n in crashes.most_common():
            print(f"  [{n:5}] {k}")
    if findings:
        print("== INVARIANT VIOLATIONS ==")
        for k, n in findings.most_common():
            print(f"  [{n:5}] {k}")
    return findings, examples, crashes, crash_examples


if __name__ == "__main__":
    f, ex, cr, crex = main()
    if cr:
        k = cr.most_common(1)[0][0]
        print("\n--- first crash repro ---")
        print(crex[k][3][-1800:])
    if f:
        k = f.most_common(1)[0][0]
        blob, orders, pm, msg = ex[k]
        print("\n--- first invariant repro ---")
        print("msg    :", msg)
        print("cash   :", blob["cash"])
        print("pos    :", blob["positions"])
        print("orders :", orders)
        print("prices :", {s: pm[s] for s in {o['symbol'] for o in orders}})
        print("settings:", blob["settings"])
