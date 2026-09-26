"""Re-check that every fund_cost group really holds "the same thing" (FC-1).

For every PAIR inside each group in app/fund_cost.py: return correlation over two years of Yahoo
adjusted closes (dividends reinvested), and the two-year total-return gap with the fee gap taken out.
A pair passes at correlation >= 0.995 and a fee-adjusted gap of at most 3 percentage points.

Weekly returns, not daily, and four-weekly when either side is a mutual fund. A mutual fund prices
once a day at a NAV with fair-value adjustments, while an ETF trades all day, so two funds holding
identical stocks can look different day to day from timing alone: VXUS vs FTIHX correlates at 0.986
daily. That noise washes out over longer intervals and a real difference does not — FXNAX's tracking
error against AGG falls from 1.36 to 0.32 points a year going from daily to four-weekly returns,
while IEMG vs EEM stays near 1.2 at every interval. Fee-adjusted, because the fee IS the difference
being compared: GBTC trails IBIT by about 3 points over two years mostly because it charges 1.25
points a year more, and a raw gap test would throw out exactly the comparison worth making.

Run: `.venv/bin/python research/fund_peers.py` from the repo root (it imports app.fund_cost for the
groups; the measuring itself is stdlib). Exits 1 if any pair fails.
"""
from __future__ import annotations

import http.cookiejar
import itertools
import json
import math
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.fund_cost import GROUPS, MUTUAL_FUNDS  # noqa: E402

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
MIN_WEEKLY_CORR = 0.995
MAX_ADJ_GAP_PP = 3.0

jar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return opener.open(req, timeout=25).read()


def closes(sym: str) -> dict[str, float]:
    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        try:
            d = json.loads(_get(f"https://{host}/v8/finance/chart/{sym}?range=2y&interval=1d"))
            r = d["chart"]["result"][0]
            adj = r["indicators"].get("adjclose", [{}])[0].get("adjclose") or r["indicators"]["quote"][0]["close"]
            return {time.strftime("%Y-%m-%d", time.gmtime(t)): c for t, c in zip(r["timestamp"], adj) if c}
        except Exception:  # noqa: BLE001 — try the other host
            continue
    return {}


def fees(symbols: list[str]) -> dict[str, float | None]:
    try:
        opener.open(urllib.request.Request("https://fc.yahoo.com", headers={"User-Agent": UA}), timeout=15)
    except Exception:  # noqa: BLE001 — fc.yahoo.com answers 404 but still sets the cookie
        pass
    crumb = _get("https://query1.finance.yahoo.com/v1/test/getcrumb").decode()
    out: dict[str, float | None] = {}
    for i in range(0, len(symbols), 60):
        chunk = ",".join(symbols[i:i + 60])
        d = json.loads(_get(f"https://query2.finance.yahoo.com/v7/finance/quote?symbols={chunk}&crumb={crumb}"))
        for q in d["quoteResponse"]["result"]:
            out[q["symbol"]] = q.get("netExpenseRatio")
    return out


def corr(a: dict, b: dict, step: int) -> tuple[float | None, list[str]]:
    days = sorted(set(a) & set(b))[::step]
    ra = [a[days[i]] / a[days[i - 1]] - 1 for i in range(1, len(days))]
    rb = [b[days[i]] / b[days[i - 1]] - 1 for i in range(1, len(days))]
    n = len(ra)
    if n < 20:
        return None, days
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va, vb = sum((x - ma) ** 2 for x in ra), sum((y - mb) ** 2 for y in rb)
    return cov / math.sqrt(va * vb), days


def main() -> int:
    syms = sorted({m for g in GROUPS for m in g.members})
    with ThreadPoolExecutor(8) as ex:
        data = dict(zip(syms, ex.map(closes, syms)))
    fee = fees(syms)
    failed = 0
    for g in GROUPS:
        print(f"== {g.id}: {g.label}")
        for a, b in itertools.combinations(g.members, 2):
            if not data[a] or not data[b]:
                print(f"   {a:6} {b:6} NO DATA")
                failed += 1
                continue
            step = 20 if a in MUTUAL_FUNDS or b in MUTUAL_FUNDS else 5
            c, days = corr(data[a], data[b], step)
            if c is None:
                print(f"   {a:6} {b:6} too little shared history ({len(days)} points)")
                failed += 1
                continue
            first, last = days[0], days[-1]
            years = (time.mktime(time.strptime(last, "%Y-%m-%d")) -
                     time.mktime(time.strptime(first, "%Y-%m-%d"))) / (365.25 * 86400)
            gap = ((data[b][last] / data[b][first]) - (data[a][last] / data[a][first])) * 100
            fa, fb = fee.get(a), fee.get(b)
            # Unknown fee (SPYM on Yahoo): judge the raw gap, which is the stricter test.
            adj = gap - ((fa - fb) * years if fa is not None and fb is not None else 0.0)
            ok = c >= MIN_WEEKLY_CORR and abs(adj) <= MAX_ADJ_GAP_PP
            failed += 0 if ok else 1
            label = "4-weekly" if step == 20 else "weekly"
            print(f"   {a:6} {b:6} {label:8} {c:.4f}  gap {gap:+6.2f}pp  fee-adjusted {adj:+6.2f}pp"
                  f"  {'ok' if ok else 'FAIL'}")
    print(f"\n{failed} failing pair(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
