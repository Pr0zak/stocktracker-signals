"""Re-measure the ETF arms' cap groups (app/etf_arm.py ETF_GROUPS).

Two years of Yahoo daily adjusted closes for every ETF in the fund catalogue, daily log returns on
shared dates, and complete-linkage clustering at 0.90: two funds share a group only if every pair in
the group correlates at 0.90 or more. Complete linkage, not single: single linkage chains funds that
are each close to a neighbour but not to each other (SPMO into the S&P via QQQ), which is the error
fund_overlap's "move together" sets were built to avoid.

T-bill funds are skipped: their prices barely move, so a daily correlation between them measures
rounding (SGOV-BIL ~0.64) rather than sameness. ETF_GROUPS groups them by what they hold.

Prints each measured group with its lowest pairwise correlation, then every difference from
ETF_GROUPS. Exits 1 on a difference, so a re-run after the market has moved says plainly whether
the map still matches. A difference is a prompt to look, not an automatic edit: regrouping changes
which targets a standing plan can reach (see sandbox_job.allocation_gap).

Run: `.venv/bin/python research/etf_groups.py` from the repo root. Stdlib only for the measuring;
it imports app.etf_arm and app.fund_catalog for the lists and research/fund_peers.py for the fetch.
"""
from __future__ import annotations

import itertools
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.etf_arm import ETF_GROUPS, universe  # noqa: E402
from fund_peers import closes  # noqa: E402

BAR = 0.90
MIN_SHARED_DAYS = 60
SKIP = {"SGOV", "BIL"}


def returns(c: dict[str, float]) -> dict[str, float]:
    ks = sorted(c)
    return {ks[i]: math.log(c[ks[i]] / c[ks[i - 1]]) for i in range(1, len(ks)) if c[ks[i - 1]] > 0}


def corr(ra: dict[str, float], rb: dict[str, float]) -> float | None:
    ks = sorted(set(ra) & set(rb))
    if len(ks) < MIN_SHARED_DAYS:
        return None
    x = [ra[k] for k in ks]
    y = [rb[k] for k in ks]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((i - mx) * (j - my) for i, j in zip(x, y))
    sxx = sum((i - mx) ** 2 for i in x)
    syy = sum((j - my) ** 2 for j in y)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else None


def main() -> int:
    syms = [s for s in universe() if s not in SKIP]
    with ThreadPoolExecutor(8) as ex:
        data = dict(zip(syms, ex.map(closes, syms)))
    missing = [s for s, d in data.items() if not d]
    if missing:
        print("no history:", " ".join(missing))
    syms = [s for s in syms if data[s]]
    rets = {s: returns(data[s]) for s in syms}
    c: dict[tuple[str, str], float | None] = {}
    for a, b in itertools.combinations(syms, 2):
        c[(a, b)] = c[(b, a)] = corr(rets[a], rets[b])

    def link(x: list[str], y: list[str]) -> float:
        vals = [c[(a, b)] for a in x for b in y]
        return -1.0 if any(v is None for v in vals) else min(vals)

    clusters = [[s] for s in syms]
    while True:
        best, bi, bj = -2.0, -1, -1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                v = link(clusters[i], clusters[j])
                if v > best:
                    best, bi, bj = v, i, j
        if best < BAR:
            break
        clusters[bi] += clusters[bj]
        del clusters[bj]

    measured = {frozenset(k) for k in clusters if len(k) > 1}
    for k in sorted(measured, key=len, reverse=True):
        low = min(c[(a, b)] for a, b in itertools.combinations(sorted(k), 2))
        print(f"[{len(k)}] {low:.3f}  {' '.join(sorted(k))}")

    mapped = {frozenset(m for m in members if m not in SKIP and m in syms)
              for members in ETF_GROUPS.values()}
    mapped = {k for k in mapped if len(k) > 1}
    diff = False
    for k in sorted(measured - mapped, key=len, reverse=True):
        print("measured, not in ETF_GROUPS:", " ".join(sorted(k)))
        diff = True
    for k in sorted(mapped - measured, key=len, reverse=True):
        print("in ETF_GROUPS, not measured:", " ".join(sorted(k)))
        diff = True
    print("matches ETF_GROUPS" if not diff else "DIFFERS from ETF_GROUPS")
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main())
