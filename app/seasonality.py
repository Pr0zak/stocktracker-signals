"""
Seasonality (AIE-6): typical per-calendar-month price action from multi-year monthly history.

For each month (Jan..Dec) we compute the average month-over-month return, the hit rate (% of years that
month was up), and best/worst, plus the CURRENT month's tendency and the strongest/weakest months. It's
WEAK, sample-limited context (a handful of years per month), framed as such for the analyst — a modest
tilt, never a timing signal on its own.
"""
from __future__ import annotations

import datetime as dt
import statistics

import httpx

from .market import _adjusted_closes, _fetch_chart

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


async def seasonality(client: httpx.AsyncClient, symbol: str) -> dict | None:
    """Per-calendar-month seasonal return stats from ~10y of monthly bars. None if too little history."""
    try:
        data = await _fetch_chart(client, symbol, rng="10y", interval="1mo")
        result = data["chart"]["result"][0]
    except Exception:  # noqa: BLE001 — seasonality is enrichment, never a blocker
        return None
    ts = result.get("timestamp") or []
    closes = _adjusted_closes(result)
    bars: list[tuple[int, int, float]] = []   # (year, month, close), null bars dropped
    for i in range(len(ts)):
        c = closes[i] if i < len(closes) else None
        if c is None:
            continue
        d = dt.datetime.fromtimestamp(ts[i], dt.timezone.utc)
        bars.append((d.year, d.month, float(c)))
    if len(bars) < 24:   # need ~2 years of months to say anything
        return None

    # Month-over-month return, attributed to the later month (the return realized DURING that month).
    returns_by_month: dict[int, list[float]] = {m: [] for m in range(1, 13)}
    for i in range(1, len(bars)):
        _, _, c0 = bars[i - 1]
        _, m1, c1 = bars[i]
        if c0 > 0:
            returns_by_month[m1].append((c1 / c0 - 1.0) * 100.0)

    months: list[dict] = []
    for m in range(1, 13):
        r = returns_by_month[m]
        if not r:
            months.append({"month": m, "name": _MONTHS[m - 1], "n": 0,
                           "avg_pct": None, "hit_rate": None, "best_pct": None, "worst_pct": None})
            continue
        months.append({
            "month": m, "name": _MONTHS[m - 1], "n": len(r),
            "avg_pct": round(statistics.mean(r), 2),
            "hit_rate": round(100.0 * sum(1 for x in r if x > 0) / len(r)),
            "best_pct": round(max(r), 2), "worst_pct": round(min(r), 2),
        })

    valid = [mo for mo in months if mo["n"] >= 3 and mo["avg_pct"] is not None]
    if not valid:
        return None
    span_years = bars[-1][0] - bars[0][0] + 1
    cur_month = dt.date.today().month
    current = next((mo for mo in months if mo["month"] == cur_month and mo["avg_pct"] is not None), None)
    best = max(valid, key=lambda mo: mo["avg_pct"])
    worst = min(valid, key=lambda mo: mo["avg_pct"])
    return {
        "years": span_years,
        "sample_note": f"~{span_years} years of monthly data — small per-month sample, treat as a weak tilt",
        "months": months,
        "current_month": current,
        "best_month": {"name": best["name"], "avg_pct": best["avg_pct"], "hit_rate": best["hit_rate"]},
        "worst_month": {"name": worst["name"], "avg_pct": worst["avg_pct"], "hit_rate": worst["hit_rate"]},
    }


def compact(data: dict | None) -> dict | None:
    """Slim block for the analyst snapshot — the current-month tendency + best/worst months."""
    if not data:
        return None
    cur = data.get("current_month") or {}
    if not cur:
        return None
    return {
        "current_month": {"name": cur.get("name"), "avg_pct": cur.get("avg_pct"),
                          "hit_rate": cur.get("hit_rate"), "n": cur.get("n")},
        "best_month": data.get("best_month"),
        "worst_month": data.get("worst_month"),
        "years": data.get("years"),
    }
