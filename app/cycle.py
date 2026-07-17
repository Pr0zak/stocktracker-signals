"""
Crypto long-term context: Bitcoin halving cycle position + multi-year trend metrics.

Halvings happen every 210,000 blocks (~4 years); past dates are fixed history and the next is
estimable from average block time. The classic cycle pattern (run-up peaking ~12-18 months after a
halving) has only FOUR observations ever — so everything here is fed to the analyst as weak-sample
context with the sample size attached, never as a law.

Long-term trend metrics come from Yahoo's max-range weekly bars (BTC back to 2014): price vs the
200-week SMA (the classic BTC long-cycle support), Mayer Multiple (price / 200-day SMA), distance
from all-time high, and a past-cycle analog (what happened 12 months after this same cycle position
in each prior cycle). Cached in-process for 6h per symbol.
"""
from __future__ import annotations

import time
from datetime import date, datetime

import httpx

from .market import _fetch_chart

# Block-schedule history (fixed) + next estimate from ~10-minute average block time.
HALVINGS = [date(2012, 11, 28), date(2016, 7, 9), date(2020, 5, 11), date(2024, 4, 19)]
NEXT_HALVING_EST = date(2028, 4, 15)

_cache: dict[str, tuple[float, dict]] = {}
_TTL = 6 * 3600


def halving_cycle(today: date | None = None) -> dict:
    """Where we are in the current halving cycle (BTC and BTC-tracking assets)."""
    today = today or date.today()
    last = max(h for h in HALVINGS if h <= today)
    span = (NEXT_HALVING_EST - last).days
    since = (today - last).days
    pct = round(since / span * 100, 1) if span > 0 else None
    phase = (
        "post-halving year (historically the strong phase)" if since <= 550
        else "mid-cycle" if since <= 1000
        else "late-cycle / pre-halving (historically the weak phase)"
    )
    return {
        "last_halving": last.isoformat(),
        "next_halving_est": NEXT_HALVING_EST.isoformat(),
        "days_since_halving": since,
        "days_to_next_est": (NEXT_HALVING_EST - today).days,
        "cycle_pct": pct,
        "phase": phase,
        "sample_size_note": "only 4 halvings have ever occurred — weak-sample context, not a law",
    }


async def _weekly_max(client: httpx.AsyncClient, symbol: str) -> tuple[list[str], list[float]]:
    data = await _fetch_chart(client, symbol, rng="max", interval="1wk")
    result = data["chart"]["result"][0]
    ts = result.get("timestamp") or []
    closes_raw = (result.get("indicators", {}).get("quote") or [{}])[0].get("close") or []
    dates, closes = [], []
    for i in range(len(ts)):
        c = closes_raw[i] if i < len(closes_raw) else None
        if c is None:
            continue
        dates.append(time.strftime("%Y-%m-%d", time.gmtime(ts[i])))
        closes.append(float(c))
    return dates, closes


def _cycle_analog(dates: list[str], closes: list[float], cycle_pct: float | None) -> dict | None:
    """Median 12-month-forward return from this same position in each PRIOR completed cycle."""
    if cycle_pct is None or len(closes) < 200:
        return None
    idx = {d: i for i, d in enumerate(dates)}
    fwd: list[float] = []
    for a, b in zip(HALVINGS, HALVINGS[1:]):
        span = (b - a).days
        target = date.fromordinal(a.toordinal() + int(span * cycle_pct / 100))
        # nearest weekly bar at/after the target date
        i = next((j for j, d in enumerate(dates) if d >= target.isoformat()), None)
        if i is None or i + 52 >= len(closes):
            continue
        fwd.append((closes[i + 52] - closes[i]) / closes[i] * 100)
    if not fwd:
        return None
    fwd.sort()
    return {
        "prior_cycles_measured": len(fwd),
        "median_fwd_12mo_pct": round(fwd[len(fwd) // 2], 1),
        "worst_fwd_12mo_pct": round(fwd[0], 1),
        "best_fwd_12mo_pct": round(fwd[-1], 1),
    }


async def crypto_context(client: httpx.AsyncClient, symbol: str, daily_closes: list[float]) -> dict:
    """The blocks merged into a crypto snapshot: long-term trend always; halving cycle for BTC-*."""
    sym = symbol.upper()
    hit = _cache.get(sym)
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]
    out: dict = {}
    try:
        dates, weekly = await _weekly_max(client, sym)
        if len(weekly) >= 30:
            price = weekly[-1]
            lt: dict = {"history_years": round(len(weekly) / 52, 1)}
            if len(weekly) >= 200:
                sma200w = sum(weekly[-200:]) / 200
                lt["price_vs_200w_sma_pct"] = round((price - sma200w) / sma200w * 100, 1)
            ath = max(weekly)
            lt["pct_off_all_time_high"] = round((price - ath) / ath * 100, 1)
            if len(weekly) >= 156:  # 3-year CAGR
                p0 = weekly[-156]
                lt["cagr_3y_pct"] = round(((price / p0) ** (1 / 3) - 1) * 100, 1)
            if len(daily_closes) >= 200:  # Mayer Multiple from the daily series we already have
                lt["mayer_multiple"] = round(daily_closes[-1] / (sum(daily_closes[-200:]) / 200), 2)
            out["long_term_trend"] = lt
            if sym.startswith("BTC"):
                cyc = halving_cycle()
                analog = _cycle_analog(dates, weekly, cyc.get("cycle_pct"))
                if analog:
                    cyc["past_cycle_analog"] = analog
                out["btc_halving_cycle"] = cyc
    except Exception:  # noqa: BLE001 — long-term context is enrichment, never a blocker
        return out
    _cache[sym] = (time.time(), out)
    return out
