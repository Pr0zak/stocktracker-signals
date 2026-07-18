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

from .market import _adjusted_closes, _fetch_chart, _rsi

# Block-schedule history (fixed) + next estimate from ~10-minute average block time.
HALVINGS = [date(2012, 11, 28), date(2016, 7, 9), date(2020, 5, 11), date(2024, 4, 19)]
NEXT_HALVING_EST = date(2028, 4, 15)

_cache: dict[str, tuple[float, dict]] = {}
_TTL = 6 * 3600
_spy_cache: tuple[float, tuple[list[str], list[float]]] | None = None  # shared SPY weekly for touch studies


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


async def _weekly_max(client: httpx.AsyncClient, symbol: str) -> tuple[list[str], list[float], list[float]]:
    # Explicit 10y beats "max": Yahoo's max+1wk combo truncates to ~3y for some crypto symbols.
    data = await _fetch_chart(client, symbol, rng="10y", interval="1wk")
    result = data["chart"]["result"][0]
    ts = result.get("timestamp") or []
    adj = _adjusted_closes(result)  # splits/dividends adjusted — critical over a 10y 200WMA window
    vols_raw = (result.get("indicators", {}).get("quote") or [{}])[0].get("volume") or []
    dates, closes, vols = [], [], []
    for i in range(len(ts)):
        c = adj[i] if i < len(adj) else None
        if c is None:
            continue
        dates.append(time.strftime("%Y-%m-%d", time.gmtime(ts[i])))
        closes.append(float(c))
        v = vols_raw[i] if i < len(vols_raw) else None
        vols.append(float(v) if v is not None else 0.0)
    return dates, closes, vols


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


# mungbeans' 7-band "distance from the 200-week line" zone, cut on % from the WMA (negative = below).
_ZONES = ((-10, "extreme_value"), (-5, "deep_value"), (0, "below_line"),
          (5, "at_doorstep"), (10, "getting_close"), (15, "approaching"))


def _zone(pct_from_wma: float) -> str:
    for cut, name in _ZONES:
        if pct_from_wma <= cut:
            return name
    return "above"


def _volume_signal(closes: list[float], volumes: list[float]) -> dict | None:
    """mungbeans' weekly volume read: relative volume + an up/down-week accumulation ratio →
    quiet_accumulation / capitulation / breakout_volume / distribution / accumulation / neutral."""
    n = len(volumes)
    if n < 15 or len(closes) < 15:
        return None
    avg = sum(volumes[-14:]) / 14
    if avg <= 0:
        return None
    rvol = volumes[-1] / avg
    up_v: list[float] = []
    down_v: list[float] = []
    for i in range(n - 14, n):
        if i <= 0:
            continue
        (up_v if closes[i] >= closes[i - 1] else down_v).append(volumes[i])
    up_avg = sum(up_v) / len(up_v) if up_v else 0.0
    down_avg = sum(down_v) / len(down_v) if down_v else 0.0
    accum = up_avg / down_avg if down_avg > 0 else (2.0 if up_avg > 0 else 1.0)
    last_up = closes[-1] >= closes[-2]
    if rvol < 0.5 and accum > 1.2:
        sig = "quiet_accumulation"
    elif rvol > 2.0 and not last_up:
        sig = "capitulation"
    elif rvol > 2.0 and last_up:
        sig = "breakout_volume"
    elif accum < 0.7:
        sig = "distribution"
    elif accum > 1.5:
        sig = "accumulation"
    else:
        sig = "neutral"
    return {"volume_signal": sig, "rvol_14": round(rvol, 2), "accumulation_ratio": round(accum, 2)}


def long_term_trend(
    weekly: list[float], daily_closes: list[float], weekly_volumes: list[float] | None = None,
) -> dict | None:
    """Multi-year trend block for ANY symbol (stocks + crypto), computed from max-range weekly bars.

    200-week SMA and where price sits relative to it — the below-the-line zone and week-over-week
    direction (mungbeans' recovering/deepening) — plus a 14-week RSI oversold read, Mayer Multiple,
    distance from the all-time high, and 3-year CAGR. Returns None when there isn't enough weekly
    history to say anything; the 200-week fields are omitted (not zero) below ~4 years of data.
    """
    if len(weekly) < 30:
        return None
    price = weekly[-1]
    lt: dict = {"history_years": round(len(weekly) / 52, 1)}
    if len(weekly) >= 200:
        sma200w = sum(weekly[-200:]) / 200
        pct = (price - sma200w) / sma200w * 100
        lt["sma_200w"] = round(sma200w, 2)
        lt["price_vs_200w_sma_pct"] = round(pct, 1)
        lt["below_line"] = price < sma200w
        lt["zone"] = _zone(pct)
        if len(weekly) >= 201:  # week-over-week direction of the distance to the line
            prev_sma = sum(weekly[-201:-1]) / 200
            prev_pct = (weekly[-2] - prev_sma) / prev_sma * 100
            wow = pct - prev_pct
            lt["price_vs_200w_wow_pp"] = round(wow, 2)
            lt["direction"] = (
                ("recovering" if wow >= 0 else "deepening") if price < sma200w
                else ("moving_away" if wow >= 0 else "approaching")
            )
        rsi_w = _rsi(weekly, 14)  # 14-WEEK RSI (distinct from the 14-day daily RSI in the snapshot)
        if rsi_w is not None:
            lt["rsi_14w"] = round(rsi_w, 1)
            lt["weekly_oversold"] = rsi_w < 30
    ath = max(weekly)
    lt["pct_off_all_time_high"] = round((price - ath) / ath * 100, 1)
    if len(weekly) >= 156:  # 3-year CAGR
        p0 = weekly[-156]
        lt["cagr_3y_pct"] = round(((price / p0) ** (1 / 3) - 1) * 100, 1)
    if len(daily_closes) >= 200:  # Mayer Multiple from the daily series we already have
        lt["mayer_multiple"] = round(daily_closes[-1] / (sum(daily_closes[-200:]) / 200), 2)
    if weekly_volumes is not None:  # weekly accumulation/distribution read
        vsig = _volume_signal(weekly, weekly_volumes)
        if vsig:
            lt.update(vsig)
    return lt


def _rolling_sma(values: list[float], window: int = 200, min_periods: int = 50) -> list[float | None]:
    """Trailing-`window` SMA at each index, emitting once `min_periods` bars exist (mungbeans'
    min_periods leniency, so an episode can be dated before a full 200-week window has accrued)."""
    out: list[float | None] = [None] * len(values)
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        seg = values[lo:i + 1]
        if len(seg) >= min_periods:
            out[i] = sum(seg) / len(seg)
    return out


def _touch_episode_starts(weekly: list[float], sma: list[float | None], recovery_weeks: int = 2) -> list[int]:
    """Index of the first week below the 200-week line for each distinct episode. Hysteresis: an
    episode ends only after `recovery_weeks` consecutive weeks back above the line (so a one-week
    poke above doesn't split one dip into two)."""
    starts: list[int] = []
    in_ep = False
    above_run = 0
    for i, s in enumerate(sma):
        if s is None:
            continue
        below = weekly[i] < s
        if not in_ep:
            if below:
                in_ep, above_run = True, 0
                starts.append(i)
        elif below:
            above_run = 0
        else:
            above_run += 1
            if above_run >= recovery_weeks:
                in_ep = False
    return starts


def wma_touch_study(
    dates: list[str], weekly: list[float], spy_dates: list[str], spy_weekly: list[float],
    recovery_weeks: int = 2,
) -> dict | None:
    """"What happened the last N times this name was below its 200-week line" — the forward
    12-/24-month return distribution from each episode's start, vs the S&P 500 over the same windows.

    Evidence context, not a signal (small, self-selected sample). Return is measured from the episode
    START (first week below the line). Returns None with under ~4 years of weekly history; touch_count
    can be 0 for a name that has 200-week history but never dipped below in the sample.
    """
    if len(weekly) < 200:
        return None
    sma = _rolling_sma(weekly)
    starts = _touch_episode_starts(weekly, sma, recovery_weeks)

    def _spy_fwd(date_str: str, weeks: int) -> float | None:
        j = next((k for k, d in enumerate(spy_dates) if d >= date_str), None)  # nearest SPY bar at/after
        if j is None or j + weeks >= len(spy_weekly) or spy_weekly[j] == 0:
            return None
        return (spy_weekly[j + weeks] - spy_weekly[j]) / spy_weekly[j] * 100

    f12: list[float] = []
    f24: list[float] = []
    s12: list[float] = []
    s24: list[float] = []
    for s in starts:
        base = weekly[s]
        if base <= 0:
            continue
        if s + 52 < len(weekly):
            f12.append((weekly[s + 52] - base) / base * 100)
            sr = _spy_fwd(dates[s], 52)
            if sr is not None:
                s12.append(sr)
        if s + 104 < len(weekly):
            f24.append((weekly[s + 104] - base) / base * 100)
            sr = _spy_fwd(dates[s], 104)
            if sr is not None:
                s24.append(sr)

    def _med(xs: list[float]) -> float | None:
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else None

    def _pct_pos(xs: list[float]) -> int | None:
        return round(sum(x > 0 for x in xs) / len(xs) * 100) if xs else None

    last_sma = sma[-1]
    out: dict = {
        "touch_count": len(starts),
        "measured_12m": len(f12),  # episodes old enough to have a full 12-month forward window
        "currently_below": (weekly[-1] < last_sma) if last_sma is not None else None,
    }
    if f12:
        out["median_fwd_12m_pct"] = round(_med(f12), 1)
        out["avg_fwd_12m_pct"] = round(sum(f12) / len(f12), 1)
        out["pct_positive_12m"] = _pct_pos(f12)
        if s12:
            out["spy_avg_fwd_12m_pct"] = round(sum(s12) / len(s12), 1)
    if f24:
        out["median_fwd_24m_pct"] = round(_med(f24), 1)
        out["pct_positive_24m"] = _pct_pos(f24)
        if s24:
            out["spy_avg_fwd_24m_pct"] = round(sum(s24) / len(s24), 1)
    return out


async def spy_weekly(client: httpx.AsyncClient) -> tuple[list[str], list[float]]:
    """S&P 500 weekly bars for touch-study benchmarking, cached 6h and shared across symbols."""
    global _spy_cache
    if _spy_cache and time.time() - _spy_cache[0] < _TTL:
        return _spy_cache[1]
    dates, closes, _ = await _weekly_max(client, "^GSPC")
    _spy_cache = (time.time(), (dates, closes))
    return _spy_cache[1]


async def crypto_context(client: httpx.AsyncClient, symbol: str, daily_closes: list[float]) -> dict:
    """Long-term context for ANY symbol: the trend block always; the halving cycle only for BTC-*.
    Cached in-process 6h per symbol. (Kept the crypto_context name for its existing callers; the
    trend block itself is asset-agnostic and also powers the equity /trend endpoint.)"""
    sym = symbol.upper()
    hit = _cache.get(sym)
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]
    out: dict = {}
    try:
        dates, weekly, weekly_vol = await _weekly_max(client, sym)
        lt = long_term_trend(weekly, daily_closes, weekly_vol)
        if lt:
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
