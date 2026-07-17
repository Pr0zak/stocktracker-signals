"""
Market data + a compact technical summary for the analyst.

Fetches ~1 year of daily bars from Yahoo's public chart endpoint (query1→query2 failover, same
approach as the Android app) and derives the latest reading of the indicators the app's Tier-1
engine uses. This mirrors ChartMath.kt so the LLM sees the same numbers the phone does.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import httpx

_UA = "Mozilla/5.0 (Linux; Android 14; Mobile) StockTracker-Signals/1.0"
_HOSTS = ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]


@dataclass
class Series:
    symbol: str
    closes: list[float]
    volumes: list[float | None]
    fifty_two_high: float | None
    fifty_two_low: float | None
    currency: str


async def _fetch_chart(client: httpx.AsyncClient, symbol: str, rng: str = "1y", interval: str = "1d") -> dict:
    enc = symbol.upper().replace("^", "%5E")
    path = f"v8/finance/chart/{enc}?range={rng}&interval={interval}"
    last_err: Exception | None = None
    for host in _HOSTS:
        try:
            r = await client.get(f"https://{host}/{path}", headers={"User-Agent": _UA}, timeout=15)
            r.raise_for_status()
            data = r.json()
            if data.get("chart", {}).get("error"):
                raise ValueError(data["chart"]["error"])
            return data
        except Exception as e:  # noqa: BLE001 — fail over to the next host
            last_err = e
    raise RuntimeError(f"Yahoo chart fetch failed for {symbol}: {last_err}")


async def fetch_series(client: httpx.AsyncClient, symbol: str) -> Series:
    data = await _fetch_chart(client, symbol)
    result = data["chart"]["result"][0]
    meta = result.get("meta", {})
    ts = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    raw_closes = quote.get("close") or []
    raw_vols = quote.get("volume") or []
    closes: list[float] = []
    vols: list[float | None] = []
    for i in range(len(ts)):
        c = raw_closes[i] if i < len(raw_closes) else None
        if c is None:  # Yahoo pads gaps with null
            continue
        closes.append(float(c))
        v = raw_vols[i] if i < len(raw_vols) else None
        vols.append(float(v) if v is not None else None)
    return Series(
        symbol=symbol.upper(),
        closes=closes,
        volumes=vols,
        fifty_two_high=meta.get("fiftyTwoWeekHigh"),
        fifty_two_low=meta.get("fiftyTwoWeekLow"),
        currency=meta.get("currency", "USD"),
    )


# --- causal indicators (latest value only) — ports of ChartMath.kt ---

def _sma(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def _ema_series(values: list[float], period: int) -> list[float | None]:
    if len(values) < period:
        return [None] * len(values)
    out: list[float | None] = [None] * len(values)
    k = 2.0 / (period + 1)
    ema = sum(values[:period]) / period
    out[period - 1] = ema
    for i in range(period, len(values)):
        ema = values[i] * k + ema * (1 - k)
        out[i] = ema
    return out


def _rsi(values: list[float], period: int = 14) -> float | None:
    if len(values) <= period:
        return None
    gain = loss = 0.0
    for i in range(1, period + 1):
        ch = values[i] - values[i - 1]
        gain += max(ch, 0.0)
        loss += max(-ch, 0.0)
    avg_gain, avg_loss = gain / period, loss / period
    for i in range(period + 1, len(values)):
        ch = values[i] - values[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(ch, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-ch, 0.0)) / period
    if avg_loss == 0.0:
        return 50.0 if avg_gain == 0.0 else 100.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def _macd(values: list[float]) -> tuple[float | None, float | None, float | None]:
    fast, slow = _ema_series(values, 12), _ema_series(values, 26)
    line = [(f - s) if (f is not None and s is not None) else None for f, s in zip(fast, slow)]
    tail = [x for x in line if x is not None]
    if len(tail) < 9:
        return (line[-1] if line else None, None, None)
    sig = _ema_series(tail, 9)
    macd_now, sig_now = tail[-1], sig[-1]
    hist = macd_now - sig_now if (macd_now is not None and sig_now is not None) else None
    return macd_now, sig_now, hist


def _bollinger_pct_b(values: list[float], period: int = 20) -> float | None:
    if len(values) < period:
        return None
    window = values[-period:]
    mid = sum(window) / period
    sd = math.sqrt(sum((x - mid) ** 2 for x in window) / period)
    if sd == 0.0:
        return None
    upper, lower = mid + 2 * sd, mid - 2 * sd
    return (values[-1] - lower) / (upper - lower)


def _stochastic_k(values: list[float], period: int = 14) -> float | None:
    if len(values) < period:
        return None
    window = values[-period:]
    lo, hi = min(window), max(window)
    return 100.0 * (values[-1] - lo) / (hi - lo) if hi > lo else 50.0


def relative_strength(closes: list[float], bench: list[float] | None, period: int = 63) -> float | None:
    """3-month price/benchmark ratio slope — the momentum proxy, aligned tail-to-tail."""
    if not bench:
        return None
    n = min(len(closes), len(bench))
    if n <= period:
        return None
    c, b = closes[-n:], bench[-n:]
    if b[-1] == 0 or b[-1 - period] == 0 or c[-1 - period] == 0:
        return None
    ratio_now = c[-1] / b[-1]
    ratio_prev = c[-1 - period] / b[-1 - period]
    return ratio_now / ratio_prev - 1.0 if ratio_prev else None


def summarize(series: Series, bench_closes: list[float] | None) -> dict:
    """The compact, LLM-facing snapshot of where this name sits technically."""
    c = series.closes
    price = c[-1]
    sma20, sma50 = _sma(c, 20), _sma(c, 50)
    macd_line, macd_sig, macd_hist = _macd(c)
    rs = relative_strength(c, bench_closes)
    return {
        "symbol": series.symbol,
        "currency": series.currency,
        "price": round(price, 4),
        "rsi14": _round(_rsi(c)),
        "macd_line": _round(macd_line),
        "macd_signal": _round(macd_sig),
        "macd_hist": _round(macd_hist),
        "sma20": _round(sma20),
        "sma50": _round(sma50),
        "pct_vs_sma20": _pct(price, sma20),
        "pct_vs_sma50": _pct(price, sma50),
        "golden_cross": (sma20 > sma50) if (sma20 and sma50) else None,
        "bollinger_pct_b": _round(_bollinger_pct_b(c)),
        "stochastic_k": _round(_stochastic_k(c)),
        "fifty_two_week_high": series.fifty_two_high,
        "fifty_two_week_low": series.fifty_two_low,
        "pct_off_52w_high": _pct(price, series.fifty_two_high),
        "rel_strength_3mo_vs_benchmark": _round(rs, 4),
    }


def _round(v: float | None, ndigits: int = 2) -> float | None:
    return round(v, ndigits) if v is not None else None


def _pct(a: float, b: float | None) -> float | None:
    return round((a / b - 1.0) * 100.0, 2) if (b not in (None, 0)) else None
