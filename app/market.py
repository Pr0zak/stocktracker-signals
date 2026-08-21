"""
Market data + a compact technical summary for the analyst.

Fetches ~1 year of daily bars from Yahoo's public chart endpoint (query1→query2 failover, same
approach as the Android app) and derives the latest reading of the indicators the app's Tier-1
engine uses. This mirrors ChartMath.kt so the LLM sees the same numbers the phone does.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import httpx

_UA = "Mozilla/5.0 (Linux; Android 14; Mobile) StockTracker-Signals/1.0"
_HOSTS = ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]


@dataclass
class Series:
    symbol: str
    closes: list[float]
    opens: list[float | None]   # session opens — needed for overnight-gap detection (see gaps.py)
    volumes: list[float | None]
    dates: list[str]  # YYYYMMDD per bar — lets shorts.py align SEC/FINRA data to prices
    fifty_two_high: float | None
    fifty_two_low: float | None
    currency: str
    source: str = "yahoo"  # "webull" when Yahoo had no data and the fallback supplied bars
    # Per-bar high/low, index-aligned with `closes` and SCALED ONTO THE ADJUSTED BASIS (see the
    # rescaling comment in fetch_series). None for a bar means the venue omitted it — never 0.0.
    # These are defaulted and last on purpose: scan_job.py's memory backfill builds a truncated
    # Series that predates them, so they must not become required positional fields.
    highs: list[float | None] = field(default_factory=list)
    lows: list[float | None] = field(default_factory=list)


async def _fetch_chart(client: httpx.AsyncClient, symbol: str, rng: str = "1y", interval: str = "1d") -> dict:
    enc = symbol.upper().replace("^", "%5E")
    path = f"v8/finance/chart/{enc}?range={rng}&interval={interval}&includeAdjustedClose=true"
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


def _adjusted_closes(result: dict) -> list[float | None]:
    """Split/dividend-adjusted close per bar, falling back to raw close where Yahoo omits adjclose.

    Adjusted close is the correct series for any moving-average / RSI / return math: without it a
    split (e.g. NVDA 10:1) drops the raw close ~90% in one bar and corrupts the 200-week SMA, the
    forward-return studies, and below-line / cross detection. Indices and crypto carry no adjclose
    array (no splits/dividends) — for them raw close already equals adjusted, so we fall back per bar.
    """
    ind = result.get("indicators", {})
    raw = (ind.get("quote") or [{}])[0].get("close") or []
    adj_holder = ind.get("adjclose") or []
    adj = (adj_holder[0].get("adjclose") if adj_holder else None) or []
    n = len(result.get("timestamp") or [])
    out: list[float | None] = []
    for i in range(n):
        a = adj[i] if i < len(adj) else None
        if a is None:  # bar missing from adjclose (or no adjclose at all) — use raw close
            a = raw[i] if i < len(raw) else None
        out.append(float(a) if a is not None else None)
    return out


async def _webull_series(client: httpx.AsyncClient, symbol: str) -> Series:
    """Build a Series from Webull daily bars — the fallback for warrants/OTC Yahoo doesn't carry.
    Raises if Webull has nothing either, so callers see a normal 'no data' failure."""
    from . import webull  # local import: webull has no market dependency, keeps import order clean
    bars = await webull.history(client, symbol)
    if not bars:
        raise RuntimeError(f"no data for {symbol} (Yahoo + Webull)")
    closes = [b["c"] for b in bars]
    opens: list[float | None] = [b.get("o") for b in bars]
    vols: list[float | None] = [b.get("v") for b in bars]
    # No adjusted-basis rescaling here, unlike the Yahoo path: Webull hands back one raw OHLC set,
    # so `closes` are already raw too and high/low/close share a single basis. (The cost is that a
    # split inside a Webull-only symbol's history distorts the closes themselves — that is a
    # pre-existing property of this fallback, not something high/low make worse.)
    highs: list[float | None] = [b.get("h") for b in bars]
    lows: list[float | None] = [b.get("l") for b in bars]
    dates = [time.strftime("%Y%m%d", time.gmtime(b["t"] / 1000)) for b in bars]
    recent = closes[-252:]
    return Series(
        symbol=symbol.upper(), closes=closes, opens=opens, volumes=vols, dates=dates,
        fifty_two_high=max(recent), fifty_two_low=min(recent), currency="USD", source="webull",
        highs=highs, lows=lows,
    )


async def fetch_series(client: httpx.AsyncClient, symbol: str, rng: str = "1y", *, fallback: bool = True) -> Series:
    """Daily bars for `symbol`. `rng` is a Yahoo range string ("1y", "2y", "5y") — the memory
    backfill wants more history than a signal does, everything else takes the default.

    `fallback=False` makes a Yahoo failure raise instead of reaching for Webull. That switch exists
    for the bulk swing scan: across 3,000+ symbols even a 2% Yahoo failure rate is ~60 names, and
    each one would burn up to 30s of Yahoo timeouts (two hosts x 15s) and then ~27s more of Webull
    search+chart against an unofficial, reverse-engineered endpoint — half an hour of wall clock and
    a rate-limit magnet, to rescue a handful of warrants a swing scan does not want anyway. It
    defaults True so every interactive caller keeps the rescue it has today.
    """
    try:
        data = await _fetch_chart(client, symbol, rng=rng)
    except Exception:  # noqa: BLE001 — Yahoo has nothing; try the Webull fallback (warrants/OTC)
        if not fallback:
            raise
        return await _webull_series(client, symbol)
    result = data["chart"]["result"][0]
    meta = result.get("meta", {})
    ts = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    adj_closes = _adjusted_closes(result)
    raw_vols = quote.get("volume") or []
    raw_opens = quote.get("open") or []
    raw_highs = quote.get("high") or []
    raw_lows = quote.get("low") or []
    raw_closes = quote.get("close") or []
    closes: list[float] = []
    opens: list[float | None] = []
    vols: list[float | None] = []
    highs: list[float | None] = []
    lows: list[float | None] = []
    dates: list[str] = []
    for i in range(len(ts)):
        c = adj_closes[i] if i < len(adj_closes) else None
        if c is None:  # Yahoo pads gaps with null
            continue
        closes.append(float(c))
        o = raw_opens[i] if i < len(raw_opens) else None
        # `opens` is knowingly left RAW while highs/lows below are rescaled. gaps.py measures
        # opens[-1] against the ADJUSTED closes[-2], so the split bar of a 10:1 reads as a ~90%
        # overnight gap. That is a real latent bug, but correcting it changes the output of a
        # shipped feature, so it gets its own commit rather than riding along with this one.
        opens.append(float(o) if o is not None else None)
        v = raw_vols[i] if i < len(raw_vols) else None
        vols.append(float(v) if v is not None else None)
        # Yahoo's quote arrays (open/high/low/close) are RAW prices, but `closes` above are the
        # split/dividend-ADJUSTED series from _adjusted_closes. Parking a raw high next to an
        # adjusted close silently corrupts every range-derived metric across a split: after NVDA's
        # 10:1 the pre-split bars would report (high-low)/close about 10x too wide, so ADR20 would
        # flag a placid mega-cap as a volatile mover and CLV would read off a bar whose high sits
        # 10x above its own close. Rescale each bar by its OWN adjusted/raw close ratio, which is
        # exactly the cumulative split+dividend factor Yahoo already applied to that bar's close;
        # the ratio is 1.0 for every bar after the last corporate action. Missing or zero raw close
        # (Yahoo nulls a bar it has adjclose for) leaves the bar unscaled rather than dropping it.
        rc = raw_closes[i] if i < len(raw_closes) else None
        f = (float(c) / float(rc)) if rc else 1.0
        h = raw_highs[i] if i < len(raw_highs) else None
        highs.append(float(h) * f if h is not None else None)
        lo = raw_lows[i] if i < len(raw_lows) else None
        lows.append(float(lo) * f if lo is not None else None)
        dates.append(time.strftime("%Y%m%d", time.gmtime(ts[i])))
    return Series(
        symbol=symbol.upper(),
        closes=closes,
        opens=opens,
        volumes=vols,
        dates=dates,
        fifty_two_high=meta.get("fiftyTwoWeekHigh"),
        fifty_two_low=meta.get("fiftyTwoWeekLow"),
        currency=meta.get("currency", "USD"),
        highs=highs,
        lows=lows,
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
        # Date of the last bar (YYYYMMDD). Tells the analyst how fresh the data is, and gives
        # memory.py a stable anchor to measure realized forward returns from.
        "as_of_date": series.dates[-1] if series.dates else None,
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
