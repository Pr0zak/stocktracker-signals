"""
Options data layer — Yahoo chain fetch (cookie+crumb handshake) + Black-Scholes greeks.

Yahoo's `v7/finance/options` endpoint lives on the same host as the chart feed (`market.py`) but,
unlike the chart path, it rejects an anonymous request with `401 Invalid Crumb`. It needs a small
handshake first (verified working 2026-07-20):

    1. GET https://fc.yahoo.com/                      -> sets an `A3` session cookie (404 body is fine)
    2. GET .../v1/test/getcrumb   (with that cookie)  -> returns a short crumb string
    3. GET .../v7/finance/options/{SYM}?crumb={CRUMB} (with the cookie) -> the chain JSON

The cookie+crumb are cached module-level and re-fetched automatically on a 401 (they expire). We
reuse the caller's `httpx.AsyncClient` (same client/style as market.py) and seed the cached cookies
onto it so a fresh client works on the first call rather than eating a wasted 401 round-trip.

Yahoo gives IV per contract but NOT the greeks, so we compute delta/gamma/theta/vega ourselves with
Black-Scholes (pure math, no scipy). Everything else the app needs (mid, spread%, breakeven,
expected move) is derived deterministically here too. See stocktracker/docs/options-roadmap.md Part 8.

Not investment advice.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# A realistic desktop-browser UA — Yahoo's crumb endpoint is pickier than the chart host.
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)
_HOSTS = ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]
_COOKIE_URL = "https://fc.yahoo.com/"
_CRUMB_URL = "https://query1.finance.yahoo.com/v1/test/getcrumb"

_SECONDS_PER_DAY = 86_400.0
_DAYS_PER_YEAR = 365.0

# Risk-free rate for the greeks. ~4.3% ≈ the 13-week T-bill (^IRX) mid-2026.
# TODO(OC-*): optionally pull ^IRX live via market._fetch_chart and cache daily; not a blocker.
RISK_FREE_RATE = 0.043

# --- module-level auth cache (cookie + crumb), refreshed on 401 ---
_crumb: str | None = None
_cookies: dict[str, str] = {}
_auth_lock = asyncio.Lock()


# ======================================================================================
# Typed structures (dataclasses, matching market.Series style — dataclasses.asdict()-friendly)
# ======================================================================================

@dataclass
class OptionContract:
    """One call or put. The `bid`/`ask`/`iv`/… block is pulled from Yahoo; everything from `mid`
    down is computed by `annotate_expiry` (None until then)."""
    type: str  # "call" | "put"
    contract_symbol: str
    strike: float
    expiration: int  # unix ts
    bid: float | None = None
    ask: float | None = None
    last_price: float | None = None
    implied_volatility: float | None = None  # Yahoo's IV, as a decimal (0.43 == 43%)
    open_interest: int | None = None
    volume: int | None = None
    in_the_money: bool | None = None
    # --- derived / greeks (filled by annotate_expiry) ---
    mid: float | None = None
    spread_pct: float | None = None
    breakeven: float | None = None
    expected_move: float | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None  # per calendar day
    vega: float | None = None   # per 1 vol point (1% change in IV)


@dataclass
class ExpiryChain:
    """The calls + puts for a single expiration."""
    expiration: int          # unix ts
    expiration_iso: str      # YYYY-MM-DD (UTC)
    calls: list[OptionContract] = field(default_factory=list)
    puts: list[OptionContract] = field(default_factory=list)
    # Filled by annotate_expiry: the ATM read used for the whole-expiry expected move.
    dte_days: float | None = None
    atm_iv: float | None = None
    expected_move: float | None = None  # spot * atm_iv * sqrt(dte/365), in price units


@dataclass
class OptionChain:
    """One symbol's chain as Yahoo returns it: a full list of available expirations + strikes, plus
    the contracts for the ONE expiry this fetch loaded (`expiry`). Yahoo serves one expiry per call —
    pass `expiry_ts` to `fetch_chain` to load a different one."""
    symbol: str
    spot: float | None
    currency: str
    market_state: str | None          # e.g. "REGULAR", "CLOSED", "POSTPOST" — drives the delayed badge
    quote_delayed: bool               # True when bid/ask are stale (outside regular hours)
    expirations: list[dict] = field(default_factory=list)  # [{"ts": int, "iso": str}, ...]
    strikes: list[float] = field(default_factory=list)
    expiry: ExpiryChain | None = None


# ======================================================================================
# Chain fetch — the cookie + crumb handshake
# ======================================================================================

def _headers() -> dict[str, str]:
    return {
        "User-Agent": _UA,
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }


async def _authenticate(client: httpx.AsyncClient) -> str:
    """Run the two-step handshake and cache the resulting cookie(s) + crumb. Returns the crumb."""
    global _crumb, _cookies
    # 1) Prime the session cookie. fc.yahoo.com answers 404 but still sends Set-Cookie (A3) — don't
    #    raise_for_status here; we only care about the cookie jar it populates on the client.
    try:
        await client.get(_COOKIE_URL, headers=_headers(), timeout=15, follow_redirects=True)
    except Exception:  # noqa: BLE001 — a transport hiccup here still often leaves a usable cookie
        pass
    # 2) Ask for a crumb using that cookie.
    r = await client.get(_CRUMB_URL, headers=_headers(), timeout=15)
    r.raise_for_status()
    crumb = r.text.strip()
    if not crumb or "<" in crumb or len(crumb) > 40:  # HTML/empty means the cookie didn't take
        raise RuntimeError(f"Yahoo returned no usable crumb (got {crumb!r:.60})")
    _crumb = crumb
    _cookies = {c.name: c.value for c in client.cookies.jar}
    return crumb


async def _ensure_auth(
    client: httpx.AsyncClient, *, force: bool = False, stale: str | None = None
) -> str:
    """Return a valid crumb, (re)authenticating if needed. Seeds cached cookies onto `client` so a
    freshly-created client works on its first request instead of taking a 401 detour.

    On a forced refresh after a 401, pass the crumb that just failed as `stale`: if another task
    already swapped in a newer crumb while we were waiting on the lock, we adopt that one instead of
    running the handshake a second time (compare-and-swap under the existing lock)."""
    async with _auth_lock:
        if force and stale is not None and _crumb and _crumb != stale:
            # A concurrent task already refreshed the crumb — use theirs, don't re-handshake.
            if _cookies:
                client.cookies.update(_cookies)
            return _crumb
        if force or not _crumb:
            return await _authenticate(client)
        if _cookies:
            client.cookies.update(_cookies)  # cheap; no-op when this client already has them
        return _crumb


async def _get_options(client: httpx.AsyncClient, symbol: str, crumb: str, expiry_ts: int | None) -> httpx.Response:
    """GET the options JSON with query1→query2 failover (network errors), returning the raw Response
    so the caller can react to a 401 by re-authenticating."""
    enc = symbol.upper().replace("^", "%5E")
    path = f"v7/finance/options/{enc}"
    # Let httpx URL-encode the query — a crumb can contain +, &, / which naive string-building
    # would corrupt (producing a bogus/invalid-crumb request).
    params: dict[str, object] = {"crumb": crumb}
    if expiry_ts:
        params["date"] = int(expiry_ts)
    last_err: Exception | None = None
    for host in _HOSTS:
        try:
            return await client.get(f"https://{host}/{path}", params=params, headers=_headers(), timeout=20)
        except Exception as e:  # noqa: BLE001 — try the next host on a transport error
            last_err = e
    raise RuntimeError(f"Yahoo options fetch failed for {symbol}: {last_err}")


def _f(v) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _i(v) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _parse_contract(raw: dict, kind: str) -> OptionContract:
    return OptionContract(
        type=kind,
        contract_symbol=str(raw.get("contractSymbol", "")),
        strike=_f(raw.get("strike")) or 0.0,
        expiration=_i(raw.get("expiration")) or 0,
        bid=_f(raw.get("bid")),
        ask=_f(raw.get("ask")),
        last_price=_f(raw.get("lastPrice")),
        implied_volatility=_f(raw.get("impliedVolatility")),
        open_interest=_i(raw.get("openInterest")),
        volume=_i(raw.get("volume")),
        in_the_money=raw.get("inTheMoney"),
    )


async def fetch_chain(
    client: httpx.AsyncClient, symbol: str, expiry_ts: int | None = None
) -> OptionChain:
    """Fetch one symbol's option chain from Yahoo (cookie+crumb handshake, cached + auto-re-auth on
    401). Yahoo returns ONE expiry's contracts per call plus the full expiration/strike lists; pass
    `expiry_ts` (a unix ts from `chain.expirations`) to load a specific expiration.

    Returns an `OptionChain` with spot, market state, all expirations, all strikes, and the loaded
    expiry's calls/puts. Greeks/derived fields are None until you run `annotate_expiry`.
    """
    crumb = await _ensure_auth(client)
    resp = await _get_options(client, symbol, crumb, expiry_ts)
    if resp.status_code == 401:  # crumb/cookie expired — re-auth once and retry
        crumb = await _ensure_auth(client, force=True, stale=crumb)
        resp = await _get_options(client, symbol, crumb, expiry_ts)
    resp.raise_for_status()
    data = resp.json()

    oc = data.get("optionChain") or {}
    if oc.get("error"):
        raise RuntimeError(f"Yahoo options error for {symbol}: {oc['error']}")
    results = oc.get("result") or []
    if not results:
        raise RuntimeError(f"no option chain for {symbol}")
    result = results[0]

    quote = result.get("quote") or {}
    spot = _f(quote.get("regularMarketPrice"))
    market_state = quote.get("marketState")
    # Yahoo option quotes are ~15-min delayed and bid/ask go stale/0 outside regular hours.
    quote_delayed = market_state not in (None, "REGULAR")

    expirations = [
        {"ts": int(ts), "iso": time.strftime("%Y-%m-%d", time.gmtime(int(ts)))}
        for ts in (result.get("expirationDates") or [])
    ]
    strikes = [f for f in (_f(s) for s in (result.get("strikes") or [])) if f is not None]

    expiry: ExpiryChain | None = None
    opts = result.get("options") or []
    if opts:
        blk = opts[0]
        exp_ts = _i(blk.get("expirationDate")) or 0
        expiry = ExpiryChain(
            expiration=exp_ts,
            expiration_iso=time.strftime("%Y-%m-%d", time.gmtime(exp_ts)) if exp_ts else "",
            calls=[_parse_contract(c, "call") for c in (blk.get("calls") or [])],
            puts=[_parse_contract(p, "put") for p in (blk.get("puts") or [])],
        )

    return OptionChain(
        symbol=str(quote.get("symbol", symbol.upper())),
        spot=spot,
        currency=str(quote.get("currency", "USD")),
        market_state=market_state,
        quote_delayed=quote_delayed,
        expirations=expirations,
        strikes=strikes,
        expiry=expiry,
    )


# ======================================================================================
# Black-Scholes greeks (pure functions — no scipy; normal CDF/PDF via math.erf)
# ======================================================================================

def norm_cdf(x: float) -> float:
    """Standard-normal cumulative distribution, N(x), via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """Standard-normal probability density, φ(x)."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black_scholes_greeks(
    spot: float,
    strike: float,
    t: float,
    sigma: float,
    r: float = RISK_FREE_RATE,
    is_call: bool = True,
) -> dict[str, float]:
    """Black-Scholes greeks for a European option on a non-dividend-paying underlying.

    Args:
        spot:   underlying price S
        strike: strike K
        t:      time to expiry in YEARS (e.g. 30 days -> 30/365)
        sigma:  implied volatility as a DECIMAL (0.30 == 30%)
        r:      annual risk-free rate as a decimal
        is_call: True for a call, False for a put

    Returns a dict with:
        delta  — call ∈ [0,1], put ∈ [-1,0]
        gamma  — ∂delta/∂S (same for calls and puts)
        theta  — time decay PER CALENDAR DAY (negative for long options)
        vega   — sensitivity to a 1-vol-POINT (1%) change in IV (positive for long options)

    Degenerate inputs (t<=0, sigma<=0, spot<=0, strike<=0) collapse to the option's intrinsic
    behaviour: delta is the ITM indicator (1/0 call, -1/0 put), the rest are 0.
    """
    if spot <= 0 or strike <= 0 or t <= 0 or sigma <= 0:
        if is_call:
            delta = 1.0 if spot > strike else 0.0
        else:
            delta = -1.0 if spot < strike else 0.0
        return {"delta": delta, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    pdf_d1 = norm_pdf(d1)

    gamma = pdf_d1 / (spot * sigma * sqrt_t)
    vega_per_1pt = spot * pdf_d1 * sqrt_t / 100.0  # raw vega is per 1.00 (100%) vol; /100 -> per 1%

    if is_call:
        delta = norm_cdf(d1)
        theta_year = (
            -(spot * pdf_d1 * sigma) / (2.0 * sqrt_t)
            - r * strike * math.exp(-r * t) * norm_cdf(d2)
        )
    else:
        delta = norm_cdf(d1) - 1.0
        theta_year = (
            -(spot * pdf_d1 * sigma) / (2.0 * sqrt_t)
            + r * strike * math.exp(-r * t) * norm_cdf(-d2)
        )

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta_year / _DAYS_PER_YEAR,  # per calendar day
        "vega": vega_per_1pt,
    }


# ======================================================================================
# Derived helpers + annotation
# ======================================================================================

def mid_price(bid: float | None, ask: float | None) -> float | None:
    """Mid = (bid+ask)/2, but ONLY when both sides are genuinely quoted (> 0). A 0 bid (common
    outside regular hours) would otherwise halve the true premium, so we return None instead and let
    callers fall back to last_price via `_limit_price`."""
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return None


def spread_pct(bid: float | None, ask: float | None) -> float | None:
    """(ask-bid)/mid — the bid/ask spread as a fraction of mid (liquidity proxy)."""
    m = mid_price(bid, ask)
    if m is None or m <= 0 or bid is None or ask is None:
        return None
    return (ask - bid) / m


def breakeven(strike: float, premium: float | None, is_call: bool) -> float | None:
    """Expiry breakeven: call = strike + premium, put = strike - premium."""
    if premium is None:
        return None
    return strike + premium if is_call else strike - premium


def expected_move(spot: float | None, iv: float | None, dte_days: float) -> float | None:
    """One-standard-deviation move to expiry, in price units: spot * IV * sqrt(DTE/365)."""
    if spot is None or iv is None or iv <= 0 or dte_days < 0:
        return None
    return spot * iv * math.sqrt(dte_days / _DAYS_PER_YEAR)


def _atm_iv(expiry: ExpiryChain, spot: float) -> float | None:
    """IV of the call nearest the money — the sensible single IV for a whole-expiry expected move."""
    with_iv = [c for c in expiry.calls if c.implied_volatility and c.strike > 0]
    if not with_iv:
        return None
    nearest = min(with_iv, key=lambda c: abs(c.strike - spot))
    return nearest.implied_volatility


def _round(v: float | None, n: int) -> float | None:
    return round(v, n) if v is not None else None


def annotate_expiry(
    chain: OptionChain,
    expiry: ExpiryChain | None = None,
    *,
    r: float = RISK_FREE_RATE,
    now_ts: float | None = None,
) -> ExpiryChain:
    """Annotate every contract in `expiry` (default: the chain's loaded expiry) in place with mid,
    spread%, breakeven, per-contract expected move, and Black-Scholes greeks, using the chain's spot.
    Also stamps the expiry-level ATM IV / DTE / expected move. Returns the annotated ExpiryChain.
    """
    expiry = expiry or chain.expiry
    if expiry is None:
        raise ValueError("chain has no loaded expiry to annotate (fetch one first)")
    spot = chain.spot
    now = now_ts if now_ts is not None else time.time()
    dte_days = max(0.0, (expiry.expiration - now) / _SECONDS_PER_DAY) if expiry.expiration else 0.0
    t_years = dte_days / _DAYS_PER_YEAR

    expiry.dte_days = round(dte_days, 3)
    if spot and spot > 0:
        expiry.atm_iv = _atm_iv(expiry, spot)
        expiry.expected_move = _round(expected_move(spot, expiry.atm_iv, dte_days), 4)

    for c in (*expiry.calls, *expiry.puts):
        is_call = c.type == "call"
        m = mid_price(c.bid, c.ask)
        c.mid = _round(m, 4)
        c.spread_pct = _round(spread_pct(c.bid, c.ask), 4)
        # Prefer mid for breakeven; fall back to lastPrice when bid/ask are stale/closed.
        premium = m if m is not None else c.last_price
        c.breakeven = _round(breakeven(c.strike, premium, is_call), 4)
        c.expected_move = _round(expected_move(spot, c.implied_volatility, dte_days), 4)
        if spot and spot > 0 and c.implied_volatility and c.implied_volatility > 0 and t_years > 0:
            g = black_scholes_greeks(spot, c.strike, t_years, c.implied_volatility, r, is_call)
            c.delta = _round(g["delta"], 4)
            c.gamma = _round(g["gamma"], 6)
            c.theta = _round(g["theta"], 4)
            c.vega = _round(g["vega"], 4)
    return expiry


# ======================================================================================
# OC-6a — ATM-IV history logging + IV rank
#
# The nightly scan (scan_job.py) appends one ATM-IV point per stock per day to
# data/iv_history.jsonl. `iv_rank` reads that history so the /options suggester can say whether
# today's implied vol is cheap or rich relative to the last ~year — the gate that decides whether a
# debit spread (OC-6b) is the smarter structure. All file access is best-effort: a missing/short/
# corrupt file yields [] (and a null rank), never an exception. See docs/options-roadmap.md.
# ======================================================================================

_DATA_DIR = Path(os.environ.get("SIGNALS_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
IV_HISTORY = _DATA_DIR / "iv_history.jsonl"

IV_HISTORY_WINDOW = 252   # ~one trading year of daily ATM-IV points
IV_RANK_MIN_POINTS = 20   # below this the rank is "still building" -> null
HIGH_IV_RANK = 50.0       # at/above this, IV is "rich" -> prefer the debit spread (OC-6b)


def _iter_iv_rows():
    """Yield each parsed row of data/iv_history.jsonl. Tolerates a missing/unreadable file and skips
    corrupt/partial lines — never raises."""
    if not IV_HISTORY.exists():
        return
    try:
        lines = IV_HISTORY.read_text().splitlines()
    except Exception:  # noqa: BLE001 — unreadable file behaves like an empty one
        return
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:  # noqa: BLE001 — skip a corrupt/partial line
            continue


def append_iv_history(symbol: str, atm_iv: float | None, *, date_str: str | None = None) -> bool:
    """Best-effort append of one ATM-IV point to data/iv_history.jsonl as {"symbol","date","atm_iv"}.
    One line per symbol per day (a same-day duplicate is skipped). Returns True if a line was written.
    NEVER raises — the nightly scan must not break on IV logging."""
    if atm_iv is None or atm_iv <= 0:
        return False
    sym = (symbol or "").upper()
    day = date_str or dt.date.today().isoformat()
    try:
        for row in _iter_iv_rows():  # de-dupe: one line/symbol/day
            if row.get("symbol") == sym and row.get("date") == day:
                return False
        IV_HISTORY.parent.mkdir(parents=True, exist_ok=True)
        with IV_HISTORY.open("a") as f:
            f.write(json.dumps({"symbol": sym, "date": day, "atm_iv": round(float(atm_iv), 6)}) + "\n")
        return True
    except Exception:  # noqa: BLE001 — logging is best-effort
        return False


def load_iv_history(symbol: str, *, window: int = IV_HISTORY_WINDOW) -> list[float]:
    """The last `window` ATM-IV points for `symbol` (oldest→newest) from data/iv_history.jsonl.
    Tolerates a missing/short/corrupt file (returns [])."""
    sym = (symbol or "").upper()
    vals = [
        float(r["atm_iv"]) for r in _iter_iv_rows()
        if r.get("symbol") == sym and isinstance(r.get("atm_iv"), (int, float)) and r["atm_iv"] > 0
    ]
    return vals[-window:]


def iv_rank(values: list[float], current: float | None = None, *, min_points: int = IV_RANK_MIN_POINTS) -> float | None:
    """IV rank = (current - min) / (max - min) * 100 over the historical `values` window, clamped to
    [0, 100]. `current` defaults to the most recent point in `values` (pass the live ATM IV to rank
    today against history). Returns None when there are fewer than `min_points` usable points (rank is
    "still building") or the window is flat (max == min)."""
    vals = [v for v in values if v is not None and v > 0]
    if len(vals) < min_points:
        return None
    cur = current if (current is not None and current > 0) else vals[-1]
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return None
    rank = (cur - lo) / (hi - lo) * 100.0
    return round(max(0.0, min(100.0, rank)), 1)


# ======================================================================================
# OC-1 — no-LLM call suggester (green/yellow/red light + delta-picked contracts)
#
# This is a deterministic structuring layer on top of the data layer above. It does NOT call an LLM
# (the analyst paragraph is OC-7). The "direction" input is a mechanical read of the SAME ChartMath
# indicators market.summarize() computes — the ones that feed the app's Signals card and /signal —
# so on this free path we stand in for the LLM verdict without calling it. See docs/options-roadmap.md
# Parts 2, 3, 5, 9. Not investment advice.
# ======================================================================================

# Expiry window (calendar days to expiration).
PREFERRED_DTE_LOW = 45
PREFERRED_DTE_HIGH = 90
TARGET_DTE = 60
MIN_DTE = 30
TARGET_DATE_BUFFER_DAYS = 3  # a chosen expiry must clear the user's target date by at least this

# Strike selection by delta, per risk profile.
TARGET_DELTAS = {"safer": 0.70, "balanced": 0.50, "cheaper": 0.32}
PROFILE_ORDER = ("safer", "balanced", "cheaper")

# OC-6b — the debit-spread short leg: near an entry-plan price target if one is derivable, else the
# ~0.30-delta OTM call (must be a HIGHER strike than the long leg).
SPREAD_SHORT_DELTA = 0.30

# Liquidity / caution thresholds (Part 2).
OI_FLOOR = 250
WIDE_SPREAD = 0.10  # bid/ask spread as a fraction of mid
BULLISH_SCORE = 55  # directional score at/above this reads bullish (roadmap "conviction >= ~55")


def directional_read(summary: dict) -> dict:
    """A deterministic bullish/neutral/bearish read + 0-100 score from a `market.summarize()` dict.

    This is the no-LLM stand-in for the app's Signals-card direction on the options path. It combines
    the SAME indicators the LLM analyst is shown (MA structure, MACD, relative strength, RSI, distance
    from the 52-week high) into one directional score centred on 50 (neutral). It is intentionally
    simple and transparent — it is a go/no-go gate for suggesting a call, not a price forecast.
    """
    score = 50.0
    reasons: list[str] = []

    gc = summary.get("golden_cross")
    if gc is True:
        score += 8
        reasons.append("SMA20 above SMA50")
    elif gc is False:
        score -= 8
        reasons.append("SMA20 below SMA50")

    for key, label in (("pct_vs_sma20", "20-day"), ("pct_vs_sma50", "50-day")):
        v = summary.get(key)
        if v is None:
            continue
        if v > 0:
            score += 6
        elif v < 0:
            score -= 6
            reasons.append(f"price under its {label} average")

    mh = summary.get("macd_hist")
    if mh is not None:
        if mh > 0:
            score += 8
            reasons.append("MACD positive")
        elif mh < 0:
            score -= 8
            reasons.append("MACD negative")

    rs = summary.get("rel_strength_3mo_vs_benchmark")
    if rs is not None:
        if rs > 0:
            score += 8
            reasons.append("outperforming the market")
        elif rs < 0:
            score -= 8
            reasons.append("lagging the market")

    rsi = summary.get("rsi14")
    if rsi is not None:
        score += 3 if rsi >= 50 else -3
        if rsi > 75:
            score -= 3
            reasons.append("RSI overbought")
        elif rsi < 25:
            score += 2
            reasons.append("RSI oversold")

    off = summary.get("pct_off_52w_high")
    if off is not None:
        if off > -5:
            score += 4
            reasons.append("near its 52-week high")
        elif off < -25:
            score -= 3

    score = max(0.0, min(100.0, score))
    if score >= BULLISH_SCORE:
        signal = "bullish"
    elif score <= 45:
        signal = "bearish"
    else:
        signal = "neutral"
    return {"signal": signal, "score": int(round(score)), "bullish": signal == "bullish", "reasons": reasons}


# --- small formatting/parse helpers ---

def _utc_date(ts: float) -> dt.date:
    """The UTC calendar date of a unix timestamp (matches the chain's gmtime-based iso strings)."""
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).date()


def _parse_iso_date(s: str | None) -> dt.date | None:
    if not s:
        return None
    try:
        return dt.date.fromisoformat(str(s)[:10])
    except ValueError:
        return None


def _fmt_strike(strike: float) -> str:
    return str(int(strike)) if float(strike).is_integer() else f"{strike:g}"


def _mmddyy(ts: int) -> str:
    return time.strftime("%m/%d/%y", time.gmtime(int(ts)))


def _limit_price(c: OptionContract) -> float | None:
    """The limit to quote: contract mid, falling back to last trade when bid/ask are stale/closed."""
    if c.mid is not None and c.mid > 0:
        return c.mid
    if c.last_price is not None and c.last_price > 0:
        return c.last_price
    return None


# --- expiry selection ---

def select_expiry(
    chain: OptionChain,
    *,
    now: float | None = None,
    target_date: str | None = None,
    earnings_date: str | None = None,
) -> tuple[dict | None, list[str]]:
    """Pick the best expiration from `chain.expirations`. Returns ({"ts","iso","dte"} | None, warnings).

    Preference order: an expiry 45-90 DTE, clearing any `target_date` (+buffer) and NOT straddling a
    known earnings date (IV-crush), closest to ~60 DTE. If earnings exclusion empties the window we
    relax it. If nothing lands 45-90 DTE we fall back to the nearest > 30 DTE (with a warning), and
    failing that the longest-dated expiry available.
    """
    warnings: list[str] = []
    now = now if now is not None else time.time()
    tgt = _parse_iso_date(target_date)
    earn = _parse_iso_date(earnings_date)

    cands: list[dict] = []
    for e in chain.expirations:
        ts = int(e["ts"])
        dte = (ts - now) / _SECONDS_PER_DAY
        if dte <= 0:
            continue
        exp_date = _utc_date(ts)
        if tgt is not None and exp_date < tgt + dt.timedelta(days=TARGET_DATE_BUFFER_DAYS):
            continue  # can't express the user's timeframe — the option would expire too early
        # Round DTE ONCE, up front, and classify off the SAME integer we later display, so a 44.6-DTE
        # expiry (shown as 45) can't be flagged out-of-window while its rationale calls it in-window.
        cands.append({"ts": ts, "iso": e["iso"], "dte": dte, "dte_int": int(round(dte)), "date": exp_date})
    if not cands:
        return None, warnings

    def straddles_earnings(c: dict) -> bool:
        return earn is not None and _utc_date(now) < earn <= c["date"]

    in_window = [c for c in cands if PREFERRED_DTE_LOW <= c["dte_int"] <= PREFERRED_DTE_HIGH]
    pool = in_window or None
    if pool:
        clear = [c for c in pool if not straddles_earnings(c)]
        if clear:  # only drop earnings-straddlers when doing so leaves something
            pool = clear
        elif earn is not None:
            warnings.append("all 45-90 day expiries straddle the next earnings date — IV-crush risk")
        chosen = min(pool, key=lambda c: abs(c["dte"] - TARGET_DTE))
    else:
        longer = [c for c in cands if c["dte_int"] > MIN_DTE]
        chosen = (
            min(longer, key=lambda c: abs(c["dte"] - TARGET_DTE)) if longer
            else max(cands, key=lambda c: c["dte"])
        )
        warnings.append(
            f"no expiry in the 45-90 day window — using the closest available (~{chosen['dte_int']} DTE)"
        )
    return {"ts": chosen["ts"], "iso": chosen["iso"], "dte": chosen["dte_int"]}, warnings


# --- strike selection ---

def _pick_by_delta(calls: list[OptionContract], target: float, *, prefer_oi: bool = False) -> OptionContract | None:
    """The call whose delta is nearest `target`. When `prefer_oi`, favour a liquid (OI>=250) strike
    among those within 0.12 delta of target before falling back to the outright nearest."""
    valid = [c for c in calls if c.delta is not None and 0.0 < c.delta < 1.0]
    if not valid:
        return None
    if prefer_oi:
        liquid = [c for c in valid if (c.open_interest or 0) >= OI_FLOOR and abs(c.delta - target) <= 0.12]
        if liquid:
            return min(liquid, key=lambda c: abs(c.delta - target))
    return min(valid, key=lambda c: abs(c.delta - target))


def _ordered_profiles(style: str) -> list[str]:
    """Profiles with the requested `style` first, then the remaining two in canonical order."""
    style = style if style in TARGET_DELTAS else "balanced"
    return [style] + [p for p in PROFILE_ORDER if p != style]


def _candidate(c: OptionContract, profile: str, *, symbol: str, spot: float, budget: float | None) -> dict:
    limit = _limit_price(c)
    cost = round(limit * 100.0, 2) if limit is not None else None
    # How many whole contracts the budget buys (None when no budget or no cost to size against).
    affordable = (
        int(math.floor(budget / cost)) if (budget and budget > 0 and cost and cost > 0) else None
    )
    # The quantity we actually suggest / put on the order ticket. It is whatever you can afford, else
    # the 1-contract minimum — NEVER 0, since a BUY-0 ticket is meaningless and would let max_loss
    # read 0 while the ticket still risks a full premium.
    n = affordable if (affordable and affordable > 0) else 1
    # The `contracts` field the app renders: the affordable count when a budget was given (1 when the
    # budget can't cover even one contract — we still show the 1-contract minimum), else None so the
    # user sizes it themselves.
    if budget and budget > 0:
        contracts: int | None = affordable if (affordable and affordable >= 1) else 1
    else:
        contracts = None
    # INVARIANT: max_loss == cost × the ticket quantity (never 0 while a BUY-n ticket exists).
    max_loss = round(cost * n, 2) if cost is not None else None
    be = c.breakeven
    be_pct = round((be - spot) / spot * 100.0, 2) if (be is not None and spot) else None
    ticket = (
        f"BUY {n} {symbol} {_mmddyy(c.expiration)} {_fmt_strike(c.strike)} C "
        f"@ {limit:.2f} LMT" if limit is not None else ""
    )
    return {
        "profile": profile,
        "contract_symbol": c.contract_symbol,
        "strike": c.strike,
        "limit_price": round(limit, 2) if limit is not None else None,
        "cost": cost,
        "max_loss": max_loss,
        "contracts": contracts,
        "breakeven": round(be, 2) if be is not None else None,
        "breakeven_pct": be_pct,
        "delta": c.delta,
        "theta": c.theta,
        "iv": c.implied_volatility,
        "spread_pct": c.spread_pct,
        "open_interest": c.open_interest,
        "expected_move": round(c.expected_move, 2) if c.expected_move is not None else None,
        "order_ticket": ticket,
    }


def build_candidates(
    expiry: ExpiryChain, *, symbol: str, spot: float, style: str, budget: float | None,
) -> list[dict]:
    """Up to three delta-picked call candidates (safer/balanced/cheaper), the requested `style` first,
    de-duped when two profiles collapse onto the same contract."""
    out: list[dict] = []
    seen: set[str] = set()
    for profile in _ordered_profiles(style):
        pick = _pick_by_delta(expiry.calls, TARGET_DELTAS[profile], prefer_oi=(profile == "balanced"))
        if pick is None or pick.contract_symbol in seen:
            continue
        cand = _candidate(pick, profile, symbol=symbol, spot=spot, budget=budget)
        if cand["limit_price"] is None:  # unquotable (no mid, no last) — skip
            continue
        seen.add(pick.contract_symbol)
        out.append(cand)
    return out


# --- OC-6b debit-spread helpers ---

def _find_call(expiry: ExpiryChain, contract_symbol: str) -> OptionContract | None:
    """The loaded call contract matching `contract_symbol` (the long leg is picked as a candidate dict,
    but the spread math needs the underlying contract for its mid)."""
    return next((c for c in expiry.calls if c.contract_symbol == contract_symbol), None)


def _spread_short_leg(
    calls: list[OptionContract], *, long_strike: float, target_price: float | None = None,
) -> OptionContract | None:
    """Pick the SELL leg of a debit call spread: a quotable call at a HIGHER strike than the long leg.
    If a price `target_price` is derivable, take the nearest strike at/above it; else the call nearest
    ~0.30 delta. Returns None when no higher quotable strike exists."""
    higher = [c for c in calls if c.strike > long_strike and _limit_price(c) is not None]
    if not higher:
        return None
    if target_price is not None:
        at_or_above = [c for c in higher if c.strike >= target_price]
        if at_or_above:
            return min(at_or_above, key=lambda c: c.strike)
    with_delta = [c for c in higher if c.delta is not None and 0.0 < c.delta < 1.0]
    pool = with_delta or higher
    return min(pool, key=lambda c: abs((c.delta if c.delta is not None else 0.0) - SPREAD_SHORT_DELTA))


def _debit_spread(
    expiry: ExpiryChain, ref: dict | None, *, target_price: float | None,
) -> dict | None:
    """Build the OC-6b debit-call-spread alternative from the leading (balanced) candidate `ref`:
    BUY that call, SELL a higher call. Returns the alternative block, or None when it can't be formed
    (no ref, no higher strike, or non-positive net debit)."""
    if ref is None:
        return None
    long_c = _find_call(expiry, ref["contract_symbol"])
    if long_c is None:
        return None
    short_c = _spread_short_leg(expiry.calls, long_strike=long_c.strike, target_price=target_price)
    if short_c is None or short_c.strike <= long_c.strike:
        return None
    long_mid = _limit_price(long_c)
    short_mid = _limit_price(short_c)
    if long_mid is None or short_mid is None:
        return None
    net_debit = long_mid - short_mid  # per share; long (lower strike) costs more than short
    if net_debit <= 0:  # degenerate quotes — a debit spread must cost something
        return None
    width = short_c.strike - long_c.strike
    return {
        "structure": "debit_call_spread",
        "long_strike": long_c.strike,
        "short_strike": short_c.strike,
        "net_debit": round(net_debit, 2),
        "cost": round(net_debit * 100.0, 2),
        "max_profit": round((width - net_debit) * 100.0, 2),
        "max_loss": round(net_debit * 100.0, 2),
        "breakeven": round(long_c.strike + net_debit, 2),
        "note": (
            f"Debit call spread: buy the {_fmt_strike(long_c.strike)} call and sell the "
            f"{_fmt_strike(short_c.strike)} call. Caps your gain at the short strike but cuts the cost "
            "(and the max loss) vs the plain long call — the smarter structure when IV is rich. Not "
            "investment advice."
        ),
    }


# --- assembly (the full response body) ---

def assemble_suggestion(
    chain: OptionChain,
    expiry: ExpiryChain,
    summary: dict,
    *,
    chosen: dict,
    style: str = "balanced",
    budget: float | None = None,
    earnings_date: str | None = None,
    now: float | None = None,
    extra_warnings: list[str] | None = None,
    iv_rank: float | None = None,
    target_price: float | None = None,
) -> dict:
    """Build the OC-1 `/options/{symbol}` response body from an already-fetched + annotated chain.

    Pure/deterministic: `chain` must already carry `expiry` annotated via `annotate_expiry`. `summary`
    is a `market.summarize()` dict (for the directional read); `earnings_date` is a YYYY-MM-DD string
    (or None). The route layer handles fetching and passes everything in.
    """
    now = now if now is not None else time.time()
    symbol = chain.symbol
    spot = chain.spot or 0.0
    warnings: list[str] = list(extra_warnings or [])

    direction = directional_read(summary)
    candidates = build_candidates(expiry, symbol=symbol, spot=spot, style=style, budget=budget)

    # Style-first guarantee (Part 9): if the requested style's strike was unquotable and got dropped,
    # the app still leads with candidates[0] — say so instead of silently showing a different profile.
    want = style if style in TARGET_DELTAS else "balanced"
    if candidates and not any(c["profile"] == want for c in candidates):
        warnings.append(f"the {want} strike wasn't quotable — showing {candidates[0]['profile']} instead")

    # Earnings-in-window: does a known earnings date fall between now and the chosen expiry?
    earn = _parse_iso_date(earnings_date)
    exp_date = _utc_date(int(chosen["ts"]))
    earnings_in_window = bool(earn and _utc_date(now) < earn <= exp_date)
    earnings_block = (
        {"date": earn.isoformat(), "in_window": earnings_in_window} if earn else None
    )

    # The balanced pick drives the liquidity gate (or the first candidate if it de-duped away).
    ref = next((c for c in candidates if c["profile"] == "balanced"), candidates[0] if candidates else None)

    # --- warnings (Part 2) ---
    if earnings_in_window:
        warnings.append(f"earnings on {earn.isoformat()} is before expiry — expect an IV drop (crush) after")
    dte = chosen["dte"]
    if dte < PREFERRED_DTE_LOW or dte > PREFERRED_DTE_HIGH:
        warnings.append(f"chosen expiry is {dte} DTE, outside the preferred 45-90 day window")
    for c in candidates:
        tag = c["profile"]
        if c["spread_pct"] is not None and c["spread_pct"] > WIDE_SPREAD:
            warnings.append(f"{tag} strike has a wide bid/ask spread ({c['spread_pct']*100:.0f}%) — use a limit order")
        if (c["open_interest"] or 0) < OI_FLOOR:
            warnings.append(f"{tag} strike has thin open interest ({c['open_interest'] or 0}) — may be hard to fill")
        if budget and c["cost"] and c["cost"] > budget:
            warnings.append(f"a single {tag} contract is ${c['cost']:.0f} — above your ${budget:.0f} budget")
    if chain.quote_delayed:
        warnings.append("quotes are delayed / market closed — bid-ask liquidity can't be confirmed right now")

    # --- traffic light (Part 2) ---
    ref_wide = bool(ref and ref["spread_pct"] is not None and ref["spread_pct"] > WIDE_SPREAD)
    ref_thin = bool(ref and (ref["open_interest"] or 0) < OI_FLOOR)
    ref_unconfirmed = bool(ref and ref["spread_pct"] is None)  # market closed → can't verify spread
    outside_window = dte < PREFERRED_DTE_LOW or dte > PREFERRED_DTE_HIGH
    cautions: list[str] = []
    if earnings_in_window:
        cautions.append("earnings before expiry (IV-crush risk)")
    if ref_wide:
        cautions.append("wide bid/ask spread on the suggested contract")
    if ref_thin:
        cautions.append("thin open interest on the suggested contract")
    if ref_unconfirmed:
        cautions.append("market closed — contract liquidity can't be confirmed")
    if outside_window:
        cautions.append("expiry sits outside the 45-90 day window")

    if ref is None:
        light = "red"
        light_reason = "no quotable contracts for the chosen expiry"
    elif not direction["bullish"]:
        light = "red"
        light_reason = "directional signal says wait/sell — options amplify direction"
    elif not cautions:
        light = "green"
        light_reason = (
            f"direction reads bullish (score {direction['score']}) and the suggested contract is liquid "
            "with no earnings before expiry"
        )
    else:
        light = "yellow"
        light_reason = f"direction reads bullish (score {direction['score']}), but {cautions[0]}"

    # --- OC-6b: debit-spread alternative + IV-rank gating ---
    alternative = _debit_spread(expiry, ref, target_price=target_price)
    # Recommend the spread when IV is rich (rank >= ~50) AND a spread is actually available.
    recommend_alternative = bool(iv_rank is not None and iv_rank >= HIGH_IV_RANK and alternative is not None)

    atm_iv = expiry.atm_iv
    ctx_bits: list[str] = []
    if atm_iv:
        ctx_bits.append(f"ATM IV ~{atm_iv*100:.0f}%")
    if iv_rank is not None:
        ctx_bits.append(f"IV rank ~{iv_rank:.0f}")
    else:
        ctx_bits.append("IV rank still building (need ~20 days of history)")
    ctx = (" (" + ", ".join(ctx_bits) + ")") if ctx_bits else ""
    if alternative is not None and recommend_alternative:
        spread_bit = (
            " A debit call spread alternative is available and PREFERRED here — IV rank is high, so "
            "selling a higher call to offset the rich premium is the smarter structure."
        )
    elif alternative is not None:
        spread_bit = " A debit call spread alternative is available (caps upside but costs less) if you'd rather spend less."
    else:
        spread_bit = ""
    structure_note = f"Plain long call{ctx}.{spread_bit}"

    # expiry rationale — one sentence explaining the pick.
    if outside_window:
        rationale = f"{dte} days out — no expiry landed in the 45-90 day sweet spot, so this is the closest usable one."
    elif earnings_in_window:
        rationale = f"{dte} days out (in the 45-90 day sweet spot); note earnings falls before it."
    else:
        rationale = f"{dte} days out — in the 45-90 day sweet spot and clear of the next earnings date."

    return {
        "symbol": symbol,
        "spot": round(spot, 4) if spot else None,
        "as_of": now,
        "quote_delayed": chain.quote_delayed,
        "light": light,
        "light_reason": light_reason,
        "expiry": {"ts": int(chosen["ts"]), "iso": chosen["iso"], "dte": dte, "rationale": rationale},
        "expected_move": round(expiry.expected_move, 2) if expiry.expected_move is not None else None,
        "structure": "long_call",
        "structure_note": structure_note,
        "candidates": candidates,
        "warnings": list(dict.fromkeys(warnings)),  # de-dup, keep order
        "earnings": earnings_block,
        # --- OC-6/OC-7 additive fields (nullable; existing keys above are unchanged) ---
        "iv_rank": iv_rank,
        "alternative": alternative,
        "recommend_alternative": recommend_alternative,
        "analyst": None,  # OC-7: filled by the /options route when deep=true; null otherwise
    }


# ======================================================================================
# OC-8 — the wheel (no-LLM cash-secured-put + covered-call suggesters)
#
# Two sibling income/accumulation tools, both premium-COLLECTING and IRA-eligible at Fidelity
# (cash-secured / covered — never naked). See docs/options-roadmap.md Part 10.
#
#   • Cash-secured put  — sell a put at a strike you'd happily own at; reserve strike×100 in cash.
#                         Keep the premium if it stays up; get assigned 100 shares/contract at the
#                         strike (net cost = strike − premium, BELOW today) if it dips.
#   • Covered call      — on ≥100 shares already held, sell a call at/above a target for income;
#                         keep the premium, or get called away (capped upside) at the strike.
#
# Theta favours the seller, so the wheel leans SHORTER-dated than the OC-1 debit-call window — the
# put side ~25-50 DTE (target ~35), the call side ~25-45 DTE. This is decision support, NOT advice:
# a CSP and a covered call at the same strike are the SAME mildly-bullish bet with FULL downside
# below the strike — the guardrail is "only sell puts on names you'd happily own at that strike."
# ======================================================================================

# Short-dated income windows (calendar days to expiration). Shorter than OC-1's 45-90.
PUT_DTE_LOW, PUT_DTE_HIGH, PUT_DTE_TARGET = 25, 50, 35
CALL_DTE_LOW, CALL_DTE_HIGH, CALL_DTE_TARGET = 25, 45, 35
WHEEL_MIN_DTE = 14  # the shortest we'll fall back to when nothing lands in-window

# Cash-secured-put strikes by assignment likelihood (|put delta|). Aggressive = near the money (high
# chance you get the shares); conservative = deep OTM (low chance, you mostly just bank the premium).
PUT_TARGET_DELTAS = {"aggressive": 0.45, "balanced": 0.30, "conservative": 0.20}
PUT_PROFILE_ORDER = ("aggressive", "balanced", "conservative")  # descending assignment probability

# Covered-call default strike when no explicit target is given: ~0.30 delta OTM.
COVERED_CALL_TARGET_DELTA = 0.30

HIGH_IV = 0.80  # a candidate IV at/above this gets a "premiums are rich, but so is the priced move" note

PUT_NOTE = (
    "Cash-secured put: you set aside strike x 100 in cash and get paid the premium now. If the stock "
    "stays up you keep the premium; if it dips you're assigned 100 shares/contract at the strike "
    "(net cost = strike - premium, shown). Only sell puts on names you'd happily own at that strike "
    "- you carry the full downside below it. Not investment advice."
)
COVERED_CALL_NOTE = (
    "Covered call: income on shares you already own. If the stock rises past the strike your shares "
    "are called away (sold) at the strike - capped upside - but you keep the premium. "
    "Not investment advice."
)


# --- shared short-dated expiry picker (used by both wheel legs) ---

def select_wheel_expiry(
    chain: OptionChain,
    *,
    low: int,
    high: int,
    target: int,
    min_dte: int = WHEEL_MIN_DTE,
    now: float | None = None,
    earnings_date: str | None = None,
) -> tuple[dict | None, list[str]]:
    """Pick a short-dated income expiry from `chain.expirations`. Returns ({"ts","iso","dte"}|None, warns).

    Prefers an expiry `low`-`high` DTE, NOT straddling a known earnings date (assignment/gap risk),
    closest to ~`target`. If the earnings filter empties the window we relax it (with a warning). If
    nothing lands in-window we fall back to the nearest >= `min_dte`, and failing that the longest
    available. Mirrors `select_expiry` but with the wheel's shorter windows and no target-date input.
    """
    warnings: list[str] = []
    now = now if now is not None else time.time()
    earn = _parse_iso_date(earnings_date)

    cands: list[dict] = []
    for e in chain.expirations:
        ts = int(e["ts"])
        dte = (ts - now) / _SECONDS_PER_DAY
        if dte <= 0:
            continue
        # Round DTE once and classify off the SAME integer we later display (matches select_expiry).
        cands.append({"ts": ts, "iso": e["iso"], "dte": dte, "dte_int": int(round(dte)), "date": _utc_date(ts)})
    if not cands:
        return None, warnings

    def straddles_earnings(c: dict) -> bool:
        return earn is not None and _utc_date(now) < earn <= c["date"]

    in_window = [c for c in cands if low <= c["dte_int"] <= high]
    pool = in_window or None
    if pool:
        clear = [c for c in pool if not straddles_earnings(c)]
        if clear:
            pool = clear
        elif earn is not None:
            warnings.append(f"all {low}-{high} day expiries straddle the next earnings date — assignment/gap risk")
        chosen = min(pool, key=lambda c: abs(c["dte"] - target))
    else:
        longer = [c for c in cands if c["dte_int"] >= min_dte]
        chosen = (
            min(longer, key=lambda c: abs(c["dte"] - target)) if longer
            else max(cands, key=lambda c: c["dte"])
        )
        warnings.append(
            f"no expiry in the {low}-{high} day window — using the closest available (~{chosen['dte_int']} DTE)"
        )
    return {"ts": chosen["ts"], "iso": chosen["iso"], "dte": chosen["dte_int"]}, warnings


def _wheel_rationale(dte: int, low: int, high: int, *, earnings_in_window: bool) -> str:
    if dte < low or dte > high:
        return f"{dte} days out — no expiry landed in the {low}-{high} day window, so this is the closest usable one."
    if earnings_in_window:
        return f"{dte} days out (short-dated so theta works for the seller); note earnings falls before it."
    return f"{dte} days out — short-dated, so time decay (theta) works in your favour as the seller, and clear of the next earnings date."


# --- cash-secured put suggester ---

def _pick_put_by_abs_delta(
    puts: list[OptionContract], target_abs: float, *, prefer_oi: bool = False
) -> OptionContract | None:
    """The put whose |delta| is nearest `target_abs`. Put delta is negative, so we compare on abs().
    When `prefer_oi`, favour a liquid (OI>=250) strike within 0.12 of target before the outright nearest."""
    valid = [p for p in puts if p.delta is not None and -1.0 < p.delta < 0.0]
    if not valid:
        return None
    if prefer_oi:
        liquid = [p for p in valid if (p.open_interest or 0) >= OI_FLOOR and abs(abs(p.delta) - target_abs) <= 0.12]
        if liquid:
            return min(liquid, key=lambda p: abs(abs(p.delta) - target_abs))
    return min(valid, key=lambda p: abs(abs(p.delta) - target_abs))


def _ordered_put_profiles(style: str) -> list[str]:
    """Profiles with the requested `style` first, then the rest in canonical (assignment-prob) order."""
    style = style if style in PUT_TARGET_DELTAS else "balanced"
    return [style] + [p for p in PUT_PROFILE_ORDER if p != style]


def _put_candidate(
    p: OptionContract, profile: str, *, symbol: str, spot: float, cash: float, dte: int
) -> dict | None:
    """One cash-secured-put candidate, sized off the ticket quantity n (never a $0 figure on a live
    ticket). Returns None if the contract has no quotable premium (no mid, no last)."""
    limit = _limit_price(p)  # the premium per share we'd quote (mid, else last trade)
    if limit is None:
        return None
    reserve_per = p.strike * 100.0  # cash-secured: strike × 100 set aside per contract
    # How many whole contracts the cash secures; n is the ticket quantity — never 0 (min 1) so no
    # figure reads $0 while a SELL-n ticket exists (the OC-1 sizing invariant).
    affordable = int(math.floor(cash / reserve_per)) if (cash > 0 and reserve_per > 0) else 0
    n = affordable if affordable > 0 else 1

    net_cost = p.strike - limit  # your effective purchase price per share if assigned
    return {
        "profile": profile,
        "contract_symbol": p.contract_symbol,
        "strike": p.strike,
        "limit_price": round(limit, 2),
        "premium_income": round(limit * 100.0 * n, 2),
        "net_cost_per_share": round(net_cost, 2),
        "discount_vs_spot_pct": round((net_cost - spot) / spot * 100.0, 2) if spot else None,
        "cash_to_reserve": round(reserve_per * n, 2),
        "contracts": n,
        "static_yield_pct": round(limit / p.strike * 100.0, 2) if p.strike else None,
        "annualized_yield_pct": round(limit / p.strike * 100.0 * _DAYS_PER_YEAR / dte, 2) if (p.strike and dte > 0) else None,
        "assignment_prob_pct": round(abs(p.delta) * 100.0),
        "breakeven": round(net_cost, 2),  # strike − premium (same as net cost per share)
        "delta": p.delta,
        "theta": p.theta,
        "iv": p.implied_volatility,
        "open_interest": p.open_interest,
        "spread_pct": p.spread_pct,
        "order_ticket": f"SELL {n} {symbol} {_mmddyy(p.expiration)} {_fmt_strike(p.strike)} P @ {limit:.2f} LMT",
    }


def assemble_put_suggestion(
    chain: OptionChain,
    expiry: ExpiryChain,
    *,
    chosen: dict,
    cash: float,
    style: str = "balanced",
    earnings_date: str | None = None,
    now: float | None = None,
    extra_warnings: list[str] | None = None,
) -> dict:
    """Build the OC-8 `/puts/{symbol}` response body from an already-fetched + annotated chain.

    Pure/deterministic: `chain.expiry` must already be annotated via `annotate_expiry`. `cash` is the
    reserve the user can set aside (sizes the contract count); `style` (aggressive|balanced|conservative)
    is surfaced first. Up to three delta-picked put candidates, de-duped when two collapse onto the
    same strike.
    """
    now = now if now is not None else time.time()
    symbol = chain.symbol
    spot = chain.spot or 0.0
    dte = int(chosen["dte"])
    warnings: list[str] = list(extra_warnings or [])

    candidates: list[dict] = []
    seen: set[str] = set()
    for profile in _ordered_put_profiles(style):
        pick = _pick_put_by_abs_delta(expiry.puts, PUT_TARGET_DELTAS[profile], prefer_oi=(profile == "balanced"))
        if pick is None or pick.contract_symbol in seen:
            continue
        cand = _put_candidate(pick, profile, symbol=symbol, spot=spot, cash=cash, dte=dte)
        if cand is None:  # unquotable (no mid, no last) — skip
            continue
        seen.add(pick.contract_symbol)
        candidates.append(cand)

    # Earnings-in-window: does a known earnings date fall between now and the chosen expiry?
    earn = _parse_iso_date(earnings_date)
    exp_date = _utc_date(int(chosen["ts"]))
    earnings_in_window = bool(earn and _utc_date(now) < earn <= exp_date)
    earnings_block = {"date": earn.isoformat(), "in_window": earnings_in_window} if earn else None

    # --- warnings ---
    if earnings_in_window:
        warnings.append(f"earnings on {earn.isoformat()} is before expiry — a gap-down could leave you assigned at a loss")
    if dte < PUT_DTE_LOW or dte > PUT_DTE_HIGH:
        warnings.append(f"chosen expiry is {dte} DTE, outside the preferred {PUT_DTE_LOW}-{PUT_DTE_HIGH} day window")
    for c in candidates:
        tag = c["profile"]
        if c["spread_pct"] is not None and c["spread_pct"] > WIDE_SPREAD:
            warnings.append(f"{tag} strike has a wide bid/ask spread ({c['spread_pct']*100:.0f}%) — use a limit order")
        if (c["open_interest"] or 0) < OI_FLOOR:
            warnings.append(f"{tag} strike has thin open interest ({c['open_interest'] or 0}) — may be hard to fill")
        if c["iv"] is not None and c["iv"] >= HIGH_IV:
            warnings.append(f"{tag} strike IV is elevated (~{c['iv']*100:.0f}%) — premium is rich but the market is pricing a big move")
        if cash < c["strike"] * 100.0:  # can't fully cash-secure even one contract at this strike
            warnings.append(
                f"${cash:.0f} cash is below the {tag} strike's ${c['strike']*100:.0f} reserve (strike x 100) — "
                "showing a 1-contract minimum; you'd need that set aside to be fully cash-secured"
            )
    if chain.quote_delayed:
        warnings.append("quotes are delayed / market closed — bid-ask liquidity can't be confirmed right now")

    rationale = _wheel_rationale(dte, PUT_DTE_LOW, PUT_DTE_HIGH, earnings_in_window=earnings_in_window)
    return {
        "symbol": symbol,
        "spot": round(spot, 4) if spot else None,
        "as_of": now,
        "quote_delayed": chain.quote_delayed,
        "expiry": {"ts": int(chosen["ts"]), "iso": chosen["iso"], "dte": dte, "rationale": rationale},
        "candidates": candidates,
        "warnings": list(dict.fromkeys(warnings)),  # de-dup, keep order
        "earnings": earnings_block,
        "note": PUT_NOTE,
    }


# --- covered call suggester ---

def _pick_covered_call(
    calls: list[OptionContract], *, spot: float, target: float | None
) -> tuple[OptionContract | None, bool]:
    """Pick ONE quotable call: the nearest strike AT/ABOVE `target` if given, else ~0.30 delta (OTM).
    Returns (contract|None, target_fallback) — target_fallback is True when a `target` was given but
    sits above every available strike, so we fell back to the delta pick."""
    quotable = [c for c in calls if c.delta is not None and 0.0 < c.delta < 1.0 and _limit_price(c) is not None]
    if not quotable:
        return None, False
    if target is not None:
        at_or_above = [c for c in quotable if c.strike >= target]
        if at_or_above:
            return min(at_or_above, key=lambda c: (c.strike, abs(c.strike - target))), False
        # target above every strike — fall back to the delta pick and flag it
        return min(quotable, key=lambda c: abs(c.delta - COVERED_CALL_TARGET_DELTA)), True
    return min(quotable, key=lambda c: abs(c.delta - COVERED_CALL_TARGET_DELTA)), False


def assemble_covered_call(
    chain: OptionChain,
    expiry: ExpiryChain,
    *,
    shares: int,
    chosen: dict,
    target: float | None = None,
    now: float | None = None,
    extra_warnings: list[str] | None = None,
) -> dict | None:
    """Build the OC-8 `/covered_call/{symbol}` response body. Returns None when no call is quotable
    (the route turns that into a 400). `shares` (>=100, validated by the route) sizes `contracts`."""
    now = now if now is not None else time.time()
    symbol = chain.symbol
    spot = chain.spot or 0.0
    dte = int(chosen["dte"])
    contracts = shares // 100
    warnings: list[str] = list(extra_warnings or [])

    pick, target_fallback = _pick_covered_call(expiry.calls, spot=spot, target=target)
    if pick is None:
        return None

    limit = _limit_price(pick) or 0.0
    premium_income = round(limit * 100.0 * contracts, 2)
    candidate = {
        "contract_symbol": pick.contract_symbol,
        "strike": pick.strike,
        "limit_price": round(limit, 2),
        "premium_income": premium_income,
        "premium_yield_pct": round(limit / spot * 100.0, 2) if spot else None,
        "annualized_yield_pct": round(limit / spot * 100.0 * _DAYS_PER_YEAR / dte, 2) if (spot and dte > 0) else None,
        "assignment_prob_pct": round((pick.delta or 0.0) * 100.0),
        # Upside to the strike (can be negative if the strike is below spot) + the premium, from today.
        "called_away_gain_from_here": round(((pick.strike - spot) * 100.0 * contracts) + premium_income, 2),
        "delta": pick.delta,
        "theta": pick.theta,
        "iv": pick.implied_volatility,
        "open_interest": pick.open_interest,
        "spread_pct": pick.spread_pct,
        "order_ticket": f"SELL {contracts} {symbol} {_mmddyy(pick.expiration)} {_fmt_strike(pick.strike)} C @ {limit:.2f} LMT",
    }

    # --- warnings ---
    if target is not None and target_fallback:
        warnings.append(
            f"your ${target:g} target is above every listed strike — showing the ~{int(COVERED_CALL_TARGET_DELTA*100)}-delta call instead"
        )
    if pick.strike < spot:
        warnings.append(
            f"the strike (${_fmt_strike(pick.strike)}) is below the current price (${spot:.2f}) — "
            "if called away you'd sell below today's price; consider a higher strike"
        )
    if candidate["spread_pct"] is not None and candidate["spread_pct"] > WIDE_SPREAD:
        warnings.append(f"the call has a wide bid/ask spread ({candidate['spread_pct']*100:.0f}%) — use a limit order")
    if (candidate["open_interest"] or 0) < OI_FLOOR:
        warnings.append(f"the call has thin open interest ({candidate['open_interest'] or 0}) — may be hard to fill")
    if chain.quote_delayed:
        warnings.append("quotes are delayed / market closed — bid-ask liquidity can't be confirmed right now")
    # NOTE: ex-dividend early-assignment (deep-ITM calls exercised for the dividend) is intentionally
    # not emitted — this service has no ex-dividend DATE source, and the spec says skip rather than
    # fabricate one. Wire it in here if a dividend-calendar feed is added later.

    rationale = _wheel_rationale(dte, CALL_DTE_LOW, CALL_DTE_HIGH, earnings_in_window=False)
    return {
        "symbol": symbol,
        "spot": round(spot, 4) if spot else None,
        "as_of": now,
        "quote_delayed": chain.quote_delayed,
        "shares": shares,
        "contracts": contracts,
        "expiry": {"ts": int(chosen["ts"]), "iso": chosen["iso"], "dte": dte, "rationale": rationale},
        "candidate": candidate,
        "warnings": list(dict.fromkeys(warnings)),  # de-dup, keep order
        "note": COVERED_CALL_NOTE,
    }
