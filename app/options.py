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
import math
import time
from dataclasses import dataclass, field

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


async def _ensure_auth(client: httpx.AsyncClient, *, force: bool = False) -> str:
    """Return a valid crumb, (re)authenticating if needed. Seeds cached cookies onto `client` so a
    freshly-created client works on its first request instead of taking a 401 detour."""
    async with _auth_lock:
        if force or not _crumb:
            return await _authenticate(client)
        if _cookies:
            client.cookies.update(_cookies)  # cheap; no-op when this client already has them
        return _crumb


async def _get_options(client: httpx.AsyncClient, symbol: str, crumb: str, expiry_ts: int | None) -> httpx.Response:
    """GET the options JSON with query1→query2 failover (network errors), returning the raw Response
    so the caller can react to a 401 by re-authenticating."""
    enc = symbol.upper().replace("^", "%5E")
    qs = f"?crumb={crumb}" + (f"&date={int(expiry_ts)}" if expiry_ts else "")
    path = f"v7/finance/options/{enc}{qs}"
    last_err: Exception | None = None
    for host in _HOSTS:
        try:
            return await client.get(f"https://{host}/{path}", headers=_headers(), timeout=20)
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
        crumb = await _ensure_auth(client, force=True)
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
    """Mid = (bid+ask)/2. Falls back to whichever side exists (0 bid outside hours is common)."""
    if bid is not None and ask is not None and (bid > 0 or ask > 0):
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
