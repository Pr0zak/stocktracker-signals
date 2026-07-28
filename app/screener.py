"""MB-15 / MB-18 — the 200-week value screen: rank a universe by our OWN thesis, with no LLM.

mungbeans' actual product is a weekly pure-Python pass that surfaces names trading below their
200-week line. Everything needed is already computed by `cycle.long_term_trend`; what was missing was
a way to run it across more than one symbol and order the results.

Two deliberate constraints carried over from the analysis:

* **This score is SEPARATE from the 0-100 momentum score and must stay that way.** Folding a
  mean-reversion value tilt into a momentum score produces a number that means nothing in either
  direction.
* **It is CONTEXT, not a buy signal.** The touch study run on this codebase found below-the-line dips
  *underperformed* SPY on a 12/24-month forward horizon (NKE, PFE samples). The endpoint says so, and
  `value_score` is explicitly a "how dislocated" measure, not "how good a buy".

Price-based only, so it costs one Yahoo weekly-bars call per symbol and no fundamentals quota. Drill
into a name for the quality/FCF/insider context cards.
"""
from __future__ import annotations

import asyncio

import httpx

# Yahoo predefined screeners with a VALUE tilt. `discover.py` deliberately uses momentum-flavoured
# ones (actives, gainers, growth) for /recommendations; ranking those by dislocation would mostly
# surface names that are dislocated *because they just ran*, which is the opposite of the thesis.
VALUE_SCREENS = ("undervalued_large_caps", "undervalued_growth_stocks", "aggressive_small_caps")

_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
_MIN_PRICE = 5.0
_MIN_CAP = 2_000_000_000.0
_CONCURRENCY = 6

# Weights for the composite. Each component is clipped to [0, 1] first so no single axis can run away
# and dominate — an unbounded z-score would let one freak name outrank everything on its own.
_W_DISLOCATION = 0.40   # how far below the 200-week line
_W_DRAWDOWN_Z = 0.25    # how unusual that drawdown is for THIS symbol's own history
_W_OVERSOLD = 0.20      # 14-week RSI
_W_RECOVERING = 0.15    # turning back up, rather than still falling


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def value_score(trend: dict | None) -> dict | None:
    """Score one symbol's `long_term_trend` block. Returns None when the history can't support it.

    Higher = more dislocated below its own long-term trend. Components are returned alongside the
    score so the card can show WHY a name ranks where it does; a screener that emits a bare number
    the user cannot interrogate is not usable for a decision they take with money.
    """
    if not trend or "price_vs_200w_sma_pct" not in trend:
        return None   # under ~4 years of weekly history the 200-week fields are absent, not zero

    pct = float(trend["price_vs_200w_sma_pct"])
    # 0 at the line, 1.0 at 50% below it. Above the line scores 0 — this screen is about names
    # BELOW their trend, and a name 30% above should not earn partial credit for being "less above".
    dislocation = _clip01(-pct / 50.0)

    z = trend.get("drawdown_z")
    # -2 sigma or worse = 1.0. Positive z (shallower drawdown than usual) = 0.
    drawdown = _clip01(-float(z) / 2.0) if isinstance(z, (int, float)) else 0.0

    rsi = trend.get("rsi_14w")
    # 1.0 at RSI 20, 0 at RSI 50.
    oversold = _clip01((50.0 - float(rsi)) / 30.0) if isinstance(rsi, (int, float)) else 0.0

    direction = trend.get("direction")
    # mungbeans' key nuance: still-falling ("deepening") is a knife, not a discount. Recovering earns
    # the full weight, deepening earns nothing — the difference between the two is the whole point.
    recovering = 1.0 if direction == "recovering" else 0.0

    score = (_W_DISLOCATION * dislocation + _W_DRAWDOWN_Z * drawdown
             + _W_OVERSOLD * oversold + _W_RECOVERING * recovering)

    return {
        "value_score": round(100.0 * score, 1),
        "components": {
            "dislocation": round(dislocation, 3),
            "drawdown_z": round(drawdown, 3),
            "oversold": round(oversold, 3),
            "recovering": round(recovering, 3),
        },
        "below_line": bool(trend.get("below_line")),
        "price_vs_200w_sma_pct": pct,
        "rsi_14w": trend.get("rsi_14w"),
        "direction": direction,
        "zone": trend.get("zone"),
        "drawdown_z_raw": z,
        "pct_off_10y_high": trend.get("pct_off_10y_high"),
        "history_years": trend.get("history_years"),
    }


def rank(rows: list[dict], *, limit: int, below_line_only: bool) -> list[dict]:
    """Order scored rows. Ties break on the raw dislocation so the order is deterministic."""
    keep = [r for r in rows if r] if not below_line_only else [r for r in rows if r and r["below_line"]]
    keep.sort(key=lambda r: (-r["value_score"], r.get("price_vs_200w_sma_pct", 0.0), r["symbol"]))
    return keep[:limit]


async def _screen(client: httpx.AsyncClient, scr: str, count: int) -> list[str]:
    try:
        r = await client.get(
            "https://query2.finance.yahoo.com/v1/finance/screener/predefined/saved",
            params={"scrIds": scr, "count": count}, headers=_UA, timeout=15.0)
        r.raise_for_status()
        quotes = (r.json().get("finance", {}).get("result") or [{}])[0].get("quotes") or []
    except Exception:  # noqa: BLE001 — one dead screener must not kill the whole screen
        return []
    out = []
    for q in quotes:
        sym = (q.get("symbol") or "").upper()
        price = q.get("regularMarketPrice") or 0.0
        cap = q.get("marketCap") or 0.0
        if not sym or q.get("quoteType") not in ("EQUITY", "ETF"):
            continue
        if float(price or 0) < _MIN_PRICE or float(cap or 0) < _MIN_CAP:
            continue
        out.append(sym)
    return out


async def build_universe(client: httpx.AsyncClient, *, extra: list[str] | None = None,
                         per_screen: int = 25) -> list[str]:
    """Value-tilted candidate pool, plus anything the caller pins (e.g. the user's watchlist)."""
    got = await asyncio.gather(*[_screen(client, s, per_screen) for s in VALUE_SCREENS])
    seen: list[str] = []
    for sym in [s for batch in got for s in batch] + [s.upper() for s in (extra or [])]:
        if sym not in seen:
            seen.append(sym)
    return seen


async def screen(client: httpx.AsyncClient, trend_of, symbols: list[str], *,
                 limit: int = 20, below_line_only: bool = True) -> tuple[list[dict], list[str], list[str]]:
    """Score `symbols` concurrently. Returns (ranked rows, too_short, failed).

    `too_short` and `failed` are SEPARATE because they are different facts: the first means the name
    genuinely lacks ~4 years of weekly history, the second means we could not fetch it (a rate limit,
    a timeout). Collapsing them told the user "not enough history to score AAPL" during a Yahoo rate
    limit — a confident statement about the company that was really a statement about our network.

    `trend_of(client, symbol) -> dict | None` is injected so this is testable without network and so
    the caller owns caching.
    """
    sem = asyncio.Semaphore(_CONCURRENCY)

    async def one(sym: str):
        async with sem:
            try:
                t = await trend_of(client, sym)
            except Exception:  # noqa: BLE001 — a single bad symbol must not sink the screen
                return sym, None, "failed"
            scored = value_score(t)
            if scored is None:
                return sym, None, "too_short"
            return sym, {"symbol": sym, **scored}, None

    results = await asyncio.gather(*[one(s) for s in symbols])
    rows = [r for _, r, _ in results if r]
    too_short = [s for s, r, why in results if r is None and why == "too_short"]
    failed = [s for s, r, why in results if r is None and why == "failed"]
    return rank(rows, limit=limit, below_line_only=below_line_only), too_short, failed
