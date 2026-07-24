"""
Market-now pulse (AIE-5): a compact snapshot of what US markets are doing RIGHT NOW — session phase,
major indices, VIX, sector-ETF rotation, and the user's watchlist movers — fed to the analyst for an
instant plain-language overview. Reuses the options.py cookie+crumb handshake for ONE batched Yahoo
v7 quote across every symbol.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import httpx

from . import market_calendar, options

_ET = ZoneInfo("America/New_York")

# (symbol, label) — the tape we read for the overview.
INDICES = [("^GSPC", "S&P 500"), ("^IXIC", "Nasdaq"), ("^DJI", "Dow"), ("^RUT", "Russell 2000")]
VIX = "^VIX"
SECTORS = [
    ("XLK", "Technology"), ("XLF", "Financials"), ("XLE", "Energy"), ("XLV", "Health Care"),
    ("XLI", "Industrials"), ("XLY", "Cons. Discretionary"), ("XLP", "Cons. Staples"),
    ("XLU", "Utilities"), ("XLB", "Materials"), ("XLRE", "Real Estate"), ("XLC", "Comm. Services"),
]


def session_phase(now: dt.datetime | None = None) -> str:
    """US equity session phase in ET: PRE (4:00-9:30), REGULAR (9:30-16:00), AFTER (16:00-20:00), else
    CLOSED. Weekends AND full NYSE holidays are CLOSED (holidays via market_calendar — this closes the
    old v1 gap where a holiday read as its normal weekday phase)."""
    et = (now or dt.datetime.now(dt.timezone.utc)).astimezone(_ET)
    if et.weekday() >= 5 or market_calendar.is_market_holiday(et.date()):
        return "CLOSED"
    mins = et.hour * 60 + et.minute
    if 4 * 60 <= mins < 9 * 60 + 30:
        return "PRE"
    if 9 * 60 + 30 <= mins < 16 * 60:
        return "REGULAR"
    if 16 * 60 <= mins < 20 * 60:
        return "AFTER"
    return "CLOSED"


def _num(v) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


async def fetch_quotes(client: httpx.AsyncClient, symbols: list[str]) -> dict[str, dict]:
    """One batched Yahoo v7 quote for all symbols (cookie+crumb via options._ensure_auth, re-auth on a
    401). Returns {SYMBOL: {price, pct, pre_pct, post_pct, state, name}}. Raises only if every host
    fails; unknown symbols are simply absent from the result."""
    if not symbols:
        return {}
    crumb = await options._ensure_auth(client)
    params = {"symbols": ",".join(s.upper() for s in symbols), "crumb": crumb}
    data = None
    last_err: Exception | None = None
    for host in options._HOSTS:
        try:
            url = f"https://{host}/v7/finance/quote"
            r = await client.get(url, params=params, headers=options._headers(), timeout=20)
            if r.status_code == 401:  # crumb/cookie expired — re-auth once and retry
                crumb = await options._ensure_auth(client, force=True, stale=crumb)
                params["crumb"] = crumb
                r = await client.get(url, params=params, headers=options._headers(), timeout=20)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:  # noqa: BLE001 — fail over to the next host
            last_err = e
    if data is None:
        raise RuntimeError(f"Yahoo quote fetch failed: {last_err}")
    out: dict[str, dict] = {}
    for q in (data.get("quoteResponse") or {}).get("result") or []:
        sym = (q.get("symbol") or "").upper()
        if not sym:
            continue
        out[sym] = {
            "price": _num(q.get("regularMarketPrice")),
            "pct": _num(q.get("regularMarketChangePercent")),
            "pre_pct": _num(q.get("preMarketChangePercent")),
            "post_pct": _num(q.get("postMarketChangePercent")),
            "state": q.get("marketState"),
            "name": q.get("shortName") or sym,
        }
    return out


def _sess_pct(row: dict, phase: str) -> float | None:
    """The % move that matters for the current session: pre/post outside RTH, else the regular day move."""
    if phase == "PRE" and row.get("pre_pct") is not None:
        return row["pre_pct"]
    if phase == "AFTER" and row.get("post_pct") is not None:
        return row["post_pct"]
    return row.get("pct")


def _r2(v) -> float | None:
    return round(v, 2) if isinstance(v, (int, float)) else None


def assemble(quotes: dict[str, dict], watchlist: list[str], phase: str, *, top: int = 3) -> dict:
    """Turn a quote map into the compact analyst-facing snapshot (pure — easy to unit-test)."""
    wl = [s.upper() for s in (watchlist or [])]
    indices = [{"name": name, "symbol": s, "pct": _r2((quotes.get(s) or {}).get("pct"))} for s, name in INDICES]
    vix_row = quotes.get(VIX) or {}
    sectors = [{"name": name, "symbol": s, "pct": _r2((quotes.get(s) or {}).get("pct"))}
               for s, name in SECTORS if (quotes.get(s) or {}).get("pct") is not None]
    sectors.sort(key=lambda x: x["pct"], reverse=True)
    movers = [{"symbol": s, "pct": _r2(_sess_pct(quotes[s], phase))}
              for s in wl if s in quotes and _sess_pct(quotes[s], phase) is not None]
    movers.sort(key=lambda x: x["pct"], reverse=True)
    return {
        "session": phase,
        "as_of_et": dt.datetime.now(_ET).strftime("%Y-%m-%d %H:%M ET"),
        "indices": indices,
        "vix": {"level": _r2(vix_row.get("price")), "pct": _r2(vix_row.get("pct"))},
        "sector_leaders": sectors[:top],
        "sector_laggards": (sectors[-top:][::-1] if len(sectors) > top else sectors[::-1]),
        "watchlist_movers": {
            "up": [m for m in movers[:top] if m["pct"] is not None and m["pct"] > 0],
            "down": [m for m in movers[::-1][:top] if m["pct"] is not None and m["pct"] < 0],
        },
        "watchlist_count": len(wl),
    }


async def build_snapshot(client: httpx.AsyncClient, watchlist: list[str], *, top: int = 3) -> dict:
    """Fetch the tape + watchlist in one batched quote and assemble the market-now snapshot."""
    phase = session_phase()
    wl = [s.upper() for s in (watchlist or [])]
    quotes = await fetch_quotes(client, [s for s, _ in INDICES] + [VIX] + [s for s, _ in SECTORS] + wl)
    return assemble(quotes, wl, phase, top=top)
