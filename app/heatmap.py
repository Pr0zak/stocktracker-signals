"""Tile data for the market heat map. Pure — the LAYOUT happens on the device.

The server has no idea how wide the phone is, so it never computes rectangles. It returns, per tile,
the value to size by and the value to colour by, plus enough detail for the tap-through. The
squarified treemap runs in the app against the real viewport.

Two colour scales that must never be mixed:
  * price      — green/red, "the market moved"
  * signal     — amber, "this system has an opinion"
A tile carries `scale` saying which one applies, so the client cannot render a dip tier as a good day.
"""
from __future__ import annotations

# Ordering only — the app maps these to colour depth. Kept here so the server owns the ranking and
# the client cannot invent one.
DIP_RANK = {"mega_dip": 4, "below_line": 3, "pullback_10": 2, "pullback_5": 1}


def market_tiles(universe: dict | None, quotes: dict, *, limit: int) -> tuple[list[dict], list[str]]:
    """Size by market cap, colour by today's move.

    Returns (tiles, unpriced). A name in the universe with no quote is NOT silently dropped — it is
    named, because "we could not price it" is a different fact from "it did not move".
    """
    detail = (universe or {}).get("detail") or []
    tiles: list[dict] = []
    unpriced: list[str] = []
    for row in detail[:limit]:
        sym = row["symbol"]
        q = quotes.get(sym) or {}
        price, pct = q.get("price"), q.get("pct")
        if price is None or pct is None:
            unpriced.append(sym)
            continue
        cap = float(row.get("market_cap") or 0.0)
        if cap <= 0:
            unpriced.append(sym)
            continue
        tiles.append({
            "symbol": sym,
            "name": q.get("name") or row.get("name") or sym,
            "size": round(cap / 1e9, 2),       # $B — the area
            "value": round(float(pct), 2),      # % — the colour
            "scale": "price",
            "price": round(float(price), 2),
        })
    return tiles, unpriced


def signal_tiles(scan: dict | None, *, limit: int) -> tuple[list[dict], list[str]]:
    """Size by how far below the 52-week high, colour by the dip tier this system assigned.

    Names with no meaningful drawdown still get a floor area so they remain tappable rather than
    collapsing to an invisible sliver — a tile you cannot hit is the same as a tile that is absent.
    """
    rows = (scan or {}).get("results") or []
    tiles: list[dict] = []
    skipped: list[str] = []
    for r in rows:
        sym = r.get("symbol")
        if not sym:
            continue
        off = r.get("pct_off_52w_high")
        if off is None:
            skipped.append(sym)
            continue
        tiles.append({
            "symbol": sym,
            "name": "",
            "size": round(max(abs(float(off)), 1.5), 2),
            "value": float(DIP_RANK.get(r.get("dip") or "", 0)),
            "scale": "signal",
            "dip": r.get("dip"),
            "signal": r.get("signal"),
            "conviction": r.get("conviction"),
            "pct_off_52w_high": round(float(off), 2),
            "below_200wma": bool(r.get("below_200wma")),
        })
    tiles.sort(key=lambda t: -t["size"])
    return tiles[:limit], skipped
