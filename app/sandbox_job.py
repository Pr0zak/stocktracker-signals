"""
Pure trade-logic for the AI paper-trading sandbox — gating, deterministic order validation + fills, and
NAV computation. NO network / no LLM here: these functions take already-fetched data (quotes, an analyst
decision) and mutate an in-memory ledger blob, so they're fully unit-testable offline. The I/O
orchestration (build snapshots, call the analyst, fetch quotes, persist) lives in main.py's /sandbox/tick
endpoint, where the shared httpx client + the `_build_portfolio_snapshot`/`_snapshot` helpers already live.

The invariant that matters: the LLM only PROPOSES; this module is the sole authority on what the ledger
does. It clamps every order to the cash floor, the per-exposure cap, and the shares actually held, and
asserts cash conservation before the caller commits.
"""
from __future__ import annotations

import datetime as dt
import time
from typing import Callable
from zoneinfo import ZoneInfo

from . import market_calendar

ET = ZoneInfo("America/New_York")

# Session windows in ET seconds-of-day (mirrors market_now.session_phase, kept local to avoid a cycle).
_REG_OPEN = 9 * 3600 + 30 * 60   # 09:30
_AFTER_END = 20 * 3600           # 20:00


def now_et() -> dt.datetime:
    return dt.datetime.now(ET)


def today_et_str(now: dt.datetime | None = None) -> str:
    return (now or now_et()).date().isoformat()


def is_crypto(symbol: str) -> bool:
    return symbol.upper().endswith("-USD")


def round_shares(symbol: str, shares: float) -> float:
    """Whole shares for stocks/ETFs, 6-dp fractional for crypto (matches _sanitize_plan)."""
    return round(shares, 6) if is_crypto(symbol) else float(int(shares))


def tick_gate(blob: dict, *, now: dt.datetime | None = None, force: bool = False) -> tuple[bool, str]:
    """Whether the tick should place trades. Returns (proceed, status). `force` (a manual "run now")
    relaxes the intraday-phase check but still requires a real trading day and honours the day cursor
    only when not forced."""
    now = now or now_et()
    today = now.date()
    if not (blob.get("settings") or {}).get("master_enabled", False):
        return False, "disabled"
    if not market_calendar.is_trading_day(today):
        return False, "market_closed"
    sod = now.hour * 3600 + now.minute * 60 + now.second
    in_session = _REG_OPEN <= sod < _AFTER_END  # REGULAR or AFTER — same-day prices exist
    if not in_session and not force:
        return False, "outside_session"
    if blob.get("last_tick_date") == today.isoformat() and not force:
        return False, "already_ran"
    return True, "ok"


def positions_value(positions: list[dict], price_of: Callable[[str], float | None]) -> float:
    """Mark-to-market value of the book; positions with no fresh price contribute their last mark of 0
    (the caller flags unpriceable symbols separately and simply won't trade them)."""
    total = 0.0
    for p in positions:
        px = price_of(p["symbol"])
        if px:
            total += p["shares"] * px
    return round(total, 2)


def _find(positions: list[dict], symbol: str) -> dict | None:
    s = symbol.upper()
    for p in positions:
        if p["symbol"].upper() == s:
            return p
    return None


def exit_date_flatten_orders(blob: dict, price_of: Callable[[str], float | None]) -> list[dict] | None:
    """If today (ET) is on/after the configured exit_date, return sell-everything orders (bypasses the
    LLM). None means "not past exit date"."""
    ed = (blob.get("settings") or {}).get("exit_date")
    if not ed:
        return None
    try:
        if now_et().date() < dt.date.fromisoformat(ed):
            return None
    except ValueError:
        return None
    orders = []
    for p in blob.get("positions", []):
        if price_of(p["symbol"]):
            orders.append({"symbol": p["symbol"], "side": "sell", "shares": p["shares"],
                           "dollars": 0.0, "conviction": 100, "reason": "Exit date reached — flattening to cash."})
    return orders


def validate_and_fill(
    blob: dict,
    orders: list[dict],
    price_of: Callable[[str], float | None],
    *,
    group_of: Callable[[str], str],
    now_ts: float | None = None,
    source: str = "haiku_tick",
) -> tuple[dict, list[dict], list[dict]]:
    """Apply an analyst order list to the ledger under hard risk limits. Returns (new_blob, filled_rows,
    skipped_rows). Sells run before buys (free cash / cut exposure first). The blob is copied, not mutated
    in place. Raises AssertionError on a cash-conservation violation — the caller must NOT persist then."""
    now_ts = now_ts or time.time()
    b = {**blob, "positions": [dict(p) for p in blob.get("positions", [])]}
    s = {**b.get("settings", {})}
    cash0 = round(float(b.get("cash", 0.0)), 2)
    cash = cash0
    positions = b["positions"]

    max_pos_pct = float(s.get("max_position_pct", 20.0))
    cash_floor_pct = float(s.get("cash_floor_pct", 10.0))
    slip = float(s.get("slippage_bps", 5)) / 10_000.0
    min_conv = int(s.get("min_conviction_to_trade", 55))
    max_trades = int(s.get("max_trades_per_tick", 4))
    max_new = int(s.get("max_new_positions_per_tick", 2))
    allow_crypto = bool(s.get("allow_crypto", True))
    allow_etf = bool(s.get("allow_etf", True))  # ETF filtering is best-effort (source-tagged upstream)

    filled: list[dict] = []
    skipped: list[dict] = []
    buy_notional = 0.0
    sell_notional = 0.0
    new_positions = 0

    def _skip(o: dict, why: str) -> None:
        skipped.append({"ts": now_ts, "date": today_et_str(), "symbol": o.get("symbol", "").upper(),
                        "side": o.get("side"), "status": "skipped", "shares": 0.0, "price": None,
                        "conviction": o.get("conviction"), "source": source, "reason": o.get("reason", ""),
                        "skip_reason": why})

    def _fill(o: dict, side: str, shares: float, price: float, realized: float, cash_after: float,
              pos_after: dict | None) -> None:
        filled.append({
            "ts": now_ts, "date": today_et_str(), "symbol": o["symbol"].upper(), "side": side,
            "status": "filled", "shares": round(shares, 6), "price": round(price, 4),
            "gross": round(shares * price, 2), "cash_after": round(cash_after, 2),
            "avg_cost_after": round(pos_after["avg_cost"], 4) if pos_after else None,
            "realized_pl": round(realized, 2), "exposure_group": group_of(o["symbol"]),
            "conviction": o.get("conviction"), "source": source, "reason": o.get("reason", ""),
        })

    sells = [o for o in orders if o.get("side") == "sell"]
    buys = [o for o in orders if o.get("side") == "buy"]

    # ---- SELLS first ----
    for o in sells:
        if len(filled) >= max_trades:
            _skip(o, "max_trades_per_tick reached"); continue
        sym = o["symbol"].upper()
        px = price_of(sym)
        if not px:
            _skip(o, "no fresh price"); continue
        pos = _find(positions, sym)
        if not pos or pos["shares"] <= 0:
            _skip(o, "not held"); continue
        want = float(o.get("shares") or 0) or (float(o.get("dollars") or 0) / px)
        shares = round_shares(sym, min(want, pos["shares"]))
        if shares <= 0:
            _skip(o, "nothing to sell"); continue
        fill = px * (1 - slip)
        proceeds = shares * fill
        realized = shares * (fill - pos["avg_cost"])
        cash += proceeds
        sell_notional += proceeds
        pos["shares"] = round(pos["shares"] - shares, 8)
        b["realized_pl_total"] = round(float(b.get("realized_pl_total", 0.0)) + realized, 2)
        _fill(o, "sell", shares, fill, realized, cash, pos)
        if pos["shares"] <= 1e-9:
            positions.remove(pos)

    # ---- BUYS by conviction ----
    # Equity + per-group values are pinned AFTER sells so caps use one stable denominator for the tick.
    pv_after_sells = positions_value(positions, price_of)
    equity = round(cash + pv_after_sells, 2)
    floor = cash_floor_pct / 100.0 * equity
    group_value: dict[str, float] = {}
    for p in positions:
        px = price_of(p["symbol"])
        if px:
            g = group_of(p["symbol"])
            group_value[g] = group_value.get(g, 0.0) + p["shares"] * px

    for o in sorted(buys, key=lambda x: -int(x.get("conviction") or 0)):
        if len(filled) >= max_trades:
            _skip(o, "max_trades_per_tick reached"); continue
        sym = o["symbol"].upper()
        if is_crypto(sym) and not allow_crypto:
            _skip(o, "crypto disabled"); continue
        if int(o.get("conviction") or 0) < min_conv:
            _skip(o, f"below conviction floor ({min_conv})"); continue
        px = price_of(sym)
        if not px:
            _skip(o, "no fresh price"); continue
        available = cash - floor
        if available <= 0:
            _skip(o, "at cash floor"); continue
        g = group_of(sym)
        cap_room = max_pos_pct / 100.0 * equity - group_value.get(g, 0.0)
        if cap_room <= 0:
            _skip(o, f"exposure '{g}' at {max_pos_pct:.0f}% cap"); continue
        is_new = _find(positions, sym) is None
        if is_new and new_positions >= max_new:
            _skip(o, "max_new_positions_per_tick reached"); continue
        spend = min(float(o.get("dollars") or 0) or available, available, cap_room)
        fill = px * (1 + slip)
        shares = round_shares(sym, spend / fill)
        if shares <= 0:
            _skip(o, "cash/cap left no whole share"); continue
        cost = shares * fill
        cash -= cost
        buy_notional += cost
        group_value[g] = group_value.get(g, 0.0) + cost
        pos = _find(positions, sym)
        if pos:
            tot = pos["shares"] + shares
            pos["avg_cost"] = (pos["shares"] * pos["avg_cost"] + shares * fill) / tot
            pos["shares"] = round(tot, 8)
            pos["last_add_at"] = now_ts
        else:
            pos = {"symbol": sym, "shares": round(shares, 8), "avg_cost": fill,
                   "exposure_group": g, "opened_at": now_ts, "last_add_at": now_ts}
            positions.append(pos)
            new_positions += 1
        _fill(o, "buy", shares, fill, 0.0, cash, pos)

    b["cash"] = round(cash, 2)
    # Cash-conservation invariant (fail-closed): the caller aborts + does not persist on violation.
    delta = round(cash, 2) - cash0
    expected = round(sell_notional - buy_notional, 2)
    assert abs(delta - expected) < 0.01, f"cash not conserved: Δcash={delta} vs {expected}"
    return b, filled, skipped


def nav_row(blob: dict, *, positions_val: float, spy_price: float | None, now_ts: float | None = None) -> dict:
    """One equity-curve point: total equity (cash + marked positions) + the benchmark's shadow value."""
    now_ts = now_ts or time.time()
    cash = round(float(blob.get("cash", 0.0)), 2)
    equity = round(cash + positions_val, 2)
    bench = blob.get("benchmark") or {}
    bench_val = round(float(bench.get("shares", 0.0)) * spy_price, 2) if spy_price else None
    return {
        "ts": now_ts, "date": today_et_str(), "equity": equity, "cash": cash,
        "positions_value": round(positions_val, 2), "funded_total": round(float(blob.get("funded_total", 0.0)), 2),
        "benchmark_symbol": bench.get("symbol", "^GSPC"), "benchmark_value": bench_val,
        "num_positions": len(blob.get("positions", [])),
    }
