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
_REG_END = 16 * 3600             # 16:00 — regular session close
_AFTER_END = 20 * 3600           # 20:00 — after-hours close


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
    allow_after = bool((blob.get("settings") or {}).get("allow_after_hours", False))
    in_regular = _REG_OPEN <= sod < _REG_END
    in_after = _REG_END <= sod < _AFTER_END
    if not force:
        if in_after and not allow_after:
            return False, "after_hours_disabled"
        if not (in_regular or in_after):
            return False, "outside_session"
    if blob.get("last_tick_date") == today.isoformat() and not force:
        return False, "already_ran"
    return True, "ok"


def mark_price(p: dict, price_of: Callable[[str], float | None]) -> float | None:
    """Best available mark for a position: fresh quote, else last known mark, else cost basis.

    Marking an unpriceable holding at ZERO (the old behaviour) was wrong in two directions at once.
    It wrote an invented drawdown into the append-only NAV log, which nothing can later correct; and
    because the exposure-cap check used the same marks, the position's group looked EMPTY, so a
    sibling ticker could buy straight through the cap — a price outage silently disabled a risk
    limit. A stale mark is an approximation; zero is a fabrication.
    """
    px = price_of(p["symbol"])
    if px and px > 0:
        return float(px)
    for key in ("last_price", "avg_cost"):
        v = p.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return None


def stale_marks(positions: list[dict], price_of: Callable[[str], float | None]) -> list[str]:
    """Symbols currently valued from a stale mark, so the caller can surface it rather than hide it."""
    out: list[str] = []
    for p in positions:
        px = price_of(p["symbol"])
        if not (px and px > 0) and float(p.get("shares") or 0) > 0:
            out.append(p["symbol"])
    return out


def positions_value(positions: list[dict], price_of: Callable[[str], float | None]) -> float:
    """Mark-to-market value of the book, falling back to the last known mark (see `mark_price`)."""
    total = 0.0
    for p in positions:
        px = mark_price(p, price_of)
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
    exclude: set[str] | None = None,
    liquidation: bool = False,
) -> tuple[dict, list[dict], list[dict]]:
    """Apply an analyst order list to the ledger under hard risk limits. Returns (new_blob, filled_rows,
    skipped_rows). Sells run before buys (free cash / cut exposure first). The blob is copied, not mutated
    in place. Raises AssertionError on a cash-conservation violation — the caller must NOT persist then.

    `liquidation=True` lifts the anti-CHURN limits (per-tick trade count and the turnover cap) for an
    exit-date flatten. Those caps exist to stop the strategy trading itself into the ground; a
    liquidation the user scheduled is not churn, and throttling it left the account still holding most
    of the book on the very date they asked to be out — while the response said "flattening to cash".
    Every risk limit that protects the ACCOUNT (cash conservation, shares actually held) still applies."""
    now_ts = now_ts or time.time()
    b = {**blob, "positions": [dict(p) for p in blob.get("positions", [])]}
    s = {**b.get("settings", {})}
    cash0 = round(float(b.get("cash", 0.0)), 2)
    cash = cash0
    positions = b["positions"]

    # Stamp a fresh mark on every position we can price, so `mark_price`'s fallback is yesterday's
    # close rather than a cost basis from months ago if a quote later goes missing.
    for p in positions:
        _px = price_of(p["symbol"])
        if _px and _px > 0:
            p["last_price"] = round(float(_px), 6)

    max_pos_pct = float(s.get("max_position_pct", 20.0))
    cash_floor_pct = float(s.get("cash_floor_pct", 10.0))
    slip = float(s.get("slippage_bps", 5)) / 10_000.0
    min_conv = int(s.get("min_conviction_to_trade", 55))
    max_trades = len(orders) + 1 if liquidation else int(s.get("max_trades_per_tick", 4))
    max_new = int(s.get("max_new_positions_per_tick", 2))
    allow_crypto = bool(s.get("allow_crypto", False))          # direct spot (BTC-USD)
    allow_crypto_etf = bool(s.get("allow_crypto_etf", True))   # spot-crypto ETFs (IBIT/FBTC/FETH)
    allow_etf = bool(s.get("allow_etf", True))  # ETF filtering is best-effort (source-tagged upstream)
    respect_zones = bool(s.get("respect_entry_zones", True))
    # Brokerage realism. CASH accounts settle T+1 (US moved from T+2 on 2024-05-28), so proceeds from
    # today's sales cannot fund today's buys — reusing them is a good-faith violation that a real broker
    # would block. MARGIN accounts may reuse proceeds immediately but are subject to FINRA's pattern-day
    # -trader rule under $25k equity. (The once-a-day cadence means a same-day round trip is structurally
    # impossible, so PDT is currently moot — the guard exists so a faster cadence stays compliant.)
    account_type = str(s.get("account_type", "cash")).lower()
    is_cash_account = account_type != "margin"
    avoid_wash = bool(s.get("avoid_wash_sales", True))
    recent_loss_sales: dict[str, float] = {k.upper(): v for k, v in (blob.get("recent_loss_sales") or {}).items()}
    # Churn control: the max notional (buys+sells) allowed to change hands this tick, as a % of the
    # starting equity. 0 disables the cap. Computed off the pre-trade book so it's a stable budget.
    turnover_pct = 0.0 if liquidation else float(s.get("max_turnover_pct", 0.0) or 0.0)
    start_equity = cash0 + positions_value(positions, price_of)
    turnover_budget = (turnover_pct / 100.0 * start_equity) if turnover_pct > 0 else float("inf")
    traded_notional = 0.0

    filled: list[dict] = []
    skipped: list[dict] = []
    buy_notional = 0.0
    sell_notional = 0.0
    new_positions = 0

    def _skip(o: dict, why: str) -> None:
        skipped.append({"ts": now_ts, "date": today_et_str(), "symbol": o.get("symbol", "").upper(),
                        "side": o.get("side"), "status": "skipped", "shares": 0.0, "price": None,
                        "conviction": o.get("conviction"), "source": source, "reason": o.get("reason", ""),
                        "entry_low": o.get("entry_low"), "entry_high": o.get("entry_high"),
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
            "entry_low": o.get("entry_low"), "entry_high": o.get("entry_high"),
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
        room = turnover_budget - traded_notional
        if room <= 0:
            _skip(o, f"turnover cap ({turnover_pct:.0f}% of equity) reached"); continue
        want = float(o.get("shares") or 0) or (float(o.get("dollars") or 0) / px)
        want = min(want, room / px)   # clamp the trade to the remaining churn budget
        shares = round_shares(sym, min(want, pos["shares"]))
        if shares <= 0:
            _skip(o, "nothing to sell (or turnover cap left no room)"); continue
        fill = px * (1 - slip)
        traded_notional += shares * fill
        proceeds = shares * fill
        realized = shares * (fill - pos["avg_cost"])
        cash += proceeds
        sell_notional += proceeds
        pos["shares"] = round(pos["shares"] - shares, 8)
        b["realized_pl_total"] = round(float(b.get("realized_pl_total", 0.0)) + realized, 2)
        if realized < 0:              # a loss sale starts the 30-day wash-sale clock for this name
            recent_loss_sales[sym] = now_ts
        _fill(o, "sell", shares, fill, realized, cash, pos)
        if pos["shares"] <= 1e-9:
            positions.remove(pos)

    # ---- BUYS by conviction ----
    # Equity + per-group values are pinned AFTER sells so caps use one stable denominator for the tick.
    pv_after_sells = positions_value(positions, price_of)
    equity = round(cash + pv_after_sells, 2)
    floor = cash_floor_pct / 100.0 * equity
    # T+1: in a cash account today's sale proceeds are UNSETTLED and can't fund today's buys.
    buying_power = cash - sell_notional if is_cash_account else cash
    # Same marks as the equity figure above — if these two ever diverge, the cap is measured against
    # a different book than the one it is capping.
    group_value: dict[str, float] = {}
    for p in positions:
        px = mark_price(p, price_of)
        if px:
            g = group_of(p["symbol"])
            group_value[g] = group_value.get(g, 0.0) + p["shares"] * px

    for o in sorted(buys, key=lambda x: -int(x.get("conviction") or 0)):
        if len(filled) >= max_trades:
            _skip(o, "max_trades_per_tick reached"); continue
        sym = o["symbol"].upper()
        if exclude and (sym in exclude or sym.removesuffix("-USD") in exclude):
            _skip(o, "excluded ticker"); continue
        if is_crypto(sym) and not allow_crypto:
            _skip(o, "direct spot crypto disabled (use the ETF)"); continue
        # A non "-USD" symbol whose exposure group is BTC/ETH is a spot-crypto ETF (IBIT/FBTC/FETH…).
        if not is_crypto(sym) and group_of(sym) in ("BTC", "ETH") and not allow_crypto_etf:
            _skip(o, "crypto ETFs disabled"); continue
        if int(o.get("conviction") or 0) < min_conv:
            _skip(o, f"below conviction floor ({min_conv})"); continue
        px = price_of(sym)
        if not px:
            _skip(o, "no fresh price"); continue
        # Entry-zone discipline: don't chase. If the analyst named a zone and the market is above its
        # top, defer the buy — it stays a candidate on later ticks instead of filling at any price.
        # (Below the zone is fine: cheaper than it wanted to pay.)
        if respect_zones:
            hi = o.get("entry_high")
            try:
                hi = float(hi) if hi is not None else None
            except (TypeError, ValueError):
                hi = None
            if hi and hi > 0 and px > hi:
                _skip(o, f"price {px:.2f} above entry zone (≤{hi:.2f}) — waiting"); continue
        # Wash-sale guard (IRS §1091): rebuying a name sold at a LOSS within 30 days disallows the loss.
        if avoid_wash and sym in recent_loss_sales:
            days = (now_ts - float(recent_loss_sales[sym])) / 86_400.0
            if days < 30:
                _skip(o, f"wash-sale window ({30 - int(days)}d left since the loss sale)"); continue
        available = min(cash, buying_power) - floor
        if available <= 0:
            _skip(o, "unsettled proceeds (T+1) — funds free next session"
                  if buying_power < cash - 0.01 else "at cash floor")
            continue
        g = group_of(sym)
        cap_room = max_pos_pct / 100.0 * equity - group_value.get(g, 0.0)
        if cap_room <= 0:
            _skip(o, f"exposure '{g}' at {max_pos_pct:.0f}% cap"); continue
        is_new = _find(positions, sym) is None
        if is_new and new_positions >= max_new:
            _skip(o, "max_new_positions_per_tick reached"); continue
        room = turnover_budget - traded_notional
        if room <= 0:
            _skip(o, f"turnover cap ({turnover_pct:.0f}% of equity) reached"); continue
        spend = min(float(o.get("dollars") or 0) or available, available, cap_room, room)
        fill = px * (1 + slip)
        shares = round_shares(sym, spend / fill)
        if shares <= 0:
            _skip(o, "cash/cap/turnover left no whole share"); continue
        cost = shares * fill
        traded_notional += cost
        cash -= cost
        buying_power -= cost
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

    # Keep the wash-sale clock on the ledger, pruned to the 30-day window it governs.
    b["recent_loss_sales"] = {k: v for k, v in recent_loss_sales.items() if now_ts - float(v) < 31 * 86_400}

    # Cash-conservation invariant (fail-closed): the caller aborts + does not persist on violation.
    #
    # Checked on the RAW accumulators, not on two independently-rounded values. The previous form
    # compared round(cash,2) - cash0 against round(sell_notional - buy_notional, 2); when the exact
    # figure sat on a half-cent boundary those were both legal roundings of the SAME number that
    # differ by exactly 0.01, and the strict `< 0.01` then aborted a perfectly correct tick.
    # Slippage manufactures that third decimal and real quotes are round numbers, so the boundary is
    # hit often — measured at 1.6% of ticks, each one a lost trading day, a hole in the equity curve,
    # and once a month an orphaned deposit row already appended to the append-only log. `cash` and
    # the notionals accumulate the same floats, so the true invariant is exact to float noise.
    drift = abs(cash - (cash0 + sell_notional - buy_notional))
    assert drift < 1e-6, (
        f"cash not conserved: cash={cash!r} vs cash0+sells-buys="
        f"{cash0 + sell_notional - buy_notional!r} (drift={drift!r})"
    )
    b["cash"] = round(cash, 2)
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
