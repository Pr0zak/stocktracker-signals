"""
Congressional / political stock trades (AIE-2) — the public-official mirror of the insider block.

Source: the free, daily-refreshed kadoa-org/congress-trading-monitor dataset (House Clerk + Senate eFD
+ OGE, normalized to one JSON schema). We cache the whole ~4MB trades file to disk and index it by
ticker, then build a compact per-symbol block: recent buys/sells, notable filers, cluster + party.

Honest framing (baked into the analyst prompt, not here): the STOCK Act allows up to ~45 days between
a trade and its disclosure, so this is LAGGING and the "alpha" is weak/debated. The interesting signal
is committee relevance, cluster buying, and size — never the raw trade. Enrichment, never a blocker.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import time
from datetime import date
from pathlib import Path

import httpx

_DATA_DIR = Path(os.environ.get("SIGNALS_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))) / "congress"
_FILE = _DATA_DIR / "all_trades.json"
_URL = "https://raw.githubusercontent.com/kadoa-org/congress-trading-monitor/main/public/data/trades.json"
_REFRESH_S = 20 * 3600   # daily-ish; the upstream refreshes once a day and the data lags weeks anyway

_lock = asyncio.Lock()
_index: dict[str, list[dict]] | None = None
_index_at: float = 0.0


def _is_buy(t: dict) -> bool:
    return str(t.get("transaction_type") or "").lower().startswith("purchase")


def _is_sell(t: dict) -> bool:
    return str(t.get("transaction_type") or "").lower().startswith("sale")


def _filer(t: dict) -> str:
    return str(t.get("filer_name") or "").strip().rstrip(",").strip()


async def _load_raw(client: httpx.AsyncClient) -> list[dict]:
    """Return the full trades list, preferring a fresh disk cache, else fetching + caching. Falls back
    to a stale disk copy if the network fetch fails (better stale congress data than none)."""
    now = time.time()
    if _FILE.exists() and now - _FILE.stat().st_mtime < _REFRESH_S:
        try:
            return json.loads(_FILE.read_text())
        except Exception:  # noqa: BLE001 — corrupt cache falls through to a refetch
            pass
    try:
        r = await client.get(_URL, timeout=30, follow_redirects=True)
        r.raise_for_status()
        raw = r.json()
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        _FILE.write_text(json.dumps(raw))
        os.chmod(_FILE, 0o644)
        return raw
    except Exception:  # noqa: BLE001 — network/parse failure: serve a stale copy if we have one
        if _FILE.exists():
            try:
                return json.loads(_FILE.read_text())
            except Exception:  # noqa: BLE001
                pass
        raise


async def _ensure_index(client: httpx.AsyncClient) -> dict[str, list[dict]]:
    """Build (and cache in-process) a ticker → trades index, refreshed ~daily."""
    global _index, _index_at
    async with _lock:
        now = time.time()
        if _index is not None and now - _index_at < _REFRESH_S:
            return _index
        raw = await _load_raw(client)
        idx: dict[str, list[dict]] = {}
        for t in raw:
            tk = str(t.get("ticker") or "").upper()
            if tk and tk not in ("", "N/A", "--", "-"):
                idx.setdefault(tk, []).append(t)
        _index = idx
        _index_at = now
        return _index


def _has_cluster(buys: list[dict]) -> bool:
    """3+ DISTINCT politicians buying within any rolling 30-day window (mirrors insider clustering)."""
    dated = []
    for b in buys:
        d = str(b.get("transaction_date") or "")
        try:
            dated.append((date.fromisoformat(d), _filer(b)))
        except ValueError:
            continue
    dated.sort()
    for i, (d0, _) in enumerate(dated):
        names = {n for (d, n) in dated[i:] if (d - d0).days <= 30}
        if len(names) >= 3:
            return True
    return False


def _build_block(rows: list[dict], months: int) -> dict | None:
    cutoff = (dt.date.today() - dt.timedelta(days=months * 31)).isoformat()
    recent = [r for r in rows if str(r.get("transaction_date") or "") >= cutoff]
    if not recent:
        return None
    buys = [r for r in recent if _is_buy(r)]
    sells = [r for r in recent if _is_sell(r)]
    parties: dict[str, int] = {}
    for r in recent:
        p = r.get("party") or "?"
        parties[p] = parties.get(p, 0) + 1

    def _row(r: dict) -> dict:
        return {
            "filer": _filer(r),
            "party": r.get("party"),
            "chamber": r.get("chamber"),
            "side": "buy" if _is_buy(r) else ("sell" if _is_sell(r) else "other"),
            "amount": r.get("amount_range_label"),
            "transaction_date": r.get("transaction_date"),
            "filed_days_after": r.get("days_to_file"),
            "late": bool(r.get("is_late")),
        }

    latest = sorted(recent, key=lambda r: str(r.get("filing_date") or ""), reverse=True)[:3]
    largest_buy_high = max((int(b.get("amount_range_high") or 0) for b in buys), default=0)
    net = (
        "buying" if len(buys) > len(sells) * 1.5
        else "selling" if len(sells) > len(buys) * 1.5
        else "mixed" if buys and sells
        else "neutral"
    )
    return {
        "window_months": months,
        "trade_count": len(recent),
        "buy_count": len(buys),
        "sell_count": len(sells),
        "net_direction": net,
        "distinct_filers": len({_filer(r) for r in recent}),
        "cluster_buy": _has_cluster(buys),
        "largest_buy_amount_high": largest_buy_high,
        "parties": parties,
        "latest": [_row(r) for r in latest],
        "latest_filing_date": max((str(r.get("filing_date") or "") for r in recent), default=None),
    }


async def congress_trades(client: httpx.AsyncClient, symbol: str, months: int = 12) -> dict | None:
    """Compact congressional-trading block for one ticker over the last `months`, or None if there are
    no disclosed trades. Never raises — political-money context is enrichment, not a blocker."""
    try:
        idx = await _ensure_index(client)
    except Exception:  # noqa: BLE001
        return None
    rows = idx.get(symbol.upper())
    if not rows:
        return None
    try:
        return _build_block(rows, months)
    except Exception:  # noqa: BLE001
        return None


def compact(data: dict | None) -> dict | None:
    """Slim block for the analyst snapshot — only when there were actual disclosed trades."""
    if not data or data.get("trade_count", 0) == 0:
        return None
    return {
        "trade_count": data["trade_count"],
        "buy_count": data["buy_count"],
        "sell_count": data["sell_count"],
        "net_direction": data["net_direction"],
        "distinct_filers": data["distinct_filers"],
        "cluster_buy": data["cluster_buy"],
        "largest_buy_amount_high": data["largest_buy_amount_high"],
        "parties": data["parties"],
        "latest": data["latest"][:2],
    }
