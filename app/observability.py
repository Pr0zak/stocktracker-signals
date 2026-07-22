"""
Operations + transparency layer for the Signals service.

Everything here is best-effort and side-effect-light: an in-memory request ring (no disk), a process
start-time for uptime, cheap read-only snapshots of the on-disk artifacts the service already writes
(scan_latest.json, usage.jsonl, iv_history.jsonl, data/shorts/), and short-timeout liveness probes of
the upstream data sources. Missing/corrupt files degrade to nulls — the helpers never raise.

The web page (`GET /`) turns these into the status header, latest-scan table, data-sources panel,
cost card, and recent-activity log. Nothing here touches an LLM or blocks a request.
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import re
import os
import time
from pathlib import Path

import httpx

from . import market, options, settings_store, shorts, webull

# --- canonical on-disk locations (mirror where each writer actually writes) ---------------
# usage/iv/shorts honor SIGNALS_DATA_DIR like their owning modules; scan_latest.json is written by
# scan_job.py to the repo's data/ dir unconditionally, so we read it from the same fixed place.
_DATA_DIR = Path(os.environ.get("SIGNALS_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
USAGE_FILE = _DATA_DIR / "usage.jsonl"
IV_HISTORY = _DATA_DIR / "iv_history.jsonl"
SHORTS_DIR = _DATA_DIR / "shorts"
SCAN_LATEST = Path(__file__).resolve().parent.parent / "data" / "scan_latest.json"

# Nightly scan fires at this local time in this zone (see the systemd timer / scan_job).
SCAN_HOUR, SCAN_MINUTE = 6, 30
SCAN_TZ_NAME = "America/Chicago"

# Files under data/shorts/ that the prune button may delete (whole-market day/period caches).
# NEVER settings.json / scan_latest.json / usage.jsonl / iv_history.jsonl — none live here anyway.
_PRUNABLE_PREFIXES = ("shvol_", "ftd_")
_PRUNE_MAX_AGE_DAYS = 90


# ==========================================================================================
# Request ring buffer (in-memory only — the middleware calls record_request on every request)
# ==========================================================================================

_MAX_REQUESTS = 200
_requests: collections.deque = collections.deque(maxlen=_MAX_REQUESTS)
_STARTED_AT = time.time()

# Redact secret-bearing query params (?token=, crumb=, api_key=, …) from any diagnostic string that
# could reach the unauthenticated /api/logs or /api/sources — belt-and-suspenders even though no
# current path leaks one (probe errors don't carry the URL; handlers catch their own errors).
_SECRET_RE = re.compile(r"(?i)\b(token|crumb|api[_-]?key|apikey|key|password|secret)=[^\s&'\"]+")


def _redact(s: str) -> str:
    return _SECRET_RE.sub(r"\1=***", str(s))


def _num(v, default: float = 0.0) -> float:
    """Coerce a possibly type-malformed JSON field to float; never raises."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def record_request(method: str, path: str, status: int, ms: float, error: str | None = None) -> None:
    """Append one served request to the ring. Cheap (a deque append under the GIL) — must never do
    disk I/O or slow the request path."""
    _requests.append({
        "ts": time.time(),
        "method": str(method),
        "path": str(path),
        "status": int(status),
        "ms": round(float(ms), 1),
        "error": (_redact(str(error))[:200] if error else None),
    })


def recent(limit: int = 50) -> list[dict]:
    """The last `limit` served requests, newest first."""
    items = list(_requests)
    if limit > 0:
        items = items[-limit:]
    return items[::-1]


def recent_errors(limit: int = 20) -> list[dict]:
    """The last `limit` non-2xx (or raised) requests, newest first — the ring filtered, no separate
    store to keep in sync."""
    errs = [r for r in _requests if r.get("error") or int(r.get("status", 0)) >= 400]
    if limit > 0:
        errs = errs[-limit:]
    return errs[::-1]


def uptime_seconds() -> float:
    return time.time() - _STARTED_AT


# ==========================================================================================
# Small file helpers (tolerant JSONL/JSON readers — never raise)
# ==========================================================================================

def _iter_jsonl(path: Path):
    if not path.exists():
        return
    try:
        lines = path.read_text().splitlines()
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


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


# ==========================================================================================
# Next nightly scan time (06:30 America/Chicago)
# ==========================================================================================

def _scan_tz():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(SCAN_TZ_NAME)
    except Exception:  # noqa: BLE001 — no tzdata: fall back to naive local time
        return None


def next_scan_at(now: float | None = None) -> float:
    """Epoch seconds of the next 06:30 America/Chicago boundary (today's if still ahead, else
    tomorrow's). Falls back to the host's local 06:30 if tz data is unavailable."""
    now = time.time() if now is None else now
    tz = _scan_tz()
    base = dt.datetime.fromtimestamp(now, tz) if tz else dt.datetime.fromtimestamp(now)
    target = base.replace(hour=SCAN_HOUR, minute=SCAN_MINUTE, second=0, microsecond=0)
    if base >= target:
        target = target + dt.timedelta(days=1)
    return target.timestamp()


# ==========================================================================================
# status_snapshot — uptime, disk, last scan, next scan, cache footprint, iv progress
# ==========================================================================================

def _disk() -> dict:
    try:
        import shutil
        u = shutil.disk_usage("/")
        return {
            "used": u.used,
            "total": u.total,
            "free": u.free,
            "pct": round(u.used / u.total * 100.0, 1) if u.total else None,
        }
    except Exception:  # noqa: BLE001
        return {"used": None, "total": None, "free": None, "pct": None}


# scan signal.value buckets (analyst.Signal): strong_buy/buy -> buy, hold -> hold, sell/strong_sell -> sell
_BUY = {"strong_buy", "buy"}
_SELL = {"strong_sell", "sell"}


def _scan_summary() -> dict:
    data = _read_json(SCAN_LATEST)
    if not data:
        return {"generated_at": None, "count": 0, "counts": {"buy": 0, "hold": 0, "sell": 0},
                "total_cost": None, "changed": [], "symbols": [], "errors": []}
    results = data.get("results") or []
    counts = {"buy": 0, "hold": 0, "sell": 0}
    changed: list[str] = []
    symbols: list[str] = []
    errors: list[str] = []
    for r in results:
        sym = r.get("symbol")
        if sym:
            symbols.append(sym)
        if r.get("error"):
            if sym:
                errors.append(sym)
            continue
        sig = r.get("signal")
        if sig in _BUY:
            counts["buy"] += 1
        elif sig in _SELL:
            counts["sell"] += 1
        elif sig is not None:
            counts["hold"] += 1
        if sym and (r.get("flipped") or r.get("dip_new") or r.get("crossed_below_200wma")):
            changed.append(sym)
    total_cost = data.get("total_cost_usd")
    if total_cost is None:
        total_cost = round(sum(_num(r.get("cost_usd")) for r in results), 6)
    else:
        total_cost = _num(total_cost)
    return {
        "generated_at": data.get("generated_at"),
        "count": len([r for r in results if not r.get("error")]),
        "counts": counts,
        "total_cost": total_cost,
        "changed": changed,
        "symbols": symbols,
        "errors": errors,
    }


def _cache_footprint() -> dict:
    shorts_bytes = 0
    shorts_files = 0
    if SHORTS_DIR.exists():
        for p in SHORTS_DIR.iterdir():
            try:
                if p.is_file():
                    shorts_files += 1
                    shorts_bytes += p.stat().st_size
            except Exception:  # noqa: BLE001
                continue
    iv_days = sum(1 for _ in _iter_jsonl(IV_HISTORY))
    return {"shorts_bytes": shorts_bytes, "shorts_files": shorts_files, "iv_history_days_total": iv_days}


def iv_rank_progress(target: int | None = None) -> dict:
    """Per-symbol count of ATM-IV points accrued in iv_history.jsonl, plus the ~20-day target the
    /options IV-rank read needs, so the UI can show 'building N/20'."""
    if target is None:
        target = getattr(options, "IV_RANK_MIN_POINTS", 20)
    by_symbol: dict[str, int] = {}
    for row in _iter_jsonl(IV_HISTORY):
        sym = row.get("symbol")
        if sym:
            by_symbol[sym] = by_symbol.get(sym, 0) + 1
    return {"target": target, "symbols": dict(sorted(by_symbol.items()))}


def status_snapshot() -> dict:
    """A cheap, best-effort operations snapshot. Does a shutil.disk_usage + a few small file reads;
    never raises."""
    now = time.time()
    return {
        "now": now,
        "started_at": _STARTED_AT,
        "uptime_s": round(uptime_seconds(), 1),
        "disk": _disk(),
        "scan": _scan_summary(),
        "next_scan_at": next_scan_at(now),
        "cache": _cache_footprint(),
        "iv_progress": iv_rank_progress(),
    }


# ==========================================================================================
# cost_breakdown — read data/usage.jsonl, aggregate by kind + month math
# ==========================================================================================

def cost_breakdown(now: float | None = None) -> dict:
    """Aggregate the usage log: per-kind calls/tokens/$$, month-to-date + naive month projection,
    per-scan / per-deep averages, all-time. Day/month buckets use local time to match usage_store."""
    now = time.time() if now is None else now
    lt = time.localtime(now)
    month_prefix = time.strftime("%Y-%m", lt)

    by_kind: dict[str, dict] = {}
    all_time = 0.0
    mtd = 0.0
    for r in _iter_jsonl(USAGE_FILE):
        try:
            kind = r.get("kind") or "?"
            cost = float(r.get("cost_usd", 0.0) or 0.0)
            tokens = int(r.get("input_tokens", 0) or 0) + int(r.get("output_tokens", 0) or 0)
            ts = float(r.get("ts", 0) or 0)
        except (TypeError, ValueError):
            continue  # a type-malformed but valid-JSON row degrades to skipped, never 500s the card
        k = by_kind.setdefault(kind, {"calls": 0, "tokens": 0, "usd": 0.0})
        k["calls"] += 1
        k["tokens"] += tokens
        k["usd"] += cost
        all_time += cost
        if time.strftime("%Y-%m", time.localtime(ts)) == month_prefix:
            mtd += cost

    day_of_month = lt.tm_mday
    import calendar
    days_in_month = calendar.monthrange(lt.tm_year, lt.tm_mon)[1]
    projected = (mtd / day_of_month * days_in_month) if day_of_month else mtd

    def _avg(kind: str) -> float | None:
        k = by_kind.get(kind)
        return round(k["usd"] / k["calls"], 6) if (k and k["calls"]) else None

    return {
        "by_kind": {k: {"calls": v["calls"], "tokens": v["tokens"], "usd": round(v["usd"], 6)}
                    for k, v in sorted(by_kind.items())},
        "month_to_date_usd": round(mtd, 6),
        "projected_month_usd": round(projected, 6),
        "per_scan_avg_usd": _avg("scan"),
        "per_deep_avg_usd": _avg("deep"),
        "all_time_usd": round(all_time, 6),
    }


# ==========================================================================================
# prune_shorts_cache — delete stale whole-market caches under data/shorts/
# ==========================================================================================

def prune_shorts_cache(max_age_days: int = _PRUNE_MAX_AGE_DAYS, now: float | None = None) -> dict:
    """Delete stale shvol_*/ftd_* files under data/shorts/ (by mtime, older than `max_age_days`) and
    report bytes freed. Only touches those two prefixes inside data/shorts/ — never settings.json,
    scan_latest.json, usage.jsonl, or iv_history.jsonl (which live outside this dir anyway)."""
    now = time.time() if now is None else now
    cutoff = now - max_age_days * 86400
    deleted = 0
    freed = 0
    if SHORTS_DIR.exists():
        for p in SHORTS_DIR.iterdir():
            try:
                if not p.is_file() or not p.name.startswith(_PRUNABLE_PREFIXES):
                    continue
                st = p.stat()
                if st.st_mtime < cutoff:
                    size = st.st_size
                    p.unlink()
                    deleted += 1
                    freed += size
            except Exception:  # noqa: BLE001 — best-effort; skip anything we can't stat/remove
                continue
    return {"deleted_files": deleted, "bytes_freed": freed, "max_age_days": max_age_days}


# ==========================================================================================
# probe_sources — concurrent, short-timeout liveness checks of the upstream data sources
# ==========================================================================================

import asyncio  # noqa: E402 — kept beside the async probes it serves

_PROBE_TIMEOUT = 3.5   # per-source cap
_PROBE_BUDGET = 8.0    # whole-panel backstop


def _short(s: str, n: int = 120) -> str:
    s = _redact(" ".join(str(s).split()))
    return s[:n]


def _recent_business_day(now: float | None = None) -> str:
    d = dt.date.fromtimestamp(time.time() if now is None else now) - dt.timedelta(days=1)
    while d.weekday() >= 5:  # walk back over the weekend
        d -= dt.timedelta(days=1)
    return d.strftime("%Y%m%d")


async def _measure(name: str, coro) -> dict:
    """Run one probe coroutine (which returns `(detail, status)`), time it, and never raise."""
    t0 = time.perf_counter()
    try:
        detail, status = await asyncio.wait_for(coro, _PROBE_TIMEOUT)
    except asyncio.TimeoutError:
        return {"name": name, "status": "down", "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "detail": f"timeout >{_PROBE_TIMEOUT:.0f}s"}
    except Exception as e:  # noqa: BLE001 — any transport/parse error is a 'down', not a crash
        return {"name": name, "status": "down", "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "detail": _short(str(e))}
    return {"name": name, "status": status, "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "detail": _short(detail)}


async def _p_yahoo_chart(client: httpx.AsyncClient):
    data = await market._fetch_chart(client, "BTC-USD", rng="1d")
    res = (data.get("chart") or {}).get("result") or []
    price = (res[0].get("meta") or {}).get("regularMarketPrice") if res else None
    return (f"BTC-USD chart ok (spot {price})" if price else "chart ok", "ok")


async def _p_coingecko(client: httpx.AsyncClient):
    r = await client.get("https://api.coingecko.com/api/v3/ping", timeout=_PROBE_TIMEOUT)
    return (f"HTTP {r.status_code}", "ok" if r.status_code == 200 else "warn")


async def _p_finnhub(client: httpx.AsyncClient):
    key = settings_store.get().get("finnhub_api_key", "")
    if not key:
        return ("no key configured — news/earnings context off", "warn")
    r = await client.get("https://finnhub.io/api/v1/quote",
                         params={"symbol": "AAPL", "token": key}, timeout=_PROBE_TIMEOUT)
    if r.status_code == 200:
        try:
            c = r.json().get("c")
        except Exception:  # noqa: BLE001
            c = None
        if isinstance(c, (int, float)) and c > 0:
            return (f"quote ok (AAPL {c})", "ok")
        return ("200 but empty quote", "warn")
    return (f"HTTP {r.status_code}", "warn")


async def _p_finra(client: httpx.AsyncClient):
    day = _recent_business_day()
    url = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{day}.txt"
    r = await client.head(url, headers=shorts._UA, timeout=_PROBE_TIMEOUT, follow_redirects=True)
    if r.status_code == 200:
        return (f"{day} short-vol file present", "ok")
    if r.status_code in (403, 404, 405):  # reachable, just not that exact file/method
        return (f"reachable (HTTP {r.status_code} for {day})", "warn")
    return (f"HTTP {r.status_code}", "warn")


async def _p_sec(client: httpx.AsyncClient):
    periods = shorts._ftd_periods(1)
    if periods:
        url = f"https://www.sec.gov/files/data/fails-deliver-data/cnsfails{periods[0]}.zip"
        label = f"FTD {periods[0]}"
    else:
        url = "https://www.sec.gov/"
        label = "sec.gov"
    r = await client.head(url, headers=shorts._SEC_UA, timeout=_PROBE_TIMEOUT, follow_redirects=True)
    if r.status_code == 403:
        return ("HTTP 403 (WAF/UA rejected)", "warn")
    if r.status_code < 400:
        return (f"{label} reachable (HTTP {r.status_code})", "ok")
    if r.status_code == 404:
        return (f"reachable ({label} HTTP 404)", "warn")
    return (f"HTTP {r.status_code}", "warn")


async def _p_webull(client: httpx.AsyncClient):
    r = await client.get(webull._SEARCH,
                         params={"keyword": "AAPL", "pageIndex": 1, "pageSize": 1, "regionId": 6},
                         headers=webull._HDRS, timeout=_PROBE_TIMEOUT)
    return (f"HTTP {r.status_code}", "ok" if r.status_code == 200 else "warn")


def _crumb_source() -> dict:
    """Yahoo options crumb cache state — no network, just whether the handshake has been primed."""
    try:
        cached = bool(options.crumb_status().get("cached"))
    except Exception:  # noqa: BLE001
        cached = False
    return {
        "name": "Yahoo options (crumb)",
        "status": "ok" if cached else "warn",
        "latency_ms": 0.0,
        "detail": "crumb cached" if cached else "crumb not primed yet (fetched on first options call)",
    }


async def probe_sources(client: httpx.AsyncClient) -> list[dict]:
    """Concurrently probe every upstream data source with short timeouts and a whole-panel budget.
    Returns [{name, status: 'ok'|'warn'|'down', latency_ms, detail}]. Never raises."""
    tasks = [
        _measure("Yahoo chart", _p_yahoo_chart(client)),
        _measure("CoinGecko", _p_coingecko(client)),
        _measure("Finnhub", _p_finnhub(client)),
        _measure("FINRA short-vol", _p_finra(client)),
        _measure("SEC FTD", _p_sec(client)),
        _measure("Webull", _p_webull(client)),
    ]
    try:
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), _PROBE_BUDGET)
    except asyncio.TimeoutError:
        results = []
    out: list[dict] = [_crumb_source()]
    for r in results:
        if isinstance(r, dict):
            out.append(r)
        elif isinstance(r, BaseException):  # a _measure that somehow raised — record as down
            out.append({"name": "unknown", "status": "down", "latency_ms": None, "detail": _short(str(r))})
    return out
