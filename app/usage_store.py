"""
Append-only log of Claude API usage so the settings UI can show total tokens/cost and a trend.

Records land in data/usage.jsonl (one JSON object per line). data/ is untracked, so the history
survives self-updates. Deliberately simple: append on each real (non-cached) call, aggregate on read.
Day buckets use the machine's local time (the CT is set to the user's timezone), so "today" lines up
with the user's calendar day.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

_DATA_DIR = Path(os.environ.get("SIGNALS_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
_FILE = _DATA_DIR / "usage.jsonl"
_lock = threading.Lock()

_ZERO = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}


def record(usage: dict, *, symbol: str, kind: str) -> None:
    """Append one call's usage. `usage` is the dict returned by analyst._usage(); `kind` is one of
    "scan" (nightly), "signal" (on-demand cheap), or "deep" (on-demand Opus)."""
    row = {
        "ts": time.time(),
        "symbol": (symbol or "").upper(),
        "kind": kind,
        "model": usage.get("model", ""),
        "input_tokens": int(usage.get("input_tokens", 0)),
        "output_tokens": int(usage.get("output_tokens", 0)),
        "cache_read_tokens": int(usage.get("cache_read_tokens", 0)),
        "cost_usd": float(usage.get("cost_usd", 0.0)),
    }
    with _lock:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        with _FILE.open("a") as f:
            f.write(json.dumps(row) + "\n")


def _iter_rows():
    if not _FILE.exists():
        return
    for line in _FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:  # noqa: BLE001 — skip a corrupt/partial line
            continue


def summary(days: int = 30) -> dict:
    """All-time totals + a per-day (local-time) time series for the last `days` days."""
    total_in = total_out = total_calls = 0
    total_cost = 0.0
    by_model: dict[str, dict] = {}
    by_day: dict[str, dict] = {}
    with _lock:
        rows = list(_iter_rows())

    for r in rows:
        ti = int(r.get("input_tokens", 0))
        to = int(r.get("output_tokens", 0))
        c = float(r.get("cost_usd", 0.0))
        m = r.get("model") or "?"
        total_in += ti
        total_out += to
        total_cost += c
        total_calls += 1
        bm = by_model.setdefault(m, dict(_ZERO))
        bm["calls"] += 1
        bm["input_tokens"] += ti
        bm["output_tokens"] += to
        bm["cost_usd"] += c
        day = time.strftime("%Y-%m-%d", time.localtime(r.get("ts", 0)))
        bd = by_day.setdefault(day, dict(_ZERO))
        bd["calls"] += 1
        bd["input_tokens"] += ti
        bd["output_tokens"] += to
        bd["cost_usd"] += c

    now = time.time()
    series = []
    for i in range(days - 1, -1, -1):
        day = time.strftime("%Y-%m-%d", time.localtime(now - i * 86400))
        d = by_day.get(day, _ZERO)
        series.append({
            "date": day,
            "calls": d["calls"],
            "input_tokens": d["input_tokens"],
            "output_tokens": d["output_tokens"],
            "tokens": d["input_tokens"] + d["output_tokens"],
            "cost_usd": round(d["cost_usd"], 6),
        })

    return {
        "total_calls": total_calls,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_tokens": total_in + total_out,
        "total_cost_usd": round(total_cost, 6),
        "by_model": {m: {**v, "cost_usd": round(v["cost_usd"], 6)} for m, v in by_model.items()},
        "series": series,
    }
