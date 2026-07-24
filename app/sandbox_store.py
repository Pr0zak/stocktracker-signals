"""
Persistence for the AI paper-trading sandbox.

Two shapes, mirroring the rest of the service:
  • data/sandbox.json         — the mutable ledger (cash + positions + settings + cursors), a single
                                blob fully read / fully rewritten as one consistent snapshot. Written
                                ATOMICALLY (temp file + fsync + os.replace) with a .bak of the prior
                                blob, so a crash mid-write can never corrupt the account or re-zero it.
  • data/sandbox_trades.jsonl — append-only audit log, one row per filled OR skipped order.
  • data/sandbox_nav.jsonl    — append-only equity curve, one row per tick (incl. the benchmark).

data/ is gitignored, so all of this survives the git-reset self-update. All mutations to the blob are
serialized by callers via an asyncio lock in main.py (single uvicorn worker = sole writer); the
threading.Lock here only guards the file I/O itself.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import threading
import time
from pathlib import Path

_DATA_DIR = Path(os.environ.get("SIGNALS_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
_FILE = _DATA_DIR / "sandbox.json"
_BAK = _DATA_DIR / "sandbox.json.bak"
_TRADES = _DATA_DIR / "sandbox_trades.jsonl"
_NAV = _DATA_DIR / "sandbox_nav.jsonl"
_lock = threading.Lock()

VERSION = 1

# Defaults are conservative — the sandbox does nothing until it's funded and turned on.
DEFAULT_SETTINGS = {
    "master_enabled": False,           # off until the user opts in
    "risk_tolerance": "balanced",      # conservative | balanced | aggressive
    "retirement_date": None,           # ISO yyyy-mm-dd; biases the glidepath (de-risk as it nears)
    "exit_date": None,                 # ISO yyyy-mm-dd; flatten to cash by this date
    "max_position_pct": 20.0,          # cap per EXPOSURE GROUP (BTC+FBTC count as one)
    "cash_floor_pct": 10.0,            # never deploy below this % of equity
    "allow_crypto": True,
    "allow_etf": True,
    "slippage_bps": 5,                 # modeled fill slippage (buys up, sells down)
    "max_trades_per_tick": 4,
    "max_new_positions_per_tick": 2,
    "min_conviction_to_trade": 55,     # 0-100 floor for a buy
}


def _defaults() -> dict:
    return {
        "version": VERSION,
        "created_at": None,
        "funded_total": 0.0,
        "cash": 0.0,
        "realized_pl_total": 0.0,
        "positions": [],   # [{symbol, shares, avg_cost, exposure_group, opened_at, last_add_at}]
        "benchmark": {"symbol": "^GSPC", "shares": 0.0, "cost_basis": 0.0},
        "settings": dict(DEFAULT_SETTINGS),
        "last_tick_date": None,           # ET yyyy-mm-dd — idempotency cursor
        "last_weekly_review_date": None,
        "last_strategy_note": None,
    }


def _load() -> dict:
    """Load the ledger, falling back to the .bak (last-known-good) then to fresh defaults. Settings are
    merged over DEFAULT_SETTINGS so a new setting key always has a value."""
    for p in (_FILE, _BAK):
        if p.exists():
            try:
                blob = _defaults()
                blob.update(json.loads(p.read_text()))
                blob["settings"] = {**DEFAULT_SETTINGS, **(blob.get("settings") or {})}
                return blob
            except Exception:  # noqa: BLE001 — try the .bak, then defaults
                continue
    return _defaults()


_current = _load()


def get() -> dict:
    """A deep copy of the current ledger (safe to mutate by the caller before save())."""
    with _lock:
        return copy.deepcopy(_current)


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(tmp, 0o600)
    if path.exists():
        try:
            shutil.copy2(path, _BAK)   # last-known-good before we overwrite
        except Exception:  # noqa: BLE001
            pass
    os.replace(tmp, path)              # atomic on a single filesystem


def save(blob: dict) -> dict:
    """Atomically persist the ledger and update the in-memory copy. Returns the saved blob."""
    global _current
    with _lock:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(_FILE, blob)
        _current = copy.deepcopy(blob)
        return copy.deepcopy(_current)


def _append_jsonl(path: Path, row: dict) -> None:
    with _lock:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")


def append_trade(row: dict) -> None:
    _append_jsonl(_TRADES, row)


def append_nav(row: dict) -> None:
    _append_jsonl(_NAV, row)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:  # noqa: BLE001 — skip a corrupt/partial line
            continue
    return out


def read_trades(limit: int = 200) -> list[dict]:
    """Most recent trade rows first (filled + skipped)."""
    rows = _read_jsonl(_TRADES)
    return list(reversed(rows))[: max(1, limit)]


def read_nav(days: int | None = None) -> list[dict]:
    """The NAV/equity series oldest-first; optionally only the last `days` calendar days."""
    rows = _read_jsonl(_NAV)
    if days:
        cutoff = time.time() - days * 86400
        rows = [r for r in rows if float(r.get("ts", 0)) >= cutoff]
    return rows


def reset() -> dict:
    """Wipe the ledger back to fresh defaults and rotate the append-only logs to .bak (never truncated
    in place — the history is preserved on disk). Returns the fresh blob."""
    with _lock:
        for p in (_TRADES, _NAV):
            if p.exists():
                try:
                    p.replace(p.with_suffix(p.suffix + f".bak.{int(time.time())}"))
                except Exception:  # noqa: BLE001
                    pass
    return save(_defaults())
