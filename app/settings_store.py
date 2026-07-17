"""
Runtime settings, editable from the web UI and persisted to disk.

Precedence: values saved via the UI (data/settings.json) override the initial environment seed
(.env / systemd EnvironmentFile). This lets the API key and models be changed without redeploying.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

_DATA_DIR = Path(os.environ.get("SIGNALS_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
_FILE = _DATA_DIR / "settings.json"
_lock = threading.Lock()

_EDITABLE = ("anthropic_api_key", "deep_model", "scan_model", "verdict_ttl_seconds")


def _split(s: str) -> list[str]:
    return [t.strip().upper() for t in s.replace(",", " ").split() if t.strip()]


def _defaults() -> dict:
    return {
        "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
        "deep_model": os.environ.get("DEEP_MODEL", "claude-opus-4-8"),
        "scan_model": os.environ.get("SCAN_MODEL", "claude-haiku-4-5"),
        "verdict_ttl_seconds": int(os.environ.get("VERDICT_TTL_SECONDS", "14400")),
        "watchlist": _split(os.environ.get("WATCHLIST", "")),
        "crypto_watchlist": _split(os.environ.get("CRYPTO_WATCHLIST", "")),
    }


def _load() -> dict:
    cfg = _defaults()
    if _FILE.exists():
        try:
            cfg.update(json.loads(_FILE.read_text()))
        except Exception:  # noqa: BLE001 — a corrupt file falls back to env defaults
            pass
    return cfg


_current = _load()


def get() -> dict:
    with _lock:
        return dict(_current)


def update(patch: dict) -> dict:
    """Apply a partial update. Empty strings are treated as "leave unchanged" so a blank key field
    in the UI never wipes the stored key."""
    with _lock:
        for k in ("anthropic_api_key", "deep_model", "scan_model"):
            v = patch.get(k)
            if v is not None and str(v).strip() != "":
                _current[k] = str(v).strip()
        ttl = patch.get("verdict_ttl_seconds")
        if ttl is not None:
            _current["verdict_ttl_seconds"] = max(0, int(ttl))
        for k in ("watchlist", "crypto_watchlist"):
            v = patch.get(k)
            if v is not None:
                _current[k] = _split(v) if isinstance(v, str) else [str(s).strip().upper() for s in v]
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        _FILE.write_text(json.dumps(_current, indent=2))
        os.chmod(_FILE, 0o600)
        return dict(_current)
