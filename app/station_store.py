"""The Station tab's layout: which widgets the user placed where, saved on the server so the same
station shows in every browser (Zote ytik5yyk, 2026-11-18).

One JSON file under the data dir, written atomically. The page owns the layout; this module only
checks its shape so a bad POST cannot store something the page would choke on, and caps its size
so the file stays small. Unknown widget types are refused rather than kept: a type the page does
not know would render as a blank tile nobody can configure.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

_DATA_DIR = Path(os.environ.get("SIGNALS_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
_FILE = _DATA_DIR / "station_layout.json"
_lock = threading.Lock()

TYPES = ("quote", "chart", "watchlist", "gate", "pick", "movers", "brief", "arm", "race")
COLS = 12
MAX_ROWS = 60
MAX_WIDGETS = 40
MAX_BYTES = 32_000
_ID = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def _int(v, lo: int, hi: int, name: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise ValueError(f"{name} must be a whole number from {lo} to {hi}")
    return v


def _cfg(c) -> dict:
    """Widget settings: a flat dict of short strings, numbers, booleans or lists of short strings."""
    if c is None:
        return {}
    if not isinstance(c, dict) or len(c) > 12:
        raise ValueError("a widget's settings must be an object with at most 12 keys")
    out = {}
    for k, v in c.items():
        if not isinstance(k, str) or not _ID.match(k):
            raise ValueError("settings keys are short names")
        if isinstance(v, (bool, int, float)) or v is None:
            out[k] = v
        elif isinstance(v, str) and len(v) <= 200:
            out[k] = v
        elif isinstance(v, list) and len(v) <= 60 and all(isinstance(s, str) and len(s) <= 24 for s in v):
            out[k] = list(v)
        else:
            raise ValueError(f"setting {k!r} is not a short value")
    return out


def validate(body) -> dict:
    if not isinstance(body, dict):
        raise ValueError("the layout must be an object")
    name = body.get("name") or "My station"
    if not isinstance(name, str) or len(name) > 60:
        raise ValueError("the station name is at most 60 characters")
    ws = body.get("widgets")
    if not isinstance(ws, list) or len(ws) > MAX_WIDGETS:
        raise ValueError(f"widgets must be a list of at most {MAX_WIDGETS}")
    seen, out = set(), []
    for w in ws:
        if not isinstance(w, dict):
            raise ValueError("each widget must be an object")
        wid = w.get("id")
        if not isinstance(wid, str) or not _ID.match(wid) or wid in seen:
            raise ValueError("each widget needs a unique short id")
        seen.add(wid)
        if w.get("type") not in TYPES:
            raise ValueError(f"unknown widget type {w.get('type')!r}")
        x = _int(w.get("x"), 0, COLS - 1, "x")
        width = _int(w.get("w"), 1, COLS, "w")
        if x + width > COLS:
            raise ValueError("a widget runs past the right edge")
        out.append({"id": wid, "type": w["type"], "x": x, "y": _int(w.get("y"), 0, MAX_ROWS, "y"),
                    "w": width, "h": _int(w.get("h"), 1, 12, "h"), "cfg": _cfg(w.get("cfg"))})
    doc = {"name": name.strip() or "My station", "widgets": out}
    if len(json.dumps(doc)) > MAX_BYTES:
        raise ValueError("the layout is too large")
    return doc


def load() -> dict | None:
    """The saved layout, or None when none has been saved (the page then shows its starter layout)."""
    try:
        return json.loads(_FILE.read_text())
    except (OSError, ValueError):
        return None


def save(body) -> dict:
    doc = validate(body)
    doc["saved_at"] = time.time()
    with _lock:
        _FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _FILE.with_suffix(f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(doc, indent=1))
        os.replace(tmp, _FILE)
    return doc
