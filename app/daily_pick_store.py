"""Persistence for the Daily Pick (DP-3): an append-only run log plus one small settings file.

`data/daily_picks.jsonl` holds one line per RUN. A forced re-run on the same ET date appends a second
line rather than rewriting the first, and readers take the LAST line for each date, so the log keeps
what was actually shown earlier in the day while the card shows the newest run.

`data/daily_pick_settings.json` holds the user's universe choice. It is separate from settings.json
because that file's editable keys are a published contract with the dashboard (settings_store._EDITABLE)
and this is one feature's preference.

The API process is the only writer; `_lock` serialises its threads.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

log = logging.getLogger("signals.daily_pick_store")

_DATA_DIR = Path(os.environ.get("SIGNALS_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
_RUNS = _DATA_DIR / "daily_picks.jsonl"
_SETTINGS = _DATA_DIR / "daily_pick_settings.json"
# Intraday re-checks. Separate from the run log on purpose: a re-check never replaces the morning pick
# and is never graded, so it must not be able to land in the file the grading reads.
_RECHECKS = _DATA_DIR / "daily_pick_rechecks.jsonl"
_lock = threading.Lock()

UNIVERSES = ("market", "watchlist")
_DEFAULT_SETTINGS = {"universe": "market"}


def append_run(row: dict) -> None:
    with _lock:
        _RUNS.parent.mkdir(parents=True, exist_ok=True)
        with _RUNS.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")


def _read_all() -> list[dict]:
    if not _RUNS.exists():
        return []
    out: list[dict] = []
    with _lock:
        text = _RUNS.read_text()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:  # a partial last line from a crash mid-write
            log.warning("daily_pick_store: skipping an unreadable line")
            continue
        if isinstance(row, dict) and row.get("date"):
            out.append(row)
    return out


def runs(limit: int = 60) -> list[dict]:
    """The latest run for each ET date, NEWEST FIRST."""
    latest: dict[str, dict] = {}
    for row in _read_all():
        latest[row["date"]] = row          # later lines win
    ordered = sorted(latest.values(), key=lambda r: r["date"], reverse=True)
    return ordered[:max(1, int(limit))]


def run_for(date: str) -> dict | None:
    for row in runs(limit=400):
        if row["date"] == date:
            return row
    return None


def append_recheck(row: dict) -> None:
    with _lock:
        _RECHECKS.parent.mkdir(parents=True, exist_ok=True)
        with _RECHECKS.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")


def latest_recheck(date: str) -> dict | None:
    """The newest re-check recorded for `date`, or None."""
    if not _RECHECKS.exists():
        return None
    with _lock:
        lines = _RECHECKS.read_text().splitlines()
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("date") == date:
            return row
    return None


def get_settings() -> dict:
    try:
        with _lock:
            data = json.loads(_SETTINGS.read_text()) if _SETTINGS.exists() else {}
    except (OSError, ValueError):
        log.warning("daily_pick_store: settings unreadable, using defaults")
        data = {}
    out = dict(_DEFAULT_SETTINGS)
    if data.get("universe") in UNIVERSES:
        out["universe"] = data["universe"]
    return out


def save_settings(patch: dict) -> dict:
    cur = get_settings()
    uni = patch.get("universe")
    if uni is not None:
        if uni not in UNIVERSES:
            raise ValueError(f"universe must be one of {', '.join(UNIVERSES)}")
        cur["universe"] = uni
    with _lock:
        _SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        tmp = _SETTINGS.with_suffix(f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(cur, indent=2))
        os.replace(tmp, _SETTINGS)
    return cur
