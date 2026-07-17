"""
Self-update for the CT deploy: check the git remote and pull + restart on demand (exposed as the
"Check for updates / Update" buttons in the settings UI). Only works when the deploy directory is
a git checkout of the public repo; otherwise it degrades to "not a git checkout".
"""
from __future__ import annotations

import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_VERSION_FILE = _ROOT / "VERSION"


def _git(*args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(_ROOT), *args], capture_output=True, text=True, timeout=timeout
    )


def version() -> str:
    try:
        return _VERSION_FILE.read_text().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def status() -> dict:
    if _git("rev-parse", "--is-inside-work-tree").returncode != 0:
        return {"version": version(), "git": False, "update_available": False, "behind": 0}
    _git("fetch", "--quiet", "origin", "main", timeout=30)
    behind = _git("rev-list", "--count", "HEAD..origin/main").stdout.strip()
    n = int(behind) if behind.isdigit() else 0
    return {
        "version": version(),
        "git": True,
        "update_available": n > 0,
        "behind": n,
        "local": _git("rev-parse", "--short", "HEAD").stdout.strip(),
        "remote": _git("rev-parse", "--short", "origin/main").stdout.strip(),
    }


def update() -> dict:
    before = _git("rev-parse", "--short", "HEAD").stdout.strip()
    _git("fetch", "origin", "main", timeout=60)
    reset = _git("reset", "--hard", "origin/main", timeout=30)
    after = _git("rev-parse", "--short", "HEAD").stdout.strip()
    # Reinstall deps (idempotent), then schedule a restart 2s out so this HTTP response returns first.
    subprocess.run(
        [str(_ROOT / ".venv" / "bin" / "pip"), "install", "-q", "-r", str(_ROOT / "requirements.txt")],
        capture_output=True, text=True, timeout=180,
    )
    subprocess.run(
        ["systemd-run", "--quiet", "--on-active=2", "systemctl", "restart", "signals"],
        capture_output=True, text=True, timeout=10,
    )
    return {"ok": reset.returncode == 0, "from": before, "to": after, "version": version(), "restarting": True}
