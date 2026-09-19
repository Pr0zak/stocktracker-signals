"""
Version/update-availability status for the CT deploy, exposed read-only via GET /api/version and the
settings UI's "Check for updates" button. Only reports something meaningful when the deploy directory
is a git checkout of the public repo; otherwise it degrades to "not a git checkout" (`git: False`).

OPS-6: this module used to also expose update() — fetch + `reset --hard origin/main` + restart the
service, wired to POST /api/update with no authentication at all. It's gone: the container is
deployed by rsync, not git (deploy/README.md has said so since 2026-08-21), so calling it would have
rolled the working tree backwards past every un-pushed commit and discarded untracked local state.
Deploy with the rsync skill/README instead.
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
