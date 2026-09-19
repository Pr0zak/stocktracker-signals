"""
Runtime settings, editable from the web UI and persisted to disk.

Precedence: values saved via the UI (data/settings.json) override the initial environment seed
(.env / systemd EnvironmentFile). This lets the API key and models be changed without redeploying.

OPS-5: the file is written ~70x/day and is the sole home of the Finnhub key, the Claude CLI OAuth
token, llm_provider, deep_model and the watchlist — a crash mid-write must not lose it. Writes go
through _atomic_write_json (temp file + fsync + os.replace, keeping a .bak of the prior good
content), the exact technique app/sandbox_store.py uses for its own ledger file. A parse failure on
load no longer falls back to environment defaults in silence: it tries .bak, repairs settings.json
from it if that parses, and only degrades to env defaults (loudly logged) if both are unreadable.
`source()` reports which of the three actually backs the current in-memory settings, surfaced on
/health as `settings_source` so the dashboard can show the degraded state instead of looking like a
normally-configured backend running on empty keys.

OPS-3: watchlist syncs are otherwise last-writer-wins with no notion of who sent them. A debug client
(e.g. an emulator on the LAN) POSTing a stale, short watchlist can silently overwrite the real one
right before the nightly scan reads it — see the 2026-09-11 incident, 54 symbols culled to 14 with no
log line and no diff. update() now takes the caller's client_id, refuses (WatchlistSyncRefused, HTTP
409 at the route) a sync that would strip more than watchlist_removal_guard names when the sender
isn't the client who wrote the current list, and logs n_before -> n_after plus the removed symbols on
every accepted change so a repeat is at least visible.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path

_DATA_DIR = Path(os.environ.get("SIGNALS_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
_FILE = _DATA_DIR / "settings.json"
_BAK = _FILE.with_suffix(_FILE.suffix + ".bak")
_CORRUPT = _FILE.with_suffix(_FILE.suffix + ".corrupt")
_lock = threading.Lock()
_log = logging.getLogger(__name__)

_EDITABLE = ("anthropic_api_key", "deep_model", "scan_model", "verdict_ttl_seconds", "llm_provider",
             "cli_oauth_token")

# settings_source values — see module docstring (OPS-5).
SOURCE_FILE = "file"
SOURCE_BACKUP = "backup"
SOURCE_ENV = "env"

# Default guard for OPS-3: a sync from a DIFFERENT client than the one who wrote the current
# watchlist is refused if it would remove more than this many symbols. A same-client prune (the
# normal case — a user editing their own list) is never refused, at any size.
_DEFAULT_REMOVAL_GUARD = 5


class WatchlistSyncRefused(Exception):
    """Raised by update() when a watchlist/crypto_watchlist sync trips the OPS-3 removal guard.
    Carries everything the HTTP layer needs to build the 409 body without re-deriving it."""

    def __init__(self, field: str, n_before: int, n_after: int, removed: list[str], threshold: int):
        self.field = field
        self.n_before = n_before
        self.n_after = n_after
        self.removed = removed
        self.threshold = threshold
        super().__init__(
            f"refused {field} sync: would remove {len(removed)} symbols (n_before={n_before}, "
            f"n_after={n_after}, threshold={threshold}) from a client that didn't write the current "
            "list; resend with replace=true to force"
        )

    def detail(self) -> dict:
        return {
            "error": "watchlist_sync_refused",
            "field": self.field,
            "n_before": self.n_before,
            "n_after": self.n_after,
            "removed_count": len(self.removed),
            "removed": self.removed,
            "threshold": self.threshold,
            "message": str(self),
        }


def _split(s: str) -> list[str]:
    return [t.strip().upper() for t in s.replace(",", " ").split() if t.strip()]


def _defaults() -> dict:
    return {
        "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
        "finnhub_api_key": os.environ.get("FINNHUB_API_KEY", ""),
        "deep_model": os.environ.get("DEEP_MODEL", "claude-opus-4-8"),
        "scan_model": os.environ.get("SCAN_MODEL", "claude-haiku-4-5"),
        # Which LLM backend the analyst uses: "api" (Anthropic SDK, per-token billing) or "cli"
        # (headless `claude` CLI on the machine's subscription OAuth — no per-token cost). See llm_cli.
        "llm_provider": (os.environ.get("LLM_PROVIDER", "api").strip().lower() or "api"),
        # CLI subscription token (`claude setup-token`) for cli mode, editable in the UI. Empty here
        # means "fall back to the CLAUDE_CODE_OAUTH_TOKEN env var"; a UI-set value takes precedence.
        "cli_oauth_token": "",
        # When llm_provider is "cli" and the headless CLI hits its subscription session/budget limit
        # (see llm_cli.CliBudgetExhaustedError) — as opposed to a transient rate limit, which already
        # retries — fall back to the Anthropic API for that one call if this is on AND an API key is
        # configured (analyst._cli_fallback_eligible). Off by default: flipping it on can silently
        # start spending real per-token $ against the API key instead of the $0 subscription, so it's
        # worth an explicit opt-in rather than a surprise on someone else's session limit.
        "cli_fallback_to_api": (os.environ.get("CLI_FALLBACK_TO_API", "false").strip().lower()
                                 in ("1", "true", "yes")),
        "verdict_ttl_seconds": int(os.environ.get("VERDICT_TTL_SECONDS", "14400")),
        "watchlist": _split(os.environ.get("WATCHLIST", "")),
        "crypto_watchlist": _split(os.environ.get("CRYPTO_WATCHLIST", "")),
        # Epoch seconds of the last watchlist push from the app (None until the app first syncs).
        # Doubles as an "app is connected" heartbeat since the app re-syncs every ~15 min.
        "watchlist_synced_at": None,
        # Install id of the client that sent the last accepted watchlist/crypto_watchlist sync (OPS-3).
        # None until a client that sends one syncs; used only to distinguish "the same install pruning
        # its own list" from "a different client overwriting it" — never displayed as a raw identity.
        "watchlist_synced_by": None,
        # OPS-3 removal guard, in symbol count — see WatchlistSyncRefused / update(). Editable so ops
        # can tune it without a redeploy (e.g. a household with a genuinely huge watchlist).
        "watchlist_removal_guard": int(os.environ.get("WATCHLIST_REMOVAL_GUARD", str(_DEFAULT_REMOVAL_GUARD))),
    }


def _atomic_write_json(path: Path, bak: Path, data: dict) -> None:
    """Same technique as app.sandbox_store._atomic_write_json: write to a unique temp file, fsync,
    snapshot the current file to `bak` (best-effort), then os.replace — atomic on one filesystem, so
    a crash mid-write leaves either the old file or the new one, never a half-written one."""
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(tmp, 0o600)
    if path.exists():
        try:
            shutil.copy2(path, bak)    # last-known-good before we overwrite
        except Exception:  # noqa: BLE001
            pass
    os.replace(tmp, path)              # atomic on a single filesystem


def _repair_from_backup(cfg: dict) -> None:
    """settings.json failed to parse but .bak did: write the recovered content back to settings.json
    so the service isn't running in a degraded state forever. The corrupt file is moved aside (not
    deleted — kept for forensics) rather than left in place, so the repair write's own
    _atomic_write_json call sees `path` absent and skips its "snapshot current into .bak" step —
    otherwise it would copy the CORRUPT file over the very backup we're recovering from."""
    try:
        if _FILE.exists():
            _FILE.replace(_CORRUPT)
        _atomic_write_json(_FILE, _BAK, cfg)
        _log.warning("settings.json repaired from %s (corrupt copy kept at %s)", _BAK, _CORRUPT)
    except Exception:  # noqa: BLE001 — repair is best-effort; the in-memory recovery already happened
        _log.error("settings.json repair from .bak failed", exc_info=True)


def _load() -> tuple[dict, str]:
    """Load settings, preferring data/settings.json, falling back to its .bak on a parse failure, and
    finally to environment defaults. Never silently swallows a parse failure (OPS-5) — a corrupted
    file that fell back to defaults used to look identical to a normally-configured backend running
    on different settings. Returns (config, source) where source is one of SOURCE_FILE/BACKUP/ENV."""
    if _FILE.exists():
        try:
            cfg = _defaults()
            cfg.update(json.loads(_FILE.read_text()))
            return cfg, SOURCE_FILE
        except Exception:  # noqa: BLE001 — try the .bak next, loudly
            _log.error("settings.json failed to parse; attempting recovery from %s", _BAK, exc_info=True)
    if _BAK.exists():
        try:
            cfg = _defaults()
            cfg.update(json.loads(_BAK.read_text()))
            _log.warning("settings recovered from %s after settings.json failed to parse", _BAK)
            _repair_from_backup(cfg)
            return cfg, SOURCE_BACKUP
        except Exception:  # noqa: BLE001 — both copies are bad; fall through to env defaults
            _log.error("settings.json.bak ALSO failed to parse; degrading to environment defaults",
                       exc_info=True)
    if _FILE.exists() or _BAK.exists():
        _log.error("settings.json unreadable (file and backup both failed/absent) — running on "
                   "environment defaults; any UI-saved API key, watchlist, or model choice is "
                   "unavailable until this is fixed")
    return _defaults(), SOURCE_ENV


_current, _source = _load()


def get() -> dict:
    with _lock:
        return dict(_current)


def source() -> str:
    """Which of settings.json / .bak / env defaults currently backs the in-memory settings — for
    /health's settings_source (OPS-5)."""
    with _lock:
        return _source


def update(patch: dict, *, client_id: str | None = None, replace: bool = False) -> dict:
    """Apply a partial update. Empty strings are treated as "leave unchanged" so a blank key field
    in the UI never wipes the stored key.

    client_id is the syncing client's install id (OPS-3, optional — omit for the settings UI's own
    saves, which never touch the watchlist fields). If the patch changes watchlist or
    crypto_watchlist, and that change would remove more than watchlist_removal_guard symbols, and
    client_id differs from the install id that wrote the current list, the update is refused by
    raising WatchlistSyncRefused instead of being applied — unless replace=True. A same-client sync
    (client_id == the last recorded writer, including both being None for clients that don't send an
    id at all) is never refused, regardless of size: a user pruning their own watchlist must go
    through.

    Every accepted watchlist/crypto_watchlist change is logged as n_before -> n_after plus the actual
    symbols removed, whether or not the guard applied — OPS-3 happened because that was silent."""
    global _source
    with _lock:
        for k in ("anthropic_api_key", "finnhub_api_key", "deep_model", "scan_model", "cli_oauth_token"):
            v = patch.get(k)
            if v is not None and str(v).strip() != "":
                _current[k] = str(v).strip()
        ttl = patch.get("verdict_ttl_seconds")
        if ttl is not None:
            _current["verdict_ttl_seconds"] = max(0, int(ttl))
        guard = patch.get("watchlist_removal_guard")
        if guard is not None:
            _current["watchlist_removal_guard"] = max(0, int(guard))
        prov = patch.get("llm_provider")
        if prov is not None and str(prov).strip().lower() in ("api", "cli"):
            _current["llm_provider"] = str(prov).strip().lower()

        pending: dict[str, list[str]] = {}
        for k in ("watchlist", "crypto_watchlist"):
            v = patch.get(k)
            if v is not None:
                pending[k] = _split(v) if isinstance(v, str) else [str(s).strip().upper() for s in v]

        if pending:
            last_writer = _current.get("watchlist_synced_by")
            same_client = client_id == last_writer
            threshold = int(_current.get("watchlist_removal_guard", _DEFAULT_REMOVAL_GUARD))
            if not same_client and not replace:
                for field, new_list in pending.items():
                    before = _current.get(field) or []
                    removed = sorted(set(before) - set(new_list))
                    if len(removed) > threshold:
                        raise WatchlistSyncRefused(field, len(before), len(new_list), removed, threshold)
            for field, new_list in pending.items():
                before = _current.get(field) or []
                if new_list != before:
                    removed = sorted(set(before) - set(new_list))
                    added = sorted(set(new_list) - set(before))
                    _log.info(
                        "%s sync from client=%s: %d -> %d symbols; removed=%s added=%s%s",
                        field, client_id or "(none)", len(before), len(new_list), removed, added,
                        " [forced past removal guard]" if (replace and not same_client
                                                            and len(removed) > threshold) else "",
                    )
                _current[field] = new_list
            # A patch carrying watchlist fields is an app sync (the UI's settings-save omits them) —
            # stamp the heartbeat + writer so the UI can show when the app last checked in and OPS-3's
            # guard has an identity to compare the next sync against.
            _current["watchlist_synced_at"] = time.time()
            _current["watchlist_synced_by"] = client_id

        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(_FILE, _BAK, _current)
        _source = SOURCE_FILE
        return dict(_current)
