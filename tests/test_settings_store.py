"""app/settings_store.py — OPS-3 (watchlist sync identity/guard) and OPS-5 (atomic persistence).

OPS-5: settings.json is the sole home of the Finnhub key, the Claude CLI OAuth token, llm_provider,
deep_model and the watchlist, written ~70x/day by a truncate-then-write that can lose everything on a
crash mid-write, and a `_load()` that swallowed a parse error and silently ran on environment
defaults — a corrupted file looked identical to a normally-configured backend running on different
settings. These tests pin: the write surviving a simulated crash (a half-written temp file must never
land on the real path), a corrupt settings.json recovering from .bak, and `source()` reporting each of
the three states (file / backup / env) that /health's `settings_source` surfaces.

OPS-3: on 2026-09-11 a debug client's stale 14-symbol watchlist overwrote the real 54-symbol one
moments before the nightly scan, with no log line and no way to tell the sync came from a different
client. These tests pin: a cross-client sync that removes more than the configured guard is refused
(WatchlistSyncRefused), the SAME client is never refused regardless of size, and `replace=True`
forces it through.
"""
from __future__ import annotations

import importlib
import json

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A settings_store bound to an isolated data dir, freshly reloaded per test."""
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    import app.settings_store as st
    importlib.reload(st)
    return st


# --- OPS-5: atomic write + recovery -----------------------------------------------------------

def test_atomic_write_leaves_no_partial_temp_file_and_updates_the_real_path(store):
    store.update({"deep_model": "claude-opus-x"})
    assert store._FILE.exists()
    assert json.loads(store._FILE.read_text())["deep_model"] == "claude-opus-x"
    # No leftover .tmp.<pid> — os.replace either lands the write or the old file survives untouched.
    leftovers = list(store._DATA_DIR.glob("settings.json.tmp.*"))
    assert leftovers == []


def test_atomic_write_keeps_a_bak_of_the_previous_good_content(store):
    store.update({"deep_model": "first"})
    store.update({"deep_model": "second"})
    assert store._BAK.exists()
    assert json.loads(store._BAK.read_text())["deep_model"] == "first"
    assert json.loads(store._FILE.read_text())["deep_model"] == "second"


def test_a_crash_mid_write_never_corrupts_the_real_file(store, monkeypatch):
    """Simulate a crash between the temp-file write and os.replace: os.replace itself raises. The
    real settings.json must still hold the last successfully committed content, not a truncated or
    half-written one — that's the entire point of temp-file + os.replace over truncate-then-write."""
    store.update({"deep_model": "before-crash"})
    good_before = store._FILE.read_text()

    real_replace = store.os.replace
    calls = {"n": 0}

    def _boom(src, dst):
        calls["n"] += 1
        raise OSError("simulated crash mid-write")

    monkeypatch.setattr(store.os, "replace", _boom)
    with pytest.raises(OSError):
        store.update({"deep_model": "crashes-here"})
    monkeypatch.setattr(store.os, "replace", real_replace)

    assert calls["n"] == 1
    assert store._FILE.read_text() == good_before
    assert json.loads(store._FILE.read_text())["deep_model"] == "before-crash"


def test_corrupt_settings_json_restores_from_bak_and_repairs_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    import app.settings_store as st
    importlib.reload(st)
    st.update({"deep_model": "good-value"})
    good_bak_content = st._FILE.read_text()  # becomes .bak once we write again

    st.update({"deep_model": "second-good-value"})  # now .bak == good_bak_content (deep_model=good-value)
    st._FILE.write_text("{ not json at all")  # corrupt the live file

    importlib.reload(st)  # re-run _load() against the corrupted file + valid .bak

    assert st.source() == st.SOURCE_BACKUP
    assert st.get()["deep_model"] == "good-value"
    # The file itself was repaired, not just the in-memory copy — the next load must not need .bak.
    assert json.loads(st._FILE.read_text())["deep_model"] == "good-value"
    # The corrupt content wasn't silently discarded.
    assert "not json at all" in st._CORRUPT.read_text()


def test_source_is_env_when_both_file_and_backup_are_corrupt(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    import app.settings_store as st
    importlib.reload(st)
    st._DATA_DIR.mkdir(parents=True, exist_ok=True)
    st._FILE.write_text("{ nope")
    st._BAK.write_text("[ also nope")

    importlib.reload(st)

    assert st.source() == st.SOURCE_ENV
    # Falls back to environment/hardcoded defaults, not an empty or half-parsed dict.
    assert st.get()["deep_model"]  # non-empty default model name


def test_source_is_file_on_a_normal_load(store):
    store.update({"deep_model": "x"})
    assert store.source() == store.SOURCE_FILE
    importlib.reload(store)  # reload against the settings.json just written, not a fresh tmp dir
    assert store.source() == store.SOURCE_FILE
    assert store.get()["deep_model"] == "x"


def test_source_is_env_when_no_settings_file_exists_yet(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path))
    import app.settings_store as st
    importlib.reload(st)
    assert not st._FILE.exists()
    assert st.source() == st.SOURCE_ENV


# --- OPS-3: watchlist sync identity + removal guard -------------------------------------------

def test_cross_client_sync_under_the_guard_is_never_refused(store):
    store.update({"watchlist": list(f"S{i}" for i in range(10))}, client_id="phone-1")
    # Removes 4 symbols (<= default guard of 5) from a DIFFERENT client — allowed.
    out = store.update({"watchlist": list(f"S{i}" for i in range(6))}, client_id="emulator-2")
    assert len(out["watchlist"]) == 6


def test_cross_client_sync_over_the_guard_is_refused(store):
    store.update({"watchlist": list(f"S{i}" for i in range(54))}, client_id="phone-1")
    with pytest.raises(store.WatchlistSyncRefused) as ei:
        store.update({"watchlist": list(f"S{i}" for i in range(14))}, client_id="emulator-2")
    err = ei.value
    assert err.field == "watchlist"
    assert err.n_before == 54
    assert err.n_after == 14
    assert len(err.removed) == 40
    assert err.threshold == store._DEFAULT_REMOVAL_GUARD
    d = err.detail()
    assert d["error"] == "watchlist_sync_refused"
    assert d["removed_count"] == 40
    assert "replace=true" in d["message"]
    # Refusing must not have partially applied the change.
    assert len(store.get()["watchlist"]) == 54
    assert store.get()["watchlist_synced_by"] == "phone-1"


def test_same_client_pruning_its_own_list_is_never_refused_at_any_size(store):
    store.update({"watchlist": list(f"S{i}" for i in range(54))}, client_id="phone-1")
    out = store.update({"watchlist": ["AAPL"]}, client_id="phone-1")
    assert out["watchlist"] == ["AAPL"]


def test_two_clients_that_never_send_an_id_are_treated_as_the_same_client(store):
    """Back-compat: an app that doesn't send install ids at all must not suddenly get blocked just
    because the guard now exists — client_id defaults to None on both sides."""
    store.update({"watchlist": list(f"S{i}" for i in range(54))})
    out = store.update({"watchlist": ["AAPL"]})
    assert out["watchlist"] == ["AAPL"]


def test_replace_true_forces_a_refused_sync_through(store):
    store.update({"watchlist": list(f"S{i}" for i in range(54))}, client_id="phone-1")
    out = store.update({"watchlist": list(f"S{i}" for i in range(14))}, client_id="emulator-2", replace=True)
    assert len(out["watchlist"]) == 14
    assert out["watchlist_synced_by"] == "emulator-2"


def test_removal_guard_threshold_is_configurable(store):
    store.update({"watchlist_removal_guard": 1})
    store.update({"watchlist": list(f"S{i}" for i in range(10))}, client_id="phone-1")
    # Removing 2 symbols now exceeds the lowered threshold of 1.
    with pytest.raises(store.WatchlistSyncRefused):
        store.update({"watchlist": list(f"S{i}" for i in range(8))}, client_id="emulator-2")


def test_watchlist_change_is_logged_with_before_after_and_removed_symbols(store, caplog):
    import logging
    store.update({"watchlist": ["AAPL", "MSFT", "TSLA"]}, client_id="phone-1")
    with caplog.at_level(logging.INFO, logger="app.settings_store"):
        store.update({"watchlist": ["AAPL", "MSFT"]}, client_id="phone-1")
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "3 -> 2" in joined
    assert "TSLA" in joined


def test_additions_alone_are_never_refused(store):
    store.update({"watchlist": ["AAPL"]}, client_id="phone-1")
    out = store.update({"watchlist": ["AAPL", "MSFT", "TSLA", "NVDA", "AMD", "GOOG", "AMZN"]},
                        client_id="emulator-2")
    assert len(out["watchlist"]) == 7
