"""Shared pytest setup for the signals backend test suite.

SEC-2 added a bearer-token gate (app.main.require_api_token) in front of every non-GET route plus a
handful of sensitive GET routes, and made app startup (app.main.lifespan) refuse to run at all unless
SIGNALS_API_TOKEN is set. Two consequences follow for the ~80 test files under here, both handled
once, in this file, rather than in each of them:

1. Every test that builds a TestClient needs SIGNALS_API_TOKEN set before app.main is imported or
   reloaded, or the app now refuses to start. Default it here, before test collection imports
   anything. A test that specifically wants to exercise the "unset" startup-refusal path removes it
   with monkeypatch.delenv, which pytest restores automatically at teardown.

2. Starlette's TestClient fakes its connecting peer as ("testclient", 50000) by default — nothing
   like the loopback address a real systemd-timer `curl http://127.0.0.1:8000/...` call presents (see
   deploy/*.service). require_api_token() exempts real loopback callers so those units keep working
   without ever holding the secret; if TestClient kept its fake peer, every existing test in this
   suite (none of which sends a token) would start seeing 401s. So: default every TestClient built
   anywhere in this suite to a loopback peer. A test that wants to exercise the "real remote caller"
   path (the whole point of SEC-2) passes its own non-loopback `client=(...)` explicitly, which this
   wrapper leaves alone (it only fills in a default, via setdefault).
"""
from __future__ import annotations

import os

os.environ.setdefault("SIGNALS_API_TOKEN", "pytest-signals-token-do-not-use-in-prod")

import starlette.testclient as _st  # noqa: E402 — must follow the env default above

_orig_testclient_init = _st.TestClient.__init__


def _loopback_default_init(self, app, *args, **kwargs):
    kwargs.setdefault("client", ("127.0.0.1", 55123))
    _orig_testclient_init(self, app, *args, **kwargs)


_st.TestClient.__init__ = _loopback_default_init
