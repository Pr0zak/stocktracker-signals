"""
Offline tests for the headless-CLI analyst backend (app/llm_cli.py) and analyst provider routing.
No real `claude` process is ever spawned — the subprocess / _invoke layer is mocked.
"""
from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from app import analyst, llm_cli, settings_store


class Foo(BaseModel):
    a: int
    b: str


def _env(result, **over):
    e = {
        "is_error": False,
        "stop_reason": "end_turn",
        "result": result,
        "total_cost_usd": 0.01,
        "usage": {
            "input_tokens": 5, "output_tokens": 7,
            "cache_read_input_tokens": 3, "cache_creation_input_tokens": 11,
        },
    }
    e.update(over)
    return e


# ============================ _strip_to_json ============================

def test_strip_fenced_json():
    assert llm_cli._strip_to_json('```json\n{"a":1}\n```') == '{"a":1}'


def test_strip_plain_fence():
    assert llm_cli._strip_to_json('```\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_bare():
    assert llm_cli._strip_to_json('{"a":1}') == '{"a":1}'


def test_strip_prose_wrapped():
    assert llm_cli._strip_to_json('Sure, here you go: {"a":1, "b":"x"} — hope it helps!') == '{"a":1, "b":"x"}'


# ============================ _usage_from_env ============================

def test_usage_maps_and_tags_provider():
    u = llm_cli._usage_from_env("claude-haiku-4-5", _env("{}"))
    assert u["provider"] == "cli"
    assert u["input_tokens"] == 5 and u["output_tokens"] == 7
    assert u["cache_read_tokens"] == 3 and u["cache_write_tokens"] == 11
    assert u["cost_usd"] == 0.01 and u["model"] == "claude-haiku-4-5"


def test_usage_tolerates_missing_usage_block():
    u = llm_cli._usage_from_env("m", {"result": "x"})   # no usage / cost keys
    assert u["input_tokens"] == 0 and u["output_tokens"] == 0 and u["cost_usd"] == 0.0


# ============================ structured() (patch _invoke) ============================

def _patch_invoke(monkeypatch, envs):
    seq = list(envs)
    calls = {"n": 0}

    async def fake(model, system, prompt, *, thinking=False):
        i = calls["n"]
        calls["n"] += 1
        return seq[min(i, len(seq) - 1)]

    monkeypatch.setattr(llm_cli, "_invoke", fake)
    return calls


def test_structured_ok_fenced(monkeypatch):
    _patch_invoke(monkeypatch, [_env('```json\n{"a": 1, "b": "x"}\n```')])
    obj, u = asyncio.run(llm_cli.structured("sys", "user", Foo, model="m"))
    assert isinstance(obj, Foo) and obj.a == 1 and obj.b == "x"
    assert u["provider"] == "cli"


def test_structured_retries_then_succeeds(monkeypatch):
    calls = _patch_invoke(monkeypatch, [_env("not json at all"), _env('{"a": 2, "b": "y"}')])
    obj, _ = asyncio.run(llm_cli.structured("sys", "user", Foo, model="m"))
    assert obj.a == 2 and calls["n"] == 2   # took the retry


def test_structured_raises_after_two_bad(monkeypatch):
    calls = _patch_invoke(monkeypatch, [_env("garbage"), _env("still garbage")])
    with pytest.raises(llm_cli.CliError):
        asyncio.run(llm_cli.structured("sys", "user", Foo, model="m"))
    assert calls["n"] == 2   # exactly two attempts, no infinite loop


def test_structured_rejects_schema_violation(monkeypatch):
    # valid JSON but wrong types for Foo -> pydantic fails both tries -> CliError
    _patch_invoke(monkeypatch, [_env('{"a": "notint", "b": 5}')])
    with pytest.raises(llm_cli.CliError):
        asyncio.run(llm_cli.structured("sys", "user", Foo, model="m"))


def test_text_ok(monkeypatch):
    _patch_invoke(monkeypatch, [_env("  a plain paragraph.  ")])
    out, u = asyncio.run(llm_cli.text("sys", "user", model="m"))
    assert out == "a plain paragraph." and u["provider"] == "cli"


def test_text_empty_raises(monkeypatch):
    _patch_invoke(monkeypatch, [_env("   ")])
    with pytest.raises(llm_cli.CliError):
        asyncio.run(llm_cli.text("sys", "user", model="m"))


# ============================ _invoke() (patch subprocess) ============================

class _FakeProc:
    def __init__(self, out=b"", err=b"", rc=0):
        self._out, self._err, self.returncode = out, err, rc
        self.killed = False

    async def communicate(self, input=None):
        return self._out, self._err

    def kill(self):
        self.killed = True

    async def wait(self):
        return self.returncode


def _patch_exec(monkeypatch, proc=None, raises=None):
    async def fake_exec(*a, **k):
        if raises is not None:
            raise raises
        return proc
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)


def test_invoke_ok(monkeypatch):
    import json
    proc = _FakeProc(out=json.dumps(_env('{"a":1}')).encode())
    _patch_exec(monkeypatch, proc)
    env = asyncio.run(llm_cli._invoke("m", "sys", "user"))
    assert env["result"] == '{"a":1}'


def test_invoke_nonzero_exit_raises(monkeypatch):
    _patch_exec(monkeypatch, _FakeProc(err=b"boom", rc=1))
    with pytest.raises(llm_cli.CliError):
        asyncio.run(llm_cli._invoke("m", "sys", "user"))


def test_invoke_error_envelope_raises(monkeypatch):
    import json
    _patch_exec(monkeypatch, _FakeProc(out=json.dumps(_env("x", is_error=True)).encode()))
    with pytest.raises(llm_cli.CliError):
        asyncio.run(llm_cli._invoke("m", "sys", "user"))


def test_invoke_error_envelope_surfaces_result_text(monkeypatch):
    # The `result` field carries the human-readable reason (e.g. a session-limit message) — it must
    # land in the raised error's text, or budget/rate/fatal classification has nothing to match on.
    import json
    _patch_exec(monkeypatch, _FakeProc(out=json.dumps(
        _env("You've hit your session limit · resets 9:10am", is_error=True)
    ).encode()))
    with pytest.raises(llm_cli.CliError, match="session limit"):
        asyncio.run(llm_cli._invoke("m", "sys", "user"))


def test_invoke_truncated_raises(monkeypatch):
    import json
    _patch_exec(monkeypatch, _FakeProc(out=json.dumps(_env("x", stop_reason="max_tokens")).encode()))
    with pytest.raises(llm_cli.CliError):
        asyncio.run(llm_cli._invoke("m", "sys", "user"))


def test_invoke_non_json_stdout_raises(monkeypatch):
    _patch_exec(monkeypatch, _FakeProc(out=b"<html>login</html>"))
    with pytest.raises(llm_cli.CliError):
        asyncio.run(llm_cli._invoke("m", "sys", "user"))


def test_invoke_missing_binary_raises(monkeypatch):
    _patch_exec(monkeypatch, raises=FileNotFoundError("no claude"))
    with pytest.raises(llm_cli.CliError):
        asyncio.run(llm_cli._invoke("m", "sys", "user"))


def test_child_env_excludes_auth_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok")
    monkeypatch.setenv("PATH", "/usr/bin")
    e = llm_cli._child_env(thinking=False)
    assert "ANTHROPIC_API_KEY" not in e and "ANTHROPIC_AUTH_TOKEN" not in e
    assert e.get("PATH") == "/usr/bin"   # non-auth env still passes through


def test_child_env_injects_ui_token_over_env(monkeypatch):
    # A token saved via the settings UI takes precedence over the CLAUDE_CODE_OAUTH_TOKEN service env.
    monkeypatch.setattr(settings_store, "get", lambda: {"cli_oauth_token": "sk-ant-oat-ui"})
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat-env")
    assert llm_cli._child_env(thinking=False)["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat-ui"


def test_child_env_falls_back_to_env_token(monkeypatch):
    monkeypatch.setattr(settings_store, "get", lambda: {"cli_oauth_token": ""})
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat-env")
    assert llm_cli._child_env(thinking=False)["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat-env"


def test_child_env_no_token_when_neither_set(monkeypatch):
    monkeypatch.setattr(settings_store, "get", lambda: {"cli_oauth_token": ""})
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in llm_cli._child_env(thinking=False)


def test_child_env_gates_thinking(monkeypatch):
    monkeypatch.setattr(settings_store, "get", lambda: {"cli_oauth_token": ""})
    monkeypatch.delenv("MAX_THINKING_TOKENS", raising=False)
    # scan tier: thinking disabled (Haiku verdict balloons ~15x with thinking on, for no gain)
    assert llm_cli._child_env(thinking=False).get("MAX_THINKING_TOKENS") == "0"
    # deep tier: var cleared so the model default applies (Opus errors on a 0 budget)
    assert "MAX_THINKING_TOKENS" not in llm_cli._child_env(thinking=True)


def test_invoke_passes_sanitized_env_to_subprocess(monkeypatch):
    # The child claude MUST NOT inherit ANTHROPIC_API_KEY, or it bills per-token instead of the
    # subscription OAuth (verified: a set key makes headless claude 401 on the key, ignoring OAuth).
    import json
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("HOME", "/root")
    captured = {}

    async def fake_exec(*a, **k):
        captured["env"] = k.get("env")
        return _FakeProc(out=json.dumps(_env('{"a":1}')).encode())
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    asyncio.run(llm_cli._invoke("m", "sys", "user"))
    env = captured["env"]
    assert env is not None and "ANTHROPIC_API_KEY" not in env
    assert env.get("HOME") == "/root"   # HOME (needed to find OAuth creds) still passed


def test_invoke_concurrency_is_bounded(monkeypatch):
    # 6 concurrent _invoke calls, semaphore of 2 -> never more than 2 subprocesses in flight at once.
    import json
    monkeypatch.setattr(llm_cli, "_SEM", asyncio.Semaphore(2))
    state = {"cur": 0, "max": 0}
    payload = json.dumps(_env('{"a":1}')).encode()

    class SlowProc(_FakeProc):
        async def communicate(self, input=None):
            state["cur"] += 1
            state["max"] = max(state["max"], state["cur"])
            for _ in range(5):
                await asyncio.sleep(0)   # yield so siblings can interleave (no real sleep)
            state["cur"] -= 1
            return self._out, self._err

    async def fake_exec(*a, **k):
        return SlowProc(out=payload)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    async def many():
        await asyncio.gather(*[llm_cli._invoke("m", "sys", "u") for _ in range(6)])
    asyncio.run(many())
    assert state["max"] <= 2, f"expected <=2 concurrent claude procs, saw {state['max']}"


def test_invoke_timeout_raises_and_kills(monkeypatch):
    proc = _FakeProc(out=b"{}")
    _patch_exec(monkeypatch, proc)

    async def boom(coro, *a, **k):
        coro.close()   # avoid "coroutine was never awaited" — we're simulating the timeout path
        raise asyncio.TimeoutError()
    monkeypatch.setattr(asyncio, "wait_for", boom)
    with pytest.raises(llm_cli.CliError):
        asyncio.run(llm_cli._invoke("m", "sys", "user"))
    assert proc.killed   # timed-out process is killed, not leaked


# ============================ session-limit / budget-exhaustion classification ============================

def test_is_budget_exhausted_recognizes_session_limit_phrase():
    assert llm_cli._is_budget_exhausted(Exception("You've hit your session limit"))


def test_is_budget_exhausted_recognizes_resets_at_phrasing():
    assert llm_cli._is_budget_exhausted(Exception("Session limit reached — resets at 9:10am"))


def test_is_budget_exhausted_recognizes_terse_resets_phrasing():
    # The real OPS-2 incident message: no "at" between "resets" and the clock time.
    assert llm_cli._is_budget_exhausted(Exception("You've hit your session limit · resets 9:10am"))


def test_is_budget_exhausted_false_for_generic_rate_limit():
    # A genuine transient rate limit must NOT be classified as budget exhaustion (it should still
    # retry, unlike a session-limit wall).
    assert not llm_cli._is_budget_exhausted(Exception("429 too many requests, rate limit exceeded"))


def test_invoke_resilient_does_not_retry_session_limit(monkeypatch):
    calls = {"n": 0}

    async def fake_invoke(model, system, prompt, *, thinking=False):
        calls["n"] += 1
        raise llm_cli.CliError("You've hit your session limit · resets 9:10am")
    monkeypatch.setattr(llm_cli, "_invoke", fake_invoke)

    async def fail_if_slept(*a, **k):
        raise AssertionError("must not sleep/retry against a session-limit wall")
    monkeypatch.setattr(asyncio, "sleep", fail_if_slept)

    with pytest.raises(llm_cli.CliBudgetExhaustedError):
        asyncio.run(llm_cli._invoke_resilient("m", "sys", "user"))
    assert calls["n"] == 1   # exactly one attempt — no retry into the same wall


def test_invoke_resilient_still_retries_transient_rate_limit(monkeypatch):
    # Contrast case: a genuine transient failure still gets its one retry (unchanged behavior).
    calls = {"n": 0}

    async def fake_invoke(model, system, prompt, *, thinking=False):
        calls["n"] += 1
        if calls["n"] == 1:
            raise llm_cli.CliError("429 too many requests")
        return {"result": "ok"}
    monkeypatch.setattr(llm_cli, "_invoke", fake_invoke)

    async def fast_sleep(*a, **k):
        return None
    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    env = asyncio.run(llm_cli._invoke_resilient("m", "sys", "user"))
    assert env == {"result": "ok"} and calls["n"] == 2


# ============================ analyst provider routing ============================

def _verdict():
    return analyst.Verdict(signal=analyst.Signal.buy, conviction=150, horizon="swing",
                           thesis="t", rationale=["r"], key_risks=["k"], invalidation="below 100",
                           catalysts=[])


def test_analyst_routes_to_cli_when_selected(monkeypatch):
    monkeypatch.setattr(settings_store, "get",
                        lambda: {"llm_provider": "cli", "deep_model": "d", "scan_model": "s"})
    seen = {}

    async def fake_structured(system, prompt, output_model, *, model, max_tokens=4096, thinking=False):
        seen["model"] = model
        seen["thinking"] = thinking
        return _verdict(), {"provider": "cli", "model": model}
    monkeypatch.setattr(llm_cli, "structured", fake_structured)

    v, u = asyncio.run(analyst.analyze({"symbol": "AAPL"}, deep=False))
    assert u["provider"] == "cli"
    assert seen["model"] == "s"          # scan tier for deep=False
    assert v.conviction == 100           # analyze() still clamps 0-100 on the CLI path


def test_analyst_api_path_tags_provider_api():
    # _usage() (API path) must stamp provider=api so old + API rows aggregate correctly.
    class U:
        input_tokens = 10
        output_tokens = 20
        cache_read_input_tokens = 0
        cache_creation_input_tokens = 0
    d = analyst._usage("claude-haiku-4-5", U())
    assert d["provider"] == "api"


# ============================ OPS-2: cli -> api fallback on budget exhaustion ============================

def test_cli_fallback_to_api_defaults_off(monkeypatch):
    monkeypatch.delenv("CLI_FALLBACK_TO_API", raising=False)
    assert settings_store._defaults()["cli_fallback_to_api"] is False


def _fake_api_client(parsed):
    """A stand-in for analyst._get_client() whose messages.parse() succeeds immediately — used to
    prove the analyst._parse() cli->api fallback actually completes the call on the API path."""
    class _FakeUsage:
        input_tokens = 42
        output_tokens = 7

    class _FakeResp:
        usage = _FakeUsage()
        stop_reason = "end_turn"
        parsed_output = parsed

    class _FakeMessages:
        async def parse(self, **kwargs):
            return _FakeResp()

    class _FakeClient:
        messages = _FakeMessages()

    return _FakeClient()


def test_analyst_falls_back_to_api_when_cli_budget_exhausted_and_eligible(monkeypatch):
    # Both conditions met: cli_fallback_to_api is on AND an API key is configured.
    monkeypatch.setattr(settings_store, "get", lambda: {
        "llm_provider": "cli", "deep_model": "d", "scan_model": "s",
        "cli_fallback_to_api": True, "anthropic_api_key": "sk-ant-real",
    })

    async def fake_structured(system, prompt, output_model, *, model, max_tokens=4096, thinking=False):
        raise llm_cli.CliBudgetExhaustedError("You've hit your session limit · resets 9:10am")
    monkeypatch.setattr(llm_cli, "structured", fake_structured)
    monkeypatch.setattr(analyst, "_get_client", lambda: _fake_api_client(_verdict()))

    v, u = asyncio.run(analyst.analyze({"symbol": "AAPL"}, deep=False))
    assert u["provider"] == "api"          # cost card must show this as an API-tier call
    assert u["fallback_from"] == "cli"     # ...and that it was a fallback, not a configured API run


def test_analyst_reraises_budget_exhaustion_when_fallback_setting_off(monkeypatch):
    # Setting is off (default) — no fallback, even with a key configured.
    monkeypatch.setattr(settings_store, "get", lambda: {
        "llm_provider": "cli", "deep_model": "d", "scan_model": "s",
        "cli_fallback_to_api": False, "anthropic_api_key": "sk-ant-real",
    })

    async def fake_structured(system, prompt, output_model, *, model, max_tokens=4096, thinking=False):
        raise llm_cli.CliBudgetExhaustedError("You've hit your session limit · resets 9:10am")
    monkeypatch.setattr(llm_cli, "structured", fake_structured)

    with pytest.raises(llm_cli.CliBudgetExhaustedError):
        asyncio.run(analyst.analyze({"symbol": "AAPL"}, deep=False))


def test_analyst_reraises_budget_exhaustion_when_no_api_key(monkeypatch):
    # Setting is on, but no key is configured — no fallback (nothing to fall back to).
    monkeypatch.setattr(settings_store, "get", lambda: {
        "llm_provider": "cli", "deep_model": "d", "scan_model": "s",
        "cli_fallback_to_api": True, "anthropic_api_key": "",
    })

    async def fake_structured(system, prompt, output_model, *, model, max_tokens=4096, thinking=False):
        raise llm_cli.CliBudgetExhaustedError("You've hit your session limit · resets 9:10am")
    monkeypatch.setattr(llm_cli, "structured", fake_structured)

    with pytest.raises(llm_cli.CliBudgetExhaustedError):
        asyncio.run(analyst.analyze({"symbol": "AAPL"}, deep=False))


def test_analyst_does_not_fall_back_on_non_budget_cli_error(monkeypatch):
    # A non-budget CliError (e.g. auth failure) must propagate untouched even with fallback eligible —
    # the fallback is scoped to CliBudgetExhaustedError, never a blanket safety net for any CLI error.
    monkeypatch.setattr(settings_store, "get", lambda: {
        "llm_provider": "cli", "deep_model": "d", "scan_model": "s",
        "cli_fallback_to_api": True, "anthropic_api_key": "sk-ant-real",
    })

    async def fake_structured(system, prompt, output_model, *, model, max_tokens=4096, thinking=False):
        raise llm_cli.CliError("not logged in — /login")
    monkeypatch.setattr(llm_cli, "structured", fake_structured)

    with pytest.raises(llm_cli.CliError):
        asyncio.run(analyst.analyze({"symbol": "AAPL"}, deep=False))
