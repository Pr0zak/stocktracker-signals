"""
Headless Claude Code CLI backend for the analyst — an alternative to the Anthropic API path.

When settings `llm_provider == "cli"`, analyst._parse / analyst.options_note route here instead of
the SDK. This shells out to `claude -p --output-format json` using the machine's logged-in
*subscription* OAuth credentials (~/.claude/.credentials.json), so calls draw on the Claude
subscription's rate-limit budget instead of per-token API billing (real $ spend is $0; the cost we
log is the CLI's own notional API-equivalent figure, kept only for the dashboard).

Trade-offs vs the API path (surfaced in the settings UI):
  • No schema-constrained decoding — we inject the JSON Schema in-prompt, then validate + retry once.
  • ~14k tokens of agent-harness overhead per COLD call (cached ~1h after), even with MCP + built-in
    tools stripped; quota, not dollars, is the currency.
  • Higher per-call latency (spawns a Node agent process).

Kept lean with: --strict-mcp-config (drop all MCP tool defs), --exclude-dynamic-system-prompt-sections,
--disallowedTools <all built-ins>, --max-turns 1. The user prompt goes on STDIN (never argv), so there
is no shell and no argv-length/quoting exposure regardless of how large the snapshot JSON is.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re

from pydantic import BaseModel, ValidationError

log = logging.getLogger("uvicorn.error")

# The claude binary + a hard wall-clock cap per call, both overridable from the systemd unit's env.
_CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")
_TIMEOUT_S = float(os.environ.get("CLAUDE_CLI_TIMEOUT", "180"))

# Bound how many `claude` subprocesses run at once. The nightly scan fans the whole watchlist out with
# an unbounded asyncio.gather, and unlike the pooled-HTTP API path each CLI call is a heavyweight Node
# agent process (~14k-token cold harness). Without this cap a 24-symbol scan would spawn 24 at once —
# memory pressure and a burst the subscription rate-limiter rejects, failing most rows.
_CONCURRENCY = max(1, int(os.environ.get("CLAUDE_CLI_CONCURRENCY", "3")))
_SEM = asyncio.Semaphore(_CONCURRENCY)

# Env vars that make the CLI authenticate as an API KEY (per-token billing) in preference to the
# machine's subscription OAuth. cli mode's whole premise is $0 subscription use, so these are stripped
# from the child env — verified: with a key set, headless `claude` returns 401 on a bad key instead of
# falling back to the valid ~/.claude OAuth. Stripping forces OAuth (and fails loudly if it's missing).
_AUTH_ENV_STRIP = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


def _child_env(thinking: bool) -> dict:
    """Child env for the CLI: OAuth-only (no inherited API key), with thinking gated per tier.

    The headless CLI turns on extended thinking by default, which on the scan (Haiku) tier balloons
    a ~190-token verdict to ~2-9k output tokens (≈22s and heavy quota) for no quality gain — the API
    scan path never used thinking. So we set MAX_THINKING_TOKENS=0 for non-thinking (scan) calls
    (22s→1.5s). Deep (Opus) calls DO think — and Opus errors on a 0 budget — so we clear the var and
    let the model default apply. Mirrors analyst._parse's opus/sonnet/fable thinking predicate."""
    env = {k: v for k, v in os.environ.items() if k not in _AUTH_ENV_STRIP}
    if thinking:
        env.pop("MAX_THINKING_TOKENS", None)   # deep tier: let the model think (Opus 401s on a 0 budget)
    else:
        env["MAX_THINKING_TOKENS"] = "0"        # scan tier: no thinking, matching the API scan path
    return env

# Built-in tools denied so the single-turn agent can only answer — this is a pure text/JSON
# completion, never an action. (MCP tools are already gone via --strict-mcp-config with no
# --mcp-config.) Denying a name that doesn't exist is harmless; a superset just future-proofs.
# Only long-stable core tool names — an UNKNOWN name is a non-fatal warning on newer CLIs but noise,
# and CLI tool sets vary by version, so we keep to names that have existed for many releases.
_DENY_TOOLS = "Bash,Read,Edit,Write,Glob,Grep,WebFetch,WebSearch,Task,TodoWrite,NotebookEdit"

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


class CliError(RuntimeError):
    """A headless claude call failed (spawn/timeout/non-zero exit, error envelope, or bad output)."""


def _lean_argv(model: str, system: str) -> list[str]:
    # There is no CLI flag for max_tokens; single-turn completions rarely hit the model's default
    # output cap (32k), and _invoke guards on stop_reason=="max_tokens" for parity with the API path.
    return [
        _CLAUDE_BIN, "-p",
        "--output-format", "json",
        "--model", model,
        "--system-prompt", system,
        "--max-turns", "1",
        "--strict-mcp-config",
        "--exclude-dynamic-system-prompt-sections",
        "--disallowedTools", _DENY_TOOLS,
    ]


async def _invoke(model: str, system: str, user_prompt: str, *, thinking: bool = False) -> dict:
    """Run one headless claude call; return the parsed result envelope. Raises CliError on any failure.
    The user prompt is fed on stdin so no shell/argv is involved. `thinking` gates extended thinking."""
    argv = _lean_argv(model, system)
    async with _SEM:   # cap concurrent claude processes (see _CONCURRENCY)
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=_child_env(thinking),   # OAuth only (never an API key); thinking gated per tier
            )
        except FileNotFoundError as e:
            raise CliError(f"claude CLI not found ({_CLAUDE_BIN!r}) — installed and on PATH?") from e

        try:
            out, err = await asyncio.wait_for(
                proc.communicate(input=user_prompt.encode()), timeout=_TIMEOUT_S
            )
        except asyncio.TimeoutError as e:
            proc.kill()
            try:
                await proc.wait()  # reap so we don't leak a zombie
            except Exception:  # noqa: BLE001
                pass
            raise CliError(f"claude CLI timed out after {_TIMEOUT_S:.0f}s") from e

    if proc.returncode != 0:
        detail = err.decode(errors="replace")[:300].strip()
        # The useful error (e.g. "Not logged in · Please run /login", auth/model issues) is usually in
        # the JSON envelope on STDOUT, not stderr — surface it so real failures aren't masked by the
        # non-fatal tool-deny warnings the CLI prints to stderr.
        try:
            msg = (json.loads(out.decode()).get("result") or "").strip()
            if msg:
                detail = f"{msg}{(' | ' + detail) if detail else ''}"
        except Exception:  # noqa: BLE001
            pass
        raise CliError(f"claude CLI exited {proc.returncode}: {detail[:300]}")
    try:
        env = json.loads(out.decode())
    except Exception as e:  # noqa: BLE001
        raise CliError(f"claude CLI returned non-JSON: {out.decode(errors='replace')[:200]}") from e
    if env.get("is_error"):
        raise CliError(f"claude CLI error envelope (api_error_status={env.get('api_error_status')})")
    if env.get("stop_reason") == "max_tokens":
        raise CliError("claude CLI output was truncated at the model's max output — retry")
    if "result" not in env:
        raise CliError("claude CLI envelope missing 'result'")
    return env


def _usage_from_env(model: str, env: dict) -> dict:
    """Map the CLI envelope's usage into analyst._usage()'s shape, tagged provider=cli. cost_usd is the
    CLI's own notional (API-equivalent) figure — actual spend is $0 against the subscription."""
    u = env.get("usage") or {}
    return {
        "model": model,
        "input_tokens": int(u.get("input_tokens", 0) or 0),
        "output_tokens": int(u.get("output_tokens", 0) or 0),
        "cache_read_tokens": int(u.get("cache_read_input_tokens", 0) or 0),
        "cache_write_tokens": int(u.get("cache_creation_input_tokens", 0) or 0),
        "cost_usd": round(float(env.get("total_cost_usd", 0.0) or 0.0), 6),
        "provider": "cli",
    }


def _strip_to_json(text: str) -> str:
    """Carve a JSON object out of a model reply that may be fenced or wrapped in stray prose."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = _FENCE_RE.sub("", t).strip()
    if not t.startswith("{"):
        i, j = t.find("{"), t.rfind("}")
        if i != -1 and j > i:
            t = t[i:j + 1]
    return t


async def structured(system: str, user_prompt: str, output_model: type[BaseModel], *,
                     model: str, max_tokens: int = 4096, thinking: bool = False):
    """CLI equivalent of analyst._parse: returns (validated pydantic model, usage dict). Retries once
    with a stricter nudge if the first reply doesn't parse/validate against output_model."""
    schema = json.dumps(output_model.model_json_schema())
    sys = (
        system
        + "\n\nOUTPUT FORMAT (STRICT): Respond with ONLY a single JSON object that validates against "
        "this JSON Schema. No prose, no explanation, no markdown fences.\nJSON Schema:\n" + schema
    )
    last_err: Exception | None = None
    prompt = user_prompt
    for attempt in (1, 2):
        env = await _invoke(model, sys, prompt, thinking=thinking)
        raw = _strip_to_json(env.get("result", ""))
        try:
            obj = output_model.model_validate_json(raw)
            usage = _usage_from_env(model, env)
            log.info("analyst[cli] %s in=%s out=%s cache_read=%s cache_write=%s",
                     model, usage["input_tokens"], usage["output_tokens"],
                     usage["cache_read_tokens"], usage["cache_write_tokens"])
            return obj, usage
        except (ValidationError, ValueError) as e:
            last_err = e
            log.warning("cli structured validate failed (try %s/2): %s", attempt, str(e)[:200])
            prompt = (
                user_prompt
                + "\n\nYour previous reply was not valid JSON for the required schema. Return ONLY the "
                "JSON object — no prose, no fences."
            )
    raise CliError(f"claude CLI structured output failed to validate after 2 tries: {str(last_err)[:200]}")


async def auth_probe(model: str = "claude-haiku-4-5") -> dict:
    """Cheap liveness + auth check for cli mode: one tiny headless call. Returns {ok, detail} and never
    raises — powers the settings page's 'Test CLI auth' button so the operator can confirm the CLI is
    installed and the subscription token is valid without running a full analysis."""
    try:
        env = await _invoke(model, "Reply with the single word OK and nothing else.", "OK", thinking=False)
        return {"ok": True, "detail": (env.get("result") or "").strip()[:60] or "ok"}
    except Exception as e:  # noqa: BLE001 — surface the reason (e.g. "Not logged in") to the UI
        return {"ok": False, "detail": str(e)[:200]}


async def text(system: str, user_prompt: str, *, model: str, max_tokens: int = 2048, thinking: bool = False):
    """CLI equivalent of the plain-text options_note create() call: returns (paragraph, usage dict)."""
    env = await _invoke(model, system, user_prompt, thinking=thinking)
    out = (env.get("result") or "").strip()
    if not out:
        raise CliError("claude CLI returned empty text")
    usage = _usage_from_env(model, env)
    log.info("analyst-options[cli] %s in=%s out=%s", model, usage["input_tokens"], usage["output_tokens"])
    return out, usage
