"""
The Claude "analyst" — turns a compact technical snapshot into a structured, explained verdict.

Uses the Anthropic SDK's structured-output parse() so the response always validates against the
Verdict schema. Adaptive thinking on the deep (Opus) model; the cheap scan model runs without it.
"""
from __future__ import annotations

import json
from enum import Enum

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from . import settings_store

_client: AsyncAnthropic | None = None
_client_key: str | None = None


def _get_client() -> AsyncAnthropic:
    """Lazy client, rebuilt when the API key is changed via the settings UI."""
    global _client, _client_key
    key = settings_store.get()["anthropic_api_key"]
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured — set it in the settings UI at /")
    if _client is None or key != _client_key:
        _client = AsyncAnthropic(api_key=key)
        _client_key = key
    return _client


class Signal(str, Enum):
    strong_buy = "strong_buy"
    buy = "buy"
    hold = "hold"
    sell = "sell"
    strong_sell = "strong_sell"


class Verdict(BaseModel):
    signal: Signal
    conviction: int              # 0-100
    horizon: str                 # e.g. "swing (days-weeks)" or "position (weeks-months)"
    thesis: str                  # one-sentence bottom line
    rationale: list[str]         # grounded in the provided numbers
    key_risks: list[str]         # what would make this wrong
    invalidation: str            # concrete price level / condition to bail
    catalysts: list[str]         # known events ahead (empty if none provided)


SYSTEM = """You are a disciplined technical analyst assisting one retail investor's personal \
portfolio-tracking app. You receive a compact snapshot of a single asset's DAILY technical \
indicators and must return a structured buy/sell read.

Rules:
- Ground every claim in the numbers you are given. Do NOT invent fundamentals, news, earnings, or \
price levels you were not provided.
- Weight momentum / relative strength most heavily (it is the best-evidenced factor). Treat RSI, \
Bollinger %B and Stochastic extremes as mean-reversion context, and moving-average structure as \
trend. A strongly extended move is a reason for caution, not for chasing.
- Be honest and calibrated: most technical signals do NOT reliably beat buy-and-hold after costs. \
When the picture is mixed or weak, return "hold" with lower conviction rather than forcing a call. \
Reserve high conviction for genuine confluence.
- Always give a concrete invalidation (a price level or condition that would flip your view).
- If the snapshot includes `recent_news` (headlines) or `next_earnings` (a date), use them to \
populate `catalysts` and sharpen `key_risks` — but do NOT invent news beyond what is provided.
- If the snapshot includes a `position` block, the user ALREADY HOLDS this asset (shares, average \
cost, position value, unrealized gain %). Frame the verdict as an action ON THAT POSITION: read \
"buy" as ADD, "sell" as TRIM / reduce, "hold" as keep-as-is, and say which in the thesis. Weigh the \
position's SIZE and unrealized gain — a large, well-in-profit position argues for protecting gains \
(trim into strength, tighter invalidation) over adding; a small or modestly-underwater position near \
support may justify adding. Never advise averaging down merely because it is red. Set the invalidation \
relative to their average cost when relevant. Judge size only from the numbers given — you do NOT \
know their total net worth or other holdings, so don't assume overall concentration.
- This is decision support, not investment advice."""


# Public pricing per 1M tokens (input, output) — keep in sync with the claude-api reference.
_PRICING = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5": (10.0, 50.0),
}


def _usage(model: str, u) -> dict:
    """Token counts + an estimated USD cost for one call (cache reads/writes priced in if present)."""
    in_rate, out_rate = _PRICING.get(model, (5.0, 25.0))
    cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
    cost = (
        u.input_tokens * in_rate
        + u.output_tokens * out_rate
        + cache_read * in_rate * 0.1
        + cache_write * in_rate * 1.25
    ) / 1_000_000
    return {
        "model": model,
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_read_tokens": cache_read,
        "cost_usd": round(cost, 6),
    }


def _render(summary: dict) -> str:
    return (
        "Daily technical snapshot (values are the latest bar):\n"
        + json.dumps(summary, indent=2)
        + "\n\nReturn your structured verdict."
    )


async def analyze(summary: dict, *, deep: bool = False) -> tuple[Verdict, dict]:
    cfg = settings_store.get()
    model = cfg["deep_model"] if deep else cfg["scan_model"]
    kwargs: dict = dict(
        model=model,
        max_tokens=2048,
        system=SYSTEM,
        messages=[{"role": "user", "content": _render(summary)}],
        output_format=Verdict,
    )
    # Adaptive thinking is a 4.6+ feature; only enable it for the deep-tier models.
    if any(m in model for m in ("opus-4", "sonnet-5", "fable")):
        kwargs["thinking"] = {"type": "adaptive"}

    resp = await _get_client().messages.parse(**kwargs)
    verdict = resp.parsed_output
    if verdict is None:
        raise RuntimeError(f"analyst returned no structured verdict (stop_reason={resp.stop_reason})")
    verdict.conviction = max(0, min(100, verdict.conviction))
    return verdict, _usage(model, resp.usage)
