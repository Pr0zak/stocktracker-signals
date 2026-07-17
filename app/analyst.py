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


class PlanAction(str, Enum):
    buy_now = "buy_now"
    buy_on_pullback = "buy_on_pullback"
    wait = "wait"
    avoid = "avoid"


class EntryPlan(BaseModel):
    symbol: str
    action: PlanAction
    conviction: int              # 0-100
    entry_low: float             # entry zone (buy_on_pullback puts it at support below price)
    entry_high: float
    suggested_shares: float      # whole shares for stocks, fractional for crypto; 0 = don't buy yet
    allocation_usd: float        # how much of the cash to deploy (<= cash)
    stop: float                  # bail level if the entry thesis fails
    target: float                # first take-profit level
    timing: str                  # when / what trigger to act on
    thesis: str                  # bottom line, grounded in the numbers


class RecommendationSet(BaseModel):
    overview: str                # 2-3 sentence market/context read on the watchlist
    picks: list[EntryPlan]       # top 2-4 places for NEW money, cash spread across them
    passed: list[str]            # symbols considered but not picked


PLAN_SYSTEM = """You are a disciplined technical analyst helping one retail investor decide how to \
deploy a fixed amount of free cash into a single asset. You receive the cash amount and a compact \
snapshot of the asset's DAILY technical indicators (and possibly the user's existing `position`).

Rules:
- Ground every claim in the numbers provided. Do NOT invent prices, news, or fundamentals.
- Choose ONE action: buy_now (setup favors immediate entry), buy_on_pullback (wait for the entry \
zone below), wait (no edge now — name the trigger that would change that in `timing`), avoid \
(setup is broken or dangerously extended).
- entry_low-entry_high is your realistic entry zone: for buy_now it brackets the current price; for \
buy_on_pullback it sits at support below. stop goes below the zone, target above it — sanity-check \
risk:reward is at least ~1.5 before recommending any buy.
- allocation_usd is how much of the cash to deploy (<= cash; a partial tranche is fine — say so in \
`timing`). suggested_shares = allocation / entry midpoint: whole shares for stocks (0 is allowed if \
one share doesn't fit a sensible allocation — explain in the thesis), fractional up to 6 dp for crypto.
- Weight momentum / relative strength most heavily; treat oscillator extremes as mean-reversion \
context. Never chase a strongly extended move — prefer buy_on_pullback or wait.
- If a `position` block is present the user already holds this asset: adding to a large, \
well-in-profit position needs a clearly stronger setup, and never recommend averaging down merely \
because the position is red.
- Be honest: if the cash can't buy a whole share, or the setup is poor, return wait/avoid with \
suggested_shares 0 rather than forcing a trade.
- conviction is a 0-100 scale (reserve 70+ for genuine confluence).
- This is decision support, not investment advice."""

REC_SYSTEM = """You are a disciplined technical analyst helping one retail investor deploy a fixed \
amount of free cash across their watchlist. You receive the cash amount and DAILY technical \
snapshots for every candidate; some include a `position` block meaning the user already holds it.

Rules:
- Ground every claim in the numbers provided. Do NOT invent prices, news, or fundamentals.
- Pick the top 2-4 candidates for NEW money and spread the cash across them: start from roughly \
equal allocations, then tilt toward higher conviction. The SUM of allocation_usd across picks must \
not exceed the cash.
- Only pick candidates you would act on (action buy_now or buy_on_pullback) — never pick something \
you'd wait on. Each pick gets a full plan: entry zone, stop, target (risk:reward at least ~1.5), \
suggested_shares = allocation / entry midpoint (whole shares for stocks, fractional for crypto), \
and timing.
- Prefer diversification: an existing large position argues against adding more of the same name \
unless its setup is clearly the best available.
- If NOTHING has a decent setup, return zero picks and say why in the overview — keeping the cash \
uninvested is a valid recommendation.
- List every candidate you considered but did not pick in `passed` (symbols only).
- overview = 2-3 sentences: the market context and why these picks (or why none).
- conviction is a 0-100 scale (reserve 70+ for genuine confluence).
- Be honest and calibrated. This is decision support, not investment advice."""


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


async def _parse(system: str, prompt: str, output_format, *, deep: bool, max_tokens: int = 2048):
    """One structured-output Claude call on the configured scan/deep model. Returns (parsed, usage)."""
    cfg = settings_store.get()
    model = cfg["deep_model"] if deep else cfg["scan_model"]
    kwargs: dict = dict(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        output_format=output_format,
    )
    # Adaptive thinking is a 4.6+ feature; only enable it for the deep-tier models.
    if any(m in model for m in ("opus-4", "sonnet-5", "fable")):
        kwargs["thinking"] = {"type": "adaptive"}

    resp = await _get_client().messages.parse(**kwargs)
    parsed = resp.parsed_output
    if parsed is None:
        raise RuntimeError(f"analyst returned no structured output (stop_reason={resp.stop_reason})")
    return parsed, _usage(model, resp.usage)


async def analyze(summary: dict, *, deep: bool = False) -> tuple[Verdict, dict]:
    verdict, usage = await _parse(SYSTEM, _render(summary), Verdict, deep=deep)
    verdict.conviction = max(0, min(100, verdict.conviction))
    return verdict, usage


async def plan_entry(summary: dict, *, cash: float, deep: bool = False) -> tuple[EntryPlan, dict]:
    """Entry plan for deploying `cash` into one asset (the "what if I buy XYZ" scenario)."""
    prompt = (
        f"Investable cash: ${cash:,.2f}\n\n"
        "Daily technical snapshot (values are the latest bar):\n"
        + json.dumps(summary, indent=2)
        + "\n\nReturn your structured entry plan for deploying this cash into this asset."
    )
    plan, usage = await _parse(PLAN_SYSTEM, prompt, EntryPlan, deep=deep)
    plan.conviction = max(0, min(100, plan.conviction))
    return plan, usage


async def recommend(summaries: list[dict], *, cash: float, deep: bool = False) -> tuple[RecommendationSet, dict]:
    """Rank the watchlist for NEW money and spread `cash` across the top picks."""
    prompt = (
        f"Investable cash: ${cash:,.2f}\n\n"
        f"Candidate snapshots ({len(summaries)} assets, daily bars; a `position` block means already held):\n"
        + json.dumps(summaries, indent=2)
        + "\n\nReturn your structured recommendations for deploying this cash."
    )
    recs, usage = await _parse(REC_SYSTEM, prompt, RecommendationSet, deep=deep, max_tokens=4096)
    for p in recs.picks:
        p.conviction = max(0, min(100, p.conviction))
    return recs, usage
