"""
The Claude "analyst" — turns a compact technical snapshot into a structured, explained verdict.

Uses the Anthropic SDK's structured-output parse() so the response always validates against the
Verdict schema. Adaptive thinking on the deep (Opus) model; the cheap scan model runs without it.
"""
from __future__ import annotations

import json
import logging
from enum import Enum

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from . import llm_cli, settings_store

log = logging.getLogger("uvicorn.error")

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
price levels you were not provided — though you MAY cite an `insider` or `short_pressure` block when \
one is present in the snapshot.
- Weight momentum / relative strength most heavily (it is the best-evidenced factor). Treat RSI, \
Bollinger %B and Stochastic extremes as mean-reversion context, and moving-average structure as \
trend. A strongly extended move is a reason for caution, not for chasing.
- Be honest and calibrated: most technical signals do NOT reliably beat buy-and-hold after costs. \
When the picture is mixed or weak, return "hold" with lower conviction rather than forcing a call. \
conviction is a 0-100 scale — reserve 70+ for genuine confluence; a mixed picture is ~40-55.
- Always give a concrete invalidation (a price level or condition that would flip your view).
- thesis is the bottom line in AT MOST two short sentences (under 40 words). Put supporting detail \
in `rationale` (3-5 bullets, each under 15 words), risks in `key_risks` (2-3 short bullets). Never \
restate all the numbers in the thesis — the user sees the indicator values already.
- If the snapshot includes `rule_score`, it is the app's mechanical composite read of these SAME \
indicators on a 0-100 directional scale (50 neutral, higher bullish). If your verdict materially \
disagrees with it, add ONE rationale bullet starting "Vs rule score:" explaining the difference.
- If the snapshot includes `short_pressure` (days-to-cover, short-interest change, daily \
short-volume ratio, FTD trend, state quiet/fuel/ignition): the peer-reviewed base rate (Boehmer/ \
Jones/Zhang 2008; Asquith/Pathak/Ritter 2005; NBER w21166) is that high short interest — and \
especially high DAYS-TO-COVER, the sharper of the two — predicts UNDERperformance, not squeezes, \
because shorts are informed. So "fuel" (high days-to-cover) is a mildly BEARISH tilt and a \
two-sided risk amplifier, never a buy signal. This effect is concentrated in small/illiquid names \
and is weak-to-absent for large, liquid, value-weighted stocks — don't apply it to a mega-cap or \
broad ETF. Only treat squeeze mechanics as bullish when state is "ignition" (price AND volume \
confirming), and even then keep it rare and tightly invalidated: the SEC's GME 2021 report found \
short COVERING was only a small fraction of the buy volume (retail buying drove it), so a genuine \
covering squeeze is far rarer than folklore claims. FTDs are lagging/contemporaneous (they rise \
AFTER declines) — context, never timing. The daily short-volume ratio has no documented predictive \
value; treat it as descriptive only. If `ftd_spike_history` is present it is THIS \
symbol's own record after past FTD spikes — trust it over folklore (a negative median means spikes \
were NOT bullish here). If `upcoming_within_14d` lists near-term dates (SI publication, OPEX, \
earnings, speculative t35_echo), you may cite them in `catalysts` with their real meaning — an SI \
publication reveals positioning, OPEX affects hedging flows, a t35_echo is a speculative retail \
theory and must be labeled as such if mentioned.
- If the snapshot includes `insider` (open-market Form 4 PURCHASES over the last 12 months: \
buy_count_12m, buy_total_12m, largest_buy_value, and conviction/cluster flags): this is the BULLISH \
informed-money mirror of short_pressure. Insiders buying with their own money — especially a \
conviction buy (largest >= $500k) or a cluster (3+ insiders within 30 days) — is a modest POSITIVE \
base rate, since insiders sell for many reasons but mostly buy for one. Weigh it as confirming \
context, not timing: Form 4s lag the trade by up to ~2 business days, and it never overrides what \
price and momentum are doing. Absence of insider buying is NOT bearish.
- If the snapshot includes `quality` (ROE, gross/net margin, a debt-to-equity RATIO, and \
buffett_quality / wide_moat / dividend_aristocrat flags): these are stance-NEUTRAL descriptors of \
business durability, NOT a buy or sell call. A wide-moat, low-debt, high-ROE compounder is a \
structurally safer base to be constructive on, and a leveraged, low-margin business warrants more \
caution — but quality does not time entries and must never override what price and momentum are doing.
- Crypto snapshots may include `long_term_trend` (price vs 200-week SMA, Mayer Multiple, distance \
from ATH, 3y CAGR) and, for BTC, `btc_halving_cycle` (cycle position, phase, and a past-cycle \
analog of 12-month-forward returns from this position). Use them to frame the MULTI-YEAR regime — \
price above a rising 200-week SMA is a structurally healthier base than below it — but respect the \
attached sample-size note: four halvings is anecdote-grade evidence. Never let cycle folklore \
override what current price and momentum are actually doing.
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
- If the snapshot includes `short_pressure`: high short interest is a bearish base rate, not a buy \
signal — do not plan an entry INTO a heavily-shorted name unless its state is "ignition", and then \
size smaller with a tighter stop (squeezes reverse fast). Use `upcoming_within_14d` to time entries \
around known dates (e.g. wait out earnings) and treat any t35_echo date as speculative.
- This is decision support, not investment advice."""

REC_SYSTEM = """You are a disciplined technical analyst helping one retail investor deploy a fixed \
amount of free cash. You receive the cash amount and DAILY technical snapshots for every candidate; \
some include a `position` block meaning the user already holds it. Each snapshot's `source` says \
where it came from: "watchlist" (a name the user follows) or "market_screen" (discovered via live \
market screens — actives, gainers, growth, value). Judge every candidate purely on merit; when you \
pick a market_screen name, note in its thesis why it beat the watchlist alternatives.

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
- Stay COMPACT: each pick's thesis and timing at most 25 words each. The full response must not run
long — verbosity gets it truncated.
- conviction is a 0-100 scale (reserve 70+ for genuine confluence).
- If snapshots include `short_pressure`: high short interest is a bearish base rate. Never pick a \
heavily-shorted name as a squeeze play unless its state is "ignition", and say so explicitly with \
smaller sizing and a tighter stop. Weigh `upcoming_within_14d` dates (earnings, OPEX) when choosing \
between otherwise-similar candidates; t35_echo dates are speculative.
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
        "cache_write_tokens": cache_write,
        "cost_usd": round(cost, 6),
        "provider": "api",
    }


def _render(summary: dict) -> str:
    return (
        "Daily technical snapshot (values are the latest bar):\n"
        + json.dumps(summary, indent=2)
        + "\n\nReturn your structured verdict."
    )


async def _parse(system: str, prompt: str, output_format, *, deep: bool, max_tokens: int = 4096):
    """One structured-output Claude call on the configured scan/deep model. Returns (parsed, usage)."""
    cfg = settings_store.get()
    model = cfg["deep_model"] if deep else cfg["scan_model"]
    # Adaptive thinking is a deep-tier (Opus/Sonnet/Fable) feature; the cheap scan model runs without it.
    thinking_model = any(m in model for m in ("opus-4", "sonnet-5", "fable"))
    # Provider toggle: "cli" shells out to the headless claude CLI (subscription OAuth, no per-token
    # billing); the default "api" path below uses the Anthropic SDK's schema-constrained parse().
    if cfg.get("llm_provider") == "cli":
        return await llm_cli.structured(system, prompt, output_format, model=model,
                                        max_tokens=max_tokens, thinking=thinking_model)
    kwargs: dict = dict(
        model=model,
        max_tokens=max_tokens,
        # Prompt caching was measured here and REMOVED — it doesn't help this workload:
        #  • Haiku scan (the bulk of the cost): on the output_format path the cacheable prefix is the
        #    system prompt only (~1.3k tokens; the schema isn't a cached tool there), under Haiku's ~2k
        #    minimum — so cache_control was a silent no-op (cache_write=0 across a 24-symbol scan).
        #  • Opus deep: it DID cache (~2.8k prefix) but deep calls are usually one-off, so the +25%
        #    cache-write premium never gets recouped by a within-TTL read.
        # The Batch API (~50% off input AND output) is the real scan lever — see scan_job.py.
        # The per-call in/out/cache token log below stays, for ongoing cost visibility.
        system=system,
        messages=[{"role": "user", "content": prompt}],
        output_format=output_format,
    )
    # Adaptive thinking is a 4.6+ feature; only enable it for the deep-tier models.
    if thinking_model:
        kwargs["thinking"] = {"type": "adaptive"}

    resp = await _get_client().messages.parse(**kwargs)
    u0 = resp.usage
    log.info(
        "analyst %s in=%s out=%s cache_write=%s cache_read=%s",
        model, u0.input_tokens, u0.output_tokens,
        getattr(u0, "cache_creation_input_tokens", 0) or 0,
        getattr(u0, "cache_read_input_tokens", 0) or 0,
    )
    if resp.stop_reason == "max_tokens":
        # Truncated JSON parses as garbage — fail with a clear cause instead of a pydantic stack.
        raise RuntimeError(f"analyst output hit the {max_tokens}-token cap and was truncated — retry")
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
    recs, usage = await _parse(REC_SYSTEM, prompt, RecommendationSet, deep=deep, max_tokens=8192)
    for p in recs.picks:
        p.conviction = max(0, min(100, p.conviction))
    return recs, usage


# ======================================================================================
# OC-7 — one plain-language paragraph explaining a suggested options contract (deep=Opus path).
# Plain text (not structured) — reuses the same client/key handling as the verdict path above.
# ======================================================================================

OPTIONS_NOTE_SYSTEM = """You are a friendly options coach explaining ONE specific suggested options \
trade to a BEGINNER retail investor in plain language. You receive a JSON context: the symbol and \
its price, a go/no-go "light" (green/yellow/red) with its reason, the suggested long-CALL contract \
(strike, expiry, cost, breakeven, delta), the mechanical directional read, any earnings-before-expiry \
flag, an IV-rank read, and possibly a cheaper debit-spread alternative.

Write ONE short paragraph (3-5 sentences, under ~90 words) that ties THIS contract to the directional \
thesis, for a beginner:
- Say what the trade is betting on and by when (ground it ONLY in the numbers given — never invent \
news, prices, or fundamentals).
- Mention the light and the single most important caution (earnings / high IV / thin liquidity) if \
there is one.
- If IV rank is high and a debit-spread alternative is offered, note the spread is the cheaper, \
lower-risk way to express the same view.
- Plain words, no jargon dumps, no bullet points or headers — just one paragraph.
- End by reminding this is decision support, not investment advice."""


async def options_note(context: dict, *, deep: bool = True) -> tuple[str, dict]:
    """One plain-language paragraph explaining a suggested options contract for a beginner (OC-7).
    Returns (paragraph, usage). Uses the deep (Opus) model by default — this is the on-demand
    "explain it to me" path, mirroring how /plan calls the analyst. Raises on API/empty output so the
    route can swallow it and leave `analyst` null (never a 500)."""
    cfg = settings_store.get()
    model = cfg["deep_model"] if deep else cfg["scan_model"]
    thinking_model = any(m in model for m in ("opus-4", "sonnet-5", "fable"))
    prompt = (
        "Explain this suggested options trade to a beginner:\n"
        + json.dumps(context, indent=2, default=str)
        + "\n\nReturn ONE short plain-language paragraph."
    )
    if cfg.get("llm_provider") == "cli":
        return await llm_cli.text(OPTIONS_NOTE_SYSTEM, prompt, model=model, max_tokens=2048, thinking=thinking_model)
    kwargs: dict = dict(
        model=model,
        max_tokens=2048,
        system=OPTIONS_NOTE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    # Adaptive thinking is a 4.6+ feature; only enable it for the deep-tier models (matches _parse).
    if thinking_model:
        kwargs["thinking"] = {"type": "adaptive"}

    resp = await _get_client().messages.create(**kwargs)
    u0 = resp.usage
    log.info("analyst-options %s in=%s out=%s", model, u0.input_tokens, u0.output_tokens)
    if resp.stop_reason == "max_tokens":
        raise RuntimeError(f"options paragraph hit the 2048-token cap and was truncated (stop_reason={resp.stop_reason})")
    text = "".join(
        getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text"
    ).strip()
    if not text:
        raise RuntimeError(f"analyst returned no text (stop_reason={resp.stop_reason})")
    return text, _usage(model, resp.usage)


# ======================================================================================
# AIE-5 — "Market now": a fast plain-language read of what the whole tape is doing right now.
# Plain text (like options_note), routed through the same provider toggle. Defaults to the cheap
# scan model so the on-demand button feels instant.
# ======================================================================================

MARKET_SYSTEM = """You are a market strategist giving one retail investor a fast, plain-language read of \
what US markets are doing RIGHT NOW. You receive a JSON snapshot: the session phase (PRE / REGULAR / \
AFTER / CLOSED), the major indices (S&P 500, Nasdaq, Dow, Russell 2000) with % change, the VIX level and \
its change, the 11 SPDR sector ETFs split into leaders and laggards by % change, market-wide top \
gainers/losers, and the user's own watchlist movers.

Write a tight, scannable overview — 3 to 5 short bullets, under ~130 words total:
- Lead with the tape's tone: risk-on / risk-off / mixed, grounded in the indices AND the VIX (a rising \
VIX is fear; small-caps (Russell) leading or lagging tells you about risk appetite).
- Name what is LEADING and LAGGING by sector and what it implies (e.g. defensives — staples/utilities — \
bid while tech lags = a cautious, rotational tape).
- Call out the 1-2 most notable moves in the user's WATCHLIST specifically, if any stand out.
- Ground EVERY claim in the numbers provided. Do NOT invent news, catalysts, earnings, or price levels \
you were not given; if there's no obvious driver in the data, say the move is happening without one in view.
- If the session is PRE or AFTER, say so and warn the moves are on thin, unreliable volume; if CLOSED, \
frame it as where things settled, not live action.
- No preamble, no headers. End with a short 'Not investment advice.'"""


async def market_overview(snapshot: dict, *, deep: bool = False) -> tuple[str, dict]:
    """One plain-language 'what are the markets doing right now' overview (AIE-5). Returns (text, usage).
    Defaults to the cheap scan model (deep=False) for an instant feel; deep=True uses Opus for a richer
    synthesis. Mirrors options_note (plain text) and honors the api/cli provider toggle. Raises on
    API/empty output so the route can turn it into a clean 502."""
    cfg = settings_store.get()
    model = cfg["deep_model"] if deep else cfg["scan_model"]
    thinking_model = any(m in model for m in ("opus-4", "sonnet-5", "fable"))
    prompt = (
        "Here is the current market snapshot. Give the 'what's happening right now' overview:\n"
        + json.dumps(snapshot, indent=2, default=str)
    )
    if cfg.get("llm_provider") == "cli":
        return await llm_cli.text(MARKET_SYSTEM, prompt, model=model, max_tokens=1024, thinking=thinking_model)
    kwargs: dict = dict(
        model=model,
        max_tokens=1024,
        system=MARKET_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    if thinking_model:
        kwargs["thinking"] = {"type": "adaptive"}
    resp = await _get_client().messages.create(**kwargs)
    u0 = resp.usage
    log.info("analyst-market %s in=%s out=%s", model, u0.input_tokens, u0.output_tokens)
    if resp.stop_reason == "max_tokens":
        raise RuntimeError(f"market overview hit the 1024-token cap and was truncated (stop_reason={resp.stop_reason})")
    text = "".join(
        getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text"
    ).strip()
    if not text:
        raise RuntimeError(f"analyst returned no market text (stop_reason={resp.stop_reason})")
    return text, _usage(model, resp.usage)
