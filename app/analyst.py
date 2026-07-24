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


class VerdictLevels(BaseModel):
    """Clean numeric price levels for the chart annotation (AIE-1). Any field may be null."""
    support: float | None = None             # nearest meaningful support below price
    resistance: float | None = None          # nearest meaningful resistance above price
    invalidation_price: float | None = None  # the numeric level behind `invalidation`
    target: float | None = None              # first realistic upside target


class Verdict(BaseModel):
    signal: Signal
    conviction: int              # 0-100
    horizon: str                 # e.g. "swing (days-weeks)" or "position (weeks-months)"
    thesis: str                  # one-sentence bottom line
    rationale: list[str]         # grounded in the provided numbers
    key_risks: list[str]         # what would make this wrong
    invalidation: str            # concrete price level / condition to bail
    catalysts: list[str]         # known events ahead (empty if none provided)
    levels: VerdictLevels | None = None  # numeric chart levels (AIE-1); null on older/rejected outputs


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
- Also fill `levels` — clean NUMERIC price levels for the chart, grounded ONLY in the price/indicator \
numbers provided: support (nearest meaningful support below the current price), resistance (nearest \
above), invalidation_price (the numeric price behind your `invalidation` condition), and target (a first \
realistic upside level in the direction of your call). Use null for any level you cannot justify from the \
numbers you were given — never invent a precise level.
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
- If the snapshot includes `congress` (recent House/Senate/cabinet trades in this name: buy_count, \
sell_count, net_direction, distinct_filers, cluster_buy, largest_buy_amount_high, parties, and the \
latest filers): treat this as WEAK, LAGGING context, NOT a signal. Under the STOCK Act a trade can be \
disclosed up to ~45 days after it happens, so by the time you see it the move is old, and the academic \
evidence that politicians beat the market is thin and heavily debated — do NOT trade on "follow the \
politician." The only mildly interesting reads are: a genuine CLUSTER (cluster_buy = several distinct \
members buying within 30 days), unusually large size (largest_buy_amount_high), or a committee-relevant \
name — and even then it is confirming color at most, never a reason to override price and momentum. \
Mention it in at most ONE rationale bullet if it is notable; otherwise ignore it. Selling by members \
is especially noisy (liquidity, taxes, blind trusts) — do not read it as bearish.
- If the snapshot includes `quality` (ROE, gross/net margin, a debt-to-equity RATIO, and \
buffett_quality / wide_moat / dividend_aristocrat flags): these are stance-NEUTRAL descriptors of \
business durability, NOT a buy or sell call. A wide-moat, low-debt, high-ROE compounder is a \
structurally safer base to be constructive on, and a leveraged, low-margin business warrants more \
caution — but quality does not time entries and must never override what price and momentum are doing.
- If the snapshot includes `seasonality` (current_month avg return + hit_rate over ~N years, plus the \
best/worst calendar months): this is the stock's TYPICAL price action for the calendar month, a WEAK, \
sample-limited tilt (only a handful of years per month, and seasonal edges decay as they get known). \
Treat a strong/weak current month as MILD confirming or cautioning context — e.g. a historically weak \
month with a 30% hit rate is a small reason to be patient — never as a standalone timing signal, and it \
must never override what price and momentum are doing right now. Mention it in at most one rationale \
bullet if the current month is notably strong or weak; otherwise ignore it. Always respect the small sample.
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
# AIE-5 — "Market now": a fast, STRUCTURED read of the tape right now (tone + headline + a few scannable
# points) so the app renders it grouped, not as one paragraph. Routed through the shared structured
# _parse() (api/cli). Defaults to the cheap scan model so the on-demand button feels instant.
# ======================================================================================

class MarketOverview(BaseModel):
    tone: str          # "risk-on" | "risk-off" | "mixed"
    headline: str      # one punchy bottom-line sentence
    points: list[str]  # 3-5 short, scannable points


MARKET_SYSTEM = """You are a market strategist giving one retail investor a fast, SCANNABLE read of \
what US markets are doing RIGHT NOW. You receive a JSON snapshot: the session phase (PRE / REGULAR / \
AFTER / CLOSED), major indices (S&P 500, Nasdaq, Dow, Russell 2000) with % change, the VIX level + change, \
the 11 SPDR sector ETFs split into leaders/laggards, market-wide top gainers/losers, and the user's \
watchlist movers.

Return a structured read (it renders as a tone chip + a bold headline + bullet points on a phone):
- tone: exactly one of "risk-on", "risk-off", or "mixed" — the overall tape, grounded in indices AND the \
VIX (a rising VIX is fear; small-caps (Russell) leading/lagging signals risk appetite).
- headline: ONE punchy bottom-line sentence, under ~14 words.
- points: 3 to 5 SHORT bullets, each ONE sentence under ~22 words, in this order: (1) indices + VIX read; \
(2) sector rotation — what's leading/lagging and what it implies (e.g. defensives bid + tech lagging = \
cautious); (3) the 1-2 most notable moves in the user's WATCHLIST by name; (4) if PRE/AFTER, a thin-volume \
caveat, or if CLOSED, that it's where things settled.
Ground EVERY point in the numbers — never invent news, catalysts, earnings, or price levels; if there's \
no obvious driver, say the move is happening without one in view. Do NOT add a disclaimer line (the app \
shows one). Each string is plain text, no markdown."""


async def market_overview(snapshot: dict, *, deep: bool = False) -> tuple[MarketOverview, dict]:
    """Structured 'what are the markets doing right now' read (AIE-5): tone + headline + 3-5 points, so
    the app can render it grouped rather than as one paragraph. Defaults to the cheap scan model
    (deep=False) for an instant feel; honors the api/cli provider toggle via the shared structured
    _parse(). Raises on API/empty output so the route can turn it into a clean 502."""
    prompt = (
        "Here is the current market snapshot. Give the structured 'what's happening right now' read:\n"
        + json.dumps(snapshot, indent=2, default=str)
    )
    return await _parse(MARKET_SYSTEM, prompt, MarketOverview, deep=deep, max_tokens=1024)


# ======================================================================================
# AI daily brief (AIE-3) — a once-a-morning push: the tape, the user's names on the move, and any
# catalyst landing today, compressed into a notification title + a couple of sentences.
# ======================================================================================

class DailyBrief(BaseModel):
    title: str  # notification title — a punchy morning headline, a few words
    body: str   # 2-3 sentence brief for the expanded notification
    tone: str   # "risk-on" | "risk-off" | "mixed"


DAILY_BRIEF_SYSTEM = """You are writing ONE retail investor's morning market brief — it lands as a phone \
push notification before/around the US open, so it must be tight and scannable. You receive a JSON \
snapshot: the session phase (PRE / REGULAR / AFTER / CLOSED), major indices (S&P 500, Nasdaq, Dow, \
Russell 2000) with % change, the VIX level + change, the SPDR sectors split into leaders/laggards, \
market-wide top movers, the user's WATCHLIST movers, and `catalysts_today` — a list of the user's \
tickers reporting earnings (or other dated events) today.

Return a structured brief:
- title: the notification TITLE — a punchy morning headline UNDER ~8 words, leading with the tape's tone \
or the single most notable item (e.g. "Futures soft; NVDA reports today").
- body: 2 to 3 SHORT sentences, UNDER ~55 words total, in this priority order: (1) the tape — indices + \
VIX in one line; (2) the 1-2 most notable moves in the user's OWN watchlist, by name; (3) any \
`catalysts_today` — name who reports today (this is the highest-value line when present). Skip a bucket \
if there's nothing worth saying; never pad.
- tone: exactly one of "risk-on", "risk-off", or "mixed", grounded in indices AND the VIX.
Ground EVERY claim in the snapshot numbers — never invent news, price levels, or catalysts not in \
`catalysts_today`. In PRE phase, treat moves as thin-volume futures/pre-market. Plain text, no markdown, \
no disclaimer line (the app adds one)."""


async def daily_brief(snapshot: dict, *, deep: bool = False) -> tuple[DailyBrief, dict]:
    """Structured morning brief (AIE-3): a notification title + 2-3 sentences + tone, from the same live
    snapshot market_now uses (plus today's watchlist catalysts). Cheap scan model by default; honors the
    api/cli provider toggle via the shared _parse(). Raises on API/empty output so the route 502s cleanly."""
    prompt = (
        "Here is this morning's market snapshot. Write the structured daily brief:\n"
        + json.dumps(snapshot, indent=2, default=str)
    )
    return await _parse(DAILY_BRIEF_SYSTEM, prompt, DailyBrief, deep=deep, max_tokens=768)


# ======================================================================================
# Market regime (Theme D) — a structural read of what KIND of market this is (trend + volatility),
# beyond market_now's "right this second" snapshot, with a positioning implication.
# ======================================================================================

class MarketRegime(BaseModel):
    label: str        # short 2-4 word regime label, e.g. "Risk-on uptrend"
    trend: str        # "up" | "down" | "sideways"
    volatility: str   # "calm" | "normal" | "elevated" | "stressed"
    note: str         # 1-2 sentences: what it means for positioning


REGIME_SYSTEM = """You classify the current US market REGIME — the structural backdrop, not the intraday \
tape. You receive a JSON snapshot: session phase, the major indices (S&P 500, Nasdaq, Dow, Russell 2000) \
with today's % change, the VIX level, the SPDR sectors split into leaders/laggards, the user's watchlist \
movers, and `spy_trend` — the S&P 500's structural trend: % vs its 50-day and 200-day moving averages, \
whether it's above the 200-day, RSI, and MACD histogram.

Return a structured regime read:
- label: a SHORT 2-4 word regime name (e.g. "Risk-on uptrend", "Choppy range", "Risk-off pullback", \
"High-volatility stress", "Grinding recovery").
- trend: exactly one of "up", "down", or "sideways" — grounded in the S&P vs its 50/200-day and whether \
the four indices confirm each other (small-caps leading = broad risk appetite; only mega-cap up = narrow).
- volatility: exactly one of "calm" (VIX <15), "normal" (15-20), "elevated" (20-30), or "stressed" (>30).
- note: 1-2 SHORT sentences on what this regime means for positioning — e.g. "Trend intact above both \
MAs; pullbacks toward the 50-day have been buyable" or "Below the 200-day with a rising VIX argues for \
smaller size and defense". Base it on structure (MAs, VIX, breadth), NEVER a price prediction.
Ground every field in the numbers. Plain text, no markdown, no disclaimer line (the app adds one)."""


async def market_regime(snapshot: dict, *, deep: bool = False) -> tuple[MarketRegime, dict]:
    """Structural market-regime classification (Theme D): trend + volatility + a positioning note, from
    the market snapshot plus the S&P's 50/200-day trend. Honors the api/cli provider toggle via _parse()."""
    prompt = (
        "Here is the market snapshot with the S&P's structural trend. Classify the current regime:\n"
        + json.dumps(snapshot, indent=2, default=str)
    )
    return await _parse(REGIME_SYSTEM, prompt, MarketRegime, deep=deep, max_tokens=512)


# ======================================================================================
# News → move correlation (AIE-4) — line the stock's notable recent daily moves up against dated
# headlines and judge which move was news-driven and which happened on flows/technicals alone.
# ======================================================================================

class NewsDriver(BaseModel):
    date: str               # YYYY-MM-DD of the move
    move_pct: float         # that day's % change
    headline: str | None = None  # the headline that best explains it, verbatim; null if none fits
    explanation: str        # one sentence: what drove the move (or "no clear catalyst — flows/technicals")


class NewsMoves(BaseModel):
    summary: str            # one-line read: is the stock trading on news, or on flows/technicals?
    drivers: list[NewsDriver]


NEWS_MOVES_SYSTEM = """You explain WHY a stock moved. You receive a ticker, a list of its NOTABLE recent \
daily moves (each: date + % change), and a list of dated company-news headlines (date, headline, source). \
Your job is to correlate the two — HONESTLY, without inventing anything.

Return a structured read:
- summary: ONE sentence — is this stock trading on news/catalysts right now, or drifting on flows and \
technicals with no obvious headline driver? Ground it in what you actually see.
- drivers: EXACTLY ONE entry per notable move you were given, most-recent first. For each: echo its date \
and move_pct; set `headline` to the ONE provided headline (verbatim) that best explains that day's move \
(same day, or the trading day before an open gap) or null if none genuinely fits; and write `explanation` \
— one sentence tying the move to that news, OR, when nothing fits, say plainly it moved with no clear \
catalyst in the headlines (a flows/technical/market-beta move).
Rules: NEVER invent a headline or attribute a move to news that isn't in the list. Correlation is not \
proof — hedge with "likely"/"appears". A big move with no matching headline is a REAL and useful finding, \
not a failure — say so. Plain text, no markdown, no disclaimer line (the app adds one)."""


async def news_moves(
    symbol: str, moves: list[dict], news: list[dict], *, deep: bool = False,
) -> tuple[NewsMoves, dict]:
    """Correlate a stock's notable recent daily moves with dated headlines (AIE-4). `moves` is
    [{date, move_pct}], `news` is [{date, headline, source, ...}]. Cheap scan model by default; honors
    the api/cli provider toggle via _parse(). Raises on API/empty output so the route 502s cleanly."""
    prompt = (
        f"Ticker: {symbol}\n\n"
        f"Notable recent daily moves (date, % change):\n{json.dumps(moves, indent=2, default=str)}\n\n"
        f"Dated company-news headlines:\n{json.dumps(news, indent=2, default=str)}\n\n"
        "Correlate the moves with the headlines and return the structured read."
    )
    return await _parse(NEWS_MOVES_SYSTEM, prompt, NewsMoves, deep=deep, max_tokens=1536)


# ======================================================================================
# Portfolio review — a structured, whole-portfolio read: concentration, a per-holding action list,
# and cash deployment. One structured call over lightweight technical snapshots of each holding.
# ======================================================================================

class PortfolioAction(BaseModel):
    symbol: str
    action: str          # "trim" | "hold" | "add" | "watch"
    reason: str          # one short sentence, grounded in the numbers


class PortfolioReview(BaseModel):
    health: str                    # one-line overall read of the book
    concentration: list[str]       # concentration / diversification flags (empty if well-diversified)
    actions: list[PortfolioAction]  # one entry per holding
    cash_note: str                 # what to do with idle cash


PORTFOLIO_SYSTEM = """You are a disciplined portfolio strategist reviewing one retail investor's WHOLE \
stock/crypto portfolio. You receive: cash + cash_pct, total_value, a `positions` list where each holding \
has its weight_pct, unrealized_gain_pct, price, an `exposure_group`, and key technicals (RSI, MACD \
histogram, % vs the 50-day MA, golden-cross flag, 3-month relative strength vs the S&P, and % off its \
52-week high), and `equivalent_exposures` — a map of any groups the user holds MORE THAN ONE vehicle of.

Positions that share an `exposure_group` are the SAME underlying economic exposure (e.g. BTC and a spot- \
bitcoin ETF like FBTC/IBIT are both just bitcoin; SPY and VOO are both the S&P 500). Their technicals \
will look nearly identical — do NOT treat them as independent bets.

Return a structured review:
- health: ONE sentence on the book's overall posture (diversified vs concentrated, momentum tilt, cash level).
- concentration: flag genuine risks — any single exposure over ~20-25% of the book (SUM the weights of an \
exposure_group before judging — e.g. BTC + FBTC together), several holdings that are clearly the same \
theme/sector, or a very low cash buffer. If `equivalent_exposures` is non-empty, flag that the user holds \
redundant vehicles for the same thing and suggest consolidating into the cheaper/more-liquid one. Empty \
list if it's reasonably balanced. Be specific ("BTC exposure is 40% of the book across BTC + FBTC").
- actions: EXACTLY ONE entry per holding, action ∈ trim | hold | add | watch, with a one-sentence reason \
grounded in ITS numbers: trim a large, well-in-profit, extended winner (protect gains / rebalance); add to \
an underweight name with a genuinely strong setup; hold when there's no edge; watch when it's weakening but \
not yet actionable. NEVER advise averaging down just because a position is red. Holdings in the SAME \
exposure_group MUST get a consistent stance — never tell the user to add one and trim its equivalent (a \
pointless wash); pick ONE vehicle to act on and hold the other.
- cash_note: what to do with the idle cash (deploy into the best-setup adds, keep dry powder if nothing's \
compelling, etc.), consistent with your actions.

Ground everything in the numbers. You do NOT know their total net worth, tax situation, or holdings outside \
this list — judge concentration only within this book and say so if it matters. This is decision support, \
not investment advice."""


async def review_portfolio(portfolio: dict, *, cash: float, deep: bool = False) -> tuple[PortfolioReview, dict]:
    """Structured whole-portfolio review: health, concentration flags, a per-holding action list, and a
    cash note. Honors the api/cli provider toggle via the shared structured _parse()."""
    prompt = (
        "Review this portfolio. Each position carries its weight, unrealized gain, and key technicals.\n"
        + json.dumps(portfolio, indent=2, default=str)
        + "\n\nReturn your structured portfolio review (exactly one action per holding)."
    )
    return await _parse(PORTFOLIO_SYSTEM, prompt, PortfolioReview, deep=deep, max_tokens=4096)


# ======================================================================================
# Portfolio rebalance (Theme C) — turns the review's trim/add judgment into CONCRETE sized moves:
# sell N shares of the over-weights, redeploy the proceeds + idle cash into the best-setup existing
# holdings, respecting a target max single-position weight. Actionable for manual (Fidelity) trading.
# ======================================================================================

class RebalanceMove(BaseModel):
    symbol: str
    action: str          # "sell" | "buy" | "hold"
    shares: float        # share count to trade (0 for hold)
    dollars: float       # approximate $ value of the trade (0 for hold)
    reason: str          # one short sentence, grounded in weight + setup


class RebalancePlan(BaseModel):
    summary: str                          # one-line read of what the plan does + the resulting book
    moves: list[RebalanceMove]            # concrete sized moves (one per holding worth touching; holds allowed)
    resulting_top_weight_pct: float | None = None  # est. largest position weight AFTER the moves
    cash_after: float | None = None       # est. leftover cash after the buys


REBALANCE_SYSTEM = """You are a disciplined portfolio strategist producing a CONCRETE, ACTIONABLE \
rebalance plan for one retail investor who trades MANUALLY (so give real share counts and dollar \
amounts they can enter). You receive: cash + cash_pct, total_value, a `max_position_pct` target (the \
largest weight any single EXPOSURE should have after rebalancing), a `positions` list where each holding \
has its price, shares, value, weight_pct, unrealized_gain_pct, an `exposure_group`, and key technicals \
(RSI, MACD histogram, % vs 50-day MA, golden-cross, 3-month relative strength vs the S&P, % off 52-week \
high), and `equivalent_exposures` — a map of any groups the user holds more than one vehicle of.

Positions sharing an `exposure_group` are the SAME underlying exposure (e.g. BTC and a spot-bitcoin ETF \
like FBTC/IBIT; SPY and VOO). Judge the max-weight target against the COMBINED weight of a group, act on \
ONE vehicle per group (prefer the more-liquid / lower-cost one), give the group a consistent direction, \
and NEVER sell one vehicle to buy its equivalent — that's a wash that just churns fees.

Produce a plan that ONLY trades the EXISTING holdings + deploys the idle cash (do NOT introduce new \
tickers — that's a different tool):
- summary: ONE sentence — what the plan does and the resulting posture (e.g. "Trims NVDA from 61% to \
~40% and rotates ~$900 into AAPL, leaving a balanced two-name book").
- moves: for each holding worth touching, an entry with action ∈ sell | buy | hold. For sell/buy give a \
concrete `shares` and approximate `dollars` (shares × current price). RULES: (1) trim any position over \
`max_position_pct` back toward that target — compute the dollars to sell = value − max_position_pct% × \
total_value, then shares = that ÷ price (round to whole shares for stocks; fractional OK for crypto). \
(2) Redeploy the sell proceeds + idle cash into the underweight holdings with the genuinely BEST \
technical setup (strong RS, above 50-day MA, MACD positive) — size those buys so you don't spend more \
than proceeds + cash and don't push a buy target back over the max. (3) Use `hold` (shares 0) for names \
that are fine as-is. NEVER add to a clearly weak/downtrending name just to spend cash — leave it as cash.
- resulting_top_weight_pct: your estimate of the largest single weight after the moves.
- cash_after: your estimate of leftover cash after the buys.

Trimming a big winner realizes gains — mention that in the relevant reason when it applies. You do NOT \
know their tax situation, total net worth, or outside holdings; judge only within this book. Numbers must \
tie out (buys ≤ sells + cash). This is decision support, not investment advice."""


async def rebalance_portfolio(
    portfolio: dict, *, max_position_pct: float, deep: bool = False,
) -> tuple[RebalancePlan, dict]:
    """Concrete sized rebalance moves (Theme C) from the same portfolio snapshot the review uses, plus a
    target max single-position weight. Honors the api/cli provider toggle via the shared _parse()."""
    prompt = (
        f"Rebalance this portfolio to a max single-position weight of {max_position_pct:.0f}%. "
        "Each position carries price, shares, value, weight, unrealized gain, and technicals.\n"
        + json.dumps(portfolio, indent=2, default=str)
        + "\n\nReturn the concrete rebalance plan (real share counts + dollar amounts)."
    )
    return await _parse(REBALANCE_SYSTEM, prompt, RebalancePlan, deep=deep, max_tokens=4096)


# ======================================================================================
# AI Sandbox (autonomous paper trader) — a daily decision (Haiku) that emits a unified order list to
# steer a fictional book toward a weekly strategy (Opus). The server validates + fills; the LLM only
# proposes. Both go through the shared _parse() so the api|cli provider toggle applies.
# ======================================================================================

class SandboxOrder(BaseModel):
    symbol: str
    side: str            # "buy" | "sell"
    shares: float        # whole for stocks, fractional (6dp) for crypto — the server RE-DERIVES this
    dollars: float       # approximate notional the server sizes against available cash
    conviction: int      # 0-100
    reason: str          # one short sentence grounded in the numbers


class SandboxDecision(BaseModel):
    posture: str                     # one-line read of today's plan
    orders: list[SandboxOrder]       # the concrete buys/sells for today (may be empty = hold)
    hold_reasons: list[str] = []     # short notes on notable positions deliberately left alone


SANDBOX_SYSTEM = """You are the portfolio manager of a FICTIONAL paper-trading account (no real money). \
Each trading day you make ONE decision: a short list of concrete buy/sell orders that moves the book \
toward its strategy while respecting hard risk limits. You receive JSON: `equity` and `cash`+`cash_pct`; \
a `positions` list (each with weight_pct, unrealized_gain_pct, price, `exposure_group`, and technicals — \
RSI, MACD histogram, % vs 50-day MA, golden-cross, 3-month relative strength vs the S&P, % off 52-week \
high); `candidates` (not-yet-held names to consider, each with the same technicals and a `source`); the \
account `settings` (risk_tolerance, retirement_date, exit_date, max_position_pct, cash_floor_pct, \
allow_crypto, allow_etf); and `strategy_note` — the weekly game plan (stance, cash target, per-exposure \
target weights) you should steer toward (may be null early on).

Positions/candidates that share an `exposure_group` are the SAME economic exposure (e.g. BTC and a \
spot-bitcoin ETF like FBTC/IBIT are both bitcoin; SPY/VOO are the S&P). Judge weight and caps on the \
COMBINED exposure, act on ONE vehicle per group, and NEVER sell one to buy its equivalent.

Return a `SandboxDecision`:
- posture: ONE sentence — what today's plan does and why (e.g. "Trim extended NVDA, start a half-position \
in a strong-RS healthcare name, hold the rest").
- orders: a SMALL list (usually 0-4) of concrete orders. For each: symbol, side (buy|sell), an approximate \
`dollars` notional and `shares` (whole for stocks, fractional for crypto — the server re-sizes to the \
available cash, so approximate is fine), a 0-100 `conviction`, and a one-sentence `reason` grounded in \
ITS numbers. RULES: keep at least `cash_floor_pct`% in cash; keep every exposure_group under \
`max_position_pct`%; only BUY with genuine conviction (skip weak setups — cash is a fine position); \
TRIM a large, extended, well-in-profit winner to fund a better setup or to de-risk; SELL a position \
whose thesis is breaking (relative-strength rolling over, below the 50-day, momentum gone). Do NOT churn \
— if nothing clears the bar, return an empty `orders` list and explain in `posture`.
- hold_reasons: brief notes on 1-3 notable holds.

Bias by `risk_tolerance` (conservative = more cash, defensives, broad diversification, smaller positions; \
aggressive = more concentrated, higher-beta, fully invested) and by the GLIDEPATH: as `retirement_date` \
or `exit_date` approaches, shift progressively toward cash/defensives; if past/at `exit_date`, sell \
everything to cash. If `goal_amount`/`goal_date` are set, weigh the pace needed to reach that target by \
that date against current equity — behind pace with little time justifies leaning more aggressive; well \
ahead lets you de-risk. Respect allow_crypto / allow_etf. Ground EVERY order in the numbers — never \
invent news or prices. Plain reasons, no markdown."""


async def sandbox_decision(
    book: dict, candidates: list[dict], *, cash: float, settings: dict,
    strategy_note: dict | None, deep: bool = False,
) -> tuple[SandboxDecision, dict]:
    """The daily sandbox decision (Haiku by default): a unified order list to steer the book toward the
    strategy within the risk limits. The server validates/clamps/fills afterward — this only proposes."""
    payload = {
        "equity": book.get("total_value"),
        "cash": round(cash, 2),
        "cash_pct": book.get("cash_pct"),
        "positions": book.get("positions", []),
        "candidates": candidates,
        "settings": settings,
        "strategy_note": strategy_note,
    }
    prompt = (
        "Make today's paper-trading decision for this account.\n"
        + json.dumps(payload, indent=2, default=str)
        + "\n\nReturn the SandboxDecision (a short, concrete order list; empty if nothing clears the bar)."
    )
    return await _parse(SANDBOX_SYSTEM, prompt, SandboxDecision, deep=deep, max_tokens=2048)


class TargetWeight(BaseModel):
    exposure_group: str
    target_pct: float


class StrategyNote(BaseModel):
    stance: str                      # "constructive" | "neutral" | "defensive"
    cash_target_pct: float
    targets: list[TargetWeight] = []  # the per-exposure game plan the daily tick steers toward
    themes: list[str] = []           # what to lean into
    avoid: list[str] = []            # what to steer clear of
    notes: str = ""                  # one short paragraph of reasoning


STRATEGY_SYSTEM = """You are the chief strategist for a FICTIONAL paper-trading account, setting the \
WEEKLY game plan that a daily execution model then steers the book toward. You receive JSON: the current \
`book` (positions with weights + exposure_group + technicals, cash, equity), recent `performance` \
(total return, return vs the S&P benchmark, max drawdown), a `market` snapshot (indices, VIX, sector \
leaders/laggards, the S&P's 50/200-day trend), the tradable `universe`, and the account `settings` \
(risk_tolerance, retirement_date, exit_date).

Return a `StrategyNote` — the north star for the coming week:
- stance: exactly one of "constructive", "neutral", or "defensive", grounded in the regime (S&P vs its \
200-day, VIX, breadth) AND the account's risk tolerance + time horizon.
- cash_target_pct: how much of equity to hold in cash this week (higher when defensive / near a horizon \
date / high volatility).
- targets: a handful of `exposure_group` → target_pct weights that express the plan (they need not sum \
to 100 with cash; keep each under a sensible single-name cap). Group equivalent vehicles (BTC≡FBTC).
- themes: 2-4 short things to lean into (sectors/factors), grounded in the sector rotation + relative \
strength you see.
- avoid: 1-3 short things to steer clear of.
- notes: one short paragraph tying it together.

Bias by `risk_tolerance` and the GLIDEPATH toward `retirement_date`/`exit_date` (raise cash + defense as \
they near); if `goal_amount`/`goal_date` are set, set the stance's aggressiveness by the pace needed to \
reach the goal in time. Ground everything in the numbers and the regime — no invented catalysts. Plain \
text, no markdown."""


async def strategy_review(context: dict, *, settings: dict, deep: bool = True) -> tuple[StrategyNote, dict]:
    """The weekly strategy review (Opus by default): a durable game plan the daily sandbox decision then
    executes toward. Degrades gracefully upstream — the caller keeps the prior note if this fails."""
    payload = {**context, "settings": settings}
    prompt = (
        "Set the weekly strategy for this paper-trading account.\n"
        + json.dumps(payload, indent=2, default=str)
        + "\n\nReturn the StrategyNote (stance, cash target, per-exposure targets, themes, avoid, notes)."
    )
    return await _parse(STRATEGY_SYSTEM, prompt, StrategyNote, deep=deep, max_tokens=1536)
