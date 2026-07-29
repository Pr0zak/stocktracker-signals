"""Theme C — follow the smart money: who with an information edge has been BUYING, across the book.

Two feeds already exist per-symbol (`insider.insider_buying`, `congress.congress_trades`). What was
missing is the cross-sectional view: the question is not "did anyone buy NVDA" but "of everything I
watch, where has informed money actually shown up".

Both signals are *corroborating context*, never a reason on their own:

* **Insider buys** are the stronger of the two — an officer or director spending their own money on
  the open market, disclosed within two business days. Sales are deliberately NOT scored: insiders
  sell for diversification, tax and house purchases, so a sale carries far less information than a
  buy.
* **Congressional disclosures LAG by up to ~45 days** under the STOCK Act, and the amounts are
  ranges, not figures. A congressional buy is a stale, fuzzy signal — weighted well below an insider
  buy, and the staleness is reported rather than smoothed over.

THE HONESTY RULE, as everywhere else here: a feed that could not be read is NOT the same as a feed
that came back empty. "No insider buying" and "no Finnhub key configured" must never render the
same, so every row carries `sources_seen` and the endpoint reports what it could not consult.
"""
from __future__ import annotations

import asyncio

# Insider evidence — an officer buying their own stock on the open market.
_W_INSIDER_CLUSTER = 3.0    # several insiders buying independently, close together
_W_INSIDER_CONVICTION = 2.5  # one unusually large buy
_W_INSIDER_ANY = 1.5        # any open-market buying at all

# Congressional evidence — lagged up to ~45 days, amounts are ranges. Deliberately weaker.
_W_CONGRESS_CLUSTER = 1.5
_W_CONGRESS_NET_BUY = 1.0
_W_CONGRESS_LARGE = 0.75    # a disclosed buy whose range TOP is over this many dollars
_LARGE_CONGRESS_BUY = 100_000


def score_symbol(symbol: str, insider: dict | None, congress: dict | None,
                 *, insider_available: bool = True, congress_available: bool = True) -> dict:
    """Score one name. `*_available` distinguishes "feed said nothing" from "feed unreadable".

    A row is only meaningful if at least one source was actually consulted; `sources_seen` says
    which, so a name scoring 0 because nobody bought is distinguishable from one scoring 0 because
    the key was missing.
    """
    score = 0.0
    reasons: list[str] = []
    sources_seen: list[str] = []
    unavailable: list[str] = []

    if insider_available:
        sources_seen.append("insider")
        if insider:
            if insider.get("has_cluster_buy"):
                score += _W_INSIDER_CLUSTER
                reasons.append("Several insiders bought independently")
            if insider.get("has_conviction_buy"):
                score += _W_INSIDER_CONVICTION
                reasons.append("An insider made an unusually large buy")
            n = insider.get("buy_count_12m") or 0
            if n and not (insider.get("has_cluster_buy") or insider.get("has_conviction_buy")):
                score += _W_INSIDER_ANY
                reasons.append(f"{n} open-market insider buy{'s' if n != 1 else ''} in 12 months")
    else:
        unavailable.append("insider")

    if congress_available:
        sources_seen.append("congress")
        if congress:
            buys = congress.get("buy_count") or 0
            if congress.get("cluster_buy"):
                score += _W_CONGRESS_CLUSTER
                reasons.append("Several members of Congress bought around the same time")
            elif congress.get("net_direction") == "buying" and buys:
                score += _W_CONGRESS_NET_BUY
                reasons.append(f"{buys} disclosed congressional buy{'s' if buys != 1 else ''}, "
                               f"net buying")
            if (congress.get("largest_buy_amount_high") or 0) >= _LARGE_CONGRESS_BUY:
                score += _W_CONGRESS_LARGE
                reasons.append("A disclosed congressional buy in the six-figure range")
    else:
        unavailable.append("congress")

    return {
        "symbol": symbol,
        "score": round(score, 2),
        "reasons": reasons,
        # Which feeds were actually consulted. Without this a 0 from "nobody bought" is
        # indistinguishable from a 0 from "we could not look".
        "sources_seen": sources_seen,
        "unavailable": unavailable,
        "insider": insider or None,
        "congress": congress or None,
        # The freshest disclosure we can point at, so the card can age the evidence rather than
        # implying it is current. Congressional filings are up to ~45 days behind the trade.
        "congress_latest_filing": (congress or {}).get("latest_filing_date"),
        "congress_newest_trade": (congress or {}).get("newest_trade_date"),
    }


def rank(rows: list[dict], *, limit: int) -> list[dict]:
    """Only names with actual evidence, best first. Ties break on symbol so the order is stable."""
    scored = [r for r in rows if r and r["score"] > 0]
    scored.sort(key=lambda r: (-r["score"], r["symbol"]))
    return scored[:limit]


async def sweep(symbols: list[str], insider_of, congress_of, *, limit: int = 20,
                concurrency: int = 4) -> tuple[list[dict], list[str], list[str]]:
    """Score every symbol concurrently. Returns (ranked, no_evidence, failed).

    `no_evidence` and `failed` are separate for the usual reason: the first is a finding about the
    company, the second is a finding about our fetch, and reporting one as the other is a confident
    claim about the wrong thing.
    """
    sem = asyncio.Semaphore(concurrency)

    async def one(sym: str):
        async with sem:
            ins = con = None
            ins_ok = con_ok = True
            try:
                ins = await insider_of(sym)
            except Exception:  # noqa: BLE001 — one dead feed must not sink the sweep
                ins_ok = False
            try:
                con = await congress_of(sym)
            except Exception:  # noqa: BLE001
                con_ok = False
            if not ins_ok and not con_ok:
                return sym, None, "failed"
            return sym, score_symbol(sym, ins, con,
                                     insider_available=ins_ok, congress_available=con_ok), None

    results = await asyncio.gather(*[one(s) for s in symbols])
    rows = [r for _, r, _ in results if r]
    failed = [s for s, _, why in results if why == "failed"]
    no_evidence = [r["symbol"] for r in rows if r["score"] <= 0]
    return rank(rows, limit=limit), no_evidence, failed
