"""
The macro catalyst research job (NEWS-2). Run standalone (`python -m app.macro_job`, wired to a
systemd timer) or via POST /macro/run.

Fetches the market-wide wire feed, drops the articles already folded into a previous read, hands the
rest to the cheap scan model for triage, and merges the result into the standing catalyst set. Cost is
one Haiku call per run over ~100 headlines — a fraction of a cent, so this can run several times a day.

Two properties matter more than anything else here:

  * A failed run must never look like a calm world. On failure the previous read is kept and stamped
    `last_run_failed`, so every consumer can tell "we couldn't look" from "nothing is happening".
  * An ongoing situation must stay ONE catalyst. The feed re-files the same story for days, so the
    model assigns a stable slug and `macro.merge_catalysts` folds repeats into the existing entry
    rather than stacking a dozen copies of the same war.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time

import httpx

from . import macro, settings_store, usage_store
from .analyst import macro_read

log = logging.getLogger("signals.macro")

# Headlines handed to the model per run. The feed returns 100; there is no value in trimming further
# since triage quality depends on seeing the whole picture, and 100 headlines is ~6k input tokens.
_MAX_ARTICLES = 100

# Below this many genuinely new articles, a re-grade would just re-read what we already graded. The
# standing catalysts are kept as-is and only their freshness stamp moves.
_MIN_NEW_ARTICLES = 3


async def run_macro(*, force: bool = False) -> dict:
    """One macro research pass. Returns the stored state (see macro.load_state)."""
    prior = macro.load() or {}
    prior_catalysts = prior.get("catalysts") or []
    seen_ids: list[int] = [int(i) for i in (prior.get("seen_ids") or []) if isinstance(i, (int, float))]

    async with httpx.AsyncClient() as client:
        articles = await macro.fetch_general(client, limit=_MAX_ARTICLES)

    if not articles:
        # Either no Finnhub key or the feed is down. Keep whatever read we had and SAY the run failed.
        # Writing an empty read here would turn an outage into a confident "no macro risks".
        blob = dict(prior)
        blob["last_run_failed"] = True
        blob["last_error"] = "news feed returned nothing (no key, or the fetch failed)"
        blob["last_run_at"] = time.time()
        if prior:
            macro.save(blob)
        log.warning("macro: no articles — keeping the previous read, flagged degraded")
        return macro.load_state()

    fresh = macro.unseen_articles(articles, seen_ids)
    if len(fresh) < _MIN_NEW_ARTICLES and not force and prior.get("risk_level"):
        # Nothing meaningfully new. Refresh the freshness stamp so the read isn't reported stale while
        # it is in fact current, but don't spend a model call re-reading the same headlines.
        blob = dict(prior)
        blob["generated_at"] = time.time()
        blob["as_of"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        blob["last_run_failed"] = False
        blob["last_error"] = None
        blob["last_run_at"] = time.time()
        macro.save(blob)
        log.info("macro: only %d new article(s) — read refreshed without a model call", len(fresh))
        return macro.load_state()

    # Grade the WHOLE current feed, not just the unseen slice: severity is a judgement about the
    # situation as a whole, and a model shown three stray follow-ups with no context would grade them
    # as isolated trivia. The unseen count decides WHETHER to re-grade; the model still sees the lot.
    payload = [
        {"date": a["date"], "source": a["source"], "headline": a["headline"], "summary": a["summary"]}
        for a in articles
    ]

    # Show the model what it is already tracking, so a developing story keeps its key instead of
    # being re-coined into a near-duplicate and stacked twice by the merge.
    existing = [
        {"key": c.get("key"), "title": c.get("title")}
        for c in macro.active_catalysts({"catalysts": prior_catalysts}, now=time.time())
    ][:12]

    try:
        read, usage = await macro_read(payload, existing=existing, deep=False)
    except Exception as e:  # noqa: BLE001 — keep the prior read; never publish a blank one
        blob = dict(prior)
        blob["last_run_failed"] = True
        blob["last_error"] = f"grading failed: {e}"
        blob["last_run_at"] = time.time()
        if prior:
            macro.save(blob)
        log.warning("macro: grading failed — keeping the previous read", exc_info=True)
        return macro.load_state()

    usage_store.record(usage, symbol="_macro", kind="macro")

    now = time.time()
    merged = macro.merge_catalysts(
        prior_catalysts, [c.model_dump() for c in read.catalysts], now=now,
    )
    blob = {
        "generated_at": now,
        "as_of": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "risk_level": read.risk_level,
        "headline": read.headline,
        "bullets": read.bullets,
        "catalysts": merged,
        "seen_ids": macro.trim_seen(sorted({*seen_ids, *(a["id"] for a in articles)})),
        "articles_considered": len(articles),
        "articles_new": len(fresh),
        "model": settings_store.get()["scan_model"],
        "cost_usd": usage.get("cost_usd"),
        "last_run_failed": False,
        "last_error": None,
        "last_run_at": now,
    }
    macro.save(blob)
    log.info(
        "macro: risk=%s catalysts=%d (%d new articles of %d) $%s",
        read.risk_level, len(merged), len(fresh), len(articles), usage.get("cost_usd"),
    )
    return macro.load_state()


if __name__ == "__main__":
    state = asyncio.run(run_macro())
    cats = state.get("catalysts") or []
    print(f"risk={state.get('risk_level')} · {len(cats)} catalyst(s)")
    for c in cats:
        print(f"  [{c.get('severity'):>3}] {c.get('category'):<12} {c.get('title')}")
