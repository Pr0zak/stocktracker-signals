"""
Optional news + earnings context for the analyst (Tier 3), via Finnhub's free tier. Adds recent
headlines and the next earnings date to a stock's snapshot so Claude can ground the `catalysts` and
`key_risks` fields in real events. No-op (empty) when no Finnhub key is configured, or for crypto.
"""
from __future__ import annotations

import datetime as dt
import logging
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")

import httpx

from . import settings_store

log = logging.getLogger("signals.news")

_BASE = "https://finnhub.io/api/v1"


async def fetch_context(client: httpx.AsyncClient, symbol: str) -> dict:
    key = settings_store.get().get("finnhub_api_key", "")
    if not key:
        return {}
    out: dict = {}
    today = dt.date.today()

    try:  # recent company news — a few latest headlines
        r = await client.get(
            f"{_BASE}/company-news",
            params={"symbol": symbol, "from": (today - dt.timedelta(days=10)).isoformat(),
                    "to": today.isoformat(), "token": key},
            timeout=12,
        )
        r.raise_for_status()
        news = r.json()
        if isinstance(news, list):
            heads = [n.get("headline", "").strip() for n in news if n.get("headline")]
            if heads:
                out["recent_news"] = heads[:5]
    except Exception as e:  # noqa: BLE001 — decorative context; never fail the verdict on it
        # Still logged. A verdict silently written without the news that explains the move looks
        # identical to one written on a genuinely quiet tape.
        log.warning("news: %s headline context failed (%s: %s)", symbol, type(e).__name__, e)

    try:  # next scheduled earnings date within ~90 days
        r = await client.get(
            f"{_BASE}/calendar/earnings",
            params={"from": today.isoformat(), "to": (today + dt.timedelta(days=90)).isoformat(),
                    "symbol": symbol, "token": key},
            timeout=12,
        )
        r.raise_for_status()
        cal = r.json().get("earningsCalendar") or []
        if cal:
            out["next_earnings"] = min(cal, key=lambda e: e.get("date", "9999")).get("date")
    except Exception as e:  # noqa: BLE001
        log.warning("news: %s earnings-calendar lookup failed (%s: %s)", symbol, type(e).__name__, e)

    return out


async def fetch_dated_news(client: httpx.AsyncClient, symbol: str, days: int = 16) -> list[dict] | None:
    """Company news over the last [days], each carrying its ET date so the analyst can line headlines
    up against specific price moves (AIE-4). Returns [{date: YYYY-MM-DD, headline, summary, source, url}]
    newest-first.

    Returns **None when the LOOKUP FAILED** (no key, HTTP error, rate limit, timeout) and **[] only
    when the fetch succeeded and the symbol genuinely has no coverage**. Callers must not conflate
    them.

    That distinction is the whole point of this signature. It used to return [] on every failure,
    silently and without a log line — so on 2026-08-03 GME fell 10.4% on a $1.4B convertible-notes
    dilution, the app asked why, one lookup came back empty, and /news_moves rendered "No headlines
    available for this day" and CACHED that confident claim for an hour. Finnhub had 26 articles
    including "Why GameStop Stock Just Slumped". Because nothing was logged, it is still not
    recoverable whether that lookup was rate-limited by the ~10-call burst the app fires when a
    symbol is opened, or whether Finnhub had simply not indexed yet.
    """
    key = settings_store.get().get("finnhub_api_key", "")
    if not key:
        log.warning("news: no Finnhub key configured — %s lookup skipped", symbol)
        return None
    today = dt.date.today()
    try:
        r = await client.get(
            f"{_BASE}/company-news",
            params={"symbol": symbol, "from": (today - dt.timedelta(days=days)).isoformat(),
                    "to": today.isoformat(), "token": key},
            timeout=12,
        )
        r.raise_for_status()
        raw = r.json()
    except Exception as e:  # noqa: BLE001
        # LOUD on purpose. A rate limit and "this company made no news" are the same empty list to
        # every caller downstream; the log is the only place they stay distinguishable.
        log.warning("news: %s lookup FAILED (%s: %s) — this is not 'no news'",
                    symbol, type(e).__name__, e)
        return None
    if not isinstance(raw, list):
        log.warning("news: %s returned a non-list payload (%s)", symbol, type(raw).__name__)
        return None
    out: list[dict] = []
    for n in raw:
        head = (n.get("headline") or "").strip()
        ts = n.get("datetime")
        if not head or not ts:
            continue
        try:
            # ET, as the docstring promises and as the analyst assumes when lining headlines up
            # against price moves. utcfromtimestamp pushed anything published after 20:00 ET onto the
            # FOLLOWING calendar day, so an evening headline appeared to precede the next session's
            # move rather than follow that day's.
            date = dt.datetime.fromtimestamp(int(ts), _ET).date().isoformat()
        except Exception:  # noqa: BLE001
            continue
        out.append({
            "date": date,
            "headline": head,
            "summary": (n.get("summary") or "").strip()[:280],
            "source": (n.get("source") or "").strip(),
            "url": (n.get("url") or "").strip(),
        })
    out.sort(key=lambda x: x["date"], reverse=True)
    return out
