"""
Macro / geopolitical catalyst layer — the market-wide exogenous input.

Every other signal in this service is derived from price or company fundamentals. That means the scan
can be looking at a clean technical setup on a refiner or an airline the same morning a shooting war
changes the oil tape, and nothing in the pipeline knows. This module is the missing input: the
market-moving events that are happening TO the market rather than in it.

Source: Finnhub's `/news?category=general` feed (the same key the per-symbol `news` module already
uses — no new credential). Measured 2026-08-01 on the live feed: 100 items, Reuters 52 / CNBC 40 /
Bloomberg 8, with roughly 44% carrying macro or geopolitical substance (Iran war, Suez/Hormuz
shipping, central-bank decisions, sanctions, tariffs). The feed is "latest 100", with no time-range
parameter — so a job that runs a few times a day sees a rolling window and MUST de-duplicate by
article id across runs, which `unseen_articles` does.

Storage is a single JSON blob (data/macro_latest.json) holding the graded read plus the article ids
already folded into it. Staleness is a first-class field rather than something the caller infers:
`load_state` always reports how old the read is and whether the last run failed, because a macro
layer that silently goes quiet reads to every downstream consumer as "no risks", which is the exact
failure this module exists to prevent.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

from . import settings_store
from .news import RATE
from .redact import http_error

log = logging.getLogger("signals.macro")

_ET = ZoneInfo("America/New_York")
_BASE = "https://finnhub.io/api/v1"

_DATA_DIR = Path(os.environ.get("SIGNALS_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
LATEST = _DATA_DIR / "macro_latest.json"

# How long a read stays usable. The job is wired to run several times a day; past this the read is
# still SHOWN (a day-old war is still a war) but it is flagged stale everywhere it surfaces.
STALE_AFTER_SECONDS = 8 * 3600

# Hard ceiling on how many article ids we remember. Enough to cover several runs of a 100-item feed
# without the blob growing without bound.
_MAX_SEEN_IDS = 600

# Catalysts fade rather than vanish: one that stops appearing in the feed is kept for this long so a
# quiet news day doesn't erase an ongoing situation, then drops out if nothing renews it.
CATALYST_TTL_SECONDS = 3 * 24 * 3600


async def fetch_general(client: httpx.AsyncClient, *, limit: int = 100) -> list[dict]:
    """Latest market-wide headlines, newest first. Empty on any failure or with no Finnhub key.

    Returns [{id, ts, date, headline, summary, source, url}]. `date` is the ET calendar date, matching
    the convention the per-symbol news module established so headlines line up with trading sessions.
    """
    key = settings_store.get().get("finnhub_api_key", "")
    if not key:
        return []
    if not await RATE.acquire(30.0):
        log.warning("macro: general news fetch skipped — Finnhub rate budget exhausted")
        return []
    try:
        r = await client.get(f"{_BASE}/news", params={"category": "general", "token": key}, timeout=20)
        r.raise_for_status()
        raw = r.json()
    except Exception as e:  # noqa: BLE001 — the caller decides what a failed pull means; see run_macro
        # NOT exc_info=True. The traceback's last line is httpx's message, which is the whole request
        # URL, and the Finnhub key is a query parameter in it (SEC-1). redact.http_error keeps the
        # status and the path, which is the part worth reading anyway.
        log.warning("macro: general news fetch failed (%s)", http_error(e))
        return []
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for n in raw:
        head = (n.get("headline") or "").strip()
        ts = n.get("datetime")
        aid = n.get("id")
        if not head or not ts or aid is None:
            continue
        try:
            date = dt.datetime.fromtimestamp(int(ts), _ET).date().isoformat()
        except Exception:  # noqa: BLE001
            continue
        out.append({
            "id": int(aid),
            "ts": int(ts),
            "date": date,
            "headline": head,
            "summary": (n.get("summary") or "").strip()[:400],
            "source": (n.get("source") or "").strip(),
            "url": (n.get("url") or "").strip(),
        })
    out.sort(key=lambda x: x["ts"], reverse=True)
    return out[:limit]


def unseen_articles(articles: list[dict], seen_ids: list[int]) -> list[dict]:
    """Articles not already folded into a previous read.

    The feed is a rolling "latest 100" with heavy overlap between runs, and wire services re-file the
    same story repeatedly. Without this, one war gets graded as a fresh catalyst several times a day.
    """
    seen = set(seen_ids)
    return [a for a in articles if a["id"] not in seen]


def _now() -> float:
    return time.time()


def load() -> dict | None:
    """The stored macro blob, or None when nothing has been written yet / the file is unreadable."""
    if not LATEST.exists():
        return None
    try:
        blob = json.loads(LATEST.read_text())
    except Exception:  # noqa: BLE001 — a corrupt blob is treated as absent, never as "no risks"
        log.warning("macro: stored read is unreadable", exc_info=True)
        return None
    return blob if isinstance(blob, dict) else None


def save(blob: dict) -> None:
    LATEST.parent.mkdir(parents=True, exist_ok=True)
    LATEST.write_text(json.dumps(blob, indent=2))


def age_seconds(blob: dict | None) -> float | None:
    if not blob:
        return None
    gen = blob.get("generated_at")
    if not isinstance(gen, (int, float)):
        return None
    return max(0.0, _now() - float(gen))


def is_stale(blob: dict | None) -> bool:
    age = age_seconds(blob)
    return age is None or age > STALE_AFTER_SECONDS


def active_catalysts(blob: dict | None, *, now: float | None = None) -> list[dict]:
    """Catalysts still inside their TTL, most severe first.

    A catalyst that stops being re-reported is kept for CATALYST_TTL_SECONDS from when it was last
    seen — an ongoing war doesn't stop mattering because the wires had a slow afternoon — and then
    ages out on its own.
    """
    if not blob:
        return []
    now = now if now is not None else _now()
    out = []
    for c in blob.get("catalysts") or []:
        last = c.get("last_seen")
        if isinstance(last, (int, float)) and now - float(last) > CATALYST_TTL_SECONDS:
            continue
        # Trim on READ, not just on write. A catalyst carried over from an earlier run was graded
        # under whatever length rule was in force then — the first live blobs held `why` strings of
        # 230-310 characters, which is the wall of text this was supposed to stop. Normalising here
        # fixes old entries too, instead of waiting for them to age out.
        c = dict(c)
        c["why"] = one_line(c.get("why"), 150)
        c["title"] = one_line(c.get("title"), 70)
        out.append(c)
    out.sort(key=lambda c: (c.get("severity") or 0), reverse=True)
    return out


def load_state() -> dict:
    """The macro read plus its honesty envelope, for the API and every prompt that consumes it.

    `available` is False when there is no usable read at all. Callers MUST branch on it rather than
    treating an empty catalyst list as "the world is calm" — those are different claims, and only one
    of them is something this service is entitled to make.
    """
    blob = load()
    cats = active_catalysts(blob)
    age = age_seconds(blob)
    b = blob or {}
    # `summary` was a paragraph before the bullet rewrite. A blob written by the old code is still
    # readable — its paragraph becomes the headline — so a deploy doesn't blank the card until the
    # next scheduled run replaces it.
    headline = b.get("headline") or b.get("summary") or None
    return {
        "available": blob is not None,
        "risk_level": b.get("risk_level"),
        "headline": one_line(headline, 160) if headline else None,
        "bullets": [x for x in (b.get("bullets") or []) if str(x).strip()],
        "catalysts": cats,
        "generated_at": b.get("generated_at"),
        "as_of": b.get("as_of"),
        "age_seconds": round(age) if age is not None else None,
        "stale": is_stale(blob),
        # True when the most recent RUN failed but an older read survives — the difference between
        # "nothing is happening" and "we couldn't look", which the UI has to be able to tell apart.
        "degraded": bool(b.get("last_run_failed")),
        "last_error": b.get("last_error"),
        "articles_considered": b.get("articles_considered"),
    }


def compact(state: dict, *, limit: int = 4) -> dict | None:
    """The small block injected into analyst prompts — headline risk plus the top few catalysts.

    Returns None when there is no usable read, so callers can omit the key entirely. Injecting an
    empty macro block would tell the model the backdrop is clear on exactly the days the pipeline is
    broken.
    """
    if not state.get("available") or not state.get("risk_level"):
        return None
    cats = state.get("catalysts") or []
    return {
        "risk_level": state["risk_level"],
        "as_of": state.get("as_of"),
        "stale": state.get("stale"),
        "age_hours": round((state.get("age_seconds") or 0) / 3600, 1),
        "headline": state.get("headline"),
        "bullets": state.get("bullets") or [],
        "catalysts": [
            {
                "title": c.get("title"),
                "category": c.get("category"),
                "severity": c.get("severity"),
                "direction": c.get("direction"),
                "horizon": c.get("horizon"),
                "affected": c.get("affected") or [],
                "tickers": c.get("tickers") or [],
                "confidence": c.get("confidence"),
                "why": c.get("why"),
            }
            for c in cats[:limit]
        ],
    }


def merge_catalysts(prior: list[dict], fresh: list[dict], *, now: float | None = None) -> list[dict]:
    """Fold a fresh grading into the standing set, keyed by the model's stable `key` slug.

    A rolling story ("iran-war-escalation") re-reported across days must stay ONE catalyst that keeps
    its `first_seen` and updates its severity — otherwise the same war stacks up as a dozen separate
    risks and the severity ordering becomes meaningless.
    """
    now = now if now is not None else _now()
    by_key: dict[str, dict] = {}
    for c in prior:
        k = (c.get("key") or "").strip().lower()
        if k:
            by_key[k] = dict(c)
    for c in fresh:
        k = (c.get("key") or "").strip().lower()
        if not k:
            continue
        existing = by_key.get(k)
        merged = dict(c)
        merged["key"] = k
        merged["last_seen"] = now
        merged["first_seen"] = (existing or {}).get("first_seen", now)
        # How many separate runs have re-surfaced this — a rough persistence signal that a one-off
        # headline can't fake.
        merged["seen_count"] = int((existing or {}).get("seen_count", 0)) + 1
        by_key[k] = merged
    return sorted(by_key.values(), key=lambda c: (c.get("severity") or 0), reverse=True)


def trim_seen(ids: list[int]) -> list[int]:
    return ids[-_MAX_SEEN_IDS:]


# A plausible US ticker: 1-6 letters, optional .A/.B class suffix. This is a FORMAT filter only — it
# cannot tell a live symbol from a dead one (the first live run emitted "RDS.A", retired in 2022, and
# "SHELL", which has never been a US symbol). See `affected_symbols` for why that is survivable.
_TICKER_OK = re.compile(r"^[A-Z]{1,6}(\.[A-Z])?$")


def one_line(text: str, limit: int) -> str:
    """Collapse to a single tidy line: no markdown bullets, no newlines, capped length.

    The prompts ask for short bullets, but a model will still occasionally re-add the "- " glyph it
    was told to omit or run past the ceiling. Enforcing it here means the UI never has to defend
    against a paragraph arriving where it expects a line — the failure the whole bullet rewrite was
    meant to remove.
    """
    s = " ".join(str(text or "").split())
    s = re.sub(r"^\s*(?:[-*•–—]+\s*)+", "", s)
    if len(s) <= limit:
        return s
    # Cut on a word boundary so the ellipsis doesn't land mid-word.
    cut = s[:limit].rsplit(" ", 1)[0].rstrip(" ,;:.")
    return (cut or s[:limit].rstrip()) + "…"


def sanitize_tickers(tickers: list[str]) -> list[str]:
    """Normalize the model's ticker links and drop anything that isn't ticker-shaped."""
    out: list[str] = []
    for t in tickers or []:
        s = str(t).strip().upper()
        if _TICKER_OK.match(s) and s not in out:
            out.append(s)
    return out


def affected_symbols(state: dict, known: set[str] | list[str]) -> dict[str, list[dict]]:
    """Map REAL symbols to the catalysts that name them.

    The ticker links on a catalyst are model-asserted, not validated instruments — the model will
    occasionally name a symbol that is retired or simply wrong. Intersecting against symbols the
    caller actually knows about makes that harmless by construction: an invented ticker matches
    nothing and silently disappears, instead of attaching a scary-looking risk to the wrong name.

    `affected` sector words are deliberately NOT matched here. Fuzzy mapping from "airlines" to a
    ticker is exactly the kind of guess that produces a confident, wrong claim on a stock's page.
    """
    known_set = {str(s).strip().upper() for s in known}
    out: dict[str, list[dict]] = {}
    for c in state.get("catalysts") or []:
        for t in sanitize_tickers(c.get("tickers") or []):
            if t in known_set:
                out.setdefault(t, []).append(c)
    return out
