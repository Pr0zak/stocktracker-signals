"""MB-19 — a curated, PRIMARY-SOURCED ticker universe, cached to disk.

The value screen previously drew its candidates from Yahoo's predefined screeners, which are rebuilt
server-side on every call: two runs minutes apart returned different names, and the pool was only
~58 wide. That is not a screener, it is a sampler.

Source of record is **Nasdaq Trader's consolidated symbol directory**
(`nasdaqtraded.txt`) — the official public list of everything traded on US exchanges, with test-issue
and ETF flags. Deliberately NOT mungbeans' curated list (this repo is public, and their file is their
work), and deliberately not the SEC's `company_tickers.json`, which requires a contact-bearing
User-Agent that has no business being hard-coded in a public repo.

The directory has ~13k rows including warrants, units and preferreds, so it is filtered structurally
and then by market cap via batched Yahoo quotes. The result is persisted with a build timestamp,
because rebuilding costs a few hundred HTTP calls and the answer changes on the order of weeks.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import httpx

from . import options

_DIRECTORY_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
_DATA_DIR = Path(os.environ.get("SIGNALS_DATA_DIR",
                                str(Path(__file__).resolve().parent.parent / "data")))
_FILE = _DATA_DIR / "universe.json"

_BATCH = 50           # symbols per Yahoo quote call
_CONCURRENCY = 4
DEFAULT_MIN_CAP = 2_000_000_000.0
DEFAULT_MIN_PRICE = 5.0
DEFAULT_LIMIT = 600   # top N by market cap — bounded so a full screen stays tractable
STALE_AFTER_S = 7 * 24 * 3600
# Below this share of successful quote batches the build is a subsample, not a universe, and must not
# overwrite a good one — a partial fetch stamped fresh for a week is worse than a slightly old build.
MIN_COVERAGE = 0.90


def parse_directory(text: str) -> list[dict]:
    """Parse nasdaqtraded.txt into candidate rows.

    Pipe-delimited with a header line and a trailing 'File Creation Time' footer. Dropped here:
    test issues, non-traded rows, and symbols carrying '$' or '.' — warrants, units and preferred
    classes, which have their own price behaviour and no meaningful 200-week trend of the common.
    """
    rows: list[dict] = []
    lines = text.splitlines()
    if not lines:
        return rows
    header = [h.strip() for h in lines[0].split("|")]
    try:
        i_traded = header.index("Nasdaq Traded")
        i_sym = header.index("Symbol")
        i_name = header.index("Security Name")
        i_etf = header.index("ETF")
        i_test = header.index("Test Issue")
    except ValueError:
        return rows
    for line in lines[1:]:
        parts = line.split("|")
        if len(parts) <= max(i_traded, i_sym, i_name, i_etf, i_test):
            continue                      # footer row
        sym = parts[i_sym].strip().upper()
        if parts[i_traded].strip() != "Y" or parts[i_test].strip() == "Y":
            continue
        # Structural exclusions. "$" marks preferreds/when-issued in this feed. A "." is a CLASS
        # separator (BRK.B, BF.A) — dropping every dotted symbol removed Berkshire Hathaway and
        # every other dual-class common from the universe, which was never the intent; only the
        # single-letter suffixes that denote warrants/units/rights are excluded.
        if not sym or "$" in sym:
            continue
        base, _, cls = sym.partition(".")
        if cls and cls not in ("A", "B", "C"):      # W=warrant, U=unit, R=right, WS=when-issued
            continue
        if len(base) > 5:
            continue
        rows.append({
            "symbol": sym,
            "name": parts[i_name].strip(),
            "is_etf": parts[i_etf].strip() == "Y",
        })
    return rows


async def fetch_directory(client: httpx.AsyncClient) -> list[dict]:
    r = await client.get(_DIRECTORY_URL, timeout=45.0)
    r.raise_for_status()
    return parse_directory(r.text)


async def _profile_batch(client: httpx.AsyncClient, syms: list[str]) -> dict[str, dict]:
    """Market cap + price + quote type for a batch. Reuses market_now's auth machinery rather than
    changing fetch_quotes' return contract, which other callers depend on."""
    crumb = await options._ensure_auth(client)
    params = {"symbols": ",".join(syms), "crumb": crumb}
    for host in options._HOSTS:
        try:
            url = f"https://{host}/v7/finance/quote"
            r = await client.get(url, params=params, headers=options._headers(), timeout=25)
            if r.status_code == 401:
                crumb = await options._ensure_auth(client, force=True, stale=crumb)
                params["crumb"] = crumb
                r = await client.get(url, params=params, headers=options._headers(), timeout=25)
            r.raise_for_status()
            out = {}
            for q in (r.json().get("quoteResponse") or {}).get("result") or []:
                s = (q.get("symbol") or "").upper()
                if s:
                    out[s] = {"cap": q.get("marketCap"), "price": q.get("regularMarketPrice"),
                              "type": q.get("quoteType")}
            return out
        except Exception:  # noqa: BLE001 — try the next host
            continue
    return {}   # a dead batch drops those symbols; it must not sink the whole build


async def build(client: httpx.AsyncClient, *, min_cap: float = DEFAULT_MIN_CAP,
                min_price: float = DEFAULT_MIN_PRICE, limit: int = DEFAULT_LIMIT) -> dict:
    """Fetch, filter and rank the universe. Returns the persisted-shape dict (does not write)."""
    directory = await fetch_directory(client)
    syms = [r["symbol"] for r in directory]
    meta = {r["symbol"]: r for r in directory}

    sem = asyncio.Semaphore(_CONCURRENCY)

    async def one(batch: list[str]):
        async with sem:
            return await _profile_batch(client, batch)

    batches = [syms[i:i + _BATCH] for i in range(0, len(syms), _BATCH)]
    results = await asyncio.gather(*[one(b) for b in batches])

    # _profile_batch returns {} when every host failed for that batch, so a rate limit or an outage
    # midway silently shrinks the universe — which then gets SAVED and stamped fresh for a week, and
    # the value screen serves the subsample as if it were the whole market. Measure coverage and
    # refuse to publish a build that is materially incomplete.
    empty_batches = sum(1 for got in results if not got)
    coverage = 1.0 - (empty_batches / len(batches)) if batches else 0.0

    rows: list[dict] = []
    for got in results:
        for sym, p in got.items():
            cap, price = p.get("cap"), p.get("price")
            if p.get("type") not in ("EQUITY", "ETF"):
                continue
            if not isinstance(cap, (int, float)) or not isinstance(price, (int, float)):
                continue        # an ETF often has no marketCap — excluded by the same rule
            if cap < min_cap or price < min_price:
                continue
            m = meta.get(sym, {})
            rows.append({"symbol": sym, "name": m.get("name", sym),
                         "is_etf": bool(m.get("is_etf")), "market_cap": float(cap)})
    rows.sort(key=lambda r: -r["market_cap"])
    return {
        "built_at": time.time(),
        "batch_coverage": round(coverage, 3),
        "empty_batches": empty_batches,
        "complete": coverage >= MIN_COVERAGE,
        "source": "nasdaqtrader.com symbol directory + Yahoo quote market caps",
        "directory_rows": len(directory),
        "passed_filter": len(rows),
        "min_cap": min_cap,
        "min_price": min_price,
        "symbols": [r["symbol"] for r in rows[:limit]],
        "detail": rows[:limit],
    }


def load() -> dict | None:
    try:
        return json.loads(_FILE.read_text())
    except Exception:  # noqa: BLE001 — absent or corrupt is simply "not built yet"
        return None


def save(blob: dict) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(blob))
    os.replace(tmp, _FILE)      # atomic: a crash mid-write must not leave a half-file


def is_stale(blob: dict | None, *, now: float | None = None) -> bool:
    if not blob or not blob.get("built_at"):
        return True
    return (now or time.time()) - float(blob["built_at"]) > STALE_AFTER_S
