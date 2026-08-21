"""MB-19 / SWT-1 — a curated, PRIMARY-SOURCED ticker universe, cached to disk, gated on LIQUIDITY.

The value screen previously drew its candidates from Yahoo's predefined screeners, which are rebuilt
server-side on every call: two runs minutes apart returned different names, and the pool was only
~58 wide. That is not a screener, it is a sampler.

Source of record is **Nasdaq Trader's consolidated symbol directory**
(`nasdaqtraded.txt`) — the official public list of everything traded on US exchanges, with test-issue
and ETF flags. Deliberately NOT mungbeans' curated list (this repo is public, and their file is their
work), and deliberately not the SEC's `company_tickers.json`, which requires a contact-bearing
User-Agent that has no business being hard-coded in a public repo.

WHAT THIS MODULE DECIDES, and the decision it now protects
----------------------------------------------------------
The directory has ~13k rows including warrants, units and preferreds, so it is filtered structurally
and then by a QUANTITATIVE gate before being persisted with a build timestamp (rebuilding costs a few
hundred HTTP calls and the answer changes on the order of weeks).

That quantitative gate used to be MARKET CAP >= $2B, which answered the wrong question. What a
scanner needs to know is "can this be traded in size without moving it" — a liquidity question — and
market cap is a poor proxy for it that additionally punishes the data: 13.3% of equities report no
marketCap at all, and the old build DELETED every one of them. The gate is now AVERAGE DOLLAR
VOLUME, and market cap survives only as METADATA on the row.

Measured census over the full directory (100% coverage, 2026-08-21), which is where every constant
below comes from:

    directory rows 12,620; 12,175 with price+avgVol
    equities 6,555 | ETFs 5,620
    equities with NO marketCap: 873 (13.3%)   |   ETFs with a marketCap: 9 of 5,620
    ADV$ >= $5M and px >= $1  ->  4,562 total = 3,158 equities + 1,404 ETFs
    ADV$ >= $5M and px >= $5  ->  4,332 total = 2,949 equities + 1,383 ETFs
    the old filter (cap >= $2B, px >= $5) -> 2,045 passed, top 600 stored
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from pathlib import Path

import httpx

from . import options

log = logging.getLogger("signals.universe")

_DIRECTORY_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
_DATA_DIR = Path(os.environ.get("SIGNALS_DATA_DIR",
                                str(Path(__file__).resolve().parent.parent / "data")))
_FILE = _DATA_DIR / "universe.json"

_BATCH = 50           # symbols per Yahoo quote call
_CONCURRENCY = 4

# Units, warrants and rights, as the Nasdaq directory names them: "Churchill Capital Corp XIII -
# Units", "Armada Acquisition Corp. III - Warrant". Anchored to the end of the string on purpose —
# see the parse_directory comment for why an unanchored match would delete every ADR. Measured
# against the live directory on 2026-08-21: 721 rows matched, and TSM / BABA / ASML / SONY / NVO
# were all correctly left alone.
_DERIVATIVE_NAME = re.compile(r"\s-\s(?:[^-]*\b)?(?:Units?|Warrants?|Rights?)\s*$", re.I)

# The gate. A name is admitted when it trades at least $5,000,000 of stock on an average day.
# That is the tradability question a scanner actually has: whether a position can be opened and
# closed without being the market. Market cap answered a different question (how big is the issuer)
# and answered it for only 79% of equities.
DEFAULT_MIN_DOLLAR_VOLUME = 5_000_000.0

# The price floor drops from $5 to $1. With dollar volume doing the tradability work, the $5 floor
# was a second, cruder version of the same test that misfired on real liquidity: PLUG trades ~$134M
# a day at $2.27 and the $5 floor excluded it outright. $1 remains, because sub-$1 quotes bring
# tick-size and delisting behaviour that the weekly-trend maths downstream is not built for.
DEFAULT_MIN_PRICE = 1.0

# A CEILING, not a target. The census says ~3,158 equities pass the gate, so 4,000 leaves headroom
# for the market to grow into without ever silently truncating the pool. (It was 600 under the cap
# filter, which stored barely a quarter of what passed.)
DEFAULT_LIMIT = 4000

# Market cap is no longer a filter, only metadata on the row. The constant stays at 0.0 so the
# persisted blob records the floor as EXPLICITLY ZERO rather than dropping the key — a reader
# diffing an old build against a new one sees 2e9 -> 0.0, which is exactly what changed.
DEFAULT_MIN_CAP = 0.0

STALE_AFTER_S = 7 * 24 * 3600

# Below this share of successful quote batches the build is a subsample, not a universe, and must not
# overwrite a good one — a partial fetch stamped fresh for a week is worse than a slightly old build.
MIN_COVERAGE = 0.90

# batch_coverage counts BATCHES, and Yahoo silently OMITS unknown symbols from an otherwise-200
# response, so a batch can come back healthy having answered for half of what it was asked. At 253
# batches, MIN_COVERAGE=0.90 alone tolerates ~1,250 missing symbols in a build stamped `complete`.
# symbol_coverage measures the thing that actually matters — symbols returned / symbols requested —
# and this is its floor.
MIN_SYMBOL_COVERAGE = 0.95


def parse_directory(text: str) -> list[dict]:
    """Parse nasdaqtraded.txt into candidate rows.

    Pipe-delimited with a header line and a trailing 'File Creation Time' footer. Dropped here:
    test issues, non-traded rows, symbols carrying '$' or a non-class '.' suffix, and any security
    the directory NAMES as a unit, warrant or right. All of those have their own price behaviour
    and no meaningful 200-week trend of the common.

    American Depositary Shares and New York Registry Shares are deliberately KEPT — they are
    ordinary common stock in every way that matters here, and TSM, BABA, ASML and SONY all reach
    this codebase through them.
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
        # The dotted-suffix rule above only catches instruments the feed spells with a dot. Nasdaq
        # writes most SPAC units and warrants as UNDOTTED five-letter symbols instead (XIIIU,
        # AACIW), and those slipped through: seven of them landed in the universe once the $2B cap
        # floor — which had been excluding them by accident, not by intent — was replaced with a
        # liquidity gate. A unit is a share bundled with a fraction of a warrant, so its price is
        # part trust value and part option premium; a moving average or an average daily range
        # computed over that hybrid does not describe a company, and it pollutes every
        # cross-sectional percentile it appears in.
        #
        # The SYMBOL suffix is not a safe test (five-letter tickers are not reserved for
        # derivatives), so match the directory's own naming convention instead: it spells the
        # instrument out after " - ". Anchored to the END of the name so that "American Depositary
        # Shares" and similar survive — ADRs are ordinary common stock, TSM and BABA among them,
        # and an unanchored match on "Shares"/"Depositary" would have deleted them.
        if _DERIVATIVE_NAME.search(parts[i_name]):
            continue
        # Nasdaq writes a share class with a DOT (BRK.B); Yahoo — which every downstream consumer
        # here uses for quotes and weekly bars — writes it with a HYPHEN (BRK-B). Queried with the
        # dot, Yahoo returns the symbol but with marketCap None (and types BF.B as MUTUALFUND), so
        # the cap filter dropped it silently: admitting dotted symbols to the parser achieved
        # nothing on its own. Store the form the fetchers actually need.
        if cls:
            sym = f"{base}-{cls}"
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
    """Price + average daily volume + market cap + quote type for a batch. Reuses market_now's auth
    machinery rather than changing fetch_quotes' return contract, which other callers depend on.

    `averageDailyVolume3Month` is ALREADY in the v7 quote payload this function requests — verified
    live against the real response — so carrying it costs ZERO extra HTTP calls. The previous version
    fetched it and threw it away, which is why the universe was gated on the one field (marketCap)
    that 21% of the response does not carry.
    """
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
                              "type": q.get("quoteType"),
                              "adv": q.get("averageDailyVolume3Month")}
            return out
        except Exception:  # noqa: BLE001 — one host failing is routine; try the next before giving up
            continue
    # A dead batch drops those symbols; it must not sink the whole build. Say so, because the
    # symbols are ABSENT from the result rather than filtered out of it, and coverage counts it.
    log.warning("universe: every host failed for a %d-symbol batch — those symbols are absent "
                "from this build, not excluded by it", len(syms))
    return {}


async def build(client: httpx.AsyncClient, *,
                min_dollar_volume: float = DEFAULT_MIN_DOLLAR_VOLUME,
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

    # The second, finer coverage measure: a batch that answered for 30 of its 50 symbols looks
    # perfectly healthy to the count above. Counted as DISTINCT symbols in case a host ever echoes
    # one back into a neighbouring batch.
    returned: set[str] = set()

    rows: list[dict] = []
    equity_rows = 0
    etf_rows = 0
    no_price = 0
    no_adv = 0
    for got in results:
        returned.update(got)
        for sym, p in got.items():
            m = meta.get(sym, {})
            qtype = p.get("type")
            # EQUITIES ONLY, stated on purpose. This is NOT a regression: the live universe already
            # held zero ETFs (verified — 0 of 600, no VTI, no SPY, no FBTC), because only 9 of 5,620
            # ETFs report a marketCap and the old isinstance(cap) gate deleted every one of them.
            # The exclusion was real but accidental, a side effect of a filter aimed at something
            # else; dropping the cap gate would have silently ADMITTED 1,404 funds. Make it explicit
            # and intentional instead, and count what it costs.
            if qtype == "ETF" or bool(m.get("is_etf")):
                etf_rows += 1
                continue
            if qtype != "EQUITY":
                continue        # MUTUALFUND, INDEX, CURRENCY — not a common share on a US exchange
            equity_rows += 1

            price = p.get("price")
            # Hard gate. Every downstream computation here is price-based, so no price means no
            # scan — but that is a fact about the RESPONSE, so it is counted rather than dropped.
            if not isinstance(price, (int, float)):
                no_price += 1
                continue
            if price < min_price:
                continue

            adv = p.get("adv")
            # averageDailyVolume3Month covers ~95% of the response (against marketCap's 79%), so a
            # missing one is a fetch anomaly, not a property of the company. Excluded — the gate
            # cannot be applied without it — but NAMED in the blob, never silently dropped.
            if not isinstance(adv, (int, float)):
                no_adv += 1
                continue
            if adv * price < min_dollar_volume:
                continue

            # Market cap is OPTIONAL METADATA from here on. None, never 0.0: 13.3% of equities
            # report no cap, and rendering an unreported cap as a $0 company is precisely the
            # "absent presented as a confident number" defect this codebase keeps having to fix.
            cap = p.get("cap")
            rows.append({
                "symbol": sym,
                "name": m.get("name", sym),
                "is_etf": False,          # kept for row-shape stability; ETFs cannot reach here
                "market_cap": float(cap) if isinstance(cap, (int, float)) else None,
                "dollar_volume": round(float(adv) * float(price), 2),
            })

    # THE SORT IS LOAD-BEARING — keep it CAP-DESCENDING, with unknown caps LAST.
    # /heatmap?mode=market slices uni["detail"][:limit] (main.py) and is captioned "Sized by market
    # cap". Because the ranking is unchanged, the HEAD of the list is unchanged too: the heat map's
    # top-200 slice behaves exactly as it did before, and the universe grows only at the TAIL.
    # Re-ranking by dollar volume would silently rewrite which companies appear on the map — a
    # visible product change smuggled in by a filter change. The `is None` term is the other half:
    # market_cap can now be None, and `-r["market_cap"]` alone raises TypeError the moment it is.
    # Cap-less names sort to the tail and therefore never reach the head slice.
    rows.sort(key=lambda r: (r["market_cap"] is None, -(r["market_cap"] or 0.0)))

    # None, not 0.0, when nothing was requested: unmeasurable is not "zero coverage". publish()
    # skips the gate on a coverage it cannot read rather than refusing on a number it invented.
    symbol_coverage = (len(returned) / len(syms)) if syms else None
    return {
        "built_at": time.time(),
        "batch_coverage": round(coverage, 3),
        "symbol_coverage": round(symbol_coverage, 3) if symbol_coverage is not None else None,
        "empty_batches": empty_batches,
        "complete": coverage >= MIN_COVERAGE,
        "source": "nasdaqtrader.com symbol directory + Yahoo quote price x avg daily volume",
        "directory_rows": len(directory),
        # What the gate cost, by cause, so a reader can tell what this build actually did.
        "equity_rows": equity_rows,
        "etf_rows": etf_rows,
        "no_price": no_price,
        "no_adv": no_adv,
        "passed_filter": len(rows),
        "min_dollar_volume": min_dollar_volume,
        "min_price": min_price,
        "min_cap": DEFAULT_MIN_CAP,     # 0.0 — recorded to say the cap floor is GONE, not applied
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
    # A single fixed temp path lets two concurrent builds (the nightly hook plus a manual POST)
    # interleave their writes and publish a spliced file. Unique per writer.
    tmp = _FILE.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(blob))
    os.replace(tmp, _FILE)      # atomic: a crash mid-write must not leave a half-file


def is_stale(blob: dict | None, *, now: float | None = None) -> bool:
    if not blob or not blob.get("built_at"):
        return True
    return (now or time.time()) - float(blob["built_at"]) > STALE_AFTER_S


def publish(blob: dict, *, previous: dict | None = None) -> tuple[bool, str]:
    """Decide whether a fresh build may REPLACE the stored universe, and say why.

    Shared by POST /universe/build and the nightly scan hook. It lived only in the endpoint, so the
    UNATTENDED path — the one that matters most, because nobody is watching it — would happily
    persist a half-fetched universe and stamp it fresh for a week. Exactly backwards. Both coverage
    rules therefore live HERE, in the one place both callers go through, and not in main.py.

    Returns (ok_to_save, reason). `reason` is always non-empty so every caller can log an outcome.
    """
    if not blob.get("symbols"):
        return False, "build produced no symbols"

    shortfalls: list[str] = []
    if not blob.get("complete"):
        shortfalls.append(f"batch coverage {blob.get('batch_coverage')} "
                          f"({blob.get('empty_batches')} batches failed)")
    sym_cov = blob.get("symbol_coverage")
    # An ABSENT symbol_coverage is not a failing one — a blob built before this field existed simply
    # cannot be judged on it, and refusing on a number that was never measured would be inventing
    # evidence. Only a coverage we can actually read gates the publish.
    if isinstance(sym_cov, (int, float)) and sym_cov < MIN_SYMBOL_COVERAGE:
        shortfalls.append(f"symbol coverage {sym_cov} below {MIN_SYMBOL_COVERAGE} "
                          f"(Yahoo omits unknown symbols from a 200 response, so whole batches can "
                          f"look healthy while answering for half of what they were asked)")

    if not shortfalls:
        return True, (f"rebuilt: {len(blob['symbols'])} symbols "
                      f"from {blob.get('passed_filter')} that passed the filter")
    detail = "; ".join(shortfalls)
    if previous and previous.get("symbols"):
        return False, f"refused: {detail} — keeping the previous universe"
    # No usable previous universe: a partial one beats none, but say so loudly.
    return True, (f"rebuilt PARTIAL: {detail}; {len(blob['symbols'])} symbols — "
                  f"no previous universe to fall back on")
