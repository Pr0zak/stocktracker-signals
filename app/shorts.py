"""
Short-pressure data layer: short interest, daily short volume, and fails-to-deliver.

Sources (all free, no keys):
- FINRA consolidatedShortInterest API (anonymous POST) — official bi-monthly short interest with
  days-to-cover and full history per symbol. ~9 business days publication lag.
- FINRA Reg SHO daily short-sale volume files (one CSV per day covers ALL symbols) — the only
  same/next-day signal in this family.
- SEC fails-to-deliver half-month zips — FTD share counts per settlement day. Published with a
  2-4 week lag, so treated as context/confirmation, never timing.

Evidence stance baked into the evaluation: high short interest is BEARISH on average (shorts are
informed); the squeeze case is conditional — high days-to-cover ("fuel") only matters once price
and volume confirm ("ignition"). FTD "T+35 cycle" theories have weak evidence; echo dates are
surfaced but explicitly labeled speculative.

Everything caches under data/shorts/ so watchlist-wide scans cost one download per file, not per
symbol. All fetchers are best-effort and return None rather than raise.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import statistics
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx

_DATA_DIR = Path(os.environ.get("SIGNALS_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
_DIR = _DATA_DIR / "shorts"
_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
# SEC fair-access policy asks for an identifying UA; Akamai 403s requests without an Accept header.
_SEC_UA = {
    "User-Agent": "StockTracker-Signals personal use (github.com/Pr0zak/stocktracker-signals)",
    "Accept": "*/*",
}

_SI_TTL = 12 * 3600         # FINRA SI cache per symbol
_LOCK = asyncio.Lock()      # serialize file-cache writes


def _read(name: str) -> dict | None:
    try:
        return json.loads((_DIR / name).read_text())
    except Exception:  # noqa: BLE001
        return None


def _write(name: str, obj: dict) -> None:
    _DIR.mkdir(parents=True, exist_ok=True)
    (_DIR / name).write_text(json.dumps(obj))


# --- FINRA short interest (bi-monthly, official) ---

async def short_interest(client: httpx.AsyncClient, symbol: str) -> dict | None:
    """Latest + history of official short interest for one symbol. Cached 12h."""
    sym = symbol.upper()
    cached = _read(f"si_{sym}.json")
    if cached and time.time() - cached.get("fetched_at", 0) < _SI_TTL:
        return cached
    try:
        r = await client.post(
            "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest",
            headers={"Content-Type": "application/json", "Accept": "application/json", **_UA},
            json={
                "limit": 1000,
                "compareFilters": [
                    {"compareType": "EQUAL", "fieldName": "symbolCode", "fieldValue": sym},
                ],
            },
            timeout=20,
        )
        r.raise_for_status()
        rows = sorted(r.json(), key=lambda x: x.get("settlementDate", ""))
    except Exception:  # noqa: BLE001
        return cached  # stale beats nothing
    if not rows:
        return None
    hist = [
        {
            "date": x.get("settlementDate"),
            "si": x.get("currentShortPositionQuantity"),
            "adv": x.get("averageDailyVolumeQuantity"),
            "dtc": x.get("daysToCoverQuantity"),
            "change_pct": x.get("changePercent"),
        }
        for x in rows[-24:]  # ~1 year of half-month records
    ]
    out = {"fetched_at": time.time(), "symbol": sym, "latest": hist[-1], "history": hist}
    async with _LOCK:
        _write(f"si_{sym}.json", out)
    return out


# --- FINRA daily short-sale volume (whole-market file per day) ---

async def _shvol_day(client: httpx.AsyncClient, day: str) -> dict | None:
    """Parsed {SYM: (short, total)} for one YYYYMMDD trading day (immutable → cached forever)."""
    cached = _read(f"shvol_{day}.json")
    if cached is not None:
        return cached.get("rows")
    try:
        r = await client.get(f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{day}.txt", timeout=20)
        if r.status_code != 200:
            return None  # weekend/holiday/not published yet
        rows: dict[str, list[float]] = {}
        for line in r.text.splitlines()[1:]:
            p = line.split("|")
            if len(p) >= 5 and p[0] == day:
                rows[p[1]] = [float(p[2]), float(p[4])]
    except Exception:  # noqa: BLE001
        return None
    async with _LOCK:
        _write(f"shvol_{day}.json", {"rows": rows})
    return rows


async def short_volume(client: httpx.AsyncClient, symbol: str, days: int = 5) -> dict | None:
    """Short-volume ratio per day for the last `days` trading days (walks back over the calendar)."""
    sym = symbol.upper()
    out = []
    d = date.today()
    tried = 0
    while len(out) < days and tried < days * 3 + 6:
        d -= timedelta(days=1)
        tried += 1
        if d.weekday() >= 5:
            continue
        rows = await _shvol_day(client, d.strftime("%Y%m%d"))
        if not rows or sym not in rows:
            continue
        short, total = rows[sym]
        if total > 0:
            out.append({"date": d.strftime("%Y%m%d"), "ratio": round(short / total, 4)})
    if not out:
        return None
    ratios = [x["ratio"] for x in out]
    return {"days": list(reversed(out)), "avg": round(sum(ratios) / len(ratios), 4), "latest": out[0]}


# --- SEC fails-to-deliver (half-month zips) ---

def _ftd_periods(n: int) -> list[str]:
    """Most-recent-first period codes like 202606b, honoring the ~3-week publication lag."""
    today = date.today()
    periods = []
    y, m = today.year, today.month
    for _ in range(n + 2):
        periods.append(f"{y}{m:02d}b")
        periods.append(f"{y}{m:02d}a")
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    # Drop periods that can't be published yet (half-month end + ~21 days).
    ok = []
    for p in periods:
        yy, mm, half = int(p[:4]), int(p[4:6]), p[6]
        end = date(yy, mm, 15) if half == "a" else (date(yy + (mm == 12), mm % 12 + 1, 1) - timedelta(days=1))
        if today >= end + timedelta(days=21):
            ok.append(p)
    return ok[:n]


async def _ftd_period(client: httpx.AsyncClient, period: str) -> dict | None:
    """Parsed {SYM: [[settle_date, qty, price], ...]} for one half-month. Cached forever."""
    cached = _read(f"ftd_{period}.json")
    if cached is not None:
        return cached.get("rows")
    try:
        r = await client.get(
            f"https://www.sec.gov/files/data/fails-deliver-data/cnsfails{period}.zip",
            headers=_SEC_UA, timeout=30, follow_redirects=True,
        )
        if r.status_code != 200:
            return None
        z = zipfile.ZipFile(io.BytesIO(r.content))
        text = z.read(z.namelist()[0]).decode("latin-1")
        rows: dict[str, list] = {}
        for line in text.splitlines()[1:]:
            p = line.split("|")
            if len(p) >= 6 and p[0].isdigit():
                try:
                    rows.setdefault(p[2], []).append([p[0], int(p[3]), float(p[5]) if p[5] else None])
                except ValueError:
                    continue
    except Exception:  # noqa: BLE001
        return None
    async with _LOCK:
        _write(f"ftd_{period}.json", {"rows": rows})
    return rows


async def ftd(client: httpx.AsyncClient, symbol: str, periods: int = 12) -> dict | None:
    """~6 months of FTD history for one symbol: series + trend + spike detection."""
    sym = symbol.upper()
    series: list[list] = []
    for p in reversed(_ftd_periods(periods)):  # oldest → newest
        rows = await _ftd_period(client, p)
        if rows and sym in rows:
            series.extend(sorted(rows[sym]))
    if not series:
        return None
    qtys = [q for _, q, _ in series]
    med = statistics.median(qtys) if qtys else 0
    recent = series[-6:]
    spikes = [d for d, q, _ in series if med > 0 and q > 3 * med and q > 10000]
    trend = "quiet"
    if len(series) >= 4:
        older = statistics.median(qtys[:-4]) if qtys[:-4] else 0
        newer = statistics.median(qtys[-4:])
        if older > 0 and newer > 1.8 * older:
            trend = "rising"
        elif older > 0 and newer < 0.55 * older:
            trend = "falling"
    return {
        "series": series[-40:],
        "latest": series[-1],
        "median": med,
        "trend": trend,
        "spike_dates": spikes[-8:],
        "recent_total": sum(q for _, q, _ in recent),
    }


# --- event study: what actually happened after past FTD spikes for THIS symbol ---

def event_study(spike_dates: list[str], dates: list[str], closes: list[float]) -> dict | None:
    """Median forward returns after past FTD-spike settlement days, using the symbol's own history."""
    if not spike_dates or len(closes) < 30:
        return None
    idx = {d: i for i, d in enumerate(dates)}
    fwd5, fwd10 = [], []
    for sd in spike_dates:
        i = idx.get(sd)
        if i is None:  # settlement day may be missing — take the next trading day
            later = [j for j, d in enumerate(dates) if d > sd]
            i = later[0] if later else None
        if i is None:
            continue
        if i + 5 < len(closes):
            fwd5.append((closes[i + 5] - closes[i]) / closes[i] * 100)
        if i + 10 < len(closes):
            fwd10.append((closes[i + 10] - closes[i]) / closes[i] * 100)
    if not fwd5:
        return None
    return {
        "events": len(fwd5),
        "fwd5_median_pct": round(statistics.median(fwd5), 2),
        "fwd10_median_pct": round(statistics.median(fwd10), 2) if fwd10 else None,
        "fwd10_hit_rate": round(sum(1 for x in fwd10 if x > 0) / len(fwd10), 2) if fwd10 else None,
    }


# --- upcoming dates (deterministic calendar, no fetch) ---

def _next_third_friday(after: date) -> date:
    d = date(after.year, after.month, 1)
    fridays = 0
    while True:
        if d.weekday() == 4:
            fridays += 1
            if fridays == 3 and d > after:
                return d
        d += timedelta(days=1)
        if d.day == 1:
            fridays = 0


def _add_business_days(d: date, n: int) -> date:
    while n > 0:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def upcoming_dates(ftd_spikes: list[str] | None = None) -> list[dict]:
    """Known/estimated future dates that matter for short mechanics, soonest first."""
    today = date.today()
    out: list[dict] = []
    # Next two FINRA SI settlement dates (15th & end-of-month) + estimated publication (+9 biz days).
    settles: list[date] = []
    y, m = today.year, today.month
    for _ in range(3):
        eom = (date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1))
        for c in (date(y, m, 15), eom):
            if c > today:
                settles.append(c)
        m += 1
        if m == 13:
            y, m = y + 1, 1
    for s in settles[:2]:
        out.append({"date": s.isoformat(), "label": "Short-interest settlement (FINRA)", "kind": "si_settlement"})
        out.append({
            "date": _add_business_days(s, 9).isoformat(),
            "label": "Short-interest data published (~est.)", "kind": "si_publication",
        })
    # Monthly options expiry (3rd Friday) — gamma/covering mechanics often cluster here.
    opex = _next_third_friday(today)
    out.append({"date": opex.isoformat(), "label": "Monthly options expiry (OPEX)", "kind": "opex"})
    # Speculative T+35 echo windows from recent FTD spikes — clearly labeled, weak evidence.
    for sd in (ftd_spikes or []):
        try:
            echo = datetime.strptime(sd, "%Y%m%d").date() + timedelta(days=35)
        except ValueError:
            continue
        if today <= echo <= today + timedelta(days=45):
            out.append({
                "date": echo.isoformat(),
                "label": f"T+35 echo of {sd[:4]}-{sd[4:6]}-{sd[6:]} FTD spike (speculative)",
                "kind": "t35_echo",
            })
    out.sort(key=lambda x: x["date"])
    return out[:8]


# --- the composite read ---

async def short_pressure(
    client: httpx.AsyncClient, symbol: str,
    dates: list[str] | None = None,
    closes: list[float] | None = None,
    volumes: list[float | None] | None = None,
) -> dict | None:
    """Full short-pressure block for one stock. Price/volume series (optional) unlock the
    ignition check and the event study."""
    si_task = short_interest(client, symbol)
    sv_task = short_volume(client, symbol)
    ftd_task = ftd(client, symbol)
    si, sv, f = await asyncio.gather(si_task, sv_task, ftd_task)
    if si is None and sv is None and f is None:
        return None

    reasons: list[str] = []
    dtc = (si or {}).get("latest", {}).get("dtc")
    si_change = (si or {}).get("latest", {}).get("change_pct")
    sv_avg = (sv or {}).get("avg")

    # FUEL: enough open shorts that covering takes days.
    fuel = bool(dtc and dtc >= 5)
    if fuel:
        reasons.append(f"Days-to-cover {dtc:.1f} — covering is slow (fuel)")
    if si_change is not None:
        reasons.append(f"Short interest {'+' if si_change >= 0 else ''}{si_change:.1f}% vs prior period")
    if sv_avg is not None and sv_avg >= 0.45:
        reasons.append(f"Elevated daily short-volume ratio ({sv_avg:.0%} 5-day avg)")
    if f and f.get("trend") == "rising":
        reasons.append("FTDs rising vs their 6-month median (settlement stress)")

    # IGNITION: fuel + the tape confirming (price above rising 20d avg on expanding volume).
    ignition = False
    if fuel and closes and len(closes) >= 25:
        sma20 = sum(closes[-20:]) / 20
        sma20_prev = sum(closes[-25:-5]) / 20
        vols = [v for v in (volumes or [])[-20:] if v]
        vol_expand = bool(vols) and volumes and volumes[-1] and volumes[-1] > 1.5 * (sum(vols) / len(vols))
        if closes[-1] > sma20 > sma20_prev and vol_expand:
            ignition = True
            reasons.append("Price above rising 20-day average on expanded volume (ignition)")
    state = "ignition" if ignition else ("fuel" if fuel else "quiet")

    study = None
    if f and dates and closes:
        study = event_study(f.get("spike_dates", []), dates, closes)

    return {
        "state": state,
        "days_to_cover": dtc,
        "short_interest": (si or {}).get("latest", {}).get("si"),
        "si_change_pct": si_change,
        "si_date": (si or {}).get("latest", {}).get("date"),
        "si_history": [{"date": h["date"], "dtc": h["dtc"]} for h in (si or {}).get("history", [])[-12:]],
        "short_vol_ratio_5d": sv_avg,
        "short_vol_days": (sv or {}).get("days"),
        "ftd_latest": (f or {}).get("latest"),
        "ftd_trend": (f or {}).get("trend"),
        "ftd_series": (f or {}).get("series", [])[-16:],
        "event_study": study,
        "upcoming": upcoming_dates((f or {}).get("spike_dates")),
        "reasons": reasons,
    }


def compact(sp: dict | None) -> dict | None:
    """The slim version embedded in analyst snapshots (keeps prompt tokens sane)."""
    if not sp:
        return None
    return {
        "state": sp["state"],
        "days_to_cover": sp["days_to_cover"],
        "si_change_pct": sp["si_change_pct"],
        "short_vol_ratio_5d": sp["short_vol_ratio_5d"],
        "ftd_trend": sp["ftd_trend"],
        "reasons": sp["reasons"][:4],
    }
