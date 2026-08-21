"""
Persistence for the AI paper-trading sandbox.

Three shapes, mirroring the rest of the service:
  • data/sandbox.json         — the mutable ledger (cash + positions + settings + cursors), a single
                                blob fully read / fully rewritten as one consistent snapshot. Written
                                ATOMICALLY (temp file + fsync + os.replace) with a .bak of the prior
                                blob, so a crash mid-write can never corrupt the account or re-zero it.
  • data/sandbox_trades.jsonl — append-only audit log, one row per filled OR skipped order.
  • data/sandbox_nav.jsonl    — append-only equity curve, one row per tick (incl. the benchmark).

data/ is gitignored, so all of this survives the git-reset self-update. All mutations to the blob are
serialized by callers via an asyncio lock in main.py (single uvicorn worker = sole writer); the
threading.Lock here only guards the file I/O itself.

ARMS. The account above is one ARM, named `main`. Additional arms are independent ledgers — their own
cash, positions, settings and decision engine — run against the SAME market snapshot on the same tick,
so their equity curves are comparable and the difference between them is the strategy rather than the
weather. That is the point: with one arm, every change is a guess measured against noise, and the
question actually on the table (is 52% cash the strategist's fault or the mechanism's?) has no
experiment that can answer it.

`main` deliberately keeps the original three paths rather than moving under `arms/main/`. It holds the
real history and there is no version of a migration that is safer than not migrating; extra arms live
at data/arms/<id>/ and cost the existing account nothing.
"""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path

_DATA_DIR = Path(os.environ.get("SIGNALS_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
_lock = threading.RLock()

VERSION = 1

MAIN_ARM = "main"
# "rejects" takes another arm's REVIEW-REJECTED orders and executes them. It is the control group
# for the review model: without it, a reviewer that quietly cuts good trades and one that saves the
# account from bad ones produce the same visible result -- a slightly different equity curve with no
# way to attribute it. This arm makes the rejected trades observable instead of counterfactual.
ENGINES = ("llm", "rules", "rejects")
# An arm id becomes a DIRECTORY NAME, so it is validated as one before it is ever joined to a path.
# Whitelist rather than blacklist: `..`, absolute paths, separators, NUL and unicode lookalikes are
# all excluded by construction instead of by remembering to check for each of them.
_ARM_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


class ArmError(ValueError):
    """Bad arm id, or an operation refused on a reserved arm."""


def validate_arm(arm: str) -> str:
    arm = (arm or "").strip().lower()
    if not _ARM_RE.fullmatch(arm):
        raise ArmError(
            "arm id must be 1-32 chars of a-z, 0-9, '_' or '-', starting alphanumeric")
    return arm


def _paths(arm: str) -> tuple[Path, Path, Path, Path, Path, Path]:
    """(ledger, ledger.bak, trades, nav, changes, inputs) for an arm. `main` keeps the flat layout."""
    arm = validate_arm(arm)
    d = _DATA_DIR if arm == MAIN_ARM else _DATA_DIR / "arms" / arm
    return (d / "sandbox.json", d / "sandbox.json.bak",
            d / "sandbox_trades.jsonl", d / "sandbox_nav.jsonl",
            d / "sandbox_changes.jsonl",
            d / "sandbox_inputs.jsonl")


def list_arms() -> list[str]:
    """`main` first, then any extra arms alphabetically. `main` is always listed even before its file
    exists, because it is the account every existing caller means when it says nothing at all."""
    out = [MAIN_ARM]
    root = _DATA_DIR / "arms"
    if root.is_dir():
        for p in sorted(root.iterdir()):
            if p.is_dir() and _ARM_RE.fullmatch(p.name) and p.name != MAIN_ARM:
                out.append(p.name)
    return out

# Defaults are conservative — the sandbox does nothing until it's funded and turned on.
DEFAULT_SETTINGS = {
    "master_enabled": False,           # off until the user opts in
    "risk_tolerance": "balanced",      # conservative | balanced | aggressive
    "retirement_date": None,           # ISO yyyy-mm-dd; biases the glidepath (de-risk as it nears)
    # Age-based glidepath — clearer than a date for "how much runway is left". Drives the growth-vs-
    # income tilt (see the dividend research in research/README.md).
    #
    # `birth_date` is the input; `current_age` is the legacy one and now only a fallback. An age is a
    # fact with a one-year shelf life, and this one is load-bearing — the runway is (retirement_age -
    # current_age), so a number typed once keeps reporting the runway it had on the day it was typed.
    # sandbox_job.effective_age() derives the age per tick and prefers the date whenever it is set.
    "birth_date": None,                # ISO yyyy-mm-dd; server-side only, never sent to the model
    "current_age": None,               # legacy static age; used only when birth_date is unset
    "retirement_age": 65,
    "exit_date": None,                 # ISO yyyy-mm-dd; flatten to cash by this date
    "max_position_pct": 20.0,          # cap per EXPOSURE GROUP (BTC+FBTC count as one)
    "cash_floor_pct": 10.0,            # never deploy below this % of equity
    # Crypto exposure is split in two because the vehicles differ in cost: a spot-crypto ETF
    # (IBIT/FBTC/FETH) trades like any equity inside a brokerage, while DIRECT spot (BTC-USD) usually
    # carries higher trading spreads/fees and custody friction. So ETFs are on, direct spot is off.
    "allow_crypto": False,       # direct spot crypto (BTC-USD, ETH-USD)
    "allow_crypto_etf": True,    # spot-crypto ETFs (IBIT, FBTC, FETH, …)
    # Which spot-BITCOIN ETF to use when it wants bitcoin exposure. Every vehicle in the BTC exposure
    # group holds the same asset, so the choice is about custody, liquidity and fees rather than
    # exposure — the user's call, not the model's. Enforced on BUYS server-side (see
    # sandbox_job.prefer_btc_etf); sells are never redirected, since you can only sell what you hold.
    "preferred_btc_etf": "FBTC",
    "allow_etf": True,
    # Smallest company the market screen may propose, in USD of market cap. The universe reaches the
    # whole market now, and the screens that give it breadth (actives, gainers, small-cap growth) are
    # also where the speculative end lives — this is the dial between "every listed name above the
    # junk floor" and "only established companies". ETFs are unaffected: they report AUM, not market
    # cap, and filtering a broad index fund by company size is a category error.
    "min_market_cap": 2_000_000_000.0,
    "slippage_bps": 5,                 # modeled fill slippage (buys up, sells down)
    "max_trades_per_tick": 4,
    "max_new_positions_per_tick": 2,
    "min_conviction_to_trade": 55,     # 0-100 floor for a buy
    # Second-opinion layer. A deeper model reads the proposed orders before they reach the ledger and
    # can drop individual ones or reject the lot. validate_and_fill checks ARITHMETIC; this checks
    # the argument, which is where both of this account's worst decisions went wrong (2026-08-03
    # IBIT, 2026-08-06 SPY->VTI) -- each was mechanically legal and each was reasoned badly.
    #
    # OFF by default. It costs one extra deep call per tick, and whether it earns that is exactly the
    # kind of question the comparison arms exist to answer -- so it ships switchable, per arm.
    "review_enabled": False,
    # For an engine="rejects" arm: whose rejected orders to take. Ignored by every other engine.
    "rejects_from": "",
    # Honour the analyst's per-buy entry zone: skip (and retry a later day) when the market is above
    # the zone's top, instead of chasing an extended price. Sells always execute at the market.
    "respect_entry_zones": True,
    # Goals & horizon
    "goal_amount": None,               # target $ the AI should push toward (None = no explicit goal)
    "goal_date": None,                 # ISO date by which to hit the goal
    # Funds automation
    "monthly_deposit": 0.0,            # auto-add this much fictional cash each new month (0 = off)
    # Universe
    "exclusions": [],                  # tickers the AI must never buy (uppercase, e.g. ["TSLA"])
    # Realism: mimic a retail brokerage account's actual rules.
    #  cash   = T+1 settlement, sale proceeds unusable until the next session (no PDT rule)
    #  margin = proceeds usable immediately, but the FINRA PDT rule applies under $25k equity
    "account_type": "cash",
    "avoid_wash_sales": True,          # don't rebuy a name sold at a loss within 30 days (IRS §1091)
    # Whether gains are taxed. In a TAXABLE account a sale under one year is a short-term gain taxed
    # at ordinary-income rates rather than the long-term rate, which is a real cost of trimming a
    # young winner — so the decision model is shown each position's holding period and told to weigh
    # it. Set False for an IRA/401k, where holding period is irrelevant and the model should ignore
    # it entirely rather than "protecting" a gain that is never taxed.
    "taxable_account": True,
    # Yield paid on idle cash, annualised. Real brokerage cash sits in a sweep/MMF earning roughly the
    # T-bill rate; modelling it at 0 overstates the cost of being uninvested and pushes the strategist
    # to deploy harder than reality justifies. Set 0 to model a non-interest-bearing account.
    "cash_apy_pct": 4.3,
    # Automation
    "cadence": "daily",                # "daily" | "weekly" — how often it DECIDES (NAV still marks daily)
    # Whether the tick may trade during the 16:00-20:00 ET after-hours window. OFF by default: extended
    # -hours books are thin and spreads are wide, so a fill there is less realistic and riskier. With it
    # off the tick only trades in the regular session (crypto is 24/7 and unaffected).
    "allow_after_hours": False,
    # Churn control: cap the % of equity that may CHANGE HANDS in one decision (buys+sells notional).
    # 0 = unlimited. Lower = calmer, buy-and-hold-ish; higher = lets it reposition aggressively.
    "max_turnover_pct": 25.0,
    # App-side preference, stored here so it rides with the account: push a notification per trade.
    "notify_on_trade": True,
    # Per-arm LLM backbone. None = whatever the service's scan_model is set to, which is what every
    # arm should use unless the BACKBONE is the thing being compared — an arm that pins a model and
    # also changes a limit has two variables and measures neither.
    "model": None,
}


def _defaults(arm: str = MAIN_ARM, *, engine: str = "llm", label: str | None = None) -> dict:
    return {
        "version": VERSION,
        # Identity travels INSIDE the blob as well as in its path, so a row copied out of one arm's
        # log can still say which book it came from.
        "arm": arm,
        "label": label or ("Main" if arm == MAIN_ARM else arm),
        "engine": engine,           # "llm" = the analyst decides; "rules" = mechanical, no model
        "created_at": None,
        "funded_total": 0.0,
        "cash": 0.0,
        "realized_pl_total": 0.0,
        "positions": [],   # [{symbol, shares, avg_cost, exposure_group, opened_at, last_add_at}]
        "benchmark": {"symbol": "^GSPC", "shares": 0.0, "cost_basis": 0.0},
        "settings": dict(DEFAULT_SETTINGS),
        "last_tick_date": None,           # ET yyyy-mm-dd — idempotency cursor
        # One-line read of what the last tick decided and why. The only durable record of a tick that
        # proposed NOTHING: the trade log only holds orders, so without this a deliberate hold and a
        # blocked one leave identical traces (none).
        "last_posture": "",
        # {date, approve, dropped, concerns, note} from the review model's last verdict, or {} if it
        # has never run on this arm. Distinguishing "approved" from "never ran" is the point.
        "last_review": {},
        "last_decision_date": None,       # ET yyyy-mm-dd of the last DECISION (drives weekly cadence)
        "last_deposit_month": None,       # "yyyy-mm" of the last recurring deposit
        "last_weekly_review_date": None,
        "last_strategy_note": None,
    }


def _load(arm: str = MAIN_ARM) -> dict:
    """Load an arm's ledger, falling back to its .bak (last-known-good) then to fresh defaults.
    Settings are merged over DEFAULT_SETTINGS so a new setting key always has a value."""
    f, bak = _paths(arm)[:2]   # slice, not a fixed-width unpack: the tuple grows
    for p in (f, bak):
        if p.exists():
            try:
                blob = _defaults(arm)
                blob.update(json.loads(p.read_text()))
                blob["settings"] = {**DEFAULT_SETTINGS, **(blob.get("settings") or {})}
                # The path is the authority on which arm this is. A blob restored from the wrong
                # backup, or hand-edited, must not be able to claim it belongs to another book.
                blob["arm"] = arm
                if blob.get("engine") not in ENGINES:
                    blob["engine"] = "llm"
                return blob
            except Exception:  # noqa: BLE001 — try the .bak, then defaults
                continue
    return _defaults(arm)


# Cache per arm, populated lazily. `main` is loaded eagerly to preserve the original import-time
# behaviour that the rest of the service was written against.
_cache: dict[str, dict] = {MAIN_ARM: _load(MAIN_ARM)}


def get(arm: str = MAIN_ARM) -> dict:
    """A deep copy of an arm's ledger (safe to mutate by the caller before save())."""
    arm = validate_arm(arm)
    with _lock:
        if arm not in _cache:
            _cache[arm] = _load(arm)
        return copy.deepcopy(_cache[arm])


def _atomic_write_json(path: Path, bak: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(tmp, 0o600)
    if path.exists():
        try:
            shutil.copy2(path, bak)    # last-known-good before we overwrite
        except Exception:  # noqa: BLE001
            pass
    os.replace(tmp, path)              # atomic on a single filesystem


def save(blob: dict, arm: str | None = None) -> dict:
    """Atomically persist an arm's ledger and update the in-memory copy. Returns the saved blob.

    The arm comes from the blob itself unless overridden, so a caller that read one arm and wrote it
    back cannot silently land it on another."""
    arm = validate_arm(arm or blob.get("arm") or MAIN_ARM)
    f, bak = _paths(arm)[:2]   # slice, not a fixed-width unpack: the tuple grows
    with _lock:
        f.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(f, bak, blob)
        _cache[arm] = copy.deepcopy(blob)
        return copy.deepcopy(_cache[arm])


def _append_jsonl(path: Path, row: dict) -> None:
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")


def append_trade(row: dict, arm: str = MAIN_ARM) -> None:
    _append_jsonl(_paths(arm)[2], row)


def append_nav(row: dict, arm: str = MAIN_ARM) -> None:
    _append_jsonl(_paths(arm)[3], row)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:  # noqa: BLE001 — skip a corrupt/partial line
            continue
    return out


def read_trades(limit: int = 200, arm: str = MAIN_ARM) -> list[dict]:
    """Most recent trade rows first (filled + skipped)."""
    rows = _read_jsonl(_paths(arm)[2])
    return list(reversed(rows))[: max(1, limit)]


def append_change(row: dict, arm: str = MAIN_ARM) -> None:
    """Record one settings change. Append-only, like the trade and NAV logs.

    Nothing recorded configuration changes, and the ledger is where their effect shows up. On
    2026-08-17 five settings were changed in an afternoon -- the candidate universe, the screen
    width, the market-cap floor, the affordability filter, the memory backfill rate -- and none of it
    left a trace. When the comparison arms then diverge, there is no way to tell strategy from
    configuration, which is the one question the arms exist to answer.
    """
    _append_jsonl(_paths(arm)[4], row)


def read_changes(limit: int = 100, arm: str = MAIN_ARM) -> list[dict]:
    """Most recent settings changes first."""
    rows = _read_jsonl(_paths(arm)[4])
    return list(reversed(rows))[: max(1, limit)]


def append_inputs(row: dict, arm: str = MAIN_ARM) -> None:
    """Record what the model was SHOWN this tick. Append-only, like the trade and NAV logs.

    The companion to the trade log: that says what was decided, this says what it was decided from.
    Without it a stated reason quoting a number could not be checked against anything -- and on
    2026-08-21 both the analyst and its reviewer made claims about the input data that turned out to
    be wrong, with nothing on disk able to settle either.
    """
    _append_jsonl(_paths(arm)[5], row)


def read_inputs(limit: int = 10, arm: str = MAIN_ARM) -> list[dict]:
    """Most recent input snapshots first. Small `limit` by default -- each row holds every candidate."""
    rows = _read_jsonl(_paths(arm)[5])
    return list(reversed(rows))[: max(1, limit)]


def read_nav(days: int | None = None, arm: str = MAIN_ARM) -> list[dict]:
    """The NAV/equity series oldest-first; optionally only the last `days` calendar days."""
    rows = _read_jsonl(_paths(arm)[3])
    if days:
        cutoff = time.time() - days * 86400
        rows = [r for r in rows if float(r.get("ts", 0)) >= cutoff]
    return rows


def reset(arm: str = MAIN_ARM) -> dict:
    """Wipe an arm's ledger back to fresh defaults and rotate its append-only logs to .bak (never
    truncated in place — the history is preserved on disk). Returns the fresh blob.

    The arm's IDENTITY survives the reset: a reset arm is the same experiment starting over, not a
    nameless one, and silently reverting `engine` to "llm" would turn the rules arm into a second copy
    of main without saying so."""
    arm = validate_arm(arm)
    prior = get(arm)
    with _lock:
        for p in _paths(arm)[2:]:
            if p.exists():
                try:
                    p.replace(p.with_suffix(p.suffix + f".bak.{int(time.time())}"))
                except Exception:  # noqa: BLE001
                    pass
    return save(_defaults(arm, engine=prior.get("engine", "llm"), label=prior.get("label")), arm)


def create_arm(arm: str, *, engine: str = "rules", label: str | None = None,
               settings: dict | None = None, seed_settings_from: str | None = MAIN_ARM,
               clone_from: str | None = None) -> dict:
    """Create a new arm. Refuses to overwrite an existing one.

    Settings are seeded from another arm by default, so a comparison starts as a genuine A/B — the
    variable under test is whatever the caller then overrides, not forty defaults that silently
    drifted apart from the account being compared against.

    `clone_from` copies the BOOK too — cash, positions, funded total and the benchmark shadow. That
    is usually what you want when adding an arm to a running account: an arm started empty today
    against one that has been invested for three weeks differs by a head start as well as by its
    strategy, and the head start is the larger effect for months. Cloning makes today the common
    ancestor, so everything after it is attributable. The benchmark shadow is copied with the rest,
    which is the point — both arms then measure excess return from the same starting line."""
    arm = validate_arm(arm)
    if arm == MAIN_ARM:
        raise ArmError("'main' already exists and cannot be re-created")
    if _paths(arm)[0].exists():
        raise ArmError(f"arm '{arm}' already exists")
    if engine not in ENGINES:
        raise ArmError(f"engine must be one of {ENGINES}")
    blob = _defaults(arm, engine=engine, label=label)
    if seed_settings_from:
        blob["settings"] = {**DEFAULT_SETTINGS, **(get(seed_settings_from).get("settings") or {})}
    if clone_from:
        src = get(validate_arm(clone_from))
        blob.update({
            "cash": float(src.get("cash") or 0.0),
            "positions": copy.deepcopy(src.get("positions") or []),
            "funded_total": float(src.get("funded_total") or 0.0),
            "realized_pl_total": float(src.get("realized_pl_total") or 0.0),
            "interest_total": float(src.get("interest_total") or 0.0),
            "benchmark": copy.deepcopy(src.get("benchmark") or _defaults()["benchmark"]),
            # The plan is shared at tick time, but carrying it here means the arm can be inspected
            # (and can act) before its first tick rather than looking planless.
            "last_strategy_note": copy.deepcopy(src.get("last_strategy_note")),
            # Deliberately NOT copied: last_tick_date and last_decision_date. Inheriting today's
            # cursor would gate the new arm out of its own first tick.
            "last_deposit_month": src.get("last_deposit_month"),
            # Unsettled proceeds belong to trades this arm never made.
            "unsettled": [],
            "recent_loss_sales": {},
        })
    # An arm is enabled explicitly by the caller, never by inheritance from a live account.
    blob["settings"]["master_enabled"] = False
    blob["created_at"] = time.time()
    if settings:
        blob["settings"].update(settings)
    return save(blob, arm)


def delete_arm(arm: str) -> None:
    """Remove an arm's directory entirely. `main` is not deletable — it is the real account."""
    arm = validate_arm(arm)
    if arm == MAIN_ARM:
        raise ArmError("the 'main' arm cannot be deleted")
    d = (_DATA_DIR / "arms" / arm)
    with _lock:
        _cache.pop(arm, None)
        if d.is_dir():
            shutil.rmtree(d)
