"""A weekly plan may not name one exposure group twice under two spellings.

Measured 2026-08-10. The strategist wrote a plan that passed every arithmetic check it had: targets
summed to exactly 85% against a 15% cash target, across six named groups, with the notes correctly
citing the cash drag and the blocked-trade history that motivated both numbers. It was still
unfillable, because two of those six lines were `US_EQUITY` 22% and `SP500` 18% — and `SP500` has
been an alias of `US_EQUITY` since the 2026-08-05 merge. One 40% target against a 25% cap. The 18%
had nowhere to go and became idle cash, which is the exact failure the sum-to-100 rule was added to
prevent, arriving through the vocabulary instead of the arithmetic.

`allocation_gap` caught it — but only as a report, after the note was stored, so the daily tick and
every other consumer still read two entries for one exposure. Canonicalising at ingest means the
alias survives only between parse and store.

These tests use the REAL `_exposure_group` deliberately. The last round of this bug was masked by a
unit test whose hand-written group map already did the aliasing the production code was missing, so a
green suite reported a fix that did not exist.

Updated 2026-08-27: the ex-US funds were merged into an `INTL` group (VXUS/IXUS/VEU/VEA/SCHF/SPDW/
IEFA/VWO, all 0.913-0.999 correlated with VXUS on 2y of daily returns), so `VXUS` became a retired
group label exactly as `SP500` did — a ticker that resolves to its group rather than a group name of
its own. The expectations below moved with it; the property each test pins did not.
"""
from __future__ import annotations

from app.main import _crypto_symbol, _exposure_group, _exposure_vocabulary
from app.sandbox_job import allocation_gap, canonicalize_targets


def _note(targets, cash=15.0):
    return {"cash_target_pct": cash,
            "targets": [{"exposure_group": g, "target_pct": p} for g, p in targets]}


def _as_pairs(note):
    return [(t["exposure_group"], t["target_pct"]) for t in note["targets"]]


# ---------------------------------------------------------------- the plan that motivated this

REAL_PLAN = [("US_EQUITY", 22.0), ("VXUS", 22.0), ("SP500", 18.0),
             ("XOM", 10.0), ("AMZN", 8.0), ("BTC", 5.0)]


def test_the_real_plan_collapses_to_five_groups():
    note = _note(REAL_PLAN)
    changed = canonicalize_targets(note, group_of=_exposure_group)
    assert _as_pairs(note) == [("US_EQUITY", 40.0), ("INTL", 22.0),
                               ("XOM", 10.0), ("AMZN", 8.0), ("BTC", 5.0)]
    assert any("twice" in c for c in changed)


def test_merging_preserves_the_total():
    """The percentages are redistributed, never dropped — the plan still sums to what it claimed."""
    note = _note(REAL_PLAN)
    before = sum(p for _, p in REAL_PLAN)
    canonicalize_targets(note, group_of=_exposure_group)
    assert sum(t["target_pct"] for t in note["targets"]) == before == 85.0


def test_the_merge_exposes_the_cap_breach_to_every_consumer():
    """Before: six compliant-looking lines. After: one 40% target that the audit and the tick both
    see for what it is."""
    note = _note(REAL_PLAN)
    canonicalize_targets(note, group_of=_exposure_group)
    gap = allocation_gap(note, max_position_pct=25.0, group_of=_exposure_group)
    assert gap["targets_over_cap"] == ["US_EQUITY"]
    assert gap["groups"] == 5          # not the 6 the plan claimed


def test_ticker_labels_are_resolved_to_their_group():
    """A plan naming funds instead of groups is a spelling error with one correct reading."""
    note = _note([("VTI", 20.0), ("SPY", 15.0), ("VXUS", 25.0), ("FBTC", 10.0), ("BTC", 5.0)])
    canonicalize_targets(note, group_of=_exposure_group)
    assert _as_pairs(note) == [("US_EQUITY", 35.0), ("INTL", 25.0), ("BTC", 15.0)]


# ---------------------------------------------------------------- it must not invent changes

def test_a_clean_plan_is_left_exactly_alone():
    note = _note([("US_EQUITY", 25.0), ("INTL", 25.0), ("SCHD", 20.0), ("AMZN", 15.0)])
    assert canonicalize_targets(note, group_of=_exposure_group) == []
    assert _as_pairs(note) == [("US_EQUITY", 25.0), ("INTL", 25.0),
                               ("SCHD", 20.0), ("AMZN", 15.0)]


def test_genuinely_distinct_groups_are_not_merged():
    """INTL (VXUS 0.773 vs VTI, re-measured 2026-08-27) and SCHD (0.641) do real diversification
    work and keep their own groups. The ex-US merge must not have quietly swallowed either."""
    note = _note([("US_EQUITY", 25.0), ("INTL", 25.0), ("SCHD", 25.0), ("GOLD", 10.0)])
    canonicalize_targets(note, group_of=_exposure_group)
    assert len(note["targets"]) == 4


def test_order_of_first_appearance_is_kept():
    note = _note([("VXUS", 22.0), ("SP500", 18.0), ("AMZN", 8.0), ("US_EQUITY", 22.0)])
    canonicalize_targets(note, group_of=_exposure_group)
    assert [g for g, _ in _as_pairs(note)] == ["INTL", "US_EQUITY", "AMZN"]
    assert dict(_as_pairs(note))["US_EQUITY"] == 40.0


# ---------------------------------------------------------------- degrading safely

def test_no_note_and_no_targets_are_not_errors():
    assert canonicalize_targets(None, group_of=_exposure_group) == []
    assert canonicalize_targets({"targets": []}, group_of=_exposure_group) == []


def test_a_blank_label_does_not_crash_or_merge_into_a_real_group():
    note = _note([("", 10.0), ("US_EQUITY", 25.0)])
    canonicalize_targets(note, group_of=_exposure_group)
    assert dict(_as_pairs(note))["US_EQUITY"] == 25.0


# ---------------------------------------------------------------- the vocabulary the model is shown

def test_the_vocabulary_maps_groups_to_their_members():
    vocab = _exposure_vocabulary(["VTI", "SPY", "VOO", "VXUS", "FBTC", "BTC-USD", "AMZN"])
    assert vocab["US_EQUITY"] == ["SPY", "VOO", "VTI"]
    assert vocab["BTC"] == ["BTC", "FBTC"]
    assert vocab["INTL"] == ["VXUS"]


def test_the_vocabulary_never_offers_a_retired_label():
    """`SP500` must not appear as a nameable group — that spelling is what the model copied.

    `VXUS` joined it as a retired label on 2026-08-27. It is the sharper case of the two: it is still
    a real, holdable TICKER, so a model shown it as a group name has every reason to keep writing
    plans against it, and every such plan would name one member of INTL as though it were the whole
    exposure."""
    vocab = _exposure_vocabulary(["VTI", "SPY", "SPLG", "QQQM", "VXUS"])
    assert "SP500" not in vocab
    assert "VXUS" not in vocab
    assert set(vocab) == {"US_EQUITY", "INTL"}


def test_the_benchmark_is_not_a_tradable_group():
    assert "^GSPC" not in _exposure_vocabulary(["^GSPC", "VTI"])


# ---------------------------------------------------------------- adjacent: crypto symbol shape

def test_crypto_watchlist_entries_are_normalised_not_concatenated():
    """The stored watchlist already carries the suffix, so appending one built `BTC-USD-USD`.

    Found while checking what vocabulary the strategist would receive. Latent, because `allow_crypto`
    was off and the candidate filter drops everything ending in `-USD` — malformed or not — so the
    ghosts were swept up by the same filter that hid them. Turning that setting on would have made
    every crypto candidate a symbol nothing can price.
    """
    stored = ["BTC-USD", "SHIB-USD", "DOGE-USD"]        # exactly what settings_store holds
    assert [_crypto_symbol(c) for c in stored] == stored
    assert _crypto_symbol("btc") == "BTC-USD"           # the other shape, still one suffix
    assert _exposure_vocabulary([_crypto_symbol(c) for c in stored])["BTC"] == ["BTC"]
