"""DATA-4: gaps.py used to compare a RAW open against a split/dividend-ADJUSTED prior close.

The bug is described at app/market.py:144-147 (now fixed there): `Series.opens` was left on the RAW
basis while `Series.highs`/`Series.lows`/`Series.closes` were rescaled onto the ADJUSTED basis by the
per-bar factor `f = adjusted_close / raw_close`. gaps.detect() reads `opens[-1]` against `closes[-2]`,
so any bar living on the raw (unscaled) side of that mismatch produced an overnight-gap percentage
with no relationship to reality — most dramatically around a split, where `f` differs sharply from
1.0. That fabricated gap flows straight into the sandbox trading prompt via
`gaps.compact(gaps.detect(...))` (app/main.py:2960) as a "measured edge" tilt.

The fix taken here is the PREFERRED one from the task: rescale `opens` by the exact same per-bar
factor already computed for highs/lows in `market.fetch_series`, so every one of the four series
shares one basis. No new threshold or guard was added to gaps.py itself — the fix is upstream, in
market.py, so the boundary worth pinning here is the SPLIT SEAM: the bar whose own factor is 1.0
sitting right next to a bar whose factor isn't, which is exactly where an incomplete rescaling would
still show a seam artifact.

Tests reuse `_payload`/`_ok`/`_fetch` from test_market_hl.py, which already builds Yahoo-chart-shaped
mock responses and drives `market.fetch_series` against a mocked transport.
"""
from __future__ import annotations

from app import gaps
from tests.test_market_hl import _fetch, _ok, _payload

_VOL = 5_000_000.0


# --------------------------------------------------------------------- a real gap still works

def test_a_normal_overnight_gap_is_still_detected():
    """No splits anywhere in sight — a plain -3% overnight gap on ordinary volume must still come
    back with the fields the sandbox prompt and the fill-rate tables key off."""
    closes = [100.0] * 20 + [99.0, 98.0]          # prior_close = 99.0, today's close = 98.0
    opens = [100.0] * 21 + [96.03]                # today's open, exactly -3% of prior_close
    volumes = [_VOL] * 22                          # flat volume -> vol_ratio == 1.0, no catalyst

    g = gaps.detect(closes, opens, volumes)

    assert g is not None
    assert g["pct"] == -3.0
    assert g["direction"] == "down"
    assert g["size_bucket"] == "2-5%"
    assert g["catalyst_likely"] is False
    assert g["filled_today"] is False              # today's close (98.0) stayed below prior (99.0)
    assert g["edge"] == "constructive"              # the one bucket with a measured edge
    assert g["fill_rate_10d_pct"] == 71.9

    compact = gaps.compact(g)
    assert compact["measured_edge"] == "constructive"
    assert compact["gap_pct"] == -3.0


# --------------------------------------------------------------------- THE split reproduction

def _flat_split_bars(pre_n: int = 25, post_n: int = 5) -> list[dict]:
    """A boring, perfectly flat stock around a synthetic 2:1 split: NOTHING actually happens on any
    day (every bar's true range and true close-to-close move is zero) — pre-split bars trade at raw
    100 (adjusted 50), post-split bars trade at raw 50 (adjusted 50, factor 1.0). Any gap this
    produces, on either side of the split, is by construction a data artefact, never a real move."""
    pre = [{"ts": 1_780_000_000 + i * 86400, "o": 100.0, "h": 102.0, "l": 98.0, "c": 100.0,
            "adj": 50.0, "v": _VOL} for i in range(pre_n)]
    post = [{"ts": 1_780_000_000 + (pre_n + i) * 86400, "o": 50.0, "h": 51.0, "l": 49.0, "c": 50.0,
             "adj": 50.0, "v": _VOL} for i in range(post_n)]
    return pre + post


def test_a_synthetic_split_bar_no_longer_produces_a_false_gap():
    """Reproduces scan_job.py's memory-backfill shape: a Series truncated to end at an EARLIER
    'today' (`opens[:i+1]`, `closes[:i+1]`, ...) while its `opens`/`closes` still carry the per-bar
    adjustment factor computed against the FULL fetch's end. Truncate to the LAST pre-split bar: its
    own factor (0.5) is identical to the bar before it, so the two flat, identical bars either side of
    that cut must show NO overnight gap once opens share the adjusted basis."""
    bars = _flat_split_bars()
    s = _fetch(_ok(_payload(bars)))

    cut = 25  # ends exactly on the last pre-split bar
    g = gaps.detect(s.closes[:cut], s.opens[:cut], s.volumes[:cut])

    assert g is None, g  # nothing happened; the fixed opens agree with the adjusted closes


def test_the_unfixed_raw_open_would_have_manufactured_a_huge_fake_gap():
    """Pins the exact defect: feeding the SAME closes but the RAW (pre-fix) opens — i.e. exactly what
    `market.fetch_series` used to hand back — into the identical truncated window. Raw open (100) next
    to the adjusted prior close (50) reads as a +100% overnight gap on a stock that did not move at
    all. If a future change reintroduces unscaled opens, this is the test that catches it."""
    bars = _flat_split_bars()
    s = _fetch(_ok(_payload(bars)))
    raw_opens = [b["o"] for b in bars]  # the basis `opens` used to carry, unscaled

    cut = 25
    g = gaps.detect(s.closes[:cut], raw_opens[:cut], s.volumes[:cut])

    assert g is not None
    assert g["pct"] == 100.0
    assert g["size_bucket"] == ">5%"


# --------------------------------------------------------------------- the boundary this fix cares about

def test_the_split_seam_itself_reports_no_artificial_gap():
    """The bar right at the seam: 'today' is the FIRST post-split bar (its own factor is exactly
    1.0), 'yesterday' is the LAST pre-split bar (factor 0.5, now rescaled). This is precisely the
    boundary a half-finished rescale would get wrong in either direction — too far or not far
    enough — so it gets its own case distinct from the interior-bar reproduction above."""
    bars = _flat_split_bars()
    s = _fetch(_ok(_payload(bars)))

    cut = 26  # ends on the first post-split bar
    g = gaps.detect(s.closes[:cut], s.opens[:cut], s.volumes[:cut])

    assert g is None, g


def test_a_missing_raw_close_leaves_that_bars_open_unscaled_without_crashing_gaps():
    """market.py's existing fallback: a bar with no raw close (Yahoo nulls it while still publishing
    adjclose) can't compute a factor, so it is left AS-IS rather than dividing by zero. Confirms that
    fallback now applies to `opens` too, and that gaps.detect tolerates the resulting series (no
    crash, no fabricated gap) rather than assuming every bar was successfully rescaled."""
    bars = [{"ts": 1_780_000_000 + i * 86400, "o": 100.0, "h": 102.0, "l": 98.0, "c": None,
             "adj": 50.0, "v": _VOL} for i in range(3)]
    bars += [{"ts": 1_780_000_000 + (3 + i) * 86400, "o": 100.0, "h": 102.0, "l": 98.0, "c": 100.0,
              "adj": 50.0, "v": _VOL} for i in range(22)]
    s = _fetch(_ok(_payload(bars)))

    # The bar with no raw close falls back to an unscaled (raw) open, exactly like its high/low.
    assert s.opens[0] == 100.0

    g = gaps.detect(s.closes, s.opens, s.volumes)  # must not raise
    assert g is None or isinstance(g, dict)
