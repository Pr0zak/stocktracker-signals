"""Adding a fee to minimise is adding an incentive, so the guard ships in the same commit.

On 2026-08-06 the model sold all 3 SPY to buy VTI because 0.095% is a worse expense ratio than
0.030%. Correct for new money, wrong for a holding: it banked $85.56 of short-term gains to save
about $1.50 a year, and the replacement buy was then cap-blocked so the proceeds never even reached
VTI. Two separate prompt rules forbade it in words. The lesson written down afterwards was that a
mechanically detectable constraint belongs in `validate_and_fill`, not in a paragraph — and that when
an INCENTIVE is added to the model's inputs, the guard against its pathological path goes in with it.

`_EXPENSE_RATIO_PCT` grew on 2026-08-27 to cover gold, international, dividend and bitcoin vehicles.
The sharpest new number is GLD at 0.400% beside GLDM at 0.100% — same bullion, same issuer, four
times the fee — which is exactly the shape of the argument that lost $85.56. These tests pin the
grouping that makes that swap unprofitable to attempt, so the fee data can only ever steer new money.

Correlations quoted below are 2y of daily returns measured 2026-08-27, the same basis the
_EXPOSURE_GROUP comments already cite.
"""
from __future__ import annotations

from app.main import _EXPENSE_RATIO_PCT, _exposure_group
from app.sandbox_job import intra_group_swaps


def test_the_cheap_twin_of_every_priced_gold_fund_is_the_same_exposure():
    """GLD 0.400% vs GLDM 0.100% is the biggest like-for-like gap in the map. All measured at
    0.999-1.000 against GLD, so none of them is a diversification argument."""
    for sym in ("GLD", "GLDM", "IAUM", "IAU", "SGOL", "OUNZ"):
        assert _exposure_group(sym) == "GOLD", sym
        assert sym in _EXPENSE_RATIO_PCT, sym


def test_selling_gld_to_buy_the_cheaper_gldm_is_dropped():
    """The 2026-08-06 trade, rerun with today's widest fee gap. Both legs go, not just the buy —
    a half-executed swap leaves the account in cash, which is how the SPY sale actually hurt."""
    orders = [
        {"symbol": "GLD", "side": "sell", "shares": 2,
         "reason": "0.400% expense ratio vs 0.100% for identical bullion"},
        {"symbol": "GLDM", "side": "buy", "dollars": 845},
    ]
    assert intra_group_swaps(orders, group_of=_exposure_group) == {0, 1}


def test_selling_vxus_to_buy_a_cheaper_ex_us_fund_is_dropped():
    """The same trade in the group added on 2026-08-27. VEU is 1bp cheaper than VXUS and 0.998
    correlated with it; there is no version of that swap worth a realised gain."""
    orders = [{"symbol": "VXUS", "side": "sell", "shares": 16},
              {"symbol": "VEU", "side": "buy", "dollars": 1405}]
    assert intra_group_swaps(orders, group_of=_exposure_group) == {0, 1}


def test_every_priced_ex_us_fund_shares_one_cap():
    """Ungrouped, two funds correlated 0.913-0.999 would each draw their own position cap, and a
    book could hold twice its intended international weight while believing it held two diversified
    positions. That is precisely how US_EQUITY came to exist (VTI 24.8% + SPY 21.5% = 46.3%)."""
    for sym in ("VXUS", "IXUS", "VEU", "VEA", "SCHF", "SPDW", "IEFA", "VWO"):
        assert _exposure_group(sym) == "INTL", sym


def test_plans_naming_the_old_vxus_group_still_resolve():
    """Standing plans name targets by GROUP. One written before this regrouping says "VXUS", and a
    label that stops resolving is how a plan silently became unreachable once before."""
    assert _exposure_group("VXUS") == _exposure_group("VEA") == "INTL"


def test_dividend_funds_are_priced_but_not_merged():
    """Measured against SCHD: DGRO 0.892, VYM 0.857, FDVV 0.783 — every one under the 0.90 bar this
    file uses. VYM is 2bp cheaper than SCHD and must NOT therefore look like a free swap; they are
    different indices doing different work, so the fee comparison does not apply between them."""
    groups = {s: _exposure_group(s) for s in ("SCHD", "VYM", "DGRO", "FDVV")}
    assert len(set(groups.values())) == 4, groups
    assert intra_group_swaps(
        [{"symbol": "SCHD", "side": "sell"}, {"symbol": "VYM", "side": "buy"}],
        group_of=_exposure_group) == set()


def test_us_equity_and_intl_stay_separate():
    """VXUS-VTI measured 0.773 on the same run that merged the ex-US funds. The split is deliberate
    and the new group must not have quietly swallowed it."""
    assert _exposure_group("VTI") == "US_EQUITY"
    assert _exposure_group("VXUS") == "INTL"


def test_no_mutual_fund_carries_a_ratio():
    """Yahoo returned 0.590% for FZILX and 1.760% for FZROX, both of which are 0.00% ZERO funds.
    An unverifiable number is worse than an absent one, and absent already means unknown here."""
    for sym in ("FZROX", "FZILX", "FNILX", "FTIHX", "FSKAX", "FSGGX"):
        assert _EXPENSE_RATIO_PCT.get(sym) is None, sym


def test_absent_still_means_unknown_not_free():
    """A single stock has no expense ratio at all, and must not read as the cheapest thing on offer."""
    assert _EXPENSE_RATIO_PCT.get("AMZN") is None
    from app.main import _expense_ratio
    assert _expense_ratio("AMZN") is None
    assert _expense_ratio("GLD") == 0.400
