"""MB-17 — discount vs deterioration.

The rule under test throughout: ABSENT EVIDENCE MUST NOT READ AS GOOD NEWS. A name with no
fundamentals has to come back "unclear" with `missing` populated and `assessable` False — never a
verdict that a reader would take as a clean bill of health.
"""
import pytest

from app.valuetrap import assess


def quality(**kw):
    base = {"fcf_trend": "rising", "fcf_positive_years": 5, "fcf_years": 5,
            "shares_change_pct": -2.0, "debt_to_equity": 0.3, "low_debt": True,
            "roe": 25.0, "high_roe": True}
    base.update(kw)
    return base


# ---------------------------------------------------------------- the honesty rule

def test_no_fundamentals_is_unclear_and_says_what_was_missing():
    a = assess(trend=None, quality=None, insider=None)
    assert a["verdict"] == "unclear"
    assert a["assessable"] is False
    assert "fundamentals" in a["missing"]
    assert a["green"] == [], "nothing may be presented as positive evidence when nothing was seen"


def test_a_thin_view_does_not_earn_a_verdict():
    # One weak positive is not enough to declare a discount.
    a = assess(trend=None, quality={"wide_moat": True}, insider=None)
    assert a["verdict"] == "unclear" and a["assessable"] is False


def test_missing_is_always_reported_even_on_a_confident_verdict():
    a = assess(trend={"direction": "recovering"}, quality=quality(), insider=None)
    assert a["verdict"] == "discount"
    assert "insider activity" in a["missing"], "an unseen input must be named even when the call is clear"


# ---------------------------------------------------------------- the two real verdicts

def test_a_deteriorating_business_is_called_deteriorating():
    a = assess(
        trend={"direction": "deepening"},
        quality=quality(fcf_trend="falling", fcf_positive_years=2, fcf_years=5,
                        shares_change_pct=8.0, debt_to_equity=3.5, low_debt=False,
                        roe=-12.0, high_roe=False, net_margin=-5.0),
        insider={"buy_count_12m": 0},
    )
    assert a["verdict"] == "deteriorating"
    assert a["confidence"] == "high"
    assert any("Free cash flow is falling" in x for x in a["red"])
    assert any("dilution" in x for x in a["red"])


def test_an_intact_business_that_is_merely_cheap_is_called_a_discount():
    a = assess(
        trend={"direction": "recovering"},
        quality=quality(wide_moat=True, dividend_aristocrat=True),
        insider={"buy_count_12m": 3, "has_conviction_buy": True, "has_cluster_buy": True},
    )
    assert a["verdict"] == "discount"
    assert a["confidence"] == "high"


def test_mixed_evidence_stays_unclear_rather_than_picking_a_side():
    # Genuinely balanced: falling cash flow AND dilution AND no insider buying (red) against low
    # debt, high ROE and a recovering price (green). Neither side clears the 1.5x margin.
    a = assess(
        trend={"direction": "recovering"},
        quality={"fcf_trend": "falling", "shares_change_pct": 8.0,
                 "debt_to_equity": 0.3, "low_debt": True, "roe": 25.0, "high_roe": True},
        insider={"buy_count_12m": 0},
    )
    assert a["verdict"] == "unclear", a["weights"]
    assert a["assessable"] is True, "genuinely balanced is different from not enough data"
    assert a["red"] and a["green"]


# ---------------------------------------------------------------- units, which have bitten before

def test_debt_to_equity_is_read_as_a_ratio_not_a_percent():
    # low_debt is <0.5. Treating 0.3 as "0.3%" or 3.5 as "3.5%" would inverse both readings.
    lev = assess(None, quality(debt_to_equity=3.5, low_debt=False), None)
    assert any("heavily levered" in x for x in lev["red"])
    safe = assess(None, quality(debt_to_equity=0.3, low_debt=True), None)
    assert not any("levered" in x for x in safe["red"])


def test_negative_returns_and_margins_count_against():
    a = assess(None, quality(roe=-8.0, high_roe=False, net_margin=-3.0), None)
    assert any("Return on equity is negative" in x for x in a["red"])
    assert any("Net margin is negative" in x for x in a["red"])


@pytest.mark.parametrize("pct,expect_red", [(8.0, True), (2.5, True), (0.0, False), (-3.0, False)])
def test_dilution_counts_against_and_buybacks_count_for(pct, expect_red):
    a = assess(None, quality(shares_change_pct=pct), None)
    assert any("dilution" in x for x in a["red"]) is expect_red
    if pct <= -1.0:
        assert any("buybacks" in x for x in a["green"])


def test_the_note_states_what_an_unclear_verdict_means():
    a = assess(None, None, None)
    assert "not that the business looks fine" in a["note"]


def test_a_stock_split_is_not_counted_as_dilution():
    """SMCI's 10-for-1 read as '+1074% — dilution': the worst available red flag, for a corporate
    action that dilutes nobody. Raw reported share counts cannot distinguish the two, so when
    fundamentals marks the number unreliable it must not become evidence either way."""
    a = assess(None, quality(shares_change_pct=1074.4, shares_change_reliable=False), None)
    assert not any("dilution" in x for x in a["red"]), a["red"]
    assert any("split" in x for x in a["missing"])


def test_a_reliable_share_count_still_counts():
    a = assess(None, quality(shares_change_pct=8.0, shares_change_reliable=True), None)
    assert any("dilution" in x for x in a["red"])
    b = assess(None, quality(shares_change_pct=-5.0, shares_change_reliable=True), None)
    assert any("buybacks" in x for x in b["green"])


def test_confidence_requires_BREADTH_of_evidence_not_just_weight():
    """The ratio tests are multiplicative, so with zero red evidence both are trivially satisfied.

    A name with four of six inputs unseen and no negatives came back "discount, high confidence" —
    exactly what this module's docstring promises it will never do.
    """
    thin = assess(None, {"roe": 25.0, "high_roe": True, "wide_moat": True, "buffett_quality": True,
                         "dividend_aristocrat": True, "debt_to_equity": 0.3, "low_debt": True}, None)
    assert thin["confidence"] != "high", thin
    assert thin["assessable"] is False, "one observed category is not a basis for a verdict"
    assert thin["evidence_categories_seen"] == 1

    broad = assess(
        {"direction": "recovering"},
        {"roe": 25.0, "high_roe": True, "wide_moat": True, "debt_to_equity": 0.3, "low_debt": True,
         "fcf_trend": "rising", "fcf_positive_years": 5, "fcf_years": 5,
         "shares_change_pct": -2.0, "shares_change_reliable": True},
        {"buy_count_12m": 3, "has_conviction_buy": True},
    )
    assert broad["verdict"] == "discount" and broad["confidence"] == "high"
    assert broad["evidence_categories_seen"] == 5


def test_a_dead_quality_feed_reports_the_metrics_it_could_not_read():
    """ROE and net margin were read inside `if quality:` without ever recording their absence, so a
    partial quality feed under-reported what could not be checked."""
    a = assess({"direction": "recovering"},
               {"fcf_trend": "rising", "fcf_positive_years": 5, "fcf_years": 5,
                "shares_change_pct": -2.0, "shares_change_reliable": True, "shares_years": 5,
                "debt_to_equity": 0.3, "low_debt": True},
               {"buy_count_12m": 1})
    assert "return on equity" in a["missing"]
    assert "net margin" in a["missing"]


def test_the_dilution_threshold_is_per_year_not_per_window():
    """_DILUTION_PCT is documented "y/y" but shares_change_pct spans `shares_years` reports, so
    2.5% over five years — half a percent a year — was flagged as dilution."""
    trivial = assess(None, {"shares_change_pct": 2.5, "shares_change_reliable": True,
                            "shares_years": 5}, None)
    assert not any("dilution" in x for x in trivial["red"]), trivial["red"]

    real = assess(None, {"shares_change_pct": 15.0, "shares_change_reliable": True,
                         "shares_years": 5}, None)
    assert any("dilution" in x for x in real["red"])

    # A single-year window must still behave as before.
    one_year = assess(None, {"shares_change_pct": 2.5, "shares_change_reliable": True,
                             "shares_years": 1}, None)
    assert any("dilution" in x for x in one_year["red"])
