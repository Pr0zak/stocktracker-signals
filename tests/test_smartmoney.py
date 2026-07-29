"""Theme C — follow the smart money.

Two rules under test throughout:
  * An insider buy outranks a congressional one. Insiders disclose within two business days; the
    STOCK Act allows ~45, and the amounts are ranges.
  * A feed that could not be READ is never the same as a feed that came back EMPTY.
"""
import asyncio

import pytest

from app import smartmoney as sm


def ins(cluster=False, conviction=False, n=0):
    return {"has_cluster_buy": cluster, "has_conviction_buy": conviction, "buy_count_12m": n}


def con(cluster=False, net="neutral", buys=0, largest=0, filed="2026-06-01", newest="2026-05-15"):
    return {"cluster_buy": cluster, "net_direction": net, "buy_count": buys,
            "largest_buy_amount_high": largest, "latest_filing_date": filed,
            "newest_trade_date": newest}


# ---------------------------------------------------------------- the weighting

def test_an_insider_cluster_outranks_a_congressional_cluster():
    """Insiders disclose in two business days and spend their own money; congressional filings lag
    up to ~45 days and report ranges. The ordering has to reflect that."""
    a = sm.score_symbol("A", ins(cluster=True), None)
    b = sm.score_symbol("B", None, con(cluster=True))
    assert a["score"] > b["score"]


def test_insider_selling_is_not_scored_at_all():
    # Insiders sell for tax, diversification and houses. Only buying is informative.
    quiet = sm.score_symbol("X", ins(n=0), None)
    assert quiet["score"] == 0.0 and quiet["reasons"] == []


def test_evidence_stacks_but_each_reason_is_named():
    r = sm.score_symbol("Z", ins(cluster=True, conviction=True), con(cluster=True, largest=250_000))
    assert r["score"] == pytest.approx(3.0 + 2.5 + 1.5 + 0.75)
    assert len(r["reasons"]) == 4


def test_plain_buying_scores_but_less_than_a_cluster():
    plain = sm.score_symbol("P", ins(n=2), None)["score"]
    cluster = sm.score_symbol("C", ins(cluster=True), None)["score"]
    assert 0 < plain < cluster


# ---------------------------------------------------------------- the honesty rule

def test_an_unreadable_feed_is_not_reported_as_no_buying():
    """"No insider buying" and "no Finnhub key" must never render the same."""
    empty = sm.score_symbol("A", None, con(), insider_available=True)
    blind = sm.score_symbol("A", None, con(), insider_available=False)
    assert empty["score"] == blind["score"] == 0.0     # same number...
    assert "insider" in empty["sources_seen"] and empty["unavailable"] == []
    assert "insider" in blind["unavailable"] and "insider" not in blind["sources_seen"]


def test_the_congressional_lag_is_carried_on_the_row():
    r = sm.score_symbol("A", None, con(net="buying", buys=2, filed="2026-06-20",
                                       newest="2026-05-02"))
    assert r["congress_latest_filing"] == "2026-06-20"
    assert r["congress_newest_trade"] == "2026-05-02", "the card must be able to age the evidence"


# ---------------------------------------------------------------- ranking + sweep

def test_only_names_with_evidence_are_ranked_and_order_is_stable():
    rows = [sm.score_symbol("B", ins(cluster=True), None),
            sm.score_symbol("A", ins(cluster=True), None),
            sm.score_symbol("Q", None, None)]
    out = sm.rank(rows, limit=10)
    assert [r["symbol"] for r in out] == ["A", "B"], "ties break on symbol; no-evidence excluded"


def test_a_dead_feed_for_one_symbol_does_not_sink_the_sweep():
    async def insider_of(sym):
        if sym == "BOOM":
            raise RuntimeError("finnhub down")
        return ins(cluster=True)

    async def congress_of(sym):
        if sym == "BOOM":
            raise RuntimeError("kadoa down")
        return None

    ranked, none_found, failed = asyncio.run(
        sm.sweep(["OK1", "BOOM", "OK2"], insider_of, congress_of))
    assert {r["symbol"] for r in ranked} == {"OK1", "OK2"}
    assert failed == ["BOOM"] and none_found == []


def test_one_surviving_feed_still_produces_a_row_flagged_partial():
    async def insider_of(sym):
        raise RuntimeError("finnhub down")

    async def congress_of(sym):
        return con(cluster=True)

    ranked, none_found, failed = asyncio.run(sm.sweep(["A"], insider_of, congress_of))
    assert failed == []
    assert ranked[0]["unavailable"] == ["insider"], "a partial read must say which half is missing"


def test_symbols_with_no_evidence_are_reported_separately_from_failures():
    async def insider_of(sym):
        return ins(n=0)

    async def congress_of(sym):
        return None

    ranked, none_found, failed = asyncio.run(sm.sweep(["A", "B"], insider_of, congress_of))
    assert ranked == [] and failed == []
    assert sorted(none_found) == ["A", "B"]
