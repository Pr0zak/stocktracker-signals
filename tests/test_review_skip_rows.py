"""A rejected order has to leave a trace in the trade log.

Measured 2026-08-19, the first live rejection: the review model refused an XOM buy on the `reviewed`
arm. apply_review removed it before validate_and_fill ever saw it, so nothing wrote a row -- the only
evidence a rejection had happened at all was the control arm buying the same thing. In the log, an
order the reviewer killed was indistinguishable from one the analyst never proposed, and those are
very different facts about a day's decision.

The row carries BOTH sides. `reason` stays the analyst's case for the trade; `skip_reason` is the
reviewer's case against it. The disagreement should be readable from one line rather than
reconstructible only by diffing two arms.
"""
from __future__ import annotations

from app.sandbox_job import review_skip_rows

ORDER = {"symbol": "xom", "side": "buy", "conviction": 72,
         "reason": "Energy target +Hormuz catalyst", "entry_low": 160.0, "entry_high": 168.0}


def test_the_row_keeps_the_analysts_reason_and_adds_the_reviewers():
    r = review_skip_rows([ORDER], {"approve": True, "drop_symbols": ["XOM"],
                                   "note": "Entry is extended",
                                   "concerns": ["Chasing a 7% oil spike at RSI 72"]},
                         now_ts=1.0)[0]
    assert r["reason"] == "Energy target +Hormuz catalyst"          # the case FOR
    assert "Entry is extended" in r["skip_reason"]                  # the case AGAINST
    assert "RSI 72" in r["skip_reason"]
    assert r["status"] == "skipped" and r["source"] == "review"
    assert r["symbol"] == "XOM"                                     # normalised like every other row
    assert r["shares"] == 0.0 and r["price"] is None


def test_a_blanket_rejection_says_so_rather_than_reading_as_one_dropped_order():
    r = review_skip_rows([ORDER], {"approve": False, "concerns": ["The whole plan is stale"]},
                         now_ts=1.0)[0]
    assert "rejected the whole decision" in r["skip_reason"]


def test_the_note_leads_and_the_concerns_follow():
    # The note is the reviewer's own summary; the concerns are what a reader needs in order to
    # disagree with it. Order matters because the log is scanned, not read.
    r = review_skip_rows([ORDER], {"approve": True, "note": "Summary", "concerns": ["Detail"]},
                         now_ts=1.0)[0]
    assert r["skip_reason"].index("Summary") < r["skip_reason"].index("Detail")


def test_a_silent_verdict_still_produces_a_readable_row():
    # A reviewer that drops an order and explains nothing must not yield a row whose reason is blank
    # -- that reads as a bug in the log rather than as a terse reviewer.
    r = review_skip_rows([ORDER], {"approve": True, "drop_symbols": ["XOM"]}, now_ts=1.0)[0]
    assert "no reason given" in r["skip_reason"]


def test_no_dropped_orders_means_no_rows():
    assert review_skip_rows([], {"approve": False, "concerns": ["x"]}, now_ts=1.0) == []
