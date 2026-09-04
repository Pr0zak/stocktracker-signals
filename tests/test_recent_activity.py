"""The sandbox tick's own ledger, handed back to it.

The tick was a COLD START. `sandbox_decision`'s payload was equity, cash, positions, candidates,
settings, the weekly strategy note and the allocation gaps — a snapshot of the book plus today's
names, and nothing about what this account had done. A position that was SOLD leaves no trace in
that payload except a smaller share count, which is indistinguishable from one never held.

The two behaviours that exposed it, both measured on the live paper account:

  * 2026-09-02 the analyst trimmed 8 SCHD for being extended at a weekly RSI of 76.5. On 2026-09-04
    it bought 10 back, in a reason that itself called the name overbought at 73.4, citing "largest
    allocation gap" — the gap its own sale had created two days earlier. Both decisions were locally
    sound on the information given; the contradiction was only visible from outside.
  * The GLD->GLDM swap was proposed on three consecutive days and refused every time by the same
    guard. Nothing told the model its order had been rejected, so nothing stopped it re-proposing.

So skipped rows are carried too, with the reason the server recorded — for the second case they
matter more than the fills do.
"""

from app.sandbox_job import _HISTORY_MAX, _REASON_CHARS, recent_activity

TODAY = "2026-09-04"


def _t(date, symbol, side, status="filled", shares=1.0, price=10.0, reason="", skip=""):
    row = {"date": date, "symbol": symbol, "side": side, "status": status,
           "shares": shares, "price": price, "reason": reason}
    if skip:
        row["skip_reason"] = skip
    return row


# --- the round trip it could not see -------------------------------------------------------------


def test_a_sale_days_ago_is_visible_where_the_book_alone_would_hide_it():
    trades = [
        _t("2026-09-04", "SCHD", "buy", shares=10.0, price=34.8224, reason="Largest allocation gap"),
        _t("2026-09-02", "SCHD", "sell", shares=8.0, price=35.0275, reason="Weekly RSI 76.5 — trim into strength"),
    ]
    out = recent_activity(trades, today=TODAY)
    sells = [r for r in out if r["side"] == "sell"]
    assert sells, "the sale is the fact the positions block cannot carry"
    assert sells[0]["symbol"] == "SCHD"
    assert sells[0]["date"] == "2026-09-02"
    assert "76.5" in sells[0]["your_reason"]


def test_a_refused_order_says_it_was_refused_and_why():
    trades = [_t("2026-09-03", "GLDM", "buy", status="skipped", reason="cheaper twin",
                 skip="same-exposure swap within 'GOLD' — sells and re-buys the same thing")]
    out = recent_activity(trades, today=TODAY)
    assert out[0]["status"] == "skipped"
    assert "same-exposure swap" in out[0]["skipped_because"]
    # A skipped order has no fill, so quoting shares or a price for it would be a fiction.
    assert "shares" not in out[0] and "price" not in out[0]


def test_a_filled_order_carries_what_actually_happened():
    out = recent_activity([_t("2026-09-04", "SCHD", "buy", shares=10.0, price=34.8224)], today=TODAY)
    assert out[0]["shares"] == 10.0 and out[0]["price"] == 34.8224
    assert "skipped_because" not in out[0]


# --- what is deliberately excluded ---------------------------------------------------------------


def test_cash_movements_are_not_decisions():
    trades = [
        {"date": "2026-09-04", "symbol": "CASH", "side": "interest", "status": "filled", "gross": 0.12},
        {"date": "2026-09-01", "symbol": "CASH", "side": "deposit", "status": "filled", "gross": 500.0},
        _t("2026-09-04", "SCHD", "buy"),
    ]
    out = recent_activity(trades, today=TODAY)
    assert [r["symbol"] for r in out] == ["SCHD"]


def test_anything_older_than_the_window_is_dropped():
    trades = [_t("2026-09-04", "AAA", "buy"), _t("2026-08-01", "OLD", "buy")]
    syms = [r["symbol"] for r in recent_activity(trades, today=TODAY, days=14)]
    assert syms == ["AAA"]


def test_the_window_boundary_is_inclusive_of_the_cutoff_day():
    trades = [_t("2026-08-21", "EDGE", "buy"), _t("2026-08-20", "PAST", "buy")]
    syms = [r["symbol"] for r in recent_activity(trades, today=TODAY, days=14)]
    assert syms == ["EDGE"]


def test_the_list_is_capped_so_the_prompt_cannot_grow_without_bound():
    trades = [_t("2026-09-04", f"S{i}", "buy") for i in range(60)]
    assert len(recent_activity(trades, today=TODAY)) == _HISTORY_MAX


def test_reasons_are_truncated_rather_than_quoted_in_full():
    """The goal is to remind the model what it did, not hand it back a persuasive paragraph it wrote
    about itself. Consistency with a past call is not the same as being right."""
    long = "x" * 400
    out = recent_activity([_t("2026-09-04", "AAA", "buy", reason=long)], today=TODAY)
    assert len(out[0]["your_reason"]) == _REASON_CHARS


# --- absence, not substitution -------------------------------------------------------------------


def test_no_trades_yields_an_empty_list_which_the_caller_then_omits():
    """`sandbox_decision` sends the key only when non-empty: "you have traded nothing recently" and
    "no history was supplied" are different claims, and only the first can be true here."""
    assert recent_activity([], today=TODAY) == []


def test_an_unparseable_today_admits_history_rather_than_silently_erasing_it():
    """A bad date must not make the model believe it has never traded."""
    out = recent_activity([_t("2026-09-04", "AAA", "buy")], today="not-a-date")
    assert [r["symbol"] for r in out] == ["AAA"]


def test_rows_with_no_date_are_skipped_rather_than_dated_today():
    trades = [{"symbol": "AAA", "side": "buy", "status": "filled", "shares": 1.0, "price": 1.0}]
    assert recent_activity(trades, today=TODAY) == []


def test_a_missing_reason_leaves_the_key_out_instead_of_writing_an_empty_string():
    out = recent_activity([_t("2026-09-04", "AAA", "buy", reason="")], today=TODAY)
    assert "your_reason" not in out[0]


def test_newest_first_is_preserved_so_the_model_reads_the_recent_end_first():
    trades = [_t("2026-09-04", "NEW", "buy"), _t("2026-09-03", "MID", "buy"), _t("2026-09-02", "OLD", "buy")]
    assert [r["symbol"] for r in recent_activity(trades, today=TODAY)] == ["NEW", "MID", "OLD"]
