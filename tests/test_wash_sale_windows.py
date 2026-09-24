"""The open wash-sale windows handed to the tick prompt (2026-09-24: the analyst justified a loss
sale as one that "resets wash-sale lockout", a rule it had no way to see)."""
from app.sandbox_job import wash_sale_windows

DAY = 86_400.0
NOW = 1_790_000_000.0


def test_lists_open_windows_with_days_left_soonest_first():
    got = wash_sale_windows({"schd": NOW - 0.2 * DAY, "XOM": NOW - 27.5 * DAY}, now_ts=NOW)
    assert got == [{"symbol": "XOM", "days_left": 3}, {"symbol": "SCHD", "days_left": 30}]


def test_a_window_past_30_days_is_closed():
    assert wash_sale_windows({"GLD": NOW - 30.01 * DAY}, now_ts=NOW) == []


def test_guard_off_means_no_windows():
    # The ledger will not enforce them, so they are not a constraint the model should plan around.
    assert wash_sale_windows({"SCHD": NOW - DAY}, now_ts=NOW, enabled=False) == []


def test_absent_or_malformed_ledger_is_empty_not_an_error():
    assert wash_sale_windows(None, now_ts=NOW) == []
    assert wash_sale_windows({"BAD": "x", "OK": NOW - DAY}, now_ts=NOW) == [{"symbol": "OK", "days_left": 29}]


def test_matches_the_guard_that_enforces_it():
    # Same arithmetic as validate_and_fill's skip message: "(30 - int(days))d left".
    days = 11.7
    assert wash_sale_windows({"A": NOW - days * DAY}, now_ts=NOW)[0]["days_left"] == 30 - int(days)
