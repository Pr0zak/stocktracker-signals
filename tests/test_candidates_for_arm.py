"""Every comparison arm is offered the same universe; only its own book differs.

The daily tick builds ONE candidate pool and hands it to every arm, which is what makes the arms
comparable -- same quotes, same tick, same day. But the pool is assembled against MAIN's book, and
`select_candidates` strips held names from it on purpose: a held name already reaches the model
through the positions block, with its own price and technicals, so listing it twice wastes a slot
and reads as two separate opportunities.

Correct for main, silently wrong for everyone else. An arm that does not hold what main holds was
given no row and no price for those names anywhere in its input, while `target_gaps` went on telling
it about the allocation gap it was supposed to close. Measured on the live ledgers before this was
fixed: the `reviewed` arm carried an 11% GOLD target against 0% actual for three consecutive
sessions, GLD stripped from its pool every day because MAIN holds GLD. Asked to fill a gap in an
instrument it had never been shown a price for, the model wrote the price from memory -- GLD entry
zones of $170-180, $195-210 and $195-202 against a market of $421-427, and an FBTC zone of $355-365
against $68.31. The ledger's zone guard refused all four, so no money moved, but the allocation was
never filled either and the trade log recorded a badly-priced order instead of the missing input
behind it.

The bias mattered beyond the refused orders: the further an arm's book drifted from main's, the more
of the universe vanished from its prompt, so an arm differed from main by what it could SEE as well
as by the one setting under test.
"""
from __future__ import annotations

from app.sandbox_job import candidates_for_arm


def row(sym: str, price: float = 100.0) -> dict:
    return {"symbol": sym, "price": price, "technicals": {"rsi_14": 50}}


SHARED = [row("VOO"), row("QQQM"), row("VEA")]
HELD_BY_MAIN = [row("GLD", 421.70), row("FBTC", 68.31), row("VTI")]


def syms(rows: list[dict]) -> list[str]:
    return [r["symbol"] for r in rows]


def test_arm_sees_what_main_holds_and_it_does_not():
    """The regression. An arm with no gold gets a GLD row, with the real price attached."""
    out = candidates_for_arm(SHARED, HELD_BY_MAIN, held=["VOO"], exclusions=[])
    assert "GLD" in syms(out)
    assert next(r for r in out if r["symbol"] == "GLD")["price"] == 421.70
    assert "FBTC" in syms(out)


def test_an_arm_never_sees_its_own_holdings_twice():
    """The property the original strip existed to protect, now applied per arm rather than to
    main's book only. A held name arrives through the positions block; a second row for it is a
    duplicate, not a second opportunity."""
    out = candidates_for_arm(SHARED, HELD_BY_MAIN, held=["VOO", "GLD"], exclusions=[])
    assert "VOO" not in syms(out)
    assert "GLD" not in syms(out)
    assert "QQQM" in syms(out)
    assert "FBTC" in syms(out)


def test_two_arms_holding_different_books_get_the_same_universe():
    """The invariant the comparison depends on: universe identical, books different. Anything else
    means an arm differs from main by what it can see as well as by the setting under test."""
    a = candidates_for_arm(SHARED, HELD_BY_MAIN, held=["VOO"], exclusions=[])
    b = candidates_for_arm(SHARED, HELD_BY_MAIN, held=["GLD"], exclusions=[])
    assert set(syms(a)) | {"VOO"} == set(syms(b)) | {"GLD"}


def test_the_arms_own_exclusions_are_honoured():
    out = candidates_for_arm(SHARED, HELD_BY_MAIN, held=[], exclusions=["fbtc", "QQQM"])
    assert "FBTC" not in syms(out)
    assert "QQQM" not in syms(out)
    assert "GLD" in syms(out)


def test_shared_pool_wins_a_duplicate_and_order_is_preserved():
    """An arm's list stays the pool it would have had, with the missing names appended — so a change
    here cannot reorder what the model ranks."""
    dupe = [row("VOO", 999.0)]
    out = candidates_for_arm(SHARED, dupe + HELD_BY_MAIN, held=[], exclusions=[])
    assert syms(out) == ["VOO", "QQQM", "VEA", "GLD", "FBTC", "VTI"]
    assert next(r for r in out if r["symbol"] == "VOO")["price"] == 100.0


def test_case_and_blanks_do_not_leak_a_held_name_back_in():
    out = candidates_for_arm(
        SHARED + [{"symbol": ""}, {"price": 1.0}], HELD_BY_MAIN,
        held=["voo"], exclusions=["gld"])
    assert "VOO" not in syms(out)
    assert "GLD" not in syms(out)
    assert "" not in syms(out)


def test_no_extra_rows_is_todays_behaviour_unchanged():
    """With nothing to add back, an arm holding nothing sees exactly main's pool. Keeps the change
    provably inert on the path where main holds none of the universe."""
    assert candidates_for_arm(SHARED, [], held=[], exclusions=[]) == SHARED
