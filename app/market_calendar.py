"""
US equity-market (NYSE/Nasdaq) trading-calendar helpers — computed, no heavy dependency.

`market_now.session_phase()` treats a holiday as its normal weekday phase (a documented gap), which is
fine for a "what's the tape doing" read (quotes come back flat) but WRONG for the sandbox trader, which
must not place equity fills on a full market closure. Rather than pull in `pandas_market_calendars`
(which drags in pandas) for ~9 fixed closures a year on rules that rarely change, this computes the
NYSE full-closure set per year (with the Saturday→Friday / Sunday→Monday observed shift) plus the
early-close (1pm ET) days. Only `is_trading_day()` is needed by the gate; early closes are harmless at
15:35 ET (the market is still "open" then).
"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache


def _easter(year: int) -> dt.date:
    """Gregorian Easter Sunday (Anonymous/Meeus algorithm). Good Friday = Easter − 2 days."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    lm = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lm) // 451
    month = (h + lm - 7 * m + 114) // 31
    day = ((h + lm - 7 * m + 114) % 31) + 1
    return dt.date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    """The n-th `weekday` (Mon=0…Sun=6) of `month` (1-indexed n)."""
    first = dt.date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + dt.timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    """The last `weekday` of `month`."""
    if month == 12:
        nxt = dt.date(year + 1, 1, 1)
    else:
        nxt = dt.date(year, month + 1, 1)
    last = nxt - dt.timedelta(days=1)
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)


def _observed(d: dt.date) -> dt.date:
    """NYSE fixed-date-holiday observance: Saturday → prior Friday, Sunday → following Monday."""
    if d.weekday() == 5:      # Saturday
        return d - dt.timedelta(days=1)
    if d.weekday() == 6:      # Sunday
        return d + dt.timedelta(days=1)
    return d


@lru_cache(maxsize=32)
def _holidays(year: int) -> frozenset[dt.date]:
    """The set of full NYSE closures for `year` (observed dates)."""
    days = {
        _observed(dt.date(year, 1, 1)),            # New Year's Day
        _nth_weekday(year, 1, 0, 3),               # MLK Jr. Day — 3rd Monday of January
        _nth_weekday(year, 2, 0, 3),               # Washington's Birthday — 3rd Monday of February
        _easter(year) - dt.timedelta(days=2),      # Good Friday
        _last_weekday(year, 5, 0),                 # Memorial Day — last Monday of May
        _nth_weekday(year, 9, 0, 1),               # Labor Day — 1st Monday of September
        _nth_weekday(year, 11, 3, 4),              # Thanksgiving — 4th Thursday of November
        _observed(dt.date(year, 12, 25)),          # Christmas Day
        _observed(dt.date(year, 7, 4)),            # Independence Day
    }
    # Juneteenth is a market holiday from 2022 on.
    if year >= 2022:
        days.add(_observed(dt.date(year, 6, 19)))
    return frozenset(days)


def is_market_holiday(d: dt.date) -> bool:
    """True if `d` is a full NYSE closure (ignores weekends — use is_trading_day for the full check)."""
    return d in _holidays(d.year)


def is_trading_day(d: dt.date) -> bool:
    """True if the US equity market has a regular session on `d` (weekday and not a full holiday)."""
    return d.weekday() < 5 and not is_market_holiday(d)
