"""SEC-1 — the Finnhub API key must not reach a log line.

`httpx` renders an HTTPStatusError as the whole request URL, and Finnhub takes its key as a query
parameter, so `log.warning("... (%s)", e)` on a failed lookup wrote the live credential into the
journal. Measured on the deployed container: 56 such lines on 2026-08-31, and 58 in a single second
on 2026-09-01.

These pin both layers — the call-site formatter that never builds the unsafe string, and the log
filter that catches what a call site forgot, including a traceback logged by code this repo does not
own.
"""

import logging

import httpx
import pytest

from app.redact import RedactingFilter, http_error, install_log_filter, redact

# A FABRICATED key of the same shape. The live one must never be committed — this file is in a
# public repo, and a test fixture is still a place a secret can leak from.
KEY = "FAKEKEYFAKEKEYFAKEKEYFAKEKEYFAKEKEYFAKE1"
URL = f"https://finnhub.io/api/v1/company-news?symbol=NVDA&from=2026-08-22&token={KEY}"


def _status_error(status: int = 429, url: str = URL) -> httpx.HTTPStatusError:
    req = httpx.Request("GET", url)
    try:
        httpx.Response(status, request=req).raise_for_status()
    except httpx.HTTPStatusError as e:
        return e
    raise AssertionError("expected raise_for_status to raise")


# --- the redactor ------------------------------------------------------------------------------


def test_the_exact_log_line_that_leaked_is_scrubbed():
    line = (f"news: NVDA headline context failed (HTTPStatusError: Client error '429 Too Many "
            f"Requests' for url '{URL}'")
    out = redact(line)
    assert KEY not in out
    assert "token=***" in out
    # The diagnosis survives: which symbol, which lookup, which status.
    assert "NVDA" in out and "429" in out


@pytest.mark.parametrize("param", ["token", "api_key", "api-key", "apikey", "crumb", "key",
                                   "password", "secret", "TOKEN", "Api_Key"])
def test_every_secret_bearing_parameter_name_is_covered(param):
    assert KEY not in redact(f"https://x/y?{param}={KEY}&symbol=X")


def test_a_value_ending_at_a_quote_or_ampersand_is_bounded_not_greedy():
    out = redact(f"url 'https://x/y?token={KEY}&from=2026-01-01' next")
    assert KEY not in out
    # Everything after the secret is kept — a redactor that ate the rest of the line would take the
    # diagnosis with it.
    assert "from=2026-01-01" in out and "next" in out


def test_a_line_with_no_secret_is_returned_unchanged():
    line = "news: AAPL earnings-calendar lookup failed (HTTP 503 finnhub.io/api/v1/calendar/earnings)"
    assert redact(line) == line


def test_non_string_input_does_not_raise():
    assert redact(None) == "None"
    assert KEY not in redact(_status_error())


# --- the call-site formatter -------------------------------------------------------------------


def test_http_error_reports_status_and_path_without_the_query_string():
    out = http_error(_status_error(429))
    assert out == "HTTP 429 finnhub.io/api/v1/company-news"
    assert KEY not in out and "symbol=NVDA" not in out


def test_http_error_on_a_failure_with_no_response_still_names_the_endpoint():
    req = httpx.Request("GET", URL)
    out = http_error(httpx.ReadTimeout("timed out", request=req))
    assert KEY not in out
    assert "ReadTimeout" in out and "finnhub.io/api/v1/company-news" in out


def test_http_error_on_a_bare_exception_is_redacted_rather_than_raising():
    out = http_error(RuntimeError(f"boom while calling ?token={KEY}"))
    assert KEY not in out
    assert "RuntimeError" in out


# --- the log filter ----------------------------------------------------------------------------


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(self.format(record))


@pytest.fixture()
def logger_with_filter():
    log = logging.getLogger("test.redact")
    log.propagate = False
    log.setLevel(logging.DEBUG)
    cap = _Capture()
    cap.setFormatter(logging.Formatter("%(message)s"))
    cap.addFilter(RedactingFilter())
    log.handlers = [cap]
    yield log, cap
    log.handlers = []


def test_the_filter_scrubs_a_lazily_formatted_message(logger_with_filter):
    """The %-args are not joined until getMessage(), so a filter reading only record.msg sees
    nothing and lets the key straight through."""
    log, cap = logger_with_filter
    log.warning("news: %s failed (%s)", "NVDA", _status_error())
    assert KEY not in cap.lines[0]
    assert "NVDA" in cap.lines[0]


def test_the_filter_scrubs_a_traceback(logger_with_filter):
    """macro.py logs its Finnhub failure with exc_info=True, and the traceback's last line is the
    same URL the message would have carried."""
    log, cap = logger_with_filter
    try:
        raise _status_error()
    except httpx.HTTPStatusError:
        log.warning("macro: general news fetch failed", exc_info=True)
    assert KEY not in cap.lines[0]
    assert "Traceback" in cap.lines[0]


def test_a_clean_line_survives_the_filter_intact(logger_with_filter):
    log, cap = logger_with_filter
    log.info("universe fresh, no rebuild needed (4612 symbols)")
    assert cap.lines[0] == "universe fresh, no rebuild needed (4612 symbols)"


def test_a_malformed_record_is_still_logged(logger_with_filter):
    """A filter that raises on a bad %-format would drop the line entirely — and swallowing a log
    line is how a failure becomes invisible."""
    log, cap = logger_with_filter
    log.warning("two placeholders %s %s", "only-one")
    assert cap.lines


def test_install_is_idempotent():
    h = logging.Handler()
    try:
        install_log_filter()
        logging.getLogger().addHandler(h)
        install_log_filter()
        install_log_filter()
        assert sum(isinstance(f, RedactingFilter) for f in h.filters) == 1
    finally:
        logging.getLogger().removeHandler(h)


def test_install_covers_last_resort_which_is_what_production_actually_uses():
    """Nothing in this service calls logging.basicConfig(), and uvicorn configures only its own
    loggers, so the root logger has no handler: WARNING and above reach logging.lastResort. A filter
    installed only on the root logger's (empty) handler list would redact nothing at all."""
    install_log_filter()
    assert any(isinstance(f, RedactingFilter) for f in logging.lastResort.filters)


# --- the HTTP error boundary ----------------------------------------------------------------------
#
# Found while reviewing the SEC-1 change rather than in the original report. Twenty-four routes in
# main.py answered a failed upstream with `HTTPException(detail=f"...: {e}")`, which SENDS the
# exception's text to the phone in a 502 body. The Finnhub key never reaches those (news lookups
# swallow their own failures), but the Yahoo path does: market_now.fetch_quotes raises
# `RuntimeError(f"Yahoo quote fetch failed: {last_err}")` with an httpx error whose message is the
# whole v7/finance/quote URL — and that URL carries the `crumb` session credential.


def test_the_yahoo_quote_failure_does_not_carry_the_crumb_into_its_message():
    import app.market_now as market_now

    crumb = "SeCrEtCrUmB123"
    req = httpx.Request("GET", f"https://query1.finance.yahoo.com/v7/finance/quote?symbols=AAPL&crumb={crumb}")

    async def blow_up(*a, **k):
        # Raised the way httpx raises it: the message is built from the request URL, which is the
        # whole reason the crumb ends up in it.
        httpx.Response(500, request=req).raise_for_status()

    class _Client:
        get = staticmethod(blow_up)

    async def fake_auth(client, force=False, stale=None):
        return crumb

    import asyncio

    import app.options as options
    real = options._ensure_auth
    options._ensure_auth = fake_auth
    try:
        with pytest.raises(RuntimeError) as ei:
            asyncio.run(market_now.fetch_quotes(_Client(), ["AAPL"]))
    finally:
        options._ensure_auth = real

    assert crumb not in str(ei.value)
    assert "crumb=***" in str(ei.value)


def test_every_error_body_in_main_redacts_the_exception_it_reports():
    """A grep-level invariant, so a route added later cannot quietly reintroduce the pattern."""
    import pathlib
    import re

    src = pathlib.Path(__file__).resolve().parents[1].joinpath("app", "main.py").read_text()
    raw = re.findall(r'detail=f"[^"]*\{e\}"', src)
    assert raw == [], f"these send a raw exception to the client: {raw}"
