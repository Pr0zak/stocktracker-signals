"""One place that scrubs secrets out of anything on its way to a log line.

SEC-1. `httpx` renders an HTTPStatusError as the entire request URL, and Finnhub takes its API key
as a QUERY PARAMETER, so every `log.warning(..., e)` on a failed news or earnings lookup wrote the
live key into the journal in the clear. Measured on the deployed container: 56 such lines on
2026-08-31, and 58 more inside a single second on 2026-09-01. Anyone who can read the journal — or a
log export, a support bundle, a screenshot of a terminal — has the key.

Two layers, because either one alone fails open:

  * `http_error()` at each call site, which never builds the unsafe string in the first place. It
    reports the status and the PATH and drops the query string entirely, so the useful diagnosis
    ("HTTP 429 finnhub.io/api/v1/calendar/earnings") survives while the secret does not.

  * `install_log_filter()`, which attaches a redacting filter to the handlers records actually reach,
    so a line emitted by code this repo does not own is scrubbed too. `macro.py` logs its failures
    with `exc_info=True`, and a traceback's final line carries the same URL that the message does —
    call-site discipline would never have caught that one, because the call site never formats the
    exception at all.

Rotating the key remains a separate, manual step. Redaction stops the next leak; it does not undo
the ones already sitting in the journal, in `journalctl --vacuum`-able archives, or in any backup
taken since this path was written.
"""
from __future__ import annotations

import logging
import re

# Deliberately broad: the cost of scrubbing something harmless in a log line is nil, and the cost of
# missing one is a live credential in plaintext. Matches `token=...`, `?api_key=...`, `&crumb=...`
# and friends, stopping at whatever delimiter ends the value.
_SECRET_RE = re.compile(r"(?i)\b(token|crumb|api[_-]?key|apikey|key|password|secret)=[^\s&'\"]+")


def redact(s: object) -> str:
    """`s` rendered as a string with any secret-bearing query parameter replaced by `***`."""
    return _SECRET_RE.sub(r"\1=***", str(s))


def http_error(e: BaseException) -> str:
    """A short, secret-free description of a failed HTTP call.

    Built from the response's status and the request's host and PATH — never `str(e)`, and never the
    URL's query string, which is where the key lives. Falls back to the exception type plus its
    redacted message for failures that carry no response (timeouts, DNS, connection resets), so a
    caller still learns which kind of failure it was.
    """
    resp = getattr(e, "response", None)
    req = getattr(e, "request", None) or getattr(resp, "request", None)
    url = getattr(req, "url", None)
    where = ""
    if url is not None:
        # Attribute access, not str(url): str() would put the query string back.
        where = f"{getattr(url, 'host', '') or ''}{getattr(url, 'path', '') or ''}".strip()
    status = getattr(resp, "status_code", None)
    if status is not None:
        return f"HTTP {status} {where}".strip()
    detail = redact(e).strip()
    head = f"{type(e).__name__} {where}".strip()
    return f"{head}: {detail}" if detail else head


class RedactingFilter(logging.Filter):
    """A logging filter that rewrites every record's message, and its traceback, through `redact()`.

    Attached to handlers rather than loggers on purpose: a filter on a Logger is consulted only for
    records logged directly through THAT logger, and every warning here is emitted on a child logger
    (`signals.news`) that merely propagates upward. Handler filters see everything that reaches them.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        broken = False
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001 — a mis-formatted log call must still produce a line
            # Without this the SAME exception is raised again inside the handler's format(), where
            # logging routes it to handleError and the line is dropped. Losing a warning entirely is
            # worse than printing it awkwardly, so the args are rendered as-is and the record is
            # left in a state nothing downstream can trip over.
            broken = True
            msg = f"{record.msg} [unformattable log args: {record.args!r}]"
        safe = redact(msg)
        if broken or safe != msg:
            record.msg = safe
            record.args = ()
        if record.exc_info and not record.exc_text:
            # Render the traceback here, redacted, so the formatter finds `exc_text` already set and
            # does not re-render the raw one. `exc_info` is left in place: other handlers may want it,
            # and each of them runs this filter too.
            try:
                record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
            except Exception:  # noqa: BLE001
                record.exc_text = "(traceback suppressed)"
        return True


def install_log_filter() -> int:
    """Attach `RedactingFilter` to every handler records can reach. Idempotent; returns how many it
    newly covered.

    `logging.lastResort` is in the list because it is the handler that actually carries these lines
    in production: nothing in this service calls `logging.basicConfig()`, and uvicorn configures only
    its own `uvicorn.*` loggers, so the root logger has no handler and WARNING-and-above falls
    through to lastResort. A filter installed only on the root logger's (empty) handler list would
    have redacted nothing at all.
    """
    targets: list[logging.Handler] = []
    if logging.lastResort is not None:
        targets.append(logging.lastResort)
    targets.extend(logging.getLogger().handlers)
    for name in list(logging.root.manager.loggerDict):
        obj = logging.root.manager.loggerDict.get(name)
        targets.extend(getattr(obj, "handlers", []) or [])

    added = 0
    for h in targets:
        if not any(isinstance(f, RedactingFilter) for f in h.filters):
            h.addFilter(RedactingFilter())
            added += 1
    return added
