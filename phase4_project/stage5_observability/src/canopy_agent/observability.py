"""Structured, scrubbed observability for the canopy agent.

Every significant event becomes one JSON log line — machine-parseable,
shippable to any log backend. Secrets are scrubbed BEFORE writing, so a
leaked log can never leak a key."""

import json
import logging
import re
import sys
import time
import uuid
from contextvars import ContextVar

# --- correlation id: ties all log lines from ONE query together ---
# set at the start of a request, read by every log call during it
_trace_id: ContextVar[str] = ContextVar("trace_id", default="")


# --- what to scrub ---
# keys whose VALUES are sensitive, and regex patterns for secrets in free text
_SENSITIVE_KEYS = {
    "api_key",
    "x-api-key",
    "authorization",
    "token",
    "password",
    "canopy_api_key",
    "gemini_api_key",
    "secret",
}

# a long hex string (like your API key) or Bearer token in any text
_SECRET_PATTERNS = [
    re.compile(r"\b[a-f0-9]{32,}\b", re.IGNORECASE),  # long hex (API keys)
    re.compile(r"Bearer\s+\S+", re.IGNORECASE),  # bearer tokens
]

_REDACTED = "***REDACTED***"


def _scrub(obj):
    """Recursively remove secrets from a value before logging.

    - dict keys matching sensitive names → value redacted
    - strings matching secret patterns → redacted
    - recurses into nested dicts/lists
    """
    if isinstance(obj, dict):
        return {
            k: (_REDACTED if k.lower() in _SENSITIVE_KEYS else _scrub(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    if isinstance(obj, str):
        s = obj
        for pat in _SECRET_PATTERNS:
            s = pat.sub(_REDACTED, s)
        return s
    return obj


class JsonFormatter(logging.Formatter):
    """Render each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "event": record.getMessage(),
            "trace_id": _trace_id.get(),
        }
        # structured fields passed via extra={"data": {...}}
        data = getattr(record, "data", None)
        if data is not None:
            payload["data"] = _scrub(data)  # ← scrub before writing
        return json.dumps(payload)


def get_logger() -> logging.Logger:
    log = logging.getLogger("canopy")
    if not log.handlers:
        log.setLevel(logging.INFO)
        # StreamHandler should use sys.stderr instead of sys.stdout
        # MCP server communicates over stdout (the protocol channel).
        # If logs went to stdout, they'd corrupt the MCP messages and break the protocol.
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        log.addHandler(handler)
    return log


def new_trace() -> str:
    """Start a new correlation id for one query. Call at request entry."""
    tid = uuid.uuid4().hex[:12]
    _trace_id.set(tid)
    return tid


log = get_logger()


# ----------------------------- log data flow --------------------------------------------------

# log.info(...)
#    ↓
# Python's logging creates a LogRecord (message="mcp_tool_call", data={...})
#    ↓
# the record goes to the handler (StreamHandler → stderr)
#    ↓
# the handler calls the formatter: JsonFormatter.format(record)
#    ↓
# format() builds the JSON and returns a string
#    ↓
# handler writes that string to stderr
