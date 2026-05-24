"""
Heuristic JSON repair for malformed SSE payloads from third-party providers.

Some providers emit JavaScript-style objects with unquoted keys
(e.g. {error: "overloaded"} or {rate_limit: true, retry_after: 30})
instead of proper JSON. This module attempts to repair such payloads
so the gateway can extract structured error information rather than
silently dropping the chunk.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger("auto-gateway.sse_repair")

# Match a bare identifier key that is followed by a colon.
# Capture the preceding boundary (start-of-string, opening brace, or comma
# with optional whitespace) so we can preserve it in the replacement.
# We use a negative lookahead (?!["']) to avoid re-quoting already-quoted keys.
#
# Groups:
#   1 - preceding boundary: start-of-string, { or , plus any whitespace
#   2 - the bare identifier key (letters, digits, underscore, dot)
#   3 - the colon and any whitespace after it
_UNQUOTED_KEY_PATTERN = re.compile(
    r'(^|[{,]\s*)'                   # boundary: start-of-string, { or , + optional ws
    r'(?!["\'])'                     # negative lookahead: NOT already quoted
    r'([a-zA-Z_][a-zA-Z0-9_.]*)'     # the bare identifier key
    r'(\s*:\s*)',                     # colon with optional whitespace
    re.MULTILINE,
)


def heuristic_json_repair(raw: str) -> dict[str, Any] | None:
    """Attempt to parse a malformed JSON-like string into a dict.

    Handles the common case where a third-party server emits a JavaScript
    object literal with unquoted keys (e.g. {error: "overloaded"} or
    {rate_limit: true, retry_after: 30}).

    Args:
        raw: The raw string extracted from an SSE ``data:`` line.

    Returns:
        A parsed dict if repair + parsing succeeds, or None if the string
        is unrecoverable.
    """
    if not raw or not isinstance(raw, str):
        return None

    # ----- fast path: already valid JSON -----
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
        # Valid JSON but not an object (e.g. a number, string, list).
        # These aren't useful as SSE chunks; treat as unrecoverable.
        return None
    except json.JSONDecodeError:
        pass

    # ----- heuristic repair path -----
    repaired = raw.strip()

    # Step 1: quote bare identifier keys.
    # Each match captures (boundary, key, colon).  We re-insert the
    # boundary unchanged and wrap the key in double quotes.
    repaired = _UNQUOTED_KEY_PATTERN.sub(r'\1"\2"\3', repaired)

    # Step 2: convert JavaScript literal 'true' / 'false' / 'null' if they
    #          were left unquoted as values (json.loads handles these
    #          natively once keys are properly quoted).

    try:
        result = json.loads(repaired)
        if isinstance(result, dict):
            logger.debug(
                "sserepair: successfully repaired malformed SSE payload "
                "(original first 120 chars: %.120s)",
                raw.strip(),
            )
            return result
        return None
    except json.JSONDecodeError as exc:
        logger.debug(
            "sserepair: unable to repair payload (%.120s...): %s",
            raw.strip(),
            exc,
        )
        return None


def extract_error_from_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Extract a structured error dict from a repaired payload if present.

    Recognises common error shapes from various providers:
      - {"error": "message"}
      - {"error": {"message": "...", "type": "..."}}
      - {"type": "error", "message": "..."}
      - {"code": 429, "message": "rate limited"}

    Returns a normalised dict with at least ``message``, or None if the
    payload does not appear to represent an error.
    """
    error = payload.get("error")
    if isinstance(error, str):
        return {"message": error, "type": "provider_error", "code": None}
    if isinstance(error, dict):
        return {
            "message": error.get("message", str(error)),
            "type": error.get("type", "provider_error"),
            "code": error.get("code"),
        }

    # Some providers use a flat shape
    if payload.get("type") == "error" and "message" in payload:
        return {
            "message": payload["message"],
            "type": "provider_error",
            "code": payload.get("code"),
        }

    # Rate-limit signals
    if "retry_after" in payload or "rate_limit" in payload:
        return {
            "message": payload.get("message", "Rate limited by upstream provider"),
            "type": "rate_limit_error",
            "code": payload.get("code", "rate_limit_exceeded"),
        }

    return None
