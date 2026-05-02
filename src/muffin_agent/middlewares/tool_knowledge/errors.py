"""Error classification + duplicate-block policy.

Encapsulates two questions the middleware needs to answer:

* "Is this error *permanent* for the given (tool, args)?" — drives the
  duplicate-block facet (cache key in ``failed_tool_calls`` state).
* "Have we seen this exact (tool, args) call fail before?" — short-
  circuit answer with the cached error message.

Pure logic, no IO.
"""

from __future__ import annotations

import json
from typing import Any

# Substrings that mark a *permanent* failure for the given (tool, args)
# shape — re-firing the same call cannot succeed. Transient failures
# (5xx, gateway, timeout) are handled by ``ToolRetryMiddleware``.
_PERMANENT_ERROR_HINTS: tuple[str, ...] = (
    "missing credential",
    "api_key",
    "authentication",
    "unauthorized",
    "permission denied",
    "403",
    "404",
    "not found",
    "not supported",
    "invalid parameter",
    "invalid argument",
    "unprocessable entity",
    "422",
    "less than or equal",
    "greater than or equal",
    "no estimates data",
    "no data was returned",
)


def is_permanent_error(error_msg: str) -> bool:
    """Return ``True`` when the error string matches a known permanent hint."""
    if not isinstance(error_msg, str):
        return False
    lower = error_msg.lower()
    return any(hint in lower for hint in _PERMANENT_ERROR_HINTS)


def duplicate_key(tool_call: dict[str, Any]) -> str:
    """Build the cache key used by the duplicate-block facet."""
    args_json = json.dumps(tool_call.get("args", {}), sort_keys=True, default=str)
    return f"{tool_call['name']}:{args_json}"
