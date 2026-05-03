"""Error classification and duplicate-block bookkeeping.

Pure policy — no I/O. Decides whether a given error message is permanent
(re-firing the same args cannot succeed) and renders the right
``ToolMessage`` / state update for the middleware to return.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.types import Command

# Substrings that mark a *permanent* failure for the given (tool, args)
# shape — re-firing the same call cannot succeed. Used to seed the
# duplicate-block cache. Transient failures (5xx, gateway, timeout) are
# already handled by ``ToolRetryMiddleware`` upstream.
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


def dup_key(tool_call: dict[str, Any]) -> str:
    """Cache key for the duplicate-block facet — exact (name, args)."""
    args_json = json.dumps(tool_call.get("args", {}), sort_keys=True, default=str)
    return f"{tool_call['name']}:{args_json}"


def extract_error_text(result: ToolMessage) -> str:
    """Pull a string error body out of an errored ``ToolMessage``."""
    return (
        result.content
        if isinstance(result.content, str)
        else str(result.content)
    )


@dataclass(frozen=True)
class ErrorOutcome:
    """The two pieces of information the middleware needs after an error.

    *message* is the ``ToolMessage`` to return to the LLM (either the
    one we received from an inner middleware, or a fresh one we built).
    *permanent* tells the caller whether to also short-circuit future
    identical calls via the ``failed_tool_calls`` map.
    """

    message: ToolMessage
    permanent: bool


def classify(
    *,
    error_msg: str,
    handler_result: ToolMessage | None,
    tool_call_id: str,
) -> ErrorOutcome:
    """Build the ``ToolMessage`` to surface and decide on dedup."""
    permanent = is_permanent_error(error_msg)
    if handler_result is not None:
        # Inner middleware already produced a ``ToolMessage(status="error")``
        # — pass it through verbatim instead of double-wrapping.
        return ErrorOutcome(message=handler_result, permanent=permanent)

    prefix = "Error (permanent)" if permanent else "Error"
    return ErrorOutcome(
        message=ToolMessage(
            content=f"{prefix}: {error_msg}",
            tool_call_id=tool_call_id,
        ),
        permanent=permanent,
    )


def duplicate_block_message(
    *,
    tool_call_id: str,
    prior_error: str,
) -> ToolMessage:
    """Render the message for a (tool, args) shape that already failed."""
    return ToolMessage(
        content=(
            f"DUPLICATE CALL BLOCKED: This tool was already called with "
            f"identical arguments and failed permanently. "
            f"Previous error: {prior_error}"
        ),
        tool_call_id=tool_call_id,
    )


def permanent_command(
    *,
    failed_calls: dict[str, str],
    new_dup_key: str,
    error_msg: str,
    message: ToolMessage,
) -> Command:
    """Build the ``Command`` that records a permanent failure in state."""
    return Command(
        update={
            "failed_tool_calls": {**failed_calls, new_dup_key: error_msg},
            "messages": [message],
        }
    )
