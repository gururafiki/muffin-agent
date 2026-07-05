"""Tool-telemetry record schema, state channel, and message-walk builder.

Records are reconstructed from the agent's message history in
``aafter_agent`` (no per-call state writes). They propagate from nested
``task``-invoked subagents up to the parent exactly like
``subagent_runs`` — a reducer-backed state channel the deepagents ``task``
tool merges up automatically.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, NotRequired

from langchain.agents import AgentState
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

# Field caps (chars) — keep per-thread state bounded even for chatty runs.
ARGS_PREVIEW = 300
OUTPUT_PREVIEW = 400
ERROR_PREVIEW = 300
#: Max records emitted per agent capture; on overflow a single ``truncated``
#: marker is appended and the rest dropped.
MAX_RECORDS_PER_CAPTURE = 50

# Framework/plumbing tools that are noise in a "data collection" view.
DEFAULT_EXCLUDE_TOOLS: frozenset[str] = frozenset(
    {
        "write_todos",
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
        "discover_cached_tool_outputs",
        "get_tool_output_schema",
        "write_cached_tool_output_to_backend",
    }
)

_DUPLICATE_BLOCKED_PREFIX = "DUPLICATE CALL BLOCKED"
_ERROR_PREFIXES = ("Error:", "Error (permanent):")


def merge_tool_runs(
    left: list[dict[str, Any]] | None, right: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Reducer: concatenate tool-run records across nested/parallel agents."""
    return [*(left or []), *(right or [])]


class ToolTelemetryState(AgentState):
    """Adds the ``tool_runs`` channel — a flat list of tool-execution records."""

    tool_runs: NotRequired[Annotated[list[dict[str, Any]], merge_tool_runs]]


def _cap(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def _flatten(content: Any) -> str:
    """Flatten message content (str | content blocks | other) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return str(content) if content is not None else ""


def _args_preview(args: Any) -> str:
    try:
        return _cap(json.dumps(args, default=str), ARGS_PREVIEW)
    except (TypeError, ValueError):
        return _cap(str(args), ARGS_PREVIEW)


def _classify(msg: ToolMessage, content: str) -> tuple[str, str | None]:
    """Return ``(status, error_preview)`` for a tool result message."""
    if content.startswith(_DUPLICATE_BLOCKED_PREFIX):
        return "duplicate_blocked", _cap(content, ERROR_PREVIEW)
    if getattr(msg, "status", None) == "error" or content.startswith(_ERROR_PREFIXES):
        return "error", _cap(content, ERROR_PREVIEW)
    return "ok", None


def _cache_hit(msg: ToolMessage) -> bool:
    cache = getattr(msg, "additional_kwargs", {}) or {}
    meta = cache.get("cache") if isinstance(cache, dict) else None
    return bool(isinstance(meta, dict) and meta.get("hit"))


def build_tool_records(
    messages: list[AnyMessage],
    *,
    agent_name: str,
    is_subagent: bool,
    exclude_tools: frozenset[str] = DEFAULT_EXCLUDE_TOOLS,
) -> list[dict[str, Any]]:
    """Reconstruct one record per completed tool call from *messages*.

    Pairs each ``AIMessage`` tool call with its ``ToolMessage`` result by
    ``tool_call_id``. Only THIS agent's own calls are in *messages* — nested
    subagents' calls arrive via the reducer channel, so there is no
    double-counting. The ``task`` tool is kept and tagged
    ``is_subagent_call=True`` (it is the "delegated to <subagent>" signal).
    """
    # call_id → (tool_name, args) from every AIMessage tool call.
    calls: dict[str, tuple[str, Any]] = {}
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls:
                cid = tc.get("id")
                name = tc.get("name") or ""
                if cid and name:
                    calls[cid] = (name, tc.get("args", {}))

    records: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        pair = calls.get(msg.tool_call_id)
        if pair is None:
            continue
        tool_name, args = pair
        if tool_name in exclude_tools:
            continue
        content = _flatten(getattr(msg, "content", ""))
        status, error = _classify(msg, content)
        output_preview = "" if status != "ok" else _cap(content, OUTPUT_PREVIEW)
        if len(records) >= MAX_RECORDS_PER_CAPTURE:
            records.append(
                {
                    "tool": tool_name,
                    "agent": agent_name,
                    "is_subagent_call": tool_name == "task",
                    "status": "truncated",
                    "cache_hit": False,
                    "args_preview": "",
                    "output_preview": "",
                    "error": None,
                }
            )
            break
        records.append(
            {
                "tool": tool_name,
                "agent": agent_name,
                "is_subagent_call": tool_name == "task",
                "status": status,
                "cache_hit": _cache_hit(msg),
                "args_preview": _args_preview(args),
                "output_preview": output_preview,
                "error": error,
            }
        )
    return records
