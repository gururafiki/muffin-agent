"""Full-fidelity message serialization shared by transcript + tool records."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AnyMessage

# Cap a single message's text so a chatty subagent can't bloat parent state.
_MAX_TEXT = 4000


def flatten_content(content: Any, *, cap: int = _MAX_TEXT) -> str:
    """Flatten message content (str | content blocks) to capped plain text."""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        text = "".join(parts)
    else:
        text = str(content) if content is not None else ""
    return text if len(text) <= cap else text[:cap] + "\n… (truncated)"


def cache_hit(msg: Any) -> bool:
    """Report whether a tool message was served from the tool-result cache."""
    kwargs = getattr(msg, "additional_kwargs", {}) or {}
    meta = kwargs.get("cache") if isinstance(kwargs, dict) else None
    return bool(isinstance(meta, dict) and meta.get("hit"))


def serialize_messages(messages: list[AnyMessage]) -> list[dict[str, Any]]:
    """Trim messages to the shape the UI's message renderer understands.

    Unlike the pre-consolidation transcript serializer this PRESERVES
    ``status`` and the cache-hit flag on tool messages, so tool-execution
    views can be derived from the transcript without refetching.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        d: dict[str, Any] = {
            "type": getattr(m, "type", None),
            "content": flatten_content(getattr(m, "content", "")),
        }
        name = getattr(m, "name", None)
        if name:
            d["name"] = name
        tcid = getattr(m, "tool_call_id", None)
        if tcid:
            d["tool_call_id"] = tcid
            status = getattr(m, "status", None)
            if status:
                d["status"] = status
            if cache_hit(m):
                d["cache_hit"] = True
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            d["tool_calls"] = [
                {"name": tc.get("name"), "args": tc.get("args"), "id": tc.get("id")}
                for tc in tool_calls
            ]
        out.append(d)
    return out


def first_human(messages: list[AnyMessage]) -> str:
    """Return the first human message's text (a run's task description)."""
    for m in messages:
        if getattr(m, "type", None) == "human":
            return flatten_content(getattr(m, "content", ""))
    return ""
