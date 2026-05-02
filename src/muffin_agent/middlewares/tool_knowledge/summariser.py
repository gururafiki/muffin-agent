"""LLM-based tool-failure summariser.

Distills a raw tool error into a short, action-oriented lesson the
calling agent should remember next time. Generic — no per-tool / per-
error-shape parsing. Cached upstream so the same ``(tool, error_class)``
pair only summarises once.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

DEFAULT_SUMMARISER_PROMPT = (
    "You are a tool-failure summariser. The agent will read your one-line "
    "lesson before its next tool call so it can avoid the same mistake. "
    "Mention the tool, the failing parameter or provider, and the corrective "
    "action. Be concrete. Maximum 200 characters. Return only the lesson "
    "text — no preamble, no quotes, no markdown."
)

_FALLBACK_LESSON_TEMPLATE = "{tool}: previous call failed — {error}"
_FALLBACK_TRUNCATE = 200
_LESSON_MAX_CHARS = 240


def error_class_hash(tool_name: str, error_message: str) -> str:
    """Hash ``(tool, error_class)`` for cache deduplication.

    The "class" is derived from the first 120 chars of the error so that
    different *instances* of the same problem (same tool + same fault
    pattern) collapse to one lesson, while genuinely different errors
    surface separately.
    """
    canonical = f"{tool_name}\n{error_message[:120]}"
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


async def summarise_tool_failure(
    *,
    summariser: BaseChatModel,
    tool_name: str,
    args: dict[str, Any],
    error_message: str,
    system_prompt: str = DEFAULT_SUMMARISER_PROMPT,
) -> str:
    """Ask the summariser model for a one-line lesson.

    Returns a short string. On any failure (model error, empty output,
    overlong output) returns a deterministic fallback string built from
    the raw error so the agent still gets a usable hint.
    """
    user_message = (
        f"Tool: {tool_name}\n"
        f"Args: {json.dumps(args, sort_keys=True, default=str)}\n"
        f"Error: {error_message}"
    )
    try:
        response = await summariser.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
        )
    except Exception:
        logger.debug("Summariser call failed for %s", tool_name, exc_info=True)
        return _fallback(tool_name, error_message)

    lesson = _extract_text(response).strip()
    if not lesson:
        return _fallback(tool_name, error_message)
    if len(lesson) > _LESSON_MAX_CHARS:
        lesson = lesson[:_LESSON_MAX_CHARS].rstrip() + "…"
    return lesson


def _extract_text(response: Any) -> str:
    """Pull plain text out of an ``AIMessage`` or compatible object."""
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    # Anthropic-style content blocks.
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


def fallback_lesson(tool_name: str, error_message: str) -> str:
    """Deterministic lesson used when no summariser is configured."""
    truncated = error_message.strip().replace("\n", " ")[:_FALLBACK_TRUNCATE]
    return _FALLBACK_LESSON_TEMPLATE.format(tool=tool_name, error=truncated)


# Backwards-compatible alias retained for the summariser's internal fallback.
_fallback = fallback_lesson
