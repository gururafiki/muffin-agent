"""Prompt rendering for the lessons block injected into the system message."""

from __future__ import annotations

from langchain_core.messages import SystemMessage

from .lessons import Lesson

_PROMPT_HEADER = "## Lessons learned from prior tool failures"


def render_lessons_block(lessons: list[Lesson]) -> str:
    """Render the system-prompt addendum (empty string if no lessons)."""
    if not lessons:
        return ""
    body = "\n".join(f"- `{le.tool_name}`: {le.text}" for le in lessons)
    return f"{_PROMPT_HEADER}\n{body}"


def append_block(existing: SystemMessage | None, block: str) -> SystemMessage:
    """Append *block* to *existing*, preserving both str and content-blocks content.

    deepagents >= 0.6 composes the system prompt as content blocks
    (``SystemMessage(content_blocks=[...])`` — ``content`` is a list).
    Replacing non-string content would wipe the entire base prompt
    whenever any lesson exists, so list content is preserved and the
    block appended as an extra text part.
    """
    if existing is None:
        return SystemMessage(content=block)
    content = existing.content
    if isinstance(content, str):
        combined = f"{content}\n\n{block}".strip() if content else block
        return SystemMessage(content=combined)
    if isinstance(content, list) and content:
        return SystemMessage(
            content=[*content, {"type": "text", "text": f"\n\n{block}"}]
        )
    return SystemMessage(content=block)
