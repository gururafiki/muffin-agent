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
    """Append *block* to *existing* preserving prior content where possible."""
    existing_text = (
        existing.content
        if existing is not None and isinstance(existing.content, str)
        else ""
    )
    combined = f"{existing_text}\n\n{block}".strip() if existing_text else block
    return SystemMessage(content=combined)
