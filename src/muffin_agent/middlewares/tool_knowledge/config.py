"""Configuration for ``ToolKnowledgeMiddleware`` — the tool-lessons policy."""

from __future__ import annotations

from typing import Literal

from muffin_agent.utils.base_config import BaseConfiguration

ToolLessonsMode = Literal["read_and_record", "read_only", "off"]


class ToolKnowledgeConfiguration(BaseConfiguration):
    """Per-run policy for how agents use tool-failure lessons.

    Env ``TOOL_LESSONS_MODE`` or ``configurable.tool_lessons_mode``:

    * ``read_and_record`` (default) — inject the accumulated lessons into the
      system prompt AND record new lessons (error + loop patterns).
    * ``read_only`` — inject existing lessons but record nothing new.
    * ``off`` — do not inject or record; the middleware is inert (duplicate
      permanent-failure blocking within a run still applies — that is
      per-thread correctness, not the lessons store).
    """

    tool_lessons_mode: ToolLessonsMode = "read_and_record"
