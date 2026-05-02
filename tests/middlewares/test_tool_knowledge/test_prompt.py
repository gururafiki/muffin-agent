"""Tests for the lesson prompt renderer."""

import pytest
from langchain_core.messages import SystemMessage

from muffin_agent.middlewares.tool_knowledge.lessons import Lesson
from muffin_agent.middlewares.tool_knowledge.prompt import (
    append_block,
    render_lessons_block,
)


@pytest.mark.unit
class TestRenderLessonsBlock:
    def test_empty_returns_empty(self):
        assert render_lessons_block([]) == ""

    def test_renders_header_and_each_lesson(self):
        block = render_lessons_block(
            [
                Lesson(tool_name="tool_a", text="watch out for X", created_at="t1"),
                Lesson(tool_name="tool_b", text="prefer Y", created_at="t0"),
            ]
        )
        assert "Lessons learned from prior tool failures" in block
        assert "- `tool_a`: watch out for X" in block
        assert "- `tool_b`: prefer Y" in block


@pytest.mark.unit
class TestAppendBlock:
    def test_no_existing_returns_block_only(self):
        msg = append_block(None, "BLOCK")
        assert isinstance(msg, SystemMessage)
        assert msg.content == "BLOCK"

    def test_existing_text_is_preserved_and_block_appended(self):
        msg = append_block(SystemMessage(content="base prompt"), "BLOCK")
        assert "base prompt" in msg.content
        assert "BLOCK" in msg.content
        assert msg.content.index("base prompt") < msg.content.index("BLOCK")

    def test_non_string_existing_content_is_replaced(self):
        # Anthropic-style content blocks: we don't try to merge into a list.
        existing = SystemMessage(content=[{"type": "text", "text": "structured"}])
        msg = append_block(existing, "BLOCK")
        assert msg.content == "BLOCK"
