"""Tests for the LLM-based tool-failure summariser."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from muffin_agent.middlewares.tool_knowledge.summariser import (
    error_class_hash,
    summarise_tool_failure,
)


@pytest.mark.unit
class TestErrorClassHash:
    def test_same_tool_and_error_collapse(self):
        h1 = error_class_hash("equity_fundamental_cash", "HTTP 422 limit must be ≤ 5")
        h2 = error_class_hash("equity_fundamental_cash", "HTTP 422 limit must be ≤ 5")
        assert h1 == h2

    def test_tail_variation_beyond_class_window_collapses(self):
        # Variation in the long tail (past the 120-char class window) should
        # not split the hash — only the prefix participates.
        prefix = (
            "Error calling tool 'equity_fundamental_cash': "
            "HTTP error 422: Unprocessable Entity - {'detail': "
            "[{'type': 'less_than_equal'"
        )
        assert len(prefix) > 120
        h1 = error_class_hash("equity_fundamental_cash", prefix + " ... call_id=A")
        h2 = error_class_hash("equity_fundamental_cash", prefix + " ... call_id=B")
        assert h1 == h2

    def test_different_tools_split(self):
        h1 = error_class_hash("tool_a", "boom")
        h2 = error_class_hash("tool_b", "boom")
        assert h1 != h2

    def test_different_errors_split(self):
        h1 = error_class_hash("tool_a", "auth failed")
        h2 = error_class_hash("tool_a", "rate limit hit")
        assert h1 != h2


@pytest.mark.unit
class TestSummariseToolFailure:
    @pytest.mark.asyncio
    async def test_returns_lesson_from_summariser_response(self):
        summariser = AsyncMock()
        summariser.ainvoke = AsyncMock(
            return_value=MagicMock(
                content=(
                    "equity_estimates_forward_eps: use provider=fmp; "
                    "intrinio key absent."
                )
            )
        )
        lesson = await summarise_tool_failure(
            summariser=summariser,
            tool_name="equity_estimates_forward_eps",
            args={"provider": "intrinio", "symbol": "AMZN"},
            error_message="Missing credential 'intrinio_api_key'",
        )
        assert "intrinio" in lesson.lower()
        summariser.ainvoke.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_falls_back_when_summariser_raises(self):
        summariser = AsyncMock()
        summariser.ainvoke = AsyncMock(side_effect=RuntimeError("model down"))
        lesson = await summarise_tool_failure(
            summariser=summariser,
            tool_name="tool_x",
            args={},
            error_message="kaboom",
        )
        # Fallback string still mentions tool + truncated error.
        assert "tool_x" in lesson
        assert "kaboom" in lesson

    @pytest.mark.asyncio
    async def test_falls_back_on_empty_response(self):
        summariser = AsyncMock()
        summariser.ainvoke = AsyncMock(return_value=MagicMock(content=""))
        lesson = await summarise_tool_failure(
            summariser=summariser,
            tool_name="tool_x",
            args={},
            error_message="kaboom",
        )
        assert "tool_x" in lesson

    @pytest.mark.asyncio
    async def test_truncates_overlong_response(self):
        summariser = AsyncMock()
        summariser.ainvoke = AsyncMock(return_value=MagicMock(content="x" * 1000))
        lesson = await summarise_tool_failure(
            summariser=summariser,
            tool_name="tool_x",
            args={},
            error_message="boom",
        )
        assert len(lesson) <= 241  # cap + ellipsis

    @pytest.mark.asyncio
    async def test_handles_anthropic_style_content_blocks(self):
        summariser = AsyncMock()
        summariser.ainvoke = AsyncMock(
            return_value=MagicMock(
                content=[{"type": "text", "text": "use provider=fmp instead"}]
            )
        )
        lesson = await summarise_tool_failure(
            summariser=summariser,
            tool_name="tool_x",
            args={},
            error_message="boom",
        )
        assert "fmp" in lesson
