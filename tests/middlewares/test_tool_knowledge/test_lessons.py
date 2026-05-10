"""Tests for the lesson catalog."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from muffin_agent.middlewares.tool_knowledge.lessons import (
    Lesson,
    LessonCatalog,
    lessons_namespace,
)


def _store_item(value: dict) -> MagicMock:
    item = MagicMock()
    item.value = value
    return item


@pytest.mark.unit
class TestLessonsNamespace:
    def test_namespace_shape(self):
        assert lessons_namespace("equity_price") == ("tool_lessons", "equity_price")


@pytest.mark.unit
class TestLessonCatalogHas:
    @pytest.mark.asyncio
    async def test_no_store_means_no_record(self):
        catalog = LessonCatalog(None)
        assert await catalog.has("tool_x", "boom") is False

    @pytest.mark.asyncio
    async def test_returns_true_when_store_has_entry(self):
        store = AsyncMock()
        store.aget = AsyncMock(return_value=_store_item({"lesson": "x"}))
        catalog = LessonCatalog(store)
        assert await catalog.has("tool_x", "boom") is True

    @pytest.mark.asyncio
    async def test_returns_false_when_store_read_raises(self):
        store = AsyncMock()
        store.aget = AsyncMock(side_effect=RuntimeError("store down"))
        catalog = LessonCatalog(store)
        assert await catalog.has("tool_x", "boom") is False


@pytest.mark.unit
class TestLessonCatalogRecord:
    @pytest.mark.asyncio
    async def test_no_store_is_noop(self):
        catalog = LessonCatalog(None)
        await catalog.record(
            tool_name="t", args={}, error_message="e", lesson="l"
        )  # must not raise

    @pytest.mark.asyncio
    async def test_writes_lesson_payload(self):
        store = AsyncMock()
        store.aput = AsyncMock()
        catalog = LessonCatalog(store)

        await catalog.record(
            tool_name="equity_estimates_forward_eps",
            args={"provider": "intrinio"},
            error_message="Missing credential 'intrinio_api_key'",
            lesson="use provider=fmp",
        )

        store.aput.assert_awaited_once()
        ns, _key, value = store.aput.call_args.args
        assert ns == ("tool_lessons", "equity_estimates_forward_eps")
        assert value["lesson"] == "use provider=fmp"
        assert value["tool_name"] == "equity_estimates_forward_eps"
        assert value["args_sample"] == {"provider": "intrinio"}
        assert "intrinio_api_key" in value["error_excerpt"]
        assert "created_at" in value

    @pytest.mark.asyncio
    async def test_swallows_store_write_failure(self):
        store = AsyncMock()
        store.aput = AsyncMock(side_effect=RuntimeError("write fail"))
        catalog = LessonCatalog(store)
        # Must not raise.
        await catalog.record(tool_name="t", args={}, error_message="e", lesson="l")


@pytest.mark.unit
class TestLessonCatalogLatestPerTool:
    @pytest.mark.asyncio
    async def test_no_store_returns_empty(self):
        catalog = LessonCatalog(None)
        assert await catalog.latest_per_tool(["tool_a"], cap=5) == []

    @pytest.mark.asyncio
    async def test_no_tool_names_returns_empty(self):
        store = AsyncMock()
        catalog = LessonCatalog(store)
        assert await catalog.latest_per_tool([], cap=5) == []

    @pytest.mark.asyncio
    async def test_returns_newest_first_truncated_to_cap(self):
        items = [
            _store_item(
                {"lesson": f"lesson #{i}", "created_at": f"2026-04-{i + 1:02d}"}
            )
            for i in range(7)
        ]
        store = AsyncMock()
        store.asearch = AsyncMock(return_value=items)
        catalog = LessonCatalog(store)

        result = await catalog.latest_per_tool(["tool_x"], cap=3)

        assert len(result) == 3
        assert all(isinstance(le, Lesson) for le in result)
        assert [le.text for le in result] == [
            "lesson #6",
            "lesson #5",
            "lesson #4",
        ]
        assert all(le.tool_name == "tool_x" for le in result)

    @pytest.mark.asyncio
    async def test_skips_tools_with_search_failure(self):
        items_b = [_store_item({"lesson": "good", "created_at": "2026-04-10"})]

        async def asearch(ns, *_a, **_k):
            if ns == ("tool_lessons", "tool_a"):
                raise RuntimeError("boom")
            return items_b

        store = AsyncMock()
        store.asearch = AsyncMock(side_effect=asearch)
        catalog = LessonCatalog(store)

        result = await catalog.latest_per_tool(["tool_a", "tool_b"], cap=5)
        assert [le.text for le in result] == ["good"]

    @pytest.mark.asyncio
    async def test_skips_blank_lessons(self):
        items = [
            _store_item({"lesson": "", "created_at": "2026-04-10"}),
            _store_item({"lesson": "   ", "created_at": "2026-04-09"}),
            _store_item({"lesson": "real", "created_at": "2026-04-08"}),
        ]
        store = AsyncMock()
        store.asearch = AsyncMock(return_value=items)
        catalog = LessonCatalog(store)

        result = await catalog.latest_per_tool(["tool_x"], cap=5)
        assert [le.text for le in result] == ["real"]
