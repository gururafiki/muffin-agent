"""Integration tests for ``ToolKnowledgeMiddleware``.

Component-level behaviour (error classification, store CRUD, prompt
rendering) is covered in ``test_errors.py``, ``test_lessons.py`` and
``test_prompt.py``. These tests focus on the wiring inside the middleware
itself: how it composes the components in response to LangChain hooks.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.types import Command

from muffin_agent.middlewares.tool_knowledge import ToolKnowledgeMiddleware


def _make_request(
    tool_name: str = "tool_a",
    args: dict | None = None,
    tool_call_id: str = "tc_1",
    state: dict | None = None,
    store: AsyncMock | None = None,
) -> MagicMock:
    """Build a mock ``ToolCallRequest``."""
    request = MagicMock()
    request.tool_call = {
        "name": tool_name,
        "args": args or {},
        "id": tool_call_id,
    }
    request.runtime = MagicMock()
    request.runtime.store = store
    request.state = state if state is not None else {}
    return request


def _make_store() -> AsyncMock:
    store = AsyncMock()
    store.aget = AsyncMock(return_value=None)
    store.aput = AsyncMock()
    store.asearch = AsyncMock(return_value=[])
    return store


def _summariser(response: str = "use provider=fmp") -> AsyncMock:
    s = AsyncMock()
    s.ainvoke = AsyncMock(return_value=MagicMock(content=response))
    return s


@pytest.mark.unit
class TestDuplicateBlock:
    @pytest.mark.asyncio
    async def test_duplicate_call_short_circuits_with_cached_message(self):
        prior = "Missing credential 'intrinio_api_key'"
        state = {
            "failed_tool_calls": {
                'equity_estimates_forward_eps:{"provider": "intrinio"}': prior
            }
        }
        request = _make_request(
            "equity_estimates_forward_eps",
            {"provider": "intrinio"},
            state=state,
        )
        handler = AsyncMock()  # must not be awaited

        mw = ToolKnowledgeMiddleware()
        result = await mw.awrap_tool_call(request, handler)

        handler.assert_not_awaited()
        assert isinstance(result, ToolMessage)
        assert "DUPLICATE CALL BLOCKED" in result.content
        assert prior in result.content


@pytest.mark.unit
class TestErrorRecording:
    @pytest.mark.asyncio
    async def test_permanent_exception_dedups_and_records(self):
        store = _make_store()
        request = _make_request(
            "equity_estimates_forward_eps",
            {"provider": "intrinio"},
            store=store,
        )
        handler = AsyncMock(
            side_effect=Exception("Missing credential 'intrinio_api_key'")
        )

        mw = ToolKnowledgeMiddleware(summariser=_summariser())
        result = await mw.awrap_tool_call(request, handler)

        assert isinstance(result, Command)
        assert "failed_tool_calls" in result.update
        emitted: list = result.update["messages"]
        assert "Error (permanent)" in emitted[0].content
        store.aput.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_transient_exception_returns_plain_tool_message(self):
        store = _make_store()
        request = _make_request("tool_x", store=store)
        handler = AsyncMock(side_effect=Exception("HTTP 502 Bad Gateway"))

        mw = ToolKnowledgeMiddleware(summariser=_summariser())
        result = await mw.awrap_tool_call(request, handler)

        assert isinstance(result, ToolMessage)
        assert result.content.startswith("Error: ")
        store.aput.assert_awaited_once()  # learned anyway

    @pytest.mark.asyncio
    async def test_without_summariser_records_deterministic_lesson(self):
        store = _make_store()
        request = _make_request(
            "equity_estimates_forward_eps",
            {"provider": "intrinio"},
            store=store,
        )
        handler = AsyncMock(
            side_effect=Exception("Missing credential 'intrinio_api_key'")
        )

        mw = ToolKnowledgeMiddleware()  # no summariser
        result = await mw.awrap_tool_call(request, handler)

        store.aput.assert_awaited_once()
        stored = store.aput.call_args.args[2]
        assert "intrinio" in stored["lesson"]
        assert "equity_estimates_forward_eps" in stored["lesson"]
        assert isinstance(result, Command)
        assert "failed_tool_calls" in result.update

    @pytest.mark.asyncio
    async def test_no_store_skips_lesson_persistence(self):
        request = _make_request("tool_x", store=None)
        handler = AsyncMock(side_effect=Exception("boom"))

        mw = ToolKnowledgeMiddleware(summariser=_summariser())
        result = await mw.awrap_tool_call(request, handler)

        assert isinstance(result, ToolMessage)
        assert "boom" in result.content

    @pytest.mark.asyncio
    async def test_existing_lesson_is_not_resummarised(self):
        existing_item = MagicMock()
        existing_item.value = {"lesson": "already known"}
        store = _make_store()
        store.aget = AsyncMock(return_value=existing_item)

        summariser = _summariser()
        request = _make_request("tool_x", store=store)
        handler = AsyncMock(side_effect=Exception("HTTP 502 Bad Gateway"))

        mw = ToolKnowledgeMiddleware(summariser=summariser)
        await mw.awrap_tool_call(request, handler)

        summariser.ainvoke.assert_not_awaited()
        store.aput.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_errored_tool_message_from_inner_handler_is_learned(self):
        """``ToolRetryMiddleware`` returns ``ToolMessage(status='error')`` —
        the knowledge middleware should still record the lesson."""
        store = _make_store()
        inner_result = ToolMessage(
            content="Error: HTTP 502 Bad Gateway after retry",
            tool_call_id="tc_1",
            status="error",
        )
        handler = AsyncMock(return_value=inner_result)
        request = _make_request("tool_x", store=store)

        mw = ToolKnowledgeMiddleware(summariser=_summariser())
        await mw.awrap_tool_call(request, handler)

        store.aput.assert_awaited_once()


@pytest.mark.unit
class TestPromptInjection:
    @pytest.mark.asyncio
    async def test_lessons_block_prepended_to_system_message(self):
        store = _make_store()
        item = MagicMock()
        item.value = {
            "lesson": "equity_fundamental_cash: limit must be ≤ 5 (FMP free tier)",
            "created_at": "2026-04-26T10:00:00+00:00",
        }
        store.asearch = AsyncMock(return_value=[item])

        captured: dict = {}

        async def fake_handler(req):
            captured["system"] = req.system_message
            return MagicMock()

        mw = ToolKnowledgeMiddleware()
        request = MagicMock()
        request.runtime = MagicMock()
        request.runtime.store = store
        request.system_message = SystemMessage(content="You are an analyst.")
        request.tools = [SimpleNamespace(name="equity_fundamental_cash")]
        request.override = MagicMock(
            side_effect=lambda **kw: SimpleNamespace(
                system_message=kw.get("system_message", request.system_message),
                tools=request.tools,
                runtime=request.runtime,
            )
        )

        await mw.awrap_model_call(request, fake_handler)

        sys = captured["system"]
        assert isinstance(sys, SystemMessage)
        assert "You are an analyst." in sys.content
        assert "Lessons learned from prior tool failures" in sys.content
        assert "limit must be ≤ 5" in sys.content

    @pytest.mark.asyncio
    async def test_no_lessons_means_no_override(self):
        store = _make_store()  # asearch returns []

        async def fake_handler(req):
            return req

        mw = ToolKnowledgeMiddleware()
        request = MagicMock()
        request.runtime = MagicMock()
        request.runtime.store = store
        request.system_message = SystemMessage(content="base prompt")
        request.tools = [SimpleNamespace(name="tool_x")]
        request.override = MagicMock()

        await mw.awrap_model_call(request, fake_handler)
        request.override.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_store_skips_injection(self):
        async def fake_handler(req):
            return req

        mw = ToolKnowledgeMiddleware()
        request = MagicMock()
        request.runtime = MagicMock()
        request.runtime.store = None
        request.system_message = SystemMessage(content="base prompt")
        request.tools = [SimpleNamespace(name="tool_x")]
        request.override = MagicMock()

        await mw.awrap_model_call(request, fake_handler)
        request.override.assert_not_called()

    @pytest.mark.asyncio
    async def test_per_tool_cap_truncates_lesson_list(self):
        store = _make_store()
        items = [
            MagicMock(
                **{
                    "value": {
                        "lesson": f"lesson #{i}",
                        "created_at": f"2026-04-{i + 1:02d}T00:00:00+00:00",
                    }
                }
            )
            for i in range(10)
        ]
        store.asearch = AsyncMock(return_value=items)

        captured: dict = {}

        async def fake_handler(req):
            captured["system"] = req.system_message
            return MagicMock()

        mw = ToolKnowledgeMiddleware(per_tool_cap=3)
        request = MagicMock()
        request.runtime = MagicMock()
        request.runtime.store = store
        request.system_message = SystemMessage(content="base")
        request.tools = [SimpleNamespace(name="tool_x")]
        request.override = MagicMock(
            side_effect=lambda **kw: SimpleNamespace(
                system_message=kw.get("system_message", request.system_message),
                tools=request.tools,
                runtime=request.runtime,
            )
        )

        await mw.awrap_model_call(request, fake_handler)
        sys_text = captured["system"].content
        assert sys_text.count("lesson #") == 3
        assert "lesson #9" in sys_text
        assert "lesson #0" not in sys_text


def _runtime_with_mode(mode: str, store: AsyncMock | None) -> SimpleNamespace:
    """A runtime whose ``config`` carries a ``tool_lessons_mode`` configurable."""
    return SimpleNamespace(
        config={"configurable": {"tool_lessons_mode": mode}}, store=store
    )


@pytest.mark.unit
class TestLessonsMode:
    """The ``tool_lessons_mode`` configurable gates reading and recording."""

    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch):
        # Env wins over configurable — clear it so the configurable is honoured.
        monkeypatch.delenv("TOOL_LESSONS_MODE", raising=False)

    @pytest.mark.asyncio
    async def test_off_does_not_record_on_tool_error(self):
        store = _make_store()
        request = _make_request("tool_x", store=store)
        request.runtime = _runtime_with_mode("off", store)
        handler = AsyncMock(side_effect=Exception("boom"))

        mw = ToolKnowledgeMiddleware()
        await mw.awrap_tool_call(request, handler)

        store.aput.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_read_only_does_not_record_on_tool_error(self):
        store = _make_store()
        request = _make_request("tool_x", store=store)
        request.runtime = _runtime_with_mode("read_only", store)
        handler = AsyncMock(side_effect=Exception("boom"))

        mw = ToolKnowledgeMiddleware()
        await mw.awrap_tool_call(request, handler)

        store.aput.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_read_and_record_records_on_tool_error(self):
        store = _make_store()
        request = _make_request("tool_x", store=store)
        request.runtime = _runtime_with_mode("read_and_record", store)
        handler = AsyncMock(side_effect=Exception("Missing credential 'x_api_key'"))

        mw = ToolKnowledgeMiddleware()
        await mw.awrap_tool_call(request, handler)

        store.aput.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_off_skips_prompt_injection(self):
        store = _make_store()
        item = MagicMock()
        item.value = {"lesson": "known lesson", "created_at": "2026-04-26T10:00:00Z"}
        store.asearch = AsyncMock(return_value=[item])

        async def fake_handler(req):
            return req

        mw = ToolKnowledgeMiddleware()
        request = MagicMock()
        request.runtime = _runtime_with_mode("off", store)
        request.system_message = SystemMessage(content="base prompt")
        request.tools = [SimpleNamespace(name="tool_x")]
        request.override = MagicMock()

        await mw.awrap_model_call(request, fake_handler)
        request.override.assert_not_called()  # off = no read, no inject

    @pytest.mark.asyncio
    async def test_read_only_still_injects_existing_lessons(self):
        store = _make_store()
        item = MagicMock()
        item.value = {"lesson": "known lesson", "created_at": "2026-04-26T10:00:00Z"}
        store.asearch = AsyncMock(return_value=[item])

        captured: dict = {}

        async def fake_handler(req):
            captured["system"] = req.system_message
            return MagicMock()

        mw = ToolKnowledgeMiddleware()
        request = MagicMock()
        request.runtime = _runtime_with_mode("read_only", store)
        request.system_message = SystemMessage(content="base prompt")
        request.tools = [SimpleNamespace(name="tool_x")]
        request.override = MagicMock(
            side_effect=lambda **kw: SimpleNamespace(
                system_message=kw.get("system_message", request.system_message),
                tools=request.tools,
                runtime=request.runtime,
            )
        )

        await mw.awrap_model_call(request, fake_handler)
        assert "known lesson" in captured["system"].content
