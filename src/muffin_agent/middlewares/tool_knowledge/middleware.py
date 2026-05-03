"""``ToolKnowledgeMiddleware`` — adaptive learning from tool failures.

Replaces ``ToolErrorHandlerMiddleware``. Two responsibilities, each
delegated to a focused component:

* ``errors.py`` — what counts as a permanent failure, and how to key
  the duplicate-block cache.
* ``lessons.py`` — store-backed CRUD for lessons keyed by error class.
* ``summariser.py`` — turn a raw error into a one-line lesson (LLM or
  deterministic fallback).
* ``prompt.py`` — render the ``## Lessons learned …`` block and stitch
  it onto the existing system message.

This file is just the wiring: catch tool errors → record a lesson →
optionally short-circuit identical-args repeats; on every model call,
prepend the rendered block to ``request.system_message``.
"""

from __future__ import annotations

import operator
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Annotated, Any, NotRequired

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from .errors import duplicate_key, is_permanent_error
from .lessons import LessonCatalog
from .prompt import append_block, render_lessons_block
from .summariser import (
    DEFAULT_SUMMARISER_PROMPT,
    fallback_lesson,
    summarise_tool_failure,
)

if TYPE_CHECKING:
    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langchain_core.language_models import BaseChatModel

_PER_TOOL_LESSON_CAP = 5


class ToolLessonState(AgentState):
    """Extended state — tracks per-thread permanent-failure dedup cache."""

    failed_tool_calls: NotRequired[Annotated[dict[str, str], operator.or_]]


class ToolKnowledgeMiddleware(AgentMiddleware["ToolLessonState"]):
    """Catch tool errors, learn from them, and surface lessons to the LLM.

    Args:
        summariser: Small chat model used to convert raw errors into
            short lessons. Pass the cheapest model your deployment has;
            the call is short and cached. When ``None`` the middleware
            falls back to a deterministic ``"<tool>: previous call
            failed — <error>"`` string so lessons still accumulate.
        summariser_prompt: System prompt for the summariser model.
            Defaults to :data:`DEFAULT_SUMMARISER_PROMPT`.
        per_tool_cap: Maximum lessons retained per tool in the prompt
            block (newest first).
    """

    state_schema = ToolLessonState

    def __init__(
        self,
        *,
        summariser: BaseChatModel | None = None,
        summariser_prompt: str = DEFAULT_SUMMARISER_PROMPT,
        per_tool_cap: int = _PER_TOOL_LESSON_CAP,
    ) -> None:
        """Initialize with an optional summariser model and prompt."""
        self._summariser = summariser
        self._summariser_prompt = summariser_prompt
        self._per_tool_cap = per_tool_cap
        self.tools: list[Any] = []

    # ── Tool interception ────────────────────────────────────────────

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Catch tool exceptions, dedup permanents, record lessons."""
        tool_name: str = request.tool_call["name"]
        dup_key = duplicate_key(request.tool_call)
        failed_calls: dict[str, str] = request.state.get("failed_tool_calls", {})

        if dup_key in failed_calls:
            return ToolMessage(
                content=(
                    f"DUPLICATE CALL BLOCKED: This tool was already called with "
                    f"identical arguments and failed permanently. "
                    f"Previous error: {failed_calls[dup_key]}"
                ),
                tool_call_id=request.tool_call["id"],
            )

        # Run the tool; treat raised exceptions and errored ToolMessages
        # (e.g. from ``ToolRetryMiddleware`` after retries exhaust) as
        # the same logical event.
        error_msg: str | None = None
        result: ToolMessage | Command[Any] | None = None
        try:
            result = await handler(request)
        except Exception as exc:
            error_msg = str(exc)
        else:
            if (
                isinstance(result, ToolMessage)
                and getattr(result, "status", None) == "error"
            ):
                error_msg = (
                    result.content
                    if isinstance(result.content, str)
                    else str(result.content)
                )

        if error_msg is None:
            assert result is not None
            return result

        await self._record(
            runtime=request.runtime,
            tool_name=tool_name,
            args=request.tool_call.get("args", {}),
            error_message=error_msg,
        )

        permanent = is_permanent_error(error_msg)
        if isinstance(result, ToolMessage):
            error_message_obj: ToolMessage = result
        else:
            prefix = "Error (permanent)" if permanent else "Error"
            error_message_obj = ToolMessage(
                content=f"{prefix}: {error_msg}",
                tool_call_id=request.tool_call["id"],
            )

        if permanent:
            return Command(
                update={
                    "failed_tool_calls": {**failed_calls, dup_key: error_msg},
                    "messages": [error_message_obj],
                }
            )
        return error_message_obj

    async def _record(
        self,
        *,
        runtime: Any,
        tool_name: str,
        args: dict[str, Any],
        error_message: str,
    ) -> None:
        """Summarise the error (or fall back) and persist as a lesson."""
        catalog = LessonCatalog(getattr(runtime, "store", None))
        if await catalog.has(tool_name, error_message):
            return  # Already recorded this error class.

        if self._summariser is not None:
            text = await summarise_tool_failure(
                summariser=self._summariser,
                tool_name=tool_name,
                args=args,
                error_message=error_message,
                system_prompt=self._summariser_prompt,
            )
        else:
            text = fallback_lesson(tool_name, error_message)
        await catalog.record(
            tool_name=tool_name,
            args=args,
            error_message=error_message,
            lesson=text,
        )

    # ── Prompt injection ────────────────────────────────────────────

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> Any:
        """Inject the ``## Lessons learned …`` block into the system prompt."""
        catalog = LessonCatalog(getattr(request.runtime, "store", None))
        tool_names = [
            tool.name
            for tool in request.tools
            if isinstance(getattr(tool, "name", None), str)
        ]
        lessons = await catalog.latest_per_tool(tool_names, cap=self._per_tool_cap)
        block = render_lessons_block(lessons)
        if not block:
            return await handler(request)
        return await handler(
            request.override(system_message=append_block(request.system_message, block))
        )
