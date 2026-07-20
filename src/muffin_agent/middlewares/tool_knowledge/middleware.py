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
detect repetitive tool use (loop-pattern lessons) then prepend the
rendered block to ``request.system_message``.
"""

from __future__ import annotations

import json
import operator
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Annotated, Any, NotRequired, cast

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from .config import ToolKnowledgeConfiguration, ToolLessonsMode
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
        loop_warning_threshold: Minimum number of calls to the same tool
            within a single run before a loop-pattern lesson is recorded
            and injected into the system prompt.
    """

    state_schema = ToolLessonState

    def __init__(
        self,
        *,
        summariser: BaseChatModel | None = None,
        summariser_prompt: str = DEFAULT_SUMMARISER_PROMPT,
        per_tool_cap: int = _PER_TOOL_LESSON_CAP,
        loop_warning_threshold: int = 3,
    ) -> None:
        """Initialize with an optional summariser model and prompt."""
        self._summariser = summariser
        self._summariser_prompt = summariser_prompt
        self._per_tool_cap = per_tool_cap
        self._loop_warning_threshold = loop_warning_threshold
        self.tools: list[Any] = []

    # ── Policy ───────────────────────────────────────────────────────

    @staticmethod
    def _mode(runtime: Any) -> ToolLessonsMode:
        """Resolve the tool-lessons mode from the runtime config.

        Falls back to ``read_and_record`` when config is unavailable so the
        middleware keeps its historical behaviour by default.
        """
        try:
            config = cast("RunnableConfig", getattr(runtime, "config", None) or {})
            return ToolKnowledgeConfiguration.from_runnable_config(
                config
            ).tool_lessons_mode
        except Exception:
            return "read_and_record"

    # ── History helpers ──────────────────────────────────────────────

    @staticmethod
    def _count_tool_calls_in_history(messages: list) -> dict[str, int]:
        """Count total calls per tool name across all AIMessages."""
        counts: dict[str, int] = {}
        for msg in messages:
            if isinstance(msg, AIMessage):
                for tc in msg.tool_calls:
                    name = tc.get("name") or ""
                    if name:
                        counts[name] = counts.get(name, 0) + 1
        return counts

    @staticmethod
    def _get_recent_tool_args(messages: list, tool_name: str) -> dict:
        """Return the most recent args used for tool_name from AIMessage history."""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                for tc in msg.tool_calls:
                    if tc.get("name") == tool_name:
                        return tc.get("args", {})
        return {}

    @staticmethod
    def _get_recent_tool_outputs(
        messages: list, tool_name: str, max_samples: int = 2
    ) -> list[str]:
        """Return content snippets from the most recent ToolMessages for tool_name.

        ToolMessages don't carry tool_name directly — matches via tool_call_id
        collected from AIMessages.
        """
        ids = {
            tc.get("id")
            for msg in messages
            if isinstance(msg, AIMessage)
            for tc in msg.tool_calls
            if tc.get("name") == tool_name and tc.get("id")
        }
        outputs = []
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.tool_call_id in ids:
                c = msg.content
                snippet = c[:300] if isinstance(c, str) else str(c)[:300]
                outputs.append(snippet)
        return outputs[-max_samples:]

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
                status="error",
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
                # json.dumps gives valid JSON instead of Python repr for list content.
                error_msg = (
                    result.content
                    if isinstance(result.content, str)
                    else json.dumps(result.content, default=str)
                )
            elif (
                isinstance(result, ToolMessage)
                and isinstance(result.content, str)
                and is_permanent_error(result.content)
            ):
                # Content-level permanent error (e.g. "OpenAI API key is missing" with
                # status=None). Wrap with "permanently unavailable" so the recorded
                # lesson is agent-facing — the agent can't configure infra keys, it
                # needs to know the tool is off-limits and use alternatives.
                error_msg = (
                    f"permanently unavailable in this environment: "
                    f"{result.content[:120]}"
                )

        if error_msg is None:
            assert result is not None
            return result

        # Record new lessons only in the default mode. Duplicate-blocking of
        # permanent failures (below) stays active in ALL modes — that is
        # per-thread correctness, not the cross-run lessons store.
        if self._mode(request.runtime) == "read_and_record":
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
                status="error",
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

    async def _record_loop_lesson(
        self,
        *,
        runtime: Any,
        tool_name: str,
        count: int,
        messages: list,
    ) -> None:
        """Record a lesson when a tool is called too many times without progress.

        Uses a stable ``error_message`` key so the lesson is recorded once
        even as the call count grows. Passes actual tool output samples to
        the summariser so it can reason about *why* the data is unavailable
        (e.g. auth-gated content) rather than just noting repetition.
        """
        catalog = LessonCatalog(getattr(runtime, "store", None))
        error_message = "repeated calls without retrieving useful data"
        if await catalog.has(tool_name, error_message):
            return

        if self._summariser is not None:
            sample_args = self._get_recent_tool_args(messages, tool_name)
            recent_outputs = self._get_recent_tool_outputs(messages, tool_name)
            output_context = (
                " | ".join(recent_outputs) if recent_outputs else "no output available"
            )
            text = await summarise_tool_failure(
                summariser=self._summariser,
                tool_name=tool_name,
                args=sample_args,
                error_message=(
                    f"called {count} times without progress. "
                    f"Recent output samples: {output_context[:400]}"
                ),
                system_prompt=self._summariser_prompt,
            )
        else:
            text = (
                f"{tool_name}: called {count} times without progress — "
                "if data appears unavailable or access-gated, stop retrying and "
                "report what you found or that the data could not be retrieved."
            )
        await catalog.record(
            tool_name=tool_name,
            args={},
            error_message=error_message,
            lesson=text,
        )

    # ── Prompt injection ────────────────────────────────────────────

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> Any:
        """Inject the ``## Lessons learned …`` block into the system prompt.

        Records loop-pattern lessons for overused tools BEFORE reading the
        catalog so the just-persisted lesson appears in the same call's block.

        Mode-gated: ``off`` short-circuits entirely (no read, no inject, no
        record); ``read_only`` injects existing lessons but records no new
        loop-pattern lessons; ``read_and_record`` (default) does both.
        """
        mode = self._mode(request.runtime)
        if mode == "off":
            return await handler(request)

        catalog = LessonCatalog(getattr(request.runtime, "store", None))
        tool_names = [
            n
            for tool in request.tools
            if isinstance(n := getattr(tool, "name", None), str)
        ]

        # Record loop-pattern lessons first so they're included this turn.
        if mode == "read_and_record":
            tool_counts = self._count_tool_calls_in_history(request.messages)
            for t_name, count in tool_counts.items():
                if count >= self._loop_warning_threshold:
                    await self._record_loop_lesson(
                        runtime=request.runtime,
                        tool_name=t_name,
                        count=count,
                        messages=request.messages,
                    )

        lessons = await catalog.latest_per_tool(tool_names, cap=self._per_tool_cap)
        block = render_lessons_block(lessons)
        if not block:
            return await handler(request)
        return await handler(
            request.override(system_message=append_block(request.system_message, block))
        )
