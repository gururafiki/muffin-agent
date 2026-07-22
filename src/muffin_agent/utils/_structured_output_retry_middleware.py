"""Internal middleware that recovers a missing structured response in-loop.

Wired automatically by
:class:`muffin_agent.utils.agent_builder.MuffinAgentBuilder` whenever a
response format is configured (:meth:`with_response_format` or the
subagent-refinement default).

Works around an upstream langchain bug (present through 1.3.13):
``_make_tools_to_model_edge`` exit condition 3 ends the agent loop as soon
as ANY structured-output ``ToolMessage`` exists — without checking that
parsing succeeded. When a model pairs a malformed structured-output tool
call with a regular tool call (e.g. ``write_todos``) in the same
``AIMessage``, the parse-error feedback langchain appends for retry is
never shown to the model: the loop routes through the tools node for the
sibling call and then exits. ``structured_response`` stays unset and any
downstream consumer of the unpacked state fields fails.

This guard runs FIRST among ``after_agent`` hooks (it is registered last —
``after_agent`` executes in reverse registration order). When the agent is
about to end without a structured response it jumps back to the model so
the model can read the parse-error feedback already in the transcript and
retry in-loop — far cheaper than re-running the whole agent node. If the
model never called the schema tool at all, an explicit nudge
``HumanMessage`` is injected so the jump carries new information. After
``max_attempts`` total attempts it raises
:class:`StructuredOutputRetryExhaustedError` — a direct ``Exception``
subclass, so a node-level ``RetryPolicy`` still re-runs the agent node as
the outer backstop (langgraph's ``default_retry_on`` excludes common
builtin errors like ``ValueError``/``RuntimeError`` but retries unknown
``Exception`` subclasses).

Not intended for direct use by callers.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, NotRequired, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    hook_config,
)
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime

# additional_kwargs marker identifying guard-injected nudge messages, so
# attempts are counted robustly even when the model never calls the schema
# tool (content-based matching would break on copy edits).
_NUDGE_MARKER = "structured_output_retry_nudge"


class _StructuredRetryState(AgentState):
    """Adds a private jump counter so the guard cannot livelock.

    The transcript-artifact attempt count (schema ``ToolMessage``s + nudges)
    only advances when the MODEL runs. But ``ModelCallLimitMiddleware(
    exit_behavior="end")`` can stop the model from ever running again once its
    call budget is spent: the transcript then never gains a new artifact, the
    artifact count freezes below ``max_attempts``, and ``_guard`` would jump to
    the model forever while ``ModelCallLimit`` bounces it straight back to
    ``end`` — one superstep per bounce, until the Platform ``recursion_limit``
    (10011 in prod) finally trips after ~10k iterations. This ``operator.add``
    counter increments on EVERY guard jump regardless of whether the model
    actually ran, giving the guard an execution-independent ceiling that breaks
    the livelock after ``max_attempts`` jumps.
    """

    structured_retry_jumps: NotRequired[Annotated[int, operator.add]]


class StructuredOutputRetryExhaustedError(Exception):
    """No structured response after all in-loop retries.

    Deliberately a direct ``Exception`` subclass (NOT ``RuntimeError`` /
    ``ValueError``): langgraph's ``default_retry_on`` refuses to retry the
    common builtin error types but retries unknown ``Exception`` subclasses,
    so a node-level ``RetryPolicy(max_attempts=2)`` re-runs the whole agent
    node once before the failure propagates (the propagate-errors philosophy).
    """


class _StructuredOutputRetryMiddleware(
    AgentMiddleware[_StructuredRetryState, Any, Any]
):
    """Jump back to the model when the loop ends without ``structured_response``.

    One attempt = one schema-tool ``ToolMessage`` (a parse failure or a
    success), one injected nudge, OR one guard jump (the ``structured_retry_jumps``
    counter) — whichever is highest. The jump counter is the livelock backstop:
    it advances even when the model can no longer run (see
    :class:`_StructuredRetryState`). The guard fires only when the agent is
    ending WITHOUT a structured response, so successful runs never pay for it.
    """

    state_schema = _StructuredRetryState

    def __init__(self, schema_name: str, *, max_attempts: int = 3) -> None:
        """Initialize the guard.

        Args:
            schema_name: Class name of the response-format schema — the
                structured-output strategies expose it as a tool of the same
                name, and its ``ToolMessage``s are how attempts are counted.
            max_attempts: Total structured-output attempts (schema tool calls
                + injected nudges) before giving up with
                :class:`StructuredOutputRetryExhaustedError`.
        """
        super().__init__()
        self._schema_name = schema_name
        self._max_attempts = max_attempts

    def _count_attempts(self, messages: list[AnyMessage]) -> int:
        """Count prior structured-output attempts in the transcript."""
        attempts = 0
        for message in messages:
            if isinstance(message, ToolMessage) and message.name == self._schema_name:
                attempts += 1
            elif isinstance(message, HumanMessage) and message.additional_kwargs.get(
                _NUDGE_MARKER
            ):
                attempts += 1
        return attempts

    def _guard(self, state: AgentState[Any]) -> dict[str, Any] | None:
        if state.get("structured_response") is not None:
            return None
        # Two independent ceilings: transcript artifacts (advance only when the
        # model runs) and guard jumps (advance every bounce — the livelock
        # backstop when the model can no longer run). Whichever hits first wins.
        attempts = self._count_attempts(state.get("messages", []))
        jumps = cast(int, state.get("structured_retry_jumps", 0))
        effective = max(attempts, jumps)
        if effective >= self._max_attempts:
            raise StructuredOutputRetryExhaustedError(
                f"Agent ended without a structured response after {effective} "
                f"attempt(s) at the '{self._schema_name}' output tool. The parse "
                "errors (if any) are in the message history; the model failed to "
                "produce valid output — or could not run (e.g. a model-call limit "
                "was reached)."
            )
        # Count this jump. ``operator.add`` accumulates it across guard calls, so
        # the counter climbs even when the model never re-runs to add a new
        # transcript artifact.
        update: dict[str, Any] = {"jump_to": "model", "structured_retry_jumps": 1}
        if attempts == 0:
            # The model never called the schema tool — the transcript carries
            # no feedback, so inject an explicit instruction.
            update["messages"] = [
                HumanMessage(
                    content=(
                        "You have not produced the required structured output. "
                        f"Call the `{self._schema_name}` tool now with your "
                        "final answer."
                    ),
                    additional_kwargs={_NUDGE_MARKER: True},
                )
            ]
        return update

    @hook_config(can_jump_to=["model"])
    def after_agent(
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Re-enter the loop if the agent is ending without structured output."""
        return self._guard(state)

    @hook_config(can_jump_to=["model"])
    async def aafter_agent(
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Async variant of :meth:`after_agent`."""
        return self._guard(state)
