"""Make an agent actually collect data before it is allowed to answer.

## The failure this exists for

Free/weak models routinely **single-shot the structured output with zero tool
calls and fabricate plausible evidence**. Observed in production on
`nvidia/nemotron-3-super-120b-a12b:free`: 0 TOOL spans across an entire criteria
run, with confident `data_sources` naming subagents that never executed; and on
council threads where a persona's verdict cited figures it never retrieved.

Prompt contracts do not fix this — every affected prompt already carries a
"non-negotiable data-collection contract". A weak model ignores it.

## Two jobs, deliberately in one middleware

1. **Record** what really executed (`executed_tools`), for the deterministic
   reconciliation in `criteria_analysis`'s `package` node.
2. **Enforce** (optional, `max_attempts > 0`): when the agent is about to end
   having executed nothing, jump back to the model with a corrective message.

They are together because they must agree on what counts as "collected data".
Split across two middlewares, one could enforce on a definition the other does
not record, and the disagreement would be silent.

This is **not** observability. A run's execution record — topology, transcripts,
tool calls — lives in LangGraph's checkpoints and is read from there. What is
recorded here is a short evidence list a *node* needs for a *decision*, and the
agent cannot simply be asked, because the agent is what's being checked.

## After exhaustion, accept — don't raise

Unlike `_StructuredOutputRetryMiddleware` (a missing structured response breaks
downstream unpacking, so it raises), a data-free answer is *usable* as long as it
is labelled. `_reconcile_data_sources` strips the fabricated `data_sources`, caps
confidence at 0.3 and records a limitation, and the UI says "no live data was
collected". A truthful low-confidence evaluation beats a failed run.
"""

from __future__ import annotations

import json
import operator
from typing import Annotated, Any, Generic, NotRequired

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ContextT, hook_config
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime

# Framework/filesystem plumbing a deep agent calls constantly. None of it is
# data retrieval, so counting it would defeat the check: an agent that only ever
# wrote a todo list would read as "collected data".
_PLUMBING_TOOLS: frozenset[str] = frozenset(
    {
        "write_todos",
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
        "discover_cached_tool_outputs",
        "get_tool_output_schema",
        "write_cached_tool_output_to_backend",
    }
)

_ARGS_PREVIEW_CHARS = 200

# ``additional_kwargs`` marker identifying guard-injected nudges, so attempts are
# counted robustly even after a copy edit to the message text.
_NUDGE_MARKER = "data_collection_nudge"

_NUDGE = (
    "You have not called a single data-collection tool in this run, so you have "
    "no retrieved data to reason from. Do not answer from prior knowledge. Call "
    "the tools you need now, then produce your final output using only what they "
    "return. If a tool fails, say so in your limitations rather than estimating."
)


def _args_preview(args: Any) -> str:
    """Render a tool call's arguments short and searchable."""
    if not args:
        return ""
    try:
        text = json.dumps(args, default=str)
    except (TypeError, ValueError):
        text = str(args)
    return text[:_ARGS_PREVIEW_CHARS]


def executed_tool_labels(
    messages: list[AnyMessage],
    *,
    agent_name: str | None = None,
    exclude_tools: frozenset[str] = frozenset(),
) -> list[str]:
    """Label every tool call in *messages* that actually returned.

    A call is only evidence once its ``ToolMessage`` comes back — an
    ``AIMessage`` naming a tool proves intent, not retrieval, and intent is
    exactly what a hallucinating model produces.

    The deep-agent ``task`` tool is deliberately kept: its arguments name the
    sub-agent that was delegated to, which is what a claimed ``data_sources``
    entry is usually matched against.

    *exclude_tools* must carry the agent's **structured-response schema name**.
    ``AutoStrategy`` surfaces that schema as a tool, so the synthetic final call
    that emits the answer looks exactly like a tool execution — and counting it
    would mark every evaluation as data-backed, including the single-shot
    fabrications this whole pass exists to catch.
    """
    calls: dict[str, tuple[str, Any]] = {}
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls:
                cid, name = tc.get("id"), tc.get("name") or ""
                if cid and name:
                    calls[cid] = (name, tc.get("args", {}))

    labels: list[str] = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        pair = calls.get(msg.tool_call_id)
        if pair is None:
            continue
        name, args = pair
        if name in _PLUMBING_TOOLS or name in exclude_tools:
            continue
        parts = [name, _args_preview(args)]
        if agent_name:
            parts.append(agent_name)
        labels.append(" ".join(p for p in parts if p))
    return labels


class DataCollectionGuardState(AgentState):
    """Declares the evidence channel and the guard's livelock counter."""

    executed_tools: NotRequired[Annotated[list[str], operator.add]]
    # Advances on EVERY guard jump, even when the model never re-runs. See
    # ``_guard`` — this is the ceiling that survives ``ModelCallLimitMiddleware(
    # exit_behavior="end")``, which every persona collector uses.
    data_collection_jumps: NotRequired[Annotated[int, operator.add]]


class DataCollectionGuardMiddleware(
    AgentMiddleware[DataCollectionGuardState, ContextT],
    Generic[ContextT],
):
    """Record what an agent really ran; optionally make it run something.

    Reducer-backed, so a nested ``task``-invoked sub-agent's labels merge up to
    the parent automatically. A parent that does not declare ``executed_tools``
    drops them at its boundary for free, which keeps the evidence scoped to the
    node that needs it instead of leaking run-wide.
    """

    state_schema = DataCollectionGuardState

    def __init__(
        self,
        name: str | None = None,
        *,
        exclude_tools: frozenset[str] = frozenset(),
        max_attempts: int = 0,
    ) -> None:
        """Initialise the guard.

        Args:
            name: The agent's label, included in each evidence entry.
            exclude_tools: Tool names that are not data collection. **Must**
                include the agent's structured-response schema name — see
                :func:`executed_tool_labels`.
            max_attempts: How many times to bounce a data-free answer back to
                the model. ``0`` (default) records evidence without enforcing,
                for agents where collecting nothing is a legitimate outcome.
        """
        super().__init__()
        self._name = name
        self._exclude_tools = exclude_tools
        self._max_attempts = max_attempts

    def _nudge_count(self, messages: list[AnyMessage]) -> int:
        return sum(
            1
            for m in messages
            if isinstance(m, HumanMessage) and m.additional_kwargs.get(_NUDGE_MARKER)
        )

    def _guard(self, state: DataCollectionGuardState) -> dict[str, Any] | None:
        messages = list(state.get("messages") or [])
        labels = executed_tool_labels(
            messages, agent_name=self._name, exclude_tools=self._exclude_tools
        )
        update: dict[str, Any] = {"executed_tools": labels} if labels else {}

        if labels or self._max_attempts <= 0:
            return update or None

        # Two independent ceilings, same reasoning as the structured-output
        # guard: nudges in the transcript only advance when the MODEL runs, but
        # a spent model-call budget (``exit_behavior="end"``) stops that
        # happening — so the jump counter, which advances on every bounce, is
        # what actually terminates the loop.
        jumps = int(state.get("data_collection_jumps", 0))
        if max(self._nudge_count(messages), jumps) >= self._max_attempts:
            # Give up and let the answer through. Downstream truthing labels it
            # ``data_collected=False``; a flagged answer beats a failed run.
            return update or None

        return {
            **update,
            "jump_to": "model",
            "data_collection_jumps": 1,
            "messages": [
                HumanMessage(content=_NUDGE, additional_kwargs={_NUDGE_MARKER: True})
            ],
        }

    @hook_config(can_jump_to=["model"])
    def after_agent(
        self,
        state: DataCollectionGuardState,
        runtime: Runtime[ContextT],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Record the evidence; bounce a data-free answer back to the model."""
        return self._guard(state)

    @hook_config(can_jump_to=["model"])
    async def aafter_agent(
        self,
        state: DataCollectionGuardState,
        runtime: Runtime[ContextT],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Async variant of :meth:`after_agent`."""
        return self._guard(state)
