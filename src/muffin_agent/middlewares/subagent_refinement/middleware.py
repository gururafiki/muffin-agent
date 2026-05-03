"""Subagent-refinement middleware — child and parent classes.

Two roles, two classes — each with the disjoint hook surface its role
needs. The child reads / writes findings on ``/scratch/`` and amends
the system prompt with both the static rules and the per-call prior-
findings block. The parent only amends the system prompt with the
static "how to act on gaps" rules.

Prompt content lives next to the middleware that owns it (see
``prompts.py``). The agent builder is responsible only for *which*
class to register, not for prompt construction.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Generic, NotRequired

from deepagents.backends.protocol import BackendFactory
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ContextT,
    ModelCallResult,
    ModelRequest,
    PrivateStateAttr,
)
from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool
from langgraph.runtime import Runtime

from .prompts import (
    append_block,
    child_instructions,
    parent_instructions,
    render_prior_findings_block,
)
from .schema import CollectionFindings
from .storage import (
    extract_prior_call_id,
    latest_human_text,
    read_findings,
    write_findings,
)


class SubagentRefinementState(AgentState):
    """State carrying the prior-call findings between hooks.

    ``prior_findings`` is set by ``abefore_agent`` when it discovers a
    ``prior_call_id=<id>`` marker in the latest human message and reads
    the corresponding ``/scratch/subagent_runs/<id>.json`` file. The
    ``awrap_model_call`` hook reads it back to render the per-call
    block. Marked ``PrivateStateAttr`` so it never leaks into the final
    agent output.
    """

    prior_findings: NotRequired[Annotated[dict[str, Any], PrivateStateAttr]]


def _resolve_backend(
    factory: BackendFactory, runtime: Runtime[ContextT] | ToolRuntime
):
    """Run the backend factory with whatever runtime shape we received.

    Both ``Runtime`` (from ``before_agent`` / ``after_agent``) and
    ``ToolRuntime`` (from ``wrap_tool_call`` etc.) satisfy the factory's
    informal contract — pass through to the caller's factory.
    """
    return factory(runtime)  # type: ignore[arg-type]


class SubagentRefinementMiddleware(
    AgentMiddleware[SubagentRefinementState, ContextT],
    Generic[ContextT],
):
    """Child-side: structured response + scratch cache + prompt rules.

    Args:
        backend_factory: Callable that returns the agent's filesystem
            backend given the current runtime. ``MuffinAgentBuilder``
            wires this from its composite-backend factory so the
            middleware reads/writes the same ``/scratch/`` route as
            ``read_file`` / ``write_file`` tools.
    """

    state_schema = SubagentRefinementState
    tools: list[BaseTool] = []

    def __init__(self, backend_factory: BackendFactory) -> None:
        """Initialize with the agent's backend factory."""
        self._backend_factory = backend_factory

    # ── Inbound: discover a prior_call_id and stash its findings ──

    async def abefore_agent(
        self,
        state: SubagentRefinementState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Look for ``prior_call_id=<id>`` and load the cached findings."""
        description = latest_human_text(state.get("messages", []) or [])
        prior_id = extract_prior_call_id(description)
        if prior_id is None:
            return None
        try:
            backend = _resolve_backend(self._backend_factory, runtime)
        except Exception:
            return None
        findings = await read_findings(backend, prior_id)
        if findings is None:
            return None
        return {"prior_findings": findings.model_dump()}

    # ── Per-call: amend system prompt with rules + prior data ──

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelCallResult]],
    ) -> ModelCallResult:
        """Amend the system prompt with refinement rules and prior findings."""
        block = child_instructions()
        prior_dict = request.state.get("prior_findings")
        if prior_dict:
            try:
                findings = CollectionFindings.model_validate(prior_dict)
                block = f"{block}\n\n{render_prior_findings_block(findings)}"
            except Exception:
                pass  # malformed cache — fall back to instructions only
        return await handler(
            request.override(
                system_message=append_block(request.system_message, block)
            )
        )

    # ── Outbound: persist this run's findings ──

    async def aafter_agent(
        self,
        state: SubagentRefinementState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Write the structured response to ``/scratch/subagent_runs/``."""
        structured = state.get("structured_response")
        if not isinstance(structured, CollectionFindings):
            return None
        if not structured.call_id:
            structured = structured.model_copy(
                update={"call_id": uuid.uuid4().hex[:12]}
            )
        try:
            backend = _resolve_backend(self._backend_factory, runtime)
        except Exception:
            return None
        await write_findings(backend, structured)
        return {"structured_response": structured}


class SubagentRefinementParentMiddleware(
    AgentMiddleware[AgentState, ContextT],
    Generic[ContextT],
):
    """Parent-side: amend the orchestrator's system prompt with rules.

    Has no state and no tools — only ``awrap_model_call`` to prepend
    the static "how to act on gaps" rules so the orchestrator knows
    how to read ``CollectionFindings`` and re-issue refinement calls
    with ``prior_call_id=<id>``.
    """

    tools: list[BaseTool] = []

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelCallResult]],
    ) -> ModelCallResult:
        """Prepend the parent rules to the system message."""
        return await handler(
            request.override(
                system_message=append_block(
                    request.system_message, parent_instructions()
                )
            )
        )
