"""LangGraph state schemas for the research agent.

``ResearchState`` is the full top-level pipeline state.

``ResearchClassificationFilterState`` is a smaller schema consumed by
``SkillFilterMiddleware[…]`` inside the researcher node — its only
purpose is to expose ``mode`` and ``task_type`` as flat keys so the
filter can read them.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, NotRequired

from langchain.agents import AgentState
from langchain_core.messages import BaseMessage

from muffin_agent.middlewares.agent_capture.records import merge_tool_runs
from muffin_agent.middlewares.agent_capture.tree import merge_subagent_tree

TaskType = Literal[
    "research_report",
    "comparison",
    "how_to",
    "summary",
    "debate",
    "factual_qa",
]
ResearchMode = Literal["speed", "balanced", "quality"]


class ResearchState(AgentState):
    """Pipeline state for classifier → researcher → rerank → writer.

    Contract for the ``evidence`` accumulator: callers must emit a
    ``list`` per state update (use ``[]`` for empty).  ``operator.add``
    concatenates lists.  Future parallel-source fan-out (multiple writers
    of ``evidence``) must respect this contract to avoid silent
    collisions.
    """

    # ── Input ──────────────────────────────────────────────────────────
    query: str
    chat_history: NotRequired[list[BaseMessage]]
    allowed_sources: NotRequired[list[str]]
    mode_override: NotRequired[ResearchMode]
    task_type_override: NotRequired[TaskType]
    system_instructions: NotRequired[str]

    # ── Lifted by classifier_node (flat keys for SkillFilterMiddleware) ─
    standalone_query: NotRequired[str]
    task_type: NotRequired[str]
    mode: NotRequired[str]
    sources_to_use: NotRequired[list[str]]
    skip_search: NotRequired[bool]
    classification: NotRequired[dict[str, Any]]

    # ── Researcher accumulator ─────────────────────────────────────────
    evidence: Annotated[list[dict[str, Any]], operator.add]

    # ── Rerank output ──────────────────────────────────────────────────
    reranked_evidence: NotRequired[list[dict[str, Any]]]

    # ── Final output ───────────────────────────────────────────────────
    output: NotRequired[dict[str, Any]]

    # ── Tool-execution capture (declaring the channel opts this graph in) ─
    tool_runs: NotRequired[Annotated[list[dict[str, Any]], merge_tool_runs]]
    """Tool-execution records captured by ``AgentCaptureMiddleware``. The
    researcher is a function node, so ``researcher_node`` forwards the deep
    agent's ``tool_runs`` explicitly (see ``nodes/researcher.py``)."""

    subagent_tree: NotRequired[Annotated[dict[str, Any], merge_subagent_tree]]
    """Sub-agent execution tree nodes captured by ``AgentCaptureMiddleware``,
    same forwarding scope as ``tool_runs`` above — ``researcher_node``
    forwards the deep agent's ``subagent_tree`` explicitly alongside it."""


class ResearchClassificationFilterState(AgentState):
    """State schema fed to ``SkillFilterMiddleware`` inside the researcher.

    The middleware reads ``mode`` and ``task_type`` from these flat keys
    to filter ``skills_metadata`` and inject context into the system
    prompt.  Only these two fields are filtering dimensions — keep this
    schema minimal so the middleware's category-key derivation stays
    accurate.
    """

    mode: NotRequired[str]
    task_type: NotRequired[str]
