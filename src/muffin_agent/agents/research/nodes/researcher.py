"""Researcher node: deep agent that gathers evidence.

Single ``create_deep_agent`` call per request.  Tools come from the
Firecrawl MCP (``firecrawl_search`` + ``firecrawl_scrape``) plus any
caller-supplied ``extra_tools``.  Skills under ``/skills/research/``
are loaded and filtered to the current mode + task_type via
``SkillFilterMiddleware``.

LLM-call budget is mode-driven (``research_iter_*``).  Universal
middleware (retries, fallback, tool-cache, tool-knowledge) applies.

Output: ``ResearchEvidenceFindings`` via ``response_format`` — the
node copies the evidence chunks into the state's ``evidence``
accumulator.  Free-form chat content is discarded.

Migration note: when (and if) we want streaming progress to a UI
panel, swap this node's internals to a multi-node sub-graph.  The
public state contract (``state["query"] → state["evidence"]``) is
unchanged.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from langchain.agents.structured_output import AutoStrategy
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.store.base import BaseStore

from ....middlewares import SkillFilterMiddleware
from ....model_config import ModelConfiguration
from ....utils.agent_builder import MuffinAgentBuilder
from ...data_collection.utils import get_tools
from ..config import ResearchConfiguration
from ..schemas import ResearchEvidenceFindings
from ..state import ResearchClassificationFilterState, ResearchState

logger = logging.getLogger(__name__)


FIRECRAWL_TOOLS: tuple[str, ...] = ("firecrawl_search", "firecrawl_scrape")


async def create_researcher_agent(
    config: RunnableConfig,
    *,
    mode: str,
    store: BaseStore | None = None,
    extra_tools: Sequence[BaseTool] | None = None,
):
    """Build the researcher deep agent.

    Args:
        config: Runtime ``RunnableConfig``.  Used to pull
            ``ModelConfiguration`` and ``ResearchConfiguration``.
        mode: ``"speed" | "balanced" | "quality"`` — selects the
            iteration budget passed to ``with_model_call_limit``.
        store: Shared ``BaseStore`` for cross-call tool-result caching
            and the ``/memories/`` namespace.  Optional.
        extra_tools: Caller-supplied tools.  Registered with
            ``is_cacheable=True`` and the default per-tool run cap;
            for tools with custom policies, wrap them yourself
            before passing.
    """
    research_cfg = ResearchConfiguration.from_runnable_config(config)
    model_cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = model_cfg.get_llm_for_role("orchestrator")
    summariser = model_cfg.get_summariser()
    iter_budget = research_cfg.iter_budget_for(mode)

    firecrawl_tools = await get_tools(config, allowed_tools=list(FIRECRAWL_TOOLS))

    builder = (
        MuffinAgentBuilder(primary, name="researcher")
        .with_system_prompt_template("research/researcher.jinja")
        .with_fallback_models(*fallbacks)
        .with_short_term_memory()
        .with_persistent_memory()
        .with_skills(
            ["/skills/research/"],
            filter_middleware=SkillFilterMiddleware[ResearchClassificationFilterState](
                context_header="Research Configuration",
                context_intro="Current research is configured as follows:",
                context_outro=(
                    "The available skills listed above have been pre-filtered "
                    "to match this research mode and task type. Read all of "
                    "them via `read_file` before planning."
                ),
            ),
        )
        .with_model_call_limit(run_limit=iter_budget, exit_behavior="end")
        .with_response_format(AutoStrategy(schema=ResearchEvidenceFindings))
    )
    for tool in firecrawl_tools:
        builder = builder.with_tool(tool, is_cacheable=True, run_limit=None)
    for tool in extra_tools or []:
        builder = builder.with_tool(tool)
    if store is not None:
        builder = builder.with_store(store)
    if summariser is not None:
        builder = builder.with_tool_knowledge(summariser)
    return builder.build_deep_agent()


async def researcher_node(
    state: ResearchState,
    config: RunnableConfig,
    *,
    store: BaseStore | None = None,
    extra_tools: Sequence[BaseTool] | None = None,
) -> dict[str, Any]:
    """Run the researcher deep agent and stash its evidence in state.

    Short-circuits on ``skip_search=True`` (classifier decided no
    research was needed).  Returns ``{"evidence": []}`` in that case so
    the rerank step has a sensible empty input.
    """
    if state.get("skip_search"):
        return {
            "evidence": [],
            "mode": state.get("mode", "balanced"),
            "task_type": state.get("task_type", "research_report"),
        }

    mode = state.get("mode", "balanced")
    task_type = state.get("task_type", "research_report")

    try:
        agent = await create_researcher_agent(
            config,
            mode=mode,
            store=store,
            extra_tools=extra_tools,
        )
        payload = {
            "query": state.get("standalone_query") or state.get("query", ""),
            "task_type": task_type,
            "mode": mode,
            "sources_to_use": state.get("sources_to_use") or ["web"],
        }
        # Pass classification-filtering fields on the runtime state so
        # SkillFilterMiddleware can read them.
        result = await agent.ainvoke(
            {
                "messages": [{"role": "user", "content": json.dumps(payload)}],
                "mode": mode,
                "task_type": task_type,
            },
            config=config,
        )
        structured = (
            result.get("structured_response") if isinstance(result, dict) else None
        )
    except Exception:
        logger.exception("researcher_node failed; returning empty evidence")
        return {"evidence": []}

    if structured is None:
        logger.warning("researcher_node: no structured_response on agent result")
        return {"evidence": []}

    chunks = [chunk.model_dump() for chunk in structured.evidence_chunks]
    # Forward the deep agent's captured tool_runs (its own + nested subagents')
    # so the run view's "Tool execution" panel populates. It's a function node,
    # so the records don't auto-propagate like a compiled-agent graph node.
    tool_runs = result.get("tool_runs") if isinstance(result, dict) else None
    # Same treatment for the sub-agent execution tree (see tool_runs above).
    subagent_tree = result.get("subagent_tree") if isinstance(result, dict) else None
    return {
        "evidence": chunks,
        "tool_runs": tool_runs or [],
        "subagent_tree": subagent_tree or {},
    }
