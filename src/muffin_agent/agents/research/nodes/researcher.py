"""Researcher stage: the deep agent that gathers evidence.

A compiled deep agent added to the graph via ``add_node``, so it owns its own
``checkpoint_ns`` — its transcript, tool calls and nested ``task`` sub-agents are
readable per-namespace without any capture plumbing.

Tools come from the Firecrawl MCP (``firecrawl_search`` + ``firecrawl_scrape``) plus
any caller-supplied ``extra_tools``. Skills under ``/skills/research/`` are loaded and
filtered to the current mode + task_type via ``SkillFilterMiddleware``.

**One agent per mode.** The LLM-call budget is mode-driven (``research_iter_*``:
speed=2 / balanced=6 / quality=25) and ``with_model_call_limit`` bakes it in at build
time, so the graph builds all three and routes to one on ``state["mode"]`` — see
``graph.py:_route_researcher``. The node name (``researcher_speed`` / ``_balanced`` /
``_quality``) therefore also records which depth actually ran.

Output: ``ResearcherNodeOutput`` via ``response_format``, auto-unpacked into the
``evidence`` (``operator.add``) and ``notes`` state channels. Free-form chat content
is discarded. Errors propagate — ``RetryPolicy`` on the node plus the model-fallback
chain are the resilience layers.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from langchain.agents.structured_output import AutoStrategy
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.store.base import BaseStore

from ....middlewares import SkillFilterMiddleware
from ....model_config import ModelConfiguration
from ....utils.agent_builder import MuffinAgentBuilder
from ...data_collection.utils import get_tools
from ..config import ResearchConfiguration
from ..schemas import ResearcherNodeOutput
from ..state import (
    RESEARCH_MODES,
    ResearchClassificationFilterState,
    ResearcherAgentState,
)

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
        MuffinAgentBuilder(primary, name=f"researcher_{mode}")
        .with_state_schema(ResearcherAgentState)
        .with_input_prompt_template("research/researcher.jinja")
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
        .with_response_format(AutoStrategy(schema=ResearcherNodeOutput))
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


async def build_researchers_by_mode(
    config: RunnableConfig,
    *,
    store: BaseStore | None = None,
    extra_tools: Sequence[BaseTool] | None = None,
) -> dict[str, Runnable[Any, Any]]:
    """Build one compiled researcher per mode, keyed by mode name.

    ``with_model_call_limit`` bakes the iteration budget in at build time, so the
    only way to keep per-run ``mode_override`` meaningful is to build all three and
    route. See ``graph.py:_route_researcher``.
    """
    return {
        mode: await create_researcher_agent(
            config, mode=mode, store=store, extra_tools=extra_tools
        )
        for mode in RESEARCH_MODES
    }
