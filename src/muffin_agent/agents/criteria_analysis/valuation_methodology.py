"""Stage 3: valuation methodology research.

Deep agent that performs web research to surface (a) the canonical
valuation methodology the analyst community applies to this ticker and
(b) any criteria the skill-filtered Stage 2 stream would not capture
(recent sell-side debate, narrative drivers, idiosyncratic governance
issues).

Subagents: ``web-search`` (Firecrawl-backed ReAct agent) plus
``discovery-screening`` for peer-comparison context.  No sandbox, no
persistent memory, no skills — keep the surface tight.
"""

from typing import Any

from deepagents import CompiledSubAgent
from langchain.agents.structured_output import AutoStrategy
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from typing_extensions import TypedDict

from ...model_config import ModelConfiguration
from ...utils.agent_builder import MuffinAgentBuilder
from ..data_collection import (
    create_discovery_screening_data_collection_agent,
    create_web_search_data_collection_agent,
)
from ._node_helpers import invoke_structured_agent
from .schemas import ValuationMethodologyOutput

# ── Input state schema ────────────────────────────────────────────────────────


class ValuationMethodologyInputState(TypedDict, total=False):
    """Fields read by ``valuation_methodology_node``."""

    ticker: str
    query: str
    sector: str
    sub_sector: str
    market: str
    stock_type: str
    classification: dict[str, Any]


# ── Subagent builder ──────────────────────────────────────────────────────────


async def _build_methodology_subagents(
    config: RunnableConfig,
) -> list[CompiledSubAgent]:
    """Build the methodology-research subagent set."""
    web_search_agent = await create_web_search_data_collection_agent(config)
    discovery_screening_agent = await create_discovery_screening_data_collection_agent(
        config
    )
    return [
        CompiledSubAgent(
            name="web-search",
            description=(
                "Firecrawl-backed web search and scraping agent.  Use to "
                "discover sell-side research, IR materials, recent press, "
                "industry analyst commentary, and governance disclosures.  "
                "Pass a focused query plus optional URLs."
            ),
            runnable=web_search_agent,
        ),
        CompiledSubAgent(
            name="discovery-screening",
            description=(
                "Retrieves peer-group screening data.  Use to identify peers "
                "and check whether any peer-group multiples or screens are "
                "commonly applied that aren't captured in the skill set."
            ),
            runnable=discovery_screening_agent,
        ),
    ]


# ── Agent factory ─────────────────────────────────────────────────────────────


async def create_valuation_methodology_agent(
    config: RunnableConfig,
    store: BaseStore | None = None,
):
    """Build the valuation methodology deep agent."""
    subagents = await _build_methodology_subagents(config)
    configuration = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = configuration.get_llm_for_role("orchestrator")
    summariser = configuration.get_summariser()

    builder = (
        MuffinAgentBuilder(primary, name="valuation_methodology")
        .with_system_prompt_template("criteria_analysis/valuation_methodology.jinja")
        .with_fallback_models(*fallbacks)
        .with_short_term_memory()
        .with_subagents(subagents)
        .with_response_format(AutoStrategy(schema=ValuationMethodologyOutput))
    )
    if store is not None:
        builder = builder.with_store(store)
    if summariser is not None:
        builder = builder.with_tool_knowledge(summariser)
    return builder.build_deep_agent()


# ── Node ──────────────────────────────────────────────────────────────────────


async def valuation_methodology_node(
    state: ValuationMethodologyInputState,
    config: RunnableConfig,
    *,
    store: BaseStore | None = None,
) -> dict[str, Any]:
    """Stage 3: research the valuation methodology and surface extra criteria.

    Runs in parallel with Stage 2 (``criteria_definition_node``).  Output
    written to ``valuation_methodology`` state key.
    """
    return await invoke_structured_agent(
        state=dict(state),
        config=config,
        agent_factory=create_valuation_methodology_agent,
        input_state_type=ValuationMethodologyInputState,
        state_key="valuation_methodology",
        error_fallback={"additional_criteria": []},
        store=store,
    )
