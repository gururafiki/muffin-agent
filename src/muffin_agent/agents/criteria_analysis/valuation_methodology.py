"""Stage 3: valuation methodology research.

Deep agent that performs web research to surface (a) the canonical
valuation methodology the analyst community applies to this ticker and
(b) any criteria the skill-filtered Stage 2 stream would not capture
(recent sell-side debate, narrative drivers, idiosyncratic governance
issues).

Compiled with a state schema + runtime prompt so it is added directly to
the orchestrator graph as a node.  Subagents: ``web-search``
(Firecrawl-backed ReAct agent) plus ``discovery-screening`` for
peer-comparison context.
"""

from typing import Annotated, Any

from deepagents import CompiledSubAgent, DeepAgentState
from langchain.agents.middleware.types import OmitFromSchema
from langchain.agents.structured_output import AutoStrategy
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from ...model_config import ModelConfiguration
from ...utils.agent_builder import MuffinAgentBuilder
from ..data_collection import (
    create_discovery_screening_data_collection_agent,
    create_web_search_data_collection_agent,
)
from .schemas import ValuationMethodologyNodeOutput

# ── Agent state schema ────────────────────────────────────────────────────────


class ValuationMethodologyAgentState(DeepAgentState):
    """State schema for the Stage 3 methodology agent as a graph node."""

    ticker: Annotated[str, OmitFromSchema(input=False, output=True)]
    query: Annotated[str, OmitFromSchema(input=False, output=True)]
    sector: Annotated[str, OmitFromSchema(input=False, output=True)]
    sub_sector: Annotated[str, OmitFromSchema(input=False, output=True)]
    market: Annotated[str, OmitFromSchema(input=False, output=True)]
    stock_type: Annotated[str, OmitFromSchema(input=False, output=True)]
    classification: Annotated[dict[str, Any], OmitFromSchema(input=False, output=True)]
    valuation_methodology: Annotated[
        dict[str, Any], OmitFromSchema(input=True, output=False)
    ]


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
    """Build the valuation methodology deep agent (compiled graph node)."""
    subagents = await _build_methodology_subagents(config)
    configuration = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = configuration.get_llm_for_role("orchestrator")
    summariser = configuration.get_summariser()

    builder = (
        MuffinAgentBuilder(primary, name="valuation_methodology")
        .with_state_schema(ValuationMethodologyAgentState)
        .with_input_prompt_template(
            "criteria_analysis/valuation_methodology.jinja"
        )
        .with_fallback_models(*fallbacks)
        .with_short_term_memory()
        .with_subagents(subagents)
        .with_response_format(AutoStrategy(schema=ValuationMethodologyNodeOutput))
    )
    if store is not None:
        builder = builder.with_store(store)
    if summariser is not None:
        builder = builder.with_tool_knowledge(summariser)
    return builder.build_deep_agent()
