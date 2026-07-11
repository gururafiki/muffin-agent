"""Stage 5: synthesis.

Reasoning-only ReAct agent (no subagents, no tools, no sandbox) compiled
with a state schema + runtime prompt so it is added directly to the
orchestrator graph as a node.  Reads classification, criteria_definition,
valuation_methodology, merged_criteria, and the per-criterion evaluations
and produces a final ``CriteriaAnalysisSynthesis`` (composite score,
signal, weighted breakdown, thesis paragraph).
"""

from typing import Annotated, Any

from langchain.agents import AgentState
from langchain.agents.middleware.types import OmitFromSchema
from langchain.agents.structured_output import AutoStrategy
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from ...model_config import ModelConfiguration
from ...utils.agent_builder import MuffinAgentBuilder
from .schemas import SynthesisNodeOutput

# ── Agent state schema ────────────────────────────────────────────────────────


class SynthesisAgentState(AgentState):
    """State schema for the Stage 5 synthesis agent as a graph node."""

    ticker: Annotated[str, OmitFromSchema(input=False, output=True)]
    query: Annotated[str, OmitFromSchema(input=False, output=True)]
    classification: Annotated[dict[str, Any], OmitFromSchema(input=False, output=True)]
    criteria_definition: Annotated[
        dict[str, Any], OmitFromSchema(input=False, output=True)
    ]
    valuation_methodology: Annotated[
        dict[str, Any], OmitFromSchema(input=False, output=True)
    ]
    merged_criteria: Annotated[
        list[dict[str, Any]], OmitFromSchema(input=False, output=True)
    ]
    criterion_evaluations: Annotated[
        list[dict[str, Any]], OmitFromSchema(input=False, output=True)
    ]
    synthesis: Annotated[dict[str, Any], OmitFromSchema(input=True, output=False)]


# ── Agent factory ─────────────────────────────────────────────────────────────


async def create_synthesis_agent(
    config: RunnableConfig,
    store: BaseStore | None = None,
):
    """Build the synthesis agent — reasoning only, no subagents/tools."""
    configuration = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = configuration.get_llm_for_role("reasoner")

    builder = (
        MuffinAgentBuilder(primary, name="criteria_analysis_synthesis")
        .with_state_schema(SynthesisAgentState)
        .with_input_prompt_template("criteria_analysis/synthesis.jinja")
        .with_fallback_models(*fallbacks)
        .with_response_format(AutoStrategy(schema=SynthesisNodeOutput))
    )
    if store is not None:
        builder = builder.with_store(store)
    return builder.build_react_agent()
