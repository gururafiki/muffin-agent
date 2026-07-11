"""Stage 1: ticker classification.

Lightweight deep agent compiled with a state schema + runtime prompt so
it is added DIRECTLY to the orchestrator graph as a node (the
analyst/council pattern).  The task context (ticker, query) flows in via
the graph's ``input_schema`` and is rendered into the system prompt per
model call; the structured response unpacks into the ``classification``
channel via the single-field wrapper output model.

The pure :func:`lift_classification_node` runs after the agent (or
instead of it, when the caller pre-supplied the flat classification
keys) to keep ``classification`` and the flat ``sector`` / ``sub_sector``
/ ``market`` / ``stock_type`` keys in sync.
"""

import logging
from typing import Annotated, Any

from deepagents import CompiledSubAgent, DeepAgentState
from langchain.agents.middleware.types import OmitFromSchema
from langchain.agents.structured_output import AutoStrategy
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from ...model_config import ModelConfiguration
from ...utils.agent_builder import MuffinAgentBuilder
from ..data_collection import (
    create_economy_macro_data_collection_agent,
    create_equity_fundamentals_data_collection_agent,
    create_etf_index_data_collection_agent,
)
from ..subagents import build_validation_subagent
from .schemas import TickerClassificationNodeOutput
from .state import CriteriaAnalysisState

logger = logging.getLogger(__name__)

_FLAT_KEYS = ("sector", "sub_sector", "market", "stock_type")


# ── Agent state schema ────────────────────────────────────────────────────────


class TickerClassificationAgentState(DeepAgentState):
    """State schema for the Stage 1 classification agent.

    Inputs flow IN from the parent graph (``OmitFromSchema(output=True)``
    keeps them out of the node's output); ``classification`` is written by
    the structured-response unpacker and flows OUT to the parent channel.
    """

    ticker: Annotated[str, OmitFromSchema(input=False, output=True)]
    query: Annotated[str, OmitFromSchema(input=False, output=True)]
    classification: Annotated[dict[str, Any], OmitFromSchema(input=True, output=False)]


# ── Subagent builder ──────────────────────────────────────────────────────────


async def _build_classification_subagents(
    config: RunnableConfig,
) -> list[CompiledSubAgent]:
    """Build the trimmed subagent set for ticker classification."""
    etf_index_agent = await create_etf_index_data_collection_agent(config)
    equity_fundamentals_agent = await create_equity_fundamentals_data_collection_agent(
        config
    )
    economy_macro_agent = await create_economy_macro_data_collection_agent(config)
    validation_subagent = await build_validation_subagent(config)

    return [
        CompiledSubAgent(
            name="etf-index",
            description=(
                "Retrieves ETF and index data.  Call ``etf_equity_exposure`` to "
                "determine the ticker's sector, industry, and style classification "
                "(growth vs value)."
            ),
            runnable=etf_index_agent,
        ),
        CompiledSubAgent(
            name="equity-fundamentals",
            description=(
                "Retrieves fundamental ratios (ROE, margins, growth, P/E, P/B, "
                "FCF yield).  Use to confirm value vs growth classification when "
                "ETF exposure is ambiguous."
            ),
            runnable=equity_fundamentals_agent,
        ),
        CompiledSubAgent(
            name="economy-macro",
            description=(
                "Retrieves macro context (GDP, rates, inflation).  Use to "
                "determine if the ticker's primary revenue geography is a "
                "developed or emerging market."
            ),
            runnable=economy_macro_agent,
        ),
        validation_subagent,
    ]


# ── Agent factory ─────────────────────────────────────────────────────────────


async def create_ticker_classification_agent(
    config: RunnableConfig,
    store: BaseStore | None = None,
):
    """Build the ticker classification deep agent (compiled graph node)."""
    subagents = await _build_classification_subagents(config)
    configuration = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = configuration.get_llm_for_role("orchestrator")
    summariser = configuration.get_summariser()

    builder = (
        MuffinAgentBuilder(primary, name="ticker_classification")
        .with_state_schema(TickerClassificationAgentState)
        .with_input_prompt_template(
            "criteria_analysis/ticker_classification.jinja"
        )
        .with_fallback_models(*fallbacks)
        .with_short_term_memory()
        .with_subagents(subagents)
        .with_response_format(AutoStrategy(schema=TickerClassificationNodeOutput))
    )
    if store is not None:
        builder = builder.with_store(store)
    if summariser is not None:
        builder = builder.with_tool_knowledge(summariser)
    return builder.build_deep_agent()


# ── Pure graph nodes / routing ────────────────────────────────────────────────


def route_classification_entry(state: CriteriaAnalysisState) -> str:
    """Route START: skip the LLM when all flat keys were pre-supplied."""
    if all(state.get(k) for k in ("sector", "market", "stock_type")):
        return "lift_classification"
    return "ticker_classification"


def lift_classification_node(state: CriteriaAnalysisState) -> dict[str, Any]:
    """Keep ``classification`` and the flat keys consistent.

    Two entry paths:

    * After the classification agent ran — lift the flat keys OUT of the
      ``classification`` payload (read downstream by
      ``SkillFilterMiddleware[TickerClassification]``).
    * Pre-supplied short-circuit (CLI flags) — assemble the
      ``classification`` payload FROM the flat keys without an LLM call.
    """
    classification = state.get("classification") or {}
    if classification:
        update: dict[str, Any] = {}
        for key in _FLAT_KEYS:
            value = classification.get(key)
            if value is not None:
                update[key] = value
        return update

    if all(state.get(k) for k in ("sector", "market", "stock_type")):
        return {
            "classification": {
                "ticker": state.get("ticker", ""),
                "sector": state["sector"],
                "sub_sector": state.get("sub_sector"),
                "market": state["market"],
                "stock_type": state["stock_type"],
                "rationale": "Pre-supplied via input state — classification skipped.",
                "confidence": 1.0,
                "data_sources": [],
                "limitations": [],
            }
        }

    raise ValueError(
        "ticker_classification produced no classification and no flat keys "
        "were pre-supplied"
    )
