"""Stage 1: ticker classification.

Lightweight deep agent that produces sector / sub_sector / market /
stock_type for the orchestrator's downstream stages.  When the caller
already supplies all four flat keys (CLI flags), the node short-circuits
and skips the LLM entirely.
"""

import json
import logging
from typing import Any

from deepagents import CompiledSubAgent
from langchain.agents.structured_output import AutoStrategy
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from typing_extensions import TypedDict

from ...model_config import ModelConfiguration
from ...utils.agent_builder import MuffinAgentBuilder
from ..data_collection import (
    create_economy_macro_data_collection_agent,
    create_equity_fundamentals_data_collection_agent,
    create_etf_index_data_collection_agent,
)
from ..subagents import build_validation_subagent
from .schemas import TickerClassificationOutput

logger = logging.getLogger(__name__)

# ── Input state schema ────────────────────────────────────────────────────────


class TickerClassificationInputState(TypedDict, total=False):
    """Fields read by ``ticker_classification_node`` from the outer state."""

    ticker: str
    query: str
    sector: str
    sub_sector: str
    market: str
    stock_type: str


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
    """Build the ticker classification deep agent."""
    subagents = await _build_classification_subagents(config)
    configuration = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = configuration.get_llm_for_role("orchestrator")
    summariser = configuration.get_summariser()

    builder = (
        MuffinAgentBuilder(primary, name="ticker_classification")
        .with_system_prompt_template("criteria_analysis/ticker_classification.jinja")
        .with_fallback_models(*fallbacks)
        .with_short_term_memory()
        .with_subagents(subagents)
        .with_response_format(AutoStrategy(schema=TickerClassificationOutput))
    )
    if store is not None:
        builder = builder.with_store(store)
    if summariser is not None:
        builder = builder.with_tool_knowledge(summariser)
    return builder.build_deep_agent()


# ── Node ──────────────────────────────────────────────────────────────────────


_FLAT_KEYS = ("sector", "sub_sector", "market", "stock_type")


def _shortcircuit(state: dict[str, Any]) -> dict[str, Any] | None:
    """Return a fully-formed state update if the caller pre-supplied all flat keys."""
    if all(state.get(k) for k in ("sector", "market", "stock_type")):
        classification = {
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
        return {
            "sector": state["sector"],
            "sub_sector": state.get("sub_sector"),
            "market": state["market"],
            "stock_type": state["stock_type"],
            "classification": classification,
        }
    return None


async def ticker_classification_node(
    state: TickerClassificationInputState,
    config: RunnableConfig,
    *,
    store: BaseStore | None = None,
) -> dict[str, Any]:
    """Stage 1: classify the ticker.

    Short-circuits when the caller already pre-supplied
    ``sector`` + ``market`` + ``stock_type``.  Otherwise builds a
    classification deep agent, invokes it, and lifts its structured
    output into both the full ``classification`` payload and the four
    flat state keys read by ``SkillFilterMiddleware`` downstream.
    """
    short = _shortcircuit(dict(state))
    if short is not None:
        return short

    fallback: dict[str, Any] = {"classification": {"error": "Classification failed."}}

    try:
        agent = await create_ticker_classification_agent(config, store=store)
        state_dict: dict[str, Any] = dict(state)
        context = {
            k: state_dict[k]
            for k in TickerClassificationInputState.__annotations__
            if state_dict.get(k)
        }
        result = await agent.ainvoke({"input": json.dumps(context)})
        structured = (
            result.get("structured_response") if isinstance(result, dict) else None
        )
        if structured is None:
            raw = result.get("output", "") if isinstance(result, dict) else str(result)
            return {
                "classification": {
                    "error": "Agent did not produce structured output",
                    "raw_output": raw,
                },
            }

        payload = structured.model_dump()
        update: dict[str, Any] = {"classification": payload}
        for key in _FLAT_KEYS:
            value = payload.get(key)
            if value is not None:
                update[key] = value
        return update
    except Exception:
        logger.exception("ticker_classification_node failed")
        return fallback
