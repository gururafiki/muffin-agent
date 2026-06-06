"""Shared data-collection step for the persona council.

One deep agent fetches everything every persona needs and packs it into a
``PersonaDataBundle`` via structured output.  Reused by:

* ``persona_data_collection_node`` (used by the council graph and the
  standalone single-persona graph as the first node)
* future external graphs that want personas as a sub-step

Built on the same ``MuffinAgentBuilder`` pattern as the investment-stage
deep agents (e.g. ``company_analysis``) — uses 4 data subagents
(equity-fundamentals, equity-price, equity-ownership, news) + a structured
``response_format`` enforced via ``AutoStrategy(PersonaDataBundle)``.
"""

from __future__ import annotations

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
    create_equity_fundamentals_data_collection_agent,
    create_equity_ownership_data_collection_agent,
    create_equity_price_data_collection_agent,
    create_news_data_collection_agent,
)
from .data import PersonaDataBundle

logger = logging.getLogger(__name__)


class PersonaDataCollectionInputState(TypedDict, total=False):
    """State keys read by ``persona_data_collection_node``.

    Both fields optional.  ``as_of_date`` defaults to today inside the
    agent prompt when missing.  ``benchmark`` defaults to SPY.
    """

    ticker: str
    as_of_date: str
    benchmark: str


class PersonaDataCollectionOutputState(TypedDict, total=False):
    """State keys written by ``persona_data_collection_node``."""

    data_bundle: dict[str, Any]
    """``PersonaDataBundle.model_dump()`` on success; error dict
    (``{"error": ..., "raw_output": ...}``) on failure.  Council /
    standalone graphs read this directly into their downstream nodes."""


async def _build_persona_data_subagents(
    config: RunnableConfig,
) -> list[CompiledSubAgent]:
    """Build the 4 data subagents needed to populate ``PersonaDataBundle``.

    Deliberately minimal — only the four data domains every persona needs.
    No validation subagent (the data-collection agent has a fixed schema
    to populate, so it knows what is missing without LLM validation).
    """
    fundamentals_agent = await create_equity_fundamentals_data_collection_agent(config)
    price_agent = await create_equity_price_data_collection_agent(config)
    ownership_agent = await create_equity_ownership_data_collection_agent(config)
    news_agent = await create_news_data_collection_agent(config)

    return [
        CompiledSubAgent(
            name="equity-fundamentals",
            description=(
                "Retrieves financial statements (income, balance, cash flow), "
                "key ratios (ROE, ROIC, D/E, margins), per-share metrics, and "
                "ESG scores. Used to populate `PersonaDataBundle.line_items` "
                "and `financial_metrics`."
            ),
            runnable=fundamentals_agent,
        ),
        CompiledSubAgent(
            name="equity-price",
            description=(
                "Retrieves daily OHLCV price bars for the ticker AND the "
                "benchmark (e.g. SPY) over the trailing 12 months, plus "
                "monthly market-cap history over 5 years. Used to populate "
                "`PersonaDataBundle.prices_1y`, `benchmark_prices_1y`, "
                "and `market_cap_history`."
            ),
            runnable=price_agent,
        ),
        CompiledSubAgent(
            name="equity-ownership",
            description=(
                "Retrieves insider trading activity for the trailing 12 "
                "months via `equity_ownership_insider_trading`. Used to "
                "populate `PersonaDataBundle.insider_trades` with signed "
                "`transaction_shares` (positive buy / negative sell)."
            ),
            runnable=ownership_agent,
        ),
        CompiledSubAgent(
            name="news",
            description=(
                "Retrieves company-specific news headlines and articles "
                "via `news_company` for the trailing 12 months. Used to "
                "populate `PersonaDataBundle.company_news`. Pass through "
                "any provider-supplied `sentiment` field."
            ),
            runnable=news_agent,
        ),
    ]


async def create_persona_data_collection_agent(
    config: RunnableConfig,
    store: BaseStore | None = None,
):
    """Build the persona data-collection deep agent.

    Returns a compiled deep agent whose structured response is a
    ``PersonaDataBundle``.  Mirrors the per-stage agent shape used
    elsewhere in muffin (``create_company_analysis_agent`` etc.) so it
    integrates cleanly with ``run_deep_agent_node`` / similar helpers.
    """
    subagents = await _build_persona_data_subagents(config)
    configuration = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = configuration.get_llm_for_role("orchestrator")
    summariser = configuration.get_summariser()

    builder = (
        MuffinAgentBuilder(primary, name="persona_data_collection")
        .with_system_prompt_template("personas/data_collection.jinja")
        .with_fallback_models(*fallbacks)
        .with_sandbox()
        .with_short_term_memory()
        .with_subagents(subagents)
        .with_response_format(AutoStrategy(schema=PersonaDataBundle))
        .with_store(store)
    )
    if summariser is not None:
        builder = builder.with_tool_knowledge(summariser)
    return builder.build_deep_agent()


async def persona_data_collection_node(
    state: PersonaDataCollectionInputState,
    config: RunnableConfig,
    *,
    store: BaseStore | None = None,
) -> PersonaDataCollectionOutputState:
    """Run the persona data-collection deep agent and write ``data_bundle`` to state.

    Reads ``ticker``, optional ``as_of_date``, optional ``benchmark`` from
    state.  Builds a JSON context dict from the present keys, invokes the
    agent, and returns ``{"data_bundle": <PersonaDataBundle dump or error dict>}``.

    The downstream persona nodes (council fan-out workers + single-persona
    standalone) read ``state["data_bundle"]`` directly.
    """
    try:
        agent = await create_persona_data_collection_agent(config, store=store)
        context = {
            k: state[k]
            for k in PersonaDataCollectionInputState.__annotations__
            if state.get(k)
        }
        result = await agent.ainvoke({"input": json.dumps(context)})
        structured = (
            result.get("structured_response") if isinstance(result, dict) else None
        )
        if structured is None:
            raw = result.get("output", "") if isinstance(result, dict) else str(result)
            return {
                "data_bundle": {
                    "error": "Data-collection agent did not produce structured output",
                    "raw_output": raw,
                }
            }
        return {"data_bundle": structured.model_dump()}
    except Exception:
        logger.exception("persona_data_collection_node failed")
        return {
            "data_bundle": {
                "error": "Data-collection agent raised an exception",
            }
        }
