"""Market Analyst — technical analysis via OpenBB OHLCV + ``get_indicators``."""

from __future__ import annotations

from typing import Annotated

from langchain.agents import AgentState
from langchain.agents.middleware.types import OmitFromSchema
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from ....model_config import ModelConfiguration
from ....utils.agent_builder import MuffinAgentBuilder
from ...data_collection.utils import get_tools
from ..tools import get_indicators

_MCP_TOOLS = [
    "equity_price_historical",
    "equity_price_quote",
    "equity_price_performance",
    "equity_historical_market_cap",
]


class MarketAnalystOutput(BaseModel):
    """Structured output for the Market Analyst.

    Field name matches the state-schema output field so
    ``MuffinAgentBuilder`` auto-unpacks the structured response into
    parent state.
    """

    market_report: str = Field(
        description=(
            "Markdown technical-analysis report covering selected indicators "
            "(trend / momentum / volatility / volume), price action, and an "
            "end-of-report Markdown summary table of the key readings."
        )
    )


class MarketAnalystState(AgentState):
    """State schema for the Market Analyst.

    Inherits ``messages`` (and ``structured_response``) from
    ``AgentState``. Adds the parent-state fields the analyst reads
    (input-only) and writes (output-only) via ``OmitFromSchema``.
    """

    ticker: Annotated[str, OmitFromSchema(input=False, output=True)]
    decision_date: Annotated[str, OmitFromSchema(input=False, output=True)]
    market_report: Annotated[str, OmitFromSchema(input=True, output=False)]


async def build_market_analyst_agent(
    config: RunnableConfig,
) -> CompiledStateGraph:
    """Build the Market Analyst as a compiled ReAct agent.

    Add directly to a parent graph via ``parent.add_node("market_analyst", agent,
    input_schema=AnalystInput)`` (an explicit ``{ticker, decision_date}`` schema —
    NOT ``agent.input_schema``, a property-less ``RootModel`` that maps ``{}`` and
    raises at coercion). The parent state must declare ``ticker`` and
    ``decision_date`` (read by the analyst) and ``market_report`` (written by it).
    """
    cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = cfg.get_llm_for_role("collector")
    mcp_tools = await get_tools(config, _MCP_TOOLS)

    builder = (
        MuffinAgentBuilder(primary, name="market_analyst")
        .with_fallback_models(*fallbacks)
        .with_state_schema(MarketAnalystState)
        .with_input_prompt_template(
            "trading_decision/analysts/market.jinja",
        )
        .with_response_format(MarketAnalystOutput)
    )
    for tool in mcp_tools:
        builder = builder.with_tool(tool, run_limit=4)
    builder = builder.with_tool(get_indicators, run_limit=10)
    return builder.build_react_agent()
