"""Growth specialist — compiled subgraph (collect → compute).

ReAct ``collect_data`` extracts growth/margin history + valuation ratios +
insider trades into a typed :class:`GrowthRawData`; the deterministic
``compute_growth_signal`` node scores them via the package-local
``tools.growth.score_growth_signals`` (no LLM verdict).

Upstream reference: ``ai-hedge-fund/src/agents/growth_agent.py``.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

from langchain.agents import AgentState
from langchain.agents.middleware.types import OmitFromSchema
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from ....model_config import ModelConfiguration
from ....sandbox.tools import execute_python
from ....utils.agent_builder import MuffinAgentBuilder
from ...data_collection.utils import get_tools
from ..schemas import AnalystSignal, InvestmentSignal, merge_tool_runs
from ..tools.growth import GrowthResult, score_growth_signals

logger = logging.getLogger(__name__)
_RETRY = RetryPolicy(max_attempts=2)

_MCP_TOOLS = [
    "equity_fundamental_metrics",
    "equity_fundamental_ratios",
    "equity_ownership_insider_trading",
]


# ── Evidence + signal ─────────────────────────────────────────────────────────


class GrowthEvidence(BaseModel):
    weighted_score: float
    growth_trends: dict[str, Any]
    valuation: dict[str, Any]
    margins: dict[str, Any]
    insider: dict[str, Any]
    health: dict[str, Any]


class GrowthSignal(AnalystSignal):
    agent_id: Literal["growth"] = Field(default="growth")
    evidence: GrowthEvidence


# ── RawData ───────────────────────────────────────────────────────────────────


class GrowthRawData(BaseModel):
    """Growth/margin history + latest valuation ratios + insider trades.

    All time series are **oldest → newest** (latest = last element).
    """

    revenue_growth_history: list[float | None] = Field(default_factory=list)
    eps_growth_history: list[float | None] = Field(default_factory=list)
    fcf_growth_history: list[float | None] = Field(default_factory=list)
    gross_margin_history: list[float | None] = Field(default_factory=list)
    operating_margin_history: list[float | None] = Field(default_factory=list)
    net_margin_history: list[float | None] = Field(default_factory=list)
    peg_ratio: float | None = None
    price_to_sales_ratio: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    insider_trades: list[dict[str, Any]] = Field(default_factory=list)


# ── State ─────────────────────────────────────────────────────────────────────


class GrowthInput(TypedDict, total=False):
    ticker: str
    as_of_date: str
    query: str | None


class GrowthOutput(TypedDict, total=False):
    persona_signals: list[dict[str, Any]]
    tool_runs: list[dict[str, Any]]


class GrowthState(AgentState):
    ticker: Annotated[str, OmitFromSchema(input=False, output=True)]
    as_of_date: Annotated[str, OmitFromSchema(input=False, output=True)]
    query: Annotated[str | None, OmitFromSchema(input=False, output=True)]
    revenue_growth_history: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    eps_growth_history: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    fcf_growth_history: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    gross_margin_history: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    operating_margin_history: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    net_margin_history: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    peg_ratio: Annotated[float | None, OmitFromSchema(input=True, output=False)]
    price_to_sales_ratio: Annotated[
        float | None, OmitFromSchema(input=True, output=False)
    ]
    debt_to_equity: Annotated[float | None, OmitFromSchema(input=True, output=False)]
    current_ratio: Annotated[float | None, OmitFromSchema(input=True, output=False)]
    insider_trades: Annotated[
        list[dict[str, Any]] | None, OmitFromSchema(input=True, output=False)
    ]
    persona_signals: Annotated[list[dict], OmitFromSchema(input=True, output=False)]
    tool_runs: Annotated[list[dict[str, Any]], merge_tool_runs]


# ── Mapping helper ────────────────────────────────────────────────────────────


def _to_5tier(
    tactical_signal: str, confidence: float, strong_threshold: float = 0.7
) -> InvestmentSignal:
    if tactical_signal == "bullish":
        return "strong_buy" if confidence >= strong_threshold else "buy"
    if tactical_signal == "bearish":
        return "strong_sell" if confidence >= strong_threshold else "sell"
    return "hold"


# ── Compute node ──────────────────────────────────────────────────────────────


def compute_growth_signal_node(state: GrowthState) -> dict[str, Any]:
    """Pure-Python five-dimension weighted growth scoring (no LLM)."""
    result: GrowthResult = score_growth_signals(
        state.get("revenue_growth_history"),
        state.get("eps_growth_history"),
        state.get("fcf_growth_history"),
        state.get("gross_margin_history"),
        state.get("operating_margin_history"),
        state.get("net_margin_history"),
        state.get("peg_ratio"),
        state.get("price_to_sales_ratio"),
        state.get("debt_to_equity"),
        state.get("current_ratio"),
        state.get("insider_trades"),
    )
    rating = _to_5tier(result["signal"], result["confidence"])
    reasoning = (
        f"Growth {result['signal']} (weighted {result['weighted_score']:.2f}); "
        f"growth-trend {result['growth_trends']['score']:.2f}, "
        f"valuation {result['valuation']['score']:.2f}, "
        f"margins {result['margins']['score']:.2f}, "
        f"insider {result['insider']['score']:.2f}, "
        f"health {result['health']['score']:.2f}"
    )
    sig = GrowthSignal(
        agent_id="growth",
        signal=rating,
        confidence=min(result["confidence"], 1.0),
        reasoning=reasoning,
        evidence=GrowthEvidence(
            weighted_score=result["weighted_score"],
            growth_trends=result["growth_trends"],
            valuation=result["valuation"],
            margins=result["margins"],
            insider=result["insider"],
            health=result["health"],
        ),
    )
    return {"persona_signals": [sig.model_dump()]}


# ── Data-collection sub-agent + subgraph builder ──────────────────────────────


async def _build_data_collection_agent(config: RunnableConfig) -> CompiledStateGraph:
    cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = cfg.get_llm_for_role("collector")
    mcp_tools = await get_tools(config, _MCP_TOOLS)
    builder = (
        MuffinAgentBuilder(primary, name="growth_data_collection")
        .with_fallback_models(*fallbacks)
        .with_state_schema(GrowthState)
        .with_input_prompt_template("personas_council/specialists/growth_data_collection.jinja")
        .with_response_format(GrowthRawData)
        .with_model_call_limit(run_limit=8, exit_behavior="end")
    )
    for tool in mcp_tools:
        builder = builder.with_tool(tool, run_limit=2)
    builder = builder.with_tool(execute_python, is_cacheable=False)
    return builder.build_react_agent()


async def build_growth_analysis_agent(config: RunnableConfig) -> CompiledStateGraph:
    """Build the growth specialist subgraph (ReAct collect → det. compute)."""
    data_agent = await _build_data_collection_agent(config)
    graph = StateGraph(
        GrowthState,
        input_schema=GrowthInput,
        output_schema=GrowthOutput,
    )
    graph.add_node(
        "collect_data",
        data_agent,
        input_schema=GrowthInput,
        retry_policy=_RETRY,
    )
    graph.add_node("compute_growth_signal", compute_growth_signal_node)
    graph.add_edge(START, "collect_data")
    graph.add_edge("collect_data", "compute_growth_signal")
    graph.add_edge("compute_growth_signal", END)
    return graph.compile()
