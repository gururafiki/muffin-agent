"""Fundamentals specialist — compiled subgraph (collect → compute).

Two-node :class:`StateGraph`:

1. ``collect_data`` — compiled ReAct sub-agent (``MuffinAgentBuilder``) that
   pulls the latest financial-metrics snapshot from OpenBB MCP and extracts a
   typed :class:`FundamentalsRawData` structured response.
2. ``compute_fundamentals_signal`` — deterministic Python (no LLM) via
   the package-local ``tools.fundamentals.score_fundamentals``, emitting the
   shared ``AnalystSignal`` contract.

The LLM is used only for reliable *extraction* of OpenBB ratio fields; the
scoring is fully mechanical, matching ai-hedge-fund's deterministic
``fundamentals_analyst_agent``.
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
from ..tools.fundamentals import FundamentalsResult, score_fundamentals

logger = logging.getLogger(__name__)
_RETRY = RetryPolicy(max_attempts=2)

_MCP_TOOLS = ["equity_fundamental_metrics", "equity_fundamental_ratios"]


# ── Evidence + signal ─────────────────────────────────────────────────────────


class FundamentalsEvidence(BaseModel):
    profitability: dict[str, Any]
    growth: dict[str, Any]
    financial_health: dict[str, Any]
    price_ratios: dict[str, Any]


class FundamentalsSignal(AnalystSignal):
    agent_id: Literal["fundamentals"] = Field(default="fundamentals")
    evidence: FundamentalsEvidence


# ── RawData ───────────────────────────────────────────────────────────────────


class FundamentalsRawData(BaseModel):
    """Latest financial-metrics snapshot (all decimals, e.g. 0.18 for 18%)."""

    return_on_equity: float | None = None
    net_margin: float | None = None
    operating_margin: float | None = None
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    book_value_growth: float | None = None
    current_ratio: float | None = None
    debt_to_equity: float | None = None
    free_cash_flow_per_share: float | None = None
    earnings_per_share: float | None = None
    price_to_earnings_ratio: float | None = None
    price_to_book_ratio: float | None = None
    price_to_sales_ratio: float | None = None


# ── State ─────────────────────────────────────────────────────────────────────


class FundamentalsInput(TypedDict, total=False):
    ticker: str
    as_of_date: str
    query: str | None


class FundamentalsOutput(TypedDict, total=False):
    persona_signals: list[dict[str, Any]]
    tool_runs: list[dict[str, Any]]


class FundamentalsState(AgentState):
    ticker: Annotated[str, OmitFromSchema(input=False, output=True)]
    as_of_date: Annotated[str, OmitFromSchema(input=False, output=True)]
    query: Annotated[str | None, OmitFromSchema(input=False, output=True)]
    return_on_equity: Annotated[float | None, OmitFromSchema(input=True, output=False)]
    net_margin: Annotated[float | None, OmitFromSchema(input=True, output=False)]
    operating_margin: Annotated[float | None, OmitFromSchema(input=True, output=False)]
    revenue_growth: Annotated[float | None, OmitFromSchema(input=True, output=False)]
    earnings_growth: Annotated[float | None, OmitFromSchema(input=True, output=False)]
    book_value_growth: Annotated[float | None, OmitFromSchema(input=True, output=False)]
    current_ratio: Annotated[float | None, OmitFromSchema(input=True, output=False)]
    debt_to_equity: Annotated[float | None, OmitFromSchema(input=True, output=False)]
    free_cash_flow_per_share: Annotated[
        float | None, OmitFromSchema(input=True, output=False)
    ]
    earnings_per_share: Annotated[
        float | None, OmitFromSchema(input=True, output=False)
    ]
    price_to_earnings_ratio: Annotated[
        float | None, OmitFromSchema(input=True, output=False)
    ]
    price_to_book_ratio: Annotated[
        float | None, OmitFromSchema(input=True, output=False)
    ]
    price_to_sales_ratio: Annotated[
        float | None, OmitFromSchema(input=True, output=False)
    ]
    persona_signals: Annotated[list[dict], OmitFromSchema(input=True, output=False)]
    tool_runs: Annotated[list[dict[str, Any]], merge_tool_runs]


# ── Mapping helper ────────────────────────────────────────────────────────────


def _to_5tier(
    tactical_signal: str, confidence: float, strong_threshold: float = 0.7
) -> InvestmentSignal:
    """Convert 3-tier bullish/bearish/neutral + confidence to 5-tier rating."""
    if tactical_signal == "bullish":
        return "strong_buy" if confidence >= strong_threshold else "buy"
    if tactical_signal == "bearish":
        return "strong_sell" if confidence >= strong_threshold else "sell"
    return "hold"


# ── Compute node ──────────────────────────────────────────────────────────────


_METRIC_KEYS = (
    "return_on_equity",
    "net_margin",
    "operating_margin",
    "revenue_growth",
    "earnings_growth",
    "book_value_growth",
    "current_ratio",
    "debt_to_equity",
    "free_cash_flow_per_share",
    "earnings_per_share",
    "price_to_earnings_ratio",
    "price_to_book_ratio",
    "price_to_sales_ratio",
)


def compute_fundamentals_signal_node(state: FundamentalsState) -> dict[str, Any]:
    """Pure-Python four-dimension fundamentals scoring (no LLM)."""
    metrics = {k: state.get(k) for k in _METRIC_KEYS}
    if all(metrics[k] is None for k in _METRIC_KEYS):
        sig = FundamentalsSignal(
            agent_id="fundamentals",
            signal="hold",
            confidence=0.0,
            reasoning="No financial metrics available",
            evidence=FundamentalsEvidence(
                profitability={}, growth={}, financial_health={}, price_ratios={}
            ),
        )
        return {"persona_signals": [sig.model_dump()]}

    result: FundamentalsResult = score_fundamentals(metrics)
    rating = _to_5tier(result["signal"], result["confidence"])
    reasoning = (
        f"Fundamentals {result['signal']} (conf {result['confidence']:.2f}); "
        f"profitability {result['profitability']['signal']}, "
        f"growth {result['growth']['signal']}, "
        f"health {result['financial_health']['signal']}, "
        f"price ratios {result['price_ratios']['signal']}"
    )
    sig = FundamentalsSignal(
        agent_id="fundamentals",
        signal=rating,
        confidence=min(result["confidence"], 1.0),
        reasoning=reasoning,
        evidence=FundamentalsEvidence(
            profitability=result["profitability"],
            growth=result["growth"],
            financial_health=result["financial_health"],
            price_ratios=result["price_ratios"],
        ),
    )
    return {"persona_signals": [sig.model_dump()]}


# ── Data-collection sub-agent + subgraph builder ──────────────────────────────


async def _build_data_collection_agent(config: RunnableConfig) -> CompiledStateGraph:
    cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = cfg.get_llm_for_role("collector")
    mcp_tools = await get_tools(config, _MCP_TOOLS)
    builder = (
        MuffinAgentBuilder(primary, name="fundamentals_data_collection")
        .with_fallback_models(*fallbacks)
        .with_state_schema(FundamentalsState)
        .with_input_prompt_template(
            "personas_council/specialists/fundamentals_data_collection.jinja"
        )
        .with_response_format(FundamentalsRawData)
        .with_model_call_limit(run_limit=6, exit_behavior="end")
    )
    for tool in mcp_tools:
        builder = builder.with_tool(tool, run_limit=2)
    builder = builder.with_tool(execute_python, is_cacheable=False)
    return builder.build_react_agent()


async def build_fundamentals_analysis_agent(
    config: RunnableConfig,
) -> CompiledStateGraph:
    """Build the fundamentals specialist subgraph (ReAct collect → det. compute)."""
    data_agent = await _build_data_collection_agent(config)
    graph = StateGraph(
        FundamentalsState,
        input_schema=FundamentalsInput,
        output_schema=FundamentalsOutput,
    )
    graph.add_node(
        "collect_data",
        data_agent,
        input_schema=FundamentalsInput,
        retry_policy=_RETRY,
    )
    graph.add_node("compute_fundamentals_signal", compute_fundamentals_signal_node)
    graph.add_edge(START, "collect_data")
    graph.add_edge("collect_data", "compute_fundamentals_signal")
    graph.add_edge("compute_fundamentals_signal", END)
    return graph.compile()
