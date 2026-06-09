"""Valuation specialist — compiled subgraph (collect → compute).

ReAct ``collect_data`` extracts the line items + metrics the four valuation
methods need into a typed :class:`ValuationRawData`; the deterministic
``compute_valuation_signal`` node scores them via the package-local
``tools.valuation_signal.score_valuation_signals`` (no LLM verdict).

Upstream reference: ``ai-hedge-fund/src/agents/valuation.py``.
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
from ..schemas import AnalystSignal, InvestmentSignal
from ..tools.valuation_signal import ValuationResult, score_valuation_signals

logger = logging.getLogger(__name__)
_RETRY = RetryPolicy(max_attempts=2)

_MCP_TOOLS = [
    "equity_fundamental_metrics",
    "equity_fundamental_income",
    "equity_fundamental_balance",
    "equity_fundamental_cash",
    "equity_fundamental_ratios",
    "equity_historical_market_cap",
]


# ── Evidence + signal ─────────────────────────────────────────────────────────


class ValuationEvidence(BaseModel):
    weighted_gap: float
    market_cap: float
    methods: dict[str, Any]


class ValuationSignal(AnalystSignal):
    agent_id: Literal["valuation"] = Field(default="valuation")
    evidence: ValuationEvidence


# ── RawData ───────────────────────────────────────────────────────────────────


class ValuationRawData(BaseModel):
    """Line items + metrics for the four valuation methods.

    Histories are **oldest → newest**; ``*_latest`` are single scalars.
    """

    market_cap: float | None = None
    net_income_latest: float | None = None
    depreciation_latest: float | None = None
    capital_expenditure_latest: float | None = Field(
        default=None,
        description="Latest capex as a POSITIVE number (abs of cash outflow).",
    )
    working_capital_history: list[float | None] = Field(
        default_factory=list,
        description="Working capital, oldest -> newest (for the ΔWC adjustment).",
    )
    earnings_growth: float | None = None
    revenue_growth: float | None = None
    free_cash_flow_history: list[float | None] = Field(default_factory=list)
    total_debt_latest: float | None = None
    cash_latest: float | None = None
    interest_coverage_latest: float | None = None
    debt_to_equity_latest: float | None = None
    enterprise_value_latest: float | None = None
    ev_to_ebitda_history: list[float | None] = Field(default_factory=list)
    price_to_book_ratio_latest: float | None = None
    book_value_growth: float | None = None


# ── State ─────────────────────────────────────────────────────────────────────


class ValuationInput(TypedDict, total=False):
    ticker: str
    as_of_date: str
    query: str | None


class ValuationOutput(TypedDict, total=False):
    persona_signals: list[dict[str, Any]]


class ValuationState(AgentState):
    ticker: Annotated[str, OmitFromSchema(input=False, output=True)]
    as_of_date: Annotated[str, OmitFromSchema(input=False, output=True)]
    query: Annotated[str | None, OmitFromSchema(input=False, output=True)]
    market_cap: Annotated[float | None, OmitFromSchema(input=True, output=False)]
    net_income_latest: Annotated[float | None, OmitFromSchema(input=True, output=False)]
    depreciation_latest: Annotated[
        float | None, OmitFromSchema(input=True, output=False)
    ]
    capital_expenditure_latest: Annotated[
        float | None, OmitFromSchema(input=True, output=False)
    ]
    working_capital_history: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    earnings_growth: Annotated[float | None, OmitFromSchema(input=True, output=False)]
    revenue_growth: Annotated[float | None, OmitFromSchema(input=True, output=False)]
    free_cash_flow_history: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    total_debt_latest: Annotated[float | None, OmitFromSchema(input=True, output=False)]
    cash_latest: Annotated[float | None, OmitFromSchema(input=True, output=False)]
    interest_coverage_latest: Annotated[
        float | None, OmitFromSchema(input=True, output=False)
    ]
    debt_to_equity_latest: Annotated[
        float | None, OmitFromSchema(input=True, output=False)
    ]
    enterprise_value_latest: Annotated[
        float | None, OmitFromSchema(input=True, output=False)
    ]
    ev_to_ebitda_history: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    price_to_book_ratio_latest: Annotated[
        float | None, OmitFromSchema(input=True, output=False)
    ]
    book_value_growth: Annotated[float | None, OmitFromSchema(input=True, output=False)]
    persona_signals: Annotated[list[dict], OmitFromSchema(input=True, output=False)]


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


def compute_valuation_signal_node(state: ValuationState) -> dict[str, Any]:
    """Pure-Python four-method weighted valuation scoring (no LLM)."""
    wc_hist = [v for v in (state.get("working_capital_history") or []) if v is not None]
    wc_change = (wc_hist[-1] - wc_hist[-2]) if len(wc_hist) >= 2 else 0.0
    capex = state.get("capital_expenditure_latest")
    result: ValuationResult = score_valuation_signals(
        market_cap=state.get("market_cap"),
        net_income=state.get("net_income_latest"),
        depreciation=state.get("depreciation_latest"),
        capital_expenditure=abs(capex) if capex is not None else None,
        working_capital_change=wc_change,
        earnings_growth=state.get("earnings_growth"),
        revenue_growth=state.get("revenue_growth"),
        free_cash_flow_history=state.get("free_cash_flow_history"),
        total_debt=state.get("total_debt_latest"),
        cash=state.get("cash_latest"),
        interest_coverage=state.get("interest_coverage_latest"),
        debt_to_equity=state.get("debt_to_equity_latest"),
        enterprise_value=state.get("enterprise_value_latest"),
        ev_to_ebitda_history=state.get("ev_to_ebitda_history"),
        price_to_book_ratio=state.get("price_to_book_ratio_latest"),
        book_value_growth=state.get("book_value_growth"),
    )
    rating = _to_5tier(result["signal"], result["confidence"])
    reasoning = (
        f"Valuation {result['signal']} (weighted gap {result['weighted_gap']:+.1%}, "
        f"conf {result['confidence']:.2f}) vs market cap "
        f"${result['market_cap']:,.0f}"
    )
    sig = ValuationSignal(
        agent_id="valuation",
        signal=rating,
        confidence=min(result["confidence"], 1.0),
        reasoning=reasoning,
        evidence=ValuationEvidence(
            weighted_gap=result["weighted_gap"],
            market_cap=result["market_cap"],
            methods=result["methods"],
        ),
    )
    return {"persona_signals": [sig.model_dump()]}


# ── Data-collection sub-agent + subgraph builder ──────────────────────────────


async def _build_data_collection_agent(config: RunnableConfig) -> CompiledStateGraph:
    cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = cfg.get_llm_for_role("collector")
    mcp_tools = await get_tools(config, _MCP_TOOLS)
    builder = (
        MuffinAgentBuilder(primary, name="valuation_data_collection")
        .with_fallback_models(*fallbacks)
        .with_state_schema(ValuationState)
        .with_runtime_system_prompt_template(
            "specialists/valuation_data_collection.jinja"
        )
        .with_response_format(ValuationRawData)
        .with_model_call_limit(run_limit=10, exit_behavior="end")
    )
    for tool in mcp_tools:
        builder = builder.with_tool(tool, run_limit=2)
    builder = builder.with_tool(execute_python, is_cacheable=False)
    return builder.build_react_agent()


async def build_valuation_analysis_agent(config: RunnableConfig) -> CompiledStateGraph:
    """Build the valuation specialist subgraph (ReAct collect → det. compute)."""
    data_agent = await _build_data_collection_agent(config)
    graph = StateGraph(
        ValuationState,
        input_schema=ValuationInput,
        output_schema=ValuationOutput,
    )
    graph.add_node(
        "collect_data",
        data_agent,
        input_schema=ValuationInput,
        retry_policy=_RETRY,
    )
    graph.add_node("compute_valuation_signal", compute_valuation_signal_node)
    graph.add_edge(START, "collect_data")
    graph.add_edge("collect_data", "compute_valuation_signal")
    graph.add_edge("compute_valuation_signal", END)
    return graph.compile()
