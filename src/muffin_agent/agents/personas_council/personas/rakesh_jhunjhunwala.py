"""Rakesh Jhunjhunwala persona — compiled subgraph (collect → compute → verdict).

EM growth + quality-tier DCF (12/15/18% discount based on quality).
See ``warren_buffett.py`` for canonical reference.
Reference: ``ai-hedge-fund/src/agents/rakesh_jhunjhunwala.py``.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal, cast

from langchain.agents import AgentState
from langchain.agents.middleware.types import OmitFromSchema
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from ....model_config import ModelConfiguration
from ....prompts import render_template
from ....sandbox.tools import execute_python
from ....utils.agent_builder import MuffinAgentBuilder
from ...data_collection.utils import get_tools
from ..schemas import AnalystSignal
from ..tools.scoring_helpers import compute_intrinsic_value_exit_multiple

logger = logging.getLogger(__name__)
_LLM_RETRY = RetryPolicy(max_attempts=2)


# ── Typed sub-evidence ────────────────────────────────────────────────────────


class RakeshJhunjhunwalaProfitability(BaseModel):
    return_on_equity: float | None
    operating_margin: float | None
    eps_cagr: float | None
    score: int
    max_score: int
    reasoning: str


class RakeshJhunjhunwalaGrowth(BaseModel):
    revenue_cagr: float | None
    net_income_cagr: float | None
    consistent_ni: bool
    score: int
    max_score: int
    reasoning: str


class RakeshJhunjhunwalaBalanceSheet(BaseModel):
    debt_to_assets: float | None
    current_ratio: float | None
    score: int
    max_score: int
    reasoning: str


class RakeshJhunjhunwalaCashFlow(BaseModel):
    fcf_positive_latest: bool
    pays_dividends: bool
    score: int
    max_score: int
    reasoning: str


class RakeshJhunjhunwalaManagementActions(BaseModel):
    latest_issuance: float | None
    has_buybacks: bool
    score: int
    max_score: int
    reasoning: str


class RakeshJhunjhunwalaEvidence(BaseModel):
    profitability: RakeshJhunjhunwalaProfitability
    growth: RakeshJhunjhunwalaGrowth
    balance_sheet: RakeshJhunjhunwalaBalanceSheet
    cash_flow: RakeshJhunjhunwalaCashFlow
    management_actions: RakeshJhunjhunwalaManagementActions
    quality_tier: Literal["high", "medium", "low"]
    discount_rate: float
    intrinsic_value: float | None = None
    margin_of_safety_pct: float | None = None
    market_cap: float | None = None
    total_score: float
    max_score: float


class RakeshJhunjhunwalaSignal(AnalystSignal):
    agent_id: Literal["rakesh_jhunjhunwala"] = Field(default="rakesh_jhunjhunwala")
    evidence: RakeshJhunjhunwalaEvidence


# ── RawData ───────────────────────────────────────────────────────────────────


class RakeshJhunjhunwalaRawData(BaseModel):
    revenue_series: list[float | None] = Field(default_factory=list)
    net_income_series: list[float | None] = Field(default_factory=list)
    eps_series: list[float | None] = Field(default_factory=list)
    free_cash_flow_series: list[float | None] = Field(default_factory=list)
    dividends_series: list[float | None] = Field(default_factory=list)
    issuance_or_purchase_series: list[float | None] = Field(default_factory=list)
    total_assets_series: list[float | None] = Field(default_factory=list)
    total_liabilities_series: list[float | None] = Field(default_factory=list)
    current_assets_series: list[float | None] = Field(default_factory=list)
    current_liabilities_series: list[float | None] = Field(default_factory=list)
    roe_latest: float | None = None
    operating_margin_latest: float | None = None
    market_cap: float | None = None


# ── State ─────────────────────────────────────────────────────────────────────


class RakeshJhunjhunwalaInput(TypedDict, total=False):
    ticker: str
    as_of_date: str
    query: str | None


class RakeshJhunjhunwalaOutput(TypedDict, total=False):
    persona_signals: list[dict[str, Any]]


class RakeshJhunjhunwalaState(AgentState):
    ticker: Annotated[str, OmitFromSchema(input=False, output=True)]
    as_of_date: Annotated[str, OmitFromSchema(input=False, output=True)]
    query: Annotated[str | None, OmitFromSchema(input=False, output=True)]
    revenue_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    net_income_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    eps_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    free_cash_flow_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    dividends_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    issuance_or_purchase_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    total_assets_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    total_liabilities_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    current_assets_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    current_liabilities_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    roe_latest: Annotated[float | None, OmitFromSchema(input=True, output=False)]
    operating_margin_latest: Annotated[
        float | None, OmitFromSchema(input=True, output=False)
    ]
    market_cap: Annotated[float | None, OmitFromSchema(input=True, output=False)]
    evidence: Annotated[
        RakeshJhunjhunwalaEvidence | None, OmitFromSchema(input=True, output=False)
    ]
    persona_signals: Annotated[list[dict], OmitFromSchema(input=True, output=False)]


# ── Composite scorers ─────────────────────────────────────────────────────────


def _cagr(series: list[float | None]) -> float | None:
    vals = [v for v in series if v is not None]
    if len(vals) < 2 or vals[0] is None or vals[0] <= 0 or vals[-1] <= 0:
        return None
    return (vals[-1] / vals[0]) ** (1 / (len(vals) - 1)) - 1


def _score_jhunjhunwala_profitability(
    state: RakeshJhunjhunwalaState,
) -> RakeshJhunjhunwalaProfitability:
    roe = state.get("roe_latest")
    op_margin = state.get("operating_margin_latest")
    eps_cagr = _cagr(state.get("eps_series") or [])
    score = 0
    parts: list[str] = []
    if roe is not None:
        if roe > 0.20:
            score += 3
            parts.append(f"ROE {roe:.1%}")
        elif roe > 0.15:
            score += 2
        elif roe > 0.10:
            score += 1
    if op_margin is not None:
        if op_margin > 0.20:
            score += 2
        elif op_margin > 0.15:
            score += 1
    if eps_cagr is not None:
        if eps_cagr > 0.20:
            score += 3
            parts.append(f"EPS CAGR {eps_cagr:.1%}")
        elif eps_cagr > 0.15:
            score += 2
        elif eps_cagr > 0.10:
            score += 1
    return RakeshJhunjhunwalaProfitability(
        return_on_equity=roe,
        operating_margin=op_margin,
        eps_cagr=eps_cagr,
        score=min(score, 8),
        max_score=8,
        reasoning="; ".join(parts) or "Limited",
    )


def _score_jhunjhunwala_growth(
    state: RakeshJhunjhunwalaState,
) -> RakeshJhunjhunwalaGrowth:
    rev_cagr = _cagr(state.get("revenue_series") or [])
    ni_cagr = _cagr(state.get("net_income_series") or [])
    net_income = [v for v in (state.get("net_income_series") or []) if v is not None]
    score = 0
    parts: list[str] = []
    if rev_cagr is not None:
        if rev_cagr > 0.20:
            score += 3
            parts.append(f"Rev CAGR {rev_cagr:.1%}")
        elif rev_cagr > 0.15:
            score += 2
        elif rev_cagr > 0.10:
            score += 1
    if ni_cagr is not None:
        if ni_cagr > 0.25:
            score += 3
            parts.append(f"NI CAGR {ni_cagr:.1%}")
        elif ni_cagr > 0.20:
            score += 2
        elif ni_cagr > 0.15:
            score += 1
    consistent_ni = False
    if len(net_income) >= 2:
        if all(
            net_income[i] >= net_income[i - 1] * 0.8 for i in range(1, len(net_income))
        ):
            score += 1
            consistent_ni = True
            parts.append("Consistent NI")
    return RakeshJhunjhunwalaGrowth(
        revenue_cagr=rev_cagr,
        net_income_cagr=ni_cagr,
        consistent_ni=consistent_ni,
        score=min(score, 7),
        max_score=7,
        reasoning="; ".join(parts) or "Limited",
    )


def _score_jhunjhunwala_balance(
    state: RakeshJhunjhunwalaState,
) -> RakeshJhunjhunwalaBalanceSheet:
    total_assets = [
        v for v in (state.get("total_assets_series") or []) if v is not None
    ]
    total_liab = [
        v for v in (state.get("total_liabilities_series") or []) if v is not None
    ]
    current_assets = [
        v for v in (state.get("current_assets_series") or []) if v is not None
    ]
    current_liab = [
        v for v in (state.get("current_liabilities_series") or []) if v is not None
    ]
    score = 0
    parts: list[str] = []
    de: float | None = None
    cr: float | None = None
    if total_assets and total_liab and total_assets[-1] > 0:
        de = total_liab[-1] / total_assets[-1]
        if de < 0.5:
            score += 2
            parts.append(f"D/Assets {de:.2f}")
        elif de < 0.7:
            score += 1
    if current_assets and current_liab and current_liab[-1] > 0:
        cr = current_assets[-1] / current_liab[-1]
        if cr > 2.0:
            score += 2
            parts.append(f"Current ratio {cr:.2f}")
        elif cr > 1.5:
            score += 1
    return RakeshJhunjhunwalaBalanceSheet(
        debt_to_assets=de,
        current_ratio=cr,
        score=min(score, 4),
        max_score=4,
        reasoning="; ".join(parts) or "Limited",
    )


def _score_jhunjhunwala_cash_flow(
    state: RakeshJhunjhunwalaState,
) -> RakeshJhunjhunwalaCashFlow:
    fcf = [v for v in (state.get("free_cash_flow_series") or []) if v is not None]
    dividends = [v for v in (state.get("dividends_series") or []) if v is not None]
    score = 0
    parts: list[str] = []
    fcf_positive = bool(fcf and fcf[-1] is not None and fcf[-1] > 0)
    if fcf_positive:
        score += 2
        parts.append("Positive FCF")
    pays_dividends = bool(dividends and any(d < 0 for d in dividends))
    if pays_dividends:
        score += 1
        parts.append("Pays dividends")
    return RakeshJhunjhunwalaCashFlow(
        fcf_positive_latest=fcf_positive,
        pays_dividends=pays_dividends,
        score=min(score, 3),
        max_score=3,
        reasoning="; ".join(parts) or "Limited",
    )


def _score_jhunjhunwala_management(
    state: RakeshJhunjhunwalaState,
) -> RakeshJhunjhunwalaManagementActions:
    issuance = [
        v for v in (state.get("issuance_or_purchase_series") or []) if v is not None
    ]
    score = 0
    parts: list[str] = []
    has_buybacks = False
    latest: float | None = None
    if issuance and issuance[-1] is not None:
        latest = issuance[-1]
        if latest < 0:
            score += 2
            has_buybacks = True
            parts.append("Buybacks")
        elif latest == 0:
            score += 1
            parts.append("No dilution")
    return RakeshJhunjhunwalaManagementActions(
        latest_issuance=latest,
        has_buybacks=has_buybacks,
        score=min(score, 2),
        max_score=2,
        reasoning="; ".join(parts) or "Limited",
    )


def _quality_tier(
    profitability: int, growth: int, balance: int
) -> Literal["high", "medium", "low"]:
    if profitability >= 6 and balance >= 3:
        return "high"
    if profitability >= 4:
        return "medium"
    return "low"


# ── Graph nodes ───────────────────────────────────────────────────────────────


def compute_evidence_node(state: RakeshJhunjhunwalaState) -> dict[str, Any]:
    profitability = _score_jhunjhunwala_profitability(state)
    growth = _score_jhunjhunwala_growth(state)
    balance = _score_jhunjhunwala_balance(state)
    cash_flow = _score_jhunjhunwala_cash_flow(state)
    mgmt = _score_jhunjhunwala_management(state)
    tier = _quality_tier(profitability.score, growth.score, balance.score)
    discount_rates: dict[str, float] = {"high": 0.12, "medium": 0.15, "low": 0.18}
    terminal_multiples: dict[str, float] = {"high": 18.0, "medium": 15.0, "low": 12.0}
    discount = discount_rates[tier]
    # ai-hedge-fund parity: derive sustainable growth from historical net-income
    # CAGR, then value 5y of earnings + a quality-based exit-multiple terminal.
    ni_series = [v for v in (state.get("net_income_series") or []) if v is not None]
    ni_latest = ni_series[-1] if ni_series else None
    intrinsic: float | None = None
    if ni_latest is not None and ni_latest > 0:
        positive_ni = [v for v in ni_series if v > 0]
        if len(positive_ni) < 2:
            # Conservative fallback: latest earnings × 12 (P/E of 12)
            intrinsic = ni_latest * 12
        else:
            yrs = len(positive_ni) - 1
            hist_growth = (positive_ni[-1] / positive_ni[0]) ** (1 / yrs) - 1
            if hist_growth > 0.25:
                sustainable_growth = 0.20
            elif hist_growth > 0.15:
                sustainable_growth = hist_growth * 0.8
            elif hist_growth > 0.05:
                sustainable_growth = hist_growth * 0.9
            else:
                sustainable_growth = 0.05
            intrinsic = compute_intrinsic_value_exit_multiple(
                base_cash_flow=ni_latest,
                growth_rate=sustainable_growth,
                discount_rate=discount,
                terminal_multiple=terminal_multiples[tier],
                years=5,
            )
    market_cap = state.get("market_cap")
    mos_pct: float | None = (
        (intrinsic - market_cap) / market_cap * 100
        if intrinsic is not None and market_cap and market_cap > 0
        else None
    )
    total = (
        profitability.score
        + growth.score
        + balance.score
        + cash_flow.score
        + mgmt.score
    )
    evidence = RakeshJhunjhunwalaEvidence(
        profitability=profitability,
        growth=growth,
        balance_sheet=balance,
        cash_flow=cash_flow,
        management_actions=mgmt,
        quality_tier=tier,
        discount_rate=discount,
        intrinsic_value=intrinsic,
        margin_of_safety_pct=mos_pct,
        market_cap=market_cap,
        total_score=total,
        max_score=24,
    )
    return {"evidence": evidence}


async def render_verdict_node(
    state: RakeshJhunjhunwalaState, config: RunnableConfig
) -> dict[str, Any]:
    ticker = state.get("ticker", "")
    as_of_date = state.get("as_of_date", "")
    query = state.get("query")
    evidence = state.get("evidence")
    if evidence is None:
        raise RuntimeError(
            "render_verdict_node called without evidence — "
            "compute_evidence_node must run first"
        )

    llm = ModelConfiguration.get_chat_model_for_role(
        config, "reasoner", schema=RakeshJhunjhunwalaSignal
    )
    prompt = render_template(
        "personas_council/personas/rakesh_jhunjhunwala.jinja",
        ticker=ticker,
        as_of_date=as_of_date,
        evidence=evidence,
        market_cap=state.get("market_cap"),
        query=query,
    )
    result = cast(
        RakeshJhunjhunwalaSignal,
        await llm.ainvoke(
            [
                SystemMessage(prompt),
                HumanMessage("Render your Jhunjhunwala verdict now."),
            ]
        ),
    )
    return {"persona_signals": [result.model_dump()]}


# ── Data-collection sub-agent + subgraph builder ──────────────────────────────


_MCP_TOOLS = [
    "equity_fundamental_metrics",
    "equity_fundamental_income",
    "equity_fundamental_balance",
    "equity_fundamental_cash",
    "equity_fundamental_dividends",
    "equity_historical_market_cap",
]


async def _build_data_collection_agent(config: RunnableConfig) -> CompiledStateGraph:
    cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = cfg.get_llm_for_role("collector")
    mcp_tools = await get_tools(config, _MCP_TOOLS)
    builder = (
        MuffinAgentBuilder(primary, name="rakesh_jhunjhunwala_data_collection")
        .with_fallback_models(*fallbacks)
        .with_state_schema(RakeshJhunjhunwalaState)
        .with_runtime_system_prompt_template(
            "personas_council/personas/rakesh_jhunjhunwala_data_collection.jinja"
        )
        .with_response_format(RakeshJhunjhunwalaRawData)
        .with_model_call_limit(run_limit=8, exit_behavior="end")
    )
    for tool in mcp_tools:
        builder = builder.with_tool(tool, run_limit=2)
    builder = builder.with_tool(execute_python, is_cacheable=False)
    return builder.build_react_agent()


async def build_rakesh_jhunjhunwala_agent(config: RunnableConfig) -> CompiledStateGraph:
    data_agent = await _build_data_collection_agent(config)
    graph = StateGraph(
        RakeshJhunjhunwalaState,
        input_schema=RakeshJhunjhunwalaInput,
        output_schema=RakeshJhunjhunwalaOutput,
    )
    graph.add_node(
        "collect_data",
        data_agent,
        input_schema=RakeshJhunjhunwalaInput,
        retry_policy=_LLM_RETRY,
    )
    graph.add_node("compute_evidence", compute_evidence_node)
    graph.add_node("render_verdict", render_verdict_node, retry_policy=_LLM_RETRY)
    graph.add_edge(START, "collect_data")
    graph.add_edge("collect_data", "compute_evidence")
    graph.add_edge("compute_evidence", "render_verdict")
    graph.add_edge("render_verdict", END)
    return graph.compile()
