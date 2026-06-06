"""Bill Ackman persona — compiled subgraph (collect → compute → verdict).

Activist + business quality + catalyst lens. See ``warren_buffett.py`` for the
canonical v4 reference. Reference: ``ai-hedge-fund/src/agents/bill_ackman.py``.
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

from ...model_config import ModelConfiguration
from ...prompts import render_template
from ...tools.scoring_helpers import compute_intrinsic_value_dcf
from ...utils.agent_builder import MuffinAgentBuilder
from ..data_collection.utils import get_tools
from .schemas import AnalystSignal

logger = logging.getLogger(__name__)
_LLM_RETRY = RetryPolicy(max_attempts=2)


# ── Typed sub-evidence ────────────────────────────────────────────────────────


class BillAckmanBusinessQuality(BaseModel):
    revenue_growth_pct: float | None
    latest_operating_margin: float | None
    fcf_positive_periods: int
    fcf_total_periods: int
    score: int
    max_score: int
    reasoning: str


class BillAckmanFinancialDiscipline(BaseModel):
    debt_to_equity: float | None
    dividends_paid_years: int
    dividends_window_years: int
    score: int
    max_score: int
    reasoning: str


class BillAckmanActivismPotential(BaseModel):
    shares_change_pct: float | None
    op_margin_compression_pp: float | None
    score: int
    max_score: int
    reasoning: str


class BillAckmanValuation(BaseModel):
    score: int
    max_score: int
    reasoning: str


class BillAckmanEvidence(BaseModel):
    business_quality: BillAckmanBusinessQuality
    financial_discipline: BillAckmanFinancialDiscipline
    activism_potential: BillAckmanActivismPotential
    valuation: BillAckmanValuation
    intrinsic_value: float | None = None
    margin_of_safety_pct: float | None = None
    market_cap: float | None = None
    total_score: float
    max_score: float


class BillAckmanSignal(AnalystSignal):
    agent_id: Literal["bill_ackman"] = Field(default="bill_ackman")
    evidence: BillAckmanEvidence


# ── RawData ───────────────────────────────────────────────────────────────────


class BillAckmanRawData(BaseModel):
    """Ackman MCP extraction. Series oldest -> newest."""

    revenue_series: list[float | None] = Field(default_factory=list)
    operating_margin_series: list[float | None] = Field(default_factory=list)
    free_cash_flow_series: list[float | None] = Field(default_factory=list)
    total_debt_series: list[float | None] = Field(default_factory=list)
    shareholders_equity_series: list[float | None] = Field(default_factory=list)
    dividends_series: list[float | None] = Field(
        default_factory=list,
        description=(
            "dividends_and_other_cash_distributions, SIGNED (negative = outflow), "
            "oldest -> newest."
        ),
    )
    outstanding_shares_series: list[float | None] = Field(default_factory=list)
    market_cap: float | None = None


# ── State ─────────────────────────────────────────────────────────────────────


class BillAckmanInput(TypedDict, total=False):
    ticker: str
    as_of_date: str
    query: str | None


class BillAckmanOutput(TypedDict, total=False):
    persona_signals: list[dict[str, Any]]


class BillAckmanState(AgentState):
    ticker: Annotated[str, OmitFromSchema(input=False, output=True)]
    as_of_date: Annotated[str, OmitFromSchema(input=False, output=True)]
    query: Annotated[str | None, OmitFromSchema(input=False, output=True)]
    revenue_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    operating_margin_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    free_cash_flow_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    total_debt_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    shareholders_equity_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    dividends_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    outstanding_shares_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    market_cap: Annotated[float | None, OmitFromSchema(input=True, output=True)]
    evidence: Annotated[
        BillAckmanEvidence | None, OmitFromSchema(input=True, output=True)
    ]
    persona_signals: Annotated[list[dict], OmitFromSchema(input=True, output=False)]


# ── Composite scorers ─────────────────────────────────────────────────────────


def _score_ackman_business_quality(
    state: BillAckmanState,
) -> BillAckmanBusinessQuality:
    revenues = [v for v in (state.get("revenue_series") or []) if v is not None]
    op_margins = [
        v for v in (state.get("operating_margin_series") or []) if v is not None
    ]
    fcf = [v for v in (state.get("free_cash_flow_series") or []) if v is not None]

    score = 0
    parts: list[str] = []
    growth: float | None = None
    if len(revenues) >= 2 and revenues[0] and revenues[0] > 0:
        growth = (revenues[-1] - revenues[0]) / revenues[0]
        if growth > 0.5:
            score += 2
            parts.append(f"Revenue growth {growth:.1%} over window")
        elif growth > 0.2:
            score += 1

    latest_om: float | None = op_margins[-1] if op_margins else None
    if latest_om is not None:
        if latest_om > 0.20:
            score += 2
            parts.append(f"Op margin {latest_om:.1%} (strong)")
        elif latest_om > 0.10:
            score += 1

    positives = sum(1 for f in fcf if f > 0) if fcf else 0
    if fcf and positives == len(fcf):
        score += 1
        parts.append("FCF positive every period")

    return BillAckmanBusinessQuality(
        revenue_growth_pct=growth * 100 if growth is not None else None,
        latest_operating_margin=latest_om,
        fcf_positive_periods=positives,
        fcf_total_periods=len(fcf),
        score=min(score, 5),
        max_score=5,
        reasoning="; ".join(parts) or "Limited data",
    )


def _score_ackman_financial_discipline(
    state: BillAckmanState,
) -> BillAckmanFinancialDiscipline:
    total_debt = [v for v in (state.get("total_debt_series") or []) if v is not None]
    equity = [
        v for v in (state.get("shareholders_equity_series") or []) if v is not None
    ]
    dividends = [v for v in (state.get("dividends_series") or []) if v is not None]
    score = 0
    parts: list[str] = []
    de: float | None = None
    if total_debt and equity and equity[-1] and equity[-1] > 0:
        de = total_debt[-1] / equity[-1]
        if de < 0.5:
            score += 2
            parts.append(f"D/E {de:.2f}")
        elif de < 1.0:
            score += 1
    paid_years = sum(1 for d in dividends if d < 0) if dividends else 0
    if dividends and paid_years >= len(dividends) // 2:
        score += 2
        parts.append(f"Pays dividends ({paid_years}/{len(dividends)} years)")
    return BillAckmanFinancialDiscipline(
        debt_to_equity=de,
        dividends_paid_years=paid_years,
        dividends_window_years=len(dividends),
        score=min(score, 5),
        max_score=5,
        reasoning="; ".join(parts) or "Limited data",
    )


def _score_ackman_activism_potential(
    state: BillAckmanState,
) -> BillAckmanActivismPotential:
    shares = [
        v for v in (state.get("outstanding_shares_series") or []) if v is not None
    ]
    op_margins = [
        v for v in (state.get("operating_margin_series") or []) if v is not None
    ]
    score = 0
    parts: list[str] = []
    delta: float | None = None
    gap: float | None = None
    if len(shares) >= 2 and shares[0] > 0:
        delta = (shares[-1] - shares[0]) / shares[0]
        if delta > 0.05:
            score += 2
            parts.append(f"Shares grew {delta:+.1%} (dilution — activism opportunity)")
        elif delta < -0.05:
            score += 1
            parts.append("Buybacks underway")
    if len(op_margins) >= 4:
        recent = sum(op_margins[-2:]) / 2
        older = sum(op_margins[:2]) / 2
        gap = older - recent
        if gap > 0.05:
            score += 3
            parts.append(f"Op margin compression {gap:+.1%} — turnaround thesis")
        elif gap > 0.02:
            score += 1
    return BillAckmanActivismPotential(
        shares_change_pct=delta * 100 if delta is not None else None,
        op_margin_compression_pp=gap * 100 if gap is not None else None,
        score=min(score, 5),
        max_score=5,
        reasoning="; ".join(parts) or "No activism catalysts",
    )


def _score_ackman_valuation(
    fcf_latest: float | None, market_cap: float | None
) -> tuple[BillAckmanValuation, float | None, float | None]:
    if not fcf_latest or fcf_latest <= 0 or not market_cap or market_cap <= 0:
        return (
            BillAckmanValuation(score=0, max_score=5, reasoning="Cannot compute DCF"),
            None,
            None,
        )
    iv = compute_intrinsic_value_dcf(
        base_cash_flow=fcf_latest,
        growth_rate=0.10,
        discount_rate=0.10,
        terminal_growth_rate=0.025,
        years=5,
    )
    if iv is None:
        return (
            BillAckmanValuation(score=0, max_score=5, reasoning="DCF inputs invalid"),
            None,
            None,
        )
    mos = (iv - market_cap) / market_cap
    score = 0
    parts = [f"DCF IV ${iv:,.0f}, MoS {mos:+.1%}"]
    if mos > 0.3:
        score = 5
    elif mos > 0.1:
        score = 3
    elif mos > -0.1:
        score = 1
    return (
        BillAckmanValuation(score=score, max_score=5, reasoning="; ".join(parts)),
        iv,
        mos * 100,
    )


# ── Graph nodes ───────────────────────────────────────────────────────────────


def compute_evidence_node(state: BillAckmanState) -> dict[str, Any]:
    quality = _score_ackman_business_quality(state)
    discipline = _score_ackman_financial_discipline(state)
    activism = _score_ackman_activism_potential(state)
    fcf = state.get("free_cash_flow_series") or []
    fcf_latest = fcf[-1] if fcf else None
    valuation, iv, mos_pct = _score_ackman_valuation(
        fcf_latest, state.get("market_cap")
    )

    total = quality.score + discipline.score + activism.score + valuation.score
    max_total = (
        quality.max_score
        + discipline.max_score
        + activism.max_score
        + valuation.max_score
    )

    evidence = BillAckmanEvidence(
        business_quality=quality,
        financial_discipline=discipline,
        activism_potential=activism,
        valuation=valuation,
        intrinsic_value=iv,
        margin_of_safety_pct=mos_pct,
        market_cap=state.get("market_cap"),
        total_score=total,
        max_score=max_total,
    )
    return {"evidence": evidence}


async def render_verdict_node(
    state: BillAckmanState, config: RunnableConfig
) -> dict[str, Any]:
    ticker = state.get("ticker", "")
    as_of_date = state.get("as_of_date", "")
    query = state.get("query")
    evidence = state.get("evidence")
    if evidence is None:
        raise RuntimeError(
            "render_verdict_node called without evidence — "
            "compute_evidence_node must run first in v4 subgraphs"
        )

    llm = ModelConfiguration.get_chat_model_for_role(
        config, "reasoner", schema=BillAckmanSignal
    )
    prompt = render_template(
        "personas/bill_ackman.jinja",
        ticker=ticker,
        as_of_date=as_of_date,
        evidence=evidence,
        market_cap=state.get("market_cap"),
        query=query,
    )
    result = cast(
        BillAckmanSignal,
        await llm.ainvoke(
            [SystemMessage(prompt), HumanMessage("Render your Ackman verdict now.")]
        ),
    )
    return {"persona_signals": [result.model_dump()]}


# ── Data-collection sub-agent + subgraph builder ──────────────────────────────


_MCP_TOOLS = [
    "equity_fundamental_metrics",
    "equity_fundamental_income",
    "equity_fundamental_balance",
    "equity_fundamental_cash",
    "equity_fundamental_ratios",
    "equity_ownership_share_statistics",
]


async def _build_data_collection_agent(config: RunnableConfig) -> CompiledStateGraph:
    cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = cfg.get_llm_for_role("collector")
    mcp_tools = await get_tools(config, _MCP_TOOLS)
    builder = (
        MuffinAgentBuilder(primary, name="bill_ackman_data_collection")
        .with_fallback_models(*fallbacks)
        .with_state_schema(BillAckmanState)
        .with_runtime_system_prompt_template(
            "personas/bill_ackman_data_collection.jinja"
        )
        .with_response_format(BillAckmanRawData)
        .with_model_call_limit(run_limit=8, exit_behavior="end")
    )
    for tool in mcp_tools:
        builder = builder.with_tool(tool, run_limit=2)
    return builder.build_react_agent()


async def build_bill_ackman_agent(config: RunnableConfig) -> CompiledStateGraph:
    data_agent = await _build_data_collection_agent(config)
    graph = StateGraph(
        BillAckmanState,
        input_schema=BillAckmanInput,
        output_schema=BillAckmanOutput,
    )
    graph.add_node(
        "collect_data",
        data_agent,
        input_schema=data_agent.input_schema,
        retry_policy=_LLM_RETRY,
    )
    graph.add_node("compute_evidence", compute_evidence_node)
    graph.add_node("render_verdict", render_verdict_node, retry_policy=_LLM_RETRY)
    graph.add_edge(START, "collect_data")
    graph.add_edge("collect_data", "compute_evidence")
    graph.add_edge("compute_evidence", "render_verdict")
    graph.add_edge("render_verdict", END)
    return graph.compile()


