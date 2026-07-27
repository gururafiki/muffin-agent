"""Aswath Damodaran persona — compiled subgraph.

Academic FCFF DCF + CAPM. See ``warren_buffett.py`` for the reference.
Reference: ``ai-hedge-fund/src/agents/aswath_damodaran.py``.
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
from ..tools.scoring_helpers import compute_damodaran_fcff_dcf

logger = logging.getLogger(__name__)
_LLM_RETRY = RetryPolicy(max_attempts=2)


# ── Typed sub-evidence ────────────────────────────────────────────────────────


class AswathDamodaranGrowthReinvestment(BaseModel):
    revenue_cagr: float | None
    fcff_expanding: bool
    return_on_invested_capital: float | None
    score: int
    max_score: int
    reasoning: str


class AswathDamodaranRiskProfile(BaseModel):
    beta: float | None
    debt_to_equity: float | None
    interest_coverage: float | None
    score: int
    max_score: int
    reasoning: str


class AswathDamodaranRelativeValuation(BaseModel):
    pe_ratio: float | None
    score: int
    max_score: int
    reasoning: str


class AswathDamodaranEvidence(BaseModel):
    growth_reinvestment: AswathDamodaranGrowthReinvestment
    risk_profile: AswathDamodaranRiskProfile
    relative_valuation: AswathDamodaranRelativeValuation
    intrinsic_value: float | None = None
    discount_rate: float | None = None
    margin_of_safety_pct: float | None = None
    market_cap: float | None = None
    total_score: float
    max_score: float


class AswathDamodaranSignal(AnalystSignal):
    agent_id: Literal["aswath_damodaran"] = Field(default="aswath_damodaran")
    evidence: AswathDamodaranEvidence


# ── RawData ───────────────────────────────────────────────────────────────────


class AswathDamodaranRawData(BaseModel):
    revenue_series: list[float | None] = Field(default_factory=list)
    free_cash_flow_series: list[float | None] = Field(default_factory=list)
    roic_latest: float | None = None
    beta_latest: float | None = None
    debt_to_equity_latest: float | None = None
    interest_coverage_latest: float | None = None
    pe_ratio_history: list[float | None] = Field(
        default_factory=list,
        description=(
            "Annual price-to-earnings ratio, oldest -> newest (>=5 periods for the "
            "relative-vs-historical-median valuation check)."
        ),
    )
    market_cap: float | None = None


# ── State ─────────────────────────────────────────────────────────────────────


class AswathDamodaranInput(TypedDict, total=False):
    ticker: str
    as_of_date: str
    query: str | None


class AswathDamodaranOutput(TypedDict, total=False):
    persona_signals: list[dict[str, Any]]


class AswathDamodaranState(AgentState):
    ticker: Annotated[str, OmitFromSchema(input=False, output=True)]
    as_of_date: Annotated[str, OmitFromSchema(input=False, output=True)]
    query: Annotated[str | None, OmitFromSchema(input=False, output=True)]
    revenue_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    free_cash_flow_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    roic_latest: Annotated[float | None, OmitFromSchema(input=True, output=False)]
    beta_latest: Annotated[float | None, OmitFromSchema(input=True, output=False)]
    debt_to_equity_latest: Annotated[
        float | None, OmitFromSchema(input=True, output=False)
    ]
    interest_coverage_latest: Annotated[
        float | None, OmitFromSchema(input=True, output=False)
    ]
    pe_ratio_history: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    market_cap: Annotated[float | None, OmitFromSchema(input=True, output=False)]
    evidence: Annotated[
        AswathDamodaranEvidence | None, OmitFromSchema(input=True, output=False)
    ]
    persona_signals: Annotated[list[dict], OmitFromSchema(input=True, output=False)]


# ── Composite scorers ─────────────────────────────────────────────────────────


def _score_damodaran_growth(
    state: AswathDamodaranState,
) -> tuple[AswathDamodaranGrowthReinvestment, float | None]:
    revenues = [v for v in (state.get("revenue_series") or []) if v is not None]
    fcf = [v for v in (state.get("free_cash_flow_series") or []) if v is not None]
    roic = state.get("roic_latest")
    score = 0
    parts: list[str] = []
    rev_cagr: float | None = None
    if len(revenues) >= 2 and revenues[0] > 0:
        rev_cagr = (revenues[-1] / revenues[0]) ** (1 / (len(revenues) - 1)) - 1
        if rev_cagr > 0.08:
            score += 2
            parts.append(f"Rev CAGR {rev_cagr:.1%}")
        elif rev_cagr > 0.03:
            score += 1
    fcff_expanding = bool(len(fcf) >= 2 and fcf[-1] > fcf[0])
    if fcff_expanding:
        score += 1
        parts.append("FCFF expanding")
    if roic is not None and roic > 0.10:
        score += 1
        parts.append(f"ROIC {roic:.1%}")
    return (
        AswathDamodaranGrowthReinvestment(
            revenue_cagr=rev_cagr,
            fcff_expanding=fcff_expanding,
            return_on_invested_capital=roic,
            score=min(score, 4),
            max_score=4,
            reasoning="; ".join(parts) or "Limited",
        ),
        rev_cagr,
    )


def _score_damodaran_risk(
    state: AswathDamodaranState,
) -> tuple[AswathDamodaranRiskProfile, float | None]:
    beta = state.get("beta_latest")
    de = state.get("debt_to_equity_latest")
    coverage = state.get("interest_coverage_latest")
    score = 0
    parts: list[str] = []
    if beta is not None:
        if beta < 1.3:
            score += 1
            parts.append(f"β {beta:.2f}")
    if de is not None:
        if de < 1.0:
            score += 1
    if coverage is not None and coverage > 3:
        score += 1
        parts.append(f"Interest coverage {coverage:.1f}x")
    return (
        AswathDamodaranRiskProfile(
            beta=beta,
            debt_to_equity=de,
            interest_coverage=coverage,
            score=min(score, 3),
            max_score=3,
            reasoning="; ".join(parts) or "Limited",
        ),
        beta,
    )


def _score_damodaran_relative(
    state: AswathDamodaranState,
) -> AswathDamodaranRelativeValuation:
    """P/E vs 5-yr historical median (ai-hedge-fund parity).

    +1 when TTM P/E < 0.7× median (cheap), −1 when > 1.3× median (expensive),
    0 when inline. Requires ≥5 P/E observations.
    """
    pes = [v for v in (state.get("pe_ratio_history") or []) if v is not None]
    if len(pes) < 5:
        return AswathDamodaranRelativeValuation(
            pe_ratio=pes[-1] if pes else None,
            score=0,
            max_score=1,
            reasoning="Insufficient P/E history (need 5+ periods)",
        )
    ttm_pe = pes[-1]
    median_pe = sorted(pes)[len(pes) // 2]
    if ttm_pe < 0.7 * median_pe:
        return AswathDamodaranRelativeValuation(
            pe_ratio=ttm_pe,
            score=1,
            max_score=1,
            reasoning=f"P/E {ttm_pe:.1f} vs median {median_pe:.1f} (cheap)",
        )
    if ttm_pe > 1.3 * median_pe:
        return AswathDamodaranRelativeValuation(
            pe_ratio=ttm_pe,
            score=-1,
            max_score=1,
            reasoning=f"P/E {ttm_pe:.1f} vs median {median_pe:.1f} (expensive)",
        )
    return AswathDamodaranRelativeValuation(
        pe_ratio=ttm_pe,
        score=0,
        max_score=1,
        reasoning=f"P/E {ttm_pe:.1f} inline with history",
    )


# ── Graph nodes ───────────────────────────────────────────────────────────────


def compute_evidence_node(state: AswathDamodaranState) -> dict[str, Any]:
    growth, rev_cagr = _score_damodaran_growth(state)
    risk, beta = _score_damodaran_risk(state)
    relative = _score_damodaran_relative(state)
    fcf = [v for v in (state.get("free_cash_flow_series") or []) if v is not None]
    fcf_latest = fcf[-1] if fcf else None
    intrinsic: float | None = None
    discount_rate: float | None = None
    if fcf_latest and fcf_latest > 0:
        # ai-hedge-fund parity: base growth = 5y revenue CAGR (cap 12%, fallback
        # 4%); terminal value anchored on base FCFF (helper default).
        initial_growth = min(rev_cagr if rev_cagr is not None else 0.04, 0.12)
        result = compute_damodaran_fcff_dcf(
            base_fcff=fcf_latest,
            initial_growth=initial_growth,
            beta=beta if beta is not None else 1.0,
        )
        if result is not None:
            intrinsic, discount_rate = result
    market_cap = state.get("market_cap")
    mos_pct: float | None = (
        (intrinsic - market_cap) / market_cap * 100
        if intrinsic is not None and market_cap and market_cap > 0
        else None
    )
    total = growth.score + risk.score + relative.score
    evidence = AswathDamodaranEvidence(
        growth_reinvestment=growth,
        risk_profile=risk,
        relative_valuation=relative,
        intrinsic_value=intrinsic,
        discount_rate=discount_rate,
        margin_of_safety_pct=mos_pct,
        market_cap=market_cap,
        total_score=total,
        max_score=8,
    )
    return {"evidence": evidence}


async def render_verdict_node(
    state: AswathDamodaranState, config: RunnableConfig
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
        config, "reasoner", schema=AswathDamodaranSignal
    )
    prompt = render_template(
        "personas_council/personas/aswath_damodaran.jinja",
        ticker=ticker,
        as_of_date=as_of_date,
        evidence=evidence,
        market_cap=state.get("market_cap"),
        query=query,
    )
    result = cast(
        AswathDamodaranSignal,
        await llm.ainvoke(
            [
                SystemMessage(prompt),
                HumanMessage("Render your Damodaran verdict now."),
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
    "equity_fundamental_ratios",
    "equity_estimates_consensus",
]


async def _build_data_collection_agent(config: RunnableConfig) -> CompiledStateGraph:
    cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = cfg.get_llm_for_role("collector")
    mcp_tools = await get_tools(config, _MCP_TOOLS)
    builder = (
        MuffinAgentBuilder(primary, name="aswath_damodaran_data_collection")
        .with_fallback_models(*fallbacks)
        .with_state_schema(AswathDamodaranState)
        .with_input_prompt_template(
            "personas_council/personas/aswath_damodaran_data_collection.jinja"
        )
        .with_response_format(AswathDamodaranRawData)
        .with_model_call_limit(run_limit=8, exit_behavior="end")
    )
    for tool in mcp_tools:
        builder = builder.with_tool(tool, run_limit=2)
    builder = builder.with_tool(execute_python, is_cacheable=False)
    return builder.build_react_agent()


async def build_aswath_damodaran_agent(config: RunnableConfig) -> CompiledStateGraph:
    data_agent = await _build_data_collection_agent(config)
    graph = StateGraph(
        AswathDamodaranState,
        input_schema=AswathDamodaranInput,
        output_schema=AswathDamodaranOutput,
    )
    graph.add_node(
        "collect_data",
        data_agent,
        input_schema=AswathDamodaranInput,
        retry_policy=_LLM_RETRY,
    )
    graph.add_node("compute_evidence", compute_evidence_node)
    graph.add_node("render_verdict", render_verdict_node, retry_policy=_LLM_RETRY)
    graph.add_edge(START, "collect_data")
    graph.add_edge("collect_data", "compute_evidence")
    graph.add_edge("compute_evidence", "render_verdict")
    graph.add_edge("render_verdict", END)
    return graph.compile()
