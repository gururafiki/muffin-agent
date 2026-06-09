"""Cathie Wood persona — compiled subgraph (collect → compute → verdict).

Disruptive innovation lens: revenue acceleration + R&D intensity + high-growth
DCF (20% / 15% / 25x terminal). See ``warren_buffett.py`` for the canonical
reference implementation.

Reference (upstream): ``ai-hedge-fund/src/agents/cathie_wood.py``.
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


class CathieWoodDisruptivePotential(BaseModel):
    """Revenue acceleration + gross-margin expansion + R&D intensity."""

    latest_revenue_growth: float | None
    gross_margin_change: float | None
    latest_gross_margin: float | None
    rd_intensity: float | None
    raw_score: int
    score: float
    max_score: float
    reasoning: str


class CathieWoodInnovationGrowth(BaseModel):
    """R&D growth + FCF consistency + operating leverage + capex intensity."""

    rd_growth_pct: float | None
    positive_fcf_periods: int
    fcf_total_periods: int
    op_margin_latest: float | None
    capex_intensity: float | None
    raw_score: int
    score: float
    max_score: float
    reasoning: str


class CathieWoodValuation(BaseModel):
    """High-growth exit-multiple DCF + margin of safety."""

    score: int
    max_score: int
    reasoning: str


class CathieWoodEvidence(BaseModel):
    """Wood-specific precomputed evidence."""

    disruptive_potential: CathieWoodDisruptivePotential
    innovation_growth: CathieWoodInnovationGrowth
    valuation: CathieWoodValuation
    intrinsic_value: float | None = None
    margin_of_safety_pct: float | None = None
    market_cap: float | None = None
    total_score: float
    max_score: float


class CathieWoodSignal(AnalystSignal):
    """Cathie Wood structured signal."""

    agent_id: Literal["cathie_wood"] = Field(default="cathie_wood")
    evidence: CathieWoodEvidence


# ── RawData ───────────────────────────────────────────────────────────────────


class CathieWoodRawData(BaseModel):
    """Structured MCP extraction. Time series are oldest -> newest."""

    revenue_series: list[float | None] = Field(default_factory=list)
    gross_margin_series: list[float | None] = Field(default_factory=list)
    operating_expense_series: list[float | None] = Field(default_factory=list)
    operating_margin_series: list[float | None] = Field(default_factory=list)
    research_and_development_series: list[float | None] = Field(default_factory=list)
    free_cash_flow_series: list[float | None] = Field(default_factory=list)
    capital_expenditure_series: list[float | None] = Field(
        default_factory=list,
        description="Capex as POSITIVE absolute values, oldest -> newest.",
    )
    dividends_series: list[float | None] = Field(
        default_factory=list,
        description=(
            "dividends_and_other_cash_distributions, SIGNED "
            "(negative = cash outflow), oldest -> newest."
        ),
    )
    market_cap: float | None = None


# ── State ─────────────────────────────────────────────────────────────────────


class CathieWoodInput(TypedDict, total=False):
    """Public input contract."""

    ticker: str
    as_of_date: str
    query: str | None


class CathieWoodOutput(TypedDict, total=False):
    """Public output contract."""

    persona_signals: list[dict[str, Any]]


class CathieWoodState(AgentState):
    """Cathie Wood persona subgraph state."""

    ticker: Annotated[str, OmitFromSchema(input=False, output=True)]
    as_of_date: Annotated[str, OmitFromSchema(input=False, output=True)]
    query: Annotated[str | None, OmitFromSchema(input=False, output=True)]
    revenue_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    gross_margin_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    operating_expense_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    operating_margin_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    research_and_development_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    free_cash_flow_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    capital_expenditure_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    dividends_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    market_cap: Annotated[float | None, OmitFromSchema(input=True, output=False)]
    evidence: Annotated[
        CathieWoodEvidence | None, OmitFromSchema(input=True, output=False)
    ]
    persona_signals: Annotated[list[dict], OmitFromSchema(input=True, output=False)]


# ── Composite scorers ─────────────────────────────────────────────────────────


def _score_wood_disruptive_potential(
    state: CathieWoodState,
) -> CathieWoodDisruptivePotential:
    """Revenue acceleration + margin expansion + R&D intensity (raw max 12, norm 5)."""
    revenues = [v for v in (state.get("revenue_series") or []) if v is not None]
    gross_margins = [
        v for v in (state.get("gross_margin_series") or []) if v is not None
    ]
    op_exp = [v for v in (state.get("operating_expense_series") or []) if v is not None]
    rd = [
        v for v in (state.get("research_and_development_series") or []) if v is not None
    ]

    score = 0
    parts: list[str] = []
    latest_growth: float | None = None
    margin_change: float | None = None
    latest_gross_margin = gross_margins[-1] if gross_margins else None
    rd_intensity: float | None = None

    if len(revenues) >= 3:
        growth_rates = []
        for i in range(1, len(revenues)):
            base = revenues[i - 1]
            if base and base != 0:
                growth_rates.append((revenues[i] - base) / abs(base))
        if len(growth_rates) >= 2 and growth_rates[-1] > growth_rates[0]:
            score += 2
            parts.append(
                f"Revenue acceleration "
                f"({growth_rates[0]:.1%} -> {growth_rates[-1]:.1%})"
            )
        latest_growth = growth_rates[-1] if growth_rates else None
        if latest_growth is not None:
            if latest_growth > 1.0:
                score += 3
                parts.append(f"Exceptional revenue growth {latest_growth:.1%}")
            elif latest_growth > 0.5:
                score += 2
                parts.append(f"Strong revenue growth {latest_growth:.1%}")
            elif latest_growth > 0.2:
                score += 1
                parts.append(f"Moderate revenue growth {latest_growth:.1%}")

    if len(gross_margins) >= 2:
        margin_change = gross_margins[-1] - gross_margins[0]
        if margin_change > 0.05:
            score += 2
            parts.append(f"Expanding gross margins {margin_change:+.1%}")
        elif margin_change > 0:
            score += 1
            parts.append(f"Slightly improving gross margins {margin_change:+.1%}")
        if gross_margins[-1] > 0.5:
            score += 2
            parts.append(f"High gross margin {gross_margins[-1]:.1%}")

    if len(revenues) >= 2 and len(op_exp) >= 2 and revenues[0] != 0 and op_exp[0] != 0:
        rev_growth = (revenues[-1] - revenues[0]) / abs(revenues[0])
        opex_growth = (op_exp[-1] - op_exp[0]) / abs(op_exp[0])
        if rev_growth > opex_growth:
            score += 2
            parts.append("Positive operating leverage")

    if rd and revenues and revenues[-1]:
        rd_intensity = rd[-1] / revenues[-1]
        if rd_intensity > 0.15:
            score += 3
            parts.append(f"High R&D intensity {rd_intensity:.1%}")
        elif rd_intensity > 0.08:
            score += 2
            parts.append(f"Moderate R&D {rd_intensity:.1%}")
        elif rd_intensity > 0.05:
            score += 1
            parts.append(f"Some R&D {rd_intensity:.1%}")

    max_raw = 12
    normalized = (score / max_raw) * 5
    return CathieWoodDisruptivePotential(
        latest_revenue_growth=latest_growth,
        gross_margin_change=margin_change,
        latest_gross_margin=latest_gross_margin,
        rd_intensity=rd_intensity,
        raw_score=score,
        score=normalized,
        max_score=5,
        reasoning="; ".join(parts) if parts else "Insufficient data",
    )


def _score_wood_innovation_growth(
    state: CathieWoodState,
) -> CathieWoodInnovationGrowth:
    """R&D growth + FCF + op margin + capex (raw max 15, norm 5)."""
    rd = [
        v for v in (state.get("research_and_development_series") or []) if v is not None
    ]
    revenues = [v for v in (state.get("revenue_series") or []) if v is not None]
    fcf = [v for v in (state.get("free_cash_flow_series") or []) if v is not None]
    op_margins = [
        v for v in (state.get("operating_margin_series") or []) if v is not None
    ]
    capex = [
        abs(v) for v in (state.get("capital_expenditure_series") or []) if v is not None
    ]
    dividends = [v for v in (state.get("dividends_series") or []) if v is not None]

    score = 0
    parts: list[str] = []
    rd_growth: float | None = None
    capex_intensity: float | None = None

    if len(rd) >= 2 and rd[0] != 0:
        rd_growth = (rd[-1] - rd[0]) / abs(rd[0])
        if rd_growth > 0.5:
            score += 3
            parts.append(f"Strong R&D growth {rd_growth:+.1%}")
        elif rd_growth > 0.2:
            score += 2
            parts.append(f"Moderate R&D growth {rd_growth:+.1%}")
        if len(revenues) >= 2 and revenues[0] != 0 and revenues[-1] != 0:
            start_intensity = rd[0] / revenues[0]
            end_intensity = rd[-1] / revenues[-1]
            if end_intensity > start_intensity:
                score += 2
                parts.append("Increasing R&D intensity")

    positive_fcf = 0
    if len(fcf) >= 2:
        positive_fcf = sum(1 for f in fcf if f > 0)
        fcf_growth = (fcf[-1] - fcf[0]) / abs(fcf[0]) if fcf[0] != 0 else 0
        if fcf_growth > 0.3 and positive_fcf == len(fcf):
            score += 3
            parts.append("Strong consistent FCF growth")
        elif positive_fcf >= len(fcf) * 0.75:
            score += 2
            parts.append("Consistent positive FCF")
        elif positive_fcf > len(fcf) // 2:
            score += 1
            parts.append("Moderately positive FCF")

    if len(op_margins) >= 2:
        margin_trend = op_margins[-1] - op_margins[0]
        if op_margins[-1] > 0.15 and margin_trend > 0:
            score += 3
            parts.append(f"Strong improving op margin {op_margins[-1]:.1%}")
        elif op_margins[-1] > 0.10:
            score += 2
            parts.append(f"Healthy op margin {op_margins[-1]:.1%}")
        elif margin_trend > 0:
            score += 1

    if capex and revenues and revenues[-1] and capex[-1] != 0:
        capex_intensity = capex[-1] / revenues[-1]
        capex_growth = (capex[-1] - capex[0]) / abs(capex[0]) if capex[0] != 0 else 0
        if capex_intensity > 0.10 and capex_growth > 0.2:
            score += 2
            parts.append("Heavy growth investment")
        elif capex_intensity > 0.05:
            score += 1
            parts.append("Moderate growth investment")

    if dividends and fcf and fcf[-1] != 0:
        payout = abs(dividends[-1] / fcf[-1])
        if payout < 0.2:
            score += 2
            parts.append("Reinvests heavily over dividends")
        elif payout < 0.4:
            score += 1

    max_raw = 15
    normalized = (score / max_raw) * 5
    return CathieWoodInnovationGrowth(
        rd_growth_pct=rd_growth * 100 if rd_growth is not None else None,
        positive_fcf_periods=positive_fcf,
        fcf_total_periods=len(fcf),
        op_margin_latest=op_margins[-1] if op_margins else None,
        capex_intensity=capex_intensity,
        raw_score=score,
        score=normalized,
        max_score=5,
        reasoning="; ".join(parts) if parts else "Insufficient data",
    )


def _score_wood_valuation(
    fcf_latest: float | None, market_cap: float | None
) -> tuple[CathieWoodValuation, float | None, float | None]:
    """High-growth exit-multiple DCF + MoS (max 5).

    Returns (CathieWoodValuation, intrinsic_value, margin_of_safety_pct).
    """
    if not fcf_latest or fcf_latest <= 0 or not market_cap or market_cap <= 0:
        return (
            CathieWoodValuation(
                score=0,
                max_score=5,
                reasoning="Cannot compute DCF (need positive FCF + market cap).",
            ),
            None,
            None,
        )
    iv = compute_intrinsic_value_exit_multiple(
        base_cash_flow=fcf_latest,
        growth_rate=0.20,
        discount_rate=0.15,
        terminal_multiple=25.0,
        years=5,
    )
    if iv is None:
        return (
            CathieWoodValuation(score=0, max_score=5, reasoning="DCF inputs invalid."),
            None,
            None,
        )
    mos = (iv - market_cap) / market_cap
    score = 0
    parts = [f"DCF IV ${iv:,.0f}, market cap ${market_cap:,.0f}, MoS {mos:+.1%}"]
    if mos > 0.5:
        score = 3
        parts.append("MoS > 50% — strong undervaluation")
    elif mos > 0.2:
        score = 1
        parts.append("MoS > 20% — modest undervaluation")
    elif mos < -0.5:
        parts.append("Severe overvaluation")
    return (
        CathieWoodValuation(score=score, max_score=5, reasoning="; ".join(parts)),
        iv,
        mos * 100,
    )


# ── Graph nodes ───────────────────────────────────────────────────────────────


def compute_evidence_node(state: CathieWoodState) -> dict[str, Any]:
    """Deterministic Wood evidence assembly."""
    disruptive = _score_wood_disruptive_potential(state)
    innovation = _score_wood_innovation_growth(state)
    fcf = state.get("free_cash_flow_series") or []
    fcf_latest = fcf[-1] if fcf else None
    valuation, iv, mos_pct = _score_wood_valuation(fcf_latest, state.get("market_cap"))

    total = disruptive.score + innovation.score + valuation.score
    max_total = disruptive.max_score + innovation.max_score + valuation.max_score

    evidence = CathieWoodEvidence(
        disruptive_potential=disruptive,
        innovation_growth=innovation,
        valuation=valuation,
        intrinsic_value=iv,
        margin_of_safety_pct=mos_pct,
        market_cap=state.get("market_cap"),
        total_score=total,
        max_score=max_total,
    )
    return {"evidence": evidence}


async def render_verdict_node(
    state: CathieWoodState, config: RunnableConfig
) -> dict[str, Any]:
    """Single LLM call — render CathieWoodSignal."""
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
        config, "reasoner", schema=CathieWoodSignal
    )
    prompt = render_template(
        "personas/cathie_wood.jinja",
        ticker=ticker,
        as_of_date=as_of_date,
        evidence=evidence,
        market_cap=state.get("market_cap"),
        query=query,
    )
    result = cast(
        CathieWoodSignal,
        await llm.ainvoke(
            [
                SystemMessage(prompt),
                HumanMessage("Render your Cathie Wood verdict now."),
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
    "equity_fundamental_revenue_per_segment",
    "equity_historical_market_cap",
]


async def _build_data_collection_agent(config: RunnableConfig) -> CompiledStateGraph:
    """Compiled ReAct sub-agent that fetches MCP data -> CathieWoodRawData."""
    cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = cfg.get_llm_for_role("collector")
    mcp_tools = await get_tools(config, _MCP_TOOLS)

    builder = (
        MuffinAgentBuilder(primary, name="cathie_wood_data_collection")
        .with_fallback_models(*fallbacks)
        .with_state_schema(CathieWoodState)
        .with_runtime_system_prompt_template(
            "personas/cathie_wood_data_collection.jinja"
        )
        .with_response_format(CathieWoodRawData)
        .with_model_call_limit(run_limit=8, exit_behavior="end")
    )
    for tool in mcp_tools:
        builder = builder.with_tool(tool, run_limit=2)
    builder = builder.with_tool(execute_python, is_cacheable=False)
    return builder.build_react_agent()


async def build_cathie_wood_agent(config: RunnableConfig) -> CompiledStateGraph:
    """Build the full 3-node Cathie Wood subgraph."""
    data_agent = await _build_data_collection_agent(config)
    graph = StateGraph(
        CathieWoodState,
        input_schema=CathieWoodInput,
        output_schema=CathieWoodOutput,
    )
    graph.add_node(
        "collect_data",
        data_agent,
        input_schema=CathieWoodInput,
        retry_policy=_LLM_RETRY,
    )
    graph.add_node("compute_evidence", compute_evidence_node)
    graph.add_node("render_verdict", render_verdict_node, retry_policy=_LLM_RETRY)
    graph.add_edge(START, "collect_data")
    graph.add_edge("collect_data", "compute_evidence")
    graph.add_edge("compute_evidence", "render_verdict")
    graph.add_edge("render_verdict", END)
    return graph.compile()
