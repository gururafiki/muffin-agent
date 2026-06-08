"""Mohnish Pabrai persona — compiled subgraph (collect → compute → verdict).

Dhandho weighting: 0.45 downside + 0.35 valuation + 0.20 double potential.
See ``warren_buffett.py`` for the canonical reference.
Reference: ``ai-hedge-fund/src/agents/mohnish_pabrai.py``.
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

logger = logging.getLogger(__name__)
_LLM_RETRY = RetryPolicy(max_attempts=2)


# ── Typed sub-evidence ────────────────────────────────────────────────────────


class MohnishPabraiDownsideProtection(BaseModel):
    net_cash: float | None
    current_ratio: float | None
    debt_to_equity: float | None
    fcf_positive_ratio: float | None
    score: int
    max_score: int
    reasoning: str


class MohnishPabraiValuation(BaseModel):
    fcf_yield: float | None
    capex_intensity: float | None
    score: int
    max_score: int
    reasoning: str


class MohnishPabraiDoublePotential(BaseModel):
    revenue_growth_pct: float | None
    fcf_growth_pct: float | None
    score: int
    max_score: int
    reasoning: str


class MohnishPabraiEvidence(BaseModel):
    downside_protection: MohnishPabraiDownsideProtection
    valuation: MohnishPabraiValuation
    double_potential: MohnishPabraiDoublePotential
    weighted_score: float
    market_cap: float | None = None
    total_score: float
    max_score: float


class MohnishPabraiSignal(AnalystSignal):
    agent_id: Literal["mohnish_pabrai"] = Field(default="mohnish_pabrai")
    evidence: MohnishPabraiEvidence


# ── RawData ───────────────────────────────────────────────────────────────────


class MohnishPabraiRawData(BaseModel):
    """Pabrai MCP extraction. Series oldest -> newest."""

    revenue_series: list[float | None] = Field(default_factory=list)
    free_cash_flow_series: list[float | None] = Field(default_factory=list)
    capital_expenditure_series: list[float | None] = Field(
        default_factory=list,
        description="POSITIVE absolute values.",
    )
    cash_and_equivalents_series: list[float | None] = Field(default_factory=list)
    total_debt_series: list[float | None] = Field(default_factory=list)
    shareholders_equity_series: list[float | None] = Field(default_factory=list)
    current_assets_series: list[float | None] = Field(default_factory=list)
    current_liabilities_series: list[float | None] = Field(default_factory=list)
    market_cap: float | None = None


# ── State ─────────────────────────────────────────────────────────────────────


class MohnishPabraiInput(TypedDict, total=False):
    ticker: str
    as_of_date: str
    query: str | None


class MohnishPabraiOutput(TypedDict, total=False):
    persona_signals: list[dict[str, Any]]


class MohnishPabraiState(AgentState):
    ticker: Annotated[str, OmitFromSchema(input=False, output=True)]
    as_of_date: Annotated[str, OmitFromSchema(input=False, output=True)]
    query: Annotated[str | None, OmitFromSchema(input=False, output=True)]
    revenue_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    free_cash_flow_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    capital_expenditure_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    cash_and_equivalents_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    total_debt_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    shareholders_equity_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    current_assets_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    current_liabilities_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    market_cap: Annotated[float | None, OmitFromSchema(input=True, output=True)]
    evidence: Annotated[
        MohnishPabraiEvidence | None, OmitFromSchema(input=True, output=True)
    ]
    persona_signals: Annotated[list[dict], OmitFromSchema(input=True, output=False)]


# ── Composite scorers ─────────────────────────────────────────────────────────


def _score_pabrai_downside(
    state: MohnishPabraiState,
) -> MohnishPabraiDownsideProtection:
    cash = [
        v for v in (state.get("cash_and_equivalents_series") or []) if v is not None
    ]
    total_debt = [v for v in (state.get("total_debt_series") or []) if v is not None]
    equity = [
        v for v in (state.get("shareholders_equity_series") or []) if v is not None
    ]
    current_assets = [
        v for v in (state.get("current_assets_series") or []) if v is not None
    ]
    current_liab = [
        v for v in (state.get("current_liabilities_series") or []) if v is not None
    ]
    fcf = [v for v in (state.get("free_cash_flow_series") or []) if v is not None]

    score = 0
    parts: list[str] = []
    net_cash: float | None = None
    cr: float | None = None
    de: float | None = None
    fcf_pos_ratio: float | None = None

    if cash and total_debt:
        net_cash = cash[-1] - total_debt[-1]
        if net_cash > 0:
            score += 3
            parts.append(f"Net cash {net_cash:,.0f}")
    if current_assets and current_liab and current_liab[-1] and current_liab[-1] > 0:
        cr = current_assets[-1] / current_liab[-1]
        if cr >= 2.0:
            score += 2
            parts.append(f"Current ratio {cr:.2f}")
        elif cr >= 1.2:
            score += 1
    if total_debt and equity and equity[-1] and equity[-1] > 0:
        de = total_debt[-1] / equity[-1]
        if de < 0.3:
            score += 2
            parts.append(f"D/E {de:.2f}")
        elif de < 0.7:
            score += 1
    if fcf:
        positives = sum(1 for v in fcf if v > 0)
        fcf_pos_ratio = positives / len(fcf)
        # ai-hedge-fund parity: score the FCF *trend* (recent 3-period avg vs
        # oldest), not raw consistency. Positive & improving/stable -> +2;
        # positive but declining -> +1; non-positive -> 0.
        if len(fcf) >= 3:
            recent_avg = sum(fcf[-3:]) / 3
            older = sum(fcf[:3]) / 3 if len(fcf) >= 6 else fcf[0]
            if recent_avg > 0 and recent_avg >= older:
                score += 2
                parts.append(
                    f"Positive & improving/stable FCF ({positives}/{len(fcf)} +ve)"
                )
            elif recent_avg > 0:
                score += 1
                parts.append(
                    f"Positive but declining FCF ({positives}/{len(fcf)} +ve)"
                )

    return MohnishPabraiDownsideProtection(
        net_cash=net_cash,
        current_ratio=cr,
        debt_to_equity=de,
        fcf_positive_ratio=fcf_pos_ratio,
        score=min(score, 10),
        max_score=10,
        reasoning="; ".join(parts) or "Limited data",
    )


def _score_pabrai_valuation(state: MohnishPabraiState) -> MohnishPabraiValuation:
    fcf = state.get("free_cash_flow_series") or []
    fcf_latest = fcf[-1] if fcf else None
    market_cap = state.get("market_cap")
    revenues = state.get("revenue_series") or []
    capex = state.get("capital_expenditure_series") or []

    score = 0
    parts: list[str] = []
    fcf_yield: float | None = None
    capex_intensity: float | None = None

    if fcf_latest is not None and market_cap and market_cap > 0:
        fcf_yield = fcf_latest / market_cap
        if fcf_yield > 0.10:
            score += 4
            parts.append(f"FCF yield {fcf_yield:.1%}")
        elif fcf_yield > 0.07:
            score += 3
        elif fcf_yield > 0.05:
            score += 2
        elif fcf_yield > 0.03:
            score += 1

    if revenues and capex:
        rev = revenues[-1] or 0
        cap = abs(capex[-1] or 0)
        capex_intensity = cap / rev if rev > 0 else None
        if capex_intensity is not None:
            if capex_intensity < 0.05:
                score += 2
                parts.append(f"Capex/rev {capex_intensity:.1%} (light)")
            elif capex_intensity < 0.10:
                score += 1

    return MohnishPabraiValuation(
        fcf_yield=fcf_yield,
        capex_intensity=capex_intensity,
        score=min(score, 10),
        max_score=10,
        reasoning="; ".join(parts) or "Limited data",
    )


def _score_pabrai_double_potential(
    state: MohnishPabraiState,
) -> MohnishPabraiDoublePotential:
    revenues = [v for v in (state.get("revenue_series") or []) if v is not None]
    fcf = [v for v in (state.get("free_cash_flow_series") or []) if v is not None]
    score = 0
    parts: list[str] = []
    rev_growth: float | None = None
    fcf_growth: float | None = None
    if len(revenues) >= 2 and revenues[0] and revenues[0] > 0:
        rev_growth = (revenues[-1] - revenues[0]) / revenues[0]
        if rev_growth > 0.15:
            score += 2
            parts.append(f"Revenue growth {rev_growth:+.1%}")
        elif rev_growth > 0.05:
            score += 1
    if len(fcf) >= 2 and fcf[0] != 0:
        fcf_growth = (fcf[-1] - fcf[0]) / abs(fcf[0])
        if fcf_growth > 0.20:
            score += 3
        elif fcf_growth > 0.08:
            score += 2
        elif fcf_growth > 0:
            score += 1
        parts.append(f"FCF growth {fcf_growth:+.1%}")
    return MohnishPabraiDoublePotential(
        revenue_growth_pct=rev_growth * 100 if rev_growth is not None else None,
        fcf_growth_pct=fcf_growth * 100 if fcf_growth is not None else None,
        score=min(score, 10),
        max_score=10,
        reasoning="; ".join(parts),
    )


# ── Graph nodes ───────────────────────────────────────────────────────────────


def compute_evidence_node(state: MohnishPabraiState) -> dict[str, Any]:
    downside = _score_pabrai_downside(state)
    valuation = _score_pabrai_valuation(state)
    double = _score_pabrai_double_potential(state)
    weighted = 0.45 * downside.score + 0.35 * valuation.score + 0.20 * double.score
    total = downside.score + valuation.score + double.score
    max_total = downside.max_score + valuation.max_score + double.max_score

    evidence = MohnishPabraiEvidence(
        downside_protection=downside,
        valuation=valuation,
        double_potential=double,
        weighted_score=weighted,
        market_cap=state.get("market_cap"),
        total_score=total,
        max_score=max_total,
    )
    return {"evidence": evidence}


async def render_verdict_node(
    state: MohnishPabraiState, config: RunnableConfig
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
        config, "reasoner", schema=MohnishPabraiSignal
    )
    prompt = render_template(
        "personas/mohnish_pabrai.jinja",
        ticker=ticker,
        as_of_date=as_of_date,
        evidence=evidence,
        market_cap=state.get("market_cap"),
        query=query,
    )
    result = cast(
        MohnishPabraiSignal,
        await llm.ainvoke(
            [SystemMessage(prompt), HumanMessage("Render your Pabrai verdict now.")]
        ),
    )
    return {"persona_signals": [result.model_dump()]}


# ── Data-collection sub-agent + subgraph builder ──────────────────────────────


_MCP_TOOLS = [
    "equity_fundamental_metrics",
    "equity_fundamental_income",
    "equity_fundamental_balance",
    "equity_fundamental_cash",
]


async def _build_data_collection_agent(config: RunnableConfig) -> CompiledStateGraph:
    cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = cfg.get_llm_for_role("collector")
    mcp_tools = await get_tools(config, _MCP_TOOLS)
    builder = (
        MuffinAgentBuilder(primary, name="mohnish_pabrai_data_collection")
        .with_fallback_models(*fallbacks)
        .with_state_schema(MohnishPabraiState)
        .with_runtime_system_prompt_template(
            "personas/mohnish_pabrai_data_collection.jinja"
        )
        .with_response_format(MohnishPabraiRawData)
        .with_model_call_limit(run_limit=8, exit_behavior="end")
    )
    for tool in mcp_tools:
        builder = builder.with_tool(tool, run_limit=2)
    builder = builder.with_tool(execute_python, is_cacheable=False)
    return builder.build_react_agent()


async def build_mohnish_pabrai_agent(config: RunnableConfig) -> CompiledStateGraph:
    data_agent = await _build_data_collection_agent(config)
    graph = StateGraph(
        MohnishPabraiState,
        input_schema=MohnishPabraiInput,
        output_schema=MohnishPabraiOutput,
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


