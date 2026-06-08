"""Charlie Munger persona — compiled subgraph (collect → compute → verdict).

Quality-weighted (0.35 moat + 0.25 mgmt + 0.25 predictability + 0.15 val).
See ``warren_buffett.py`` for the canonical reference.

Reference: ``ai-hedge-fund/src/agents/charlie_munger.py``.
"""

from __future__ import annotations

import logging
import statistics
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
from ..tools.scoring_helpers import score_insider_buy_ratio

logger = logging.getLogger(__name__)
_LLM_RETRY = RetryPolicy(max_attempts=2)


# ── Typed sub-evidence ────────────────────────────────────────────────────────


class CharlieMungerMoat(BaseModel):
    roic_consistency_pct: float | None
    avg_operating_margin: float | None
    capex_intensity: float | None
    has_active_rd: bool
    has_intangibles: bool
    score: int
    max_score: int
    reasoning: str


class CharlieMungerManagement(BaseModel):
    fcf_to_ni_ratio: float | None
    debt_to_equity_ratio: float | None
    cash_to_revenue_ratio: float | None
    insider_score: int
    shares_trend: Literal["decreasing", "stable", "increasing", "n/a"]
    score: int
    max_score: int
    reasoning: str


class CharlieMungerPredictability(BaseModel):
    revenue_cv: float | None
    op_income_positive_ratio: float | None
    op_margin_cv: float | None
    fcf_positive_ratio: float | None
    score: int
    max_score: int
    reasoning: str


class CharlieMungerValuation(BaseModel):
    fcf_yield: float | None
    margin_of_safety_pct: float | None
    score: int
    max_score: int
    reasoning: str


class CharlieMungerEvidence(BaseModel):
    moat_strength: CharlieMungerMoat
    management_quality: CharlieMungerManagement
    predictability: CharlieMungerPredictability
    valuation: CharlieMungerValuation
    weighted_score: float
    flags: dict[str, bool]
    market_cap: float | None = None
    total_score: float
    max_score: float


class CharlieMungerSignal(AnalystSignal):
    agent_id: Literal["charlie_munger"] = Field(default="charlie_munger")
    evidence: CharlieMungerEvidence


# ── RawData ───────────────────────────────────────────────────────────────────


class CharlieMungerRawData(BaseModel):
    """Munger MCP extraction. Time series oldest -> newest."""

    revenue_series: list[float | None] = Field(default_factory=list)
    operating_income_series: list[float | None] = Field(default_factory=list)
    operating_margin_series: list[float | None] = Field(default_factory=list)
    net_income_series: list[float | None] = Field(default_factory=list)
    free_cash_flow_series: list[float | None] = Field(default_factory=list)
    capital_expenditure_series: list[float | None] = Field(
        default_factory=list,
        description="POSITIVE absolute values, oldest -> newest.",
    )
    research_and_development_series: list[float | None] = Field(default_factory=list)
    return_on_invested_capital_series: list[float | None] = Field(default_factory=list)
    goodwill_and_intangible_assets_series: list[float | None] = Field(
        default_factory=list
    )
    total_debt_series: list[float | None] = Field(default_factory=list)
    shareholders_equity_series: list[float | None] = Field(default_factory=list)
    cash_and_equivalents_series: list[float | None] = Field(default_factory=list)
    outstanding_shares_series: list[float | None] = Field(default_factory=list)
    insider_trades: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Past 12 months of insider trades from equity_ownership_insider_trading. "
            "Each entry must contain signed `transaction_shares` (+ = buy, - = sell)."
        ),
    )
    market_cap: float | None = None


# ── State ─────────────────────────────────────────────────────────────────────


class CharlieMungerInput(TypedDict, total=False):
    ticker: str
    as_of_date: str
    query: str | None


class CharlieMungerOutput(TypedDict, total=False):
    persona_signals: list[dict[str, Any]]


class CharlieMungerState(AgentState):
    ticker: Annotated[str, OmitFromSchema(input=False, output=True)]
    as_of_date: Annotated[str, OmitFromSchema(input=False, output=True)]
    query: Annotated[str | None, OmitFromSchema(input=False, output=True)]
    revenue_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    operating_income_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    operating_margin_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    net_income_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    free_cash_flow_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    capital_expenditure_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    research_and_development_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    return_on_invested_capital_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    goodwill_and_intangible_assets_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    total_debt_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    shareholders_equity_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    cash_and_equivalents_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    outstanding_shares_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    insider_trades: Annotated[
        list[dict[str, Any]] | None, OmitFromSchema(input=True, output=True)
    ]
    market_cap: Annotated[float | None, OmitFromSchema(input=True, output=True)]
    evidence: Annotated[
        CharlieMungerEvidence | None, OmitFromSchema(input=True, output=True)
    ]
    persona_signals: Annotated[list[dict], OmitFromSchema(input=True, output=False)]


# ── Composite scorers ─────────────────────────────────────────────────────────


def _score_munger_moat(state: CharlieMungerState) -> CharlieMungerMoat:
    """Score Munger's moat dimension (max 10).

    Combines ROIC consistency, operating margin, capex intensity, R&D, and
    intangible assets.
    """
    roics = [
        v
        for v in (state.get("return_on_invested_capital_series") or [])
        if v is not None
    ]
    margins = [v for v in (state.get("operating_margin_series") or []) if v is not None]
    capex = [
        abs(v) for v in (state.get("capital_expenditure_series") or []) if v is not None
    ]
    revenues = [v for v in (state.get("revenue_series") or []) if v is not None]
    rd = [
        v for v in (state.get("research_and_development_series") or []) if v is not None
    ]
    goodwill = [
        v
        for v in (state.get("goodwill_and_intangible_assets_series") or [])
        if v is not None
    ]

    score = 0
    parts: list[str] = []
    roic_consistency: float | None = None
    avg_margin: float | None = None
    capex_intensity: float | None = None
    has_rd = bool(rd and any(v > 0 for v in rd))
    has_intangibles = bool(goodwill and any(v > 0 for v in goodwill))

    if roics:
        high_count = sum(1 for r in roics if r > 0.15)
        roic_consistency = high_count / len(roics) * 100
        if high_count / len(roics) >= 0.8:
            score += 3
            parts.append(f"ROIC > 15% in {high_count}/{len(roics)} periods")
        elif high_count / len(roics) >= 0.5:
            score += 2
            parts.append(f"ROIC > 15% in {high_count}/{len(roics)} (decent)")

    if len(margins) >= 3:
        positive = sum(1 for m in margins if m > 0.20)
        avg_margin = sum(margins) / len(margins)
        if positive >= len(margins) * 0.7:
            score += 2
            parts.append("Stable / improving operating margins > 20%")
        elif avg_margin > 0.30:
            score += 1
            parts.append("Average margin > 30%")

    if capex and revenues and revenues[-1]:
        capex_intensity = capex[-1] / revenues[-1]
        if capex_intensity < 0.05:
            score += 2
            parts.append(f"Low capex intensity {capex_intensity:.1%}")
        elif capex_intensity < 0.10:
            score += 1
            parts.append(f"Moderate capex intensity {capex_intensity:.1%}")

    if has_rd:
        score += 1
        parts.append("Active R&D investment")
    if has_intangibles:
        score += 1
        parts.append("Intangible assets present (brand / IP)")

    return CharlieMungerMoat(
        roic_consistency_pct=roic_consistency,
        avg_operating_margin=avg_margin,
        capex_intensity=capex_intensity,
        has_active_rd=has_rd,
        has_intangibles=has_intangibles,
        score=min(score, 10),
        max_score=10,
        reasoning="; ".join(parts),
    )


def _score_munger_management(state: CharlieMungerState) -> CharlieMungerManagement:
    """Score Munger's management dimension (max 10).

    Combines FCF/NI quality, leverage, cash position, insider activity, and
    share-count trajectory.
    """
    fcf = [v for v in (state.get("free_cash_flow_series") or []) if v is not None]
    net_income = [v for v in (state.get("net_income_series") or []) if v is not None]
    total_debt = [v for v in (state.get("total_debt_series") or []) if v is not None]
    equity = [
        v for v in (state.get("shareholders_equity_series") or []) if v is not None
    ]
    cash = [
        v for v in (state.get("cash_and_equivalents_series") or []) if v is not None
    ]
    revenues = [v for v in (state.get("revenue_series") or []) if v is not None]
    shares = [
        v for v in (state.get("outstanding_shares_series") or []) if v is not None
    ]
    insider_trades = state.get("insider_trades") or []

    score = 0
    parts: list[str] = []
    fcf_ni_ratio: float | None = None
    de_ratio: float | None = None
    cash_ratio: float | None = None
    shares_trend: Literal["decreasing", "stable", "increasing", "n/a"] = "n/a"

    if fcf and net_income and net_income[-1] != 0:
        fcf_ni_ratio = fcf[-1] / net_income[-1]
        if fcf_ni_ratio > 1.1:
            score += 3
            parts.append(f"FCF/NI {fcf_ni_ratio:.2f} (excellent earnings quality)")
        elif fcf_ni_ratio > 0.9:
            score += 2
            parts.append(f"FCF/NI {fcf_ni_ratio:.2f} (good)")
        elif fcf_ni_ratio > 0.7:
            score += 1
            parts.append(f"FCF/NI {fcf_ni_ratio:.2f}")

    if total_debt and equity and equity[-1] and equity[-1] > 0:
        de_ratio = total_debt[-1] / equity[-1]
        if de_ratio < 0.3:
            score += 3
            parts.append(f"D/E {de_ratio:.2f} (very conservative)")
        elif de_ratio < 0.7:
            score += 2
            parts.append(f"D/E {de_ratio:.2f}")
        elif de_ratio < 1.5:
            score += 1

    if cash and revenues and revenues[-1] and revenues[-1] > 0:
        cash_ratio = cash[-1] / revenues[-1]
        if 0.10 <= cash_ratio <= 0.25:
            score += 2
            parts.append(f"Sensible cash position {cash_ratio:.1%} of revenue")
        elif 0.05 <= cash_ratio < 0.10 or 0.25 < cash_ratio <= 0.40:
            score += 1

    insider_score_detail = score_insider_buy_ratio(insider_trades)
    insider_score_val = int(insider_score_detail.score)
    if insider_score_val >= 8:
        score += 2
        parts.append("Heavy insider buying")
    elif insider_score_val >= 6:
        score += 1
        parts.append("Balanced insider activity")
    elif insider_score_val < 5:
        score -= 1
        parts.append("Net insider selling")

    # ai-hedge-fund parity (oldest -> newest series): meaningful buybacks
    # require a >=5% reduction; "stable" is within +-5%; the dilution penalty
    # only triggers above +20% share growth.
    if len(shares) >= 2 and shares[0] > 0:
        oldest, newest = shares[0], shares[-1]
        if newest < oldest * 0.95:
            score += 2
            shares_trend = "decreasing"
            parts.append("Share count down >=5% (buybacks)")
        elif abs(newest - oldest) / oldest < 0.05:
            score += 1
            shares_trend = "stable"
        elif newest > oldest * 1.20:
            score -= 1
            shares_trend = "increasing"
            parts.append("Shares diluting >20%")
        else:
            shares_trend = "increasing"

    return CharlieMungerManagement(
        fcf_to_ni_ratio=fcf_ni_ratio,
        debt_to_equity_ratio=de_ratio,
        cash_to_revenue_ratio=cash_ratio,
        insider_score=insider_score_val,
        shares_trend=shares_trend,
        score=max(0, min(score, 10)),
        max_score=10,
        reasoning="; ".join(parts),
    )


def _score_munger_predictability(
    state: CharlieMungerState,
) -> CharlieMungerPredictability:
    """Score Munger's predictability dimension (max 10).

    Looks at revenue / operating-income / operating-margin / FCF stability.
    """
    revenues = [v for v in (state.get("revenue_series") or []) if v is not None]
    op_income = [
        v for v in (state.get("operating_income_series") or []) if v is not None
    ]
    op_margins = [
        v for v in (state.get("operating_margin_series") or []) if v is not None
    ]
    fcf = [v for v in (state.get("free_cash_flow_series") or []) if v is not None]

    score = 0
    parts: list[str] = []
    revenue_cv: float | None = None
    op_income_positive_ratio: float | None = None
    op_margin_cv: float | None = None
    fcf_positive_ratio: float | None = None

    if len(revenues) >= 4:
        mean = sum(revenues) / len(revenues)
        if mean > 0:
            revenue_cv = statistics.pstdev(revenues) / mean
            if revenue_cv < 0.10:
                score += 3
                parts.append(f"Revenue stable (CV {revenue_cv:.1%})")
            elif revenue_cv < 0.20:
                score += 2
                parts.append(f"Revenue moderately stable (CV {revenue_cv:.1%})")

    if op_income:
        positives = sum(1 for v in op_income if v > 0)
        op_income_positive_ratio = positives / len(op_income)
        if op_income_positive_ratio == 1.0:
            score += 3
            parts.append("Operating income positive every period")
        elif op_income_positive_ratio >= 0.8:
            score += 2
        elif op_income_positive_ratio >= 0.6:
            score += 1

    if len(op_margins) >= 4:
        mean = sum(op_margins) / len(op_margins)
        if mean > 0:
            op_margin_cv = statistics.pstdev(op_margins) / mean
            if op_margin_cv < 0.03:
                score += 2
                parts.append("Highly stable margins")
            elif op_margin_cv < 0.07:
                score += 1

    if fcf:
        positives = sum(1 for v in fcf if v > 0)
        fcf_positive_ratio = positives / len(fcf)
        if fcf_positive_ratio == 1.0:
            score += 2
            parts.append("FCF positive every period")
        elif fcf_positive_ratio >= 0.8:
            score += 1

    return CharlieMungerPredictability(
        revenue_cv=revenue_cv,
        op_income_positive_ratio=op_income_positive_ratio,
        op_margin_cv=op_margin_cv,
        fcf_positive_ratio=fcf_positive_ratio,
        score=min(score, 10),
        max_score=10,
        reasoning="; ".join(parts),
    )


def _score_munger_valuation(
    fcf_latest: float | None, market_cap: float | None
) -> CharlieMungerValuation:
    """Valuation dimension (max 10): FCF yield + 15x FCF margin of safety."""
    if not fcf_latest or fcf_latest <= 0 or not market_cap or market_cap <= 0:
        return CharlieMungerValuation(
            fcf_yield=None,
            margin_of_safety_pct=None,
            score=0,
            max_score=10,
            reasoning="No FCF or market cap available",
        )
    fcf_yield = fcf_latest / market_cap
    fair_value = fcf_latest * 15
    mos = (fair_value - market_cap) / market_cap

    score = 0
    parts: list[str] = []
    if fcf_yield > 0.08:
        score += 4
        parts.append(f"FCF yield {fcf_yield:.1%} (very attractive)")
    elif fcf_yield > 0.05:
        score += 3
        parts.append(f"FCF yield {fcf_yield:.1%}")
    elif fcf_yield > 0.03:
        score += 1

    if mos > 0.3:
        score += 3
        parts.append(f"MoS {mos:.1%} at 15x FCF multiple")
    elif mos > 0.1:
        score += 2
        parts.append(f"MoS {mos:.1%}")
    elif mos > -0.1:
        score += 1

    return CharlieMungerValuation(
        fcf_yield=fcf_yield,
        margin_of_safety_pct=mos * 100,
        score=min(score, 10),
        max_score=10,
        reasoning="; ".join(parts),
    )


# ── Graph nodes ───────────────────────────────────────────────────────────────


def compute_evidence_node(state: CharlieMungerState) -> dict[str, Any]:
    moat = _score_munger_moat(state)
    management = _score_munger_management(state)
    predictability = _score_munger_predictability(state)
    fcf = state.get("free_cash_flow_series") or []
    fcf_latest = fcf[-1] if fcf else None
    valuation = _score_munger_valuation(fcf_latest, state.get("market_cap"))

    weighted = (
        0.35 * moat.score
        + 0.25 * management.score
        + 0.25 * predictability.score
        + 0.15 * valuation.score
    )
    flags = {
        "moat_strong": moat.score >= 7,
        "predictable": predictability.score >= 7,
        "owner_aligned": management.score >= 7,
        "valuation_ok": valuation.score >= 5,
    }
    total = moat.score + management.score + predictability.score + valuation.score
    max_total = (
        moat.max_score
        + management.max_score
        + predictability.max_score
        + valuation.max_score
    )

    evidence = CharlieMungerEvidence(
        moat_strength=moat,
        management_quality=management,
        predictability=predictability,
        valuation=valuation,
        weighted_score=weighted,
        flags=flags,
        market_cap=state.get("market_cap"),
        total_score=total,
        max_score=max_total,
    )
    return {"evidence": evidence}


async def render_verdict_node(
    state: CharlieMungerState, config: RunnableConfig
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
        config, "reasoner", schema=CharlieMungerSignal
    )
    prompt = render_template(
        "personas/charlie_munger.jinja",
        ticker=ticker,
        as_of_date=as_of_date,
        evidence=evidence,
        market_cap=state.get("market_cap"),
        query=query,
    )
    result = cast(
        CharlieMungerSignal,
        await llm.ainvoke(
            [SystemMessage(prompt), HumanMessage("Render your Munger verdict now.")]
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
    "equity_ownership_insider_trading",
]


async def _build_data_collection_agent(config: RunnableConfig) -> CompiledStateGraph:
    cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = cfg.get_llm_for_role("collector")
    mcp_tools = await get_tools(config, _MCP_TOOLS)
    builder = (
        MuffinAgentBuilder(primary, name="charlie_munger_data_collection")
        .with_fallback_models(*fallbacks)
        .with_state_schema(CharlieMungerState)
        .with_runtime_system_prompt_template(
            "personas/charlie_munger_data_collection.jinja"
        )
        .with_response_format(CharlieMungerRawData)
        .with_model_call_limit(run_limit=8, exit_behavior="end")
    )
    for tool in mcp_tools:
        builder = builder.with_tool(tool, run_limit=2)
    builder = builder.with_tool(execute_python, is_cacheable=False)
    return builder.build_react_agent()


async def build_charlie_munger_agent(config: RunnableConfig) -> CompiledStateGraph:
    data_agent = await _build_data_collection_agent(config)
    graph = StateGraph(
        CharlieMungerState,
        input_schema=CharlieMungerInput,
        output_schema=CharlieMungerOutput,
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


