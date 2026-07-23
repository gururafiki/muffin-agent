"""Phil Fisher persona — compiled subgraph (collect → compute → verdict).

Qualitative growth + R&D + management. See ``warren_buffett.py`` for canonical
reference. Reference: ``ai-hedge-fund/src/agents/phil_fisher.py``.
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
from ..schemas import AnalystSignal, merge_subagent_tree, merge_tool_runs
from ..tools.scoring_helpers import score_insider_buy_ratio
from ..tools.sentiment import aggregate_news_sentiment

logger = logging.getLogger(__name__)
_LLM_RETRY = RetryPolicy(max_attempts=2)


# ── Typed sub-evidence ────────────────────────────────────────────────────────


class PhilFisherGrowthQuality(BaseModel):
    revenue_cagr: float | None
    eps_cagr: float | None
    rd_intensity: float | None
    score: float
    max_score: float
    reasoning: str


class PhilFisherMarginsStability(BaseModel):
    latest_operating_margin: float | None
    latest_gross_margin: float | None
    operating_margin_cv: float | None
    score: float
    max_score: float
    reasoning: str


class PhilFisherManagementEfficiency(BaseModel):
    return_on_equity: float | None
    debt_to_equity: float | None
    fcf_positive_ratio: float | None
    score: float
    max_score: float
    reasoning: str


class PhilFisherValuation(BaseModel):
    pe_ratio: float | None
    score: float
    max_score: float
    reasoning: str


class PhilFisherSentiment(BaseModel):
    bullish_articles: int
    bearish_articles: int
    total_articles: int
    score: float
    max_score: float
    reasoning: str


class PhilFisherInsiderActivity(BaseModel):
    raw_insider_score: int
    score: float
    max_score: float
    reasoning: str


class PhilFisherEvidence(BaseModel):
    growth_quality: PhilFisherGrowthQuality
    margins_stability: PhilFisherMarginsStability
    management_efficiency: PhilFisherManagementEfficiency
    valuation: PhilFisherValuation
    sentiment: PhilFisherSentiment
    insider_activity: PhilFisherInsiderActivity
    weighted_score: float
    market_cap: float | None = None
    total_score: float
    max_score: float


class PhilFisherSignal(AnalystSignal):
    agent_id: Literal["phil_fisher"] = Field(default="phil_fisher")
    evidence: PhilFisherEvidence


# ── RawData ───────────────────────────────────────────────────────────────────


class PhilFisherRawData(BaseModel):
    """Fisher MCP extraction. Series oldest -> newest."""

    revenue_series: list[float | None] = Field(default_factory=list)
    eps_series: list[float | None] = Field(default_factory=list)
    research_and_development_series: list[float | None] = Field(default_factory=list)
    operating_margin_series: list[float | None] = Field(default_factory=list)
    gross_margin_series: list[float | None] = Field(default_factory=list)
    free_cash_flow_series: list[float | None] = Field(default_factory=list)
    roe_latest: float | None = None
    debt_to_equity_latest: float | None = None
    pe_ratio_latest: float | None = None
    insider_trades: list[dict[str, Any]] = Field(default_factory=list)
    company_news: list[dict[str, Any]] = Field(default_factory=list)
    market_cap: float | None = None


# ── State ─────────────────────────────────────────────────────────────────────


class PhilFisherInput(TypedDict, total=False):
    ticker: str
    as_of_date: str
    query: str | None


class PhilFisherOutput(TypedDict, total=False):
    persona_signals: list[dict[str, Any]]
    tool_runs: list[dict[str, Any]]
    subagent_tree: dict[str, Any]


class PhilFisherState(AgentState):
    ticker: Annotated[str, OmitFromSchema(input=False, output=True)]
    as_of_date: Annotated[str, OmitFromSchema(input=False, output=True)]
    query: Annotated[str | None, OmitFromSchema(input=False, output=True)]
    revenue_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    eps_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    research_and_development_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    operating_margin_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    gross_margin_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    free_cash_flow_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    roe_latest: Annotated[float | None, OmitFromSchema(input=True, output=False)]
    debt_to_equity_latest: Annotated[
        float | None, OmitFromSchema(input=True, output=False)
    ]
    pe_ratio_latest: Annotated[float | None, OmitFromSchema(input=True, output=False)]
    insider_trades: Annotated[
        list[dict[str, Any]] | None, OmitFromSchema(input=True, output=False)
    ]
    company_news: Annotated[
        list[dict[str, Any]] | None, OmitFromSchema(input=True, output=False)
    ]
    market_cap: Annotated[float | None, OmitFromSchema(input=True, output=False)]
    evidence: Annotated[
        PhilFisherEvidence | None, OmitFromSchema(input=True, output=False)
    ]
    persona_signals: Annotated[list[dict], OmitFromSchema(input=True, output=False)]
    tool_runs: Annotated[list[dict[str, Any]], merge_tool_runs]
    subagent_tree: Annotated[dict[str, Any], merge_subagent_tree]


# ── Composite scorers ─────────────────────────────────────────────────────────


def _cagr(series: list[float | None]) -> float | None:
    vals = [v for v in series if v is not None]
    if len(vals) < 2 or vals[0] is None or vals[0] <= 0:
        return None
    if vals[-1] <= 0:
        return None
    return (vals[-1] / vals[0]) ** (1 / (len(vals) - 1)) - 1


def _score_fisher_growth(state: PhilFisherState) -> PhilFisherGrowthQuality:
    revenues = [v for v in (state.get("revenue_series") or []) if v is not None]
    eps = [v for v in (state.get("eps_series") or []) if v is not None]
    rd = [
        v for v in (state.get("research_and_development_series") or []) if v is not None
    ]
    score = 0
    parts: list[str] = []
    rev_cagr = _cagr(revenues)
    eps_cagr = _cagr(eps)
    rd_intensity: float | None = None

    if rev_cagr is not None:
        if rev_cagr > 0.20:
            score += 3
            parts.append(f"Rev CAGR {rev_cagr:.1%}")
        elif rev_cagr > 0.10:
            score += 2
        elif rev_cagr > 0.03:
            score += 1
    if eps_cagr is not None:
        if eps_cagr > 0.20:
            score += 3
            parts.append(f"EPS CAGR {eps_cagr:.1%}")
        elif eps_cagr > 0.10:
            score += 2
        elif eps_cagr > 0.03:
            score += 1
    if rd and revenues and revenues[-1]:
        rd_intensity = rd[-1] / revenues[-1]
        if 0.03 <= rd_intensity <= 0.15:
            score += 3
            parts.append(f"R&D/rev {rd_intensity:.1%} (Fisher zone)")
        elif rd_intensity > 0.15:
            score += 2
        elif rd_intensity > 0:
            score += 1
    normalised = (score / 9) * 10
    return PhilFisherGrowthQuality(
        revenue_cagr=rev_cagr,
        eps_cagr=eps_cagr,
        rd_intensity=rd_intensity,
        score=normalised,
        max_score=10,
        reasoning="; ".join(parts) or "Limited data",
    )


def _score_fisher_margins(state: PhilFisherState) -> PhilFisherMarginsStability:
    op_margins = [
        v for v in (state.get("operating_margin_series") or []) if v is not None
    ]
    gross_margins = [
        v for v in (state.get("gross_margin_series") or []) if v is not None
    ]
    score = 0
    parts: list[str] = []
    om_cv: float | None = None

    if op_margins:
        if op_margins[-1] is not None and op_margins[-1] > 0:
            score += 1
        if len(op_margins) >= 2 and op_margins[-1] >= op_margins[0]:
            score += 2
            parts.append("Stable / improving op margin")
    if gross_margins:
        if gross_margins[-1] is not None and gross_margins[-1] > 0.50:
            score += 2
            parts.append(f"Gross margin {gross_margins[-1]:.1%}")
        elif gross_margins[-1] is not None and gross_margins[-1] > 0.30:
            score += 1
    if op_margins and len(op_margins) >= 3:
        mean = sum(op_margins) / len(op_margins)
        if mean > 0:
            om_cv = statistics.pstdev(op_margins) / mean
            if om_cv < 0.02:
                score += 2
                parts.append(f"Highly stable op margin (CV {om_cv:.1%})")
            elif om_cv < 0.05:
                score += 1
    normalised = (score / 6) * 10
    return PhilFisherMarginsStability(
        latest_operating_margin=op_margins[-1] if op_margins else None,
        latest_gross_margin=gross_margins[-1] if gross_margins else None,
        operating_margin_cv=om_cv,
        score=normalised,
        max_score=10,
        reasoning="; ".join(parts) or "Limited data",
    )


def _score_fisher_management(
    state: PhilFisherState,
) -> PhilFisherManagementEfficiency:
    roe = state.get("roe_latest")
    de = state.get("debt_to_equity_latest")
    fcf = [v for v in (state.get("free_cash_flow_series") or []) if v is not None]
    score = 0
    parts: list[str] = []
    fcf_pos_ratio: float | None = None

    if roe is not None:
        if roe > 0.20:
            score += 3
            parts.append(f"ROE {roe:.1%}")
        elif roe > 0.10:
            score += 2
        elif roe > 0:
            score += 1
    if de is not None:
        if de < 0.3:
            score += 2
            parts.append(f"D/E {de:.2f}")
        elif de < 1.0:
            score += 1
    if fcf:
        positives = sum(1 for v in fcf if v > 0)
        fcf_pos_ratio = positives / len(fcf)
        if fcf_pos_ratio >= 0.8:
            score += 1
            parts.append(f"FCF positive {positives}/{len(fcf)}")
    normalised = (score / 6) * 10
    return PhilFisherManagementEfficiency(
        return_on_equity=roe,
        debt_to_equity=de,
        fcf_positive_ratio=fcf_pos_ratio,
        score=normalised,
        max_score=10,
        reasoning="; ".join(parts) or "Limited data",
    )


def _score_fisher_valuation(state: PhilFisherState) -> PhilFisherValuation:
    pe = state.get("pe_ratio_latest")
    score = 0
    parts: list[str] = []
    if pe is not None:
        if pe < 20:
            score += 2
            parts.append(f"P/E {pe:.1f}")
        elif pe < 30:
            score += 1
    normalised = (score / 4) * 10
    return PhilFisherValuation(
        pe_ratio=pe,
        score=normalised,
        max_score=10,
        reasoning="; ".join(parts) or "Cannot value",
    )


def _score_fisher_sentiment(state: PhilFisherState) -> PhilFisherSentiment:
    articles = state.get("company_news") or []
    agg = aggregate_news_sentiment(articles)
    bullish = int(agg.get("bullish_articles", 0))
    bearish = int(agg.get("bearish_articles", 0))
    total = int(agg.get("total_articles", 0))
    if total == 0:
        return PhilFisherSentiment(
            bullish_articles=0,
            bearish_articles=0,
            total_articles=0,
            score=5,
            max_score=10,
            reasoning="No news — neutral",
        )
    if bearish / max(total, 1) > 0.30:
        return PhilFisherSentiment(
            bullish_articles=bullish,
            bearish_articles=bearish,
            total_articles=total,
            score=3,
            max_score=10,
            reasoning=f"Bearish news ({bearish}/{total})",
        )
    if bullish > bearish:
        return PhilFisherSentiment(
            bullish_articles=bullish,
            bearish_articles=bearish,
            total_articles=total,
            score=8,
            max_score=10,
            reasoning=f"Bullish news ({bullish}/{total})",
        )
    return PhilFisherSentiment(
        bullish_articles=bullish,
        bearish_articles=bearish,
        total_articles=total,
        score=6,
        max_score=10,
        reasoning="Mixed news",
    )


def _score_fisher_insider(state: PhilFisherState) -> PhilFisherInsiderActivity:
    inner = score_insider_buy_ratio(state.get("insider_trades") or [])
    raw = int(inner.score)
    return PhilFisherInsiderActivity(
        raw_insider_score=raw,
        score=(raw / 8) * 10,
        max_score=10,
        reasoning=inner.details,
    )


# ── Graph nodes ───────────────────────────────────────────────────────────────


def compute_evidence_node(state: PhilFisherState) -> dict[str, Any]:
    growth = _score_fisher_growth(state)
    margins = _score_fisher_margins(state)
    mgmt = _score_fisher_management(state)
    valuation = _score_fisher_valuation(state)
    sentiment = _score_fisher_sentiment(state)
    insider = _score_fisher_insider(state)
    weighted = (
        0.30 * growth.score
        + 0.25 * mgmt.score
        + 0.20 * margins.score
        + 0.15 * valuation.score
        + 0.05 * sentiment.score
        + 0.05 * insider.score
    )
    total = (
        growth.score
        + margins.score
        + mgmt.score
        + valuation.score
        + sentiment.score
        + insider.score
    )
    evidence = PhilFisherEvidence(
        growth_quality=growth,
        margins_stability=margins,
        management_efficiency=mgmt,
        valuation=valuation,
        sentiment=sentiment,
        insider_activity=insider,
        weighted_score=weighted,
        market_cap=state.get("market_cap"),
        total_score=total,
        max_score=60,
    )
    return {"evidence": evidence}


async def render_verdict_node(
    state: PhilFisherState, config: RunnableConfig
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
        config, "reasoner", schema=PhilFisherSignal
    )
    prompt = render_template(
        "personas_council/personas/phil_fisher.jinja",
        ticker=ticker,
        as_of_date=as_of_date,
        evidence=evidence,
        market_cap=state.get("market_cap"),
        query=query,
    )
    result = cast(
        PhilFisherSignal,
        await llm.ainvoke(
            [SystemMessage(prompt), HumanMessage("Render your Fisher verdict now.")]
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
    "equity_fundamental_revenue_per_segment",
    "equity_ownership_insider_trading",
    "news_company",
]


async def _build_data_collection_agent(config: RunnableConfig) -> CompiledStateGraph:
    cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = cfg.get_llm_for_role("collector")
    mcp_tools = await get_tools(config, _MCP_TOOLS)
    builder = (
        MuffinAgentBuilder(primary, name="phil_fisher_data_collection")
        .with_fallback_models(*fallbacks)
        .with_state_schema(PhilFisherState)
        .with_input_prompt_template(
            "personas_council/personas/phil_fisher_data_collection.jinja"
        )
        .with_response_format(PhilFisherRawData)
        .with_model_call_limit(run_limit=8, exit_behavior="end")
    )
    for tool in mcp_tools:
        builder = builder.with_tool(tool, run_limit=2)
    builder = builder.with_tool(execute_python, is_cacheable=False)
    return builder.build_react_agent()


async def build_phil_fisher_agent(config: RunnableConfig) -> CompiledStateGraph:
    data_agent = await _build_data_collection_agent(config)
    graph = StateGraph(
        PhilFisherState,
        input_schema=PhilFisherInput,
        output_schema=PhilFisherOutput,
    )
    graph.add_node(
        "collect_data",
        data_agent,
        input_schema=PhilFisherInput,
        retry_policy=_LLM_RETRY,
    )
    graph.add_node("compute_evidence", compute_evidence_node)
    graph.add_node("render_verdict", render_verdict_node, retry_policy=_LLM_RETRY)
    graph.add_edge(START, "collect_data")
    graph.add_edge("collect_data", "compute_evidence")
    graph.add_edge("compute_evidence", "render_verdict")
    graph.add_edge("render_verdict", END)
    return graph.compile()
