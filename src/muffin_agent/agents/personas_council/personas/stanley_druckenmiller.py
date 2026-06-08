"""Stanley Druckenmiller persona — compiled subgraph.

Macro + momentum + asymmetric R/R. See ``warren_buffett.py`` for reference.
Reference: ``ai-hedge-fund/src/agents/stanley_druckenmiller.py``.
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
from ..tools.scoring_helpers import compute_price_momentum, score_insider_buy_ratio
from ..tools.sentiment import aggregate_news_sentiment

logger = logging.getLogger(__name__)
_LLM_RETRY = RetryPolicy(max_attempts=2)


# ── Typed sub-evidence ────────────────────────────────────────────────────────


class StanleyDruckenmillerGrowthMomentum(BaseModel):
    revenue_cagr: float | None
    eps_cagr: float | None
    momentum_pct: float | None
    score: float
    max_score: float
    reasoning: str


class StanleyDruckenmillerRiskReward(BaseModel):
    debt_to_equity: float | None
    daily_volatility: float | None
    score: float
    max_score: float
    reasoning: str


class StanleyDruckenmillerValuation(BaseModel):
    pe_ratio: float | None
    ev_to_ebit: float | None
    fcf_yield: float | None
    score: float
    max_score: float
    reasoning: str


class StanleyDruckenmillerSentiment(BaseModel):
    bullish_articles: int
    bearish_articles: int
    total_articles: int
    score: float
    max_score: float
    reasoning: str


class StanleyDruckenmillerInsiderActivity(BaseModel):
    raw_insider_score: int
    score: float
    max_score: float
    reasoning: str


class StanleyDruckenmillerEvidence(BaseModel):
    growth_momentum: StanleyDruckenmillerGrowthMomentum
    risk_reward: StanleyDruckenmillerRiskReward
    valuation: StanleyDruckenmillerValuation
    sentiment: StanleyDruckenmillerSentiment
    insider_activity: StanleyDruckenmillerInsiderActivity
    momentum_pct: float | None = None
    weighted_score: float
    market_cap: float | None = None
    total_score: float
    max_score: float


class StanleyDruckenmillerSignal(AnalystSignal):
    agent_id: Literal["stanley_druckenmiller"] = Field(default="stanley_druckenmiller")
    evidence: StanleyDruckenmillerEvidence


# ── RawData ───────────────────────────────────────────────────────────────────


class StanleyDruckenmillerRawData(BaseModel):
    revenue_series: list[float | None] = Field(default_factory=list)
    eps_series: list[float | None] = Field(default_factory=list)
    total_debt_series: list[float | None] = Field(default_factory=list)
    shareholders_equity_series: list[float | None] = Field(default_factory=list)
    pe_ratio_latest: float | None = None
    ev_to_ebit_latest: float | None = None
    fcf_yield_latest: float | None = None
    insider_trades: list[dict[str, Any]] = Field(default_factory=list)
    company_news: list[dict[str, Any]] = Field(default_factory=list)
    prices_1y: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Past 252 days of OHLCV from equity_price_historical (yfinance).",
    )
    market_cap: float | None = None


# ── State ─────────────────────────────────────────────────────────────────────


class StanleyDruckenmillerInput(TypedDict, total=False):
    ticker: str
    as_of_date: str
    query: str | None


class StanleyDruckenmillerOutput(TypedDict, total=False):
    persona_signals: list[dict[str, Any]]


class StanleyDruckenmillerState(AgentState):
    ticker: Annotated[str, OmitFromSchema(input=False, output=True)]
    as_of_date: Annotated[str, OmitFromSchema(input=False, output=True)]
    query: Annotated[str | None, OmitFromSchema(input=False, output=True)]
    revenue_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    eps_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    total_debt_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    shareholders_equity_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    pe_ratio_latest: Annotated[float | None, OmitFromSchema(input=True, output=True)]
    ev_to_ebit_latest: Annotated[float | None, OmitFromSchema(input=True, output=True)]
    fcf_yield_latest: Annotated[float | None, OmitFromSchema(input=True, output=True)]
    insider_trades: Annotated[
        list[dict[str, Any]] | None, OmitFromSchema(input=True, output=True)
    ]
    company_news: Annotated[
        list[dict[str, Any]] | None, OmitFromSchema(input=True, output=True)
    ]
    prices_1y: Annotated[
        list[dict[str, Any]] | None, OmitFromSchema(input=True, output=True)
    ]
    market_cap: Annotated[float | None, OmitFromSchema(input=True, output=True)]
    evidence: Annotated[
        StanleyDruckenmillerEvidence | None, OmitFromSchema(input=True, output=True)
    ]
    persona_signals: Annotated[list[dict], OmitFromSchema(input=True, output=False)]


# ── Composite scorers ─────────────────────────────────────────────────────────


def _cagr(series: list[float | None]) -> float | None:
    vals = [v for v in series if v is not None]
    if len(vals) < 2 or vals[0] is None or vals[0] <= 0 or vals[-1] <= 0:
        return None
    return (vals[-1] / vals[0]) ** (1 / (len(vals) - 1)) - 1


def _score_druckenmiller_growth(
    state: StanleyDruckenmillerState,
) -> tuple[StanleyDruckenmillerGrowthMomentum, float | None]:
    rev_cagr = _cagr(state.get("revenue_series") or [])
    eps_cagr = _cagr(state.get("eps_series") or [])
    prices = state.get("prices_1y") or []
    score = 0
    parts: list[str] = []
    if rev_cagr is not None:
        if rev_cagr > 0.08:
            score += 3
        elif rev_cagr > 0.04:
            score += 2
        elif rev_cagr > 0.01:
            score += 1
        parts.append(f"Rev CAGR {rev_cagr:.1%}")
    if eps_cagr is not None:
        if eps_cagr > 0.08:
            score += 3
        elif eps_cagr > 0.04:
            score += 2
        elif eps_cagr > 0.01:
            score += 1
    close_series = [b.get("close") for b in prices if b.get("close") is not None]
    momentum_pct: float | None = None
    if len(close_series) >= 20:
        mom = compute_price_momentum(close_series)
        momentum_pct = mom["total_return_pct"]
        if momentum_pct is not None:
            if momentum_pct > 50:
                score += 3
                parts.append(f"Price momentum {momentum_pct:.1f}%")
            elif momentum_pct > 20:
                score += 2
            elif momentum_pct > 0:
                score += 1
    normalised = (score / 9) * 10
    return (
        StanleyDruckenmillerGrowthMomentum(
            revenue_cagr=rev_cagr,
            eps_cagr=eps_cagr,
            momentum_pct=momentum_pct,
            score=normalised,
            max_score=10,
            reasoning="; ".join(parts) or "Limited",
        ),
        momentum_pct,
    )


def _score_druckenmiller_risk_reward(
    state: StanleyDruckenmillerState,
) -> StanleyDruckenmillerRiskReward:
    debt = [v for v in (state.get("total_debt_series") or []) if v is not None]
    equity = [
        v for v in (state.get("shareholders_equity_series") or []) if v is not None
    ]
    prices = state.get("prices_1y") or []
    closes = [b.get("close") for b in prices if b.get("close") is not None]
    score = 0
    parts: list[str] = []
    de: float | None = None
    vol: float | None = None
    if debt and equity and equity[-1] and equity[-1] > 0:
        de = debt[-1] / equity[-1]
        if de < 0.3:
            score += 3
        elif de < 0.7:
            score += 2
        elif de < 1.5:
            score += 1
        parts.append(f"D/E {de:.2f}")
    if len(closes) >= 20:
        returns = [
            (closes[i] - closes[i - 1]) / closes[i - 1]
            for i in range(1, len(closes))
            if closes[i - 1] > 0
        ]
        if returns:
            vol = statistics.pstdev(returns)
            if vol < 0.01:
                score += 3
            elif vol < 0.02:
                score += 2
            elif vol < 0.04:
                score += 1
            parts.append(f"Daily vol {vol:.2%}")
    normalised = (score / 6) * 10
    return StanleyDruckenmillerRiskReward(
        debt_to_equity=de,
        daily_volatility=vol,
        score=normalised,
        max_score=10,
        reasoning="; ".join(parts) or "Limited",
    )


def _score_druckenmiller_valuation(
    state: StanleyDruckenmillerState,
) -> StanleyDruckenmillerValuation:
    pe = state.get("pe_ratio_latest")
    ev_to_ebit = state.get("ev_to_ebit_latest")
    fcf_yield = state.get("fcf_yield_latest")
    score = 0
    parts: list[str] = []
    if pe is not None:
        if pe < 15:
            score += 2
        elif pe < 25:
            score += 1
        parts.append(f"P/E {pe:.1f}")
    if ev_to_ebit is not None:
        if ev_to_ebit < 15:
            score += 2
        elif ev_to_ebit < 25:
            score += 1
    if fcf_yield is not None:
        if fcf_yield > 0.05:
            score += 2
        elif fcf_yield > 0.03:
            score += 1
    normalised = (score / 6) * 10
    return StanleyDruckenmillerValuation(
        pe_ratio=pe,
        ev_to_ebit=ev_to_ebit,
        fcf_yield=fcf_yield,
        score=normalised,
        max_score=10,
        reasoning="; ".join(parts) or "Limited",
    )


def _score_druckenmiller_sentiment(
    state: StanleyDruckenmillerState,
) -> StanleyDruckenmillerSentiment:
    articles = state.get("company_news") or []
    agg = aggregate_news_sentiment(articles)
    bullish = int(agg.get("bullish_articles", 0))
    bearish = int(agg.get("bearish_articles", 0))
    total = int(agg.get("total_articles", 0))
    if total == 0:
        return StanleyDruckenmillerSentiment(
            bullish_articles=0,
            bearish_articles=0,
            total_articles=0,
            score=5,
            max_score=10,
            reasoning="No news",
        )
    if bearish / max(total, 1) > 0.30:
        return StanleyDruckenmillerSentiment(
            bullish_articles=bullish,
            bearish_articles=bearish,
            total_articles=total,
            score=3,
            max_score=10,
            reasoning=f"Bearish news ({bearish}/{total})",
        )
    if bullish > bearish:
        return StanleyDruckenmillerSentiment(
            bullish_articles=bullish,
            bearish_articles=bearish,
            total_articles=total,
            score=8,
            max_score=10,
            reasoning=f"Bullish news ({bullish}/{total})",
        )
    return StanleyDruckenmillerSentiment(
        bullish_articles=bullish,
        bearish_articles=bearish,
        total_articles=total,
        score=6,
        max_score=10,
        reasoning="Mixed news",
    )


def _score_druckenmiller_insider(
    state: StanleyDruckenmillerState,
) -> StanleyDruckenmillerInsiderActivity:
    inner = score_insider_buy_ratio(state.get("insider_trades") or [])
    raw = int(inner.score)
    return StanleyDruckenmillerInsiderActivity(
        raw_insider_score=raw,
        score=(raw / 8) * 10,
        max_score=10,
        reasoning=inner.details,
    )


# ── Graph nodes ───────────────────────────────────────────────────────────────


def compute_evidence_node(state: StanleyDruckenmillerState) -> dict[str, Any]:
    growth, momentum_pct = _score_druckenmiller_growth(state)
    risk = _score_druckenmiller_risk_reward(state)
    valuation = _score_druckenmiller_valuation(state)
    sentiment = _score_druckenmiller_sentiment(state)
    insider = _score_druckenmiller_insider(state)
    weighted = (
        0.35 * growth.score
        + 0.20 * risk.score
        + 0.20 * valuation.score
        + 0.15 * sentiment.score
        + 0.10 * insider.score
    )
    total = (
        growth.score + risk.score + valuation.score + sentiment.score + insider.score
    )
    evidence = StanleyDruckenmillerEvidence(
        growth_momentum=growth,
        risk_reward=risk,
        valuation=valuation,
        sentiment=sentiment,
        insider_activity=insider,
        momentum_pct=momentum_pct,
        weighted_score=weighted,
        market_cap=state.get("market_cap"),
        total_score=total,
        max_score=50,
    )
    return {"evidence": evidence}


async def render_verdict_node(
    state: StanleyDruckenmillerState, config: RunnableConfig
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
        config, "reasoner", schema=StanleyDruckenmillerSignal
    )
    prompt = render_template(
        "personas/stanley_druckenmiller.jinja",
        ticker=ticker,
        as_of_date=as_of_date,
        evidence=evidence,
        market_cap=state.get("market_cap"),
        query=query,
    )
    result = cast(
        StanleyDruckenmillerSignal,
        await llm.ainvoke(
            [
                SystemMessage(prompt),
                HumanMessage("Render your Druckenmiller verdict now."),
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
    "equity_price_historical",
    "equity_ownership_insider_trading",
    "news_company",
]


async def _build_data_collection_agent(config: RunnableConfig) -> CompiledStateGraph:
    cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = cfg.get_llm_for_role("collector")
    mcp_tools = await get_tools(config, _MCP_TOOLS)
    builder = (
        MuffinAgentBuilder(primary, name="stanley_druckenmiller_data_collection")
        .with_fallback_models(*fallbacks)
        .with_state_schema(StanleyDruckenmillerState)
        .with_runtime_system_prompt_template(
            "personas/stanley_druckenmiller_data_collection.jinja"
        )
        .with_response_format(StanleyDruckenmillerRawData)
        .with_model_call_limit(run_limit=10, exit_behavior="end")
    )
    for tool in mcp_tools:
        builder = builder.with_tool(tool, run_limit=2)
    builder = builder.with_tool(execute_python, is_cacheable=False)
    return builder.build_react_agent()


async def build_stanley_druckenmiller_agent(
    config: RunnableConfig,
) -> CompiledStateGraph:
    data_agent = await _build_data_collection_agent(config)
    graph = StateGraph(
        StanleyDruckenmillerState,
        input_schema=StanleyDruckenmillerInput,
        output_schema=StanleyDruckenmillerOutput,
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


