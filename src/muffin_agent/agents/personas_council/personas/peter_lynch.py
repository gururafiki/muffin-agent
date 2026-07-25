"""Peter Lynch persona — compiled subgraph (collect → compute → verdict).

GARP via PEG; ten-bagger hunt. See ``warren_buffett.py`` for canonical reference.
Reference: ``ai-hedge-fund/src/agents/peter_lynch.py``.
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
from ..schemas import AnalystSignal, merge_subagent_tree, merge_tool_runs
from ..tools.scoring_helpers import compute_peg_ratio, score_insider_buy_ratio
from ..tools.sentiment import aggregate_news_sentiment

logger = logging.getLogger(__name__)
_LLM_RETRY = RetryPolicy(max_attempts=2)


# ── Typed sub-evidence ────────────────────────────────────────────────────────


class PeterLynchGrowth(BaseModel):
    revenue_cagr: float | None
    eps_cagr: float | None
    score: float
    max_score: float
    reasoning: str


class PeterLynchFundamentals(BaseModel):
    debt_to_equity: float | None
    operating_margin: float | None
    fcf_latest: float | None
    score: float
    max_score: float
    reasoning: str


class PeterLynchValuation(BaseModel):
    pe_ratio: float | None
    peg_ratio: float | None
    score: float
    max_score: float
    reasoning: str


class PeterLynchSentiment(BaseModel):
    bullish_articles: int
    bearish_articles: int
    total_articles: int
    score: float
    max_score: float
    reasoning: str


class PeterLynchInsiderActivity(BaseModel):
    raw_insider_score: int
    score: float
    max_score: float
    reasoning: str


class PeterLynchEvidence(BaseModel):
    growth: PeterLynchGrowth
    fundamentals: PeterLynchFundamentals
    valuation: PeterLynchValuation
    sentiment: PeterLynchSentiment
    insider_activity: PeterLynchInsiderActivity
    peg_ratio: float | None = None
    weighted_score: float
    market_cap: float | None = None
    total_score: float
    max_score: float


class PeterLynchSignal(AnalystSignal):
    agent_id: Literal["peter_lynch"] = Field(default="peter_lynch")
    evidence: PeterLynchEvidence


# ── RawData ───────────────────────────────────────────────────────────────────


class PeterLynchRawData(BaseModel):
    """Lynch MCP extraction. Series oldest -> newest."""

    revenue_series: list[float | None] = Field(default_factory=list)
    eps_series: list[float | None] = Field(default_factory=list)
    free_cash_flow_series: list[float | None] = Field(default_factory=list)
    debt_to_equity_latest: float | None = None
    operating_margin_latest: float | None = None
    pe_ratio_latest: float | None = None
    insider_trades: list[dict[str, Any]] = Field(default_factory=list)
    company_news: list[dict[str, Any]] = Field(default_factory=list)
    market_cap: float | None = None


# ── State ─────────────────────────────────────────────────────────────────────


class PeterLynchInput(TypedDict, total=False):
    ticker: str
    as_of_date: str
    query: str | None


class PeterLynchOutput(TypedDict, total=False):
    persona_signals: list[dict[str, Any]]
    tool_runs: list[dict[str, Any]]
    subagent_tree: dict[str, Any]


class PeterLynchState(AgentState):
    ticker: Annotated[str, OmitFromSchema(input=False, output=True)]
    as_of_date: Annotated[str, OmitFromSchema(input=False, output=True)]
    query: Annotated[str | None, OmitFromSchema(input=False, output=True)]
    revenue_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    eps_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    free_cash_flow_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    debt_to_equity_latest: Annotated[
        float | None, OmitFromSchema(input=True, output=False)
    ]
    operating_margin_latest: Annotated[
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
        PeterLynchEvidence | None, OmitFromSchema(input=True, output=False)
    ]
    persona_signals: Annotated[list[dict], OmitFromSchema(input=True, output=False)]
    tool_runs: Annotated[list[dict[str, Any]], merge_tool_runs]
    subagent_tree: Annotated[dict[str, Any], merge_subagent_tree]


# ── Composite scorers ─────────────────────────────────────────────────────────


def _cagr(series: list[float | None]) -> float | None:
    vals = [v for v in series if v is not None]
    if len(vals) < 2 or vals[0] is None or vals[0] <= 0:
        return None
    years = len(vals) - 1
    if vals[-1] <= 0:
        return None
    return (vals[-1] / vals[0]) ** (1 / years) - 1


def _score_lynch_growth(state: PeterLynchState) -> PeterLynchGrowth:
    rev_cagr = _cagr(state.get("revenue_series") or [])
    eps_cagr = _cagr(state.get("eps_series") or [])
    score = 0
    parts: list[str] = []
    if rev_cagr is not None:
        if rev_cagr > 0.25:
            score += 3
            parts.append(f"Revenue CAGR {rev_cagr:.1%}")
        elif rev_cagr > 0.10:
            score += 2
        elif rev_cagr > 0.02:
            score += 1
    if eps_cagr is not None:
        if eps_cagr > 0.25:
            score += 3
            parts.append(f"EPS CAGR {eps_cagr:.1%}")
        elif eps_cagr > 0.10:
            score += 2
        elif eps_cagr > 0.02:
            score += 1
    normalised = (score / 6) * 10
    return PeterLynchGrowth(
        revenue_cagr=rev_cagr,
        eps_cagr=eps_cagr,
        score=normalised,
        max_score=10,
        reasoning="; ".join(parts) or "Limited growth data",
    )


def _score_lynch_fundamentals(state: PeterLynchState) -> PeterLynchFundamentals:
    de = state.get("debt_to_equity_latest")
    om = state.get("operating_margin_latest")
    fcf_series = state.get("free_cash_flow_series") or []
    fcf_latest = fcf_series[-1] if fcf_series else None
    score = 0
    parts: list[str] = []
    if de is not None:
        if de < 0.5:
            score += 2
            parts.append(f"D/E {de:.2f}")
        elif de < 1.0:
            score += 1
    if om is not None:
        if om > 0.20:
            score += 2
            parts.append(f"Op margin {om:.1%}")
        elif om > 0.10:
            score += 1
    if fcf_latest is not None and fcf_latest > 0:
        score += 2
        parts.append(f"FCF positive {fcf_latest:,.0f}")
    normalised = (score / 6) * 10
    return PeterLynchFundamentals(
        debt_to_equity=de,
        operating_margin=om,
        fcf_latest=fcf_latest,
        score=normalised,
        max_score=10,
        reasoning="; ".join(parts) or "Limited fundamentals",
    )


def _score_lynch_valuation(
    state: PeterLynchState,
) -> tuple[PeterLynchValuation, float | None]:
    pe = state.get("pe_ratio_latest")
    eps_cagr = _cagr(state.get("eps_series") or [])
    peg = compute_peg_ratio(pe, eps_cagr)
    score = 0
    parts: list[str] = []
    if pe is not None:
        if pe < 15:
            score += 2
            parts.append(f"P/E {pe:.1f}")
        elif pe < 25:
            score += 1
    if peg is not None:
        if peg < 1:
            score += 3
            parts.append(f"PEG {peg:.2f} (<1: cheap)")
        elif peg < 2:
            score += 2
        elif peg < 3:
            score += 1
    normalised = (score / 5) * 10
    return (
        PeterLynchValuation(
            pe_ratio=pe,
            peg_ratio=peg,
            score=normalised,
            max_score=10,
            reasoning="; ".join(parts) or "Cannot compute PEG",
        ),
        peg,
    )


def _score_lynch_sentiment(state: PeterLynchState) -> PeterLynchSentiment:
    articles = state.get("company_news") or []
    agg = aggregate_news_sentiment(articles)
    bullish = int(agg.get("bullish_articles", 0))
    bearish = int(agg.get("bearish_articles", 0))
    total = int(agg.get("total_articles", 0))
    if total == 0:
        return PeterLynchSentiment(
            bullish_articles=0,
            bearish_articles=0,
            total_articles=0,
            score=5,
            max_score=10,
            reasoning="No news data — neutral default",
        )
    if bearish / max(total, 1) > 0.30:
        return PeterLynchSentiment(
            bullish_articles=bullish,
            bearish_articles=bearish,
            total_articles=total,
            score=3,
            max_score=10,
            reasoning=f"Bearish-leaning news ({bearish}/{total})",
        )
    if bullish > bearish:
        return PeterLynchSentiment(
            bullish_articles=bullish,
            bearish_articles=bearish,
            total_articles=total,
            score=8,
            max_score=10,
            reasoning=f"Bullish news ({bullish}/{total})",
        )
    return PeterLynchSentiment(
        bullish_articles=bullish,
        bearish_articles=bearish,
        total_articles=total,
        score=6,
        max_score=10,
        reasoning=f"Mixed news ({bullish}/{bearish}/{total})",
    )


def _score_lynch_insider(state: PeterLynchState) -> PeterLynchInsiderActivity:
    insider_trades = state.get("insider_trades") or []
    inner = score_insider_buy_ratio(insider_trades)
    raw = int(inner.score)
    return PeterLynchInsiderActivity(
        raw_insider_score=raw,
        score=(raw / 8) * 10,
        max_score=10,
        reasoning=inner.details,
    )


# ── Graph nodes ───────────────────────────────────────────────────────────────


def compute_evidence_node(state: PeterLynchState) -> dict[str, Any]:
    growth = _score_lynch_growth(state)
    fundamentals = _score_lynch_fundamentals(state)
    valuation, peg = _score_lynch_valuation(state)
    sentiment = _score_lynch_sentiment(state)
    insider = _score_lynch_insider(state)
    weighted = (
        0.30 * growth.score
        + 0.25 * valuation.score
        + 0.20 * fundamentals.score
        + 0.15 * sentiment.score
        + 0.10 * insider.score
    )
    total = (
        growth.score
        + valuation.score
        + fundamentals.score
        + sentiment.score
        + insider.score
    )
    max_total = (
        growth.max_score
        + valuation.max_score
        + fundamentals.max_score
        + sentiment.max_score
        + insider.max_score
    )
    evidence = PeterLynchEvidence(
        growth=growth,
        fundamentals=fundamentals,
        valuation=valuation,
        sentiment=sentiment,
        insider_activity=insider,
        peg_ratio=peg,
        weighted_score=weighted,
        market_cap=state.get("market_cap"),
        total_score=total,
        max_score=max_total,
    )
    return {"evidence": evidence}


async def render_verdict_node(
    state: PeterLynchState, config: RunnableConfig
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
        config, "reasoner", schema=PeterLynchSignal
    )
    prompt = render_template(
        "personas_council/personas/peter_lynch.jinja",
        ticker=ticker,
        as_of_date=as_of_date,
        evidence=evidence,
        market_cap=state.get("market_cap"),
        query=query,
    )
    result = cast(
        PeterLynchSignal,
        await llm.ainvoke(
            [SystemMessage(prompt), HumanMessage("Render your Lynch verdict now.")]
        ),
    )
    return {"persona_signals": [result.model_dump()]}


# ── Data-collection sub-agent + subgraph builder ──────────────────────────────


_MCP_TOOLS = [
    "equity_fundamental_metrics",
    "equity_fundamental_income",
    "equity_fundamental_balance",
    "equity_fundamental_cash",
    "equity_estimates_forward_eps",
    "equity_ownership_insider_trading",
    "news_company",
]


async def _build_data_collection_agent(config: RunnableConfig) -> CompiledStateGraph:
    cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = cfg.get_llm_for_role("collector")
    mcp_tools = await get_tools(config, _MCP_TOOLS)
    builder = (
        MuffinAgentBuilder(primary, name="peter_lynch_data_collection")
        .with_fallback_models(*fallbacks)
        .with_state_schema(PeterLynchState)
        .with_input_prompt_template(
            "personas_council/personas/peter_lynch_data_collection.jinja"
        )
        .with_response_format(PeterLynchRawData)
        .with_model_call_limit(run_limit=8, exit_behavior="end")
    )
    for tool in mcp_tools:
        builder = builder.with_tool(tool, run_limit=2)
    builder = builder.with_tool(execute_python, is_cacheable=False)
    return builder.build_react_agent()


async def build_peter_lynch_agent(config: RunnableConfig) -> CompiledStateGraph:
    data_agent = await _build_data_collection_agent(config)
    graph = StateGraph(
        PeterLynchState,
        input_schema=PeterLynchInput,
        output_schema=PeterLynchOutput,
    )
    graph.add_node(
        "collect_data",
        data_agent,
        input_schema=PeterLynchInput,
        retry_policy=_LLM_RETRY,
    )
    graph.add_node("compute_evidence", compute_evidence_node)
    graph.add_node("render_verdict", render_verdict_node, retry_policy=_LLM_RETRY)
    graph.add_edge(START, "collect_data")
    graph.add_edge("collect_data", "compute_evidence")
    graph.add_edge("compute_evidence", "render_verdict")
    graph.add_edge("render_verdict", END)
    return graph.compile()
