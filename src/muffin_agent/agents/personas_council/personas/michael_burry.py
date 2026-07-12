"""Michael Burry persona — compiled subgraph (collect → compute → verdict).

Contrarian deep value: FCF yield + EV/EBIT + balance sheet + insider buying +
negative press. See ``warren_buffett.py`` for the canonical reference.
Reference: ``ai-hedge-fund/src/agents/michael_burry.py``.
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
from ..schemas import AnalystSignal, merge_tool_runs
from ..tools.sentiment import aggregate_insider_trades, aggregate_news_sentiment

logger = logging.getLogger(__name__)
_LLM_RETRY = RetryPolicy(max_attempts=2)


# ── Typed sub-evidence ────────────────────────────────────────────────────────


class MichaelBurryDeepValue(BaseModel):
    fcf_yield: float | None
    ev_to_ebit: float | None
    score: int
    max_score: int
    reasoning: str


class MichaelBurryBalanceSheet(BaseModel):
    debt_to_equity: float | None
    net_cash_position: bool
    score: int
    max_score: int
    reasoning: str


class MichaelBurryInsiderActivity(BaseModel):
    bullish_trades: int
    bearish_trades: int
    total_trades: int
    score: int
    max_score: int
    reasoning: str


class MichaelBurryContrarianSentiment(BaseModel):
    bearish_articles: int
    total_articles: int
    score: int
    max_score: int
    reasoning: str


class MichaelBurryEvidence(BaseModel):
    deep_value: MichaelBurryDeepValue
    balance_sheet: MichaelBurryBalanceSheet
    insider_activity: MichaelBurryInsiderActivity
    contrarian_sentiment: MichaelBurryContrarianSentiment
    fcf_yield: float | None = None
    ev_to_ebit: float | None = None
    total_score: float
    max_score: float


class MichaelBurrySignal(AnalystSignal):
    agent_id: Literal["michael_burry"] = Field(default="michael_burry")
    evidence: MichaelBurryEvidence


# ── RawData ───────────────────────────────────────────────────────────────────


class MichaelBurryRawData(BaseModel):
    """Burry MCP extraction. Series oldest -> newest."""

    free_cash_flow_series: list[float | None] = Field(default_factory=list)
    ebit_series: list[float | None] = Field(default_factory=list)
    total_debt_series: list[float | None] = Field(default_factory=list)
    shareholders_equity_series: list[float | None] = Field(default_factory=list)
    cash_and_equivalents_series: list[float | None] = Field(default_factory=list)
    insider_trades: list[dict[str, Any]] = Field(default_factory=list)
    company_news: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Past 12 months of news articles, each with `sentiment` field "
            "(positive / negative / neutral) — from news_company "
            "(provider=benzinga for sentiment scoring)."
        ),
    )
    market_cap: float | None = None


# ── State ─────────────────────────────────────────────────────────────────────


class MichaelBurryInput(TypedDict, total=False):
    ticker: str
    as_of_date: str
    query: str | None


class MichaelBurryOutput(TypedDict, total=False):
    persona_signals: list[dict[str, Any]]
    tool_runs: list[dict[str, Any]]


class MichaelBurryState(AgentState):
    ticker: Annotated[str, OmitFromSchema(input=False, output=True)]
    as_of_date: Annotated[str, OmitFromSchema(input=False, output=True)]
    query: Annotated[str | None, OmitFromSchema(input=False, output=True)]
    free_cash_flow_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    ebit_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    total_debt_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    shareholders_equity_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    cash_and_equivalents_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    insider_trades: Annotated[
        list[dict[str, Any]] | None, OmitFromSchema(input=True, output=False)
    ]
    company_news: Annotated[
        list[dict[str, Any]] | None, OmitFromSchema(input=True, output=False)
    ]
    market_cap: Annotated[float | None, OmitFromSchema(input=True, output=False)]
    evidence: Annotated[
        MichaelBurryEvidence | None, OmitFromSchema(input=True, output=False)
    ]
    persona_signals: Annotated[list[dict], OmitFromSchema(input=True, output=False)]
    tool_runs: Annotated[list[dict[str, Any]], merge_tool_runs]


# ── Composite scorers ─────────────────────────────────────────────────────────


def _score_burry_deep_value(state: MichaelBurryState) -> MichaelBurryDeepValue:
    """FCF yield + EV/EBIT (max 6)."""
    fcf = state.get("free_cash_flow_series") or []
    ebit = state.get("ebit_series") or []
    total_debt = state.get("total_debt_series") or []
    cash = state.get("cash_and_equivalents_series") or []
    market_cap = state.get("market_cap")

    fcf_latest = fcf[-1] if fcf else None
    ebit_latest = ebit[-1] if ebit else None
    total_debt_latest = total_debt[-1] if total_debt else None
    cash_latest = cash[-1] if cash else None

    score = 0
    parts: list[str] = []
    fcf_yield: float | None = None
    ev_to_ebit: float | None = None

    if fcf_latest is not None and market_cap and market_cap > 0:
        fcf_yield = fcf_latest / market_cap
        if fcf_yield >= 0.15:
            score += 4
            parts.append(f"FCF yield {fcf_yield:.1%} (>=15% deep value)")
        elif fcf_yield >= 0.12:
            score += 3
            parts.append(f"FCF yield {fcf_yield:.1%}")
        elif fcf_yield >= 0.08:
            score += 2

    if ebit_latest and ebit_latest > 0 and market_cap and market_cap > 0:
        ev = market_cap + (total_debt_latest or 0) - (cash_latest or 0)
        ev_to_ebit = ev / ebit_latest
        if ev_to_ebit < 6:
            score += 2
            parts.append(f"EV/EBIT {ev_to_ebit:.1f}x (very cheap)")
        elif ev_to_ebit < 10:
            score += 1
            parts.append(f"EV/EBIT {ev_to_ebit:.1f}x")

    return MichaelBurryDeepValue(
        fcf_yield=fcf_yield,
        ev_to_ebit=ev_to_ebit,
        score=min(score, 6),
        max_score=6,
        reasoning="; ".join(parts) or "Cannot compute deep-value metrics",
    )


def _score_burry_balance_sheet(state: MichaelBurryState) -> MichaelBurryBalanceSheet:
    """D/E + net cash (max 3)."""
    total_debt = [v for v in (state.get("total_debt_series") or []) if v is not None]
    equity = [
        v for v in (state.get("shareholders_equity_series") or []) if v is not None
    ]
    cash = [
        v for v in (state.get("cash_and_equivalents_series") or []) if v is not None
    ]
    score = 0
    parts: list[str] = []
    de: float | None = None
    net_cash = False
    if total_debt and equity and equity[-1] and equity[-1] > 0:
        de = total_debt[-1] / equity[-1]
        if de < 0.5:
            score += 2
            parts.append(f"D/E {de:.2f} (low)")
        elif de < 1.0:
            score += 1
    if cash and total_debt:
        if cash[-1] > total_debt[-1]:
            score += 1
            net_cash = True
            parts.append("Net cash position")
    return MichaelBurryBalanceSheet(
        debt_to_equity=de,
        net_cash_position=net_cash,
        score=min(score, 3),
        max_score=3,
        reasoning="; ".join(parts),
    )


def _score_burry_insider_activity(
    state: MichaelBurryState,
) -> MichaelBurryInsiderActivity:
    """Net insider buying (max 2)."""
    insider_trades = state.get("insider_trades") or []
    agg = aggregate_insider_trades(insider_trades)
    score = 0
    parts: list[str] = []
    if agg["signal"] == "bullish":
        if agg["bullish_trades"] > agg["bearish_trades"] * 2:
            score = 2
            parts.append(
                f"Heavy net buying ({agg['bullish_trades']}/{agg['total_trades']})"
            )
        else:
            score = 1
            parts.append(
                f"Net insider buying ({agg['bullish_trades']}/{agg['total_trades']})"
            )
    elif agg["signal"] == "bearish":
        parts.append(
            f"Net insider selling ({agg['bearish_trades']}/{agg['total_trades']})"
        )
    else:
        parts.append("No insider activity signal")
    return MichaelBurryInsiderActivity(
        bullish_trades=int(agg.get("bullish_trades", 0)),
        bearish_trades=int(agg.get("bearish_trades", 0)),
        total_trades=int(agg.get("total_trades", 0)),
        score=score,
        max_score=2,
        reasoning="; ".join(parts),
    )


def _score_burry_contrarian_sentiment(
    state: MichaelBurryState,
) -> MichaelBurryContrarianSentiment:
    """5+ negative articles = +1. Burry: a wall of hate is a friend."""
    articles = state.get("company_news") or []
    agg = aggregate_news_sentiment(articles)
    bearish = int(agg.get("bearish_articles", 0))
    total = int(agg.get("total_articles", 0))
    score = 0
    parts: list[str] = []
    if bearish >= 5:
        score = 1
        parts.append(f"{bearish} bearish headlines — contrarian setup")
    else:
        parts.append(f"Insufficient bearish coverage ({bearish} articles)")
    return MichaelBurryContrarianSentiment(
        bearish_articles=bearish,
        total_articles=total,
        score=score,
        max_score=1,
        reasoning="; ".join(parts),
    )


# ── Graph nodes ───────────────────────────────────────────────────────────────


def compute_evidence_node(state: MichaelBurryState) -> dict[str, Any]:
    deep_value = _score_burry_deep_value(state)
    balance = _score_burry_balance_sheet(state)
    insider = _score_burry_insider_activity(state)
    contrarian = _score_burry_contrarian_sentiment(state)
    total = deep_value.score + balance.score + insider.score + contrarian.score
    max_total = (
        deep_value.max_score
        + balance.max_score
        + insider.max_score
        + contrarian.max_score
    )
    evidence = MichaelBurryEvidence(
        deep_value=deep_value,
        balance_sheet=balance,
        insider_activity=insider,
        contrarian_sentiment=contrarian,
        fcf_yield=deep_value.fcf_yield,
        ev_to_ebit=deep_value.ev_to_ebit,
        total_score=total,
        max_score=max_total,
    )
    return {"evidence": evidence}


async def render_verdict_node(
    state: MichaelBurryState, config: RunnableConfig
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
        config, "reasoner", schema=MichaelBurrySignal
    )
    prompt = render_template(
        "personas_council/personas/michael_burry.jinja",
        ticker=ticker,
        as_of_date=as_of_date,
        evidence=evidence,
        market_cap=state.get("market_cap"),
        query=query,
    )
    result = cast(
        MichaelBurrySignal,
        await llm.ainvoke(
            [SystemMessage(prompt), HumanMessage("Render your Burry verdict now.")]
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
    "news_company",
]


async def _build_data_collection_agent(config: RunnableConfig) -> CompiledStateGraph:
    cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = cfg.get_llm_for_role("collector")
    mcp_tools = await get_tools(config, _MCP_TOOLS)
    builder = (
        MuffinAgentBuilder(primary, name="michael_burry_data_collection")
        .with_fallback_models(*fallbacks)
        .with_state_schema(MichaelBurryState)
        .with_input_prompt_template(
            "personas_council/personas/michael_burry_data_collection.jinja"
        )
        .with_response_format(MichaelBurryRawData)
        .with_model_call_limit(run_limit=8, exit_behavior="end")
    )
    for tool in mcp_tools:
        builder = builder.with_tool(tool, run_limit=2)
    builder = builder.with_tool(execute_python, is_cacheable=False)
    return builder.build_react_agent()


async def build_michael_burry_agent(config: RunnableConfig) -> CompiledStateGraph:
    data_agent = await _build_data_collection_agent(config)
    graph = StateGraph(
        MichaelBurryState,
        input_schema=MichaelBurryInput,
        output_schema=MichaelBurryOutput,
    )
    graph.add_node(
        "collect_data",
        data_agent,
        input_schema=MichaelBurryInput,
        retry_policy=_LLM_RETRY,
    )
    graph.add_node("compute_evidence", compute_evidence_node)
    graph.add_node("render_verdict", render_verdict_node, retry_policy=_LLM_RETRY)
    graph.add_edge(START, "collect_data")
    graph.add_edge("collect_data", "compute_evidence")
    graph.add_edge("compute_evidence", "render_verdict")
    graph.add_edge("render_verdict", END)
    return graph.compile()
