"""News-sentiment specialist — compiled subgraph (collect+classify → aggregate).

The one LLM specialist (mirrors ai-hedge-fund's ``news_sentiment.py``). The
ReAct ``collect_data`` node fetches recent company news AND classifies each
headline's sentiment (positive / negative / neutral + confidence) into a typed
:class:`NewsSentimentRawData`; the deterministic ``aggregate_sentiment`` node
combines them into the shared ``AnalystSignal``.

Upstream confidence blend preserved: ``0.7 × avg(matching-headline confidence) +
0.3 × signal-proportion``.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

from langchain.agents import AgentState
from langchain.agents.middleware.types import OmitFromSchema
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from ....model_config import ModelConfiguration
from ....utils.agent_builder import MuffinAgentBuilder
from ...data_collection.utils import get_tools
from ..schemas import AnalystSignal, InvestmentSignal

logger = logging.getLogger(__name__)
_RETRY = RetryPolicy(max_attempts=2)

_MCP_TOOLS = ["news_company"]


# ── Evidence + signal ─────────────────────────────────────────────────────────


class NewsSentimentEvidence(BaseModel):
    total_articles: int
    bullish_articles: int
    bearish_articles: int
    neutral_articles: int


class NewsSentimentSignal(AnalystSignal):
    agent_id: Literal["news_sentiment"] = Field(default="news_sentiment")
    evidence: NewsSentimentEvidence


# ── RawData ───────────────────────────────────────────────────────────────────


class ArticleSentiment(BaseModel):
    """One classified news headline."""

    title: str = ""
    sentiment: Literal["positive", "negative", "neutral"] = "neutral"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class NewsSentimentRawData(BaseModel):
    """Recent company news headlines with per-headline sentiment classification."""

    articles: list[ArticleSentiment] = Field(default_factory=list)


# ── State ─────────────────────────────────────────────────────────────────────


class NewsSentimentInput(TypedDict, total=False):
    ticker: str
    as_of_date: str
    query: str | None


class NewsSentimentOutput(TypedDict, total=False):
    persona_signals: list[dict[str, Any]]


class NewsSentimentState(AgentState):
    ticker: Annotated[str, OmitFromSchema(input=False, output=True)]
    as_of_date: Annotated[str, OmitFromSchema(input=False, output=True)]
    query: Annotated[str | None, OmitFromSchema(input=False, output=True)]
    articles: Annotated[
        list[ArticleSentiment] | None, OmitFromSchema(input=True, output=False)
    ]
    persona_signals: Annotated[list[dict], OmitFromSchema(input=True, output=False)]


# ── Mapping helper ────────────────────────────────────────────────────────────


def _to_5tier(
    tactical_signal: str, confidence: float, strong_threshold: float = 0.7
) -> InvestmentSignal:
    if tactical_signal == "bullish":
        return "strong_buy" if confidence >= strong_threshold else "buy"
    if tactical_signal == "bearish":
        return "strong_sell" if confidence >= strong_threshold else "sell"
    return "hold"


# ── Aggregate node ────────────────────────────────────────────────────────────


def aggregate_sentiment_node(state: NewsSentimentState) -> dict[str, Any]:
    """Deterministic aggregation of classified headlines (no LLM)."""
    raw_articles = state.get("articles") or []
    articles = [
        a if isinstance(a, ArticleSentiment) else ArticleSentiment.model_validate(a)
        for a in raw_articles
    ]
    total = len(articles)
    if total == 0:
        sig = NewsSentimentSignal(
            agent_id="news_sentiment",
            signal="hold",
            confidence=0.0,
            reasoning="No company news available",
            evidence=NewsSentimentEvidence(
                total_articles=0,
                bullish_articles=0,
                bearish_articles=0,
                neutral_articles=0,
            ),
        )
        return {"persona_signals": [sig.model_dump()]}

    bullish = sum(1 for a in articles if a.sentiment == "positive")
    bearish = sum(1 for a in articles if a.sentiment == "negative")
    neutral = total - bullish - bearish

    if bullish > bearish:
        overall = "bullish"
        match = "positive"
    elif bearish > bullish:
        overall = "bearish"
        match = "negative"
    else:
        overall = "neutral"
        match = "neutral"

    # Upstream confidence: 0.7 × avg matching-headline confidence + 0.3 × proportion
    matching_conf = [a.confidence for a in articles if a.sentiment == match]
    proportion = max(bullish, bearish) / total
    if matching_conf:
        avg_conf = sum(matching_conf) / len(matching_conf)
        confidence = 0.7 * avg_conf + 0.3 * proportion
    else:
        confidence = proportion

    rating = _to_5tier(overall, confidence)
    sig = NewsSentimentSignal(
        agent_id="news_sentiment",
        signal=rating,
        confidence=min(confidence, 1.0),
        reasoning=(
            f"News sentiment {overall} (conf {confidence:.2f}); "
            f"{bullish} bullish / {bearish} bearish / {neutral} neutral "
            f"of {total} headlines"
        ),
        evidence=NewsSentimentEvidence(
            total_articles=total,
            bullish_articles=bullish,
            bearish_articles=bearish,
            neutral_articles=neutral,
        ),
    )
    return {"persona_signals": [sig.model_dump()]}


# ── Data-collection sub-agent + subgraph builder ──────────────────────────────


async def _build_data_collection_agent(config: RunnableConfig) -> CompiledStateGraph:
    cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = cfg.get_llm_for_role("collector")
    mcp_tools = await get_tools(config, _MCP_TOOLS)
    builder = (
        MuffinAgentBuilder(primary, name="news_sentiment_data_collection")
        .with_fallback_models(*fallbacks)
        .with_state_schema(NewsSentimentState)
        .with_runtime_system_prompt_template(
            "personas_council/specialists/news_sentiment_data_collection.jinja"
        )
        .with_response_format(NewsSentimentRawData)
        .with_model_call_limit(run_limit=6, exit_behavior="end")
    )
    for tool in mcp_tools:
        builder = builder.with_tool(tool, run_limit=2)
    return builder.build_react_agent()


async def build_news_sentiment_analysis_agent(
    config: RunnableConfig,
) -> CompiledStateGraph:
    """Build the news-sentiment specialist subgraph (LLM classify → det. aggregate)."""
    data_agent = await _build_data_collection_agent(config)
    graph = StateGraph(
        NewsSentimentState,
        input_schema=NewsSentimentInput,
        output_schema=NewsSentimentOutput,
    )
    graph.add_node(
        "collect_data",
        data_agent,
        input_schema=NewsSentimentInput,
        retry_policy=_RETRY,
    )
    graph.add_node("aggregate_sentiment", aggregate_sentiment_node)
    graph.add_edge(START, "collect_data")
    graph.add_edge("collect_data", "aggregate_sentiment")
    graph.add_edge("aggregate_sentiment", END)
    return graph.compile()
