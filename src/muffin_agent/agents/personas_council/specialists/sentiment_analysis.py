"""Sentiment specialist — compiled deterministic subgraph.

Three-node :class:`StateGraph`:

* ``plan_fetch`` — pure Python; emits ONE synthetic ``AIMessage`` carrying BOTH tool
  calls (insider trades + company news). No LLM.
* ``fetch`` — a bare :class:`~langgraph.prebuilt.ToolNode`. Several tool calls in one
  message are executed **concurrently**, so this keeps the previous two-node parallel
  fan-out without the extra nodes.
* ``compute_sentiment_signal`` — the deterministic 30/70 weighted insider+news
  aggregation, reading each payload back off its ``ToolMessage``.

**Still no LLM call** — fully deterministic, mirrors ai-hedge-fund's upstream
``sentiment.py``. The fetches now produce genuine AIMessage/ToolMessage pairs so they
are visible wherever this namespace's messages are read; see ``_fetch_tools``.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal, cast

from langchain.agents.middleware.types import OmitFromSchema
from langchain_core.messages import AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import RetryPolicy
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from ..schemas import AnalystSignal, InvestmentSignal
from ..tools.sentiment import combine_sentiment_signals
from ._fetch_tools import (
    deterministic_tool_call,
    fetch_company_news,
    fetch_insider_trades,
    parse_result_rows,
    plan_message,
    tool_payload,
)

logger = logging.getLogger(__name__)
_RETRY = RetryPolicy(max_attempts=2)
_NEWS_LOOKBACK_DAYS = 365


# ── Evidence + signal ─────────────────────────────────────────────────────────


class SentimentEvidence(BaseModel):
    combined_signal: str
    combined_confidence: float
    insider: dict[str, Any]
    news: dict[str, Any]
    weighted_bullish: float
    weighted_bearish: float
    insider_weight: float
    news_weight: float


class SentimentSignal(AnalystSignal):
    agent_id: Literal["sentiment"] = Field(default="sentiment")
    evidence: SentimentEvidence


# ── State ─────────────────────────────────────────────────────────────────────


class SentimentAnalysisInput(TypedDict, total=False):
    ticker: str
    as_of_date: str
    query: str | None


class SentimentAnalysisOutput(TypedDict, total=False):
    persona_signals: list[dict[str, Any]]


class SentimentAnalysisState(TypedDict, total=False):
    ticker: Annotated[str, OmitFromSchema(input=False, output=True)]
    as_of_date: Annotated[str, OmitFromSchema(input=False, output=True)]
    query: Annotated[str | None, OmitFromSchema(input=False, output=True)]
    # The fetches' AIMessage + ToolMessage pairs. Internal to this subgraph — the
    # explicit output_schema keeps them out of CouncilState.
    messages: Annotated[list[AnyMessage], add_messages]
    persona_signals: Annotated[list[dict], OmitFromSchema(input=True, output=False)]


# ── Mapping helpers ───────────────────────────────────────────────────────────


def _to_5tier(
    tactical_signal: str, confidence: float, strong_threshold: float = 0.7
) -> InvestmentSignal:
    """Convert 3-tier bullish/bearish/neutral + confidence to 5-tier rating."""
    if tactical_signal == "bullish":
        return "strong_buy" if confidence >= strong_threshold else "buy"
    if tactical_signal == "bearish":
        return "strong_sell" if confidence >= strong_threshold else "sell"
    return "hold"


def _build_reasoning(combined: dict[str, Any]) -> str:
    insider = combined["insider"]
    news = combined["news"]
    return (
        f"Combined {combined['signal']} (conf {combined['confidence']:.2f}); "
        f"insider {insider['signal']} ({insider['bullish_trades']}/"
        f"{insider['total_trades']} buys); "
        f"news {news['signal']} ({news['bullish_articles']} bull / "
        f"{news['bearish_articles']} bear / {news['neutral_articles']} neutral)"
    )


def _empty_fallback(reason: str) -> SentimentSignal:
    return SentimentSignal(
        agent_id="sentiment",
        signal="hold",
        confidence=0.0,
        reasoning=reason,
        evidence=SentimentEvidence(
            combined_signal="neutral",
            combined_confidence=0.0,
            insider={
                "signal": "neutral",
                "confidence": 0.0,
                "total_trades": 0,
                "bullish_trades": 0,
                "bearish_trades": 0,
                "net_share_change": 0.0,
            },
            news={
                "signal": "neutral",
                "confidence": 0.0,
                "total_articles": 0,
                "bullish_articles": 0,
                "bearish_articles": 0,
                "neutral_articles": 0,
            },
            weighted_bullish=0.0,
            weighted_bearish=0.0,
            insider_weight=0.3,
            news_weight=0.7,
        ),
    )


# ── Graph nodes ───────────────────────────────────────────────────────────────


def plan_fetch_node(state: SentimentAnalysisState) -> dict[str, Any]:
    """Emit BOTH fetches as tool calls in one message (executed concurrently)."""
    ticker = state.get("ticker") or ""
    as_of_date = state.get("as_of_date") or datetime.now(UTC).date().isoformat()
    end_dt = datetime.strptime(as_of_date[:10], "%Y-%m-%d").date()
    start_dt = end_dt - timedelta(days=_NEWS_LOOKBACK_DAYS)
    calls = [
        deterministic_tool_call(
            fetch_insider_trades.name, {"symbol": ticker, "limit": 100}
        ),
        deterministic_tool_call(
            fetch_company_news.name,
            {
                "symbol": ticker,
                "start_date": start_dt.isoformat(),
                "end_date": end_dt.isoformat(),
                "limit": 50,
                "provider": "benzinga",
            },
        ),
    ]
    return {
        "messages": [
            plan_message(calls, note=f"Fetching insider trades + news for {ticker}")
        ]
    }


def route_after_plan(state: SentimentAnalysisState) -> str:
    """Skip the fetches entirely when there is no ticker to fetch for."""
    return "fetch" if (state.get("ticker") or "") else "compute_sentiment_signal"


def compute_sentiment_signal_node(
    state: SentimentAnalysisState,
) -> dict[str, Any]:
    """Pure-Python 30/70 weighted aggregation (no LLM)."""
    messages = state.get("messages")
    insider_trades = parse_result_rows(
        tool_payload(messages, fetch_insider_trades.name)
    )
    company_news = parse_result_rows(tool_payload(messages, fetch_company_news.name))
    if not insider_trades and not company_news:
        sig = _empty_fallback("No insider or news data available")
        return {"persona_signals": [sig.model_dump()]}

    combined = combine_sentiment_signals(insider_trades, company_news)
    rating = _to_5tier(combined["signal"], combined["confidence"])
    sig = SentimentSignal(
        agent_id="sentiment",
        signal=rating,
        confidence=min(combined["confidence"], 1.0),
        reasoning=_build_reasoning(cast(dict[str, Any], combined)),
        evidence=SentimentEvidence(
            combined_signal=combined["signal"],
            combined_confidence=combined["confidence"],
            insider=dict(combined["insider"]),
            news=dict(combined["news"]),
            weighted_bullish=combined["weighted_bullish"],
            weighted_bearish=combined["weighted_bearish"],
            insider_weight=combined["insider_weight"],
            news_weight=combined["news_weight"],
        ),
    )
    return {"persona_signals": [sig.model_dump()]}


# ── Subgraph builder ──────────────────────────────────────────────────────────


def build_sentiment_analysis_agent() -> CompiledStateGraph:
    """Build the deterministic sentiment-analysis subgraph (no LLM)."""
    graph = StateGraph(
        SentimentAnalysisState,
        input_schema=SentimentAnalysisInput,
        output_schema=SentimentAnalysisOutput,
    )
    graph.add_node("plan_fetch", plan_fetch_node)
    graph.add_node(
        "fetch",
        ToolNode([fetch_insider_trades, fetch_company_news]),
        retry_policy=_RETRY,
    )
    graph.add_node("compute_sentiment_signal", compute_sentiment_signal_node)
    graph.add_edge(START, "plan_fetch")
    graph.add_conditional_edges(
        "plan_fetch",
        route_after_plan,
        ["fetch", "compute_sentiment_signal"],
    )
    graph.add_edge("fetch", "compute_sentiment_signal")
    graph.add_edge("compute_sentiment_signal", END)
    return graph.compile()
