"""Sentiment specialist — compiled deterministic subgraph.

Three-node parallel-fetch :class:`StateGraph`:

* ``fetch_insider_trades`` and ``fetch_company_news`` run in parallel from
  ``START`` (both deterministic MCP calls via ``cached_invoke``).
* ``compute_sentiment_signal`` runs after the implicit barrier and
  applies the deterministic 30/70 weighted insider+news aggregation.

**No LLM call** — fully deterministic, mirrors ai-hedge-fund's upstream
``sentiment.py``.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal, cast

from langchain.agents.middleware.types import OmitFromSchema
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_store
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from ....middlewares.tool_result_cache import cached_invoke
from ...data_collection.utils import get_tools
from ..schemas import AnalystSignal, InvestmentSignal
from ..tools.sentiment import combine_sentiment_signals

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
    insider_trades: Annotated[
        list[dict[str, Any]] | None, OmitFromSchema(input=True, output=True)
    ]
    company_news: Annotated[
        list[dict[str, Any]] | None, OmitFromSchema(input=True, output=True)
    ]
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


# ── Response parsing ──────────────────────────────────────────────────────────


def _parse_response(raw: Any) -> list[dict[str, Any]]:
    payload: Any = raw
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list):
            return [r for r in results if isinstance(r, dict)]
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    return []


# ── Graph nodes ───────────────────────────────────────────────────────────────


async def fetch_insider_trades_node(
    state: SentimentAnalysisState, config: RunnableConfig
) -> dict[str, Any]:
    """Deterministic insider-trading fetch via ``cached_invoke``."""
    ticker = state.get("ticker") or ""
    if not ticker:
        return {"insider_trades": []}
    store = get_store()
    tools = await get_tools(config, ["equity_ownership_insider_trading"])
    if not tools:
        return {"insider_trades": []}
    try:
        raw = await cached_invoke(tools[0], {"symbol": ticker, "limit": 100}, store)
    except Exception:
        logger.exception("sentiment fetch_insider_trades failed for %s", ticker)
        return {"insider_trades": []}
    return {"insider_trades": _parse_response(raw)}


async def fetch_company_news_node(
    state: SentimentAnalysisState, config: RunnableConfig
) -> dict[str, Any]:
    """Deterministic news fetch via ``cached_invoke`` (benzinga provider).

    benzinga is the only OpenBB news provider that consistently exposes a
    per-article ``sentiment`` field, which is what
    ``combine_sentiment_signals`` reads in pure Python.
    """
    ticker = state.get("ticker") or ""
    as_of_date = state.get("as_of_date") or datetime.now(UTC).date().isoformat()
    if not ticker:
        return {"company_news": []}
    store = get_store()
    tools = await get_tools(config, ["news_company"])
    if not tools:
        return {"company_news": []}
    end_dt = datetime.strptime(as_of_date[:10], "%Y-%m-%d").date()
    start_dt = end_dt - timedelta(days=_NEWS_LOOKBACK_DAYS)
    args = {
        "provider": "benzinga",
        "symbol": ticker,
        "start_date": start_dt.isoformat(),
        "end_date": end_dt.isoformat(),
        "limit": 50,
    }
    try:
        raw = await cached_invoke(tools[0], args, store)
    except Exception:
        logger.exception("sentiment fetch_company_news failed for %s", ticker)
        return {"company_news": []}
    return {"company_news": _parse_response(raw)}


def compute_sentiment_signal_node(
    state: SentimentAnalysisState,
) -> dict[str, Any]:
    """Pure-Python 30/70 weighted aggregation (no LLM)."""
    insider_trades = state.get("insider_trades") or []
    company_news = state.get("company_news") or []
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
    graph.add_node(
        "fetch_insider_trades", fetch_insider_trades_node, retry_policy=_RETRY
    )
    graph.add_node("fetch_company_news", fetch_company_news_node, retry_policy=_RETRY)
    graph.add_node("compute_sentiment_signal", compute_sentiment_signal_node)
    graph.add_edge(START, "fetch_insider_trades")
    graph.add_edge(START, "fetch_company_news")
    graph.add_edge("fetch_insider_trades", "compute_sentiment_signal")
    graph.add_edge("fetch_company_news", "compute_sentiment_signal")
    graph.add_edge("compute_sentiment_signal", END)
    return graph.compile()


