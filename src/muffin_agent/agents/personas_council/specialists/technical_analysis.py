"""Technical analysis specialist — compiled deterministic subgraph.

Three-node :class:`StateGraph`:

1. ``plan_fetch`` — pure Python; computes the OHLCV window and emits a synthetic
   ``AIMessage`` carrying one tool call. No LLM.
2. ``fetch_ohlcv`` — a bare :class:`~langgraph.prebuilt.ToolNode` that executes it,
   appending a real ``ToolMessage``. The tool goes through ``cached_invoke`` under the
   MCP tool's own name, so the cache is still shared with personas and
   middleware-driven tool calls.
3. ``compute_technical_signal`` — pure Python 5-strategy ensemble via the
   package-local ``tools.technicals``.

**Still no LLM call anywhere** — execution is exactly as deterministic as the previous
``cached_invoke``-from-a-node version. The difference is that the fetch now produces a
genuine AIMessage/ToolMessage pair in ``values.messages``, so it is visible to anything
reading this namespace's messages (UI execution tree, LangSmith, evals) with no
special-casing. See ``_fetch_tools`` for the rationale.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal

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
from ..tools.technicals import (
    StrategyResult,
    combine_technical_signals,
    compute_mean_reversion_signal,
    compute_momentum_signal,
    compute_stat_arb_signal,
    compute_trend_signal,
    compute_volatility_regime_signal,
)
from ._fetch_tools import (
    deterministic_tool_call,
    fetch_equity_ohlcv,
    parse_result_rows,
    plan_message,
    tool_payload,
)

logger = logging.getLogger(__name__)
_RETRY = RetryPolicy(max_attempts=2)

_PRICE_LOOKBACK_DAYS = 365


# ── Evidence + signal ─────────────────────────────────────────────────────────


class TechnicalEvidence(BaseModel):
    trend: dict[str, Any]
    mean_reversion: dict[str, Any]
    momentum: dict[str, Any]
    volatility_regime: dict[str, Any]
    stat_arb: dict[str, Any]
    weighted: dict[str, Any]


class TechnicalSignal(AnalystSignal):
    agent_id: Literal["technicals"] = Field(default="technicals")
    evidence: TechnicalEvidence


# ── State ─────────────────────────────────────────────────────────────────────


class TechnicalAnalysisInput(TypedDict, total=False):
    ticker: str
    as_of_date: str
    query: str | None


class TechnicalAnalysisOutput(TypedDict, total=False):
    persona_signals: list[dict[str, Any]]


class TechnicalAnalysisState(TypedDict, total=False):
    ticker: Annotated[str, OmitFromSchema(input=False, output=True)]
    as_of_date: Annotated[str, OmitFromSchema(input=False, output=True)]
    query: Annotated[str | None, OmitFromSchema(input=False, output=True)]
    # The fetch's AIMessage/ToolMessage pair. Internal to this subgraph — the
    # explicit output_schema keeps it out of CouncilState.
    messages: Annotated[list[AnyMessage], add_messages]
    persona_signals: Annotated[list[dict], OmitFromSchema(input=True, output=False)]


# ── Mapping helpers ───────────────────────────────────────────────────────────


def _to_5tier(
    tactical_signal: str, confidence: float, strong_threshold: float = 0.7
) -> InvestmentSignal:
    """Convert a 3-tier bullish/bearish/neutral + confidence to 5-tier rating."""
    if tactical_signal == "bullish":
        return "strong_buy" if confidence >= strong_threshold else "buy"
    if tactical_signal == "bearish":
        return "strong_sell" if confidence >= strong_threshold else "sell"
    return "hold"


def _build_reasoning(
    weighted: StrategyResult, per_strategy: dict[str, StrategyResult]
) -> str:
    parts: list[str] = []
    parts.append(f"Ensemble {weighted['signal']} (conf {weighted['confidence']:.2f})")
    bullish_strats = [n for n, r in per_strategy.items() if r["signal"] == "bullish"]
    bearish_strats = [n for n, r in per_strategy.items() if r["signal"] == "bearish"]
    if bullish_strats:
        parts.append(f"bullish: {', '.join(bullish_strats)}")
    if bearish_strats:
        parts.append(f"bearish: {', '.join(bearish_strats)}")
    return "; ".join(parts)


def _empty_fallback(reason: str) -> TechnicalSignal:
    empty: dict[str, Any] = {"signal": "neutral", "confidence": 0.0, "metrics": {}}
    return TechnicalSignal(
        agent_id="technicals",
        signal="hold",
        confidence=0.0,
        reasoning=reason,
        evidence=TechnicalEvidence(
            trend=empty,
            mean_reversion=empty,
            momentum=empty,
            volatility_regime=empty,
            stat_arb=empty,
            weighted=empty,
        ),
    )


# ── Graph nodes ───────────────────────────────────────────────────────────────


def plan_fetch_node(state: TechnicalAnalysisState) -> dict[str, Any]:
    """Compute the OHLCV window and emit the tool call for ``ToolNode`` (no LLM)."""
    ticker = state.get("ticker") or ""
    as_of_date = state.get("as_of_date") or datetime.now(UTC).date().isoformat()
    end_dt = datetime.strptime(as_of_date[:10], "%Y-%m-%d").date()
    start_dt = end_dt - timedelta(days=_PRICE_LOOKBACK_DAYS)
    call = deterministic_tool_call(
        fetch_equity_ohlcv.name,
        {
            "symbol": ticker,
            "start_date": start_dt.isoformat(),
            "end_date": end_dt.isoformat(),
            "interval": "1d",
            "provider": "yfinance",
        },
    )
    return {
        "messages": [
            plan_message(
                [call], note=f"Fetching {_PRICE_LOOKBACK_DAYS}d OHLCV for {ticker}"
            )
        ]
    }


def route_after_plan(state: TechnicalAnalysisState) -> str:
    """Skip the fetch entirely when there is no ticker to fetch for."""
    return "fetch_ohlcv" if (state.get("ticker") or "") else "compute_technical_signal"


def compute_technical_signal_node(
    state: TechnicalAnalysisState,
) -> dict[str, Any]:
    """Pure-Python 5-strategy ensemble (no LLM)."""
    prices_1y = parse_result_rows(
        tool_payload(state.get("messages"), fetch_equity_ohlcv.name)
    )
    if len(prices_1y) < 20:
        sig = _empty_fallback("Insufficient OHLCV history (need 20+ daily bars)")
        return {"persona_signals": [sig.model_dump()]}

    trend = compute_trend_signal(prices_1y)
    mr = compute_mean_reversion_signal(prices_1y)
    momentum = compute_momentum_signal(prices_1y)
    vol = compute_volatility_regime_signal(prices_1y)
    stat = compute_stat_arb_signal(prices_1y)
    per_strategy: dict[str, StrategyResult] = {
        "trend": trend,
        "mean_reversion": mr,
        "momentum": momentum,
        "volatility": vol,
        "stat_arb": stat,
    }
    weighted = combine_technical_signals(per_strategy)
    rating = _to_5tier(weighted["signal"], weighted["confidence"])
    sig = TechnicalSignal(
        agent_id="technicals",
        signal=rating,
        confidence=min(weighted["confidence"], 1.0),
        reasoning=_build_reasoning(weighted, per_strategy),
        evidence=TechnicalEvidence(
            trend=dict(trend),
            mean_reversion=dict(mr),
            momentum=dict(momentum),
            volatility_regime=dict(vol),
            stat_arb=dict(stat),
            weighted=dict(weighted),
        ),
    )
    return {"persona_signals": [sig.model_dump()]}


# ── Subgraph builder ──────────────────────────────────────────────────────────


def build_technical_analysis_agent() -> CompiledStateGraph:
    """Build the deterministic technical-analysis subgraph (no LLM)."""
    graph = StateGraph(
        TechnicalAnalysisState,
        input_schema=TechnicalAnalysisInput,
        output_schema=TechnicalAnalysisOutput,
    )
    graph.add_node("plan_fetch", plan_fetch_node)
    graph.add_node("fetch_ohlcv", ToolNode([fetch_equity_ohlcv]), retry_policy=_RETRY)
    graph.add_node("compute_technical_signal", compute_technical_signal_node)
    graph.add_edge(START, "plan_fetch")
    graph.add_conditional_edges(
        "plan_fetch",
        route_after_plan,
        ["fetch_ohlcv", "compute_technical_signal"],
    )
    graph.add_edge("fetch_ohlcv", "compute_technical_signal")
    graph.add_edge("compute_technical_signal", END)
    return graph.compile()
