"""Sentiment analysis specialist — insider trades + news sentiment.

Wraps :mod:`muffin_agent.tools.sentiment` (insider/news aggregation +
30/70 weighted combine) into a single :class:`AnalystSignal`.  **No LLM
call** — fully deterministic, mirrors ai-hedge-fund's upstream
``sentiment.py``.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from ...tools.sentiment import combine_sentiment_signals
from ..personas._base import PersonaInputState, PersonaOutputState
from ..personas.schemas import AnalystSignal, InvestmentSignal
from ._base import SpecialistSpec, register_specialist

logger = logging.getLogger(__name__)


# ── Evidence + signal ─────────────────────────────────────────────────────────


class SentimentEvidence(BaseModel):
    """Combined insider + news sentiment breakdown."""

    combined_signal: str
    combined_confidence: float
    insider: dict[str, Any]
    news: dict[str, Any]
    weighted_bullish: float
    weighted_bearish: float
    insider_weight: float
    news_weight: float


class SentimentSignal(AnalystSignal):
    """Sentiment specialist structured signal."""

    agent_id: Literal["sentiment"] = Field(default="sentiment")
    evidence: SentimentEvidence


# ── Mapping ───────────────────────────────────────────────────────────────────


def _to_5tier(
    tactical_signal: str, confidence: float, strong_threshold: float = 0.7
) -> InvestmentSignal:
    """Map 3-tier sentiment + confidence to 5-tier ``InvestmentSignal``."""
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


# ── Node ──────────────────────────────────────────────────────────────────────


def _empty_fallback() -> SentimentSignal:
    """Hold signal when no insider or news data is available."""
    return SentimentSignal(
        agent_id="sentiment",
        signal="hold",
        confidence=0.0,
        reasoning="No insider or news data available",
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


async def sentiment_analysis_node(
    state: PersonaInputState, config: RunnableConfig
) -> PersonaOutputState:
    """Compute the sentiment ensemble from insider trades + news.

    Fully deterministic.  Reads ``data_bundle.insider_trades`` and
    ``data_bundle.company_news``, runs the 30/70 weighted aggregation
    from :mod:`tools.sentiment`, and maps the 3-tier verdict to a 5-tier
    ``InvestmentSignal``.
    """
    data_bundle = state.get("data_bundle") or {}
    if not data_bundle or "error" in data_bundle:
        return {"persona_signals": [_empty_fallback().model_dump()]}

    insider_trades = data_bundle.get("insider_trades") or []
    company_news = data_bundle.get("company_news") or []

    if not insider_trades and not company_news:
        return {"persona_signals": [_empty_fallback().model_dump()]}

    combined = combine_sentiment_signals(insider_trades, company_news)
    rating = _to_5tier(combined["signal"], combined["confidence"])
    sig = SentimentSignal(
        agent_id="sentiment",
        signal=rating,
        confidence=min(combined["confidence"], 1.0),
        reasoning=_build_reasoning(combined),
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


# ── Registry entry ────────────────────────────────────────────────────────────


SPECIALIST_SPEC = register_specialist(
    SpecialistSpec(
        slug="sentiment",
        display_name="Sentiment Analysis",
        investing_style=(
            "30/70 weighted insider trades + news sentiment aggregation. "
            "Deterministic, no LLM."
        ),
        node=sentiment_analysis_node,
        signal_schema=SentimentSignal,
    )
)
