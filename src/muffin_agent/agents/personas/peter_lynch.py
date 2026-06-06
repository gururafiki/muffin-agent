"""Peter Lynch persona — GARP + PEG + ten-bagger hunting.

Five sub-scores weighted 30/25/20/15/10 (growth / valuation / fundamentals /
sentiment / insider).  Mirrors ai-hedge-fund's ``peter_lynch.py``.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from ...model_config import ModelConfiguration
from ...prompts import render_template
from ...tools.scoring_helpers import (
    compute_peg_ratio,
    score_insider_buy_ratio,
)
from ...tools.sentiment import aggregate_news_sentiment
from ._base import (
    PersonaInputState,
    PersonaOutputState,
    PersonaSpec,
    register_persona,
)
from .schemas import AnalystSignal, ScoreDetail

logger = logging.getLogger(__name__)


class PeterLynchEvidence(BaseModel):
    """Persona-specific evidence — sub-scores, computed metrics."""

    growth: ScoreDetail
    fundamentals: ScoreDetail
    valuation: ScoreDetail
    sentiment: ScoreDetail
    insider_activity: ScoreDetail
    peg_ratio: float | None
    weighted_score: float
    market_cap: float | None
    total_score: float
    max_score: float


class PeterLynchSignal(AnalystSignal):
    """Persona structured signal with narrowed evidence type."""

    agent_id: Literal["peter_lynch"] = Field(default="peter_lynch")
    evidence: PeterLynchEvidence


def _score_growth(line_items: dict[str, list[float | None]]) -> ScoreDetail:
    revenues = [v for v in line_items.get("revenue", []) if v is not None]
    eps = [v for v in line_items.get("earnings_per_share", []) if v is not None]
    score = 0
    parts: list[str] = []

    def cagr(series: list[float]) -> float | None:
        if len(series) < 2 or series[0] <= 0:
            return None
        years = len(series) - 1
        return (series[-1] / series[0]) ** (1 / years) - 1

    rev_cagr = cagr(revenues)
    eps_cagr = cagr(eps)
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
    return ScoreDetail(
        score=normalised,
        max_score=10,
        details="; ".join(parts) or "Limited growth data",
    )


def _score_fundamentals(
    latest_metrics: dict[str, Any], line_items: dict[str, list[float | None]]
) -> ScoreDetail:
    de = latest_metrics.get("debt_to_equity")
    om = latest_metrics.get("operating_margin")
    fcf = line_items.get("free_cash_flow", [])
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
    if fcf and fcf[-1] is not None and fcf[-1] > 0:
        score += 2
        parts.append(f"FCF positive {fcf[-1]:,.0f}")
    normalised = (score / 6) * 10
    return ScoreDetail(
        score=normalised,
        max_score=10,
        details="; ".join(parts) or "Limited fundamentals",
    )


def _score_valuation(
    latest_metrics: dict[str, Any], line_items: dict[str, list[float | None]]
) -> tuple[ScoreDetail, float | None]:
    pe = latest_metrics.get("price_to_earnings_ratio")
    eps = line_items.get("earnings_per_share", [])
    growth = None
    if len(eps) >= 2 and eps[0] and eps[0] > 0:
        years = len(eps) - 1
        if eps[-1] > 0:
            growth = (eps[-1] / eps[0]) ** (1 / years) - 1
    peg = compute_peg_ratio(pe, growth)
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
    return ScoreDetail(
        score=normalised, max_score=10, details="; ".join(parts) or "Cannot compute PEG"
    ), peg


def _score_sentiment(articles: list[dict[str, Any]]) -> ScoreDetail:
    agg = aggregate_news_sentiment(articles)
    bullish = agg["bullish_articles"]
    bearish = agg["bearish_articles"]
    total = agg["total_articles"]
    if total == 0:
        return ScoreDetail(
            score=5, max_score=10, details="No news data — neutral default"
        )
    if bearish / max(total, 1) > 0.30:
        return ScoreDetail(
            score=3, max_score=10, details=f"Bearish-leaning news ({bearish}/{total})"
        )
    if bullish > bearish:
        return ScoreDetail(
            score=8, max_score=10, details=f"Bullish news ({bullish}/{total})"
        )
    return ScoreDetail(
        score=6, max_score=10, details=f"Mixed news ({bullish}/{bearish}/{total})"
    )


def _score_insider(insider_trades: list[dict[str, Any]]) -> ScoreDetail:
    inner = score_insider_buy_ratio(insider_trades)
    # Convert 0–8 scale to 0–10
    return ScoreDetail(
        score=(inner.score / 8) * 10,
        max_score=10,
        details=inner.details,
    )


def _compute_lynch_facts(data_bundle: dict[str, Any]) -> PeterLynchEvidence:
    line_items = data_bundle.get("line_items", {})
    metrics = data_bundle.get("financial_metrics", [])
    latest_metrics = metrics[0] if metrics else {}
    insider_trades = data_bundle.get("insider_trades", [])
    news = data_bundle.get("company_news", [])
    market_cap = data_bundle.get("market_cap")

    growth = _score_growth(line_items)
    fundamentals = _score_fundamentals(latest_metrics, line_items)
    valuation, peg = _score_valuation(latest_metrics, line_items)
    sentiment = _score_sentiment(news)
    insider = _score_insider(insider_trades)
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
    return PeterLynchEvidence(
        growth=growth,
        fundamentals=fundamentals,
        valuation=valuation,
        sentiment=sentiment,
        insider_activity=insider,
        peg_ratio=peg,
        weighted_score=weighted,
        market_cap=market_cap,
        total_score=total,
        max_score=max_total,
    )


async def peter_lynch_node(
    state: PersonaInputState, config: RunnableConfig
) -> PersonaOutputState:
    """Render the Peter Lynch verdict from state["data_bundle"]."""
    ticker = state.get("ticker", "")
    query = state.get("query")
    data_bundle = state.get("data_bundle") or {}

    if not data_bundle or "error" in data_bundle:
        fallback = PeterLynchSignal(
            agent_id="peter_lynch",
            signal="hold",
            confidence=0.0,
            reasoning="Insufficient data",
            evidence=PeterLynchEvidence(
                growth=ScoreDetail(score=0, max_score=10, details="no data"),
                fundamentals=ScoreDetail(score=0, max_score=10, details="no data"),
                valuation=ScoreDetail(score=0, max_score=10, details="no data"),
                sentiment=ScoreDetail(score=0, max_score=10, details="no data"),
                insider_activity=ScoreDetail(score=0, max_score=10, details="no data"),
                peg_ratio=None,
                weighted_score=0,
                market_cap=None,
                total_score=0,
                max_score=50,
            ),
        )
        return {"persona_signals": [fallback.model_dump()]}

    evidence = _compute_lynch_facts(data_bundle)
    llm = ModelConfiguration.get_chat_model_for_role(
        config, "reasoner", schema=PeterLynchSignal
    )
    prompt = render_template(
        "personas/peter_lynch.jinja",
        ticker=ticker,
        as_of_date=data_bundle.get("as_of_date", ""),
        facts=evidence.model_dump(mode="json"),
        query=query,
        persona_display_name="Peter Lynch",
        persona_slug="peter_lynch",
        signal_schema_name="PeterLynchSignal",
    )
    result = cast(
        PeterLynchSignal,
        await llm.ainvoke(
            [SystemMessage(prompt), HumanMessage("Render your Lynch verdict now.")]
        ),
    )
    return {"persona_signals": [result.model_dump()]}


PERSONA_SPEC = register_persona(
    PersonaSpec(
        slug="peter_lynch",
        display_name="Peter Lynch",
        investing_style=(
            "GARP — growth at a reasonable price; PEG ratio focus; invest in "
            "what you know; ten-bagger hunt"
        ),
        node=peter_lynch_node,
        signal_schema=PeterLynchSignal,
    )
)
