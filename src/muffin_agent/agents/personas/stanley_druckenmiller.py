"""Stanley Druckenmiller persona — macro + momentum + asymmetric R/R.

Five weighted dimensions: growth+momentum 35%, risk_reward 20%,
valuation 20%, sentiment 15%, insider 10%.  Reads price data for momentum,
so requires ``prices_1y`` in the data bundle.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any, Literal, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from ...model_config import ModelConfiguration
from ...prompts import render_template
from ...tools.scoring_helpers import compute_price_momentum, score_insider_buy_ratio
from ...tools.sentiment import aggregate_news_sentiment
from ._base import (
    PersonaInputState,
    PersonaOutputState,
    PersonaSpec,
    register_persona,
)
from .schemas import AnalystSignal, ScoreDetail

logger = logging.getLogger(__name__)


class StanleyDruckenmillerEvidence(BaseModel):
    """Persona-specific evidence — sub-scores, computed metrics."""

    growth_momentum: ScoreDetail
    risk_reward: ScoreDetail
    valuation: ScoreDetail
    sentiment: ScoreDetail
    insider_activity: ScoreDetail
    momentum_pct: float | None
    weighted_score: float
    market_cap: float | None
    total_score: float
    max_score: float


class StanleyDruckenmillerSignal(AnalystSignal):
    """Persona structured signal with narrowed evidence type."""

    agent_id: Literal["stanley_druckenmiller"] = Field(default="stanley_druckenmiller")
    evidence: StanleyDruckenmillerEvidence


def _cagr(series: list[float]) -> float | None:
    if len(series) < 2 or series[0] <= 0:
        return None
    return (series[-1] / series[0]) ** (1 / (len(series) - 1)) - 1


def _score_growth_momentum(
    line_items: dict[str, list[float | None]], prices: list[dict[str, Any]]
) -> tuple[ScoreDetail, float | None]:
    revenues = [v for v in line_items.get("revenue", []) if v is not None]
    eps = [v for v in line_items.get("earnings_per_share", []) if v is not None]
    score = 0
    parts: list[str] = []
    rev_cagr = _cagr(revenues)
    eps_cagr = _cagr(eps)
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
    momentum_pct = None
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
    return ScoreDetail(
        score=normalised, max_score=10, details="; ".join(parts) or "Limited"
    ), momentum_pct


def _score_risk_reward(
    line_items: dict[str, list[float | None]], prices: list[dict[str, Any]]
) -> ScoreDetail:
    debt = [v for v in line_items.get("total_debt", []) if v is not None]
    equity = [v for v in line_items.get("shareholders_equity", []) if v is not None]
    closes = [b.get("close") for b in prices if b.get("close") is not None]
    score = 0
    parts: list[str] = []
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
    return ScoreDetail(
        score=normalised, max_score=10, details="; ".join(parts) or "Limited"
    )


def _score_valuation(latest_metrics: dict[str, Any]) -> ScoreDetail:
    pe = latest_metrics.get("price_to_earnings_ratio")
    ev_to_ebit = latest_metrics.get("ev_to_ebit")
    fcf_yield = latest_metrics.get("free_cash_flow_yield")
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
    return ScoreDetail(
        score=normalised, max_score=10, details="; ".join(parts) or "Limited"
    )


def _score_sentiment(articles: list[dict[str, Any]]) -> ScoreDetail:
    agg = aggregate_news_sentiment(articles)
    bullish = agg["bullish_articles"]
    bearish = agg["bearish_articles"]
    total = agg["total_articles"]
    if total == 0:
        return ScoreDetail(score=5, max_score=10, details="No news")
    if bearish / max(total, 1) > 0.30:
        return ScoreDetail(
            score=3, max_score=10, details=f"Bearish news ({bearish}/{total})"
        )
    if bullish > bearish:
        return ScoreDetail(
            score=8, max_score=10, details=f"Bullish news ({bullish}/{total})"
        )
    return ScoreDetail(score=6, max_score=10, details="Mixed news")


def _score_insider(insider_trades: list[dict[str, Any]]) -> ScoreDetail:
    inner = score_insider_buy_ratio(insider_trades)
    return ScoreDetail(
        score=(inner.score / 8) * 10, max_score=10, details=inner.details
    )


def _compute_druckenmiller_facts(
    data_bundle: dict[str, Any],
) -> StanleyDruckenmillerEvidence:
    line_items = data_bundle.get("line_items", {})
    metrics = data_bundle.get("financial_metrics", [])
    latest_metrics = metrics[0] if metrics else {}
    insider_trades = data_bundle.get("insider_trades", [])
    news = data_bundle.get("company_news", [])
    prices = data_bundle.get("prices_1y", [])
    market_cap = data_bundle.get("market_cap")

    growth, momentum = _score_growth_momentum(line_items, prices)
    risk = _score_risk_reward(line_items, prices)
    valuation = _score_valuation(latest_metrics)
    sentiment = _score_sentiment(news)
    insider = _score_insider(insider_trades)
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
    return StanleyDruckenmillerEvidence(
        growth_momentum=growth,
        risk_reward=risk,
        valuation=valuation,
        sentiment=sentiment,
        insider_activity=insider,
        momentum_pct=momentum,
        weighted_score=weighted,
        market_cap=market_cap,
        total_score=total,
        max_score=50,
    )


async def stanley_druckenmiller_node(
    state: PersonaInputState, config: RunnableConfig
) -> PersonaOutputState:
    """Render the Stanley Druckenmiller verdict from state["data_bundle"]."""
    ticker = state.get("ticker", "")
    query = state.get("query")
    data_bundle = state.get("data_bundle") or {}

    if not data_bundle or "error" in data_bundle:
        fallback = StanleyDruckenmillerSignal(
            agent_id="stanley_druckenmiller",
            signal="hold",
            confidence=0.0,
            reasoning="Insufficient data",
            evidence=StanleyDruckenmillerEvidence(
                growth_momentum=ScoreDetail(score=0, max_score=10, details="no data"),
                risk_reward=ScoreDetail(score=0, max_score=10, details="no data"),
                valuation=ScoreDetail(score=0, max_score=10, details="no data"),
                sentiment=ScoreDetail(score=0, max_score=10, details="no data"),
                insider_activity=ScoreDetail(score=0, max_score=10, details="no data"),
                momentum_pct=None,
                weighted_score=0,
                market_cap=None,
                total_score=0,
                max_score=50,
            ),
        )
        return {"persona_signals": [fallback.model_dump()]}

    evidence = _compute_druckenmiller_facts(data_bundle)
    llm = ModelConfiguration.get_chat_model_for_role(
        config, "reasoner", schema=StanleyDruckenmillerSignal
    )
    prompt = render_template(
        "personas/stanley_druckenmiller.jinja",
        ticker=ticker,
        as_of_date=data_bundle.get("as_of_date", ""),
        facts=evidence.model_dump(mode="json"),
        query=query,
        persona_display_name="Stanley Druckenmiller",
        persona_slug="stanley_druckenmiller",
        signal_schema_name="StanleyDruckenmillerSignal",
    )
    result = cast(
        StanleyDruckenmillerSignal,
        await llm.ainvoke(
            [SystemMessage(prompt), HumanMessage("Render your Druckenmiller verdict.")]
        ),
    )
    return {"persona_signals": [result.model_dump()]}


PERSONA_SPEC = register_persona(
    PersonaSpec(
        slug="stanley_druckenmiller",
        display_name="Stanley Druckenmiller",
        investing_style=(
            "Macro + momentum + asymmetric risk/reward; price momentum as "
            "explicit factor; willing to pay for true growth leaders"
        ),
        node=stanley_druckenmiller_node,
        signal_schema=StanleyDruckenmillerSignal,
    )
)
