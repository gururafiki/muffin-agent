"""Phil Fisher persona — qualitative growth + R&D + management.

Weighted: 0.30 growth_quality + 0.25 mgmt + 0.20 margins + 0.15 valuation +
0.05 insider + 0.05 sentiment.  Mirrors ai-hedge-fund's ``phil_fisher.py``.
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
from ...tools.scoring_helpers import score_insider_buy_ratio
from ...tools.sentiment import aggregate_news_sentiment
from ._base import (
    PersonaInputState,
    PersonaOutputState,
    PersonaSpec,
    register_persona,
)
from .schemas import AnalystSignal, ScoreDetail

logger = logging.getLogger(__name__)


class PhilFisherEvidence(BaseModel):
    """Persona-specific evidence — sub-scores, computed metrics."""

    growth_quality: ScoreDetail
    margins_stability: ScoreDetail
    management_efficiency: ScoreDetail
    valuation: ScoreDetail
    sentiment: ScoreDetail
    insider_activity: ScoreDetail
    weighted_score: float
    market_cap: float | None
    total_score: float
    max_score: float


class PhilFisherSignal(AnalystSignal):
    """Persona structured signal with narrowed evidence type."""

    agent_id: Literal["phil_fisher"] = Field(default="phil_fisher")
    evidence: PhilFisherEvidence


def _cagr(series: list[float]) -> float | None:
    if len(series) < 2 or series[0] <= 0:
        return None
    return (series[-1] / series[0]) ** (1 / (len(series) - 1)) - 1


def _score_growth_quality(line_items: dict[str, list[float | None]]) -> ScoreDetail:
    revenues = [v for v in line_items.get("revenue", []) if v is not None]
    eps = [v for v in line_items.get("earnings_per_share", []) if v is not None]
    rd = [v for v in line_items.get("research_and_development", []) if v is not None]
    score = 0
    parts: list[str] = []
    rev_cagr = _cagr(revenues)
    eps_cagr = _cagr(eps)
    if rev_cagr is not None:
        if rev_cagr > 0.20:
            score += 3
            parts.append(f"Rev CAGR {rev_cagr:.1%}")
        elif rev_cagr > 0.10:
            score += 2
        elif rev_cagr > 0.03:
            score += 1
    if eps_cagr is not None:
        if eps_cagr > 0.20:
            score += 3
            parts.append(f"EPS CAGR {eps_cagr:.1%}")
        elif eps_cagr > 0.10:
            score += 2
        elif eps_cagr > 0.03:
            score += 1
    if rd and revenues and revenues[-1]:
        ratio = rd[-1] / revenues[-1]
        if 0.03 <= ratio <= 0.15:
            score += 3
            parts.append(f"R&D/rev {ratio:.1%} (Fisher zone)")
        elif ratio > 0.15:
            score += 2
        elif ratio > 0:
            score += 1
    normalised = (score / 9) * 10
    return ScoreDetail(
        score=normalised, max_score=10, details="; ".join(parts) or "Limited data"
    )


def _score_margins_stability(line_items: dict[str, list[float | None]]) -> ScoreDetail:
    op_margins = [v for v in line_items.get("operating_margin", []) if v is not None]
    gross_margins = [v for v in line_items.get("gross_margin", []) if v is not None]
    score = 0
    parts: list[str] = []
    if op_margins:
        if op_margins[-1] is not None and op_margins[-1] > 0:
            score += 1
        if len(op_margins) >= 2 and op_margins[-1] >= op_margins[0]:
            score += 2
            parts.append("Stable / improving op margin")
    if gross_margins:
        if gross_margins[-1] is not None and gross_margins[-1] > 0.50:
            score += 2
            parts.append(f"Gross margin {gross_margins[-1]:.1%}")
        elif gross_margins[-1] is not None and gross_margins[-1] > 0.30:
            score += 1
    if op_margins and len(op_margins) >= 3:
        mean = sum(op_margins) / len(op_margins)
        if mean > 0:
            cv = statistics.pstdev(op_margins) / mean
            if cv < 0.02:
                score += 2
                parts.append(f"Highly stable op margin (CV {cv:.1%})")
            elif cv < 0.05:
                score += 1
    normalised = (score / 6) * 10
    return ScoreDetail(
        score=normalised, max_score=10, details="; ".join(parts) or "Limited data"
    )


def _score_management_efficiency(
    latest_metrics: dict[str, Any], line_items: dict[str, list[float | None]]
) -> ScoreDetail:
    roe = latest_metrics.get("return_on_equity")
    de = latest_metrics.get("debt_to_equity")
    fcf = [v for v in line_items.get("free_cash_flow", []) if v is not None]
    score = 0
    parts: list[str] = []
    if roe is not None:
        if roe > 0.20:
            score += 3
            parts.append(f"ROE {roe:.1%}")
        elif roe > 0.10:
            score += 2
        elif roe > 0:
            score += 1
    if de is not None:
        if de < 0.3:
            score += 2
            parts.append(f"D/E {de:.2f}")
        elif de < 1.0:
            score += 1
    if fcf:
        positives = sum(1 for v in fcf if v > 0)
        if positives / len(fcf) >= 0.8:
            score += 1
            parts.append(f"FCF positive {positives}/{len(fcf)}")
    normalised = (score / 6) * 10
    return ScoreDetail(
        score=normalised, max_score=10, details="; ".join(parts) or "Limited data"
    )


def _score_valuation(latest_metrics: dict[str, Any]) -> ScoreDetail:
    pe = latest_metrics.get("price_to_earnings_ratio")
    score = 0
    parts: list[str] = []
    if pe is not None:
        if pe < 20:
            score += 2
            parts.append(f"P/E {pe:.1f}")
        elif pe < 30:
            score += 1
    normalised = (score / 4) * 10
    return ScoreDetail(
        score=normalised, max_score=10, details="; ".join(parts) or "Cannot value"
    )


def _score_sentiment(articles: list[dict[str, Any]]) -> ScoreDetail:
    agg = aggregate_news_sentiment(articles)
    bullish = agg["bullish_articles"]
    bearish = agg["bearish_articles"]
    total = agg["total_articles"]
    if total == 0:
        return ScoreDetail(score=5, max_score=10, details="No news — neutral")
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


def _compute_fisher_facts(data_bundle: dict[str, Any]) -> PhilFisherEvidence:
    line_items = data_bundle.get("line_items", {})
    metrics = data_bundle.get("financial_metrics", [])
    latest_metrics = metrics[0] if metrics else {}
    insider_trades = data_bundle.get("insider_trades", [])
    news = data_bundle.get("company_news", [])
    market_cap = data_bundle.get("market_cap")

    growth = _score_growth_quality(line_items)
    margins = _score_margins_stability(line_items)
    mgmt = _score_management_efficiency(latest_metrics, line_items)
    valuation = _score_valuation(latest_metrics)
    sentiment = _score_sentiment(news)
    insider = _score_insider(insider_trades)
    weighted = (
        0.30 * growth.score
        + 0.25 * mgmt.score
        + 0.20 * margins.score
        + 0.15 * valuation.score
        + 0.05 * sentiment.score
        + 0.05 * insider.score
    )
    total = (
        growth.score
        + margins.score
        + mgmt.score
        + valuation.score
        + sentiment.score
        + insider.score
    )
    max_total = 60
    return PhilFisherEvidence(
        growth_quality=growth,
        margins_stability=margins,
        management_efficiency=mgmt,
        valuation=valuation,
        sentiment=sentiment,
        insider_activity=insider,
        weighted_score=weighted,
        market_cap=market_cap,
        total_score=total,
        max_score=max_total,
    )


async def phil_fisher_node(
    state: PersonaInputState, config: RunnableConfig
) -> PersonaOutputState:
    """Render the Phil Fisher verdict from state["data_bundle"]."""
    ticker = state.get("ticker", "")
    query = state.get("query")
    data_bundle = state.get("data_bundle") or {}

    if not data_bundle or "error" in data_bundle:
        fallback = PhilFisherSignal(
            agent_id="phil_fisher",
            signal="hold",
            confidence=0.0,
            reasoning="Insufficient data",
            evidence=PhilFisherEvidence(
                growth_quality=ScoreDetail(score=0, max_score=10, details="no data"),
                margins_stability=ScoreDetail(score=0, max_score=10, details="no data"),
                management_efficiency=ScoreDetail(
                    score=0, max_score=10, details="no data"
                ),
                valuation=ScoreDetail(score=0, max_score=10, details="no data"),
                sentiment=ScoreDetail(score=0, max_score=10, details="no data"),
                insider_activity=ScoreDetail(score=0, max_score=10, details="no data"),
                weighted_score=0,
                market_cap=None,
                total_score=0,
                max_score=60,
            ),
        )
        return {"persona_signals": [fallback.model_dump()]}

    evidence = _compute_fisher_facts(data_bundle)
    llm = ModelConfiguration.get_chat_model_for_role(
        config, "reasoner", schema=PhilFisherSignal
    )
    prompt = render_template(
        "personas/phil_fisher.jinja",
        ticker=ticker,
        as_of_date=data_bundle.get("as_of_date", ""),
        facts=evidence.model_dump(mode="json"),
        query=query,
        persona_display_name="Phil Fisher",
        persona_slug="phil_fisher",
        signal_schema_name="PhilFisherSignal",
    )
    result = cast(
        PhilFisherSignal,
        await llm.ainvoke(
            [SystemMessage(prompt), HumanMessage("Render your Fisher verdict now.")]
        ),
    )
    return {"persona_signals": [result.model_dump()]}


PERSONA_SPEC = register_persona(
    PersonaSpec(
        slug="phil_fisher",
        display_name="Phil Fisher",
        investing_style=(
            "Qualitative growth + scuttlebutt; R&D 3-15% of rev; long-term "
            "compounders; willing to pay up for quality"
        ),
        node=phil_fisher_node,
        signal_schema=PhilFisherSignal,
    )
)
