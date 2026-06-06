"""Aswath Damodaran persona — academic FCFF DCF with CAPM cost of equity.

Three sub-scores (max 8): growth and reinvestment (4), risk profile (3),
relative valuation (1).  10-year FCFF DCF with CAPM-derived discount rate.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from ...model_config import ModelConfiguration
from ...prompts import render_template
from ...tools.scoring_helpers import compute_damodaran_fcff_dcf
from ._base import (
    PersonaInputState,
    PersonaOutputState,
    PersonaSpec,
    register_persona,
)
from .schemas import AnalystSignal, ScoreDetail

logger = logging.getLogger(__name__)


class AswathDamodaranEvidence(BaseModel):
    """Persona-specific evidence — sub-scores, computed metrics."""

    growth_reinvestment: ScoreDetail
    risk_profile: ScoreDetail
    relative_valuation: ScoreDetail
    intrinsic_value: float | None
    discount_rate: float | None
    margin_of_safety: float | None
    market_cap: float | None
    total_score: float
    max_score: float


class AswathDamodaranSignal(AnalystSignal):
    """Persona structured signal with narrowed evidence type."""

    agent_id: Literal["aswath_damodaran"] = Field(default="aswath_damodaran")
    evidence: AswathDamodaranEvidence


def _score_growth_reinvestment(
    line_items: dict[str, list[float | None]], latest_metrics: dict[str, Any]
) -> tuple[ScoreDetail, float | None]:
    revenues = [v for v in line_items.get("revenue", []) if v is not None]
    fcf = [v for v in line_items.get("free_cash_flow", []) if v is not None]
    score = 0
    parts: list[str] = []
    rev_cagr: float | None = None
    if len(revenues) >= 2 and revenues[0] > 0:
        rev_cagr = (revenues[-1] / revenues[0]) ** (1 / (len(revenues) - 1)) - 1
        if rev_cagr > 0.08:
            score += 2
            parts.append(f"Rev CAGR {rev_cagr:.1%}")
        elif rev_cagr > 0.03:
            score += 1
    if len(fcf) >= 2 and fcf[-1] > fcf[0]:
        score += 1
        parts.append("FCFF expanding")
    roic = latest_metrics.get("return_on_invested_capital")
    if roic is not None and roic > 0.10:
        score += 1
        parts.append(f"ROIC {roic:.1%}")
    return ScoreDetail(
        score=min(score, 4), max_score=4, details="; ".join(parts) or "Limited"
    ), rev_cagr


def _score_risk_profile(
    latest_metrics: dict[str, Any],
) -> tuple[ScoreDetail, float | None]:
    beta = latest_metrics.get("beta")
    de = latest_metrics.get("debt_to_equity")
    coverage = latest_metrics.get("interest_coverage")
    score = 0
    parts: list[str] = []
    if beta is not None:
        if beta < 1.3:
            score += 1
            parts.append(f"β {beta:.2f}")
    if de is not None:
        if de < 1.0:
            score += 1
    if coverage is not None and coverage > 3:
        score += 1
        parts.append(f"Interest coverage {coverage:.1f}×")
    return ScoreDetail(
        score=min(score, 3), max_score=3, details="; ".join(parts) or "Limited"
    ), beta


def _score_relative_valuation(latest_metrics: dict[str, Any]) -> ScoreDetail:
    """Light proxy — would normally be sector-median P/E comparison."""
    pe = latest_metrics.get("price_to_earnings_ratio")
    if pe is None:
        return ScoreDetail(score=0, max_score=1, details="No P/E available")
    if pe < 15:
        return ScoreDetail(
            score=1, max_score=1, details=f"P/E {pe:.1f} (cheap absolute)"
        )
    return ScoreDetail(score=0, max_score=1, details=f"P/E {pe:.1f}")


def _compute_damodaran_facts(data_bundle: dict[str, Any]) -> AswathDamodaranEvidence:
    line_items = data_bundle.get("line_items", {})
    metrics = data_bundle.get("financial_metrics", [])
    latest = metrics[0] if metrics else {}
    market_cap = data_bundle.get("market_cap")
    fcf_series = [v for v in line_items.get("free_cash_flow", []) if v is not None]
    fcf_latest = fcf_series[-1] if fcf_series else None

    growth, rev_cagr = _score_growth_reinvestment(line_items, latest)
    risk, beta = _score_risk_profile(latest)
    relative = _score_relative_valuation(latest)
    intrinsic: float | None = None
    discount_rate: float | None = None
    if fcf_latest and fcf_latest > 0:
        initial_growth = min(rev_cagr if rev_cagr is not None else 0.05, 0.12)
        result = compute_damodaran_fcff_dcf(
            base_fcff=fcf_latest,
            initial_growth=max(initial_growth, 0.0),
            beta=beta if beta is not None else 1.0,
        )
        if result is not None:
            intrinsic, discount_rate = result
    mos = (
        (intrinsic - market_cap) / market_cap
        if intrinsic is not None and market_cap and market_cap > 0
        else None
    )
    total = growth.score + risk.score + relative.score
    return AswathDamodaranEvidence(
        growth_reinvestment=growth,
        risk_profile=risk,
        relative_valuation=relative,
        intrinsic_value=intrinsic,
        discount_rate=discount_rate,
        margin_of_safety=mos,
        market_cap=market_cap,
        total_score=total,
        max_score=8,
    )


async def aswath_damodaran_node(
    state: PersonaInputState, config: RunnableConfig
) -> PersonaOutputState:
    """Render the Aswath Damodaran verdict from state["data_bundle"]."""
    ticker = state.get("ticker", "")
    query = state.get("query")
    data_bundle = state.get("data_bundle") or {}

    if not data_bundle or "error" in data_bundle:
        fallback = AswathDamodaranSignal(
            agent_id="aswath_damodaran",
            signal="hold",
            confidence=0.0,
            reasoning="Insufficient data",
            evidence=AswathDamodaranEvidence(
                growth_reinvestment=ScoreDetail(
                    score=0, max_score=4, details="no data"
                ),
                risk_profile=ScoreDetail(score=0, max_score=3, details="no data"),
                relative_valuation=ScoreDetail(score=0, max_score=1, details="no data"),
                intrinsic_value=None,
                discount_rate=None,
                margin_of_safety=None,
                market_cap=None,
                total_score=0,
                max_score=8,
            ),
        )
        return {"persona_signals": [fallback.model_dump()]}

    evidence = _compute_damodaran_facts(data_bundle)
    llm = ModelConfiguration.get_chat_model_for_role(
        config, "reasoner", schema=AswathDamodaranSignal
    )
    prompt = render_template(
        "personas/aswath_damodaran.jinja",
        ticker=ticker,
        as_of_date=data_bundle.get("as_of_date", ""),
        facts=evidence.model_dump(mode="json"),
        query=query,
        persona_display_name="Aswath Damodaran",
        persona_slug="aswath_damodaran",
        signal_schema_name="AswathDamodaranSignal",
    )
    result = cast(
        AswathDamodaranSignal,
        await llm.ainvoke(
            [SystemMessage(prompt), HumanMessage("Render your Damodaran verdict.")]
        ),
    )
    return {"persona_signals": [result.model_dump()]}


PERSONA_SPEC = register_persona(
    PersonaSpec(
        slug="aswath_damodaran",
        display_name="Aswath Damodaran",
        investing_style=(
            "Academic FCFF DCF + CAPM cost of equity (rf 4% + β × 5% ERP); "
            "story-then-numbers approach; MoS ≥25%"
        ),
        node=aswath_damodaran_node,
        signal_schema=AswathDamodaranSignal,
    )
)
