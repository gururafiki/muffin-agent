"""Cathie Wood persona — disruptive innovation, exponential growth.

High-growth DCF (20% growth, 15% disc, 25× terminal) + R&D intensity +
revenue acceleration scoring.  Mirrors ai-hedge-fund's ``cathie_wood.py``.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from ...model_config import ModelConfiguration
from ...prompts import render_template
from ...tools.scoring_helpers import compute_intrinsic_value_exit_multiple
from ._base import (
    PersonaInputState,
    PersonaOutputState,
    PersonaSpec,
    register_persona,
)
from .schemas import AnalystSignal, ScoreDetail

logger = logging.getLogger(__name__)


class CathieWoodEvidence(BaseModel):
    """Wood-specific evidence."""

    disruptive_potential: ScoreDetail
    innovation_growth: ScoreDetail
    valuation: ScoreDetail
    intrinsic_value: float | None
    margin_of_safety: float | None
    market_cap: float | None
    total_score: float
    max_score: float


class CathieWoodSignal(AnalystSignal):
    """Cathie Wood structured signal."""

    agent_id: Literal["cathie_wood"] = Field(default="cathie_wood")
    evidence: CathieWoodEvidence


def _score_disruptive_potential(
    line_items: dict[str, list[float | None]],
) -> ScoreDetail:
    """Score Wood's disruptive-potential dimension (max 5, normalised from 12)."""
    revenues = [v for v in line_items.get("revenue", []) if v is not None]
    gross_margins = [v for v in line_items.get("gross_margin", []) if v is not None]
    op_exp = [v for v in line_items.get("operating_expense", []) if v is not None]
    rd = [v for v in line_items.get("research_and_development", []) if v is not None]

    score = 0
    parts: list[str] = []

    # Revenue growth: oldest→newest. Acceleration = recent growth > older growth.
    if len(revenues) >= 3:
        growth_rates = []
        for i in range(1, len(revenues)):
            base = revenues[i - 1]
            if base and base != 0:
                growth_rates.append((revenues[i] - base) / abs(base))
        if len(growth_rates) >= 2 and growth_rates[-1] > growth_rates[0]:
            score += 2
            parts.append(
                f"Revenue acceleration ({growth_rates[0]:.1%} → {growth_rates[-1]:.1%})"
            )
        latest_growth = growth_rates[-1] if growth_rates else 0
        if latest_growth > 1.0:
            score += 3
            parts.append(f"Exceptional revenue growth {latest_growth:.1%}")
        elif latest_growth > 0.5:
            score += 2
            parts.append(f"Strong revenue growth {latest_growth:.1%}")
        elif latest_growth > 0.2:
            score += 1
            parts.append(f"Moderate revenue growth {latest_growth:.1%}")

    if len(gross_margins) >= 2:
        margin_change = gross_margins[-1] - gross_margins[0]
        if margin_change > 0.05:
            score += 2
            parts.append(f"Expanding gross margins {margin_change:+.1%}")
        elif margin_change > 0:
            score += 1
            parts.append(f"Slightly improving gross margins {margin_change:+.1%}")
        if gross_margins[-1] > 0.5:
            score += 2
            parts.append(f"High gross margin {gross_margins[-1]:.1%}")

    if len(revenues) >= 2 and len(op_exp) >= 2 and revenues[0] != 0 and op_exp[0] != 0:
        rev_growth = (revenues[-1] - revenues[0]) / abs(revenues[0])
        opex_growth = (op_exp[-1] - op_exp[0]) / abs(op_exp[0])
        if rev_growth > opex_growth:
            score += 2
            parts.append("Positive operating leverage")

    if rd and revenues:
        rd_intensity = rd[-1] / revenues[-1] if revenues[-1] else 0
        if rd_intensity > 0.15:
            score += 3
            parts.append(f"High R&D intensity {rd_intensity:.1%}")
        elif rd_intensity > 0.08:
            score += 2
            parts.append(f"Moderate R&D {rd_intensity:.1%}")
        elif rd_intensity > 0.05:
            score += 1
            parts.append(f"Some R&D {rd_intensity:.1%}")

    max_raw = 12
    normalized = (score / max_raw) * 5
    return ScoreDetail(
        score=normalized,
        max_score=5,
        details="; ".join(parts) if parts else "Insufficient data",
        metrics={"raw_score": score},
    )


def _score_innovation_growth(line_items: dict[str, list[float | None]]) -> ScoreDetail:
    """Score Wood's innovation-growth dimension (max 5, normalised from 15)."""
    rd = [v for v in line_items.get("research_and_development", []) if v is not None]
    revenues = [v for v in line_items.get("revenue", []) if v is not None]
    fcf = [v for v in line_items.get("free_cash_flow", []) if v is not None]
    op_margins = [v for v in line_items.get("operating_margin", []) if v is not None]
    capex = [abs(v) for v in line_items.get("capital_expenditure", []) if v is not None]
    dividends = [
        v
        for v in line_items.get("dividends_and_other_cash_distributions", [])
        if v is not None
    ]

    score = 0
    parts: list[str] = []

    if len(rd) >= 2 and rd[0] != 0:
        rd_growth = (rd[-1] - rd[0]) / abs(rd[0])
        if rd_growth > 0.5:
            score += 3
            parts.append(f"Strong R&D growth {rd_growth:+.1%}")
        elif rd_growth > 0.2:
            score += 2
            parts.append(f"Moderate R&D growth {rd_growth:+.1%}")
        if len(revenues) >= 2 and revenues[0] != 0 and revenues[-1] != 0:
            start_intensity = rd[0] / revenues[0]
            end_intensity = rd[-1] / revenues[-1]
            if end_intensity > start_intensity:
                score += 2
                parts.append("Increasing R&D intensity")

    if len(fcf) >= 2:
        positive_fcf = sum(1 for f in fcf if f > 0)
        if fcf[0] != 0:
            fcf_growth = (fcf[-1] - fcf[0]) / abs(fcf[0])
        else:
            fcf_growth = 0
        if fcf_growth > 0.3 and positive_fcf == len(fcf):
            score += 3
            parts.append("Strong consistent FCF growth")
        elif positive_fcf >= len(fcf) * 0.75:
            score += 2
            parts.append("Consistent positive FCF")
        elif positive_fcf > len(fcf) // 2:
            score += 1
            parts.append("Moderately positive FCF")

    if len(op_margins) >= 2:
        margin_trend = op_margins[-1] - op_margins[0]
        if op_margins[-1] > 0.15 and margin_trend > 0:
            score += 3
            parts.append(f"Strong improving op margin {op_margins[-1]:.1%}")
        elif op_margins[-1] > 0.10:
            score += 2
            parts.append(f"Healthy op margin {op_margins[-1]:.1%}")
        elif margin_trend > 0:
            score += 1

    if capex and revenues and revenues[-1] and capex[-1] != 0:
        capex_intensity = capex[-1] / revenues[-1]
        if capex[0] != 0:
            capex_growth = (capex[-1] - capex[0]) / abs(capex[0])
        else:
            capex_growth = 0
        if capex_intensity > 0.10 and capex_growth > 0.2:
            score += 2
            parts.append("Heavy growth investment")
        elif capex_intensity > 0.05:
            score += 1
            parts.append("Moderate growth investment")

    if dividends and fcf and fcf[-1] != 0:
        # dividends are negative outflows; we look at the magnitude
        payout = abs(dividends[-1] / fcf[-1])
        if payout < 0.2:
            score += 2
            parts.append("Reinvests heavily over dividends")
        elif payout < 0.4:
            score += 1

    max_raw = 15
    normalized = (score / max_raw) * 5
    return ScoreDetail(
        score=normalized,
        max_score=5,
        details="; ".join(parts) if parts else "Insufficient data",
        metrics={"raw_score": score},
    )


def _score_valuation(
    fcf_latest: float | None, market_cap: float | None
) -> tuple[ScoreDetail, float | None, float | None]:
    """High-growth DCF + MoS (max 5, max from raw 4 normalised).

    Returns ``(ScoreDetail, intrinsic_value, margin_of_safety)``.
    """
    if not fcf_latest or fcf_latest <= 0 or not market_cap or market_cap <= 0:
        return (
            ScoreDetail(
                score=0,
                max_score=5,
                details="Cannot compute DCF (need positive FCF + market cap)",
            ),
            None,
            None,
        )
    iv = compute_intrinsic_value_exit_multiple(
        base_cash_flow=fcf_latest,
        growth_rate=0.20,
        discount_rate=0.15,
        terminal_multiple=25.0,
        years=5,
    )
    if iv is None:
        return (
            ScoreDetail(
                score=0,
                max_score=5,
                details="DCF inputs invalid",
            ),
            None,
            None,
        )
    mos = (iv - market_cap) / market_cap
    score = 0
    parts = [f"DCF IV ${iv:,.0f}, market cap ${market_cap:,.0f}, MoS {mos:+.1%}"]
    if mos > 0.5:
        score = 3
        parts.append("MoS > 50% — strong undervaluation")
    elif mos > 0.2:
        score = 1
        parts.append("MoS > 20% — modest undervaluation")
    elif mos < -0.5:
        parts.append("Severe overvaluation")
    return ScoreDetail(score=score, max_score=5, details="; ".join(parts)), iv, mos


def _compute_wood_facts(data_bundle: dict[str, Any]) -> CathieWoodEvidence:
    """Compute Cathie Wood evidence from a PersonaDataBundle dict."""
    line_items = data_bundle.get("line_items", {})
    market_cap = data_bundle.get("market_cap")
    fcf_series = line_items.get("free_cash_flow", [])
    fcf_latest = fcf_series[-1] if fcf_series else None

    disruptive = _score_disruptive_potential(line_items)
    innovation = _score_innovation_growth(line_items)
    valuation, iv, mos = _score_valuation(fcf_latest, market_cap)

    total = disruptive.score + innovation.score + valuation.score
    max_total = disruptive.max_score + innovation.max_score + valuation.max_score
    return CathieWoodEvidence(
        disruptive_potential=disruptive,
        innovation_growth=innovation,
        valuation=valuation,
        intrinsic_value=iv,
        margin_of_safety=mos,
        market_cap=market_cap,
        total_score=total,
        max_score=max_total,
    )


async def cathie_wood_node(
    state: PersonaInputState, config: RunnableConfig
) -> PersonaOutputState:
    """Render the Cathie Wood verdict."""
    ticker = state.get("ticker", "")
    query = state.get("query")
    data_bundle = state.get("data_bundle") or {}

    if not data_bundle or "error" in data_bundle:
        logger.warning("cathie_wood_node: data_bundle missing or errored")
        fallback = CathieWoodSignal(
            agent_id="cathie_wood",
            signal="hold",
            confidence=0.0,
            reasoning="Insufficient data — defaulting to hold",
            evidence=CathieWoodEvidence(
                disruptive_potential=ScoreDetail(
                    score=0, max_score=5, details="no data"
                ),
                innovation_growth=ScoreDetail(score=0, max_score=5, details="no data"),
                valuation=ScoreDetail(score=0, max_score=5, details="no data"),
                intrinsic_value=None,
                margin_of_safety=None,
                market_cap=None,
                total_score=0,
                max_score=15,
            ),
        )
        return {"persona_signals": [fallback.model_dump()]}

    evidence = _compute_wood_facts(data_bundle)
    llm = ModelConfiguration.get_chat_model_for_role(
        config, "reasoner", schema=CathieWoodSignal
    )
    prompt = render_template(
        "personas/cathie_wood.jinja",
        ticker=ticker,
        as_of_date=data_bundle.get("as_of_date", ""),
        facts=evidence.model_dump(mode="json"),
        query=query,
        persona_display_name="Cathie Wood",
        persona_slug="cathie_wood",
        signal_schema_name="CathieWoodSignal",
    )
    result = cast(
        CathieWoodSignal,
        await llm.ainvoke(
            [
                SystemMessage(prompt),
                HumanMessage("Render your Cathie Wood verdict now."),
            ]
        ),
    )
    return {"persona_signals": [result.model_dump()]}


PERSONA_SPEC = register_persona(
    PersonaSpec(
        slug="cathie_wood",
        display_name="Cathie Wood",
        investing_style=(
            "Disruptive innovation, exponential growth, R&D intensity ≥15%, "
            "high-growth DCF (20%/15%/25× terminal), 5-year exponential thesis"
        ),
        node=cathie_wood_node,
        signal_schema=CathieWoodSignal,
    )
)
