"""Mohnish Pabrai persona — Dhandho ("heads I win, tails I don't lose much").

Weighted: 0.45 × downside protection + 0.35 × valuation + 0.20 × double potential.
Mirrors ai-hedge-fund's ``mohnish_pabrai.py``.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from ...model_config import ModelConfiguration
from ...prompts import render_template
from ._base import (
    PersonaInputState,
    PersonaOutputState,
    PersonaSpec,
    register_persona,
)
from .schemas import AnalystSignal, ScoreDetail

logger = logging.getLogger(__name__)


class MohnishPabraiEvidence(BaseModel):
    """Persona-specific evidence — sub-scores, computed metrics."""

    downside_protection: ScoreDetail
    valuation: ScoreDetail
    double_potential: ScoreDetail
    weighted_score: float
    market_cap: float | None
    total_score: float
    max_score: float


class MohnishPabraiSignal(AnalystSignal):
    """Persona structured signal with narrowed evidence type."""

    agent_id: Literal["mohnish_pabrai"] = Field(default="mohnish_pabrai")
    evidence: MohnishPabraiEvidence


def _score_downside_protection(
    line_items: dict[str, list[float | None]],
) -> ScoreDetail:
    cash = [v for v in line_items.get("cash_and_equivalents", []) if v is not None]
    total_debt = [v for v in line_items.get("total_debt", []) if v is not None]
    equity = [v for v in line_items.get("shareholders_equity", []) if v is not None]
    current_assets = [v for v in line_items.get("current_assets", []) if v is not None]
    current_liab = [
        v for v in line_items.get("current_liabilities", []) if v is not None
    ]
    fcf = [v for v in line_items.get("free_cash_flow", []) if v is not None]

    score = 0
    parts: list[str] = []
    if cash and total_debt and cash[-1] > total_debt[-1]:
        score += 3
        parts.append(f"Net cash ${cash[-1] - total_debt[-1]:,.0f}")
    if current_assets and current_liab and current_liab[-1] and current_liab[-1] > 0:
        cr = current_assets[-1] / current_liab[-1]
        if cr >= 2.0:
            score += 2
            parts.append(f"Current ratio {cr:.2f}")
        elif cr >= 1.2:
            score += 1
    if total_debt and equity and equity[-1] and equity[-1] > 0:
        de = total_debt[-1] / equity[-1]
        if de < 0.3:
            score += 2
            parts.append(f"D/E {de:.2f}")
        elif de < 0.7:
            score += 1
    if fcf:
        positives = sum(1 for v in fcf if v > 0)
        if positives >= len(fcf) * 0.8:
            score += 2 if positives == len(fcf) else 1
            parts.append(f"FCF positive in {positives}/{len(fcf)} periods")
    return ScoreDetail(
        score=min(score, 10), max_score=10, details="; ".join(parts) or "Limited data"
    )


def _score_valuation(
    fcf_latest: float | None,
    market_cap: float | None,
    revenues: list[float | None],
    capex: list[float | None],
) -> ScoreDetail:
    score = 0
    parts: list[str] = []
    if fcf_latest is not None and market_cap and market_cap > 0:
        yield_ = fcf_latest / market_cap
        if yield_ > 0.10:
            score += 4
            parts.append(f"FCF yield {yield_:.1%}")
        elif yield_ > 0.07:
            score += 3
        elif yield_ > 0.05:
            score += 2
        elif yield_ > 0.03:
            score += 1
    if revenues and capex:
        rev = revenues[-1] or 0
        cap = abs(capex[-1] or 0)
        intensity = cap / rev if rev > 0 else 1
        if intensity < 0.05:
            score += 2
            parts.append(f"Capex/rev {intensity:.1%} (light)")
        elif intensity < 0.10:
            score += 1
    return ScoreDetail(
        score=min(score, 10), max_score=10, details="; ".join(parts) or "Limited data"
    )


def _score_double_potential(line_items: dict[str, list[float | None]]) -> ScoreDetail:
    revenues = [v for v in line_items.get("revenue", []) if v is not None]
    fcf = [v for v in line_items.get("free_cash_flow", []) if v is not None]

    score = 0
    parts: list[str] = []
    if len(revenues) >= 2 and revenues[0] and revenues[0] > 0:
        growth = (revenues[-1] - revenues[0]) / revenues[0]
        if growth > 0.15:
            score += 2
            parts.append(f"Revenue growth {growth:+.1%}")
        elif growth > 0.05:
            score += 1
    if len(fcf) >= 2 and fcf[0] != 0:
        fcf_growth = (fcf[-1] - fcf[0]) / abs(fcf[0])
        if fcf_growth > 0.20:
            score += 3
        elif fcf_growth > 0.08:
            score += 2
        elif fcf_growth > 0:
            score += 1
        parts.append(f"FCF growth {fcf_growth:+.1%}")
    return ScoreDetail(score=min(score, 10), max_score=10, details="; ".join(parts))


def _compute_pabrai_facts(data_bundle: dict[str, Any]) -> MohnishPabraiEvidence:
    line_items = data_bundle.get("line_items", {})
    market_cap = data_bundle.get("market_cap")
    fcf_series = line_items.get("free_cash_flow", [])
    fcf_latest = fcf_series[-1] if fcf_series else None
    revenues = line_items.get("revenue", [])
    capex = line_items.get("capital_expenditure", [])

    downside = _score_downside_protection(line_items)
    valuation = _score_valuation(fcf_latest, market_cap, revenues, capex)
    double = _score_double_potential(line_items)
    weighted = 0.45 * downside.score + 0.35 * valuation.score + 0.20 * double.score
    total = downside.score + valuation.score + double.score
    max_total = downside.max_score + valuation.max_score + double.max_score
    return MohnishPabraiEvidence(
        downside_protection=downside,
        valuation=valuation,
        double_potential=double,
        weighted_score=weighted,
        market_cap=market_cap,
        total_score=total,
        max_score=max_total,
    )


async def mohnish_pabrai_node(
    state: PersonaInputState, config: RunnableConfig
) -> PersonaOutputState:
    """Render the Mohnish Pabrai verdict from state["data_bundle"]."""
    ticker = state.get("ticker", "")
    query = state.get("query")
    data_bundle = state.get("data_bundle") or {}

    if not data_bundle or "error" in data_bundle:
        fallback = MohnishPabraiSignal(
            agent_id="mohnish_pabrai",
            signal="hold",
            confidence=0.0,
            reasoning="Insufficient data",
            evidence=MohnishPabraiEvidence(
                downside_protection=ScoreDetail(
                    score=0, max_score=10, details="no data"
                ),
                valuation=ScoreDetail(score=0, max_score=10, details="no data"),
                double_potential=ScoreDetail(score=0, max_score=10, details="no data"),
                weighted_score=0,
                market_cap=None,
                total_score=0,
                max_score=30,
            ),
        )
        return {"persona_signals": [fallback.model_dump()]}

    evidence = _compute_pabrai_facts(data_bundle)
    llm = ModelConfiguration.get_chat_model_for_role(
        config, "reasoner", schema=MohnishPabraiSignal
    )
    prompt = render_template(
        "personas/mohnish_pabrai.jinja",
        ticker=ticker,
        as_of_date=data_bundle.get("as_of_date", ""),
        facts=evidence.model_dump(mode="json"),
        query=query,
        persona_display_name="Mohnish Pabrai",
        persona_slug="mohnish_pabrai",
        signal_schema_name="MohnishPabraiSignal",
    )
    result = cast(
        MohnishPabraiSignal,
        await llm.ainvoke(
            [SystemMessage(prompt), HumanMessage("Render your Pabrai verdict now.")]
        ),
    )
    return {"persona_signals": [result.model_dump()]}


PERSONA_SPEC = register_persona(
    PersonaSpec(
        slug="mohnish_pabrai",
        display_name="Mohnish Pabrai",
        investing_style=(
            "Dhandho: heads I win, tails I don't lose much; 0.45 downside + "
            "0.35 valuation + 0.20 double potential; capex-light + FCF-yield focus"
        ),
        node=mohnish_pabrai_node,
        signal_schema=MohnishPabraiSignal,
    )
)
