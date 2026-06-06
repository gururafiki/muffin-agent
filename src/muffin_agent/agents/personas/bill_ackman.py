"""Bill Ackman persona — activist + business quality + catalyst.

Four sub-scores (max 20 total): business quality, financial discipline,
activism potential, valuation.  Mirrors ai-hedge-fund's ``bill_ackman.py``.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from ...model_config import ModelConfiguration
from ...prompts import render_template
from ...tools.scoring_helpers import compute_intrinsic_value_dcf
from ._base import (
    PersonaInputState,
    PersonaOutputState,
    PersonaSpec,
    register_persona,
)
from .schemas import AnalystSignal, ScoreDetail

logger = logging.getLogger(__name__)


class BillAckmanEvidence(BaseModel):
    """Persona-specific evidence — sub-scores, computed metrics."""

    business_quality: ScoreDetail
    financial_discipline: ScoreDetail
    activism_potential: ScoreDetail
    valuation: ScoreDetail
    intrinsic_value: float | None
    margin_of_safety: float | None
    market_cap: float | None
    total_score: float
    max_score: float


class BillAckmanSignal(AnalystSignal):
    """Persona structured signal with narrowed evidence type."""

    agent_id: Literal["bill_ackman"] = Field(default="bill_ackman")
    evidence: BillAckmanEvidence


def _score_business_quality(line_items: dict[str, list[float | None]]) -> ScoreDetail:
    revenues = [v for v in line_items.get("revenue", []) if v is not None]
    op_margins = [v for v in line_items.get("operating_margin", []) if v is not None]
    fcf = [v for v in line_items.get("free_cash_flow", []) if v is not None]

    score = 0
    parts: list[str] = []
    if len(revenues) >= 2 and revenues[0] and revenues[0] > 0:
        growth = (revenues[-1] - revenues[0]) / revenues[0]
        if growth > 0.5:
            score += 2
            parts.append(f"Revenue growth {growth:.1%} over window")
        elif growth > 0.2:
            score += 1

    if op_margins and op_margins[-1] is not None:
        if op_margins[-1] > 0.20:
            score += 2
            parts.append(f"Op margin {op_margins[-1]:.1%} (strong)")
        elif op_margins[-1] > 0.10:
            score += 1

    if fcf:
        positive = sum(1 for f in fcf if f > 0)
        if positive == len(fcf):
            score += 1
            parts.append("FCF positive every period")

    return ScoreDetail(
        score=min(score, 5), max_score=5, details="; ".join(parts) or "Limited data"
    )


def _score_financial_discipline(
    line_items: dict[str, list[float | None]],
) -> ScoreDetail:
    total_debt = [v for v in line_items.get("total_debt", []) if v is not None]
    equity = [v for v in line_items.get("shareholders_equity", []) if v is not None]
    dividends = [
        v
        for v in line_items.get("dividends_and_other_cash_distributions", [])
        if v is not None
    ]
    score = 0
    parts: list[str] = []
    if total_debt and equity and equity[-1] and equity[-1] > 0:
        de = total_debt[-1] / equity[-1]
        if de < 0.5:
            score += 2
            parts.append(f"D/E {de:.2f}")
        elif de < 1.0:
            score += 1
    if dividends:
        paid = sum(1 for d in dividends if d < 0)
        if paid >= len(dividends) // 2:
            score += 2
            parts.append(f"Pays dividends ({paid}/{len(dividends)} years)")
    return ScoreDetail(
        score=min(score, 5), max_score=5, details="; ".join(parts) or "Limited data"
    )


def _score_activism_potential(line_items: dict[str, list[float | None]]) -> ScoreDetail:
    """Ackman's activism radar (max 5).

    Share count trends + own-margin trend (proxy for sector gap).
    """
    shares = [v for v in line_items.get("outstanding_shares", []) if v is not None]
    op_margins = [v for v in line_items.get("operating_margin", []) if v is not None]
    score = 0
    parts: list[str] = []
    if len(shares) >= 2 and shares[0] > 0:
        delta = (shares[-1] - shares[0]) / shares[0]
        if delta > 0.05:
            score += 2
            parts.append(f"Shares grew {delta:+.1%} (dilution — activism opportunity)")
        elif delta < -0.05:
            score += 1
            parts.append("Buybacks underway")
    if len(op_margins) >= 4:
        recent = sum(op_margins[-2:]) / 2
        older = sum(op_margins[:2]) / 2
        gap = older - recent
        if gap > 0.05:
            score += 3
            parts.append(f"Op margin compression {gap:+.1%} — turnaround thesis")
        elif gap > 0.02:
            score += 1
    return ScoreDetail(
        score=min(score, 5),
        max_score=5,
        details="; ".join(parts) or "No activism catalysts",
    )


def _score_valuation(
    fcf_latest: float | None, market_cap: float | None
) -> tuple[ScoreDetail, float | None, float | None]:
    if not fcf_latest or fcf_latest <= 0 or not market_cap or market_cap <= 0:
        return (
            ScoreDetail(score=0, max_score=5, details="Cannot compute DCF"),
            None,
            None,
        )
    iv = compute_intrinsic_value_dcf(
        base_cash_flow=fcf_latest,
        growth_rate=0.10,
        discount_rate=0.10,
        terminal_growth_rate=0.025,
        years=5,
    )
    if iv is None:
        return (
            ScoreDetail(score=0, max_score=5, details="DCF inputs invalid"),
            None,
            None,
        )
    mos = (iv - market_cap) / market_cap
    score = 0
    parts = [f"DCF IV ${iv:,.0f}, MoS {mos:+.1%}"]
    if mos > 0.3:
        score = 5
    elif mos > 0.1:
        score = 3
    elif mos > -0.1:
        score = 1
    return ScoreDetail(score=score, max_score=5, details="; ".join(parts)), iv, mos


def _compute_ackman_facts(data_bundle: dict[str, Any]) -> BillAckmanEvidence:
    line_items = data_bundle.get("line_items", {})
    market_cap = data_bundle.get("market_cap")
    fcf_series = line_items.get("free_cash_flow", [])
    fcf_latest = fcf_series[-1] if fcf_series else None

    quality = _score_business_quality(line_items)
    discipline = _score_financial_discipline(line_items)
    activism = _score_activism_potential(line_items)
    valuation, iv, mos = _score_valuation(fcf_latest, market_cap)
    total = quality.score + discipline.score + activism.score + valuation.score
    max_total = (
        quality.max_score
        + discipline.max_score
        + activism.max_score
        + valuation.max_score
    )
    return BillAckmanEvidence(
        business_quality=quality,
        financial_discipline=discipline,
        activism_potential=activism,
        valuation=valuation,
        intrinsic_value=iv,
        margin_of_safety=mos,
        market_cap=market_cap,
        total_score=total,
        max_score=max_total,
    )


async def bill_ackman_node(
    state: PersonaInputState, config: RunnableConfig
) -> PersonaOutputState:
    """Render the Bill Ackman verdict from state["data_bundle"]."""
    ticker = state.get("ticker", "")
    query = state.get("query")
    data_bundle = state.get("data_bundle") or {}

    if not data_bundle or "error" in data_bundle:
        fallback = BillAckmanSignal(
            agent_id="bill_ackman",
            signal="hold",
            confidence=0.0,
            reasoning="Insufficient data",
            evidence=BillAckmanEvidence(
                business_quality=ScoreDetail(score=0, max_score=5, details="no data"),
                financial_discipline=ScoreDetail(
                    score=0, max_score=5, details="no data"
                ),
                activism_potential=ScoreDetail(score=0, max_score=5, details="no data"),
                valuation=ScoreDetail(score=0, max_score=5, details="no data"),
                intrinsic_value=None,
                margin_of_safety=None,
                market_cap=None,
                total_score=0,
                max_score=20,
            ),
        )
        return {"persona_signals": [fallback.model_dump()]}

    evidence = _compute_ackman_facts(data_bundle)
    llm = ModelConfiguration.get_chat_model_for_role(
        config, "reasoner", schema=BillAckmanSignal
    )
    prompt = render_template(
        "personas/bill_ackman.jinja",
        ticker=ticker,
        as_of_date=data_bundle.get("as_of_date", ""),
        facts=evidence.model_dump(mode="json"),
        query=query,
        persona_display_name="Bill Ackman",
        persona_slug="bill_ackman",
        signal_schema_name="BillAckmanSignal",
    )
    result = cast(
        BillAckmanSignal,
        await llm.ainvoke(
            [SystemMessage(prompt), HumanMessage("Render your Ackman verdict now.")]
        ),
    )
    return {"persona_signals": [result.model_dump()]}


PERSONA_SPEC = register_persona(
    PersonaSpec(
        slug="bill_ackman",
        display_name="Bill Ackman",
        investing_style=(
            "Concentrated activist; business quality + financial discipline + "
            "identifiable catalysts; multi-year horizon"
        ),
        node=bill_ackman_node,
        signal_schema=BillAckmanSignal,
    )
)
