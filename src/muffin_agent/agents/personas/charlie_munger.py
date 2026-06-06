"""Charlie Munger persona — mental models + quality + predictability.

Quality-biased weighting: 0.35 moat + 0.25 management + 0.25 predictability +
0.15 valuation.  Uses ROIC consistency (Munger's favourite metric), FCF
yield-based valuation, and insider buy ratio.  Mirrors ai-hedge-fund's
``charlie_munger.py`` scoring shape.
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
from ._base import (
    PersonaInputState,
    PersonaOutputState,
    PersonaSpec,
    register_persona,
)
from .schemas import AnalystSignal, ScoreDetail

logger = logging.getLogger(__name__)


class CharlieMungerEvidence(BaseModel):
    """Munger evidence — four 0–10 sub-scores plus quality-weighted total."""

    moat_strength: ScoreDetail
    management_quality: ScoreDetail
    predictability: ScoreDetail
    valuation: ScoreDetail
    weighted_score: float
    """0.35 × moat + 0.25 × mgmt + 0.25 × predictability + 0.15 × valuation
    (all on 0–10 scale)."""
    flags: dict[str, bool]
    """Boolean filters Munger applies — strong moat, predictable cash flows,
    owner-aligned management, low leverage, positive MoS."""
    market_cap: float | None
    total_score: float
    max_score: float


class CharlieMungerSignal(AnalystSignal):
    """Munger structured signal."""

    agent_id: Literal["charlie_munger"] = Field(default="charlie_munger")
    evidence: CharlieMungerEvidence


def _score_moat(line_items: dict[str, list[float | None]]) -> ScoreDetail:
    """Score Munger's moat dimension (max 10).

    ROIC consistency + margin + capex + R&D + intangibles.
    """
    roics = [
        v for v in line_items.get("return_on_invested_capital", []) if v is not None
    ]
    margins = [v for v in line_items.get("operating_margin", []) if v is not None]
    capex = [abs(v) for v in line_items.get("capital_expenditure", []) if v is not None]
    revenues = [v for v in line_items.get("revenue", []) if v is not None]
    rd = [v for v in line_items.get("research_and_development", []) if v is not None]
    goodwill = [
        v for v in line_items.get("goodwill_and_intangible_assets", []) if v is not None
    ]

    score = 0
    parts: list[str] = []

    if roics:
        high_count = sum(1 for r in roics if r > 0.15)
        consistency = high_count / len(roics)
        if consistency >= 0.8:
            score += 3
            parts.append(f"ROIC > 15% in {high_count}/{len(roics)} periods")
        elif consistency >= 0.5:
            score += 2
            parts.append(f"ROIC > 15% in {high_count}/{len(roics)} (decent)")

    if len(margins) >= 3:
        positive = sum(1 for m in margins if m > 0.20)
        if positive >= len(margins) * 0.7:
            score += 2
            parts.append("Stable / improving operating margins > 20%")
        elif sum(margins) / len(margins) > 0.30:
            score += 1
            parts.append("Average margin > 30%")

    if capex and revenues:
        latest_intensity = capex[-1] / revenues[-1] if revenues[-1] else 1
        if latest_intensity < 0.05:
            score += 2
            parts.append(f"Low capex intensity {latest_intensity:.1%}")
        elif latest_intensity < 0.10:
            score += 1
            parts.append(f"Moderate capex intensity {latest_intensity:.1%}")

    if rd and any(v > 0 for v in rd):
        score += 1
        parts.append("Active R&D investment")

    if goodwill and any(v > 0 for v in goodwill):
        score += 1
        parts.append("Intangible assets present (brand / IP)")

    return ScoreDetail(score=min(score, 10), max_score=10, details="; ".join(parts))


def _score_management(
    line_items: dict[str, list[float | None]],
    insider_trades: list[dict[str, Any]],
) -> ScoreDetail:
    """Score Munger's management dimension (max 10).

    FCF/NI quality + D/E + cash position + insiders + share count.
    """
    fcf = [v for v in line_items.get("free_cash_flow", []) if v is not None]
    net_income = [v for v in line_items.get("net_income", []) if v is not None]
    total_debt = [v for v in line_items.get("total_debt", []) if v is not None]
    equity = [v for v in line_items.get("shareholders_equity", []) if v is not None]
    cash = [v for v in line_items.get("cash_and_equivalents", []) if v is not None]
    revenues = [v for v in line_items.get("revenue", []) if v is not None]
    shares = [v for v in line_items.get("outstanding_shares", []) if v is not None]

    score = 0
    parts: list[str] = []

    if fcf and net_income and net_income[-1] != 0:
        ratio = fcf[-1] / net_income[-1]
        if ratio > 1.1:
            score += 3
            parts.append(f"FCF/NI {ratio:.2f} (excellent earnings quality)")
        elif ratio > 0.9:
            score += 2
            parts.append(f"FCF/NI {ratio:.2f} (good)")
        elif ratio > 0.7:
            score += 1
            parts.append(f"FCF/NI {ratio:.2f}")

    if total_debt and equity and equity[-1] and equity[-1] > 0:
        de = total_debt[-1] / equity[-1]
        if de < 0.3:
            score += 3
            parts.append(f"D/E {de:.2f} (very conservative)")
        elif de < 0.7:
            score += 2
            parts.append(f"D/E {de:.2f}")
        elif de < 1.5:
            score += 1

    if cash and revenues and revenues[-1] and revenues[-1] > 0:
        cash_ratio = cash[-1] / revenues[-1]
        if 0.10 <= cash_ratio <= 0.25:
            score += 2
            parts.append(f"Sensible cash position {cash_ratio:.1%} of revenue")
        elif 0.05 <= cash_ratio < 0.10 or 0.25 < cash_ratio <= 0.40:
            score += 1

    insider_score = score_insider_buy_ratio(insider_trades)
    if insider_score.score >= 8:
        score += 2
        parts.append("Heavy insider buying")
    elif insider_score.score >= 6:
        score += 1
        parts.append("Balanced insider activity")
    elif insider_score.score < 5:
        score -= 1
        parts.append("Net insider selling")

    if len(shares) >= 2:
        if shares[-1] < shares[0]:
            score += 2
            parts.append("Share count decreasing (buybacks)")
        elif shares[-1] == shares[0]:
            score += 1
        else:
            score -= 1
            parts.append("Shares diluting")

    return ScoreDetail(
        score=max(0, min(score, 10)),
        max_score=10,
        details="; ".join(parts),
    )


def _score_predictability(line_items: dict[str, list[float | None]]) -> ScoreDetail:
    """Score Munger's predictability dimension (max 10).

    Looks at revenue, op income, margin, FCF stability.
    """
    revenues = [v for v in line_items.get("revenue", []) if v is not None]
    op_income = [v for v in line_items.get("operating_income", []) if v is not None]
    op_margins = [v for v in line_items.get("operating_margin", []) if v is not None]
    fcf = [v for v in line_items.get("free_cash_flow", []) if v is not None]

    score = 0
    parts: list[str] = []

    if len(revenues) >= 4:
        mean = sum(revenues) / len(revenues)
        if mean > 0:
            cv = statistics.pstdev(revenues) / mean
            if cv < 0.10:
                score += 3
                parts.append(f"Revenue stable (CV {cv:.1%})")
            elif cv < 0.20:
                score += 2
                parts.append(f"Revenue moderately stable (CV {cv:.1%})")

    if op_income:
        positives = sum(1 for v in op_income if v > 0)
        ratio = positives / len(op_income)
        if ratio == 1.0:
            score += 3
            parts.append("Operating income positive every period")
        elif ratio >= 0.8:
            score += 2
        elif ratio >= 0.6:
            score += 1

    if len(op_margins) >= 4:
        mean = sum(op_margins) / len(op_margins)
        if mean > 0:
            cv = statistics.pstdev(op_margins) / mean
            if cv < 0.03:
                score += 2
                parts.append("Highly stable margins")
            elif cv < 0.07:
                score += 1

    if fcf:
        positives = sum(1 for v in fcf if v > 0)
        ratio = positives / len(fcf)
        if ratio == 1.0:
            score += 2
            parts.append("FCF positive every period")
        elif ratio >= 0.8:
            score += 1

    return ScoreDetail(score=min(score, 10), max_score=10, details="; ".join(parts))


def _score_valuation(fcf_latest: float | None, market_cap: float | None) -> ScoreDetail:
    """Score Munger's valuation dimension (max 10).

    Uses FCF yield + multiple-based MoS.
    """
    if not fcf_latest or fcf_latest <= 0 or not market_cap or market_cap <= 0:
        return ScoreDetail(
            score=0,
            max_score=10,
            details="No FCF or market cap available",
        )
    fcf_yield = fcf_latest / market_cap
    score = 0
    parts: list[str] = []
    if fcf_yield > 0.08:
        score += 4
        parts.append(f"FCF yield {fcf_yield:.1%} (very attractive)")
    elif fcf_yield > 0.05:
        score += 3
        parts.append(f"FCF yield {fcf_yield:.1%}")
    elif fcf_yield > 0.03:
        score += 1

    # Multiple-based fair value: 15x FCF is fair, 10x cheap, 20x optimistic
    fair_value = fcf_latest * 15
    mos = (fair_value - market_cap) / market_cap
    if mos > 0.3:
        score += 3
        parts.append(f"MoS {mos:.1%} at 15× FCF multiple")
    elif mos > 0.1:
        score += 2
        parts.append(f"MoS {mos:.1%}")
    elif mos > -0.1:
        score += 1

    return ScoreDetail(score=min(score, 10), max_score=10, details="; ".join(parts))


def _compute_munger_facts(data_bundle: dict[str, Any]) -> CharlieMungerEvidence:
    """Compute Munger evidence from a PersonaDataBundle dict."""
    line_items = data_bundle.get("line_items", {})
    insider_trades = data_bundle.get("insider_trades", [])
    market_cap = data_bundle.get("market_cap")
    fcf_series = line_items.get("free_cash_flow", [])
    fcf_latest = fcf_series[-1] if fcf_series else None

    moat = _score_moat(line_items)
    management = _score_management(line_items, insider_trades)
    predictability = _score_predictability(line_items)
    valuation = _score_valuation(fcf_latest, market_cap)

    weighted = (
        0.35 * moat.score
        + 0.25 * management.score
        + 0.25 * predictability.score
        + 0.15 * valuation.score
    )

    flags = {
        "moat_strong": moat.score >= 7,
        "predictable": predictability.score >= 7,
        "owner_aligned": management.score >= 7,
        "valuation_ok": valuation.score >= 5,
    }
    total = moat.score + management.score + predictability.score + valuation.score
    max_total = (
        moat.max_score
        + management.max_score
        + predictability.max_score
        + valuation.max_score
    )

    return CharlieMungerEvidence(
        moat_strength=moat,
        management_quality=management,
        predictability=predictability,
        valuation=valuation,
        weighted_score=weighted,
        flags=flags,
        market_cap=market_cap,
        total_score=total,
        max_score=max_total,
    )


async def charlie_munger_node(
    state: PersonaInputState, config: RunnableConfig
) -> PersonaOutputState:
    """Render the Charlie Munger verdict."""
    ticker = state.get("ticker", "")
    query = state.get("query")
    data_bundle = state.get("data_bundle") or {}

    if not data_bundle or "error" in data_bundle:
        fallback = CharlieMungerSignal(
            agent_id="charlie_munger",
            signal="hold",
            confidence=0.0,
            reasoning="Insufficient data — defaulting to hold",
            evidence=CharlieMungerEvidence(
                moat_strength=ScoreDetail(score=0, max_score=10, details="no data"),
                management_quality=ScoreDetail(
                    score=0, max_score=10, details="no data"
                ),
                predictability=ScoreDetail(score=0, max_score=10, details="no data"),
                valuation=ScoreDetail(score=0, max_score=10, details="no data"),
                weighted_score=0,
                flags={
                    "moat_strong": False,
                    "predictable": False,
                    "owner_aligned": False,
                    "valuation_ok": False,
                },
                market_cap=None,
                total_score=0,
                max_score=40,
            ),
        )
        return {"persona_signals": [fallback.model_dump()]}

    evidence = _compute_munger_facts(data_bundle)
    llm = ModelConfiguration.get_chat_model_for_role(
        config, "reasoner", schema=CharlieMungerSignal
    )
    prompt = render_template(
        "personas/charlie_munger.jinja",
        ticker=ticker,
        as_of_date=data_bundle.get("as_of_date", ""),
        facts=evidence.model_dump(mode="json"),
        query=query,
        persona_display_name="Charlie Munger",
        persona_slug="charlie_munger",
        signal_schema_name="CharlieMungerSignal",
    )
    result = cast(
        CharlieMungerSignal,
        await llm.ainvoke(
            [SystemMessage(prompt), HumanMessage("Render your Munger verdict now.")]
        ),
    )
    return {"persona_signals": [result.model_dump()]}


PERSONA_SPEC = register_persona(
    PersonaSpec(
        slug="charlie_munger",
        display_name="Charlie Munger",
        investing_style=(
            "Quality + mental models; ROIC consistency; quality-weighted "
            "(0.35 moat / 0.25 mgmt / 0.25 predictability / 0.15 val)"
        ),
        node=charlie_munger_node,
        signal_schema=CharlieMungerSignal,
    )
)
