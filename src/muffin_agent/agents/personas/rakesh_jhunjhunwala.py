"""Rakesh Jhunjhunwala persona — EM growth + quality-tier DCF.

Five sub-scores (max 24): profitability (8), growth (7), balance sheet (4),
cash flow (3), management actions (2).  Quality-tier DCF uses discount
rates 12–18% based on overall quality (the 'Big Bull's signature).
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


class RakeshJhunjhunwalaEvidence(BaseModel):
    """Persona-specific evidence — sub-scores, computed metrics."""

    profitability: ScoreDetail
    growth: ScoreDetail
    balance_sheet: ScoreDetail
    cash_flow: ScoreDetail
    management_actions: ScoreDetail
    quality_tier: Literal["high", "medium", "low"]
    discount_rate: float
    intrinsic_value: float | None
    margin_of_safety: float | None
    market_cap: float | None
    total_score: float
    max_score: float


class RakeshJhunjhunwalaSignal(AnalystSignal):
    """Persona structured signal with narrowed evidence type."""

    agent_id: Literal["rakesh_jhunjhunwala"] = Field(default="rakesh_jhunjhunwala")
    evidence: RakeshJhunjhunwalaEvidence


def _score_profitability(
    latest_metrics: dict[str, Any], line_items: dict[str, list[float | None]]
) -> ScoreDetail:
    roe = latest_metrics.get("return_on_equity")
    op_margin = latest_metrics.get("operating_margin")
    eps = [v for v in line_items.get("earnings_per_share", []) if v is not None]
    score = 0
    parts: list[str] = []
    if roe is not None:
        if roe > 0.20:
            score += 3
            parts.append(f"ROE {roe:.1%}")
        elif roe > 0.15:
            score += 2
        elif roe > 0.10:
            score += 1
    if op_margin is not None:
        if op_margin > 0.20:
            score += 2
        elif op_margin > 0.15:
            score += 1
    if len(eps) >= 2 and eps[0] > 0 and eps[-1] > 0:
        cagr = (eps[-1] / eps[0]) ** (1 / (len(eps) - 1)) - 1
        if cagr > 0.20:
            score += 3
            parts.append(f"EPS CAGR {cagr:.1%}")
        elif cagr > 0.15:
            score += 2
        elif cagr > 0.10:
            score += 1
    return ScoreDetail(
        score=min(score, 8), max_score=8, details="; ".join(parts) or "Limited"
    )


def _score_growth(line_items: dict[str, list[float | None]]) -> ScoreDetail:
    revenues = [v for v in line_items.get("revenue", []) if v is not None]
    net_income = [v for v in line_items.get("net_income", []) if v is not None]
    score = 0
    parts: list[str] = []
    if len(revenues) >= 2 and revenues[0] > 0:
        cagr = (revenues[-1] / revenues[0]) ** (1 / (len(revenues) - 1)) - 1
        if cagr > 0.20:
            score += 3
            parts.append(f"Rev CAGR {cagr:.1%}")
        elif cagr > 0.15:
            score += 2
        elif cagr > 0.10:
            score += 1
    if len(net_income) >= 2 and net_income[0] > 0 and net_income[-1] > 0:
        cagr = (net_income[-1] / net_income[0]) ** (1 / (len(net_income) - 1)) - 1
        if cagr > 0.25:
            score += 3
            parts.append(f"NI CAGR {cagr:.1%}")
        elif cagr > 0.20:
            score += 2
        elif cagr > 0.15:
            score += 1
        if all(
            net_income[i] >= net_income[i - 1] * 0.8 for i in range(1, len(net_income))
        ):
            score += 1
            parts.append("Consistent NI")
    return ScoreDetail(
        score=min(score, 7), max_score=7, details="; ".join(parts) or "Limited"
    )


def _score_balance_sheet(line_items: dict[str, list[float | None]]) -> ScoreDetail:
    total_assets = [v for v in line_items.get("total_assets", []) if v is not None]
    total_liab = [v for v in line_items.get("total_liabilities", []) if v is not None]
    current_assets = [v for v in line_items.get("current_assets", []) if v is not None]
    current_liab = [
        v for v in line_items.get("current_liabilities", []) if v is not None
    ]
    score = 0
    parts: list[str] = []
    if total_assets and total_liab and total_assets[-1] > 0:
        de = total_liab[-1] / total_assets[-1]
        if de < 0.5:
            score += 2
            parts.append(f"D/Assets {de:.2f}")
        elif de < 0.7:
            score += 1
    if current_assets and current_liab and current_liab[-1] > 0:
        cr = current_assets[-1] / current_liab[-1]
        if cr > 2.0:
            score += 2
            parts.append(f"Current ratio {cr:.2f}")
        elif cr > 1.5:
            score += 1
    return ScoreDetail(
        score=min(score, 4), max_score=4, details="; ".join(parts) or "Limited"
    )


def _score_cash_flow(line_items: dict[str, list[float | None]]) -> ScoreDetail:
    fcf = [v for v in line_items.get("free_cash_flow", []) if v is not None]
    dividends = [
        v
        for v in line_items.get("dividends_and_other_cash_distributions", [])
        if v is not None
    ]
    score = 0
    parts: list[str] = []
    if fcf and fcf[-1] is not None and fcf[-1] > 0:
        score += 2
        parts.append("Positive FCF")
    if dividends and any(d < 0 for d in dividends):
        score += 1
        parts.append("Pays dividends")
    return ScoreDetail(
        score=min(score, 3), max_score=3, details="; ".join(parts) or "Limited"
    )


def _score_management_actions(line_items: dict[str, list[float | None]]) -> ScoreDetail:
    issuance = [
        v
        for v in line_items.get("issuance_or_purchase_of_equity_shares", [])
        if v is not None
    ]
    score = 0
    parts: list[str] = []
    if issuance and issuance[-1] is not None:
        if issuance[-1] < 0:
            score += 2
            parts.append("Buybacks")
        elif issuance[-1] == 0:
            score += 1
            parts.append("No dilution")
    return ScoreDetail(
        score=min(score, 2), max_score=2, details="; ".join(parts) or "Limited"
    )


def _quality_tier(
    profitability: float, growth: float, balance: float
) -> Literal["high", "medium", "low"]:
    if profitability >= 6 and balance >= 3:
        return "high"
    if profitability >= 4:
        return "medium"
    return "low"


def _compute_jhunjhunwala_facts(
    data_bundle: dict[str, Any],
) -> RakeshJhunjhunwalaEvidence:
    line_items = data_bundle.get("line_items", {})
    metrics = data_bundle.get("financial_metrics", [])
    latest = metrics[0] if metrics else {}
    market_cap = data_bundle.get("market_cap")
    net_income_series = [v for v in line_items.get("net_income", []) if v is not None]
    net_income_latest = net_income_series[-1] if net_income_series else None

    profitability = _score_profitability(latest, line_items)
    growth = _score_growth(line_items)
    balance = _score_balance_sheet(line_items)
    cash_flow = _score_cash_flow(line_items)
    mgmt = _score_management_actions(line_items)
    tier = _quality_tier(profitability.score, growth.score, balance.score)
    discount_rates: dict[str, float] = {"high": 0.12, "medium": 0.15, "low": 0.18}
    discount = discount_rates[tier]
    intrinsic = None
    if net_income_latest and net_income_latest > 0:
        intrinsic = compute_intrinsic_value_dcf(
            base_cash_flow=net_income_latest,
            growth_rate=0.12,
            discount_rate=discount,
            terminal_growth_rate=0.04,
            years=5,
        )
    mos = (
        (intrinsic - market_cap) / market_cap
        if intrinsic is not None and market_cap and market_cap > 0
        else None
    )
    total = (
        profitability.score
        + growth.score
        + balance.score
        + cash_flow.score
        + mgmt.score
    )
    max_total = 8 + 7 + 4 + 3 + 2
    return RakeshJhunjhunwalaEvidence(
        profitability=profitability,
        growth=growth,
        balance_sheet=balance,
        cash_flow=cash_flow,
        management_actions=mgmt,
        quality_tier=tier,
        discount_rate=discount,
        intrinsic_value=intrinsic,
        margin_of_safety=mos,
        market_cap=market_cap,
        total_score=total,
        max_score=max_total,
    )


async def rakesh_jhunjhunwala_node(
    state: PersonaInputState, config: RunnableConfig
) -> PersonaOutputState:
    """Render the Rakesh Jhunjhunwala verdict from state["data_bundle"]."""
    ticker = state.get("ticker", "")
    query = state.get("query")
    data_bundle = state.get("data_bundle") or {}

    if not data_bundle or "error" in data_bundle:
        fallback = RakeshJhunjhunwalaSignal(
            agent_id="rakesh_jhunjhunwala",
            signal="hold",
            confidence=0.0,
            reasoning="Insufficient data",
            evidence=RakeshJhunjhunwalaEvidence(
                profitability=ScoreDetail(score=0, max_score=8, details="no data"),
                growth=ScoreDetail(score=0, max_score=7, details="no data"),
                balance_sheet=ScoreDetail(score=0, max_score=4, details="no data"),
                cash_flow=ScoreDetail(score=0, max_score=3, details="no data"),
                management_actions=ScoreDetail(score=0, max_score=2, details="no data"),
                quality_tier="low",
                discount_rate=0.18,
                intrinsic_value=None,
                margin_of_safety=None,
                market_cap=None,
                total_score=0,
                max_score=24,
            ),
        )
        return {"persona_signals": [fallback.model_dump()]}

    evidence = _compute_jhunjhunwala_facts(data_bundle)
    llm = ModelConfiguration.get_chat_model_for_role(
        config, "reasoner", schema=RakeshJhunjhunwalaSignal
    )
    prompt = render_template(
        "personas/rakesh_jhunjhunwala.jinja",
        ticker=ticker,
        as_of_date=data_bundle.get("as_of_date", ""),
        facts=evidence.model_dump(mode="json"),
        query=query,
        persona_display_name="Rakesh Jhunjhunwala",
        persona_slug="rakesh_jhunjhunwala",
        signal_schema_name="RakeshJhunjhunwalaSignal",
    )
    result = cast(
        RakeshJhunjhunwalaSignal,
        await llm.ainvoke(
            [SystemMessage(prompt), HumanMessage("Render your Jhunjhunwala verdict.")]
        ),
    )
    return {"persona_signals": [result.model_dump()]}


PERSONA_SPEC = register_persona(
    PersonaSpec(
        slug="rakesh_jhunjhunwala",
        display_name="Rakesh Jhunjhunwala",
        investing_style=(
            "EM growth + quality-tier DCF (12/15/18% discount based on quality); "
            "30% MoS target; buybacks signal management conviction"
        ),
        node=rakesh_jhunjhunwala_node,
        signal_schema=RakeshJhunjhunwalaSignal,
    )
)
