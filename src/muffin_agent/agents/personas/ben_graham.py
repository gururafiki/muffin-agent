"""Ben Graham persona — deep value via NCAV and Graham Number.

Single-LLM-call persona faithful to ai-hedge-fund's ``ben_graham.py``.
Three sub-scores: earnings stability (max 4), financial strength (max 5),
valuation via NCAV + Graham Number (max 6).  Total max = 15.

Reference (upstream): ``ai-hedge-fund/src/agents/ben_graham.py``.
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
    compute_graham_number,
    compute_ncav_per_share,
)
from ._base import (
    PersonaInputState,
    PersonaOutputState,
    PersonaSpec,
    register_persona,
)
from .schemas import AnalystSignal, ScoreDetail

logger = logging.getLogger(__name__)


# ── Evidence + signal ─────────────────────────────────────────────────────────


class BenGrahamEvidence(BaseModel):
    """Graham-specific evidence (earnings stability, strength, valuation)."""

    earnings_stability: ScoreDetail
    financial_strength: ScoreDetail
    valuation: ScoreDetail

    ncav_per_share: float | None
    """Net Current Asset Value per share = (CA − total liabilities) / shares."""
    graham_number: float | None
    """sqrt(22.5 × EPS × BVPS); the classic Graham intrinsic estimate."""
    current_price: float | None
    """market_cap / outstanding_shares, the per-share market price used in MoS."""
    margin_of_safety_graham: float | None
    """(graham_number − current_price) / current_price, decimal."""
    is_net_net: bool
    """True when NCAV > market cap — classic Graham net-net."""

    total_score: float
    max_score: float


class BenGrahamSignal(AnalystSignal):
    """Narrowed signal with typed Graham evidence."""

    agent_id: Literal["ben_graham"] = Field(default="ben_graham")
    evidence: BenGrahamEvidence


# ── Sub-scoring helpers ───────────────────────────────────────────────────────


def _score_earnings_stability(eps_series: list[float | None]) -> ScoreDetail:
    """Score Graham's earnings stability (max 4).

    Expects EPS in **oldest → newest** order (the data_collection
    contract).  Components: ``all positive`` (+3), ``80%+ positive`` (+2),
    ``EPS grew oldest→latest`` (+1).
    """
    eps = [v for v in eps_series if v is not None]
    if len(eps) < 2:
        return ScoreDetail(score=0, max_score=4, details="Insufficient EPS history")
    positives = sum(1 for e in eps if e > 0)
    score = 0
    parts: list[str] = []
    if positives == len(eps):
        score += 3
        parts.append("EPS positive in every period")
    elif positives >= int(len(eps) * 0.8):
        score += 2
        parts.append(f"EPS positive in {positives}/{len(eps)} periods")
    else:
        parts.append(f"EPS negative in {len(eps) - positives}/{len(eps)} periods")
    if eps[-1] > eps[0]:
        score += 1
        parts.append("EPS grew from earliest to latest period")
    else:
        parts.append("EPS did not grow over the window")
    return ScoreDetail(
        score=score,
        max_score=4,
        details="; ".join(parts),
        metrics={"latest_eps": eps[-1], "oldest_eps": eps[0]},
    )


def _score_financial_strength(
    current_assets: float | None,
    current_liabilities: float | None,
    total_assets: float | None,
    total_liabilities: float | None,
    dividends_series: list[float | None],
) -> ScoreDetail:
    """Score Graham's financial-strength dimension (max 5).

    Current ratio ≥ 2 → +2, ≥ 1.5 → +1.  Debt ratio < 0.5 → +2, < 0.8 → +1.
    Dividend record: majority of years had outflows → +1.
    """
    score = 0
    parts: list[str] = []

    if current_assets is not None and current_liabilities and current_liabilities > 0:
        cr = current_assets / current_liabilities
        if cr >= 2.0:
            score += 2
            parts.append(f"Current ratio {cr:.2f} (≥2 strong)")
        elif cr >= 1.5:
            score += 1
            parts.append(f"Current ratio {cr:.2f} (moderate)")
        else:
            parts.append(f"Current ratio {cr:.2f} (weak)")
    else:
        parts.append("Current ratio unavailable")

    if total_assets and total_liabilities is not None and total_assets > 0:
        de = total_liabilities / total_assets
        if de < 0.5:
            score += 2
            parts.append(f"Debt ratio {de:.2f} (conservative)")
        elif de < 0.8:
            score += 1
            parts.append(f"Debt ratio {de:.2f} (acceptable)")
        else:
            parts.append(f"Debt ratio {de:.2f} (high)")
    else:
        parts.append("Debt ratio unavailable")

    divs = [d for d in dividends_series if d is not None]
    if divs:
        paid_years = sum(1 for d in divs if d < 0)
        if paid_years >= len(divs) // 2 + 1:
            score += 1
            parts.append(f"Dividends paid in {paid_years}/{len(divs)} years (majority)")
        elif paid_years > 0:
            parts.append(f"Dividends paid in {paid_years}/{len(divs)} years (minority)")
        else:
            parts.append("No dividends paid")
    else:
        parts.append("No dividend data")

    return ScoreDetail(score=score, max_score=5, details="; ".join(parts))


def _score_valuation(
    current_assets: float | None,
    total_liabilities: float | None,
    outstanding_shares_latest: float | None,
    eps_latest: float | None,
    bvps_latest: float | None,
    market_cap: float | None,
) -> tuple[ScoreDetail, dict[str, Any]]:
    """Graham valuation: NCAV check + Graham Number MoS (max 6).

    Returns ``(ScoreDetail, extra_facts)`` where ``extra_facts`` carries
    the computed NCAV / Graham number / margin of safety for the persona
    evidence model.
    """
    score = 0
    parts: list[str] = []
    extras: dict[str, Any] = {
        "ncav_per_share": None,
        "graham_number": None,
        "current_price": None,
        "margin_of_safety_graham": None,
        "is_net_net": False,
    }

    if (
        outstanding_shares_latest is None
        or outstanding_shares_latest <= 0
        or market_cap is None
        or market_cap <= 0
    ):
        parts.append("Cannot compute Graham valuation (missing market data)")
        return ScoreDetail(score=0, max_score=6, details="; ".join(parts)), extras

    current_price = market_cap / outstanding_shares_latest
    extras["current_price"] = current_price

    ncav_per_share = compute_ncav_per_share(
        current_assets, total_liabilities, outstanding_shares_latest
    )
    extras["ncav_per_share"] = ncav_per_share

    if ncav_per_share is not None:
        ncav_total = ncav_per_share * outstanding_shares_latest
        if ncav_total > market_cap:
            score += 4
            extras["is_net_net"] = True
            parts.append("NCAV > market cap — classic Graham net-net")
        elif ncav_per_share >= current_price * 0.67:
            score += 2
            parts.append("NCAV ≥ 67% of price — moderate net-net discount")

    graham_number = compute_graham_number(eps_latest, bvps_latest)
    extras["graham_number"] = graham_number
    if graham_number is not None and current_price > 0:
        mos = (graham_number - current_price) / current_price
        extras["margin_of_safety_graham"] = mos
        if mos > 0.5:
            score += 3
            parts.append(f"Graham Number margin of safety {mos:.1%} (≥50%)")
        elif mos > 0.2:
            score += 1
            parts.append(f"Graham Number margin of safety {mos:.1%}")
        else:
            parts.append(f"Graham Number margin of safety {mos:.1%} (low)")
    else:
        parts.append("Graham Number unavailable (need positive EPS and BVPS)")

    return ScoreDetail(score=score, max_score=6, details="; ".join(parts)), extras


# ── Fact computer ─────────────────────────────────────────────────────────────


def _compute_graham_facts(data_bundle: dict[str, Any]) -> BenGrahamEvidence:
    """Compute Graham evidence from a PersonaDataBundle dict."""
    line_items: dict[str, list[float | None]] = data_bundle.get("line_items", {})
    market_cap: float | None = data_bundle.get("market_cap")

    eps_series = line_items.get("earnings_per_share", [])
    dividends_series = line_items.get("dividends_and_other_cash_distributions", [])

    # Latest values come from the chronologically-last entry per
    # data_collection contract (oldest → newest).
    def _last(name: str) -> float | None:
        series = line_items.get(name, [])
        return series[-1] if series else None

    earnings_stability = _score_earnings_stability(eps_series)
    financial_strength = _score_financial_strength(
        _last("current_assets"),
        _last("current_liabilities"),
        _last("total_assets"),
        _last("total_liabilities"),
        dividends_series,
    )
    valuation, extras = _score_valuation(
        _last("current_assets"),
        _last("total_liabilities"),
        _last("outstanding_shares"),
        _last("earnings_per_share"),
        _last("book_value_per_share"),
        market_cap,
    )

    total = earnings_stability.score + financial_strength.score + valuation.score
    max_total = (
        earnings_stability.max_score
        + financial_strength.max_score
        + valuation.max_score
    )
    return BenGrahamEvidence(
        earnings_stability=earnings_stability,
        financial_strength=financial_strength,
        valuation=valuation,
        ncav_per_share=extras["ncav_per_share"],
        graham_number=extras["graham_number"],
        current_price=extras["current_price"],
        margin_of_safety_graham=extras["margin_of_safety_graham"],
        is_net_net=extras["is_net_net"],
        total_score=total,
        max_score=max_total,
    )


# ── Node ──────────────────────────────────────────────────────────────────────


async def ben_graham_node(
    state: PersonaInputState, config: RunnableConfig
) -> PersonaOutputState:
    """Render the Ben Graham verdict from state["data_bundle"]."""
    ticker = state.get("ticker", "")
    query = state.get("query")
    data_bundle = state.get("data_bundle") or {}

    if not data_bundle or "error" in data_bundle:
        logger.warning("ben_graham_node: data_bundle missing or errored")
        fallback = BenGrahamSignal(
            agent_id="ben_graham",
            signal="hold",
            confidence=0.0,
            reasoning="Insufficient data — defaulting to hold",
            evidence=BenGrahamEvidence(
                earnings_stability=ScoreDetail(score=0, max_score=4, details="no data"),
                financial_strength=ScoreDetail(score=0, max_score=5, details="no data"),
                valuation=ScoreDetail(score=0, max_score=6, details="no data"),
                ncav_per_share=None,
                graham_number=None,
                current_price=None,
                margin_of_safety_graham=None,
                is_net_net=False,
                total_score=0,
                max_score=15,
            ),
        )
        return {"persona_signals": [fallback.model_dump()]}

    evidence = _compute_graham_facts(data_bundle)
    llm = ModelConfiguration.get_chat_model_for_role(
        config, "reasoner", schema=BenGrahamSignal
    )
    prompt = render_template(
        "personas/ben_graham.jinja",
        ticker=ticker,
        as_of_date=data_bundle.get("as_of_date", ""),
        facts=evidence.model_dump(mode="json"),
        query=query,
        persona_display_name="Benjamin Graham",
        persona_slug="ben_graham",
        signal_schema_name="BenGrahamSignal",
    )
    result = cast(
        BenGrahamSignal,
        await llm.ainvoke(
            [
                SystemMessage(prompt),
                HumanMessage("Render your Graham verdict now."),
            ]
        ),
    )
    return {"persona_signals": [result.model_dump()]}


PERSONA_SPEC = register_persona(
    PersonaSpec(
        slug="ben_graham",
        display_name="Benjamin Graham",
        investing_style=(
            "Deep value, net-net hunting, Graham Number ≥50% margin of "
            "safety, financial strength (current ratio ≥2, debt <50% of assets)"
        ),
        node=ben_graham_node,
        signal_schema=BenGrahamSignal,
    )
)
