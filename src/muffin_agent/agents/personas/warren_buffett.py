"""Warren Buffett persona — compiled subgraph (collect → compute → verdict).

Three-node :class:`StateGraph` subgraph implementing the v4 persona pattern:

1. ``collect_data`` — compiled ReAct sub-agent built via
   :class:`MuffinAgentBuilder`.  Curated 6 OpenBB MCP tools fetch
   fundamentals + market cap; the LLM extracts typed fields into a
   :class:`WarrenBuffettRawData` structured response.  The
   ``_StructuredResponseToStateMiddleware`` auto-unpacks RawData fields
   into state.

2. ``compute_evidence`` — deterministic Python.  Calls the six composite
   scorers (``_score_buffett_fundamentals`` / ``_consistency`` / ``_moat``
   / ``_pricing_power`` / ``_book_value_growth`` / ``_management``) and
   composes a typed :class:`WarrenBuffettEvidence`.

3. ``render_verdict`` — single LLM call with ``schema=WarrenBuffettSignal``.

A legacy bridge ``warren_buffett_node`` (and ``_compute_buffett_facts``)
is preserved so the existing ``PERSONA_REGISTRY`` / council graph / CLI
keep working until Phase 5 of the refactor rewrites them to call
``build_warren_buffett_agent`` directly.

Reference (upstream): ``ai-hedge-fund/src/agents/warren_buffett.py``.
"""

from __future__ import annotations

import logging
import math
from typing import Annotated, Any, Literal, cast

from langchain.agents import AgentState
from langchain.agents.middleware.types import OmitFromSchema
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from ...model_config import ModelConfiguration
from ...prompts import render_template
from ...tools.scoring_helpers import (
    compute_buffett_3stage_dcf,
    compute_owner_earnings,
    score_current_ratio,
    score_debt_to_equity,
    score_operating_margin,
    score_roe,
)
from ...utils.agent_builder import MuffinAgentBuilder
from ..data_collection.utils import get_tools
from .schemas import AnalystSignal

logger = logging.getLogger(__name__)

_LLM_RETRY = RetryPolicy(max_attempts=2)


# ── Typed sub-evidence Pydantics (replace generic ScoreDetail) ────────────────


class WarrenBuffettFundamentals(BaseModel):
    """Latest-period quality snapshot."""

    roe_score: int
    roe_value: float | None
    debt_to_equity_score: int
    debt_to_equity_value: float | None
    operating_margin_score: int
    operating_margin_value: float | None
    current_ratio_score: int
    current_ratio_value: float | None
    total_score: int
    max_score: int
    reasoning: str


class WarrenBuffettConsistency(BaseModel):
    """Multi-year earnings trajectory."""

    score: int
    max_score: int
    periods_analysed: int
    earnings_growth_pct: float | None
    strictly_growing: bool
    reasoning: str


class WarrenBuffettMoat(BaseModel):
    """Durable competitive advantage indicators."""

    roe_consistency_pct: float | None
    roe_high_count: int
    roe_periods: int
    avg_operating_margin: float | None
    operating_margins_improving: bool
    asset_turnover_above_one: bool
    performance_stability: float | None
    score: int
    max_score: int
    reasoning: str


class WarrenBuffettPricingPower(BaseModel):
    """Gross margin trend + level."""

    avg_gross_margin: float | None
    recent_avg_gross_margin: float | None
    older_avg_gross_margin: float | None
    margin_direction: Literal["expanding", "improving", "stable", "declining", "n/a"]
    score: int
    max_score: int
    reasoning: str


class WarrenBuffettBookValueGrowth(BaseModel):
    """BVPS trajectory + CAGR."""

    bvps_latest: float | None
    bvps_oldest: float | None
    bvps_cagr_pct: float | None
    growing_period_ratio: float | None
    score: int
    max_score: int
    reasoning: str


class WarrenBuffettManagement(BaseModel):
    """Buybacks + dividends heuristics from the latest cash-flow statement."""

    net_buybacks_latest: float | None
    dividends_latest: float | None
    score: int
    max_score: int
    reasoning: str


class WarrenBuffettEvidence(BaseModel):
    """Buffett-specific precomputed evidence (v4 typed shape).

    Each sub-aspect uses its own narrow Pydantic model — no generic
    ``ScoreDetail``.  Composite scorers compose into this evidence
    deterministically; the verdict LLM receives it via the Jinja template
    with granular attribute access.
    """

    fundamentals: WarrenBuffettFundamentals
    consistency: WarrenBuffettConsistency
    moat: WarrenBuffettMoat
    pricing_power: WarrenBuffettPricingPower
    book_value_growth: WarrenBuffettBookValueGrowth
    management_quality: WarrenBuffettManagement

    owner_earnings: float | None = Field(
        default=None,
        description=(
            "Buffett's preferred earnings measure: NI + D&A - maintenance capex."
        ),
    )
    intrinsic_value: float | None = Field(
        default=None,
        description="3-stage DCF on owner earnings with 15% conservatism haircut.",
    )
    market_cap: float | None = Field(
        default=None, description="Latest market capitalisation."
    )
    margin_of_safety_pct: float | None = Field(
        default=None,
        description=(
            "(intrinsic_value - market_cap) / market_cap, in percent (decimal x 100)."
        ),
    )

    total_score: float
    max_score: float


class WarrenBuffettSignal(AnalystSignal):
    """Narrowed signal type with a typed ``WarrenBuffettEvidence``."""

    agent_id: Literal["warren_buffett"] = Field(default="warren_buffett")
    evidence: WarrenBuffettEvidence


# ── RawData (output of the data_react sub-agent) ──────────────────────────────


class BuffettMetricsRow(BaseModel):
    """One annual snapshot of Buffett-relevant financial metrics.

    Field names match OpenBB ``equity_fundamental_metrics`` for easy
    LLM-driven extraction.
    """

    return_on_equity: float | None = Field(
        default=None, description="Return on equity (decimal, e.g. 0.18 for 18%)."
    )
    debt_to_equity: float | None = Field(
        default=None, description="Total debt / shareholders' equity."
    )
    operating_margin: float | None = Field(
        default=None, description="Operating margin (decimal)."
    )
    current_ratio: float | None = Field(
        default=None, description="Current assets / current liabilities."
    )
    asset_turnover: float | None = Field(
        default=None, description="Revenue / total assets."
    )


class WarrenBuffettRawData(BaseModel):
    """Structured MCP extraction — the data_react sub-agent's response_format.

    The collect_data ReAct loop populates these fields by calling the MCP tools
    enumerated in ``_MCP_TOOLS`` below.  Field descriptions teach the LLM
    which endpoint to query and which field to extract.

    All time series are **oldest → newest** (5 years recommended).
    """

    metrics_history: list[BuffettMetricsRow] = Field(
        default_factory=list,
        description=(
            "Up to 5 annual snapshots from equity_fundamental_metrics, "
            "most-recent FIRST.  Each row carries ROE, D/E, operating margin, "
            "current ratio, asset turnover."
        ),
    )
    net_income_series: list[float | None] = Field(
        default_factory=list,
        description=(
            "Annual net income, oldest → newest, from equity_fundamental_income."
        ),
    )
    revenue_series: list[float | None] = Field(
        default_factory=list,
        description="Annual revenue, oldest → newest, from equity_fundamental_income.",
    )
    gross_margin_series: list[float | None] = Field(
        default_factory=list,
        description=(
            "Annual gross margin (decimal, e.g. 0.42 for 42%), oldest → newest. "
            "Extract from equity_fundamental_ratios, OR compute as "
            "gross_profit / revenue from equity_fundamental_income."
        ),
    )
    shareholders_equity_series: list[float | None] = Field(
        default_factory=list,
        description=(
            "Annual shareholders' equity, oldest -> newest, "
            "from equity_fundamental_balance."
        ),
    )
    outstanding_shares_series: list[float | None] = Field(
        default_factory=list,
        description=(
            "Annual diluted shares outstanding, oldest → newest, from "
            "equity_fundamental_balance OR equity_fundamental_income."
        ),
    )
    depreciation_amortization_series: list[float | None] = Field(
        default_factory=list,
        description="Annual D&A, oldest → newest, from equity_fundamental_cash.",
    )
    capital_expenditure_series: list[float | None] = Field(
        default_factory=list,
        description=(
            "Annual capex (always reported as a POSITIVE number), oldest → newest, "
            "from equity_fundamental_cash. Take abs() of negative cash-outflow."
        ),
    )
    latest_issuance_or_purchase_of_equity_shares: float | None = Field(
        default=None,
        description=(
            "Latest annual net stock issuance from equity_fundamental_cash "
            "(net of repurchases — NEGATIVE means net buybacks)."
        ),
    )
    latest_dividends_and_other_cash_distributions: float | None = Field(
        default=None,
        description=(
            "Latest annual dividend payments (NEGATIVE on cash-flow statement "
            "= cash outflow to shareholders) from equity_fundamental_cash."
        ),
    )
    market_cap: float | None = Field(
        default=None,
        description=(
            "Latest market capitalisation (USD), from equity_historical_market_cap."
        ),
    )


# ── State schema ──────────────────────────────────────────────────────────────


class WarrenBuffettInput(TypedDict, total=False):
    """Public input contract — what the council provides to this persona.

    Used as the ``input_schema`` of both the inner data_react sub-agent
    (forwarded via ``OmitFromSchema``) and the outer subgraph (passed
    explicitly to ``StateGraph(..., input_schema=...)``).
    """

    ticker: str
    as_of_date: str
    query: str | None


class WarrenBuffettOutput(TypedDict, total=False):
    """Public output contract for the Warren Buffett persona subgraph.

    Single-element ``persona_signals`` list that the council's
    ``operator.add`` reducer accumulates.
    """

    persona_signals: list[dict[str, Any]]


class WarrenBuffettState(AgentState):
    """Internal state schema for the Warren Buffett persona subgraph.

    Inherits ``messages`` + ``structured_response`` from ``AgentState`` (used
    by the data_react sub-agent's ReAct loop).  RawData fields land here via
    ``_StructuredResponseToStateMiddleware`` auto-unpack.  The outer
    ``StateGraph`` uses :class:`WarrenBuffettInput` / :class:`WarrenBuffettOutput`
    as its public boundary contracts — those filter what the parent council
    sees regardless of the internal state shape.
    """

    # Inputs from the council
    ticker: Annotated[str, OmitFromSchema(input=False, output=True)]
    as_of_date: Annotated[str, OmitFromSchema(input=False, output=True)]
    query: Annotated[str | None, OmitFromSchema(input=False, output=True)]

    # RawData fields (populated by collect_data's structured_response auto-unpack)
    metrics_history: Annotated[
        list[BuffettMetricsRow] | None, OmitFromSchema(input=True, output=True)
    ]
    net_income_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    revenue_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    gross_margin_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    shareholders_equity_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    outstanding_shares_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    depreciation_amortization_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    capital_expenditure_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    latest_issuance_or_purchase_of_equity_shares: Annotated[
        float | None, OmitFromSchema(input=True, output=True)
    ]
    latest_dividends_and_other_cash_distributions: Annotated[
        float | None, OmitFromSchema(input=True, output=True)
    ]
    market_cap: Annotated[float | None, OmitFromSchema(input=True, output=True)]

    # Computed evidence (filled by compute_evidence)
    evidence: Annotated[
        WarrenBuffettEvidence | None, OmitFromSchema(input=True, output=True)
    ]

    # Output to the council (single-element list, accumulated via parent reducer)
    persona_signals: Annotated[list[dict], OmitFromSchema(input=True, output=False)]


# ── Composite scorers (pure Python, take typed inputs) ────────────────────────


def _score_buffett_fundamentals(
    latest: BuffettMetricsRow | None,
) -> WarrenBuffettFundamentals:
    """Score the latest quality snapshot (ROE / D/E / op margin / current ratio).

    Mirrors upstream's ``analyze_fundamentals`` exactly: max = 10 (3+3+2+2).
    """
    if latest is None:
        return WarrenBuffettFundamentals(
            roe_score=0,
            roe_value=None,
            debt_to_equity_score=0,
            debt_to_equity_value=None,
            operating_margin_score=0,
            operating_margin_value=None,
            current_ratio_score=0,
            current_ratio_value=None,
            total_score=0,
            max_score=10,
            reasoning="No metrics available.",
        )
    roe = score_roe(latest.return_on_equity)
    de = score_debt_to_equity(latest.debt_to_equity)
    om = score_operating_margin(latest.operating_margin)
    cr = score_current_ratio(latest.current_ratio)
    return WarrenBuffettFundamentals(
        roe_score=int(roe.score),
        roe_value=latest.return_on_equity,
        debt_to_equity_score=int(de.score),
        debt_to_equity_value=latest.debt_to_equity,
        operating_margin_score=int(om.score),
        operating_margin_value=latest.operating_margin,
        current_ratio_score=int(cr.score),
        current_ratio_value=latest.current_ratio,
        total_score=int(roe.score + de.score + om.score + cr.score),
        max_score=int(roe.max_score + de.max_score + om.max_score + cr.max_score),
        reasoning="; ".join([roe.details, de.details, om.details, cr.details]),
    )


def _score_buffett_consistency(
    net_income_series: list[float | None] | None,
) -> WarrenBuffettConsistency:
    """Strictly-increasing net income across ≥4 periods earns +3.

    Series in **chronological order** (oldest → newest).  Max = 3.
    """
    if not net_income_series:
        return WarrenBuffettConsistency(
            score=0,
            max_score=3,
            periods_analysed=0,
            earnings_growth_pct=None,
            strictly_growing=False,
            reasoning="No earnings history.",
        )
    series = [v for v in net_income_series if v is not None]
    if len(series) < 4:
        return WarrenBuffettConsistency(
            score=0,
            max_score=3,
            periods_analysed=len(series),
            earnings_growth_pct=None,
            strictly_growing=False,
            reasoning=(
                f"Insufficient earnings history (need 4+ periods, got {len(series)})."
            ),
        )
    strictly_growing = all(series[i + 1] > series[i] for i in range(len(series) - 1))
    if strictly_growing:
        growth_pct = (
            (series[-1] - series[0]) / abs(series[0]) * 100 if series[0] != 0 else 0.0
        )
        return WarrenBuffettConsistency(
            score=3,
            max_score=3,
            periods_analysed=len(series),
            earnings_growth_pct=growth_pct,
            strictly_growing=True,
            reasoning=(
                f"Consistent earnings growth over {len(series)} periods "
                f"({growth_pct:.1f}% total)."
            ),
        )
    return WarrenBuffettConsistency(
        score=0,
        max_score=3,
        periods_analysed=len(series),
        earnings_growth_pct=None,
        strictly_growing=False,
        reasoning=f"Inconsistent earnings growth across {len(series)} periods.",
    )


def _score_buffett_moat(
    metrics_history: list[BuffettMetricsRow] | None,
) -> WarrenBuffettMoat:
    """Score moat dimensions across the metrics history (newest first).

    Components (upstream max = 5):

    * ROE consistency: 80%+ periods with ROE > 15% → +2; 60%+ → +1
    * Margin level + improvement: avg op margin > 20% AND improving → +1
    * Asset turnover > 1.0 in any period → +1
    * Performance stability (CV of ROE + margin) > 70% → +1
    """
    if not metrics_history or len(metrics_history) < 5:
        return WarrenBuffettMoat(
            roe_consistency_pct=None,
            roe_high_count=0,
            roe_periods=0,
            avg_operating_margin=None,
            operating_margins_improving=False,
            asset_turnover_above_one=False,
            performance_stability=None,
            score=0,
            max_score=5,
            reasoning="Insufficient history for moat analysis (need 5+ periods).",
        )

    score = 0
    reasoning: list[str] = []

    roes = [
        m.return_on_equity for m in metrics_history if m.return_on_equity is not None
    ]
    margins = [
        m.operating_margin for m in metrics_history if m.operating_margin is not None
    ]
    turnovers = [
        m.asset_turnover for m in metrics_history if m.asset_turnover is not None
    ]

    roe_consistency_pct: float | None = None
    roe_high_count = 0
    if len(roes) >= 5:
        roe_high_count = sum(1 for r in roes if r > 0.15)
        roe_consistency_pct = roe_high_count / len(roes) * 100
        if roe_high_count / len(roes) >= 0.8:
            score += 2
            reasoning.append(
                f"Excellent ROE consistency ({roe_high_count}/{len(roes)} >15%)"
            )
        elif roe_high_count / len(roes) >= 0.6:
            score += 1
            reasoning.append(
                f"Good ROE consistency ({roe_high_count}/{len(roes)} >15%)"
            )
        else:
            reasoning.append(f"Inconsistent ROE ({roe_high_count}/{len(roes)} >15%)")

    avg_margin: float | None = None
    margins_improving = False
    if len(margins) >= 5:
        half = max(1, len(margins) // 2)
        recent = margins[:half]
        older = margins[-half:]
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        avg_margin = sum(margins) / len(margins)
        if avg_margin > 0.20 and recent_avg >= older_avg:
            score += 1
            margins_improving = True
            reasoning.append(
                f"Stable / improving operating margins ({avg_margin:.1%} avg)"
            )

    turnover_above_one = bool(turnovers and any(t > 1.0 for t in turnovers))
    if turnover_above_one:
        score += 1
        reasoning.append("Efficient asset utilisation (turnover > 1.0)")

    performance_stability: float | None = None
    if len(roes) >= 5 and len(margins) >= 5:
        roe_mean = sum(roes) / len(roes)
        roe_var = sum((r - roe_mean) ** 2 for r in roes) / len(roes)
        margin_mean = sum(margins) / len(margins)
        margin_var = sum((m - margin_mean) ** 2 for m in margins) / len(margins)
        roe_stab = 1 - (math.sqrt(roe_var) / roe_mean) if roe_mean > 0 else 0.0
        margin_stab = (
            1 - (math.sqrt(margin_var) / margin_mean) if margin_mean > 0 else 0.0
        )
        performance_stability = (roe_stab + margin_stab) / 2
        if performance_stability > 0.7:
            score += 1
            reasoning.append(
                f"High performance stability "
                f"({performance_stability:.1%}) - strong moat"
            )

    return WarrenBuffettMoat(
        roe_consistency_pct=roe_consistency_pct,
        roe_high_count=roe_high_count,
        roe_periods=len(roes),
        avg_operating_margin=avg_margin,
        operating_margins_improving=margins_improving,
        asset_turnover_above_one=turnover_above_one,
        performance_stability=performance_stability,
        score=min(score, 5),
        max_score=5,
        reasoning="; ".join(reasoning) if reasoning else "Limited moat evidence.",
    )


def _score_buffett_pricing_power(
    gross_margin_series: list[float | None] | None,
) -> WarrenBuffettPricingPower:
    """Score gross-margin trend + level.

    Accepts an **oldest → newest** series (matches RawData convention);
    reverses internally so "recent" = newest half.  Max = 5.
    """
    if not gross_margin_series:
        return WarrenBuffettPricingPower(
            avg_gross_margin=None,
            recent_avg_gross_margin=None,
            older_avg_gross_margin=None,
            margin_direction="n/a",
            score=0,
            max_score=5,
            reasoning="No gross-margin history available.",
        )
    margins_newest_first = list(reversed(gross_margin_series))
    margins = [m for m in margins_newest_first if m is not None]
    if len(margins) < 3:
        return WarrenBuffettPricingPower(
            avg_gross_margin=None,
            recent_avg_gross_margin=None,
            older_avg_gross_margin=None,
            margin_direction="n/a",
            score=0,
            max_score=5,
            reasoning=f"Insufficient gross-margin history ({len(margins)} periods).",
        )

    score = 0
    reasoning: list[str] = []

    half = max(2, len(margins) // 2)
    recent = margins[:half]
    older = margins[-half:]
    recent_avg = sum(recent) / len(recent)
    older_avg = sum(older) / len(older)

    direction: Literal["expanding", "improving", "stable", "declining", "n/a"]
    if recent_avg > older_avg + 0.02:
        direction = "expanding"
        score += 3
        reasoning.append(
            f"Expanding margins ({older_avg:.1%} → {recent_avg:.1%}) — pricing power"
        )
    elif recent_avg > older_avg:
        direction = "improving"
        score += 2
        reasoning.append(
            f"Improving gross margins ({older_avg:.1%} → {recent_avg:.1%})"
        )
    elif abs(recent_avg - older_avg) < 0.01:
        direction = "stable"
        score += 1
        reasoning.append(f"Stable gross margins around {recent_avg:.1%}")
    else:
        direction = "declining"
        reasoning.append(
            f"Declining gross margins ({older_avg:.1%} → {recent_avg:.1%})"
        )

    avg_margin = sum(margins) / len(margins)
    if avg_margin > 0.5:
        score += 2
        reasoning.append(f"Consistently high gross margins ({avg_margin:.1%})")
    elif avg_margin > 0.3:
        score += 1
        reasoning.append(f"Decent gross margins ({avg_margin:.1%})")

    return WarrenBuffettPricingPower(
        avg_gross_margin=avg_margin,
        recent_avg_gross_margin=recent_avg,
        older_avg_gross_margin=older_avg,
        margin_direction=direction,
        score=min(score, 5),
        max_score=5,
        reasoning="; ".join(reasoning),
    )


def _score_buffett_book_value_growth(
    shareholders_equity_series: list[float | None] | None,
    outstanding_shares_series: list[float | None] | None,
) -> WarrenBuffettBookValueGrowth:
    """Score book-value-per-share growth across the available history.

    Series **oldest → newest** (RawData convention).  Reverses internally
    so [0] is newest BVPS.  Max = 5.
    """
    if not shareholders_equity_series or not outstanding_shares_series:
        return WarrenBuffettBookValueGrowth(
            bvps_latest=None,
            bvps_oldest=None,
            bvps_cagr_pct=None,
            growing_period_ratio=None,
            score=0,
            max_score=5,
            reasoning="No equity / shares history available.",
        )

    # Pair oldest → newest, then build BVPS list in same order
    pairs = list(
        zip(shareholders_equity_series, outstanding_shares_series, strict=False)
    )
    bvps_chronological: list[float] = []
    for equity, shares in pairs:
        if equity is None or shares is None or shares <= 0:
            continue
        bvps_chronological.append(equity / shares)

    if len(bvps_chronological) < 3:
        return WarrenBuffettBookValueGrowth(
            bvps_latest=None,
            bvps_oldest=None,
            bvps_cagr_pct=None,
            growing_period_ratio=None,
            score=0,
            max_score=5,
            reasoning=(
                f"Insufficient BVPS history ({len(bvps_chronological)} valid periods)."
            ),
        )

    # Reverse to newest-first for "growing periods" check
    bvps_newest_first = list(reversed(bvps_chronological))
    score = 0
    reasoning: list[str] = []

    growing_periods = sum(
        1
        for i in range(len(bvps_newest_first) - 1)
        if bvps_newest_first[i] > bvps_newest_first[i + 1]
    )
    growth_rate = growing_periods / (len(bvps_newest_first) - 1)
    if growth_rate >= 0.8:
        score += 3
        reasoning.append(
            f"Consistent BVPS growth "
            f"({growing_periods}/{len(bvps_newest_first) - 1} periods)"
        )
    elif growth_rate >= 0.6:
        score += 2
        reasoning.append(
            f"Good BVPS growth pattern ({growing_periods}/{len(bvps_newest_first) - 1})"
        )
    elif growth_rate >= 0.4:
        score += 1
        reasoning.append(
            f"Moderate BVPS growth ({growing_periods}/{len(bvps_newest_first) - 1})"
        )
    else:
        reasoning.append(
            f"Inconsistent BVPS growth ({growing_periods}/{len(bvps_newest_first) - 1})"
        )

    oldest_bv = bvps_chronological[0]
    latest_bv = bvps_chronological[-1]
    years = len(bvps_chronological) - 1
    cagr_pct: float | None = None
    if oldest_bv > 0 and latest_bv > 0:
        cagr = (latest_bv / oldest_bv) ** (1 / years) - 1
        cagr_pct = cagr * 100
        if cagr > 0.15:
            score += 2
            reasoning.append(f"Excellent BVPS CAGR {cagr:.1%}")
        elif cagr > 0.10:
            score += 1
            reasoning.append(f"Good BVPS CAGR {cagr:.1%}")
        else:
            reasoning.append(f"BVPS CAGR {cagr:.1%}")
    elif oldest_bv < 0 < latest_bv:
        score += 3
        reasoning.append("Excellent: BVPS improved from negative to positive")
    elif oldest_bv > 0 > latest_bv:
        reasoning.append("Warning: BVPS deteriorated from positive to negative")

    return WarrenBuffettBookValueGrowth(
        bvps_latest=latest_bv,
        bvps_oldest=oldest_bv,
        bvps_cagr_pct=cagr_pct,
        growing_period_ratio=growth_rate,
        score=min(score, 5),
        max_score=5,
        reasoning="; ".join(reasoning),
    )


def _score_buffett_management(
    issuance_or_purchase_latest: float | None,
    dividends_latest: float | None,
) -> WarrenBuffettManagement:
    """Score Buffett's management heuristics from the latest cash-flow statement.

    * net stock issuance < 0 (net buybacks) → +1
    * dividends paid (cash outflow, negative) → +1

    Max = 2.
    """
    score = 0
    reasoning: list[str] = []

    if issuance_or_purchase_latest is not None and issuance_or_purchase_latest < 0:
        score += 1
        reasoning.append("Net share buybacks — shareholder-friendly")
    elif issuance_or_purchase_latest is not None and issuance_or_purchase_latest > 0:
        reasoning.append("Recent equity issuance — potential dilution")
    else:
        reasoning.append("No significant equity issuance / buyback activity")

    if dividends_latest is not None and dividends_latest < 0:
        score += 1
        reasoning.append("Pays dividends consistently")
    else:
        reasoning.append("No or minimal dividends")

    return WarrenBuffettManagement(
        net_buybacks_latest=issuance_or_purchase_latest,
        dividends_latest=dividends_latest,
        score=score,
        max_score=2,
        reasoning="; ".join(reasoning),
    )


# ── Graph nodes (deterministic compute + LLM verdict) ─────────────────────────


def compute_evidence_node(state: WarrenBuffettState) -> dict[str, Any]:
    """Deterministic: read RawData fields from state, compose ``WarrenBuffettEvidence``.

    Never crosses an LLM boundary. Pure Python — same composite scorers
    are called from the legacy bridge (``_compute_buffett_facts``) too.
    """
    metrics_history = state.get("metrics_history") or []
    # State may carry plain dicts (e.g. via legacy bridge) — coerce to BuffettMetricsRow
    metrics_rows: list[BuffettMetricsRow] = [
        m if isinstance(m, BuffettMetricsRow) else BuffettMetricsRow.model_validate(m)
        for m in metrics_history
    ]

    latest_metrics = metrics_rows[0] if metrics_rows else None

    fundamentals = _score_buffett_fundamentals(latest_metrics)
    consistency = _score_buffett_consistency(state.get("net_income_series"))
    moat = _score_buffett_moat(metrics_rows)
    pricing_power = _score_buffett_pricing_power(state.get("gross_margin_series"))
    book_value_growth = _score_buffett_book_value_growth(
        state.get("shareholders_equity_series"),
        state.get("outstanding_shares_series"),
    )
    management_quality = _score_buffett_management(
        state.get("latest_issuance_or_purchase_of_equity_shares"),
        state.get("latest_dividends_and_other_cash_distributions"),
    )

    # Owner earnings + intrinsic value + MoS
    ni_series = state.get("net_income_series") or []
    dna_series = state.get("depreciation_amortization_series") or []
    capex_series = state.get("capital_expenditure_series") or []
    net_income_latest = ni_series[-1] if ni_series else None
    dna_latest = dna_series[-1] if dna_series else None
    capex_latest_signed = capex_series[-1] if capex_series else None
    capex_latest = abs(capex_latest_signed) if capex_latest_signed is not None else None

    owner_earnings = compute_owner_earnings(
        net_income_latest, dna_latest, capex_latest, maintenance_capex_ratio=0.85
    )
    intrinsic_value: float | None = (
        compute_buffett_3stage_dcf(owner_earnings)
        if owner_earnings is not None and owner_earnings > 0
        else None
    )
    market_cap = state.get("market_cap")
    mos_pct: float | None = (
        (intrinsic_value - market_cap) / market_cap * 100
        if intrinsic_value is not None and market_cap and market_cap > 0
        else None
    )

    total = (
        fundamentals.total_score
        + consistency.score
        + moat.score
        + pricing_power.score
        + book_value_growth.score
        + management_quality.score
    )
    max_total = (
        fundamentals.max_score
        + consistency.max_score
        + moat.max_score
        + pricing_power.max_score
        + book_value_growth.max_score
        + management_quality.max_score
    )

    evidence = WarrenBuffettEvidence(
        fundamentals=fundamentals,
        consistency=consistency,
        moat=moat,
        pricing_power=pricing_power,
        book_value_growth=book_value_growth,
        management_quality=management_quality,
        owner_earnings=owner_earnings,
        intrinsic_value=intrinsic_value,
        market_cap=market_cap,
        margin_of_safety_pct=mos_pct,
        total_score=total,
        max_score=max_total,
    )
    return {"evidence": evidence}


async def render_verdict_node(
    state: WarrenBuffettState, config: RunnableConfig
) -> dict[str, Any]:
    """Single LLM call: read evidence, render WarrenBuffettSignal.

    The Jinja template receives the typed ``WarrenBuffettEvidence`` Pydantic
    instance (not ``.model_dump()``) so the prompt can use granular dotted
    attribute access for both data and conditional rule application.
    """
    ticker = state.get("ticker", "")
    as_of_date = state.get("as_of_date", "")
    query = state.get("query")
    evidence = state.get("evidence")
    if evidence is None:
        raise RuntimeError(
            "render_verdict_node called without evidence — "
            "compute_evidence_node must run first in v4 subgraphs"
        )

    llm = ModelConfiguration.get_chat_model_for_role(
        config, "reasoner", schema=WarrenBuffettSignal
    )
    prompt = render_template(
        "personas/warren_buffett.jinja",
        ticker=ticker,
        as_of_date=as_of_date,
        evidence=evidence,
        market_cap=state.get("market_cap"),
        query=query,
    )
    result = cast(
        WarrenBuffettSignal,
        await llm.ainvoke(
            [
                SystemMessage(prompt),
                HumanMessage("Render your Buffett verdict now."),
            ]
        ),
    )
    return {"persona_signals": [result.model_dump()]}


# ── Data-collection sub-agent + subgraph builder ──────────────────────────────


_MCP_TOOLS = [
    "equity_fundamental_metrics",
    "equity_fundamental_income",
    "equity_fundamental_balance",
    "equity_fundamental_cash",
    "equity_fundamental_ratios",
    "equity_historical_market_cap",
]


async def _build_data_collection_agent(config: RunnableConfig) -> CompiledStateGraph:
    """Compiled ReAct sub-agent that fetches MCP data → ``WarrenBuffettRawData``."""
    cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = cfg.get_llm_for_role("collector")
    mcp_tools = await get_tools(config, _MCP_TOOLS)

    builder = (
        MuffinAgentBuilder(primary, name="warren_buffett_data_collection")
        .with_fallback_models(*fallbacks)
        .with_state_schema(WarrenBuffettState)
        .with_runtime_system_prompt_template(
            "personas/warren_buffett_data_collection.jinja"
        )
        .with_response_format(WarrenBuffettRawData)
        .with_model_call_limit(run_limit=8, exit_behavior="end")
    )
    for tool in mcp_tools:
        builder = builder.with_tool(tool, run_limit=2)
    return builder.build_react_agent()


async def build_warren_buffett_agent(config: RunnableConfig) -> CompiledStateGraph:
    """Build the full 3-node Warren Buffett subgraph.

    Add to a parent graph (e.g. the council) via::

        agent = await build_warren_buffett_agent(config)
        parent.add_node("warren_buffett", agent, input_schema=agent.input_schema)
    """
    data_agent = await _build_data_collection_agent(config)
    graph = StateGraph(
        WarrenBuffettState,
        input_schema=WarrenBuffettInput,
        output_schema=WarrenBuffettOutput,
    )
    graph.add_node(
        "collect_data",
        data_agent,
        input_schema=data_agent.input_schema,
        retry_policy=_LLM_RETRY,
    )
    graph.add_node("compute_evidence", compute_evidence_node)
    graph.add_node("render_verdict", render_verdict_node, retry_policy=_LLM_RETRY)
    graph.add_edge(START, "collect_data")
    graph.add_edge("collect_data", "compute_evidence")
    graph.add_edge("compute_evidence", "render_verdict")
    graph.add_edge("render_verdict", END)
    return graph.compile()


