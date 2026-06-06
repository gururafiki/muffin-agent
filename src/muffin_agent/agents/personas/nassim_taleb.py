"""Nassim Taleb persona — antifragility, tail risk, convexity, via negativa.

Uses price-series data for tail-risk metrics (skew, kurtosis, drawdown,
vol regime), plus standard line items for fragility / antifragility
scoring.  Mirrors ai-hedge-fund's ``nassim_taleb.py`` (which is the only
upstream persona that pulls prices directly).
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
from ...tools.scoring_helpers import compute_volatility_metrics
from ._base import (
    PersonaInputState,
    PersonaOutputState,
    PersonaSpec,
    register_persona,
)
from .schemas import AnalystSignal, ScoreDetail

logger = logging.getLogger(__name__)


class NassimTalebEvidence(BaseModel):
    """Persona-specific evidence — sub-scores, computed metrics."""

    tail_risk: ScoreDetail
    antifragility: ScoreDetail
    convexity: ScoreDetail
    fragility: ScoreDetail
    skin_in_game: ScoreDetail
    vol_regime: ScoreDetail
    annualized_volatility: float | None
    skewness: float | None
    excess_kurtosis: float | None
    max_drawdown_pct: float | None
    total_score: float
    max_score: float


class NassimTalebSignal(AnalystSignal):
    """Persona structured signal with narrowed evidence type."""

    agent_id: Literal["nassim_taleb"] = Field(default="nassim_taleb")
    evidence: NassimTalebEvidence


def _daily_returns_from_bars(bars: list[dict[str, Any]]) -> list[float]:
    closes = [b.get("close") for b in bars if b.get("close") is not None]
    returns: list[float] = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        if prev and prev > 0:
            returns.append((closes[i] - prev) / prev)
    return returns


def _score_tail_risk(
    returns: list[float],
) -> tuple[ScoreDetail, dict[str, float | None]]:
    metrics = compute_volatility_metrics(returns)
    score = 0
    parts: list[str] = []
    skew = metrics["skewness"]
    kurt = metrics["excess_kurtosis"]
    dd = metrics["max_drawdown_pct"]

    if kurt is not None:
        if kurt > 5:
            score += 2
            parts.append(f"Excess kurtosis {kurt:.1f} — fat tails")
        elif kurt > 2:
            score += 1
    if skew is not None:
        if skew > 0.5:
            score += 2
            parts.append(f"Positive skew {skew:.2f} — upside-heavy distribution")
        elif skew > -0.5:
            score += 1
    if dd is not None:
        if dd > -15:  # max_drawdown_pct is negative
            score += 2
            parts.append(f"Max drawdown {dd:.1f}% (shallow)")
        elif dd > -30:
            score += 1
    if metrics["annualized_volatility"] is not None:
        parts.append(f"Annualised vol {metrics['annualized_volatility']:.1%}")

    return ScoreDetail(
        score=min(score, 8), max_score=8, details="; ".join(parts) or "No price data"
    ), metrics


def _score_antifragility(
    line_items: dict[str, list[float | None]], market_cap: float | None
) -> ScoreDetail:
    """Net cash, low debt, stable margins, consistent FCF."""
    cash = [v for v in line_items.get("cash_and_equivalents", []) if v is not None]
    debt = [v for v in line_items.get("total_debt", []) if v is not None]
    assets = [v for v in line_items.get("total_assets", []) if v is not None]
    margins = [v for v in line_items.get("operating_margin", []) if v is not None]
    fcf = [v for v in line_items.get("free_cash_flow", []) if v is not None]
    score = 0
    parts: list[str] = []
    if cash and debt:
        net_cash = cash[-1] - debt[-1]
        if net_cash > 0:
            score += 2
            if market_cap and net_cash > 0.20 * market_cap:
                score += 1
                parts.append(f"Net cash {net_cash:,.0f} (>20% of mkt cap)")
            else:
                parts.append(f"Net cash {net_cash:,.0f}")
        elif debt and assets and debt[-1] / assets[-1] < 0.30:
            score += 1
    if debt and assets and assets[-1] and assets[-1] > 0:
        de = debt[-1] / assets[-1]
        if de < 0.30:
            score += 2
            parts.append(f"D/assets {de:.2f}")
        elif de < 0.50:
            score += 1
    if margins and len(margins) >= 3:
        mean = sum(margins) / len(margins)
        if mean > 0:
            cv = statistics.pstdev(margins) / mean
            if cv < 0.15 and mean > 0.15:
                score += 3
                parts.append(f"Stable margins (CV {cv:.1%}, mean {mean:.1%})")
            elif cv < 0.30 and mean > 0.10:
                score += 2
            elif cv < 0.30:
                score += 1
    if fcf:
        positives = sum(1 for v in fcf if v > 0)
        if positives == len(fcf):
            score += 2
            parts.append("FCF positive every period")
        elif positives >= len(fcf) // 2:
            score += 1
    return ScoreDetail(
        score=min(score, 10), max_score=10, details="; ".join(parts) or "Limited data"
    )


def _score_convexity(
    line_items: dict[str, list[float | None]], market_cap: float | None
) -> ScoreDetail:
    """R&D intensity + cash optionality + FCF yield."""
    rd = [v for v in line_items.get("research_and_development", []) if v is not None]
    revenues = [v for v in line_items.get("revenue", []) if v is not None]
    cash = [v for v in line_items.get("cash_and_equivalents", []) if v is not None]
    fcf = [v for v in line_items.get("free_cash_flow", []) if v is not None]
    score = 0
    parts: list[str] = []
    if rd and revenues and revenues[-1] and revenues[-1] > 0:
        intensity = rd[-1] / revenues[-1]
        if intensity > 0.15:
            score += 3
            parts.append(f"R&D intensity {intensity:.1%} (high convexity)")
        elif intensity > 0.08:
            score += 2
        elif intensity > 0.03:
            score += 1
    if cash and market_cap and market_cap > 0:
        ratio = cash[-1] / market_cap
        if ratio > 0.30:
            score += 3
            parts.append(f"Cash {ratio:.1%} of mkt cap (option value)")
        elif ratio > 0.15:
            score += 2
        elif ratio > 0.05:
            score += 1
    if fcf and market_cap and market_cap > 0:
        yield_ = fcf[-1] / market_cap
        if yield_ > 0.10:
            score += 2
            parts.append(f"FCF yield {yield_:.1%}")
        elif yield_ > 0.05:
            score += 1
    return ScoreDetail(
        score=min(score, 10), max_score=10, details="; ".join(parts) or "Limited data"
    )


def _score_fragility(line_items: dict[str, list[float | None]]) -> ScoreDetail:
    """Via negativa — score what should be AVOIDED.  High score = LESS fragile."""
    debt = [v for v in line_items.get("total_debt", []) if v is not None]
    equity = [v for v in line_items.get("shareholders_equity", []) if v is not None]
    ebit = [v for v in line_items.get("ebit", []) if v is not None]
    interest = [v for v in line_items.get("interest_expense", []) if v is not None]
    net_income = [v for v in line_items.get("net_income", []) if v is not None]
    score = 0
    parts: list[str] = []
    if debt and equity and equity[-1] and equity[-1] > 0:
        de = debt[-1] / equity[-1]
        if de < 0.5:
            score += 3
            parts.append(f"Low D/E {de:.2f}")
        elif de < 1.0:
            score += 2
        elif de < 2.0:
            score += 1
        else:
            parts.append(f"Fragile leverage D/E {de:.2f}")
    if ebit and interest and interest[-1] and interest[-1] > 0:
        cov = ebit[-1] / interest[-1]
        if cov > 10:
            score += 2
            parts.append(f"Interest coverage {cov:.1f}×")
        elif cov > 5:
            score += 1
    if net_income and len(net_income) >= 3:
        mean = sum(net_income) / len(net_income)
        if mean > 0:
            cv = statistics.pstdev(net_income) / mean
            if cv < 0.20:
                score += 2
                parts.append(f"Stable earnings (CV {cv:.1%})")
            elif cv < 0.50:
                score += 1
    return ScoreDetail(
        score=min(score, 8), max_score=8, details="; ".join(parts) or "Limited data"
    )


def _score_skin_in_game(insider_trades: list[dict[str, Any]]) -> ScoreDetail:
    buys = sum(1 for t in insider_trades if (t.get("transaction_shares") or 0) > 0)
    sells = sum(1 for t in insider_trades if (t.get("transaction_shares") or 0) < 0)
    if buys + sells == 0:
        return ScoreDetail(score=0, max_score=4, details="No insider activity data")
    ratio = buys / max(sells, 1)
    if ratio > 2:
        return ScoreDetail(
            score=4, max_score=4, details=f"Strong net buying ({buys}/{sells})"
        )
    if ratio > 0.5:
        return ScoreDetail(score=3, max_score=4, details=f"Net buying ({buys}/{sells})")
    if buys > 0:
        return ScoreDetail(
            score=2, max_score=4, details=f"Some buying ({buys}/{sells})"
        )
    return ScoreDetail(score=1, max_score=4, details="No insider buying")


def _score_vol_regime(returns: list[float]) -> ScoreDetail:
    """Score vol regime — Taleb's "turkey problem" detector."""
    if len(returns) < 63:
        return ScoreDetail(score=0, max_score=6, details="Insufficient price history")
    recent = returns[-21:]
    older = returns[-63:-21] if len(returns) >= 63 else returns[:-21]
    recent_vol = statistics.pstdev(recent) if recent else 0
    older_vol = statistics.pstdev(older) if older else 0
    if older_vol == 0:
        return ScoreDetail(
            score=0, max_score=6, details="Zero historical vol — anomaly"
        )
    regime = recent_vol / older_vol
    score = 0
    parts: list[str] = []
    if regime < 0.7:
        score = 0
        parts.append(
            f"Vol regime {regime:.2f} — dangerously suppressed (turkey problem)"
        )
    elif regime < 0.9:
        score = 1
    elif regime < 1.3:
        score = 3
        parts.append(f"Normal vol regime {regime:.2f}")
    elif regime < 2.0:
        score = 4
        parts.append(f"Elevated vol regime {regime:.2f} (potentially attractive entry)")
    else:
        score = 2
        parts.append(f"Extreme vol regime {regime:.2f}")
    return ScoreDetail(score=score, max_score=6, details="; ".join(parts))


def _compute_taleb_facts(data_bundle: dict[str, Any]) -> NassimTalebEvidence:
    line_items = data_bundle.get("line_items", {})
    market_cap = data_bundle.get("market_cap")
    insider_trades = data_bundle.get("insider_trades", [])
    prices_1y = data_bundle.get("prices_1y", [])
    returns = _daily_returns_from_bars(prices_1y)

    tail_risk, vol_metrics = _score_tail_risk(returns)
    antifragility = _score_antifragility(line_items, market_cap)
    convexity = _score_convexity(line_items, market_cap)
    fragility = _score_fragility(line_items)
    skin = _score_skin_in_game(insider_trades)
    vol_regime = _score_vol_regime(returns)

    total = (
        tail_risk.score
        + antifragility.score
        + convexity.score
        + fragility.score
        + skin.score
        + vol_regime.score
    )
    max_total = (
        tail_risk.max_score
        + antifragility.max_score
        + convexity.max_score
        + fragility.max_score
        + skin.max_score
        + vol_regime.max_score
    )

    return NassimTalebEvidence(
        tail_risk=tail_risk,
        antifragility=antifragility,
        convexity=convexity,
        fragility=fragility,
        skin_in_game=skin,
        vol_regime=vol_regime,
        annualized_volatility=vol_metrics["annualized_volatility"],
        skewness=vol_metrics["skewness"],
        excess_kurtosis=vol_metrics["excess_kurtosis"],
        max_drawdown_pct=vol_metrics["max_drawdown_pct"],
        total_score=total,
        max_score=max_total,
    )


async def nassim_taleb_node(
    state: PersonaInputState, config: RunnableConfig
) -> PersonaOutputState:
    """Render the Nassim Taleb verdict from state["data_bundle"]."""
    ticker = state.get("ticker", "")
    query = state.get("query")
    data_bundle = state.get("data_bundle") or {}

    if not data_bundle or "error" in data_bundle:
        fallback = NassimTalebSignal(
            agent_id="nassim_taleb",
            signal="hold",
            confidence=0.0,
            reasoning="Insufficient data",
            evidence=NassimTalebEvidence(
                tail_risk=ScoreDetail(score=0, max_score=8, details="no data"),
                antifragility=ScoreDetail(score=0, max_score=10, details="no data"),
                convexity=ScoreDetail(score=0, max_score=10, details="no data"),
                fragility=ScoreDetail(score=0, max_score=8, details="no data"),
                skin_in_game=ScoreDetail(score=0, max_score=4, details="no data"),
                vol_regime=ScoreDetail(score=0, max_score=6, details="no data"),
                annualized_volatility=None,
                skewness=None,
                excess_kurtosis=None,
                max_drawdown_pct=None,
                total_score=0,
                max_score=46,
            ),
        )
        return {"persona_signals": [fallback.model_dump()]}

    evidence = _compute_taleb_facts(data_bundle)
    llm = ModelConfiguration.get_chat_model_for_role(
        config, "reasoner", schema=NassimTalebSignal
    )
    prompt = render_template(
        "personas/nassim_taleb.jinja",
        ticker=ticker,
        as_of_date=data_bundle.get("as_of_date", ""),
        facts=evidence.model_dump(mode="json"),
        query=query,
        persona_display_name="Nassim Taleb",
        persona_slug="nassim_taleb",
        signal_schema_name="NassimTalebSignal",
    )
    result = cast(
        NassimTalebSignal,
        await llm.ainvoke(
            [SystemMessage(prompt), HumanMessage("Render your Taleb verdict now.")]
        ),
    )
    return {"persona_signals": [result.model_dump()]}


PERSONA_SPEC = register_persona(
    PersonaSpec(
        slug="nassim_taleb",
        display_name="Nassim Taleb",
        investing_style=(
            "Antifragility + convexity + via negativa; tail-risk-aware; "
            "barbell strategy; penalises fragility & turkey-problem vol"
        ),
        node=nassim_taleb_node,
        signal_schema=NassimTalebSignal,
    )
)
