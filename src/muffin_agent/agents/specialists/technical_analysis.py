"""Technical analysis specialist — deterministic 5-strategy ensemble.

Wraps :mod:`muffin_agent.tools.technicals` (trend, mean-reversion,
momentum, vol-regime, stat-arb) into a single :class:`AnalystSignal`.
**No LLM call** — fully deterministic, mirrors ai-hedge-fund's
upstream ``technicals.py``.

Maps the 3-tier internal ``TacticalSignal`` (bullish/bearish/neutral) to
the 5-tier ``InvestmentSignal`` using the combined-confidence threshold:
``confidence ≥ 0.7`` promotes a directional signal to its strong variant.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from ...tools.technicals import (
    StrategyResult,
    combine_technical_signals,
    compute_mean_reversion_signal,
    compute_momentum_signal,
    compute_stat_arb_signal,
    compute_trend_signal,
    compute_volatility_regime_signal,
)
from ..personas._base import PersonaInputState, PersonaOutputState
from ..personas.schemas import AnalystSignal, InvestmentSignal
from ._base import SpecialistSpec, register_specialist

logger = logging.getLogger(__name__)


# ── Evidence + signal ─────────────────────────────────────────────────────────


class TechnicalEvidence(BaseModel):
    """Per-strategy results from the 5-strategy ensemble."""

    trend: dict[str, Any]
    mean_reversion: dict[str, Any]
    momentum: dict[str, Any]
    volatility_regime: dict[str, Any]
    stat_arb: dict[str, Any]
    weighted: dict[str, Any]
    """Combined ensemble signal with per-strategy contribution metrics."""


class TechnicalSignal(AnalystSignal):
    """Technicals structured signal."""

    agent_id: Literal["technicals"] = Field(default="technicals")
    evidence: TechnicalEvidence


# ── Mapping helpers ───────────────────────────────────────────────────────────


def _to_5tier(
    tactical_signal: str, confidence: float, strong_threshold: float = 0.7
) -> InvestmentSignal:
    """Convert a 3-tier ``bullish/bearish/neutral`` + confidence to 5-tier rating.

    - ``bullish`` with confidence ≥ ``strong_threshold`` → ``strong_buy``
    - ``bullish`` otherwise → ``buy``
    - ``neutral`` → ``hold``
    - ``bearish`` otherwise → ``sell``
    - ``bearish`` with confidence ≥ ``strong_threshold`` → ``strong_sell``
    """
    if tactical_signal == "bullish":
        return "strong_buy" if confidence >= strong_threshold else "buy"
    if tactical_signal == "bearish":
        return "strong_sell" if confidence >= strong_threshold else "sell"
    return "hold"


def _build_reasoning(
    weighted: StrategyResult, per_strategy: dict[str, StrategyResult]
) -> str:
    """Build a one-line deterministic reasoning string from the ensemble result."""
    parts: list[str] = []
    parts.append(f"Ensemble {weighted['signal']} (conf {weighted['confidence']:.2f})")
    bullish_strats = [n for n, r in per_strategy.items() if r["signal"] == "bullish"]
    bearish_strats = [n for n, r in per_strategy.items() if r["signal"] == "bearish"]
    if bullish_strats:
        parts.append(f"bullish: {', '.join(bullish_strats)}")
    if bearish_strats:
        parts.append(f"bearish: {', '.join(bearish_strats)}")
    return "; ".join(parts)


# ── Node ──────────────────────────────────────────────────────────────────────


def _empty_fallback() -> TechnicalSignal:
    """Hold signal when no price data is available."""
    empty: dict[str, Any] = {"signal": "neutral", "confidence": 0.0, "metrics": {}}
    return TechnicalSignal(
        agent_id="technicals",
        signal="hold",
        confidence=0.0,
        reasoning="No price data available",
        evidence=TechnicalEvidence(
            trend=empty,
            mean_reversion=empty,
            momentum=empty,
            volatility_regime=empty,
            stat_arb=empty,
            weighted=empty,
        ),
    )


async def technical_analysis_node(
    state: PersonaInputState, config: RunnableConfig
) -> PersonaOutputState:
    """Compute the technical ensemble for ``state["data_bundle"].prices_1y``.

    Fully deterministic — runs the 5 strategies via the pandas helpers in
    ``tools/technicals.py``, combines via the weighted ensemble, then
    maps the 3-tier result to the 5-tier ``InvestmentSignal``.  No LLM
    call is made — output ``reasoning`` is a one-line deterministic
    summary.
    """
    data_bundle = state.get("data_bundle") or {}
    if not data_bundle or "error" in data_bundle:
        return {"persona_signals": [_empty_fallback().model_dump()]}

    prices_1y = data_bundle.get("prices_1y") or []
    if len(prices_1y) < 20:
        return {"persona_signals": [_empty_fallback().model_dump()]}

    trend = compute_trend_signal(prices_1y)
    mr = compute_mean_reversion_signal(prices_1y)
    momentum = compute_momentum_signal(prices_1y)
    vol = compute_volatility_regime_signal(prices_1y)
    stat = compute_stat_arb_signal(prices_1y)
    per_strategy: dict[str, StrategyResult] = {
        "trend": trend,
        "mean_reversion": mr,
        "momentum": momentum,
        "volatility": vol,
        "stat_arb": stat,
    }
    weighted = combine_technical_signals(per_strategy)

    rating = _to_5tier(weighted["signal"], weighted["confidence"])
    sig = TechnicalSignal(
        agent_id="technicals",
        signal=rating,
        confidence=min(weighted["confidence"], 1.0),
        reasoning=_build_reasoning(weighted, per_strategy),
        evidence=TechnicalEvidence(
            trend=dict(trend),
            mean_reversion=dict(mr),
            momentum=dict(momentum),
            volatility_regime=dict(vol),
            stat_arb=dict(stat),
            weighted=dict(weighted),
        ),
    )
    return {"persona_signals": [sig.model_dump()]}


# ── Registry entry ────────────────────────────────────────────────────────────


SPECIALIST_SPEC = register_specialist(
    SpecialistSpec(
        slug="technicals",
        display_name="Technical Analysis",
        investing_style=(
            "5-strategy technical ensemble: trend / mean-reversion / momentum / "
            "volatility-regime / stat-arb. Deterministic, no LLM."
        ),
        node=technical_analysis_node,
        signal_schema=TechnicalSignal,
    )
)
