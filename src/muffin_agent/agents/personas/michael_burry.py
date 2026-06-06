"""Michael Burry persona — contrarian deep value.

Four sub-scores: deep value (FCF yield + EV/EBIT), balance sheet,
insider activity, contrarian sentiment.  Mirrors ai-hedge-fund's
``michael_burry.py``.  Total max = 12.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from ...model_config import ModelConfiguration
from ...prompts import render_template
from ...tools.sentiment import aggregate_insider_trades, aggregate_news_sentiment
from ._base import (
    PersonaInputState,
    PersonaOutputState,
    PersonaSpec,
    register_persona,
)
from .schemas import AnalystSignal, ScoreDetail

logger = logging.getLogger(__name__)


class MichaelBurryEvidence(BaseModel):
    """Persona-specific evidence — sub-scores, computed metrics."""

    deep_value: ScoreDetail
    balance_sheet: ScoreDetail
    insider_activity: ScoreDetail
    contrarian_sentiment: ScoreDetail
    fcf_yield: float | None
    ev_to_ebit: float | None
    total_score: float
    max_score: float


class MichaelBurrySignal(AnalystSignal):
    """Persona structured signal with narrowed evidence type."""

    agent_id: Literal["michael_burry"] = Field(default="michael_burry")
    evidence: MichaelBurryEvidence


def _score_deep_value(
    fcf_latest: float | None,
    market_cap: float | None,
    ebit_latest: float | None,
    total_debt_latest: float | None,
    cash_latest: float | None,
) -> tuple[ScoreDetail, float | None, float | None]:
    """FCF yield (max 4) + EV/EBIT (max 2)."""
    score = 0
    parts: list[str] = []
    fcf_yield: float | None = None
    ev_to_ebit: float | None = None

    if fcf_latest is not None and market_cap and market_cap > 0:
        fcf_yield = fcf_latest / market_cap
        if fcf_yield >= 0.15:
            score += 4
            parts.append(f"FCF yield {fcf_yield:.1%} (≥15% — deep value)")
        elif fcf_yield >= 0.12:
            score += 3
            parts.append(f"FCF yield {fcf_yield:.1%}")
        elif fcf_yield >= 0.08:
            score += 2

    if ebit_latest and ebit_latest > 0 and market_cap and market_cap > 0:
        ev = market_cap + (total_debt_latest or 0) - (cash_latest or 0)
        ev_to_ebit = ev / ebit_latest
        if ev_to_ebit < 6:
            score += 2
            parts.append(f"EV/EBIT {ev_to_ebit:.1f}× (very cheap)")
        elif ev_to_ebit < 10:
            score += 1
            parts.append(f"EV/EBIT {ev_to_ebit:.1f}×")

    return (
        ScoreDetail(
            score=min(score, 6),
            max_score=6,
            details="; ".join(parts) or "Cannot compute deep-value metrics",
        ),
        fcf_yield,
        ev_to_ebit,
    )


def _score_balance_sheet(line_items: dict[str, list[float | None]]) -> ScoreDetail:
    total_debt = [v for v in line_items.get("total_debt", []) if v is not None]
    equity = [v for v in line_items.get("shareholders_equity", []) if v is not None]
    cash = [v for v in line_items.get("cash_and_equivalents", []) if v is not None]
    score = 0
    parts: list[str] = []
    if total_debt and equity and equity[-1] and equity[-1] > 0:
        de = total_debt[-1] / equity[-1]
        if de < 0.5:
            score += 2
            parts.append(f"D/E {de:.2f} (low)")
        elif de < 1.0:
            score += 1
    if cash and total_debt:
        if cash[-1] > total_debt[-1]:
            score += 1
            parts.append("Net cash position")
    return ScoreDetail(score=min(score, 3), max_score=3, details="; ".join(parts))


def _score_insider_activity(insider_trades: list[dict[str, Any]]) -> ScoreDetail:
    """Net insider buying signal (max 2)."""
    agg = aggregate_insider_trades(insider_trades)
    score = 0
    parts: list[str] = []
    if agg["signal"] == "bullish":
        if agg["bullish_trades"] > agg["bearish_trades"] * 2:
            score = 2
            parts.append(
                f"Heavy net buying ({agg['bullish_trades']}/{agg['total_trades']})"
            )
        else:
            score = 1
            parts.append(
                f"Net insider buying ({agg['bullish_trades']}/{agg['total_trades']})"
            )
    elif agg["signal"] == "bearish":
        parts.append(
            f"Net insider selling ({agg['bearish_trades']}/{agg['total_trades']})"
        )
    else:
        parts.append("No insider activity signal")
    return ScoreDetail(score=score, max_score=2, details="; ".join(parts))


def _score_contrarian_sentiment(articles: list[dict[str, Any]]) -> ScoreDetail:
    """Score contrarian sentiment (max 1).

    5+ negative headlines = +1.  Burry: a wall of hate is a friend.
    """
    agg = aggregate_news_sentiment(articles)
    if agg["bearish_articles"] >= 5:
        return ScoreDetail(
            score=1,
            max_score=1,
            details=f"{agg['bearish_articles']} bearish headlines — contrarian setup",
        )
    return ScoreDetail(
        score=0,
        max_score=1,
        details=f"Insufficient bearish coverage ({agg['bearish_articles']} articles)",
    )


def _compute_burry_facts(data_bundle: dict[str, Any]) -> MichaelBurryEvidence:
    line_items = data_bundle.get("line_items", {})
    market_cap = data_bundle.get("market_cap")
    insider_trades = data_bundle.get("insider_trades", [])
    news = data_bundle.get("company_news", [])

    fcf_series = line_items.get("free_cash_flow", [])
    ebit_series = line_items.get("ebit", [])
    debt_series = line_items.get("total_debt", [])
    cash_series = line_items.get("cash_and_equivalents", [])

    deep_value, fcf_yield, ev_to_ebit = _score_deep_value(
        fcf_series[-1] if fcf_series else None,
        market_cap,
        ebit_series[-1] if ebit_series else None,
        debt_series[-1] if debt_series else None,
        cash_series[-1] if cash_series else None,
    )
    balance = _score_balance_sheet(line_items)
    insider = _score_insider_activity(insider_trades)
    contrarian = _score_contrarian_sentiment(news)
    total = deep_value.score + balance.score + insider.score + contrarian.score
    max_total = (
        deep_value.max_score
        + balance.max_score
        + insider.max_score
        + contrarian.max_score
    )
    return MichaelBurryEvidence(
        deep_value=deep_value,
        balance_sheet=balance,
        insider_activity=insider,
        contrarian_sentiment=contrarian,
        fcf_yield=fcf_yield,
        ev_to_ebit=ev_to_ebit,
        total_score=total,
        max_score=max_total,
    )


async def michael_burry_node(
    state: PersonaInputState, config: RunnableConfig
) -> PersonaOutputState:
    """Render the Michael Burry verdict from state["data_bundle"]."""
    ticker = state.get("ticker", "")
    query = state.get("query")
    data_bundle = state.get("data_bundle") or {}

    if not data_bundle or "error" in data_bundle:
        fallback = MichaelBurrySignal(
            agent_id="michael_burry",
            signal="hold",
            confidence=0.0,
            reasoning="Insufficient data",
            evidence=MichaelBurryEvidence(
                deep_value=ScoreDetail(score=0, max_score=6, details="no data"),
                balance_sheet=ScoreDetail(score=0, max_score=3, details="no data"),
                insider_activity=ScoreDetail(score=0, max_score=2, details="no data"),
                contrarian_sentiment=ScoreDetail(
                    score=0, max_score=1, details="no data"
                ),
                fcf_yield=None,
                ev_to_ebit=None,
                total_score=0,
                max_score=12,
            ),
        )
        return {"persona_signals": [fallback.model_dump()]}

    evidence = _compute_burry_facts(data_bundle)
    llm = ModelConfiguration.get_chat_model_for_role(
        config, "reasoner", schema=MichaelBurrySignal
    )
    prompt = render_template(
        "personas/michael_burry.jinja",
        ticker=ticker,
        as_of_date=data_bundle.get("as_of_date", ""),
        facts=evidence.model_dump(mode="json"),
        query=query,
        persona_display_name="Michael Burry",
        persona_slug="michael_burry",
        signal_schema_name="MichaelBurrySignal",
    )
    result = cast(
        MichaelBurrySignal,
        await llm.ainvoke(
            [SystemMessage(prompt), HumanMessage("Render your Burry verdict now.")]
        ),
    )
    return {"persona_signals": [result.model_dump()]}


PERSONA_SPEC = register_persona(
    PersonaSpec(
        slug="michael_burry",
        display_name="Michael Burry",
        investing_style=(
            "Contrarian deep value; FCF yield ≥15%, EV/EBIT <6, low debt; "
            "negative press + solid balance sheet = buy"
        ),
        node=michael_burry_node,
        signal_schema=MichaelBurrySignal,
    )
)
