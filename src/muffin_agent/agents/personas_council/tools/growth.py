"""Deterministic growth scoring (ported from ai-hedge-fund's growth_agent.py).

Five weighted sub-scores — historical growth trend (0.40), growth-oriented
valuation (0.25), margin expansion (0.15), insider conviction (0.10), financial
health (0.10) — combined into a 0–1 weighted score and a bullish/bearish/neutral
signal. No LLM. Consumed by the ``growth`` specialist subgraph.

Upstream reference: ``ai-hedge-fund/src/agents/growth_agent.py``.

Series convention: this module takes all time series **oldest → newest** (the
muffin convention), so a positive trend slope means the metric is *accelerating*
(recent > older) — matching the upstream "# Accelerating" intent. (Upstream fed
its trend helper newest-first data, which inverted the slope sign relative to its
own comments; the oldest→newest orientation here is the corrected behaviour.)
"""

from __future__ import annotations

from typing import Any, TypedDict


class GrowthResult(TypedDict):
    """Combined growth signal."""

    signal: str  # "bullish" | "bearish" | "neutral"
    confidence: float  # 0.0–1.0
    weighted_score: float  # 0.0–1.0
    growth_trends: dict[str, Any]
    valuation: dict[str, Any]
    margins: dict[str, Any]
    insider: dict[str, Any]
    health: dict[str, Any]


_WEIGHTS = {
    "growth": 0.40,
    "valuation": 0.25,
    "margins": 0.15,
    "insider": 0.10,
    "health": 0.10,
}


def _clean(series: list[float | None] | None) -> list[float]:
    return [v for v in (series or []) if v is not None]


def trend_slope(series: list[float | None] | None) -> float:
    """Least-squares slope over an oldest→newest series (0.0 if <2 points)."""
    ys = _clean(series)
    if len(ys) < 2:
        return 0.0
    xs = list(range(len(ys)))
    n = len(ys)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys, strict=False))
    sum_x2 = sum(x * x for x in xs)
    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0:
        return 0.0
    return (n * sum_xy - sum_x * sum_y) / denom


def _latest(series: list[float | None] | None) -> float | None:
    cleaned = _clean(series)
    return cleaned[-1] if cleaned else None


def score_growth_trends(
    revenue_growth: list[float | None] | None,
    eps_growth: list[float | None] | None,
    fcf_growth: list[float | None] | None,
) -> dict[str, Any]:
    """Recent growth level + acceleration across revenue / EPS / FCF (max 1.0)."""
    score = 0.0
    rev_latest = _latest(revenue_growth)
    eps_latest = _latest(eps_growth)
    fcf_latest = _latest(fcf_growth)
    rev_trend = trend_slope(revenue_growth)
    eps_trend = trend_slope(eps_growth)
    fcf_trend = trend_slope(fcf_growth)

    if rev_latest is not None:
        if rev_latest > 0.20:
            score += 0.4
        elif rev_latest > 0.10:
            score += 0.2
        if rev_trend > 0:
            score += 0.1
    if eps_latest is not None:
        if eps_latest > 0.20:
            score += 0.25
        elif eps_latest > 0.10:
            score += 0.1
        if eps_trend > 0:
            score += 0.05
    if fcf_latest is not None and fcf_latest > 0.15:
        score += 0.1

    return {
        "score": min(score, 1.0),
        "revenue_growth": rev_latest,
        "revenue_trend": rev_trend,
        "eps_growth": eps_latest,
        "eps_trend": eps_trend,
        "fcf_growth": fcf_latest,
        "fcf_trend": fcf_trend,
    }


def score_growth_valuation(
    peg_ratio: float | None, price_to_sales_ratio: float | None
) -> dict[str, Any]:
    """PEG + P/S from a growth lens (max 1.0)."""
    score = 0.0
    if peg_ratio is not None:
        if peg_ratio < 1.0:
            score += 0.5
        elif peg_ratio < 2.0:
            score += 0.25
    if price_to_sales_ratio is not None:
        if price_to_sales_ratio < 2.0:
            score += 0.5
        elif price_to_sales_ratio < 5.0:
            score += 0.25
    return {
        "score": min(score, 1.0),
        "peg_ratio": peg_ratio,
        "price_to_sales_ratio": price_to_sales_ratio,
    }


def score_margin_trends(
    gross_margin: list[float | None] | None,
    operating_margin: list[float | None] | None,
    net_margin: list[float | None] | None,
) -> dict[str, Any]:
    """Margin level + expansion across gross / operating / net (max 1.0)."""
    score = 0.0
    gm_latest = _latest(gross_margin)
    om_latest = _latest(operating_margin)
    gm_trend = trend_slope(gross_margin)
    om_trend = trend_slope(operating_margin)
    nm_trend = trend_slope(net_margin)

    if gm_latest is not None:
        if gm_latest > 0.5:
            score += 0.2
        if gm_trend > 0:
            score += 0.2
    if om_latest is not None:
        if om_latest > 0.15:
            score += 0.2
        if om_trend > 0:
            score += 0.2
    if nm_trend > 0:
        score += 0.2

    return {
        "score": min(score, 1.0),
        "gross_margin": gm_latest,
        "gross_margin_trend": gm_trend,
        "operating_margin": om_latest,
        "operating_margin_trend": om_trend,
        "net_margin_trend": nm_trend,
    }


def score_insider_conviction(trades: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Dollar-weighted net insider flow ratio (max 1.0).

    Uses ``transaction_value`` when present (signed by ``transaction_shares``),
    falling back to share counts when value is unavailable.
    """
    buys = 0.0
    sells = 0.0
    for t in trades or []:
        shares = t.get("transaction_shares")
        value = t.get("transaction_value")
        if shares is None:
            continue
        magnitude = abs(value) if value is not None else abs(shares)
        if shares > 0:
            buys += magnitude
        elif shares < 0:
            sells += magnitude

    total = buys + sells
    net_flow_ratio = (buys - sells) / total if total > 0 else 0.0
    if net_flow_ratio > 0.5:
        score = 1.0
    elif net_flow_ratio > 0.1:
        score = 0.7
    elif net_flow_ratio > -0.1:
        score = 0.5
    else:
        score = 0.2
    return {
        "score": score,
        "net_flow_ratio": net_flow_ratio,
        "buys": buys,
        "sells": sells,
    }


def score_financial_health(
    debt_to_equity: float | None, current_ratio: float | None
) -> dict[str, Any]:
    """Leverage + liquidity penalties off a 1.0 base (floor 0.0)."""
    score = 1.0
    if debt_to_equity is not None:
        if debt_to_equity > 1.5:
            score -= 0.5
        elif debt_to_equity > 0.8:
            score -= 0.2
    if current_ratio is not None:
        if current_ratio < 1.0:
            score -= 0.5
        elif current_ratio < 1.5:
            score -= 0.2
    return {
        "score": max(score, 0.0),
        "debt_to_equity": debt_to_equity,
        "current_ratio": current_ratio,
    }


def score_growth_signals(
    revenue_growth: list[float | None] | None,
    eps_growth: list[float | None] | None,
    fcf_growth: list[float | None] | None,
    gross_margin: list[float | None] | None,
    operating_margin: list[float | None] | None,
    net_margin: list[float | None] | None,
    peg_ratio: float | None,
    price_to_sales_ratio: float | None,
    debt_to_equity: float | None,
    current_ratio: float | None,
    insider_trades: list[dict[str, Any]] | None,
) -> GrowthResult:
    """Combine the five growth dimensions with upstream weights.

    weighted_score >0.6 → bullish, <0.4 → bearish, else neutral.
    confidence = ``|weighted_score − 0.5| × 2`` (0.0–1.0).
    """
    growth = score_growth_trends(revenue_growth, eps_growth, fcf_growth)
    valuation = score_growth_valuation(peg_ratio, price_to_sales_ratio)
    margins = score_margin_trends(gross_margin, operating_margin, net_margin)
    insider = score_insider_conviction(insider_trades)
    health = score_financial_health(debt_to_equity, current_ratio)

    weighted = (
        growth["score"] * _WEIGHTS["growth"]
        + valuation["score"] * _WEIGHTS["valuation"]
        + margins["score"] * _WEIGHTS["margins"]
        + insider["score"] * _WEIGHTS["insider"]
        + health["score"] * _WEIGHTS["health"]
    )
    if weighted > 0.6:
        signal = "bullish"
    elif weighted < 0.4:
        signal = "bearish"
    else:
        signal = "neutral"
    confidence = min(abs(weighted - 0.5) * 2, 1.0)

    return GrowthResult(
        signal=signal,
        confidence=confidence,
        weighted_score=weighted,
        growth_trends=growth,
        valuation=valuation,
        margins=margins,
        insider=insider,
        health=health,
    )
