"""Deterministic fundamentals scoring (ported from ai-hedge-fund's fundamentals.py).

Pure-Python multi-metric scoring across four dimensions — profitability, growth,
financial health, and valuation (price) ratios — combined by a simple majority
vote. No LLM. Consumed by the ``fundamentals`` specialist subgraph
(``agents/specialists/fundamentals_analysis.py``).

Upstream reference: ``ai-hedge-fund/src/agents/fundamentals.py`` — thresholds and
the bullish/bearish/neutral mapping are preserved verbatim.
"""

from __future__ import annotations

from typing import Any, TypedDict


class SubSignal(TypedDict):
    """One fundamentals sub-dimension result."""

    signal: str  # "bullish" | "bearish" | "neutral"
    score: int
    details: str


class FundamentalsResult(TypedDict):
    """Combined fundamentals signal."""

    signal: str  # "bullish" | "bearish" | "neutral"
    confidence: float  # 0.0–1.0
    profitability: SubSignal
    growth: SubSignal
    financial_health: SubSignal
    price_ratios: SubSignal


def _three_tier(score: int) -> str:
    """≥2 of 3 thresholds met → bullish, 0 → bearish, else neutral."""
    if score >= 2:
        return "bullish"
    if score == 0:
        return "bearish"
    return "neutral"


def _gt(value: float | None, threshold: float) -> bool:
    return value is not None and value > threshold


def _fmt(value: float | None, *, pct: bool = False) -> str:
    """Compact value formatter for the human-readable detail strings."""
    if value is None:
        return "n/a"
    return f"{value:.1%}" if pct else f"{value:.2f}"


def score_profitability(
    return_on_equity: float | None,
    net_margin: float | None,
    operating_margin: float | None,
) -> SubSignal:
    """ROE>15% / net margin>20% / operating margin>15% (upstream thresholds)."""
    score = sum(
        (
            _gt(return_on_equity, 0.15),
            _gt(net_margin, 0.20),
            _gt(operating_margin, 0.15),
        )
    )
    return SubSignal(
        signal=_three_tier(score),
        score=score,
        details=(
            f"ROE {_fmt(return_on_equity, pct=True)}, "
            f"net margin {_fmt(net_margin, pct=True)}, "
            f"op margin {_fmt(operating_margin, pct=True)}"
        ),
    )


def score_growth(
    revenue_growth: float | None,
    earnings_growth: float | None,
    book_value_growth: float | None,
) -> SubSignal:
    """Revenue / earnings / book-value growth all vs 10% (upstream thresholds)."""
    score = sum(
        (
            _gt(revenue_growth, 0.10),
            _gt(earnings_growth, 0.10),
            _gt(book_value_growth, 0.10),
        )
    )
    return SubSignal(
        signal=_three_tier(score),
        score=score,
        details=(
            f"revenue growth {_fmt(revenue_growth, pct=True)}, "
            f"earnings growth {_fmt(earnings_growth, pct=True)}"
        ),
    )


def score_financial_health(
    current_ratio: float | None,
    debt_to_equity: float | None,
    free_cash_flow_per_share: float | None,
    earnings_per_share: float | None,
) -> SubSignal:
    """Liquidity / leverage / FCF-conversion checks (upstream thresholds)."""
    score = 0
    if _gt(current_ratio, 1.5):
        score += 1
    if debt_to_equity is not None and debt_to_equity < 0.5:
        score += 1
    if (
        free_cash_flow_per_share is not None
        and earnings_per_share is not None
        and free_cash_flow_per_share > earnings_per_share * 0.8
    ):
        score += 1
    return SubSignal(
        signal=_three_tier(score),
        score=score,
        details=(
            f"current ratio {_fmt(current_ratio)}, D/E {_fmt(debt_to_equity)}"
        ),
    )


def score_price_ratios(
    pe_ratio: float | None,
    pb_ratio: float | None,
    ps_ratio: float | None,
) -> SubSignal:
    """P/E>25 / P/B>3 / P/S>5 are expensive → the mapping is INVERTED.

    ≥2 rich ratios → bearish, 0 → bullish, else neutral.
    """
    rich = sum((_gt(pe_ratio, 25), _gt(pb_ratio, 3), _gt(ps_ratio, 5)))
    if rich >= 2:
        signal = "bearish"
    elif rich == 0:
        signal = "bullish"
    else:
        signal = "neutral"
    return SubSignal(
        signal=signal,
        score=rich,
        details=(
            f"P/E {_fmt(pe_ratio)}, P/B {_fmt(pb_ratio)}, P/S {_fmt(ps_ratio)}"
        ),
    )


def score_fundamentals(metrics: dict[str, Any]) -> FundamentalsResult:
    """Combine the four fundamentals dimensions by majority vote.

    Confidence = ``max(bullish, bearish) / 4`` (proportion of agreeing
    sub-signals), as a 0.0–1.0 fraction (upstream expressed it ×100).

    Args:
        metrics: Latest financial-metrics snapshot. Recognised keys:
            ``return_on_equity``, ``net_margin``, ``operating_margin``,
            ``revenue_growth``, ``earnings_growth``, ``book_value_growth``,
            ``current_ratio``, ``debt_to_equity``, ``free_cash_flow_per_share``,
            ``earnings_per_share``, ``price_to_earnings_ratio``,
            ``price_to_book_ratio``, ``price_to_sales_ratio``.
    """
    profitability = score_profitability(
        metrics.get("return_on_equity"),
        metrics.get("net_margin"),
        metrics.get("operating_margin"),
    )
    growth = score_growth(
        metrics.get("revenue_growth"),
        metrics.get("earnings_growth"),
        metrics.get("book_value_growth"),
    )
    health = score_financial_health(
        metrics.get("current_ratio"),
        metrics.get("debt_to_equity"),
        metrics.get("free_cash_flow_per_share"),
        metrics.get("earnings_per_share"),
    )
    price = score_price_ratios(
        metrics.get("price_to_earnings_ratio"),
        metrics.get("price_to_book_ratio"),
        metrics.get("price_to_sales_ratio"),
    )

    signals = [
        profitability["signal"],
        growth["signal"],
        health["signal"],
        price["signal"],
    ]
    bullish = signals.count("bullish")
    bearish = signals.count("bearish")
    if bullish > bearish:
        overall = "bullish"
    elif bearish > bullish:
        overall = "bearish"
    else:
        overall = "neutral"
    confidence = max(bullish, bearish) / len(signals)

    return FundamentalsResult(
        signal=overall,
        confidence=confidence,
        profitability=profitability,
        growth=growth,
        financial_health=health,
        price_ratios=price,
    )
