"""Backtest performance metrics — Sharpe / Sortino / max drawdown / exposures.

Pure-Python (statistics stdlib + math), no scipy dependency.  Used by
:func:`BacktestEngine.summary` after the walk-forward loop completes.
Mirrors ai-hedge-fund's metrics conventions: 252 trading days/year for
annualisation, default risk-free rate 4.34%.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable

DEFAULT_RISK_FREE_RATE = 0.0434
"""Annual risk-free rate default (4.34% — matches ai-hedge-fund's upstream)."""


def compute_sharpe(
    returns: Iterable[float],
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    frequency: int = 252,
) -> float | None:
    """Annualised Sharpe ratio.

    Args:
        returns: Per-period arithmetic return series.
        risk_free_rate: Annual risk-free rate as decimal (e.g. 0.04 = 4%).
        frequency: Periods per year (252 daily, 52 weekly, 12 monthly).

    Returns:
        Annualised Sharpe = ``(mean_excess / std_excess) × √frequency``.
        ``None`` when fewer than 2 returns or zero variance.
    """
    r = list(returns)
    if len(r) < 2:
        return None
    rf_per_period = (1 + risk_free_rate) ** (1 / frequency) - 1
    excess = [x - rf_per_period for x in r]
    mean_e = sum(excess) / len(excess)
    std_e = statistics.pstdev(excess)
    if std_e == 0:
        return None
    return (mean_e / std_e) * math.sqrt(frequency)


def compute_sortino(
    returns: Iterable[float],
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    frequency: int = 252,
) -> float | None:
    """Annualised Sortino ratio (downside-deviation variant)."""
    r = list(returns)
    if len(r) < 2:
        return None
    rf_per_period = (1 + risk_free_rate) ** (1 / frequency) - 1
    excess = [x - rf_per_period for x in r]
    mean_e = sum(excess) / len(excess)
    downside = [min(e, 0.0) ** 2 for e in excess]
    downside_var = sum(downside) / len(downside)
    if downside_var == 0:
        return None
    return (mean_e / math.sqrt(downside_var)) * math.sqrt(frequency)


def compute_max_drawdown(equity_curve: Iterable[float]) -> float | None:
    """Max drawdown from an equity-curve series.

    Returns a non-positive decimal (e.g. ``-0.18`` for an 18% drawdown),
    or ``None`` when fewer than 2 points.
    """
    eq = [v for v in equity_curve if v is not None]
    if len(eq) < 2:
        return None
    peak = eq[0]
    max_dd = 0.0
    for v in eq:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (v - peak) / peak
            if dd < max_dd:
                max_dd = dd
    return max_dd


def compute_returns_from_equity(equity_curve: Iterable[float]) -> list[float]:
    """Per-period arithmetic returns derived from an equity-curve series."""
    eq = list(equity_curve)
    returns: list[float] = []
    for i in range(1, len(eq)):
        prev = eq[i - 1]
        if prev and prev > 0:
            returns.append((eq[i] - prev) / prev)
    return returns


def compute_total_return(equity_curve: Iterable[float]) -> float | None:
    """Total return from first to last point in the curve."""
    eq = list(equity_curve)
    if len(eq) < 2 or eq[0] <= 0:
        return None
    return (eq[-1] - eq[0]) / eq[0]


def compute_benchmark_comparison(
    portfolio_curve: list[float],
    benchmark_prices: list[float],
) -> dict[str, float | None]:
    """Compare portfolio equity curve against a benchmark (e.g. SPY).

    Args:
        portfolio_curve: NAV series at each rebalance.
        benchmark_prices: Benchmark close prices at the same dates
            (caller is responsible for alignment).

    Returns:
        Dict with ``portfolio_total_return``, ``benchmark_total_return``,
        ``alpha`` (= portfolio − benchmark), and ``tracking_error``
        (annualised std of return diffs).  ``None`` for fields whose
        inputs are insufficient.
    """
    p_total = compute_total_return(portfolio_curve)
    b_total = compute_total_return(benchmark_prices)
    alpha = (p_total - b_total) if p_total is not None and b_total is not None else None

    p_returns = compute_returns_from_equity(portfolio_curve)
    b_returns = compute_returns_from_equity(benchmark_prices)
    n = min(len(p_returns), len(b_returns))
    if n >= 2:
        diffs = [p_returns[i] - b_returns[i] for i in range(n)]
        tracking_error = statistics.pstdev(diffs) * math.sqrt(252)
    else:
        tracking_error = None

    return {
        "portfolio_total_return": p_total,
        "benchmark_total_return": b_total,
        "alpha": alpha,
        "tracking_error": tracking_error,
    }
