"""Equity risk tools — beta, VaR/CVaR, Sharpe/Sortino, max drawdown."""

from __future__ import annotations

import math
from statistics import NormalDist as _NormalDist

from langchain_core.tools import tool
from pydantic import BaseModel

_ndist = _NormalDist()  # standard normal N(0, 1) — used for VaR/CVaR


# ── Helper arithmetic ─────────────────────────────────────────────────────────


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _sample_var(xs: list[float]) -> float:
    """Sample variance (divide by n-1)."""
    mx = _mean(xs)
    return sum((x - mx) ** 2 for x in xs) / (len(xs) - 1)


def _sample_cov(xs: list[float], ys: list[float]) -> float:
    """Sample covariance of two aligned series (divide by n-1)."""
    n = len(xs)
    mx, my = _mean(xs), _mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)


# ── Tool 1: compute_beta ──────────────────────────────────────────────────────


class BetaMetrics(BaseModel):
    """Output schema for compute_beta."""

    beta: float | None
    """OLS slope from stock ~ market regression."""

    alpha_annualized: float | None
    """Jensen's alpha annualised to yearly frequency (decimal)."""

    r_squared: float | None
    """Coefficient of determination (0.0–1.0)."""


@tool(
    parse_docstring=True,
    extras={"output_schema": BetaMetrics.model_json_schema()},
)
def compute_beta(
    returns: list[float],
    market_returns: list[float],
    frequency: str = "weekly",
) -> dict[str, float | None]:
    """Compute OLS beta, annualised alpha, and R-squared vs a market return series.

    Perform an ordinary-least-squares regression of stock returns on market
    returns to extract the CAPM beta, Jensen's alpha (annualised), and the
    coefficient of determination (R²).

    Args:
        returns: Stock arithmetic return series as decimals (e.g. 0.01 = 1%).
            Must be aligned to *market_returns* (same dates, same length).
        market_returns: Market index return series as decimals, aligned to
            *returns*. Same length required.
        frequency: Return frequency used to annualise alpha. One of
            ``"daily"`` (252 periods/year) or ``"weekly"`` (52 periods/year).
            Does not affect beta or R-squared.

    Returns:
        Dict with beta (OLS slope), alpha_annualized (Jensen's alpha scaled
        to yearly frequency, decimal), r_squared (0–1).  All values are None
        when fewer than two aligned observations are provided or when market
        variance is zero.
    """
    n = min(len(returns), len(market_returns))
    if n < 2:
        return BetaMetrics(
            beta=None, alpha_annualized=None, r_squared=None
        ).model_dump()

    r = returns[:n]
    m = market_returns[:n]

    var_m = _sample_var(m)
    if var_m == 0:
        return BetaMetrics(
            beta=None, alpha_annualized=None, r_squared=None
        ).model_dump()

    cov_rm = _sample_cov(r, m)
    beta = cov_rm / var_m
    alpha_per_period = _mean(r) - beta * _mean(m)

    annualization = 252 if frequency == "daily" else 52
    alpha_annualized = alpha_per_period * annualization

    var_r = _sample_var(r)
    if var_r > 0 and var_m > 0:
        corr = cov_rm / math.sqrt(var_r * var_m)
        r_squared: float | None = min(corr**2, 1.0)
    else:
        r_squared = None

    return BetaMetrics(
        beta=beta,
        alpha_annualized=alpha_annualized,
        r_squared=r_squared,
    ).model_dump()


# ── Tool 2: compute_var_cvar ──────────────────────────────────────────────────


class VaRResult(BaseModel):
    """Output schema for compute_var_cvar."""

    var_pct: float | None
    """Parametric VaR as a positive percentage (e.g. 8.5 means an 8.5% loss)."""

    var_dollar: float | None
    """VaR in dollars per share (current_price × var_pct / 100)."""

    cvar_pct: float | None
    """Expected Shortfall (CVaR) as a positive percentage; always ≥ var_pct."""

    cvar_dollar: float | None
    """CVaR in dollars per share."""


@tool(
    parse_docstring=True,
    extras={"output_schema": VaRResult.model_json_schema()},
)
def compute_var_cvar(
    annualized_vol_pct: float,
    current_price: float,
    confidence: float = 0.95,
    horizon_months: float = 1.0,
) -> dict[str, float | None]:
    """Compute parametric Value at Risk and Expected Shortfall (CVaR).

    Assumes returns are normally distributed with zero drift.  Both metrics
    are expressed as positive numbers representing the magnitude of loss.

    Args:
        annualized_vol_pct: Annualised volatility in percent (e.g. 25.0 for 25%).
            Obtained from ``compute_max_drawdown`` output or Block A sandbox code.
        current_price: Current stock price in reporting currency.
        confidence: Confidence level as a decimal (e.g. 0.95 for 95% VaR).
            Must be in (0.5, 1.0).
        horizon_months: Forecast horizon in months (e.g. 1.0 for 1-month VaR).
            Volatility is scaled by sqrt(horizon_months / 12).

    Returns:
        Dict with var_pct, var_dollar, cvar_pct, cvar_dollar.  All None if
        annualized_vol_pct or current_price are non-positive.
    """
    if annualized_vol_pct <= 0 or current_price <= 0:
        return VaRResult(
            var_pct=None, var_dollar=None, cvar_pct=None, cvar_dollar=None
        ).model_dump()

    sigma_annual = annualized_vol_pct / 100.0
    sigma_h = sigma_annual * math.sqrt(horizon_months / 12.0)

    z = _ndist.inv_cdf(confidence)

    # Standard normal PDF: φ(z) = exp(-z² / 2) / sqrt(2π)
    phi_z = _ndist.pdf(z)

    var_pct = z * sigma_h * 100
    cvar_pct = sigma_h * phi_z / (1 - confidence) * 100
    var_dollar = current_price * var_pct / 100
    cvar_dollar = current_price * cvar_pct / 100

    return VaRResult(
        var_pct=round(var_pct, 4),
        var_dollar=round(var_dollar, 4),
        cvar_pct=round(cvar_pct, 4),
        cvar_dollar=round(cvar_dollar, 4),
    ).model_dump()


# ── Tool 3: compute_sharpe_sortino ────────────────────────────────────────────


class RiskAdjustedReturns(BaseModel):
    """Output schema for compute_sharpe_sortino."""

    sharpe_ratio: float | None
    """Annualised Sharpe ratio = annualised excess return /
    annualised total-return std."""

    sortino_ratio: float | None
    """Annualised Sortino ratio = annualised excess return /
    annualised downside deviation.
    None when all returns are at or above the risk-free rate (no downside).
    """


@tool(
    parse_docstring=True,
    extras={"output_schema": RiskAdjustedReturns.model_json_schema()},
)
def compute_sharpe_sortino(
    returns: list[float],
    risk_free_rate_annual: float,
    frequency: str = "weekly",
) -> dict[str, float | None]:
    """Compute Sharpe and Sortino ratios from a return series.

    Both ratios are annualised.  Sortino uses downside deviation (returns below
    the risk-free rate) instead of total standard deviation.

    Args:
        returns: Arithmetic return series as decimals (e.g. 0.01 = 1%).
            At least two observations required.
        risk_free_rate_annual: Annualised risk-free rate as a decimal
            (e.g. 0.05 for 5%).  Converted to per-period rate internally.
        frequency: Return frequency for annualisation.  One of ``"daily"``
            (252 periods/year) or ``"weekly"`` (52 periods/year).

    Returns:
        Dict with sharpe_ratio and sortino_ratio.  Both None when fewer than
        two observations are provided.  sortino_ratio is None when all excess
        returns are non-negative (no downside deviation).
    """
    if len(returns) < 2:
        return RiskAdjustedReturns(sharpe_ratio=None, sortino_ratio=None).model_dump()

    annualization = 252 if frequency == "daily" else 52
    rf_per_period = (1 + risk_free_rate_annual) ** (1.0 / annualization) - 1

    excess = [r - rf_per_period for r in returns]
    mean_excess = _mean(excess)

    # Sharpe: std of total returns (rf constant, so std(r - rf) = std(r))
    var_r = _sample_var(returns)
    if var_r > 0:
        sharpe: float | None = (mean_excess / math.sqrt(var_r)) * math.sqrt(
            annualization
        )
    else:
        sharpe = None

    # Sortino: downside deviation (mean-squared negative excess returns)
    # Divide by N (not N-1) — standard Sortino convention
    downside_sq = [min(e, 0.0) ** 2 for e in excess]
    downside_var = sum(downside_sq) / len(downside_sq)
    if downside_var > 0:
        sortino: float | None = (mean_excess / math.sqrt(downside_var)) * math.sqrt(
            annualization
        )
    else:
        sortino = None

    return RiskAdjustedReturns(
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
    ).model_dump()


# ── Tool 4: compute_max_drawdown ──────────────────────────────────────────────


@tool(parse_docstring=True)
def compute_max_drawdown(prices: list[float]) -> float | None:
    """Compute the maximum drawdown from a price series.

    Maximum drawdown is the largest peak-to-trough decline observed in the
    series, expressed as a negative fraction (e.g. -0.35 for -35%).

    Args:
        prices: Ordered price series from oldest to newest.  At least two
            prices required.  All prices must be positive.

    Returns:
        Maximum drawdown as a negative decimal (e.g. -0.35).  Returns 0.0 if
        prices are strictly non-decreasing (no drawdown).  Returns None if
        fewer than two prices are provided.
    """
    if len(prices) < 2:
        return None

    peak = prices[0]
    max_dd = 0.0
    for price in prices:
        if price > peak:
            peak = price
        if peak > 0:
            dd = (price - peak) / peak
            if dd < max_dd:
                max_dd = dd
    return max_dd
