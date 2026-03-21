"""Market and macro regime tools — yield curve, factor Z-scores, VIX."""

from __future__ import annotations

import json

from langchain_core.tools import tool


@tool(parse_docstring=True)
def compute_yield_curve_metrics(
    yield_10y: float | None = None,
    yield_2y: float | None = None,
    yield_3m: float | None = None,
    tips_breakeven_10y: float | None = None,
    effr: float | None = None,
    neutral_rate: float = 2.5,
) -> str:
    """Compute yield curve slope and monetary policy metrics.

    All input yields in percent (e.g. 4.25 for 4.25%).
    Results in basis points.

    Args:
        yield_10y: 10-year Treasury yield (%).
        yield_2y: 2-year Treasury yield (%).
        yield_3m: 3-month Treasury yield (%).
        tips_breakeven_10y: 10-year TIPS breakeven inflation rate (%).
        effr: Effective Federal Funds Rate (%).
        neutral_rate: Estimated neutral policy rate (%). Default 2.5.

    Returns:
        JSON string with keys: slope_10y2y_bps, slope_10y3m_bps,
        real_yield_10y_bps, policy_rate_distance_bps.
    """
    slope_10y2y = (
        (yield_10y - yield_2y) * 100
        if yield_10y is not None and yield_2y is not None
        else None
    )
    slope_10y3m = (
        (yield_10y - yield_3m) * 100
        if yield_10y is not None and yield_3m is not None
        else None
    )
    real_yield = (
        (yield_10y - tips_breakeven_10y) * 100
        if yield_10y is not None and tips_breakeven_10y is not None
        else None
    )
    policy_dist = (effr - neutral_rate) * 100 if effr is not None else None
    return json.dumps({
        "slope_10y2y_bps": slope_10y2y,
        "slope_10y3m_bps": slope_10y3m,
        "real_yield_10y_bps": real_yield,
        "policy_rate_distance_bps": policy_dist,
    })


@tool(parse_docstring=True)
def compute_factor_zscore(
    factor_name: str,
    trailing_12m: float,
    mean_60m: float,
    std_60m: float,
) -> str:
    """Compute Z-score for a single Fama-French factor.

    Z = (trailing_12m - mean_60m) / std_60m.  |Z| > 1.5 indicates an
    unusually strong signal.

    Call once per factor (HML, SMB, MOM, RMW, CMA).  Multiple calls can
    be issued in parallel.

    Args:
        factor_name: Factor identifier (e.g. "HML", "SMB", "MOM").
        trailing_12m: Factor's trailing 12-month return.
        mean_60m: Factor's 60-month rolling mean return.
        std_60m: Factor's 60-month rolling standard deviation.
            Must be positive.

    Returns:
        JSON string with factor_name and z_score (null if std is zero).
    """
    z: float | None = None
    if std_60m > 0:
        z = (trailing_12m - mean_60m) / std_60m
    return json.dumps({"factor_name": factor_name, "z_score": z})


@tool(parse_docstring=True)
def compute_vix_regime(vix_level: float) -> str:
    """Classify VIX level into a volatility regime label.

    Args:
        vix_level: Current VIX index level.

    Returns:
        One of: "complacency" (<15), "normal" (15-20),
        "elevated" (20-30), "crisis" (>30).
    """
    if vix_level < 15:
        return "complacency"
    if vix_level < 20:
        return "normal"
    if vix_level < 30:
        return "elevated"
    return "crisis"
