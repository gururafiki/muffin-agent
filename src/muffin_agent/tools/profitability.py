"""Profitability and growth tools — ROIC, FCF conversion, accruals, CAGR."""

from __future__ import annotations

from langchain_core.tools import tool


@tool(parse_docstring=True)
def compute_roic(
    ebit: float,
    tax_rate: float,
    equity: float,
    debt: float,
    cash: float,
) -> float | None:
    """Compute Return on Invested Capital (ROIC) as a percentage.

    ROIC = NOPAT / Invested Capital, where NOPAT = EBIT * (1 - tax_rate)
    and Invested Capital = equity + debt - cash.

    Args:
        ebit: Earnings before interest and taxes.
        tax_rate: Effective tax rate as a decimal (e.g. 0.21 for 21%).
        equity: Total shareholders' equity.
        debt: Total debt.
        cash: Cash and equivalents.

    Returns:
        ROIC as a percentage (e.g. 12.5 for 12.5%), or None if invested
        capital is zero/negative.
    """
    nopat = ebit * (1 - tax_rate)
    invested_capital = equity + debt - cash
    if invested_capital <= 0:
        return None
    return (nopat / invested_capital) * 100


@tool(parse_docstring=True)
def compute_fcf_conversion(
    fcf: float,
    net_income: float,
) -> float | None:
    """Compute Free Cash Flow conversion ratio as a percentage.

    FCF conversion = FCF / Net Income * 100.

    Args:
        fcf: Free cash flow.
        net_income: Net income. Must be positive.

    Returns:
        FCF conversion percentage, or None if net income is zero/negative.
    """
    if net_income <= 0:
        return None
    return (fcf / net_income) * 100


@tool(parse_docstring=True)
def compute_accruals_ratio(
    net_income: float,
    fcf: float,
    total_assets_current: float,
    total_assets_prior: float | None = None,
) -> float | None:
    """Compute accruals ratio for earnings quality assessment.

    Accruals ratio = (Net Income - FCF) / avg Total Assets.
    Values >0.15 suggest earnings may not be fully cash-backed.

    Args:
        net_income: Net income.
        fcf: Free cash flow.
        total_assets_current: Current period total assets.
        total_assets_prior: Prior period total assets. If not
            provided, uses current period only.

    Returns:
        Accruals ratio as a decimal, or None if average assets are
        zero/negative.
    """
    avg_assets = (
        (total_assets_current + total_assets_prior) / 2
        if total_assets_prior is not None
        else total_assets_current
    )
    if avg_assets <= 0:
        return None
    return (net_income - fcf) / avg_assets


@tool(parse_docstring=True)
def compute_revenue_cagr(
    revenue_start: float,
    revenue_end: float,
    years: int,
) -> float | None:
    """Compute revenue Compound Annual Growth Rate (CAGR) as a percentage.

    Args:
        revenue_start: Revenue at the start of the period. Must be positive.
        revenue_end: Revenue at the end of the period.
        years: Number of years in the period. Must be positive.

    Returns:
        CAGR as a percentage (e.g. 8.3 for 8.3%), or None if inputs
        are invalid.
    """
    if revenue_start <= 0 or years <= 0:
        return None
    return ((revenue_end / revenue_start) ** (1 / years) - 1) * 100
