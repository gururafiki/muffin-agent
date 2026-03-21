"""Credit risk tools — debt ratios, interest coverage, Altman Z-Score."""

from __future__ import annotations

from langchain_core.tools import tool


@tool(parse_docstring=True)
def compute_net_debt_to_ebitda(
    debt: float,
    cash: float,
    ebitda: float,
) -> float | None:
    """Compute Net Debt / EBITDA leverage ratio.

    Args:
        debt: Total debt.
        cash: Cash and equivalents.
        ebitda: Earnings before interest, taxes, depreciation, and
            amortization. Must be positive.

    Returns:
        Net Debt / EBITDA ratio (e.g. 2.5x), or None if EBITDA is
        zero/negative.
    """
    if ebitda <= 0:
        return None
    return (debt - cash) / ebitda


@tool(parse_docstring=True)
def compute_interest_coverage(
    ebit: float,
    interest_expense: float,
) -> float | None:
    """Compute interest coverage ratio (EBIT / Interest Expense).

    Args:
        ebit: Earnings before interest and taxes.
        interest_expense: Annual interest expense. Must be positive.

    Returns:
        Interest coverage ratio (e.g. 8.5x), or None if interest expense
        is zero/negative.
    """
    if interest_expense <= 0:
        return None
    return ebit / interest_expense


@tool(parse_docstring=True)
def compute_altman_z_score(
    working_capital: float,
    retained_earnings: float,
    ebit: float,
    market_cap: float,
    total_liabilities: float,
    total_assets: float,
    revenue: float,
) -> float | None:
    """Compute Altman Z-Score for financial distress prediction.

    Interpretation: >2.99 = safe zone; 1.81-2.99 = grey zone;
    <1.81 = distress zone.

    Args:
        working_capital: Current assets minus current liabilities.
        retained_earnings: Cumulative retained earnings.
        ebit: Earnings before interest and taxes.
        market_cap: Market capitalization.
        total_liabilities: Total liabilities. Must be positive.
        total_assets: Total assets. Must be positive.
        revenue: Total revenue.

    Returns:
        Altman Z-Score, or None if total assets or liabilities are
        zero/negative.
    """
    if total_assets <= 0 or total_liabilities <= 0:
        return None
    return (
        1.2 * (working_capital / total_assets)
        + 1.4 * (retained_earnings / total_assets)
        + 3.3 * (ebit / total_assets)
        + 0.6 * (market_cap / total_liabilities)
        + 1.0 * (revenue / total_assets)
    )
