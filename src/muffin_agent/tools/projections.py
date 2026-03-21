"""Financial modeling tools — 3-year projections and sensitivity analysis."""

from __future__ import annotations

import json

from langchain_core.tools import tool


@tool(parse_docstring=True)
def project_three_year_financials(
    baseline_revenue: float,
    base_year: int,
    tax_rate: float,
    da_pct_rev: float,
    rev_growth_y1: float,
    rev_growth_y2: float,
    rev_growth_y3: float,
    ebitda_margin_y1: float,
    ebitda_margin_y2: float,
    ebitda_margin_y3: float,
    capex_rate_y1: float,
    capex_rate_y2: float,
    capex_rate_y3: float,
    total_debt_0: float,
    cash_0: float,
    equity_0: float,
    fixed_assets_0: float,
    nwc_pct_rev: float,
    diluted_shares: float | None = None,
    net_borrowing: float = 0.0,
    dividends: float = 0.0,
    buybacks: float = 0.0,
) -> str:
    """Project 3-year income statement, FCF, and balance sheet for one scenario.

    Call once per scenario (bull, base, bear).  All 3 calls can be issued
    in parallel.  All rates and margins are decimals (e.g. 0.08 for 8%).

    Args:
        baseline_revenue: Last known annual revenue.
        base_year: Last fiscal year (e.g. 2025).
        tax_rate: Effective tax rate (decimal).
        da_pct_rev: Depreciation & amortization as fraction of revenue.
        rev_growth_y1: Year+1 revenue growth rate (decimal).
        rev_growth_y2: Year+2 revenue growth rate (decimal).
        rev_growth_y3: Year+3 revenue growth rate (decimal).
        ebitda_margin_y1: Year+1 EBITDA margin (decimal).
        ebitda_margin_y2: Year+2 EBITDA margin (decimal).
        ebitda_margin_y3: Year+3 EBITDA margin (decimal).
        capex_rate_y1: Year+1 capex/revenue ratio (decimal).
        capex_rate_y2: Year+2 capex/revenue ratio (decimal).
        capex_rate_y3: Year+3 capex/revenue ratio (decimal).
        total_debt_0: Starting total debt.
        cash_0: Starting cash.
        equity_0: Starting shareholders' equity.
        fixed_assets_0: Starting PP&E / fixed assets.
        nwc_pct_rev: Net working capital as fraction of revenue.
        diluted_shares: Diluted share count. None means EPS will be null.
        net_borrowing: Annual change in debt. 0 = status quo.
        dividends: Annual dividend outflow.
        buybacks: Annual share repurchase spend.

    Returns:
        JSON string: list of 3 yearly projections, each with income
        statement (revenue, ebitda, ebit, eps, fcf, margins) and
        balance sheet (total_debt, cash, net_debt, working_capital,
        total_assets, shareholders_equity).
    """
    rev_growth_rates = [rev_growth_y1, rev_growth_y2, rev_growth_y3]
    ebitda_margins = [ebitda_margin_y1, ebitda_margin_y2, ebitda_margin_y3]
    capex_rates = [capex_rate_y1, capex_rate_y2, capex_rate_y3]

    # --- Income statement + FCF projections ---
    projections: list[dict] = []
    revenue = baseline_revenue
    for i, (g, m, c) in enumerate(
        zip(rev_growth_rates, ebitda_margins, capex_rates)
    ):
        year = base_year + i + 1
        revenue = revenue * (1 + g)
        ebitda = revenue * m
        da = revenue * da_pct_rev
        ebit = ebitda - da
        net_income = ebit * (1 - tax_rate)
        eps = net_income / diluted_shares if diluted_shares else None
        capex = revenue * c
        fcf = net_income + da - capex
        projections.append({
            "year": year,
            "revenue": revenue,
            "revenue_growth_pct": g * 100,
            "ebitda": ebitda,
            "ebitda_margin_pct": m * 100,
            "ebit": ebit,
            "ebit_margin_pct": (ebit / revenue) * 100 if revenue else None,
            "eps": eps,
            "fcf": fcf,
            "fcf_margin_pct": (fcf / revenue) * 100 if revenue else None,
        })

    # --- Balance sheet projections ---
    total_debt = total_debt_0
    cash = cash_0
    equity = equity_0
    fixed_assets = fixed_assets_0

    for i, proj in enumerate(projections):
        net_income = proj["ebit"] * (1 - tax_rate)
        fcf = proj["fcf"]

        total_debt = total_debt + net_borrowing
        cash = cash + fcf - dividends - buybacks + net_borrowing
        if cash < 0:
            cash = 0

        working_capital = proj["revenue"] * nwc_pct_rev
        equity = equity + net_income - dividends - buybacks

        fixed_assets = (
            fixed_assets
            + proj["revenue"] * capex_rates[i]
            - proj["revenue"] * da_pct_rev
        )
        total_assets = working_capital + fixed_assets + cash

        proj.update({
            "total_debt": total_debt,
            "cash": cash,
            "net_debt": total_debt - cash,
            "working_capital": working_capital,
            "total_assets": total_assets,
            "shareholders_equity": equity,
        })

    return json.dumps(projections)


@tool(parse_docstring=True)
def compute_sensitivity(
    baseline_revenue: float,
    ebit_margin: float,
    tax_rate: float,
    diluted_shares: float,
    capex: float,
) -> str:
    """Compute EPS and FCF sensitivity to assumption changes.

    Measure how much EPS changes per +1pp revenue growth, per +1pp
    margin improvement, and how much FCF changes per +10% capex swing.

    Args:
        baseline_revenue: Annual revenue.
        ebit_margin: EBIT margin as a decimal (e.g. 0.15 for 15%).
        tax_rate: Effective tax rate as a decimal (e.g. 0.21).
        diluted_shares: Diluted share count.
        capex: Annual capital expenditure.

    Returns:
        JSON string with keys: delta_eps_per_rev_1pp,
        delta_eps_per_margin_1pp, delta_fcf_per_capex_10pct.
    """
    if not diluted_shares or not baseline_revenue:
        return json.dumps({
            "delta_eps_per_rev_1pp": None,
            "delta_eps_per_margin_1pp": None,
            "delta_fcf_per_capex_10pct": None,
        })
    return json.dumps({
        "delta_eps_per_rev_1pp": (
            ebit_margin * baseline_revenue * 0.01 * (1 - tax_rate)
            / diluted_shares
        ),
        "delta_eps_per_margin_1pp": (
            baseline_revenue * 0.01 * (1 - tax_rate) / diluted_shares
        ),
        "delta_fcf_per_capex_10pct": capex * 0.10,
    })
