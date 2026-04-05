"""Financial computation tools for investment agents.

Organized by financial domain:

- ``profitability`` — ROIC, FCF conversion, accruals ratio, revenue CAGR
- ``credit_risk`` — net debt/EBITDA, interest coverage, Altman Z-Score
- ``sector`` — sector relative performance, peer dispersion
- ``macro`` — yield curve metrics, factor Z-scores, VIX regime
- ``projections`` — 3-year financial projections, sensitivity analysis
- ``risk`` — beta, VaR/CVaR, Sharpe/Sortino, max drawdown
- ``valuation`` — WACC, DCF (blended exit-multiple + Gordon Growth),
  multiples-based fair value, scenario-weighted NAV
- ``web`` — convert_document (MarkItDown); web search via LangChain SearxNG
  integration; scraping/crawling via Firecrawl MCP
"""

from .credit_risk import (
    compute_altman_z_score,
    compute_interest_coverage,
    compute_net_debt_to_ebitda,
)
from .macro import (
    compute_factor_zscore,
    compute_vix_regime,
    compute_yield_curve_metrics,
)
from .profitability import (
    compute_accruals_ratio,
    compute_fcf_conversion,
    compute_revenue_cagr,
    compute_roic,
)
from .projections import (
    compute_sensitivity,
    project_three_year_financials,
)
from .risk import (
    compute_beta,
    compute_max_drawdown,
    compute_sharpe_sortino,
    compute_var_cvar,
)
from .sector import (
    compute_peer_dispersion,
    compute_sector_relative_performance,
)
from .valuation import (
    compute_dcf,
    compute_multiples_value,
    compute_scenario_weighted_value,
    compute_wacc,
)
from .web import convert_document

__all__ = [
    "compute_accruals_ratio",
    "compute_altman_z_score",
    "compute_beta",
    "compute_dcf",
    "compute_factor_zscore",
    "compute_fcf_conversion",
    "compute_interest_coverage",
    "compute_max_drawdown",
    "compute_multiples_value",
    "compute_net_debt_to_ebitda",
    "compute_peer_dispersion",
    "compute_revenue_cagr",
    "compute_roic",
    "compute_scenario_weighted_value",
    "compute_sector_relative_performance",
    "compute_sensitivity",
    "compute_sharpe_sortino",
    "compute_var_cvar",
    "compute_vix_regime",
    "compute_wacc",
    "compute_yield_curve_metrics",
    "convert_document",
    "project_three_year_financials",
]
