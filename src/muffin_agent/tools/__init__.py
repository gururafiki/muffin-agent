"""Financial computation tools for investment agents.

Organized by financial domain:

- ``profitability`` — ROIC, FCF conversion, accruals ratio, revenue CAGR
- ``credit_risk`` — net debt/EBITDA, interest coverage, Altman Z-Score
- ``sector`` — sector relative performance, peer dispersion
- ``macro`` — yield curve metrics, factor Z-scores, VIX regime
- ``projections`` — 3-year financial projections, sensitivity analysis
"""

from muffin_agent.tools.credit_risk import (
    compute_altman_z_score,
    compute_interest_coverage,
    compute_net_debt_to_ebitda,
)
from muffin_agent.tools.macro import (
    compute_factor_zscore,
    compute_vix_regime,
    compute_yield_curve_metrics,
)
from muffin_agent.tools.profitability import (
    compute_accruals_ratio,
    compute_fcf_conversion,
    compute_revenue_cagr,
    compute_roic,
)
from muffin_agent.tools.projections import (
    compute_sensitivity,
    project_three_year_financials,
)
from muffin_agent.tools.sector import (
    compute_peer_dispersion,
    compute_sector_relative_performance,
)

__all__ = [
    "compute_accruals_ratio",
    "compute_altman_z_score",
    "compute_factor_zscore",
    "compute_fcf_conversion",
    "compute_interest_coverage",
    "compute_net_debt_to_ebitda",
    "compute_peer_dispersion",
    "compute_revenue_cagr",
    "compute_roic",
    "compute_sector_relative_performance",
    "compute_sensitivity",
    "compute_vix_regime",
    "compute_yield_curve_metrics",
    "project_three_year_financials",
]
