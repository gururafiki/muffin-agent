"""Stage 8: Risk & Downside / Stress Testing."""

from typing import Any

from langchain_core.runnables import RunnableConfig

from muffin_agent.pipeline.state import TickerAnalysisState


async def risk_assessment_node(
    state: TickerAnalysisState, config: RunnableConfig
) -> dict[str, Any]:
    """Stage 8: Risk & Downside / Stress Testing.

    Quantifies idiosyncratic, systematic, and liquidity risk.  Calculates
    factor loadings (beta, size, value), implied-volatility-based tail risk,
    short-interest crowding risk, and stress-test scenarios tied to the macro
    regime from ``market_regime``.

    Runs in **parallel** with ``forecasting_node`` (Group 2) after Group 1
    completes.  Gated by ``company_analysis`` (leverage / earnings volatility
    baseline) and ``market_regime`` (macro stress scenarios).

    Its output flows into ``valuation_node`` (Group 3) to inform the discount
    rate and downside price target.

    Inputs (from state):
        ticker: Equity ticker symbol.
        company_analysis: Leverage, earnings volatility, business-risk baseline.
        market_regime: Macro stress-scenario parameters.

    Outputs (state update):
        risk_assessment: dict containing, e.g.:
            - beta: float — market beta (trailing 2-year weekly)
            - factor_loadings: dict — size, value, momentum, quality exposures
            - annualised_vol: float
            - max_drawdown_1y: float
            - var_95_1m: float — 95% 1-month VaR
            - implied_vol: dict — 30d, 60d, 90d ATM IV from options surface
            - put_call_skew: float — 25-delta risk reversal
            - short_interest_pct: float
            - days_to_cover: float
            - crowding_risk: str — low / moderate / high
            - stress_scenarios: list[dict] — named scenarios with ΔP/L estimates
            - ex_ante_stop_level: float | None — suggested stop-loss price
            - risk_flags: list[str]

    Planned agent type:
        Deep agent (``create_deep_agent``) with 4–5 data collection subagents
        and a ``SandboxFactory`` backend (for Python-based risk calculations:
        beta regression, VaR, Sharpe, Sortino, Fama-French regressions).

    Data collection subagents:
        - ``equity-price`` ⭐ primary
            equity_price_historical + execute_python
            (price history → beta, vol, max drawdown, VaR, Sharpe, Sortino)
        - ``options`` ⭐ primary
            derivatives_options_chains, derivatives_options_surface
            (IV term structure, 25-delta skew, market-implied tail risk)
        - ``fama-french`` ⭐ primary
            famafrench_factors, famafrench_us_portfolio_returns,
            famafrench_breakpoints
            (factor loading regression, systematic risk attribution)
        - ``equity-ownership``
            equity_shorts_short_interest, equity_shorts_short_volume,
            equity_shorts_fails_to_deliver
            (crowding, short-squeeze probability)
        - ``fixed-income``
            fixedincome_spreads_tcm
            (credit-spread environment for stress scenario calibration)
        - ``economy-macro``
            economy_risk_premium, economy_composite_leading_indicator
            (macro-scenario parameters, equity-risk-premium estimate)
    """
    raise NotImplementedError("risk_assessment_node not yet implemented")
