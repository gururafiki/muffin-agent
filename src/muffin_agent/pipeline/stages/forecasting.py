"""Stage 6: Forecasting & Scenario Modeling."""

from typing import Any

from langchain_core.runnables import RunnableConfig

from muffin_agent.pipeline.state import TickerAnalysisState


async def forecasting_node(
    state: TickerAnalysisState, config: RunnableConfig
) -> dict[str, Any]:
    """Stage 6: Forecasting & Scenario Modeling.

    Builds a forward 3-statement financial model anchored to analyst consensus
    and historical trends from ``company_analysis``.  Produces bull / base /
    bear scenarios with explicit revenue, margin, and EPS assumptions.

    Runs in **parallel** with ``risk_assessment_node`` (Group 2) after Group 1
    completes.  This node is gated by ``company_analysis`` (needs historical
    financials) and ``market_regime`` (needs macro assumptions for WACC and
    top-down revenue growth anchors).

    Inputs (from state):
        ticker: Equity ticker symbol.
        query: Investment mandate.
        company_analysis: Historical 3-statement data and quality assessment.
        market_regime: Macro assumptions (GDP growth, rates, inflation).

    Outputs (state update):
        forecast: dict containing, e.g.:
            - base_case: dict — revenue, EBITDA, EPS, FCF for next 3 years
            - bull_case: dict — optimistic scenario with key driver changes
            - bear_case: dict — downside scenario with key risk triggers
            - consensus_anchoring: dict — estimate revision trend, surprise history
            - key_assumptions: list[dict] — sensitivity drivers (price/volume,
              margins, capex, working capital)
            - revision_momentum: str — upward / flat / downward
            - earnings_quality_flags: list[str]

    Planned agent type:
        Deep agent (``create_deep_agent``) with 3 data collection subagents
        and a ``SandboxFactory`` backend (for Python-based scenario modeling).

    Data collection subagents:
        - ``equity-estimates`` ⭐ primary (all 8 tools)
            equity_estimates_consensus, equity_estimates_forward_ebitda,
            equity_estimates_forward_eps, equity_estimates_forward_pe,
            equity_estimates_forward_sales, equity_estimates_historical,
            equity_estimates_price_target, equity_estimates_price_target_consensus
        - ``equity-fundamentals``
            equity_fundamental_income, equity_fundamental_cash,
            equity_fundamental_balance, equity_fundamental_historical_eps
            (historical baseline for model calibration)
        - ``economy-macro``
            economy_gdp_forecast, economy_cpi, economy_interest_rates
            (macro inputs for revenue growth and WACC assumptions)
        - ``currency-commodities``
            commodity_short_term_energy_outlook, currency_price_historical
            (FX / commodity cost assumptions for exposed companies)
    """
    raise NotImplementedError("forecasting_node not yet implemented")
