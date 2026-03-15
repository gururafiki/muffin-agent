"""Stage 2: Market Regime & Top-Down Context."""

from typing import Any

from langchain_core.runnables import RunnableConfig

from muffin_agent.agents.investment.state import TickerAnalysisState


async def market_regime_node(
    state: TickerAnalysisState, config: RunnableConfig
) -> dict[str, Any]:
    """Stage 2: Market Regime & Top-Down Context.

    Classifies the current macro and liquidity regime, identifies factor
    tailwinds and headwinds, and sets the top-down frame within which the
    individual company analysis will be interpreted.

    Runs in **parallel** with ``sector_analysis_node`` and
    ``company_analysis_node`` (Group 1).  Its output flows into
    ``forecasting_node`` and ``risk_assessment_node`` (both Group 2) as
    contextual input for macro assumptions and stress-scenario design.

    In ``screening_graph`` this node runs **once** on the outer graph before
    the per-ticker fan-out, so the regime assessment is shared across all
    candidates rather than recomputed per ticker.

    Inputs (from state):
        query: Investment mandate (used to focus regime assessment on
               relevant geographies / factor styles).

    Outputs (state update):
        market_regime: dict containing, e.g.:
            - regime_label: str — e.g. "Late-cycle expansion",
              "Risk-off / flight-to-quality", "Reflation"
            - factor_assessment: dict — which Fama-French factors are
              trending (value vs. growth tilt, size, momentum, quality)
            - yield_curve: dict — curve shape, inversion depth, credit spreads
            - macro_summary: str — 2–3 sentence narrative
            - key_risks: list[str] — top macro tail risks
            - recommended_positioning: dict — beta range, gross/net guidance

    Planned agent type:
        Deep agent (``create_deep_agent``) with 4–5 data collection subagents.

    Data collection subagents:
        - ``economy-macro`` ⭐ primary
            economy_cpi, economy_gdp_real, economy_interest_rates,
            economy_indicators, economy_survey_university_of_michigan,
            economy_survey_sloos, economy_fomc_documents,
            economy_unemployment, economy_fred_series,
            economy_composite_leading_indicator, economy_risk_premium
        - ``fixed-income`` ⭐ primary
            fixedincome_government_yield_curve, fixedincome_rate_effr,
            fixedincome_rate_sofr, fixedincome_spreads_tcm,
            fixedincome_corporate_spot_rates, fixedincome_rate_effr_forecast
        - ``fama-french``
            famafrench_factors, famafrench_us_portfolio_returns,
            famafrench_breakpoints (factor-regime classification)
        - ``currency-commodities``
            commodity_price_spot (oil, gold as economic health signals),
            currency_snapshots
        - ``etf-index``
            index_sp500_multiples, etf_price_performance
            (market-level valuation and breadth)
    """
    raise NotImplementedError("market_regime_node not yet implemented")
