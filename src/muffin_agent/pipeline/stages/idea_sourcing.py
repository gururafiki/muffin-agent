"""Stage 1: Idea Sourcing & Screening."""

from typing import Any

from langchain_core.runnables import RunnableConfig

from muffin_agent.pipeline.state import PipelineState


async def idea_sourcing_node(
    state: PipelineState, config: RunnableConfig
) -> dict[str, Any]:
    """Stage 1: Idea Sourcing & Screening.

    Screens the market to produce a ranked list of candidate tickers that
    match the investment mandate expressed in ``state["query"]``.

    This is the *only* stage exclusive to ``screening_graph``; the
    ``analysis_graph`` bypasses it entirely (the caller provides a ticker).
    Replace this node to plug in an entirely different sourcing strategy
    (e.g. country → sector drill-down, factor tilt, thematic basket, etc.).

    Inputs (from state):
        query: Investment mandate, e.g. "undervalued small-cap growth stocks
               in emerging markets", "daily gainers with improving earnings
               revisions", or "US semiconductors with wide moat".

    Outputs (state update):
        tickers: Ordered list of candidate ticker symbols (most attractive
                 first), e.g. ``["NVDA", "AMD", "AVGO"]``.

    Planned agent type:
        Deep agent (``create_deep_agent``) orchestrating 3–4 data collection
        subagents with a custom screening prompt.

    Data collection subagents:
        - ``discovery-screening`` ⭐ primary
            equity_screener, equity_discovery_gainers, equity_discovery_losers,
            equity_discovery_active, equity_discovery_undervalued_growth,
            equity_discovery_undervalued_large_caps,
            equity_discovery_growth_tech, equity_discovery_aggressive_small_caps,
            equity_compare_peers, equity_profile, equity_search
        - ``etf-index``
            etf_sectors, index_sp500_multiples, etf_price_performance
            (sector-level filtering and relative performance context)
        - ``economy-macro``
            economy_indicators, economy_country_profile
            (for country-aware or macro-filter screens)
        - ``news``
            news_world (macro headlines to surface thematic ideas)

    Example replacement strategies (plug-in alternatives):
        - ``idea_sourcing_country_sector_node``: screens by country GDP growth
          → selects best sector → screens for stocks within that sector.
        - ``idea_sourcing_factor_node``: uses Fama-French factor returns to
          tilt the screen toward value, momentum, or quality factors.
        - ``idea_sourcing_watchlist_node``: statically returns a pre-defined
          watchlist (useful for testing downstream stages).
    """
    raise NotImplementedError("idea_sourcing_node not yet implemented")
