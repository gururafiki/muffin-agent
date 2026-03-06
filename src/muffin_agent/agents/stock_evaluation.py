"""Stock evaluation agent.

Deep agent that orchestrates data collection subagents, validates collected
data, and produces a scored stock assessment with reasoning.
"""

from deepagents import CompiledSubAgent, create_deep_agent

from ..config import Configuration
from ..prompts import render_template
from .data_collection import (
    create_currency_commodities_data_collection_agent,
    create_discovery_screening_data_collection_agent,
    create_economy_macro_data_collection_agent,
    create_equity_estimates_data_collection_agent,
    create_equity_fundamentals_data_collection_agent,
    create_equity_ownership_data_collection_agent,
    create_equity_price_data_collection_agent,
    create_etf_index_data_collection_agent,
    create_fixed_income_data_collection_agent,
    create_news_data_collection_agent,
    create_options_data_collection_agent,
)


async def create_stock_evaluation_agent(config: Configuration):
    """Build the stock evaluation deep agent.

    Create a deep agent that delegates data collection to equity-fundamentals
    and equity-price subagents, then validates, analyzes, and scores the stock.
    """
    currency_commodities_agent = (
        await create_currency_commodities_data_collection_agent(config)
    )
    discovery_screening_agent = await create_discovery_screening_data_collection_agent(
        config
    )
    economy_macro_agent = await create_economy_macro_data_collection_agent(config)
    etf_index_agent = await create_etf_index_data_collection_agent(config)
    fixed_income_agent = await create_fixed_income_data_collection_agent(config)
    fundamentals_agent = await create_equity_fundamentals_data_collection_agent(config)
    price_agent = await create_equity_price_data_collection_agent(config)
    estimates_agent = await create_equity_estimates_data_collection_agent(config)
    ownership_agent = await create_equity_ownership_data_collection_agent(config)
    news_agent = await create_news_data_collection_agent(config)
    options_agent = await create_options_data_collection_agent(config)

    subagents = [
        CompiledSubAgent(
            name="equity-fundamentals",
            description=(
                "Retrieves fundamental financial data: income statements, "
                "balance sheets, cash flow, ratios, metrics, EPS, dividends, "
                "revenue segments, management info, ESG scores, earnings "
                "transcripts, SEC filings. Use for any fundamental analysis "
                "data needs."
            ),
            runnable=fundamentals_agent,
        ),
        CompiledSubAgent(
            name="equity-price",
            description=(
                "Retrieves stock price data: current quotes, historical OHLCV, "
                "price performance across timeframes, market cap history, "
                "bid/ask spreads. Use for any price or market data needs."
            ),
            runnable=price_agent,
        ),
        CompiledSubAgent(
            name="equity-estimates",
            description=(
                "Retrieves analyst estimates data: consensus estimates, price "
                "targets, forward EPS, forward EBITDA, forward PE, forward "
                "sales, analyst rating breakdowns. Use for forward-looking "
                "valuation and analyst sentiment data."
            ),
            runnable=estimates_agent,
        ),
        CompiledSubAgent(
            name="equity-ownership",
            description=(
                "Retrieves ownership and short interest data: major holders, "
                "institutional ownership, insider trades, share statistics, "
                "13F filings, government trades, short interest, short volume, "
                "fails-to-deliver. Use for conviction signals and short squeeze risk."
            ),
            runnable=ownership_agent,
        ),
        CompiledSubAgent(
            name="news",
            description=(
                "Retrieves news and sentiment data: recent company news articles "
                "with sentiment signals, and global/macro news headlines. Use for "
                "catalyst identification, sentiment analysis, and event-driven context."
            ),
            runnable=news_agent,
        ),
        CompiledSubAgent(
            name="options",
            description=(
                "Retrieves options data: full options chains with Greeks "
                "(delta, gamma, theta, vega, IV) and implied volatility surface. "
                "Use for options flow analysis, implied volatility assessment, "
                "and put/call ratio signals."
            ),
            runnable=options_agent,
        ),
        CompiledSubAgent(
            name="economy-macro",
            description=(
                "Retrieves macroeconomic data: GDP, CPI, unemployment, interest "
                "rates, FOMC documents, FRED series, consumer/business surveys, "
                "and shipping volumes. Use for macro environment assessment and "
                "discount rate context."
            ),
            runnable=economy_macro_agent,
        ),
        CompiledSubAgent(
            name="fixed-income",
            description=(
                "Retrieves fixed income data: interest rates (SOFR, EFFR, ECB), "
                "yield curves, Treasury rates/prices, TIPS, corporate bonds, "
                "spreads. Use for discount rate, WACC, and credit spread context."
            ),
            runnable=fixed_income_agent,
        ),
        CompiledSubAgent(
            name="etf-index",
            description=(
                "Retrieves ETF and index data: ETF info, sector/country weights, "
                "holdings, index levels, S&P 500 valuation multiples, and which "
                "ETFs hold a given stock. Use for benchmark and sector context."
            ),
            runnable=etf_index_agent,
        ),
        CompiledSubAgent(
            name="discovery-screening",
            description=(
                "Retrieves market-wide discovery and screening data: equity screener, "
                "top gainers/losers/active stocks, earnings/IPO/dividend calendars, "
                "peer comparisons, sector group valuations, company profiles, and "
                "dark pool volume. Use for peer context, upcoming catalysts, and "
                "relative market positioning."
            ),
            runnable=discovery_screening_agent,
        ),
        CompiledSubAgent(
            name="currency-commodities",
            description=(
                "Retrieves currency, commodity, and crypto data: FX rates and "
                "history, commodity spot prices (WTI, Brent, gold, copper), "
                "EIA energy outlooks, and crypto price history. Use for FX "
                "exposure, commodity input cost, and energy sector context."
            ),
            runnable=currency_commodities_agent,
        ),
    ]

    prompt = render_template("stock_evaluation.jinja")
    llm = config.get_llm()

    return create_deep_agent(
        model=llm,
        system_prompt=prompt,
        subagents=subagents
    )
