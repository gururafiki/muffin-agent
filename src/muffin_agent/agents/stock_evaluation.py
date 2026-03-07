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
    create_fama_french_data_collection_agent,
    create_fixed_income_data_collection_agent,
    create_news_data_collection_agent,
    create_options_data_collection_agent,
    create_regulatory_filings_data_collection_agent,
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
    fama_french_agent = await create_fama_french_data_collection_agent(config)
    ownership_agent = await create_equity_ownership_data_collection_agent(config)
    news_agent = await create_news_data_collection_agent(config)
    options_agent = await create_options_data_collection_agent(config)
    regulatory_filings_agent = await create_regulatory_filings_data_collection_agent(
        config
    )

    subagents = [
        CompiledSubAgent(
            name="equity-fundamentals",
            description=(
                "Retrieves fundamental financial data: income statements, "
                "balance sheets, cash flow statements, financial ratios "
                "(ROE, ROIC, D/E, current ratio), per-share metrics (EPS, "
                "dividends), revenue segments, management info, ESG scores, "
                "earnings transcripts, and SEC filings. Use for profitability "
                "analysis, balance sheet health, cash generation assessment, "
                "and Quality dimension scoring."
            ),
            runnable=fundamentals_agent,
        ),
        CompiledSubAgent(
            name="equity-price",
            description=(
                "Retrieves stock price data: current quotes, historical daily "
                "OHLCV prices, price performance across timeframes (1D to 5Y), "
                "market cap history, and bid/ask spreads. Use for current "
                "valuation multiples, trend analysis, historical price context, "
                "and Momentum dimension scoring."
            ),
            runnable=price_agent,
        ),
        CompiledSubAgent(
            name="equity-estimates",
            description=(
                "Retrieves analyst estimates and forward-looking data: "
                "consensus EPS/revenue estimates, price targets (mean, high, "
                "low), forward P/E, forward EV/EBITDA, forward sales, analyst "
                "rating breakdowns (buy/hold/sell), and estimate revision "
                "history. Use for Growth projections, Valuation vs forward "
                "multiples, and Catalyst scoring (estimate revisions signal)."
            ),
            runnable=estimates_agent,
        ),
        CompiledSubAgent(
            name="equity-ownership",
            description=(
                "Retrieves ownership and short interest data: major holders, "
                "institutional ownership changes, insider trades (buys/sells), "
                "share statistics (float, shares outstanding), 13F filings, "
                "government trades, short interest, short volume, and "
                "fails-to-deliver. Use for insider conviction signals, "
                "institutional sentiment, short squeeze risk assessment, "
                "and Catalyst dimension scoring."
            ),
            runnable=ownership_agent,
        ),
        CompiledSubAgent(
            name="news",
            description=(
                "Retrieves news and sentiment data: recent company-specific "
                "news articles with sentiment signals (bullish/bearish/neutral), "
                "and global/macro news headlines. Use for catalyst identification, "
                "event-driven context (earnings surprises, M&A, regulatory "
                "actions), market sentiment assessment, and Catalyst dimension "
                "scoring."
            ),
            runnable=news_agent,
        ),
        CompiledSubAgent(
            name="options",
            description=(
                "Retrieves options market data: full options chains with Greeks "
                "(delta, gamma, theta, vega, IV) and implied volatility surface "
                "across expirations. Use for implied volatility assessment, "
                "put/call ratio signals, options flow analysis, and Risk "
                "dimension scoring (market-implied risk)."
            ),
            runnable=options_agent,
        ),
        CompiledSubAgent(
            name="economy-macro",
            description=(
                "Retrieves macroeconomic data: GDP growth, CPI/inflation, "
                "unemployment, interest rates, FOMC documents and minutes, "
                "FRED economic series, consumer/business confidence surveys, "
                "and shipping volumes. Use for macro environment assessment, "
                "cyclical positioning, discount rate context, and Risk "
                "dimension scoring (macro headwinds/tailwinds)."
            ),
            runnable=economy_macro_agent,
        ),
        CompiledSubAgent(
            name="fixed-income",
            description=(
                "Retrieves fixed income and rates data: benchmark rates (SOFR, "
                "EFFR, ECB), Treasury yield curves, Treasury rates/prices, "
                "TIPS breakevens, corporate bond yields, and credit spreads "
                "(IG/HY). Use for discount rate derivation, WACC calculation, "
                "credit spread context, and Valuation dimension scoring "
                "(risk-free rate, cost of capital)."
            ),
            runnable=fixed_income_agent,
        ),
        CompiledSubAgent(
            name="etf-index",
            description=(
                "Retrieves ETF and index data: ETF profiles, sector/country "
                "weights, holdings, index levels (S&P 500, Nasdaq, etc.), "
                "S&P 500 valuation multiples, and which ETFs hold a given "
                "stock. Use for benchmark comparisons, sector context, "
                "relative valuation, and Valuation dimension scoring "
                "(market-level multiples)."
            ),
            runnable=etf_index_agent,
        ),
        CompiledSubAgent(
            name="discovery-screening",
            description=(
                "Retrieves market-wide discovery and screening data: equity "
                "screener, top gainers/losers/active stocks, peer comparisons "
                "with financial metrics, sector group valuations, company "
                "profiles, earnings/IPO/dividend calendars, and dark pool "
                "volume. Use for peer-relative analysis, upcoming catalyst "
                "identification, and Valuation dimension scoring (sector "
                "and peer multiples)."
            ),
            runnable=discovery_screening_agent,
        ),
        CompiledSubAgent(
            name="currency-commodities",
            description=(
                "Retrieves currency, commodity, and crypto data: FX rates "
                "and history (major/emerging pairs), commodity spot prices "
                "(WTI, Brent, gold, copper, natural gas), EIA energy outlooks, "
                "and crypto price history. Use for FX exposure assessment, "
                "commodity input cost analysis, energy sector context, and "
                "Risk dimension scoring (commodity/FX headwinds)."
            ),
            runnable=currency_commodities_agent,
        ),
        CompiledSubAgent(
            name="regulatory-filings",
            description=(
                "Retrieves regulatory and filing data: SEC filings via CIK lookup "
                "(ticker-to-CIK mapping, symbol map, institution search), filing "
                "headers and raw HTML filing documents, SIC code lookup, SEC "
                "schema directory, and SEC litigation RSS feed; CFTC Commitment "
                "of Traders (COT) reports and report search; and US congressional "
                "bills (bill metadata, full text, document URL listing). Use for "
                "regulatory risk assessment, compliance monitoring, legislative "
                "risk analysis (pending bills affecting the sector), COT-based "
                "commodity positioning context, and Risk dimension scoring "
                "(regulatory/legal headwinds)."
            ),
            runnable=regulatory_filings_agent,
        ),
        CompiledSubAgent(
            name="fama-french",
            description=(
                "Retrieves Fama-French academic factor data: 3-factor and 5-factor "
                "model returns (market/Mkt-RF, size/SMB, value/HML, profitability/RMW, "
                "investment/CMA), US portfolio returns sorted by size/value/momentum, "
                "regional and country portfolio returns, international index returns, "
                "and size/value breakpoints. Does NOT use a stock ticker — provides "
                "market-wide factor data. Use for factor exposure analysis, "
                "quantitative risk decomposition, style tilts (growth vs value, "
                "small vs large cap), and Risk/Valuation dimension scoring "
                "(factor risk premiums)."
            ),
            runnable=fama_french_agent,
        ),
    ]

    prompt = render_template("stock_evaluation.jinja")
    llm = config.get_llm()

    return create_deep_agent(
        model=llm,
        system_prompt=prompt,
        subagents=subagents
    )
