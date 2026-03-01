"""Stock evaluation agent.

Deep agent that orchestrates data collection subagents, validates collected
data, and produces a scored stock assessment with reasoning.
"""

from deepagents import CompiledSubAgent, create_deep_agent
from deepagents.backends import FilesystemBackend

from ..config import Configuration
from ..prompts import render_template
from .data_collection import (
    create_equity_fundamentals_data_collection_agent,
    create_equity_price_data_collection_agent,
)


async def create_stock_evaluation_agent(config: Configuration):
    """Build the stock evaluation deep agent.

    Create a deep agent that delegates data collection to equity-fundamentals
    and equity-price subagents, then validates, analyzes, and scores the stock.
    """
    fundamentals_agent = await create_equity_fundamentals_data_collection_agent(config)
    price_agent = await create_equity_price_data_collection_agent(config)

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
    ]

    prompt = render_template("stock_evaluation.jinja")
    llm = config.get_llm()

    return create_deep_agent(
        model=llm,
        system_prompt=prompt,
        subagents=subagents
    )
