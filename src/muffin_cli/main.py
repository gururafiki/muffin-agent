"""Muffin CLI — command-line interface for stock analysis agents."""

import asyncio
from typing import Annotated

import typer
from langchain_core.messages import HumanMessage

from muffin_cli.output import StreamPrinter

app = typer.Typer()


@app.callback()
def _callback() -> None:
    """Muffin — multi-agent stock analysis CLI."""


async def _stream_fundamentals(ticker: str, query: str | None) -> None:
    """Build the equity fundamentals agent and stream output."""
    from langchain_core.runnables import RunnableConfig

    from muffin_agent.agents.data_collection import create_equity_fundamentals_agent
    from muffin_agent.config import Configuration
    from muffin_agent.utils.observability import setup_tracing

    config = Configuration.from_runnable_config(RunnableConfig(configurable={}))
    callbacks = setup_tracing(session_id=ticker)
    agent = await create_equity_fundamentals_agent(config)

    prompt = (
        f"Ticker: {ticker}. {query}"
        if query
        else f"Get comprehensive fundamental data for {ticker}"
    )

    printer = StreamPrinter()
    async for chunk, _metadata in agent.astream(
        {"messages": [HumanMessage(prompt)]},
        config=RunnableConfig(callbacks=callbacks, recursion_limit=40),
        stream_mode="messages",
    ):
        printer.print_chunk(chunk)
    printer.finish()


@app.command()
def fundamentals(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g. AAPL)")],
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="Custom query (overrides default)"),
    ] = None,
) -> None:
    """Retrieve equity fundamental data for a given ticker."""
    asyncio.run(_stream_fundamentals(ticker, query))


async def _stream_price(ticker: str, query: str | None) -> None:
    """Build the equity price agent and stream output."""
    from langchain_core.runnables import RunnableConfig

    from muffin_agent.agents.data_collection import create_equity_price_agent
    from muffin_agent.config import Configuration
    from muffin_agent.utils.observability import setup_tracing

    config = Configuration.from_runnable_config(RunnableConfig(configurable={}))
    callbacks = setup_tracing(session_id=ticker)
    agent = await create_equity_price_agent(config)

    prompt = (
        f"Ticker: {ticker}. {query}"
        if query
        else f"Get comprehensive price data for {ticker}"
    )

    printer = StreamPrinter()
    async for chunk, _metadata in agent.astream(
        {"messages": [HumanMessage(prompt)]},
        config=RunnableConfig(callbacks=callbacks, recursion_limit=40),
        stream_mode="messages",
    ):
        printer.print_chunk(chunk)
    printer.finish()


@app.command()
def price(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g. AAPL)")],
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="Custom query (overrides default)"),
    ] = None,
) -> None:
    """Retrieve equity price data for a given ticker."""
    asyncio.run(_stream_price(ticker, query))


async def _stream_estimates(ticker: str, query: str | None) -> None:
    """Build the equity estimates agent and stream output."""
    from langchain_core.runnables import RunnableConfig

    from muffin_agent.agents.data_collection import (
        create_equity_estimates_data_collection_agent,
    )
    from muffin_agent.config import Configuration
    from muffin_agent.utils.observability import setup_tracing

    config = Configuration.from_runnable_config(RunnableConfig(configurable={}))
    callbacks = setup_tracing(session_id=ticker)
    agent = await create_equity_estimates_data_collection_agent(config)

    prompt = (
        f"Ticker: {ticker}. {query}"
        if query
        else f"Get comprehensive analyst estimates data for {ticker}"
    )

    printer = StreamPrinter()
    async for chunk, _metadata in agent.astream(
        {"messages": [HumanMessage(prompt)]},
        config=RunnableConfig(callbacks=callbacks, recursion_limit=40),
        stream_mode="messages",
    ):
        printer.print_chunk(chunk)
    printer.finish()


@app.command()
def estimates(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g. AAPL)")],
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="Custom query (overrides default)"),
    ] = None,
) -> None:
    """Retrieve analyst estimates data for a given ticker."""
    asyncio.run(_stream_estimates(ticker, query))


async def _stream_ownership(ticker: str, query: str | None) -> None:
    """Build the equity ownership agent and stream output."""
    from langchain_core.runnables import RunnableConfig

    from muffin_agent.agents.data_collection import (
        create_equity_ownership_data_collection_agent,
    )
    from muffin_agent.config import Configuration
    from muffin_agent.utils.observability import setup_tracing

    config = Configuration.from_runnable_config(RunnableConfig(configurable={}))
    callbacks = setup_tracing(session_id=ticker)
    agent = await create_equity_ownership_data_collection_agent(config)

    prompt = (
        f"Ticker: {ticker}. {query}"
        if query
        else f"Get comprehensive ownership and short interest data for {ticker}"
    )

    printer = StreamPrinter()
    async for chunk, _metadata in agent.astream(
        {"messages": [HumanMessage(prompt)]},
        config=RunnableConfig(callbacks=callbacks, recursion_limit=40),
        stream_mode="messages",
    ):
        printer.print_chunk(chunk)
    printer.finish()


@app.command()
def ownership(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g. AAPL)")],
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="Custom query (overrides default)"),
    ] = None,
) -> None:
    """Retrieve ownership and short interest data for a given ticker."""
    asyncio.run(_stream_ownership(ticker, query))


async def _stream_news(ticker: str, query: str | None) -> None:
    """Build the news agent and stream output."""
    from langchain_core.runnables import RunnableConfig

    from muffin_agent.agents.data_collection import create_news_data_collection_agent
    from muffin_agent.config import Configuration
    from muffin_agent.utils.observability import setup_tracing

    config = Configuration.from_runnable_config(RunnableConfig(configurable={}))
    callbacks = setup_tracing(session_id=ticker)
    agent = await create_news_data_collection_agent(config)

    prompt = (
        f"Ticker: {ticker}. {query}"
        if query
        else f"Get recent news and sentiment for {ticker}"
    )

    printer = StreamPrinter()
    async for chunk, _metadata in agent.astream(
        {"messages": [HumanMessage(prompt)]},
        config=RunnableConfig(callbacks=callbacks, recursion_limit=40),
        stream_mode="messages",
    ):
        printer.print_chunk(chunk)
    printer.finish()


@app.command()
def news(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g. AAPL)")],
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="Custom query (overrides default)"),
    ] = None,
) -> None:
    """Retrieve recent news and sentiment data for a given ticker."""
    asyncio.run(_stream_news(ticker, query))


async def _stream_options(ticker: str, query: str | None) -> None:
    """Build the options agent and stream output."""
    from langchain_core.runnables import RunnableConfig

    from muffin_agent.agents.data_collection import create_options_data_collection_agent
    from muffin_agent.config import Configuration
    from muffin_agent.utils.observability import setup_tracing

    config = Configuration.from_runnable_config(RunnableConfig(configurable={}))
    callbacks = setup_tracing(session_id=ticker)
    agent = await create_options_data_collection_agent(config)

    prompt = (
        f"Ticker: {ticker}. {query}"
        if query
        else f"Get options chain and implied volatility surface for {ticker}"
    )

    printer = StreamPrinter()
    async for chunk, _metadata in agent.astream(
        {"messages": [HumanMessage(prompt)]},
        config=RunnableConfig(callbacks=callbacks, recursion_limit=40),
        stream_mode="messages",
    ):
        printer.print_chunk(chunk)
    printer.finish()


@app.command()
def options(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g. AAPL)")],
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="Custom query (overrides default)"),
    ] = None,
) -> None:
    """Retrieve options chain and implied volatility surface for a given ticker."""
    asyncio.run(_stream_options(ticker, query))


async def _stream_economy_macro(ticker: str, query: str | None) -> None:
    """Build the economy macro agent and stream output."""
    from langchain_core.runnables import RunnableConfig

    from muffin_agent.agents.data_collection import (
        create_economy_macro_data_collection_agent,
    )
    from muffin_agent.config import Configuration
    from muffin_agent.utils.observability import setup_tracing

    config = Configuration.from_runnable_config(RunnableConfig(configurable={}))
    callbacks = setup_tracing(session_id=ticker)
    agent = await create_economy_macro_data_collection_agent(config)

    prompt = (
        f"Ticker: {ticker}. {query}"
        if query
        else (
            f"Provide current macroeconomic conditions and outlook relevant to {ticker}"
        )
    )

    printer = StreamPrinter()
    async for chunk, _metadata in agent.astream(
        {"messages": [HumanMessage(prompt)]},
        config=RunnableConfig(callbacks=callbacks, recursion_limit=40),
        stream_mode="messages",
    ):
        printer.print_chunk(chunk)
    printer.finish()


@app.command()
def economy_macro(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g. AAPL)")],
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="Custom query (overrides default)"),
    ] = None,
) -> None:
    """Retrieve macroeconomic indicators and economic data."""
    asyncio.run(_stream_economy_macro(ticker, query))


async def _stream_fixed_income(ticker: str, query: str | None) -> None:
    """Build the fixed income agent and stream output."""
    from langchain_core.runnables import RunnableConfig

    from muffin_agent.agents.data_collection import (
        create_fixed_income_data_collection_agent,
    )
    from muffin_agent.config import Configuration
    from muffin_agent.utils.observability import setup_tracing

    config = Configuration.from_runnable_config(RunnableConfig(configurable={}))
    callbacks = setup_tracing(session_id=ticker)
    agent = await create_fixed_income_data_collection_agent(config)

    prompt = (
        f"Ticker: {ticker}. {query}"
        if query
        else (
            f"Get current interest rates, yield curve, and key fixed income"
            f" data relevant to {ticker}"
        )
    )

    printer = StreamPrinter()
    async for chunk, _metadata in agent.astream(
        {"messages": [HumanMessage(prompt)]},
        config=RunnableConfig(callbacks=callbacks, recursion_limit=40),
        stream_mode="messages",
    ):
        printer.print_chunk(chunk)
    printer.finish()


@app.command()
def fixed_income(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g. AAPL)")],
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="Custom query (overrides default)"),
    ] = None,
) -> None:
    """Retrieve interest rates, yield curves, and bond market data."""
    asyncio.run(_stream_fixed_income(ticker, query))


async def _stream_discovery_screening(ticker: str, query: str | None) -> None:
    """Build the discovery and screening agent and stream output."""
    from langchain_core.runnables import RunnableConfig

    from muffin_agent.agents.data_collection import (
        create_discovery_screening_data_collection_agent,
    )
    from muffin_agent.config import Configuration
    from muffin_agent.utils.observability import setup_tracing

    config = Configuration.from_runnable_config(RunnableConfig(configurable={}))
    callbacks = setup_tracing(session_id=ticker)
    agent = await create_discovery_screening_data_collection_agent(config)

    prompt = (
        f"Ticker: {ticker}. {query}"
        if query
        else (
            f"Get peer companies, sector group valuation, and upcoming earnings "
            f"calendar for {ticker}"
        )
    )

    printer = StreamPrinter()
    async for chunk, _metadata in agent.astream(
        {"messages": [HumanMessage(prompt)]},
        config=RunnableConfig(callbacks=callbacks, recursion_limit=40),
        stream_mode="messages",
    ):
        printer.print_chunk(chunk)
    printer.finish()


@app.command()
def discovery_screening(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g. AAPL)")],
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="Custom query (overrides default)"),
    ] = None,
) -> None:
    """Retrieve discovery and screening data for peer and market context."""
    asyncio.run(_stream_discovery_screening(ticker, query))


async def _stream_etf_index(ticker: str, query: str | None) -> None:
    """Build the ETF and index agent and stream output."""
    from langchain_core.runnables import RunnableConfig

    from muffin_agent.agents.data_collection import (
        create_etf_index_data_collection_agent,
    )
    from muffin_agent.config import Configuration
    from muffin_agent.utils.observability import setup_tracing

    config = Configuration.from_runnable_config(RunnableConfig(configurable={}))
    callbacks = setup_tracing(session_id=ticker)
    agent = await create_etf_index_data_collection_agent(config)

    prompt = (
        f"Ticker: {ticker}. {query}"
        if query
        else f"Get ETF exposure, sector weights, and index context for {ticker}"
    )

    printer = StreamPrinter()
    async for chunk, _metadata in agent.astream(
        {"messages": [HumanMessage(prompt)]},
        config=RunnableConfig(callbacks=callbacks, recursion_limit=40),
        stream_mode="messages",
    ):
        printer.print_chunk(chunk)
    printer.finish()


@app.command()
def etf_index(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g. AAPL)")],
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="Custom query (overrides default)"),
    ] = None,
) -> None:
    """Retrieve ETF and index data for sector and benchmark context."""
    asyncio.run(_stream_etf_index(ticker, query))


async def _stream_currency_commodities(ticker: str, query: str | None) -> None:
    """Build the currency, commodity, and crypto agent and stream output."""
    from langchain_core.runnables import RunnableConfig

    from muffin_agent.agents.data_collection import (
        create_currency_commodities_data_collection_agent,
    )
    from muffin_agent.config import Configuration
    from muffin_agent.utils.observability import setup_tracing

    config = Configuration.from_runnable_config(RunnableConfig(configurable={}))
    callbacks = setup_tracing(session_id=ticker)
    agent = await create_currency_commodities_data_collection_agent(config)

    prompt = (
        f"Ticker: {ticker}. {query}"
        if query
        else (
            f"Get FX rates, commodity spot prices, and energy outlook"
            f" relevant to {ticker}"
        )
    )

    printer = StreamPrinter()
    async for chunk, _metadata in agent.astream(
        {"messages": [HumanMessage(prompt)]},
        config=RunnableConfig(callbacks=callbacks, recursion_limit=40),
        stream_mode="messages",
    ):
        printer.print_chunk(chunk)
    printer.finish()


@app.command()
def currency_commodities(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g. AAPL)")],
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="Custom query (overrides default)"),
    ] = None,
) -> None:
    """Retrieve currency, commodity, and crypto data."""
    asyncio.run(_stream_currency_commodities(ticker, query))


async def _stream_fama_french(ticker: str, query: str | None) -> None:
    """Build the Fama-French agent and stream output."""
    from langchain_core.runnables import RunnableConfig

    from muffin_agent.agents.data_collection import (
        create_fama_french_data_collection_agent,
    )
    from muffin_agent.config import Configuration
    from muffin_agent.utils.observability import setup_tracing

    config = Configuration.from_runnable_config(RunnableConfig(configurable={}))
    callbacks = setup_tracing(session_id=ticker)
    agent = await create_fama_french_data_collection_agent(config)

    prompt = (
        f"Ticker: {ticker}. {query}"
        if query
        else (
            f"Get Fama-French factor data and portfolio returns"
            f" for market context relevant to {ticker}"
        )
    )

    printer = StreamPrinter()
    async for chunk, _metadata in agent.astream(
        {"messages": [HumanMessage(prompt)]},
        config=RunnableConfig(callbacks=callbacks, recursion_limit=40),
        stream_mode="messages",
    ):
        printer.print_chunk(chunk)
    printer.finish()


@app.command()
def fama_french(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g. AAPL)")],
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="Custom query (overrides default)"),
    ] = None,
) -> None:
    """Retrieve Fama-French factor model and portfolio return data."""
    asyncio.run(_stream_fama_french(ticker, query))


async def _stream_regulatory_filings(ticker: str, query: str | None) -> None:
    """Build the regulatory filings agent and stream output."""
    from langchain_core.runnables import RunnableConfig

    from muffin_agent.agents.data_collection import (
        create_regulatory_filings_data_collection_agent,
    )
    from muffin_agent.config import Configuration
    from muffin_agent.utils.observability import setup_tracing

    config = Configuration.from_runnable_config(RunnableConfig(configurable={}))
    callbacks = setup_tracing(session_id=ticker)
    agent = await create_regulatory_filings_data_collection_agent(config)

    prompt = (
        f"Ticker: {ticker}. {query}"
        if query
        else f"Get comprehensive regulatory and filing data for {ticker}"
    )

    printer = StreamPrinter()
    async for chunk, _metadata in agent.astream(
        {"messages": [HumanMessage(prompt)]},
        config=RunnableConfig(callbacks=callbacks, recursion_limit=40),
        stream_mode="messages",
    ):
        printer.print_chunk(chunk)
    printer.finish()


@app.command()
def regulatory_filings(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g. AAPL)")],
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="Custom query (overrides default)"),
    ] = None,
) -> None:
    """Retrieve SEC filings, CFTC reports, and congressional bill data."""
    asyncio.run(_stream_regulatory_filings(ticker, query))


async def _stream_criterion(ticker: str, criterion: str, query: str | None) -> None:
    """Build the criterion evaluation agent and stream output."""
    from langchain_core.runnables import RunnableConfig

    from muffin_agent.agents import create_criterion_evaluation_agent
    from muffin_agent.config import Configuration
    from muffin_agent.utils.observability import setup_tracing

    config = Configuration.from_runnable_config(RunnableConfig(configurable={}))
    callbacks = setup_tracing(session_id=ticker)
    agent = await create_criterion_evaluation_agent(config)

    prompt = f"Ticker: {ticker}. Criterion: {criterion}" + (
        f" {query}" if query else ""
    )

    printer = StreamPrinter()
    async for chunk, _metadata in agent.astream(
        {"messages": [HumanMessage(prompt)]},
        config=RunnableConfig(callbacks=callbacks, recursion_limit=40),
        stream_mode="messages",
    ):
        printer.print_chunk(chunk)
    printer.finish()


@app.command()
def criterion(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g. AAPL)")],
    criterion_text: Annotated[
        str,
        typer.Option("--criterion", "-c", help="The investment criterion to evaluate"),
    ],
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="Additional context or constraints"),
    ] = None,
) -> None:
    """Evaluate a single investment criterion for a given ticker."""
    asyncio.run(_stream_criterion(ticker, criterion_text, query))


async def _stream_evaluate(ticker: str, query: str | None) -> None:
    """Build the stock evaluation agent and stream output."""
    from langchain_core.runnables import RunnableConfig

    from muffin_agent.agents import create_stock_evaluation_agent
    from muffin_agent.config import Configuration
    from muffin_agent.utils.observability import setup_tracing

    config = Configuration.from_runnable_config(RunnableConfig(configurable={}))
    callbacks = setup_tracing(session_id=ticker)
    agent = await create_stock_evaluation_agent(config)

    prompt = (
        f"Ticker: {ticker}. {query}"
        if query
        else (
            f"Evaluate {ticker} stock — analyze fundamentals"
            " and price data to produce a scored assessment"
        )
    )

    printer = StreamPrinter()
    async for chunk, _metadata in agent.astream(
        {"messages": [HumanMessage(prompt)]},
        config=RunnableConfig(callbacks=callbacks, recursion_limit=40),
        stream_mode="messages",
    ):
        printer.print_chunk(chunk)
    printer.finish()


@app.command()
def evaluate(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g. AAPL)")],
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="Custom query (overrides default)"),
    ] = None,
) -> None:
    """Evaluate a stock with scored assessment using data collection subagents."""
    asyncio.run(_stream_evaluate(ticker, query))


def main() -> None:
    """Entry point for the `muffin` CLI."""
    app()


if __name__ == "__main__":
    main()
