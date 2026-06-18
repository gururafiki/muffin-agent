"""Muffin CLI — command-line interface for stock analysis agents."""

import asyncio
import os
from typing import Annotated

import typer
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver

from muffin_cli.output import StreamPrinter

app = typer.Typer()


@app.callback()
def _callback() -> None:
    """Muffin — multi-agent stock analysis CLI."""


async def _stream_fundamentals(ticker: str, query: str | None) -> None:
    """Build the equity fundamentals agent and stream output."""
    from langchain_core.runnables import RunnableConfig

    from muffin_agent.agents.data_collection import create_equity_fundamentals_agent
    from muffin_agent.model_config import ModelConfiguration
    from muffin_agent.utils.observability import setup_tracing

    config = ModelConfiguration.from_runnable_config(RunnableConfig(configurable={}))
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
    from muffin_agent.model_config import ModelConfiguration
    from muffin_agent.utils.observability import setup_tracing

    config = ModelConfiguration.from_runnable_config(RunnableConfig(configurable={}))
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
    from muffin_agent.model_config import ModelConfiguration
    from muffin_agent.utils.observability import setup_tracing

    config = ModelConfiguration.from_runnable_config(RunnableConfig(configurable={}))
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
    from muffin_agent.model_config import ModelConfiguration
    from muffin_agent.utils.observability import setup_tracing

    config = ModelConfiguration.from_runnable_config(RunnableConfig(configurable={}))
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
    from muffin_agent.model_config import ModelConfiguration
    from muffin_agent.utils.observability import setup_tracing

    config = ModelConfiguration.from_runnable_config(RunnableConfig(configurable={}))
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
    from muffin_agent.model_config import ModelConfiguration
    from muffin_agent.utils.observability import setup_tracing

    config = ModelConfiguration.from_runnable_config(RunnableConfig(configurable={}))
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
    from muffin_agent.model_config import ModelConfiguration
    from muffin_agent.utils.observability import setup_tracing

    config = ModelConfiguration.from_runnable_config(RunnableConfig(configurable={}))
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
    from muffin_agent.model_config import ModelConfiguration
    from muffin_agent.utils.observability import setup_tracing

    config = ModelConfiguration.from_runnable_config(RunnableConfig(configurable={}))
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
    from muffin_agent.model_config import ModelConfiguration
    from muffin_agent.utils.observability import setup_tracing

    config = ModelConfiguration.from_runnable_config(RunnableConfig(configurable={}))
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
    from muffin_agent.model_config import ModelConfiguration
    from muffin_agent.utils.observability import setup_tracing

    config = ModelConfiguration.from_runnable_config(RunnableConfig(configurable={}))
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
    from muffin_agent.model_config import ModelConfiguration
    from muffin_agent.utils.observability import setup_tracing

    config = ModelConfiguration.from_runnable_config(RunnableConfig(configurable={}))
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
    from muffin_agent.model_config import ModelConfiguration
    from muffin_agent.utils.observability import setup_tracing

    config = ModelConfiguration.from_runnable_config(RunnableConfig(configurable={}))
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
    from muffin_agent.model_config import ModelConfiguration
    from muffin_agent.utils.observability import setup_tracing

    config = ModelConfiguration.from_runnable_config(RunnableConfig(configurable={}))
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


async def _stream_web_search(query: str, ticker: str | None) -> None:
    """Build the web search agent and stream output."""
    from langchain_core.runnables import RunnableConfig

    from muffin_agent.agents.data_collection import (
        create_web_search_data_collection_agent,
    )
    from muffin_agent.model_config import ModelConfiguration
    from muffin_agent.utils.observability import setup_tracing

    session_id = ticker or query[:40].replace(" ", "-")
    config = ModelConfiguration.from_runnable_config(RunnableConfig(configurable={}))
    callbacks = setup_tracing(session_id=session_id)
    agent = await create_web_search_data_collection_agent(config)

    prompt = f"Ticker: {ticker}. {query}" if ticker else query

    printer = StreamPrinter()
    async for chunk, _metadata in agent.astream(
        {"messages": [HumanMessage(prompt)]},
        config=RunnableConfig(callbacks=callbacks, recursion_limit=40),
        stream_mode="messages",
    ):
        printer.print_chunk(chunk)
    printer.finish()


@app.command()
def web_search(
    query: Annotated[str, typer.Argument(help="Search query or URL to scrape")],
    ticker: Annotated[
        str | None,
        typer.Option("--ticker", "-t", help="Optional ticker symbol for context"),
    ] = None,
) -> None:
    """Search the web or scrape a URL using SearxNG and Firecrawl."""
    asyncio.run(_stream_web_search(query, ticker))


async def _stream_criterion(ticker: str, criterion: str, query: str | None) -> None:
    """Build the criterion evaluation agent and stream output."""
    from langchain_core.runnables import RunnableConfig

    from muffin_agent.agents import create_criterion_evaluation_agent
    from muffin_agent.model_config import ModelConfiguration
    from muffin_agent.utils.observability import setup_tracing

    config = ModelConfiguration.from_runnable_config(RunnableConfig(configurable={}))
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
    from muffin_agent.model_config import ModelConfiguration
    from muffin_agent.utils.observability import setup_tracing

    config = ModelConfiguration.from_runnable_config(RunnableConfig(configurable={}))
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


def _get_checkpointer() -> BaseCheckpointSaver:
    """Create a SQLite checkpointer backed by ``~/.muffin/checkpoints.db``."""
    import sqlite3
    from pathlib import Path

    from langgraph.checkpoint.sqlite import SqliteSaver

    db_path = Path.home() / ".muffin" / "checkpoints.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(db_path)))


async def _run_analyze(ticker: str, query: str | None, user: str) -> None:
    """Run the full analysis pipeline for a single ticker and print final thesis."""
    import json

    from langchain_core.runnables import RunnableConfig
    from langgraph.store.memory import InMemoryStore

    from muffin_agent.agents import build_investment_analysis_graph
    from muffin_agent.utils.observability import setup_tracing

    callbacks = setup_tracing(session_id=ticker)
    store = InMemoryStore()
    graph = build_investment_analysis_graph(
        checkpointer=_get_checkpointer(),
        store=store,
    )

    mandate = query or f"Produce a complete investment analysis for {ticker}"

    result = await graph.ainvoke(
        {"ticker": ticker, "query": mandate},
        config=RunnableConfig(
            callbacks=callbacks,
            configurable={"thread_id": ticker, "user_id": user},
        ),
    )

    thesis = result.get("thesis", {})
    typer.echo(json.dumps(thesis, indent=2, default=str))


@app.command()
def analyze(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g. AAPL)")],
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="Investment mandate / analysis focus"),
    ] = None,
    user: Annotated[
        str,
        typer.Option(
            "--user",
            help="User id for per-user long-term memory namespace "
            "(defaults to $USER or 'default-user')",
        ),
    ] = os.environ.get("USER", "default-user"),
) -> None:
    """Run the full 7-stage investment analysis pipeline for a given ticker.

    Stages run with maximum parallelism:
    Group 1 (parallel): market-regime, sector-analysis, company-analysis
    Group 2 (parallel): forecasting, risk-assessment
    Group 3 (sequential): valuation → thesis-synthesis

    NOTE: individual stages are not yet implemented (NotImplementedError).
    """
    asyncio.run(_run_analyze(ticker, query, user))


async def _run_screen(query: str, max_tickers: int, user: str) -> None:
    """Run the screening pipeline and print comparison results."""
    import json

    from langchain_core.runnables import RunnableConfig
    from langgraph.store.memory import InMemoryStore

    from muffin_agent.agents import build_equity_screening_graph
    from muffin_agent.utils.observability import setup_tracing

    session_id = f"screen-{query[:40].replace(' ', '-')}"
    callbacks = setup_tracing(session_id=session_id)
    store = InMemoryStore()
    graph = build_equity_screening_graph(
        checkpointer=_get_checkpointer(),
        store=store,
    )

    result = await graph.ainvoke(
        {"query": query, "tickers": [], "theses": []},
        config=RunnableConfig(
            callbacks=callbacks,
            configurable={
                "thread_id": session_id,
                "max_tickers": max_tickers,
                "user_id": user,
            },
        ),
    )

    comparison = result.get("comparison", {})
    typer.echo(json.dumps(comparison, indent=2, default=str))


@app.command()
def screen(
    query: Annotated[
        str,
        typer.Option("--query", "-q", help="Investment mandate / screening objective"),
    ],
    max_tickers: Annotated[
        int,
        typer.Option("--max-tickers", "-n", help="Max candidates to evaluate in depth"),
    ] = 5,
    user: Annotated[
        str,
        typer.Option(
            "--user",
            help="User id for per-user long-term memory namespace "
            "(defaults to $USER or 'default-user')",
        ),
    ] = os.environ.get("USER", "default-user"),
) -> None:
    """Auto-discover investment ideas and run the full pipeline on each candidate.

    Workflow:
      1. idea-sourcing: screen the market matching the query
      2. shared context: market-regime + sector-analysis (run once)
      3. per-ticker analysis: parallel full pipeline for each candidate
      4. comparison: rank and select the best ideas

    NOTE: individual stages are not yet implemented (NotImplementedError).
    """
    asyncio.run(_run_screen(query, max_tickers, user))


# ── Persona council CLI ──────────────────────────────────────────────────────


async def _run_persona(slug: str, ticker: str, query: str | None, user: str) -> None:
    """Run a single persona's verdict on a ticker and print the signal."""
    import json

    from langchain_core.runnables import RunnableConfig
    from langgraph.store.memory import InMemoryStore

    # v4: import the persona's build factory directly. Adding a new persona =
    # add an entry to PERSONA_BUILDERS (see agents/personas/council_graph.py).
    from muffin_agent.agents.personas_council.council_graph import PERSONA_BUILDERS
    from muffin_agent.utils.observability import setup_tracing

    builders_by_slug = dict(PERSONA_BUILDERS)
    if slug not in builders_by_slug:
        typer.echo(
            f"Unknown persona slug {slug!r}. Available: {sorted(builders_by_slug)}",
            err=True,
        )
        raise typer.Exit(code=1)

    session_id = f"persona-{slug}-{ticker}"
    callbacks = setup_tracing(session_id=session_id)
    config = RunnableConfig(
        callbacks=callbacks,
        configurable={"thread_id": session_id, "user_id": user},
    )

    store = InMemoryStore()  # noqa: F841 — kept for parity; subgraph reads via get_store()
    builder = builders_by_slug[slug]
    graph = await builder(config)

    result = await graph.ainvoke({"ticker": ticker, "query": query}, config=config)

    signals = result.get("persona_signals") or []
    if not signals:
        typer.echo(f"No signal returned for {slug} / {ticker}", err=True)
        raise typer.Exit(code=1)
    typer.echo(json.dumps(signals[0], indent=2, default=str))


@app.command()
def persona(
    slug: Annotated[
        str,
        typer.Argument(
            help="Persona slug (e.g. warren_buffett, ben_graham, cathie_wood)"
        ),
    ],
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g. AAPL)")],
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="Investment mandate / framing"),
    ] = None,
    user: Annotated[
        str,
        typer.Option(
            "--user",
            help="User id for per-user long-term memory namespace "
            "(defaults to $USER or 'default-user')",
        ),
    ] = os.environ.get("USER", "default-user"),
) -> None:
    """Render a single persona's verdict on a ticker.

    Runs the shared data-collection step then a single persona node.  Outputs
    the persona's structured signal (rating, confidence, reasoning, evidence).

    Available personas: warren_buffett, ben_graham, cathie_wood, charlie_munger,
    bill_ackman, michael_burry, mohnish_pabrai, nassim_taleb, peter_lynch,
    phil_fisher, rakesh_jhunjhunwala, stanley_druckenmiller, aswath_damodaran.
    """
    asyncio.run(_run_persona(slug, ticker, query, user))


async def _run_council(
    ticker: str, query: str | None, user: str, include_specialists: bool = False
) -> None:
    """Run the full 13-persona council on a ticker and print the synthesis."""
    import json

    from langchain_core.runnables import RunnableConfig
    from langgraph.store.memory import InMemoryStore

    from muffin_agent.agents.personas_council import build_council_graph
    from muffin_agent.utils.observability import setup_tracing

    session_id = f"council-{ticker}"
    callbacks = setup_tracing(session_id=session_id)
    store = InMemoryStore()
    config = RunnableConfig(
        callbacks=callbacks,
        configurable={"thread_id": session_id, "user_id": user},
    )
    graph = await build_council_graph(
        config,
        checkpointer=_get_checkpointer(),
        store=store,
        include_specialists=include_specialists,
    )

    result = await graph.ainvoke({"ticker": ticker, "query": query}, config=config)

    output = {
        "ticker": ticker,
        "council_synthesis": result.get("council_synthesis", {}),
        "persona_signals": result.get("persona_signals", []),
    }
    typer.echo(json.dumps(output, indent=2, default=str))


@app.command()
def council(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g. AAPL)")],
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="Investment mandate / framing"),
    ] = None,
    user: Annotated[
        str,
        typer.Option(
            "--user",
            help="User id for per-user long-term memory namespace "
            "(defaults to $USER or 'default-user')",
        ),
    ] = os.environ.get("USER", "default-user"),
    include_specialists: Annotated[
        bool,
        typer.Option(
            "--include-specialists",
            help="Also run the 6 specialist agents (technicals / sentiment / "
            "fundamentals / growth / valuation / news_sentiment)",
        ),
    ] = False,
) -> None:
    """Run the 13-persona investor council and print the synthesised verdict.

    Fans out to all 13 personas in parallel — each fetches its own data via
    curated OpenBB MCP tools — then synthesises into a single
    ``CouncilSynthesisOutput`` with a 5-tier consensus rating, vote breakdown,
    dissent summary, and key uncertainties.  Pass ``--include-specialists`` to
    add the six specialist signal agents to the fan-in.
    """
    asyncio.run(_run_council(ticker, query, user, include_specialists))


# ── Specialist signal CLI (technicals / sentiment) ──────────────────────────


async def _run_specialist(slug: str, ticker: str, query: str | None, user: str) -> None:
    """Run a single specialist's signal on a ticker and print the result.

    Handles both the fully-deterministic sync specialists (technicals,
    sentiment) and the metric-heavy async specialists that take ``config``
    (fundamentals, growth, valuation, news_sentiment — persona-style).
    """
    import json

    from langchain_core.runnables import RunnableConfig

    from muffin_agent.agents.personas_council.specialists import (
        build_fundamentals_analysis_agent,
        build_growth_analysis_agent,
        build_news_sentiment_analysis_agent,
        build_sentiment_analysis_agent,
        build_technical_analysis_agent,
        build_valuation_analysis_agent,
    )
    from muffin_agent.utils.observability import setup_tracing

    sync_builders = {
        "technicals": build_technical_analysis_agent,
        "sentiment": build_sentiment_analysis_agent,
    }
    async_builders = {
        "fundamentals": build_fundamentals_analysis_agent,
        "growth": build_growth_analysis_agent,
        "valuation": build_valuation_analysis_agent,
        "news_sentiment": build_news_sentiment_analysis_agent,
    }
    if slug not in sync_builders and slug not in async_builders:
        available = sorted([*sync_builders, *async_builders])
        typer.echo(
            f"Unknown specialist slug {slug!r}. Available: {available}",
            err=True,
        )
        raise typer.Exit(code=1)

    session_id = f"specialist-{slug}-{ticker}"
    callbacks = setup_tracing(session_id=session_id)
    config = RunnableConfig(
        callbacks=callbacks,
        configurable={"thread_id": session_id, "user_id": user},
    )

    if slug in sync_builders:
        graph = sync_builders[slug]()
    else:
        graph = await async_builders[slug](config)

    result = await graph.ainvoke({"ticker": ticker, "query": query}, config=config)

    signals = result.get("persona_signals") or []
    if not signals:
        typer.echo(f"No signal returned for {slug} / {ticker}", err=True)
        raise typer.Exit(code=1)
    typer.echo(json.dumps(signals[0], indent=2, default=str))


@app.command()
def technicals(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g. AAPL)")],
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="Investment mandate / framing"),
    ] = None,
    user: Annotated[
        str,
        typer.Option(
            "--user",
            help="User id for per-user long-term memory namespace "
            "(defaults to $USER or 'default-user')",
        ),
    ] = os.environ.get("USER", "default-user"),
) -> None:
    """Run the technical-analysis specialist (5-strategy ensemble).

    Computes trend / mean-reversion / momentum / volatility-regime /
    stat-arb signals over the 1-year OHLCV series, then a weighted
    ensemble.  Fully deterministic — no LLM call.  Output is a
    ``TechnicalSignal`` with 5-tier rating and per-strategy evidence.
    """
    asyncio.run(_run_specialist("technicals", ticker, query, user))


@app.command()
def sentiment(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g. AAPL)")],
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="Investment mandate / framing"),
    ] = None,
    user: Annotated[
        str,
        typer.Option(
            "--user",
            help="User id for per-user long-term memory namespace "
            "(defaults to $USER or 'default-user')",
        ),
    ] = os.environ.get("USER", "default-user"),
) -> None:
    """Run the sentiment-analysis specialist (insider + news aggregation).

    Computes a 30/70 weighted combination of insider-trade direction and
    company news sentiment over the trailing 12 months.  Fully
    deterministic — no LLM call.  Output is a ``SentimentSignal`` with
    5-tier rating and insider/news breakdown.
    """
    asyncio.run(_run_specialist("sentiment", ticker, query, user))


@app.command(name="fundamentals-signal")
def fundamentals_signal(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g. AAPL)")],
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="Investment mandate / framing"),
    ] = None,
    user: Annotated[
        str,
        typer.Option(
            "--user",
            help="User id for per-user long-term memory namespace "
            "(defaults to $USER or 'default-user')",
        ),
    ] = os.environ.get("USER", "default-user"),
) -> None:
    """Run the fundamentals specialist (profitability / growth / health / ratios).

    A ReAct step extracts the latest financial-metrics snapshot; scoring is
    deterministic (majority vote across four dimensions).  Output is a
    ``FundamentalsSignal`` with a 5-tier rating.  (Named ``fundamentals-signal``
    to avoid clashing with the ``fundamentals`` data-collection command.)
    """
    asyncio.run(_run_specialist("fundamentals", ticker, query, user))


@app.command()
def growth(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g. AAPL)")],
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="Investment mandate / framing"),
    ] = None,
    user: Annotated[
        str,
        typer.Option(
            "--user",
            help="User id for per-user long-term memory namespace "
            "(defaults to $USER or 'default-user')",
        ),
    ] = os.environ.get("USER", "default-user"),
) -> None:
    """Run the growth specialist (growth / valuation / margins / insider / health).

    A ReAct step extracts multi-year growth + margin history; scoring is a
    deterministic weighted blend.  Output is a ``GrowthSignal`` (5-tier).
    """
    asyncio.run(_run_specialist("growth", ticker, query, user))


@app.command()
def valuation(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g. AAPL)")],
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="Investment mandate / framing"),
    ] = None,
    user: Annotated[
        str,
        typer.Option(
            "--user",
            help="User id for per-user long-term memory namespace "
            "(defaults to $USER or 'default-user')",
        ),
    ] = os.environ.get("USER", "default-user"),
) -> None:
    """Run the valuation specialist (DCF / owner earnings / EV-EBITDA / RIM).

    A ReAct step extracts the line items + metrics; scoring is a deterministic
    weighted intrinsic-value gap vs market cap.  Output is a ``ValuationSignal``
    (5-tier).
    """
    asyncio.run(_run_specialist("valuation", ticker, query, user))


@app.command(name="news-sentiment")
def news_sentiment(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g. AAPL)")],
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="Investment mandate / framing"),
    ] = None,
    user: Annotated[
        str,
        typer.Option(
            "--user",
            help="User id for per-user long-term memory namespace "
            "(defaults to $USER or 'default-user')",
        ),
    ] = os.environ.get("USER", "default-user"),
) -> None:
    """Run the news-sentiment specialist (LLM headline classification).

    The one LLM specialist: a ReAct step fetches recent company news and
    classifies each headline's sentiment; aggregation is deterministic.
    Output is a ``NewsSentimentSignal`` (5-tier).
    """
    asyncio.run(_run_specialist("news_sentiment", ticker, query, user))


async def _stream_criteria(
    ticker: str,
    query: str | None,
    sector: str | None,
    sub_sector: str | None,
    market: str | None,
    stock_type: str | None,
) -> None:
    """Build the criteria definition agent and stream output."""
    from langchain_core.runnables import RunnableConfig

    from muffin_agent.agents.criteria_definition import create_criteria_definition_agent
    from muffin_agent.model_config import ModelConfiguration
    from muffin_agent.utils.observability import setup_tracing

    config = ModelConfiguration.from_runnable_config(RunnableConfig(configurable={}))
    callbacks = setup_tracing(session_id=ticker)
    agent = await create_criteria_definition_agent(config)

    prompt = (
        f"Ticker: {ticker}. {query}"
        if query
        else f"Define valuation criteria for {ticker}"
    )

    state: dict = {"messages": [HumanMessage(prompt)]}
    if sector:
        state["sector"] = sector
    if sub_sector:
        state["sub_sector"] = sub_sector
    if market:
        state["market"] = market
    if stock_type:
        state["stock_type"] = stock_type

    printer = StreamPrinter()
    async for chunk, _metadata in agent.astream(
        state,
        config=RunnableConfig(callbacks=callbacks, recursion_limit=40),
        stream_mode="messages",
    ):
        printer.print_chunk(chunk)
    printer.finish()


@app.command()
def criteria(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g. AAPL)")],
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="Custom query or investment mandate"),
    ] = None,
    sector: Annotated[
        str | None,
        typer.Option("--sector", "-s", help="Sector tag (e.g. banking, software-saas)"),
    ] = None,
    sub_sector: Annotated[
        str | None,
        typer.Option("--sub-sector", help="Sub-sector tag (e.g. life for insurance)"),
    ] = None,
    market: Annotated[
        str | None,
        typer.Option("--market", "-m", help="Market type: developed or emerging"),
    ] = None,
    stock_type: Annotated[
        str | None,
        typer.Option("--stock-type", "-t", help="Stock type: value or growth"),
    ] = None,
) -> None:
    """Define valuation criteria for a ticker using sector-specific skills.

    Pre-classify the ticker with flags to skip classification:

        muffin criteria AAPL --sector software-saas -m developed
    """
    asyncio.run(_stream_criteria(ticker, query, sector, sub_sector, market, stock_type))


async def _run_criteria_analyze(
    ticker: str,
    query: str | None,
    sector: str | None,
    sub_sector: str | None,
    market: str | None,
    stock_type: str | None,
    user: str,
) -> None:
    """Run the criteria-driven analysis pipeline and print the synthesis."""
    import json

    from langchain_core.runnables import RunnableConfig
    from langgraph.store.memory import InMemoryStore

    from muffin_agent.agents import build_criteria_analysis_graph
    from muffin_agent.utils.observability import setup_tracing

    callbacks = setup_tracing(session_id=ticker)
    store = InMemoryStore()
    graph = build_criteria_analysis_graph(
        checkpointer=_get_checkpointer(),
        store=store,
    )

    mandate = query or f"Produce a criteria-driven analysis for {ticker}"

    initial_state: dict = {"ticker": ticker, "query": mandate}
    if sector:
        initial_state["sector"] = sector
    if sub_sector:
        initial_state["sub_sector"] = sub_sector
    if market:
        initial_state["market"] = market
    if stock_type:
        initial_state["stock_type"] = stock_type

    result = await graph.ainvoke(
        initial_state,
        config=RunnableConfig(
            callbacks=callbacks,
            configurable={"thread_id": ticker, "user_id": user},
        ),
    )

    synthesis = result.get("synthesis", {})
    typer.echo(json.dumps(synthesis, indent=2, default=str))


@app.command(name="criteria-analyze")
def criteria_analyze(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g. AAPL)")],
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="Investment mandate / analysis focus"),
    ] = None,
    sector: Annotated[
        str | None,
        typer.Option("--sector", "-s", help="Pre-supplied sector tag (skips Stage 1)"),
    ] = None,
    sub_sector: Annotated[
        str | None,
        typer.Option("--sub-sector", help="Pre-supplied sub-sector tag"),
    ] = None,
    market: Annotated[
        str | None,
        typer.Option(
            "--market", "-m", help="Pre-supplied market: developed or emerging"
        ),
    ] = None,
    stock_type: Annotated[
        str | None,
        typer.Option(
            "--stock-type", "-t", help="Pre-supplied stock type: value or growth"
        ),
    ] = None,
    user: Annotated[
        str,
        typer.Option(
            "--user",
            help="User id for per-user long-term memory namespace "
            "(defaults to $USER or 'default-user')",
        ),
    ] = os.environ.get("USER", "default-user"),
) -> None:
    """Run the criteria-driven analysis pipeline.

    Pipeline:
      1. ticker classification (skipped if --sector + --market + --stock-type
         are all supplied)
      2. criteria definition (skill-filtered) and valuation methodology (web
         research) — parallel
      3. merge criteria (deterministic dedup)
      4. per-criterion evaluation (parallel fan-out)
      5. synthesis

    Pre-classify to skip Stage 1:

        muffin criteria-analyze JPM --sector banking -m developed -t value
    """
    asyncio.run(
        _run_criteria_analyze(
            ticker, query, sector, sub_sector, market, stock_type, user
        )
    )


async def _run_research(
    query: str,
    mode: str | None,
    sources: list[str] | None,
    task_type: str | None,
    user: str,
    thread: str | None,
) -> None:
    """Run the research pipeline and render the cited answer."""
    from langchain_core.runnables import RunnableConfig
    from langgraph.store.memory import InMemoryStore

    from muffin_agent.agents.research import build_research_graph
    from muffin_agent.utils.observability import setup_tracing
    from muffin_cli.output import render_research_output

    session_id = thread or f"research-{query[:40].replace(' ', '-')}"
    callbacks = setup_tracing(session_id=session_id)
    store = InMemoryStore()
    graph = build_research_graph(
        checkpointer=_get_checkpointer(),
        store=store,
    )

    initial_state: dict = {"query": query}
    if sources:
        initial_state["allowed_sources"] = sources
    if mode:
        initial_state["mode_override"] = mode
    if task_type:
        initial_state["task_type_override"] = task_type

    result = await graph.ainvoke(
        initial_state,
        config=RunnableConfig(
            callbacks=callbacks,
            configurable={"thread_id": session_id, "user_id": user},
        ),
    )

    output = result.get("output") or {}
    if not output:
        typer.echo("(research pipeline returned no output)")
        return

    render_research_output(output)


@app.command()
def research(
    query: Annotated[str, typer.Argument(help="The research question")],
    mode: Annotated[
        str | None,
        typer.Option(
            "--mode",
            "-m",
            help="Override classifier-chosen mode: speed | balanced | quality",
        ),
    ] = None,
    sources: Annotated[
        str | None,
        typer.Option(
            "--sources",
            "-s",
            help="Comma-separated allowed sources (default: web)",
        ),
    ] = None,
    task_type: Annotated[
        str | None,
        typer.Option(
            "--task-type",
            "-t",
            help=(
                "Override classifier-chosen task_type: research_report, "
                "comparison, how_to, summary, debate, factual_qa"
            ),
        ),
    ] = None,
    user: Annotated[
        str,
        typer.Option(
            "--user",
            help="User id for per-user long-term memory namespace "
            "(defaults to $USER or 'default-user')",
        ),
    ] = os.environ.get("USER", "default-user"),
    thread: Annotated[
        str | None,
        typer.Option(
            "--thread",
            help="Thread id for follow-up continuity (default: slug of query)",
        ),
    ] = None,
) -> None:
    """Run the deep research agent on a query and render a cited answer.

    Pipeline: classifier → researcher → rerank → writer.

    Examples:
        muffin research "Latest news on Anthropic Claude 4.7"
        muffin research "Postgres vs MySQL for OLTP" --mode quality
        muffin research "What is 2+2?"                    # skip_search path
        muffin research "How do I set up pgvector?" -t how_to -m quality
    """
    sources_list = [s.strip() for s in sources.split(",")] if sources else None
    asyncio.run(_run_research(query, mode, sources_list, task_type, user, thread))


async def _run_decide(
    ticker: str,
    narrative: str | None,
    query: str | None,
    user: str,
    invest_rounds: int,
    risk_rounds: int,
    reflection_enabled: bool,
    decision_date: str | None,
) -> None:
    """Run the trading_decision pipeline for a ticker.

    The four analyst agents fetch their own data via OpenBB MCP (price
    history, fundamentals, news, ownership) and Firecrawl MCP (social /
    web). The caller provides only the ticker + optional framing
    (``query`` / ``narrative``); the package is self-contained and does
    NOT consume outputs from any other muffin pipeline.
    """
    import json

    from langchain_core.runnables import RunnableConfig
    from langgraph.store.memory import InMemoryStore

    from muffin_agent.agents.trading_decision import build_trading_decision_graph
    from muffin_agent.utils.observability import setup_tracing

    callbacks = setup_tracing(session_id=f"decide-{ticker}")
    # NOTE: InMemoryStore does not persist across CLI invocations, so the
    # reflection-memory layer only accumulates learning within a single
    # Python process. Wire a PostgresStore (or LangGraph Platform's
    # injected store) for true cross-session persistence.
    store = InMemoryStore()
    configurable: dict = {
        "thread_id": f"decide-{ticker}",
        "user_id": user,
        "max_investment_debate_rounds": invest_rounds,
        "max_risk_debate_rounds": risk_rounds,
        "reflection_enabled": reflection_enabled,
    }
    if decision_date:
        configurable["decision_date"] = decision_date
    config = RunnableConfig(
        callbacks=callbacks,
        configurable=configurable,
        recursion_limit=100,
    )

    graph = await build_trading_decision_graph(
        config,
        checkpointer=_get_checkpointer(),
        store=store,
    )

    initial_state: dict = {"ticker": ticker.upper()}
    if decision_date:
        initial_state["decision_date"] = decision_date
    if query:
        initial_state["query"] = query
    if narrative:
        initial_state["narrative"] = narrative

    result = await graph.ainvoke(initial_state, config=config)

    decision = result.get("portfolio_decision", {})
    typer.echo(json.dumps(decision, indent=2, default=str))


@app.command()
def decide(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g. AAPL)")],
    narrative: Annotated[
        str | None,
        typer.Option(
            "--narrative",
            "-n",
            help=(
                "Optional free-form research notes. Layered into the "
                "downstream Bull/Bear/Judge/Trader/PM prompts alongside the "
                "four analyst reports the pipeline generates."
            ),
        ),
    ] = None,
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="Investment mandate / analysis focus"),
    ] = None,
    user: Annotated[
        str,
        typer.Option(
            "--user",
            help=(
                "User id for per-user reflection memory namespace "
                "(defaults to $USER or 'default-user')"
            ),
        ),
    ] = os.environ.get("USER", "default-user"),
    invest_rounds: Annotated[
        int,
        typer.Option(
            "--invest-rounds",
            help="Max Bull-Bear debate rounds (default 2 = 4 turns)",
        ),
    ] = 2,
    risk_rounds: Annotated[
        int,
        typer.Option(
            "--risk-rounds",
            help="Max risk-debate rounds (default 1 = 3 turns)",
        ),
    ] = 1,
    reflection_enabled: Annotated[
        bool,
        typer.Option(
            "--reflection/--no-reflection",
            help="Enable outcome-driven reflection memory (default on)",
        ),
    ] = True,
    decision_date: Annotated[
        str | None,
        typer.Option(
            "--decision-date",
            help=(
                "Override the decision date (YYYY-MM-DD). Defaults to today UTC. "
                "Pinned for deterministic testing."
            ),
        ),
    ] = None,
) -> None:
    """Run the full trading-decision pipeline for a ticker.

    Pipeline:
      1. reflector_resolve — resolve prior pending decisions with realised returns
      2. Analysts (parallel) — Market / Fundamentals / News / Social
      3. Bull-Bear debate — N rounds (default 2)
      4. Investment Judge — synthesise 5-tier signal
      5. Trader — operational entry / stop / sizing
      6. Aggressive-Conservative-Neutral risk debate — M rounds (default 1)
      7. Portfolio Manager — canonical 5-tier decision
      8. decision_writeback — persist as pending for future reflection

    Examples::

      muffin decide AAPL
      muffin decide AAPL --query "long-term hold candidate"
      muffin decide AAPL --narrative "Recent earnings call mentioned X..."

    The four analysts fetch their own data via OpenBB MCP and Firecrawl
    MCP — make sure both services are running (`docker compose up -d
    openbb-mcp firecrawl-mcp`).

    Reflection memory uses an in-process InMemoryStore in the CLI today
    so prior decisions are not persisted across invocations. Wire a
    PostgresStore (or run on LangGraph Platform) for cross-session
    persistence.
    """
    asyncio.run(
        _run_decide(
            ticker,
            narrative,
            query,
            user,
            invest_rounds,
            risk_rounds,
            reflection_enabled,
            decision_date,
        )
    )


def main() -> None:
    """Entry point for the `muffin` CLI."""
    app()


if __name__ == "__main__":
    main()
