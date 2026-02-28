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

    from muffin_agent.agents.data_collection.equity_fundamentals import build_graph
    from muffin_agent.config import Configuration
    from muffin_agent.utils.observability import setup_tracing

    config = Configuration.from_runnable_config(RunnableConfig(configurable={}))
    callbacks = setup_tracing(session_id=ticker)
    agent = await build_graph(config)

    prompt = (
        f"Ticker: {ticker}. {query}"
        if query
        else f"Get comprehensive fundamental data for {ticker}"
    )

    printer = StreamPrinter()
    async for chunk, _metadata in agent.astream(
        {"messages": [HumanMessage(prompt)]},
        config=RunnableConfig(callbacks=callbacks),
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

    from muffin_agent.agents.data_collection.equity_price import build_graph
    from muffin_agent.config import Configuration
    from muffin_agent.utils.observability import setup_tracing

    config = Configuration.from_runnable_config(RunnableConfig(configurable={}))
    callbacks = setup_tracing(session_id=ticker)
    agent = await build_graph(config)

    prompt = (
        f"Ticker: {ticker}. {query}"
        if query
        else f"Get comprehensive price data for {ticker}"
    )

    printer = StreamPrinter()
    async for chunk, _metadata in agent.astream(
        {"messages": [HumanMessage(prompt)]},
        config=RunnableConfig(callbacks=callbacks),
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


def main() -> None:
    """Entry point for the `muffin` CLI."""
    app()


if __name__ == "__main__":
    main()
