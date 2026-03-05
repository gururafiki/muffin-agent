"""Graph entry point for LangGraph Platform deployment.

Exposes a module-level ``graph`` variable (a ``CompiledStateGraph``) that the
LangGraph server imports at startup.  All configuration is read from environment
variables via ``Configuration.from_runnable_config``.

The OpenBB MCP server must be reachable when this module is imported because MCP
tools are fetched eagerly during graph construction.
"""

import asyncio

from muffin_agent.agents import create_stock_evaluation_agent
from muffin_agent.config import Configuration


async def _build_graph():
    config = Configuration.from_runnable_config({"configurable": {}})
    return await create_stock_evaluation_agent(config)


graph = asyncio.run(_build_graph())
