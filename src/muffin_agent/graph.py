"""Graph entry point for LangGraph Platform deployment.

Exposes a module-level ``graph`` variable (a ``CompiledStateGraph``) that the
LangGraph server imports at startup.  All configuration is read from environment
variables via ``Configuration.from_runnable_config``.

Sandbox lifecycle
-----------------
No containers are created at import time.  ``create_stock_evaluation_agent``
attaches a ``SandboxRegistry`` as the deepagents backend.  The registry
provisions one ``OpenSandboxBackend`` per ``thread_id`` on the first tool call
for that conversation and reuses it for all subsequent calls in the same thread.
Parallel conversations are fully isolated.

The OpenBB MCP server must be reachable when this module is imported because
MCP tools are fetched eagerly during graph construction.
"""

import asyncio

from muffin_agent.agents import create_stock_evaluation_agent
from muffin_agent.config import Configuration


async def _build_graph():
    config = Configuration.from_runnable_config({"configurable": {}})
    return await create_stock_evaluation_agent(config)


graph = asyncio.run(_build_graph())
