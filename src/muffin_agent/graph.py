"""Graph entry point for LangGraph Platform deployment.

Exposes a module-level ``graph`` variable (a ``CompiledStateGraph``) that the
LangGraph server imports at startup.  All configuration is read from environment
variables via ``Configuration.from_runnable_config``.

Sandbox lifecycle
-----------------
No containers are created at import time.  ``create_stock_evaluation_agent``
attaches ``get_backend`` (from ``langchain-opensandbox``, re-exported by
:mod:`muffin_agent.sandbox`) as the deepagents backend.  It is a lazy factory:
it dials nothing until the agent actually touches the sandbox, then finds or
creates one tagged with the conversation's ``thread_id`` and reuses it for the
rest of that thread.  Parallel conversations are fully isolated.

The OpenBB MCP server must be reachable when this module is imported: it builds
the graph at import time, and graph construction loads MCP tools.  Discovery is
cached per connection set (see ``agents/data_collection/utils.py``), so the cost
is paid once per process rather than per request.
"""

import asyncio

from muffin_agent.agents import create_stock_evaluation_agent
from muffin_agent.utils.observability import instrument_graph


async def _build_graph():
    return await create_stock_evaluation_agent({"configurable": {}})


graph = instrument_graph(asyncio.run(_build_graph()))
