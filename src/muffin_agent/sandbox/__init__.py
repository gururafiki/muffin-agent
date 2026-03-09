"""OpenSandbox integration for muffin agents.

Provides two integration points:

1. **SandboxFactory** — a deepagents ``BackendFactory`` that provisions one
   ``OpenSandboxBackend`` per ``thread_id`` and reuses it for all tool calls
   within that conversation. Pass it as ``backend=`` to ``create_deep_agent``.
   On a cache miss it calls ``SandboxSync.connect()`` to reconnect to an
   existing container before falling back to creating a new one.

2. **create_python_execution_tool** — returns a LangChain async tool that
   creates a fresh ``Sandbox`` for each ``execute_python`` invocation, runs
   the code, and closes the container afterwards. Add the tool to data
   collection agents via the ``custom_tools`` argument of ``get_tools()``.

Usage::

    from muffin_agent.sandbox import SandboxFactory, create_python_execution_tool

    # Deep agent backend: one container per conversation
    registry = SandboxFactory(config)
    agent = create_deep_agent(model=llm, backend=registry, ...)

    # Standalone execution tool: fresh container per call
    tool = create_python_execution_tool(config)
    tools = await get_tools(config, MCP_TOOLS, custom_tools=[tool])
"""

from .backend import (
    OpenSandboxBackend,
    SandboxFactory,
    create_opensandbox_backend,
    create_opensandbox_sandbox,
)
from .tool import create_python_execution_tool

__all__ = [
    "OpenSandboxBackend",
    "SandboxFactory",
    "create_opensandbox_backend",
    "create_opensandbox_sandbox",
    "create_python_execution_tool",
]
