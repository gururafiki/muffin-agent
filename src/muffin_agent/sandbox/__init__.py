"""OpenSandbox integration for muffin agents.

Provides two integration points:

1. **OpenSandboxBackend** — a deepagents ``BaseSandbox`` implementation backed
   by an OpenSandbox container. Pass it as ``backend=`` to ``create_deep_agent``
   to give deep agents file system access and shell execution inside a secure
   container.

2. **create_python_execution_tool** — a factory that returns a LangChain tool
   wrapping ``execute_python``. Add the tool to data collection agents via the
   ``custom_tools`` argument of ``get_tools()``.

Usage::

    from muffin_agent.sandbox import (
        OpenSandboxBackend,
        create_opensandbox_backend,
        create_python_execution_tool,
    )

    # Create backend (async — call from an async context)
    backend = await create_opensandbox_backend(config)

    # Option A: use as deepagents backend
    agent = create_deep_agent(model=llm, backend=backend, ...)

    # Option B: use as a standalone tool
    tool = create_python_execution_tool(backend)
    tools = await get_tools(config, MCP_TOOLS, custom_tools=[tool])
"""

from .backend import OpenSandboxBackend, create_opensandbox_backend
from .tool import create_python_execution_tool

__all__ = [
    "OpenSandboxBackend",
    "create_opensandbox_backend",
    "create_python_execution_tool",
]
