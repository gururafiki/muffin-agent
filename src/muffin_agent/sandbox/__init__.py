"""OpenSandbox integration for muffin agents.

Provides two integration points:

1. **OpenSandboxBackend** — a deepagents ``BaseSandbox`` implementation backed
   by an OpenSandbox container. Pass it as ``backend=`` to ``create_deep_agent``
   to give deep agents file system access and shell execution inside a secure
   container. Created synchronously via ``create_opensandbox_backend``.

2. **create_python_execution_tool** — a factory that returns a LangChain async
   tool using the native OpenSandbox async SDK. Add the tool to data collection
   agents via the ``custom_tools`` argument of ``get_tools()``. Requires an
   async ``Sandbox`` from ``create_opensandbox_sandbox``.

Usage::

    from muffin_agent.sandbox import (
        OpenSandboxBackend,
        create_opensandbox_backend,
        create_opensandbox_sandbox,
        create_python_execution_tool,
    )

    # Option A: deep agent backend (sync setup)
    backend = create_opensandbox_backend(config)
    agent = create_deep_agent(model=llm, backend=backend, ...)

    # Option B: standalone tool (async setup)
    sandbox = await create_opensandbox_sandbox(config)
    tool = create_python_execution_tool(sandbox)
    tools = await get_tools(config, MCP_TOOLS, custom_tools=[tool])
"""

from .backend import (
    OpenSandboxBackend,
    create_opensandbox_backend,
    create_opensandbox_sandbox,
)
from .tool import create_python_execution_tool

__all__ = [
    "OpenSandboxBackend",
    "create_opensandbox_backend",
    "create_opensandbox_sandbox",
    "create_python_execution_tool",
]
