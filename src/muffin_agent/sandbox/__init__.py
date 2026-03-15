"""OpenSandbox integration for muffin agents.

Provides three integration points:

1. **get_backend** — ``BackendFactory`` function that discovers or creates
   a sandbox by ``thread_id`` metadata via the OpenSandbox API. Works with
   both ``ToolRuntime`` and ``Runtime`` contexts (``thread_id`` comes from
   ``langgraph.config.get_config()``). Pass as ``backend=get_backend`` to
   ``create_deep_agent``.

2. **get_sandbox** / **aget_sandbox** — Sync and async functions that find
   or create a sandbox for the current thread. Used internally by
   ``execute_python`` and available for direct use when a raw sandbox
   instance is needed.

3. **execute_python** — LangChain async tool that discovers the sandbox for
   the current thread and executes Python code in it. If no sandbox exists,
   creates one automatically.

Usage::

    from muffin_agent.sandbox import get_backend, execute_python

    # Deep agent: sandbox discovered/created lazily per conversation
    agent = create_deep_agent(
        model=llm, backend=get_backend, ...
    )

    # Subagent with execute_python tool
    agent = create_agent(
        model=llm, tools=[execute_python], ...
    )

Limitations:
    - If the sandbox dies mid-conversation (e.g. 1-hour timeout, container
      crash), a new container is created transparently on the next call.
      Any in-sandbox state (installed packages, written files) is lost.
"""

from .backend import OpenSandboxBackend
from .factory import aget_sandbox, get_backend, get_sandbox
from .tools import execute_python

__all__ = [
    "OpenSandboxBackend",
    "aget_sandbox",
    "execute_python",
    "get_backend",
    "get_sandbox",
]
