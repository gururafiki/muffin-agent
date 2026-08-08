"""Sandbox integration for muffin agents.

The OpenSandbox backend, the thread-scoped sandbox factory and the generic
``execute_python`` tool live in
`langchain-opensandbox <https://github.com/gururafiki/langchain-opensandbox>`_
— they are not finance-specific and were extracted so the community can use
them. This package re-exports them so muffin's own import sites stay stable,
and adds the two bridge tools that ARE muffin's, because they route through
``AccessControlledStore`` and its namespace permissions.

What comes from the package:

1. **get_backend** — a ``BackendFactory`` that discovers or creates a sandbox
   by ``thread_id`` metadata. Works with both ``ToolRuntime`` and ``Runtime``
   contexts (``thread_id`` comes from ``langgraph.config.get_config()``). Pass
   as ``backend=get_backend`` to ``create_deep_agent``, or call
   ``MuffinAgentBuilder.with_sandbox()``.

2. **get_sandbox** / **aget_sandbox** — sync and async functions returning the
   raw sandbox for the current thread, for callers that need the client itself.

3. **execute_python** — LangChain async tool running Python in that sandbox.
   Used for ad-hoc calculations not covered by the deterministic financial
   tools in :mod:`muffin_agent.tools`.

What is muffin's, defined in :mod:`muffin_agent.sandbox.tools`:

4. **write_store_data_to_sandbox** / **read_sandbox_file_to_store** — bridge
   tools moving data between the LangGraph store and the sandbox filesystem.
   Namespace access is governed by ``StoreConfiguration``.

Settings are unchanged: ``OPENSANDBOX_URL``, ``OPENSANDBOX_API_KEY``,
``OPENSANDBOX_IMAGE`` and ``OPENSANDBOX_USE_SERVER_PROXY``, environment first,
then the run's ``configurable``.

Limitations:
    If the sandbox dies mid-conversation (e.g. 1-hour timeout, container
    crash), a new container is created transparently on the next call.
    Any in-sandbox state (installed packages, written files) is lost.
"""

from langchain_opensandbox import OpenSandboxSandbox, OpenSandboxSettings
from langchain_opensandbox.factory import aget_sandbox, get_backend, get_sandbox
from langchain_opensandbox.tools import execute_python

from .tools import (
    read_sandbox_file_to_store,
    write_store_data_to_sandbox,
)

__all__ = [
    "OpenSandboxSandbox",
    "OpenSandboxSettings",
    "aget_sandbox",
    "execute_python",
    "get_backend",
    "get_sandbox",
    "read_sandbox_file_to_store",
    "write_store_data_to_sandbox",
]
