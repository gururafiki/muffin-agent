"""StoreAccessMiddleware — registers generic store CRUD tools."""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from .tools import (
    store_delete,
    store_get,
    store_list_namespaces,
    store_put,
    store_search,
)


class StoreAccessMiddleware(AgentMiddleware):
    """Register generic store CRUD tools with an agent.

    Provides ``store_get``, ``store_put``, ``store_delete``,
    ``store_search``, and ``store_list_namespaces``.  Namespace access
    control is enforced by each tool via ``StoreConfiguration``.
    """

    def __init__(self) -> None:
        """Initialize with the five store CRUD tools."""
        self.tools = [
            store_get,
            store_put,
            store_delete,
            store_search,
            store_list_namespaces,
        ]

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Pass through — no interception logic yet."""
        return await handler(request)
