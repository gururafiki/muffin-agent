"""The MCP-tool seam — fixture-backed fake tools + the patch helper.

We patch ``MultiServerMCPClient`` (not ``get_tools``) so the **real**
``get_tools`` name-filter keeps running — only the network round-trip is faked.
The fake tools mirror ``langchain-mcp-adapters`` exactly: a ``StructuredTool``
with a permissive JSON-schema ``args_schema`` and a ``**kwargs`` coroutine, so any
arguments the scripted ``tool_turn`` supplies validate and the fixture content is
returned verbatim.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.tools import StructuredTool

from .fixtures import available_tool_names, load_fixture

# Mirrors how langchain-mcp-adapters builds tools: a JSON-schema dict + a
# **kwargs coroutine. ``additionalProperties: True`` accepts any scripted args.
_PERMISSIVE_ARGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": True,
}


def _make_fake_tool(name: str, scenario: str) -> StructuredTool:
    async def _call(**kwargs: Any) -> str | list:  # noqa: ARG001 — args ignored
        return load_fixture(name, scenario)

    return StructuredTool(
        name=name,
        description=f"Integration fixture stand-in for the {name!r} MCP tool.",
        args_schema=_PERMISSIVE_ARGS_SCHEMA,
        coroutine=_call,
    )


def build_fake_mcp_tools(scenario: str = "aapl") -> list[StructuredTool]:
    """Build one fake ``StructuredTool`` per tool that has a fixture.

    The real ``get_tools`` filters this superset down to each agent's allowlist,
    so a single fixture library serves every graph.
    """
    return [_make_fake_tool(name, scenario) for name in available_tool_names()]


@contextmanager
def patch_mcp(scenario: str = "aapl") -> Iterator[list[StructuredTool]]:
    """Patch ``MultiServerMCPClient`` so MCP loads return fixture-backed tools.

    The real ``McpConfiguration`` and the real ``get_tools`` name-filter still
    run; only the client's network ``get_tools()`` call is replaced.
    """
    tools = build_fake_mcp_tools(scenario)
    client = MagicMock(name="FakeMultiServerMCPClient")
    client.get_tools = AsyncMock(return_value=tools)
    with patch(
        "muffin_agent.agents.data_collection.utils.MultiServerMCPClient",
        return_value=client,
    ):
        yield tools
