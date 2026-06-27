"""The MCP-tool seam — fixture-backed fake tools + the patch helper.

We patch ``MultiServerMCPClient`` (not ``get_tools``) so the **real**
``get_tools`` name-filter keeps running — only the network round-trip is faked.
The fake tools mirror ``langchain-mcp-adapters`` exactly: a ``StructuredTool``
with a JSON-schema dict ``args_schema`` and a ``**kwargs`` coroutine.

For OpenBB tools the ``args_schema`` is the **real** ``inputSchema`` pulled from
the OpenBB catalogue (``openbb_mcp_tools.json``, resolved via
``openbb_catalogue_path``), so each fake advertises the genuine
argument signature (`provider`, `symbol`, …) the production tool exposes. The
schema is *not* enforced at call time (``StructuredTool`` passes a dict schema
through without validation — verified), so scripted ``tool_turn`` args never need
to match it; the fidelity is purely in what the bound tool advertises. Tools
without an entry (Firecrawl, local ``@tool``s) fall back to a permissive schema.
"""

from __future__ import annotations

import functools
import json
from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.tools import StructuredTool

from .fixtures import available_tool_names, load_fixture, openbb_catalogue_path

# Permissive fallback for tools absent from the OpenBB catalogue (e.g. Firecrawl).
# ``additionalProperties: True`` accepts any scripted args.
_PERMISSIVE_ARGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": True,
}


@functools.lru_cache(maxsize=1)
def _openbb_input_schemas() -> dict[str, dict[str, Any]]:
    """Map OpenBB tool name → its real ``inputSchema`` (loaded once, lazily).

    Sourced from the OpenBB catalogue (``openbb_mcp_tools.json``, resolved via
    :func:`openbb_catalogue_path` — the same source the fixtures are authored from).
    Returns ``{}`` if the catalogue is missing so the harness still works (every tool
    then falls back to the permissive schema).
    """
    path = openbb_catalogue_path()
    if path is None:
        return {}
    data = json.loads(path.read_text())
    tools = data["tools"] if isinstance(data, dict) else data
    return {
        t["name"]: t["inputSchema"]
        for t in tools
        if isinstance(t, dict) and "inputSchema" in t
    }


def args_schema_for(name: str) -> dict[str, Any]:
    """Real OpenBB ``inputSchema`` for *name*, or the permissive fallback."""
    return _openbb_input_schemas().get(name, _PERMISSIVE_ARGS_SCHEMA)


def _make_fake_tool(name: str, scenario: str) -> StructuredTool:
    async def _call(**kwargs: Any) -> str | list:  # noqa: ARG001 — args ignored
        return load_fixture(name, scenario)

    return StructuredTool(
        name=name,
        description=f"Integration fixture stand-in for the {name!r} MCP tool.",
        args_schema=args_schema_for(name),
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
