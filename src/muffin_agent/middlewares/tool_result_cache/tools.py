"""LangChain tools for the tool result cache middleware.

Provides tools that let agents interact with cached tool results:

- ``discover_cached_tool_outputs`` — scan the shared store for cached results.
- ``get_tool_output_schema`` — look up output schema for any tool by name.
- ``write_cached_tool_output_to_backend`` — materialize cached data to sandbox.

The sandbox is discovered by ``thread_id`` metadata via the OpenSandbox API.
If no running sandbox exists, a new one is created automatically.
"""

import importlib
import json
import pkgutil

from langchain_core.tools import BaseTool, tool
from langchain_mcp_adapters.sessions import create_session
from langgraph.prebuilt import ToolRuntime
from mcp.types import PaginatedRequestParams

from ...mcp_config import McpConfiguration
from ...sandbox.factory import aget_sandbox
from .config import ToolResultCacheConfiguration

# ── Cache discovery ──────────────────────────────────────────────────────────


@tool
async def discover_cached_tool_outputs(runtime: ToolRuntime) -> str:
    """Discover all cached tool results available in the shared store.

    Scans the store for cached entries and returns a JSON array describing
    each result: tool name, original arguments, timestamp, content size,
    and store key (``args_hash``).

    Call this before collecting data to avoid duplicate MCP tool calls.

    Args:
        runtime: Injected by LangGraph ToolNode.

    Returns:
        JSON array of metadata entries, or ``"[]"`` if nothing is cached.
    """
    store = runtime.store
    if store is None:
        return "[]"

    namespaces = await store.alist_namespaces(prefix=("cache",))
    entries = []
    for ns in namespaces:
        items = await store.asearch(ns)
        for item in items:
            val = item.value
            entries.append(
                {
                    "tool_name": val.get("tool_name", ns[-1]),
                    "args": val.get("args", {}),
                    "cached_at": val.get("cached_at"),
                    "content_size": val.get("content_size", 0),
                    "store_key": item.key,
                }
            )
    return json.dumps(entries)


# ── Write cached data to sandbox ─────────────────────────────────────────────


@tool(parse_docstring=True)
async def write_cached_tool_output_to_backend(
    tool_name: str,
    args_hash: str,
    runtime: ToolRuntime,
    file_path: str | None = None,
) -> str:
    """Write a cached tool result from the store to a sandbox file.

    Materializes cached data to the sandbox filesystem so it can be loaded
    by ``execute_python`` code.  Call ``discover_cached_tool_outputs`` first
    to find available cache entries and their ``store_key`` (args_hash).

    Args:
        tool_name: The cached tool name (e.g. "equity_price_historical").
        args_hash: The store key from ``discover_cached_tool_outputs`` output.
        file_path: Optional custom file path. Defaults to
            ``/data/cache/{tool_name}/{args_hash}.json``.
        runtime: Injected by LangGraph ToolNode.

    Returns:
        Confirmation message with the file path, or an error message.
    """
    store = runtime.store
    if store is None:
        return "Error: no store available"

    item = await store.aget(("cache", tool_name), args_hash)
    if item is None or not item.value.get("content"):
        return f"Error: no cached result for tool '{tool_name}' with hash '{args_hash}'"

    content = item.value["content"]
    target = file_path or f"/data/cache/{tool_name}/{args_hash}.json"

    sandbox = await aget_sandbox(runtime)
    async with sandbox:
        try:
            await sandbox.files.write_file(target, content)
        except Exception as exc:
            return f"Error writing to sandbox: {exc}"

    return (
        f"Data written to {target} ({len(content)} chars). "
        f"Load it in execute_python with: json.load(open('{target}'))"
    )


# ── Schema lookup ────────────────────────────────────────────────────────────


def _find_python_tool_schema(
    tool_name: str,
    packages: list[str],
) -> dict | None:
    """Find output_schema for a Python tool by scanning configured packages.

    Iterates all modules in the given packages, inspects module-level
    attributes for ``BaseTool`` instances, and returns the pre-computed
    JSON schema dict from ``extras["output_schema"]`` if present.
    """
    for pkg_path in packages:
        try:
            pkg = importlib.import_module(pkg_path)
        except ImportError:
            continue
        pkg_paths = getattr(pkg, "__path__", None)
        if pkg_paths is None:
            continue
        for info in pkgutil.iter_modules(pkg_paths):
            mod = importlib.import_module(f"{pkg_path}.{info.name}")
            for attr in vars(mod).values():
                extras = getattr(attr, "extras", None) or {}
                if (
                    isinstance(attr, BaseTool)
                    and attr.name == tool_name
                    and extras.get("output_schema")
                ):
                    return extras["output_schema"]
    return None


@tool(parse_docstring=True)
async def get_tool_output_schema(tool_name: str, runtime: ToolRuntime) -> str:
    """Get the output schema for a tool by name.

    For Python tools: returns the JSON schema stored in ``extras``.
    For MCP tools: returns the full JSON Schema from the MCP server.

    Args:
        tool_name: Tool name (e.g. "equity_price_historical",
            "compute_yield_curve_metrics").
        runtime: Injected by LangGraph ToolNode.

    Returns:
        JSON string with schema, or a message if not found.
    """
    # 1. Auto-scan Python tools for extras["output_schema"] (fast, no I/O).
    cache_config = ToolResultCacheConfiguration.from_runnable_config(
        runtime.config,
    )
    schema = _find_python_tool_schema(tool_name, cache_config.tool_schema_packages)
    if schema is not None:
        return json.dumps(schema)

    # 2. Try MCP tools via session.list_tools().
    config = McpConfiguration.from_runnable_config(runtime.config)
    for conn in config.get_mcp_connections().values():
        async with create_session(conn) as session:
            await session.initialize()
            cursor: str | None = None
            while True:
                params = PaginatedRequestParams(cursor=cursor) if cursor else None
                result = await session.list_tools(params=params)
                for t in result.tools:
                    if t.name == tool_name and t.outputSchema is not None:
                        return json.dumps(t.outputSchema)
                if not result.nextCursor:
                    break
                cursor = result.nextCursor

    return f"No output schema found for tool '{tool_name}'"
