# Tool Result Cache Middleware

## Problem

Within a single analysis run, identical tool calls execute multiple times:

- **Cross-agent duplication**: `economy-macro` subagent runs in both `market_regime` (Group 1) and `forecasting` (Group 2), fetching the same FRED series twice. Each investment node creates fresh subagent instances with isolated graph state — no deduplication.
- **Intra-agent retries**: Deep agent orchestration loops may re-invoke subagents that call the same MCP tools. Agent retries after partial failures repeat tool calls that already succeeded.
- **Data embedded in LLM context**: When agents fetch large datasets (price history, financial statements) and later need to process them in sandbox, the LLM embeds data in `execute_python` code strings — wasting context window and risking truncation.

Previously, only `ToolErrorHandler` deduplicated — and only for **permanently failed** tool calls. Successful results had no deduplication.

## Solution

`ToolResultCacheMiddleware` caches successful tool results in a shared **`InMemoryStore`** (via `ToolRuntime.store`) so that identical calls across different agents are deduplicated automatically.

### Design principles

1. **Store as shared cache**: All agents in a graph share the same `InMemoryStore` (passed via `graph.compile(store=store)` → `create_deep_agent(store=store)`). Results cached by `economy-macro` in `market_regime` are visible to `economy-macro` in `forecasting`.
2. **No instance mutation after `__init__`**: Per Deep Agents middleware docs, mutating `self.x` in middleware hooks leads to race conditions with concurrent tool calls. All state is in the shared store.
3. **Sandbox materialization on demand**: Cached data lives in-memory (fast reads). When agents need to process data with `execute_python`, they call `write_tool_output_to_backend` to materialize specific entries to sandbox files.

## How It Works

### Store namespace design

```
namespace: ("cache", "{tool_name}")
key: "{args_hash}" (12-char SHA-256 of sorted JSON args)
value: {
    "content": "<tool result string>",
    "tool_name": "equity_price_historical",
    "args": {"symbol": "AAPL", "period": "5y"},
    "cached_at": "2026-03-22T14:30:00+00:00",
    "content_size": 48210,
}
```

### Middleware flow

```
Tool call arrives
    │
    ├─ Not in cacheable_tools? → Pass through to handler
    │
    ├─ store.aget(namespace, key) returns item? → Return "[Cached result — ...]<content>"
    │
    └─ store.aget returns None (cache miss) → Execute handler
         │
         ├─ Success? → store.aput(namespace, key, value), return "result\n[Data cached — ...]"
         │
         └─ Error? → Return as-is (not cached)
```

### Cache miss (first call)

```
ToolErrorHandler → FilesystemMiddleware → ToolResultCacheMiddleware → MCP handler
                                              ↓
                                         execute tool
                                              ↓
                                    store.aput(("cache", tool), hash, {content, metadata})
                                              ↓
                                    return: "result...\n[Data cached — tool: ..., args_hash: ...]"
                                              ↓
                                    FilesystemMiddleware: evicts if > 20K tokens
                                              ↓
                                    agent sees: result (or eviction ref) + cache annotation
```

### Cache hit (duplicate call)

```
ToolErrorHandler → FilesystemMiddleware → ToolResultCacheMiddleware
                                              ↓
                                         store.aget(("cache", tool), hash)
                                              ↓
                                    return: "[Cached result — tool: ..., args_hash: ...]\n<content>"
                                              ↓
                                    small message, no eviction needed
                                              ↓
                                    agent sees: cache annotation + content (no MCP call)
```

### Middleware composition

#### Data collection agents

```python
middleware=[
    ToolErrorHandler(),                   # outer: blocks duplicate permanent failures
    FilesystemMiddleware(backend=...),    # middle: file tools + auto-eviction
    ToolResultCacheMiddleware(            # inner: store-based cache + dedup
        cacheable_tools=frozenset(MCP_TOOLS),
    ),
]
```

**Order matters**: `ToolErrorHandler` is outermost — a duplicate call to a permanently failed tool is blocked before reaching the cache. `FilesystemMiddleware` provides file tools and auto-evicts large results. `ToolResultCacheMiddleware` is innermost — it wraps the actual tool execution.

#### Investment agents (deep agents)

```python
create_deep_agent(
    ...
    store=store,
    middleware=[
        ToolResultCacheMiddleware(
            cacheable_tools=frozenset({
                "compute_yield_curve_metrics",
                "compute_factor_zscore",
                ...
            }),
        ),
    ],
)
```

User middleware is appended after the standard deep agent stack (TodoList, Filesystem, Summarization, etc.), so `ToolResultCacheMiddleware` wraps the tool execution at the innermost layer.

### `cacheable_tools` filtering

Each data collection agent passes its `MCP_TOOLS` list so only MCP tools are cached. Filesystem tools (`ls`, `read_file`, etc.) and `execute_python` pass through uncached.

Investment agents pass their computation tool names (e.g., `compute_roic`, `compute_yield_curve_metrics`).

### Store wiring

The `InMemoryStore` is created at the CLI entry point and threaded through the graph:

```
CLI: store = InMemoryStore()
  → build_investment_analysis_graph(store=store)
    → graph.compile(store=store)
    → partial(market_regime_node, store=store)  # etc.
      → create_market_regime_agent(config, store=store)
        → create_deep_agent(store=store)
          → ToolNode injects store into ToolRuntime
            → ToolResultCacheMiddleware reads request.runtime.store
```

### `discover_cached_data` tool

`ToolResultCacheMiddleware` automatically registers a `discover_cached_data` tool via its `tools` attribute. The tool queries the store for all namespaces under `("cache",)` and returns a JSON array of metadata entries. Agents call this before collecting data to see what is already available.

The tool is NOT included in `cacheable_tools` — it always executes fresh to return current cache state.

### `write_tool_output_to_backend` tool

Also auto-registered. Reads a cached entry from the store and writes its content to a sandbox file for use by `execute_python`. Takes `tool_name` and `args_hash` from `discover_cached_data` output, plus an optional custom `file_path`.

**Agent workflow**: `discover_cached_data` → `get_tool_output_schema` → `write_tool_output_to_backend` → `execute_python`

### `get_tool_output_schema` tool

Also auto-registered. Agents call it with a tool name to get the JSON Schema describing the tool's output format:

- **Python tools**: Auto-scans all modules in `muffin_agent.tools` for `BaseTool` instances with `extras["output_schema"]`. Tools define output schemas via co-located Pydantic models and `@tool(extras={"output_schema": Model.model_json_schema()})`.
- **MCP tools**: Creates a temporary session via `langchain_mcp_adapters.sessions.create_session`, calls `session.list_tools()` with pagination, and returns the `outputSchema` field from the matching MCP `Tool` object.

Python tools are checked first (fast, in-memory). MCP fallback only runs if no Python tool matches.

### Prompt integration

Agents receive instructions about cached data via Jinja2 partials:

- **Data collection**: `{% include 'data_collection/_data_file_rules.jinja' %}` — explains store-based caching and how to materialize data to sandbox files.
- **Investment**: `{% include 'investment/_sandbox_data_rules.jinja' %}` — explains the discover → schema → write → execute workflow.

## Files

| File | Change |
|------|--------|
| `src/muffin_agent/agents/middleware.py` | `ToolResultCacheMiddleware` (store-based), `_args_hash()`, `_is_error_content()` |
| `src/muffin_agent/agents/data_collection/utils.py` | `data_collection_middleware()` helper |
| `src/muffin_agent/sandbox/tools.py` | `discover_cached_data`, `write_tool_output_to_backend`, `get_tool_output_schema` tools |
| `src/muffin_agent/sandbox/__init__.py` | Exports all sandbox tools |
| `src/muffin_agent/agents/investment/utils.py` | `run_deep_agent_node()` with `store` param |
| 4 investment agent files | Add `store` param, pass to `create_deep_agent` |
| `src/muffin_agent/agents/investment_analysis.py` | `build_investment_analysis_graph(store=store)` |
| `src/muffin_agent/agents/equity_screening.py` | `build_equity_screening_graph(store=store)` |
| `src/muffin_cli/main.py` | Create `InMemoryStore()` and pass to graph builders |
| 13 data collection agent files | Use `data_collection_middleware(MCP_TOOLS)` |
| `src/muffin_agent/prompts/data_collection/_data_file_rules.jinja` | Store-based caching instructions |
| `src/muffin_agent/prompts/investment/_sandbox_data_rules.jinja` | Discover → schema → write → execute workflow |
| `tests/agents/test_middleware.py` | 17 tests (store-based cache) |
| `tests/sandbox/test_tool.py` | `discover_cached_data`, `write_tool_output_to_backend`, `get_tool_output_schema` tests |
| `tests/agents/test_investment_utils.py` | Graph builder store tests |

## Verification

```bash
pytest tests/agents/test_middleware.py tests/sandbox/test_tool.py tests/agents/test_investment_utils.py -v
```

Tests verify:
- Cache miss: handler executes, result written to store, annotation appended
- Cache hit: handler NOT called, returns annotation + cached content
- Error results not cached
- `cacheable_tools` whitelist correctly filters
- Non-whitelisted tools pass through uncached
- Store write failure returns original result gracefully
- Empty store read treated as miss
- Command results from handler pass through uncached
- No store (None) passes through without caching
- Store value contains required metadata fields
- `discover_cached_data` returns JSON from store, handles empty/no store
- `write_tool_output_to_backend` writes to sandbox, handles not found/no store/custom path
- `get_tool_output_schema` returns Python tool schema, MCP schema, and not-found message
- Graph builders accept and wire store parameter

## Limitations & Future Work

- **No cache eviction**: Store grows within a single graph execution. Not a practical issue (tool calls are bounded by agent turn limits).
- **In-memory only**: Cache is scoped to a single process. A future improvement could use LangGraph's `PostgresStore` for persistent cross-session caching.
- **No cross-process sharing**: Multiple graph invocations in different processes don't share cache. For LangGraph Platform deployments, `PostgresStore` would enable this.
