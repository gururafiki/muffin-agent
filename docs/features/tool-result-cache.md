# Tool Result Cache Middleware

## Problem

Within a single agent invocation, identical tool calls can be executed multiple times:

- Deep agent orchestration loops may re-invoke subagents that call the same MCP tools
- Agent retries after partial failures repeat tool calls that already succeeded
- Different subagents may independently fetch the same data (e.g., both `economy-macro` and `forecasting` subagents calling the same FRED series)

Previously, only `ToolErrorHandler` deduplicated — and only for **permanently failed** tool calls. Successful results had no deduplication, leading to redundant API calls, increased latency, and higher MCP server load.

## Solution

`ToolResultCacheMiddleware` caches successful tool results by `(tool_name, args)` key via graph state and returns cached results on duplicate calls. Works for both investment agents (`create_deep_agent`) and data collection agents (`create_agent`) since both accept `middleware=`.

### Design principles

1. **No instance mutation after `__init__`**: Per Deep Agents middleware docs, mutating `self.x` in middleware hooks leads to race conditions with concurrent tool calls. All state writes use `Command(update=...)` to update graph state immutably. Only `self.cacheable_tools` is set at construction time (as a `frozenset` for explicit immutability).

2. **Graph state as cache**: The cache lives in `cached_tool_results: dict[str, str]` in graph state, making it thread-scoped, concurrency-safe, and automatically cleaned up when the graph run completes.

## How It Works

### Cache key

A deterministic key is created from tool name and sorted args:

```python
def cache_key(tool_call: dict) -> str:
    args_json = json.dumps(tool_call.get("args", {}), sort_keys=True)
    return f"{tool_call['name']}:{args_json}"
```

### Middleware flow

```
Tool call arrives
    │
    ├─ Not in cacheable_tools? → Pass through to handler
    │
    ├─ Key in cached_tool_results? → Return ToolMessage("[cached] ...")
    │
    └─ Execute handler
         │
         ├─ Success? → Command(update={cached_tool_results: {key: content}})
         │
         └─ Error? → Return as-is (not cached)
```

### State schema

```python
class ToolResultCacheState(AgentState):
    cached_tool_results: NotRequired[Annotated[dict[str, str], operator.or_]]
```

The `operator.or_` reducer merges cache entries from concurrent tool calls without overwriting.

### Middleware composition with ToolErrorHandler

Data collection agents compose both middleware:

```python
middleware=[
    ToolErrorHandler(),              # outermost: blocks duplicate permanent failures
    ToolResultCacheMiddleware(),     # inner: caches successful results
]
```

**Order matters**: `ToolErrorHandler` is first (outermost). For a duplicate call to a permanently failed tool, `ToolErrorHandler` blocks it before `ToolResultCacheMiddleware` sees it. For a duplicate call to a successful tool, `ToolErrorHandler` passes through and `ToolResultCacheMiddleware` returns the cached result.

## Files Changed

| File | Change |
|------|--------|
| `src/muffin_agent/agents/middleware.py` | New: `cache_key()`, `ToolResultCacheState`, `ToolResultCacheMiddleware` |
| `src/muffin_agent/agents/investment/market_regime.py` | Add middleware with cacheable computation tools |
| `src/muffin_agent/agents/investment/sector_analysis.py` | Same |
| `src/muffin_agent/agents/investment/company_analysis.py` | Same |
| `src/muffin_agent/agents/investment/forecasting.py` | Same |
| `src/muffin_agent/agents/data_collection/utils.py` | Import shared `cache_key()`, compose with `ToolErrorHandler` |
| All 13 data collection agent files | Add `ToolResultCacheMiddleware()` to middleware list |
| `tests/agents/test_middleware.py` | 9 tests: cache hit/miss, whitelist filtering, error non-caching |

## Configuration

### Investment agents (deterministic computation tools)

Each investment agent caches its own set of deterministic tools via `cacheable_tools`:

```python
ToolResultCacheMiddleware(
    cacheable_tools=frozenset({
        "compute_yield_curve_metrics",
        "compute_factor_zscore",
        "compute_vix_regime",
        "compute_sector_relative_performance",
    })
)
```

### Data collection agents (all MCP tools)

MCP tools are cached without filtering — within a single analysis run, market data does not change:

```python
ToolResultCacheMiddleware()  # cacheable_tools=None → cache all
```

## Verification

```bash
pytest tests/agents/test_middleware.py -v
```

Tests verify:
- First call executes handler and caches result
- Second call with same args returns `[cached]` prefix without executing handler
- Different args are not cached together
- Error results are not cached
- `cacheable_tools` whitelist correctly filters
- Non-whitelisted tools pass through uncached

## Limitations & Future Work

- **No cache eviction**: Cache grows unbounded within a single run. Not a practical issue (single-run tool calls are bounded by agent turn limits), but could matter for very long-running analyses.
- **String-only cache values**: `cached_tool_results` stores `dict[str, str]`. Large tool results (e.g., full financial statements) are stored as strings, which increases graph state size.
- **No cross-run caching**: Cache is scoped to a single graph invocation. A future improvement could use LangGraph's `Store` for persistent caching of market data with TTL-based expiry (e.g., "FRED GDP data is valid for 24 hours").
