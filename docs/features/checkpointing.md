# Checkpointing Support

## Problem

When running investment analysis via the CLI, a failure midway through the 7-stage pipeline (e.g., a network outage during forecasting) requires restarting the entire pipeline from scratch. There is no way to resume from the last successful node.

On LangGraph Platform (Docker deployment), the server automatically injects a `PostgresSaver` checkpointer via `DATABASE_URI`. However, the standalone graph builder functions (`build_investment_analysis_graph`, `build_equity_screening_graph`) did not accept a checkpointer parameter, so CLI usage had no persistence.

## Solution

Add an optional `checkpointer` parameter to both graph builder functions. This enables state persistence for CLI usage while remaining a no-op for LangGraph Platform (where the server handles checkpointing).

## How It Works

### Per-ticker analysis graph

```python
from langgraph.checkpoint.base import BaseCheckpointSaver

def build_investment_analysis_graph(
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    ...
    return graph.compile(checkpointer=checkpointer)
```

### Equity screening graph

```python
def build_equity_screening_graph(
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    ...
    return graph.compile(checkpointer=checkpointer)
```

### Subgraph scoping

The screening graph calls `build_investment_analysis_graph()` internally for each ticker. The inner per-ticker subgraph is **always** compiled without a checkpointer — only the outer screening graph gets one. This follows LangGraph's subgraph checkpointer scoping: the outer checkpointer captures the full state including subgraph results, while the inner subgraph runs without its own persistence layer.

### Usage examples

**CLI (with persistence)**:
```python
from langgraph.checkpoint.memory import InMemorySaver

graph = build_investment_analysis_graph(checkpointer=InMemorySaver())
result = await graph.ainvoke(state, config={"configurable": {"thread_id": "my-analysis"}})

# Resume after failure:
result = await graph.ainvoke(None, config={"configurable": {"thread_id": "my-analysis"}})

# Time travel:
for state in graph.get_state_history(config):
    print(state.metadata["step"], state.values.keys())
```

**LangGraph Platform (automatic)**: Pass `None` (default). The server injects `PostgresSaver` via `DATABASE_URI` at deployment time.

**SQLite (durable CLI persistence)**:
```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as saver:
    graph = build_investment_analysis_graph(checkpointer=saver)
```

## Files Changed

| File | Change |
|------|--------|
| `src/muffin_agent/agents/investment_analysis.py` | `checkpointer` param on `build_investment_analysis_graph()` |
| `src/muffin_agent/agents/equity_screening.py` | `checkpointer` param on `build_equity_screening_graph()` |
| `tests/agents/test_investment_utils.py` | Tests for graph compilation with and without checkpointer |

## Configuration

| Context | Checkpointer | Notes |
|---------|-------------|-------|
| CLI (development) | `InMemorySaver()` | In-memory, lost on process exit |
| CLI (durable) | `AsyncSqliteSaver` | Persists to disk, survives restarts |
| LangGraph Platform | `None` (default) | Server auto-injects `PostgresSaver` |

## Verification

```bash
pytest tests/agents/test_investment_utils.py -v
```

Tests verify:
- Graph compiles successfully with `checkpointer=None` (default)
- Graph compiles successfully with `InMemorySaver()`
- Existing tests pass unchanged (backward compatible)

## Limitations & Future Work

- **No automatic resume in CLI**: The CLI entrypoint does not yet wire up a checkpointer or detect incomplete runs. This is infrastructure only — the CLI integration is a future task.
- **In-memory only tested**: SQLite and Postgres checkpointers are not tested in the unit test suite (would require database fixtures).
- **No subgraph checkpointing**: Inner per-ticker subgraphs in the screening graph do not checkpoint individually. A failure mid-ticker requires re-running that ticker from scratch (though outer state is preserved).
