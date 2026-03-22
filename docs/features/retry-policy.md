# Retry Policy for Investment Nodes

## Problem

Investment stage nodes (market regime, sector analysis, company analysis, etc.) call external services via MCP tools and sandbox containers. These calls can fail transiently due to network timeouts, rate limits, or container restarts.

LangGraph provides `RetryPolicy` for automatic node-level retries, but it only works when the node function raises an exception. Previously, `run_deep_agent_node` caught **all** exceptions and returned a fallback error dict — meaning `RetryPolicy` never had a chance to fire.

## Solution

Split error handling in `run_deep_agent_node` into two categories:

1. **Transient errors** — re-raised so LangGraph `RetryPolicy` can retry the node
2. **Permanent errors** — caught and returned as fallback dicts (existing behaviour)

## How It Works

### Transient error detection

`TRANSIENT_ERRORS` is a tuple of exception types that indicate temporary failures:

```python
TRANSIENT_ERRORS = (
    ConnectionError,
    TimeoutError,
    httpx.NetworkError,
    httpx.TimeoutException,
)
```

When `run_deep_agent_node` catches one of these, it re-raises instead of returning a fallback:

```python
except TRANSIENT_ERRORS:
    logger.warning("Transient error in '%s', propagating for retry", state_key)
    raise  # RetryPolicy will catch this
except Exception:
    logger.exception("Investment node '%s' failed", state_key)
    return {state_key: {**fallback, "error": "Agent raised an exception"}}
```

### RetryPolicy on graph nodes

All data-dependent nodes are registered with a shared retry policy:

```python
_RETRY = RetryPolicy(max_attempts=2, initial_interval=5.0)

graph.add_node("market_regime", market_regime_node, retry_policy=_RETRY)
graph.add_node("sector_analysis", sector_analysis_node, retry_policy=_RETRY)
graph.add_node("company_analysis", company_analysis_node, retry_policy=_RETRY)
graph.add_node("forecasting", forecasting_node, retry_policy=_RETRY)
graph.add_node("risk_assessment", risk_assessment_node, retry_policy=_RETRY)
graph.add_node("valuation", valuation_node, retry_policy=_RETRY)
```

`thesis_synthesis` has no retry policy because it is a pure reasoning node with no external dependencies.

The equity screening graph applies the same policy to its shared context nodes (`idea_sourcing`, `market_regime`, `sector_analysis`).

## Files Changed

| File | Change |
|------|--------|
| `src/muffin_agent/agents/investment/utils.py` | `TRANSIENT_ERRORS` tuple; split exception handling in `run_deep_agent_node` |
| `src/muffin_agent/agents/investment_analysis.py` | `_RETRY` policy on all data-dependent nodes |
| `src/muffin_agent/agents/equity_screening.py` | Same retry policy on shared context nodes |
| `tests/agents/test_investment_utils.py` | Tests for transient vs permanent error propagation |

## Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_attempts` | 2 | One retry (total 2 attempts). Limits cost while handling transient blips. |
| `initial_interval` | 5.0s | Enough time for rate-limit windows to reset. |

## Verification

```bash
pytest tests/agents/test_investment_utils.py -v
```

Tests verify:
- `ConnectionError` propagates (not caught)
- `TimeoutError` propagates
- `ValueError` (non-transient) produces fallback dict
- Structured output extraction still works normally

## Limitations & Future Work

- **Fixed retry count**: All nodes share the same policy. A future improvement could tune retries per node (e.g., more retries for nodes calling flaky APIs).
- **No exponential backoff**: `initial_interval` is flat. LangGraph `RetryPolicy` supports `backoff_factor` — could be added if retry storms become an issue.
- **Error types are hardcoded**: If new transient error types emerge (e.g., from a new MCP server), `TRANSIENT_ERRORS` must be updated manually.
