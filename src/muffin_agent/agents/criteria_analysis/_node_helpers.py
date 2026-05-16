"""Shared invocation helpers for criteria_analysis graph nodes.

The orchestrator nodes share a common pattern: build a deep agent, invoke
it with a JSON-serialised context dict, extract ``structured_response``,
return a single state-key update or an error fallback dict.

Mirrors ``investment.utils.run_deep_agent_node`` but passes the original
``RunnableConfig`` through to the factory (rather than converting to
``ModelConfiguration`` first), which is the safer of the two patterns
when factories build subagents that themselves call
``McpConfiguration.from_runnable_config(config)``.
"""

import json
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

logger = logging.getLogger(__name__)


async def invoke_structured_agent(
    *,
    state: dict[str, Any],
    config: RunnableConfig,
    agent_factory: Callable[..., Coroutine[Any, Any, Any]],
    input_state_type: type,
    state_key: str,
    error_fallback: dict[str, Any] | None = None,
    store: BaseStore | None = None,
) -> dict[str, Any]:
    """Build a deep agent, invoke it, extract structured output.

    Args:
        state: Outer state dict.
        config: Original ``RunnableConfig`` (passed through to factory).
        agent_factory: Async callable ``(RunnableConfig, store=...) -> agent``.
        input_state_type: TypedDict whose annotation keys are read from
            *state* to build the agent's input context.
        state_key: Key under which to write the result.
        error_fallback: Extra fields merged into the error dict on failure.
        store: Shared ``BaseStore`` for cross-agent caching.

    Returns:
        ``{state_key: <dump or error dict>}``.
    """
    fallback = error_fallback or {}

    try:
        agent = await agent_factory(config, store=store)
        context = {
            k: state[k] for k in input_state_type.__annotations__ if state.get(k)
        }
        result = await agent.ainvoke({"input": json.dumps(context)})
        structured = (
            result.get("structured_response") if isinstance(result, dict) else None
        )
        if structured is None:
            raw = result.get("output", "") if isinstance(result, dict) else str(result)
            return {
                state_key: {
                    **fallback,
                    "error": "Agent did not produce structured output",
                    "raw_output": raw,
                }
            }
        return {state_key: structured.model_dump()}
    except Exception:
        logger.exception("criteria_analysis node '%s' failed", state_key)
        return {
            state_key: {
                **fallback,
                "error": "Agent raised an exception",
            }
        }
