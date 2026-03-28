"""Shared utilities for investment stage nodes."""

import json
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from ...model_config import ModelConfiguration

logger = logging.getLogger(__name__)


async def run_deep_agent_node(
    state: dict[str, Any],
    config: RunnableConfig,
    agent_factory: Callable[..., Coroutine[Any, Any, Any]],
    input_state_type: type,
    state_key: str,
    error_fallback: dict[str, Any] | None = None,
    store: BaseStore | None = None,
) -> dict[str, Any]:
    """Run a deep agent and extract its structured output into a state update.

    Encapsulates the pattern shared by all investment stage nodes:
    1. Build ``Configuration`` from ``RunnableConfig``
    2. Create the agent via *agent_factory*
    3. Build a context dict from *state* using *input_state_type* annotations
    4. Invoke the agent
    5. Extract ``structured_response`` and return ``{state_key: ...}``
    6. On failure (no structured output or exception), return an error dict

    Args:
        state: The current LangGraph state dict.
        config: LangGraph ``RunnableConfig`` (carries thread_id, configurable, etc.).
        agent_factory: Async callable ``(Configuration, store=...) -> deep_agent``.
        input_state_type: TypedDict class whose ``__annotations__`` keys are read
            from *state* to build the agent's input context.
        state_key: Key under which to write the result in the returned state update.
        error_fallback: Extra fields merged into the error dict on failure.
            For example ``{"regime_label": "unknown"}`` for market_regime_node.
        store: Shared ``BaseStore`` for cross-agent tool result caching.

    Returns:
        A single-key dict ``{state_key: <structured output or error dict>}``.
    """
    fallback = error_fallback or {}

    try:
        configuration = ModelConfiguration.from_runnable_config(config)
        agent = await agent_factory(configuration, store=store)

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
        logger.exception("Investment node '%s' failed", state_key)
        return {
            state_key: {
                **fallback,
                "error": "Agent raised an exception",
            }
        }
