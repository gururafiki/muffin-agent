"""Shared utilities for investment stage nodes."""

import json
import logging
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from muffin_agent.agents.investment.validators import get_validator
from muffin_agent.config import Configuration

logger = logging.getLogger(__name__)

_AGENTS_MD = Path(__file__).resolve().parent / "AGENTS.md"
"""Path to the shared cross-run memory file for investment agents."""


def load_agent_memory() -> str:
    """Load cross-run memory from ``AGENTS.md``.

    Returns the file contents, or an empty string if the file is missing
    or contains only the seed template (no real observations yet).
    """
    try:
        content = _AGENTS_MD.read_text()
    except FileNotFoundError:
        return ""

    # Skip injection if the file has no real entries (only seed template)
    for section in ("## Observations", "## Sector Trends", "## Model Calibration"):
        idx = content.find(section)
        if idx == -1:
            continue
        # Find the next section or end of file
        next_section = content.find("\n## ", idx + len(section))
        end = next_section if next_section != -1 else len(content)
        block = content[idx + len(section) : end]
        # Strip HTML comments and whitespace
        stripped = block.replace("<!-- ", "").replace(" -->", "")
        seed_prefixes = ("Append ", "Notes on ", "Format:")
        for line in stripped.strip().splitlines():
            line = line.strip()
            if line and not any(line.startswith(p) for p in seed_prefixes):
                return content  # Has real content
    return ""

# Transient errors that should propagate so LangGraph RetryPolicy can retry
# the node. All other exceptions are caught and produce a fallback dict.
TRANSIENT_ERRORS = (
    ConnectionError,
    TimeoutError,
    httpx.NetworkError,
    httpx.TimeoutException,
)


async def run_deep_agent_node(
    state: dict[str, Any],
    config: RunnableConfig,
    agent_factory: Callable[[Configuration], Coroutine[Any, Any, Any]],
    input_state_type: type,
    state_key: str,
    error_fallback: dict[str, Any] | None = None,
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
        agent_factory: Async callable ``(Configuration) -> deep_agent``.
        input_state_type: TypedDict class whose ``__annotations__`` keys are read
            from *state* to build the agent's input context.
        state_key: Key under which to write the result in the returned state update.
        error_fallback: Extra fields merged into the error dict on failure.
            For example ``{"regime_label": "unknown"}`` for market_regime_node.

    Returns:
        A single-key dict ``{state_key: <structured output or error dict>}``.
    """
    fallback = error_fallback or {}

    try:
        configuration = Configuration.from_runnable_config(config)
        agent = await agent_factory(configuration)

        context = {
            k: state[k]
            for k in input_state_type.__annotations__
            if state.get(k)
        }
        result = await agent.ainvoke({"input": json.dumps(context)})

        structured = (
            result.get("structured_response") if isinstance(result, dict) else None
        )
        if structured is None:
            raw = (
                result.get("output", "")
                if isinstance(result, dict)
                else str(result)
            )
            return {
                state_key: {
                    **fallback,
                    "error": "Agent did not produce structured output",
                    "raw_output": raw,
                }
            }

        result_dict = structured.model_dump()
        validator = get_validator(type(structured))
        if validator:
            validation_warnings = validator(result_dict)
            if validation_warnings:
                result_dict["_validation_warnings"] = validation_warnings
                logger.warning(
                    "Validation warnings for '%s': %s",
                    state_key,
                    validation_warnings,
                )
        return {state_key: result_dict}

    except TRANSIENT_ERRORS:
        logger.warning(
            "Transient error in '%s', propagating for retry", state_key
        )
        raise
    except Exception:
        logger.exception("Investment node '%s' failed", state_key)
        return {
            state_key: {
                **fallback,
                "error": "Agent raised an exception",
            }
        }
