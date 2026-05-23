"""Shared fixtures and helpers for trading_decision tests.

The new architecture invokes LLMs directly via
``ModelConfiguration.from_runnable_config(config).get_llm_for_role(role)``
inside each node, then wraps the returned chat model with
``.with_fallbacks(...).with_retry(...)`` (and optionally
``.with_structured_output(...)``).

To stub this cleanly we expose a ``FakeLLM`` class whose ``with_*`` methods
all return ``self`` and whose ``ainvoke`` returns the response the test
configured. Tests patch ``ModelConfiguration.from_runnable_config`` on the
relevant module to return a stub config whose ``get_llm_for_role`` yields
``[FakeLLM(...)]``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage


class FakeLLM:
    """Test double for a chat model wired through the muffin resolution chain."""

    def __init__(self, response: Any):
        self.response = response
        self.invocations: list[list[Any]] = []

    def with_fallbacks(self, fallbacks):  # noqa: ARG002 — accept and ignore
        return self

    def with_retry(self, **kwargs):  # noqa: ARG002
        return self

    def with_structured_output(self, schema):  # noqa: ARG002
        return self

    async def ainvoke(self, messages, config=None):  # noqa: ARG002
        self.invocations.append(messages)
        return self.response


def fake_model_config(response: Any) -> MagicMock:
    """Build a stub ``ModelConfiguration`` returning ``[FakeLLM(response)]``.

    The returned object can be used as the return value of a
    ``ModelConfiguration.from_runnable_config`` patch. Its
    ``get_llm_for_role(role)`` yields a list containing a single
    ``FakeLLM`` configured with the supplied response.

    Usage::

        with patch.object(
            bull_researcher_module.ModelConfiguration,
            "from_runnable_config",
            return_value=fake_model_config(AIMessage("bull text")),
        ):
            update = await bull_researcher_node(state, {})
    """
    cfg = MagicMock()
    cfg.get_llm_for_role.return_value = [FakeLLM(response)]
    return cfg


def ai(text: str) -> AIMessage:
    """Tiny helper — build an ``AIMessage`` with text content."""
    return AIMessage(content=text)
