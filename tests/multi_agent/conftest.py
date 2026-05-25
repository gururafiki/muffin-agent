"""Shared test helpers for the multi_agent conference framework.

Mirrors the pattern in ``tests/agents/test_trading_decision/conftest.py`` —
participants and judges call ``ModelConfiguration.get_chat_model_for_role``
which composes the returned chat model via ``with_fallbacks`` / ``with_retry``
/ optional ``with_structured_output``. ``FakeLLM`` short-circuits all of
those by returning ``self`` so tests drive a single configured response.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage


class FakeLLM:
    """Test double for a chat model wired through the muffin resolution chain.

    ``with_fallbacks`` / ``with_retry`` / ``with_structured_output`` all
    return ``self`` so the surrounding builder calls don't break. ``ainvoke``
    returns the response the test configured and records the message list it
    was called with (for assertions on prompt content).
    """

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


def fake_model_config(response: Any) -> tuple[MagicMock, FakeLLM]:
    """Build a stub ``ModelConfiguration`` with a single shared FakeLLM.

    Returns ``(config_mock, fake_llm)`` so the test can assert on
    ``fake_llm.invocations`` after the participant runs. Use this for
    tests that exercise ONE LLM call.
    """
    fake_llm = FakeLLM(response)
    cfg = MagicMock()
    cfg.get_llm_for_role.return_value = [fake_llm]
    return cfg, fake_llm


def fake_model_config_seq(
    *responses: Any,
) -> tuple[MagicMock, list[FakeLLM]]:
    """Build a stub returning a fresh FakeLLM per ``get_llm_for_role`` call.

    Each call pops the next response from ``responses``; the last response
    is reused if the queue runs out (so tests that don't care about
    overflow don't have to repeat themselves). Returns
    ``(config_mock, fakes_list)`` — ``fakes_list`` is populated as each
    LLM call creates a new FakeLLM, so the test can inspect
    ``fakes_list[i].invocations`` per-turn after the conference runs.
    """
    cfg = MagicMock()
    queue: list[Any] = list(responses) or [AIMessage("default")]
    fakes: list[FakeLLM] = []
    counter = {"i": 0}

    def _get_llm_for_role(role: str):  # noqa: ARG001
        i = min(counter["i"], len(queue) - 1)
        counter["i"] += 1
        fake = FakeLLM(queue[i])
        fakes.append(fake)
        return [fake]

    cfg.get_llm_for_role.side_effect = _get_llm_for_role
    return cfg, fakes


def ai(text: str) -> AIMessage:
    """Build an ``AIMessage`` with text content."""
    return AIMessage(content=text)
