"""E2E integration test — equity-price data-collection ReAct agent.

Worked example #1. Runs the *real* ReAct loop end to end while mocking only the
external boundaries:

* **LLM** — :func:`patch_llm` scripts the model turns (tool call → answer).
* **MCP** — :func:`patch_mcp` serves fixture-backed tools; the real ``get_tools``
  name-filter still selects this agent's allowlist.
* **Sandbox** — :func:`patch_sandbox` (only Test B, which scripts ``execute_python``).

Everything else is real: ``MuffinAgentBuilder``, the middleware stack
(retry / cache / tool-knowledge), the ``ToolNode``, routing, and prompt rendering.

This is the canonical template for any single-graph ReAct agent — copy it,
swap the factory + scripted tool calls, and drop in the tool's fixture file.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from muffin_agent.agents.data_collection.equity_price import (
    create_equity_price_data_collection_agent,
)

from ._harness import final, patch_llm, patch_mcp, patch_sandbox, tool_turn

pytestmark = pytest.mark.asyncio


def _tool_messages(messages: list) -> list[ToolMessage]:
    return [m for m in messages if isinstance(m, ToolMessage)]


def _final_text(messages: list) -> str:
    ai = [m for m in messages if isinstance(m, AIMessage)]
    return ai[-1].content if ai else ""


async def test_quote_happy_path_llm_and_mcp(config):
    """Scripted: call equity_price_quote, then summarize.

    The agent uses ``.with_sandbox()``, so ``patch_sandbox()`` is required (the
    FilesystemMiddleware resolves the backend on every model call) — but the
    sandbox is never *executed* on this path.
    """
    script = (
        tool_turn("equity_price_quote", {"symbol": "AAPL"}),
        final("AAPL is trading near $201.50, up ~0.6% on the day."),
    )
    with patch_mcp(scenario="aapl"), patch_sandbox(), patch_llm(*script) as cursor:
        agent = await create_equity_price_data_collection_agent(config)
        result = await agent.ainvoke(
            {"messages": [HumanMessage("Get the latest quote for AAPL.")]},
            config=config,
        )

    # The real ReAct loop executed the (fixture-backed) MCP tool exactly once.
    tool_msgs = _tool_messages(result["messages"])
    assert len(tool_msgs) == 1
    assert tool_msgs[0].name == "equity_price_quote"
    # Fixture content surfaced verbatim as the ToolMessage (real envelope shape).
    assert '"last_price": 201.5' in tool_msgs[0].content
    assert '"provider": "yfinance"' in tool_msgs[0].content
    # Final scripted answer is the terminal message.
    assert "201.50" in _final_text(result["messages"])
    # The script was consumed exactly: one tool turn + one answer turn.
    assert cursor.consumed == 2


async def test_execute_python_uses_sandbox_seam(config):
    """Scripted: call execute_python; patch_sandbox returns a canned stdout."""
    script = (
        tool_turn("execute_python", {"code": "print(201.5 * 100)"}),
        final("Position value is $20,150."),
    )
    with (
        patch_mcp(scenario="aapl"),
        patch_sandbox(execute_output="20150.0\n"),
        patch_llm(*script) as cursor,
    ):
        agent = await create_equity_price_data_collection_agent(config)
        result = await agent.ainvoke(
            {"messages": [HumanMessage("Compute 100 shares of AAPL at the quote.")]},
            config=config,
        )

    tool_msgs = _tool_messages(result["messages"])
    assert len(tool_msgs) == 1
    assert tool_msgs[0].name == "execute_python"
    assert tool_msgs[0].content.strip() == "20150.0"
    assert "20,150" in _final_text(result["messages"])
    assert cursor.consumed == 2
