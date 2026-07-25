"""Tests for offloading heavy per-subagent detail to the Store."""

import pytest
from langgraph.store.memory import InMemoryStore

from muffin_agent.middlewares.agent_capture.detail_store import offload_subagent_detail


@pytest.mark.unit
@pytest.mark.asyncio
async def test_offload_roundtrip() -> None:
    store = InMemoryStore()
    ok = await offload_subagent_detail(
        store,
        "t1",
        "p:1",
        messages=[{"type": "human", "content": "hi"}],
        tool_runs=[{"tool": "x"}],
        output={"s": 1},
    )
    assert ok
    item = await store.aget(("subagent_detail", "t1"), "p:1")
    assert item.value["messages"]
    assert item.value["tool_runs"]
    assert item.value["output"] == {"s": 1}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_offload_none_store_is_noop() -> None:
    assert (
        await offload_subagent_detail(
            None, "t1", "p:1", messages=[], tool_runs=[], output=None
        )
        is False
    )
