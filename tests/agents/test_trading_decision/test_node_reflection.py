"""Tests for the reflection-layer node functions + reflector helper."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langgraph.store.memory import InMemoryStore

from muffin_agent.agents.trading_decision.reflection import (
    reflector as reflector_module,
)
from muffin_agent.agents.trading_decision.reflection.memory import ReflectionMemory
from muffin_agent.agents.trading_decision.reflection.reflector import (
    reflect_on_decision,
)
from muffin_agent.agents.trading_decision.reflection.resolver import (
    reflector_resolve_node,
)
from muffin_agent.agents.trading_decision.reflection.writeback import (
    decision_writeback_node,
)
from muffin_agent.agents.trading_decision.schemas import Outcome

from .conftest import ai, fake_model_config


def _decision(rating: str = "buy") -> dict:
    return {
        "rating": rating,
        "executive_summary": "exec.",
        "investment_thesis": "thesis.",
        "time_horizon": "3-6m",
        "position_sizing": "2% NAV",
        "key_risks_remaining": [],
        "confidence": 0.7,
        "incorporates_past_lessons": False,
    }


def _outcome(raw: float = 5.0, alpha: float = 1.0) -> Outcome:
    return Outcome(
        raw_return_pct=raw,
        alpha_return_pct=alpha,
        holding_days=5,
        decision_action="buy",
    )


def _fixed_fetcher(outcome: Outcome | None):
    async def fetcher(**kwargs: Any) -> Outcome | None:
        return outcome

    return fetcher


# ── reflect_on_decision ──────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
class TestReflectOnDecision:
    async def test_returns_llm_text(self):
        with patch.object(
            reflector_module.ModelConfiguration,
            "from_runnable_config",
            return_value=fake_model_config(ai("Bull held; alpha +1.0%.")),
        ):
            text = await reflect_on_decision(
                {},
                ticker="AAPL",
                decision_date="2026-05-17",
                decision=_decision(),
                outcome=_outcome().model_dump(),
            )

        assert text == "Bull held; alpha +1.0%."

    async def test_includes_decision_and_outcome_in_prompt(self):
        captured = fake_model_config(ai("reflection text"))
        with patch.object(
            reflector_module.ModelConfiguration,
            "from_runnable_config",
            return_value=captured,
        ):
            await reflect_on_decision(
                {},
                ticker="AAPL",
                decision_date="2026-05-17",
                decision=_decision(),
                outcome=_outcome(raw=4.2, alpha=1.3).model_dump(),
            )

        system_msg = captured.get_llm_for_role.return_value[0].invocations[0][0]
        assert "AAPL" in system_msg.content
        assert "2026-05-17" in system_msg.content
        assert '"raw_return_pct": 4.2' in system_msg.content


# ── reflector_resolve_node ────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
class TestReflectorResolveNode:
    async def test_returns_empty_block_when_no_store(self):
        config = {"configurable": {"user_id": "alice"}}
        state = {"ticker": "AAPL"}
        result = await reflector_resolve_node(state, config, store=None)
        assert result["past_reflections"] == ""
        assert result["resolved_decisions"] == []
        assert "decision_date" in result

    async def test_returns_empty_when_reflection_disabled(self):
        store = InMemoryStore()
        memory = ReflectionMemory(store, user_id="alice")
        await memory.write_pending(
            ticker="AAPL", date="2026-05-10", decision=_decision()
        )
        config = {"configurable": {"user_id": "alice", "reflection_enabled": False}}
        state = {"ticker": "AAPL"}
        result = await reflector_resolve_node(state, config, store=store)

        assert result["past_reflections"] == ""
        # Pending was NOT resolved.
        assert len(await memory.list_pending()) == 1

    async def test_returns_empty_when_no_user_id(self, monkeypatch):
        monkeypatch.delenv("MEMORY_DEBUG_USER_ID", raising=False)
        store = InMemoryStore()
        state = {"ticker": "AAPL"}
        result = await reflector_resolve_node(state, {}, store=store)
        assert result["past_reflections"] == ""

    async def test_resolves_pending_and_persists(self):
        store = InMemoryStore()
        memory = ReflectionMemory(store, user_id="alice")
        await memory.write_pending(
            ticker="AAPL", date="2026-05-10", decision=_decision()
        )
        config = {
            "configurable": {
                "user_id": "alice",
                "decision_date": "2026-05-17",
            }
        }
        state = {"ticker": "AAPL"}

        with patch(
            "muffin_agent.agents.trading_decision.reflection.resolver.reflect_on_decision",
            AsyncMock(return_value="Bull thesis held; alpha +1.0%."),
        ):
            result = await reflector_resolve_node(
                state,
                config,
                store=store,
                outcomes_fetcher=_fixed_fetcher(_outcome()),
            )

        # Past reflections block contains the now-resolved entry.
        block = result["past_reflections"]
        assert "AAPL 2026-05-10" in block
        assert "Bull thesis held" in block

        # Pending is gone, resolved exists.
        assert await memory.list_pending() == []
        resolved = await memory.list_resolved_for_ticker("AAPL")
        assert len(resolved) == 1
        assert resolved[0].reflection == "Bull thesis held; alpha +1.0%."

    async def test_leaves_pending_when_outcome_unavailable(self):
        store = InMemoryStore()
        memory = ReflectionMemory(store, user_id="alice")
        await memory.write_pending(
            ticker="AAPL", date="2026-05-10", decision=_decision()
        )
        config = {"configurable": {"user_id": "alice"}}
        state = {"ticker": "AAPL"}

        # Fetcher returns None — entry should stay pending.
        result = await reflector_resolve_node(
            state, config, store=store, outcomes_fetcher=_fixed_fetcher(None)
        )

        assert result["past_reflections"] == ""
        assert len(await memory.list_pending()) == 1

    async def test_decision_date_override_from_config(self):
        config = {"configurable": {"user_id": "alice", "decision_date": "2026-05-17"}}
        state = {"ticker": "AAPL"}
        result = await reflector_resolve_node(state, config, store=None)
        assert result["decision_date"] == "2026-05-17"


# ── decision_writeback_node ──────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
class TestDecisionWritebackNode:
    async def test_writes_pending_entry(self):
        store = InMemoryStore()
        config = {"configurable": {"user_id": "alice"}}
        state = {
            "ticker": "AAPL",
            "portfolio_decision": _decision(),
            "decision_date": "2026-05-17",
        }
        await decision_writeback_node(state, config, store=store)

        memory = ReflectionMemory(store, user_id="alice")
        pending = await memory.list_pending()
        assert len(pending) == 1
        assert pending[0].ticker == "AAPL"
        assert pending[0].date == "2026-05-17"

    async def test_skips_when_reflection_disabled(self):
        store = InMemoryStore()
        config = {"configurable": {"user_id": "alice", "reflection_enabled": False}}
        state = {
            "ticker": "AAPL",
            "portfolio_decision": _decision(),
            "decision_date": "2026-05-17",
        }
        await decision_writeback_node(state, config, store=store)

        memory = ReflectionMemory(store, user_id="alice")
        assert await memory.list_pending() == []

    async def test_skips_errored_decision(self):
        store = InMemoryStore()
        config = {"configurable": {"user_id": "alice"}}
        state = {
            "ticker": "AAPL",
            "portfolio_decision": {"rating": "hold", "error": "PM failed"},
            "decision_date": "2026-05-17",
        }
        await decision_writeback_node(state, config, store=store)

        memory = ReflectionMemory(store, user_id="alice")
        assert await memory.list_pending() == []
