"""Tests for ``reflector_resolve_node`` and ``decision_writeback_node`` (PR 4)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langgraph.store.memory import InMemoryStore

from muffin_agent.agents.trading_decision.nodes import (
    decision_writeback_node,
    reflector_resolve_node,
)
from muffin_agent.agents.trading_decision.reflection.memory import ReflectionMemory
from muffin_agent.agents.trading_decision.schemas import (
    AnalysisContext,
    Outcome,
)


def _ctx(ticker: str = "AAPL") -> AnalysisContext:
    return AnalysisContext(ticker=ticker, narrative="Notes.")


def _decision(rating: str = "buy") -> dict:
    return {
        "rating": rating,
        "executive_summary": "Buy AAPL.",
        "investment_thesis": "Bull held.",
        "time_horizon": "3-6m",
        "position_sizing": "2% NAV",
        "key_risks_remaining": ["China"],
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


def _fixed_outcome_fetcher(outcome: Outcome | None):
    """Build a stub fetcher that always returns the given outcome."""

    async def fetcher(**kwargs: Any) -> Outcome | None:
        return outcome

    return fetcher


# ── reflector_resolve_node ───────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
class TestReflectorResolveNode:
    async def test_returns_empty_block_when_no_store(self):
        config = {"configurable": {"user_id": "alice"}}
        state = {"analysis_context": _ctx()}
        result = await reflector_resolve_node(state, config, store=None)
        assert result["past_reflections"] == ""
        assert result["resolved_decisions"] == []
        # decision_date defaults to today UTC.
        assert "decision_date" in result

    async def test_returns_empty_block_when_reflection_disabled(self):
        store = InMemoryStore()
        memory = ReflectionMemory(store, user_id="alice")
        await memory.write_pending(
            ticker="AAPL", date="2026-05-10", decision=_decision()
        )
        config = {"configurable": {"user_id": "alice", "reflection_enabled": False}}
        state = {"analysis_context": _ctx()}
        result = await reflector_resolve_node(state, config, store=store)
        assert result["past_reflections"] == ""
        # Pending entry should NOT have been resolved.
        assert len(await memory.list_pending()) == 1

    async def test_returns_empty_block_when_no_user_id(self, monkeypatch):
        # Ensure the debug fallback is not set so user_id is truly unresolvable.
        monkeypatch.delenv("MEMORY_DEBUG_USER_ID", raising=False)
        store = InMemoryStore()
        state = {"analysis_context": _ctx()}
        result = await reflector_resolve_node(state, {}, store=store)
        assert result["past_reflections"] == ""

    async def test_uses_explicit_decision_date_from_config(self):
        config = {"configurable": {"user_id": "alice", "decision_date": "2026-05-17"}}
        state = {"analysis_context": _ctx()}
        result = await reflector_resolve_node(state, config, store=None)
        assert result["decision_date"] == "2026-05-17"

    async def test_resolves_pending_and_persists_reflection(self):
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
        state = {"analysis_context": _ctx()}

        # Stub the reflector LLM via patching generate_reflection.
        with patch(
            "muffin_agent.agents.trading_decision.nodes.generate_reflection",
            AsyncMock(return_value="Bull thesis held; alpha +1.0%."),
        ):
            result = await reflector_resolve_node(
                state,
                config,
                store=store,
                outcomes_fetcher=_fixed_outcome_fetcher(_outcome()),
            )

        # past_reflections block now populated with the resolved entry.
        block = result["past_reflections"]
        assert "AAPL 2026-05-10" in block
        assert "alpha +1.00%" in block
        assert "Bull thesis held" in block
        # Pending entry is gone.
        assert await memory.list_pending() == []
        # Resolved entry exists.
        resolved = await memory.list_resolved_for_ticker("AAPL")
        assert len(resolved) == 1
        assert resolved[0].reflection == "Bull thesis held; alpha +1.0%."

    async def test_leaves_entry_pending_when_outcome_unavailable(self):
        store = InMemoryStore()
        memory = ReflectionMemory(store, user_id="alice")
        await memory.write_pending(
            ticker="AAPL", date="2026-05-10", decision=_decision()
        )
        config = {"configurable": {"user_id": "alice"}}
        state = {"analysis_context": _ctx()}
        # Fetcher returns None — entry stays pending.
        result = await reflector_resolve_node(
            state, config, store=store, outcomes_fetcher=_fixed_outcome_fetcher(None)
        )
        assert result["past_reflections"] == ""
        assert len(await memory.list_pending()) == 1

    async def test_swallows_fetcher_exception(self):
        store = InMemoryStore()
        memory = ReflectionMemory(store, user_id="alice")
        await memory.write_pending(
            ticker="AAPL", date="2026-05-10", decision=_decision()
        )

        async def raising_fetcher(**kwargs: Any) -> Outcome | None:
            raise RuntimeError("MCP down")

        config = {"configurable": {"user_id": "alice"}}
        state = {"analysis_context": _ctx()}
        result = await reflector_resolve_node(
            state, config, store=store, outcomes_fetcher=raising_fetcher
        )
        assert result["past_reflections"] == ""
        # Pending unchanged.
        assert len(await memory.list_pending()) == 1

    async def test_injects_cross_ticker_reflections(self):
        store = InMemoryStore()
        memory = ReflectionMemory(store, user_id="alice")
        # Write 2 pending for AAPL + 2 for MSFT.
        for ticker in ("AAPL", "MSFT"):
            await memory.write_pending(
                ticker=ticker, date="2026-05-10", decision=_decision()
            )

        config = {"configurable": {"user_id": "alice"}}
        state = {"analysis_context": _ctx("AAPL")}

        with patch(
            "muffin_agent.agents.trading_decision.nodes.generate_reflection",
            AsyncMock(return_value="r."),
        ):
            result = await reflector_resolve_node(
                state,
                config,
                store=store,
                outcomes_fetcher=_fixed_outcome_fetcher(_outcome()),
            )

        block = result["past_reflections"]
        # AAPL appears in same-ticker section.
        assert "AAPL 2026-05-10" in block
        # MSFT appears in cross-ticker section.
        assert "MSFT 2026-05-10" in block

    async def test_respects_max_same_ticker_limit(self):
        store = InMemoryStore()
        memory = ReflectionMemory(store, user_id="alice")
        # 3 prior resolved for AAPL.
        for i in range(3):
            date = f"2026-05-{10 - i:02d}"
            await memory.write_pending(ticker="AAPL", date=date, decision=_decision())
            await memory.resolve(
                ticker="AAPL",
                date=date,
                outcome=_outcome(),
                reflection=f"r-{i}",
            )

        config = {"configurable": {"user_id": "alice", "reflection_max_same_ticker": 1}}
        state = {"analysis_context": _ctx("AAPL")}
        result = await reflector_resolve_node(
            state, config, store=store, outcomes_fetcher=_fixed_outcome_fetcher(None)
        )
        block = result["past_reflections"]
        # Only the latest (2026-05-10) reflection appears.
        assert block.count("AAPL") == 1
        assert "r-0" in block


# ── decision_writeback_node ──────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
class TestDecisionWritebackNode:
    async def test_writes_pending_entry(self):
        store = InMemoryStore()
        config = {
            "configurable": {
                "user_id": "alice",
                "decision_date": "2026-05-17",
            }
        }
        state = {
            "analysis_context": _ctx(),
            "portfolio_decision": _decision(),
            "decision_date": "2026-05-17",
        }
        await decision_writeback_node(state, config, store=store)
        memory = ReflectionMemory(store, user_id="alice")
        pending = await memory.list_pending()
        assert len(pending) == 1
        assert pending[0].ticker == "AAPL"
        assert pending[0].date == "2026-05-17"
        assert pending[0].decision["rating"] == "buy"

    async def test_skips_when_no_store(self):
        config = {"configurable": {"user_id": "alice"}}
        state = {
            "analysis_context": _ctx(),
            "portfolio_decision": _decision(),
        }
        result = await decision_writeback_node(state, config, store=None)
        assert result == {}

    async def test_skips_when_reflection_disabled(self):
        store = InMemoryStore()
        config = {"configurable": {"user_id": "alice", "reflection_enabled": False}}
        state = {
            "analysis_context": _ctx(),
            "portfolio_decision": _decision(),
        }
        await decision_writeback_node(state, config, store=store)
        memory = ReflectionMemory(store, user_id="alice")
        assert await memory.list_pending() == []

    async def test_skips_when_no_user_id(self, monkeypatch):
        # Ensure the debug fallback is not set so user_id is truly unresolvable.
        monkeypatch.delenv("MEMORY_DEBUG_USER_ID", raising=False)
        store = InMemoryStore()
        state = {
            "analysis_context": _ctx(),
            "portfolio_decision": _decision(),
        }
        await decision_writeback_node(state, {}, store=store)
        items = await store.asearch(("memories",))
        assert items == []

    async def test_skips_errored_portfolio_decision(self):
        store = InMemoryStore()
        config = {"configurable": {"user_id": "alice"}}
        state = {
            "analysis_context": _ctx(),
            "portfolio_decision": {"rating": "hold", "error": "PM failed"},
        }
        await decision_writeback_node(state, config, store=store)
        memory = ReflectionMemory(store, user_id="alice")
        assert await memory.list_pending() == []
