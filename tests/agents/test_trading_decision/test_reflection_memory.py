"""Tests for ``ReflectionMemory`` (PR 4 — CRUD over BaseStore)."""

from __future__ import annotations

import pytest
from langgraph.store.memory import InMemoryStore

from muffin_agent.agents.trading_decision.reflection.memory import (
    ReflectionMemory,
    make_key,
    render_reflections_block,
    split_key,
    try_build_reflection_memory,
)
from muffin_agent.agents.trading_decision.schemas import (
    DecisionRecord,
    Outcome,
)


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


@pytest.mark.unit
class TestMakeAndSplitKey:
    def test_make_key_uppercases(self):
        assert make_key("aapl", "2026-05-17") == "AAPL:2026-05-17"

    def test_split_key(self):
        assert split_key("AAPL:2026-05-17") == ("AAPL", "2026-05-17")


@pytest.mark.unit
@pytest.mark.asyncio
class TestReflectionMemory:
    async def test_namespace_includes_user_id(self):
        store = InMemoryStore()
        memory = ReflectionMemory(store, user_id="alice")
        assert memory.namespace == ("memories", "alice", "decisions")

    async def test_requires_non_empty_user_id(self):
        with pytest.raises(ValueError):
            ReflectionMemory(InMemoryStore(), user_id="")

    async def test_write_pending_then_list(self):
        store = InMemoryStore()
        memory = ReflectionMemory(store, user_id="alice")
        await memory.write_pending(
            ticker="AAPL", date="2026-05-17", decision=_decision()
        )
        pending = await memory.list_pending()
        assert len(pending) == 1
        assert pending[0].ticker == "AAPL"
        assert pending[0].date == "2026-05-17"
        assert pending[0].status == "pending"
        assert pending[0].outcome is None

    async def test_write_pending_idempotent(self):
        store = InMemoryStore()
        memory = ReflectionMemory(store, user_id="alice")
        # Two writes for same key — second overwrites.
        await memory.write_pending(
            ticker="AAPL", date="2026-05-17", decision=_decision("buy")
        )
        await memory.write_pending(
            ticker="AAPL", date="2026-05-17", decision=_decision("strong_buy")
        )
        pending = await memory.list_pending()
        assert len(pending) == 1
        assert pending[0].decision["rating"] == "strong_buy"

    async def test_write_pending_skips_errored_decision(self):
        store = InMemoryStore()
        memory = ReflectionMemory(store, user_id="alice")
        await memory.write_pending(
            ticker="AAPL",
            date="2026-05-17",
            decision={"rating": "hold", "error": "judge failed"},
        )
        assert await memory.list_pending() == []

    async def test_resolve_transitions_pending_to_resolved(self):
        store = InMemoryStore()
        memory = ReflectionMemory(store, user_id="alice")
        await memory.write_pending(
            ticker="AAPL", date="2026-05-17", decision=_decision()
        )
        await memory.resolve(
            ticker="AAPL",
            date="2026-05-17",
            outcome=_outcome(),
            reflection="Bull held; alpha +1.0%.",
        )
        # No more pending entries.
        assert await memory.list_pending() == []
        # Resolved entry appears.
        resolved = await memory.list_resolved_for_ticker("AAPL")
        assert len(resolved) == 1
        assert resolved[0].status == "resolved"
        assert resolved[0].outcome is not None
        assert resolved[0].outcome.raw_return_pct == 5.0
        assert resolved[0].reflection == "Bull held; alpha +1.0%."

    async def test_resolve_noop_when_no_pending(self):
        store = InMemoryStore()
        memory = ReflectionMemory(store, user_id="alice")
        # No prior write — resolve silently does nothing.
        await memory.resolve(
            ticker="AAPL",
            date="2026-05-17",
            outcome=_outcome(),
            reflection="ignored",
        )
        assert await memory.list_pending() == []
        assert await memory.list_resolved_for_ticker("AAPL") == []

    async def test_list_resolved_sorts_by_date_desc(self):
        store = InMemoryStore()
        memory = ReflectionMemory(store, user_id="alice")
        # Three resolved entries with different dates.
        for date in ("2026-01-01", "2026-03-15", "2026-05-17"):
            await memory.write_pending(ticker="AAPL", date=date, decision=_decision())
            await memory.resolve(
                ticker="AAPL",
                date=date,
                outcome=_outcome(),
                reflection=f"r-{date}",
            )
        resolved = await memory.list_resolved_for_ticker("AAPL", limit=10)
        assert [r.date for r in resolved] == ["2026-05-17", "2026-03-15", "2026-01-01"]

    async def test_list_resolved_respects_limit(self):
        store = InMemoryStore()
        memory = ReflectionMemory(store, user_id="alice")
        for i in range(5):
            date = f"2026-05-{17 - i:02d}"
            await memory.write_pending(ticker="AAPL", date=date, decision=_decision())
            await memory.resolve(
                ticker="AAPL",
                date=date,
                outcome=_outcome(),
                reflection=f"r-{i}",
            )
        resolved = await memory.list_resolved_for_ticker("AAPL", limit=2)
        assert len(resolved) == 2
        # Latest first.
        assert resolved[0].date > resolved[1].date

    async def test_cross_ticker_excludes_target_ticker(self):
        store = InMemoryStore()
        memory = ReflectionMemory(store, user_id="alice")
        for ticker in ("AAPL", "MSFT", "NVDA"):
            await memory.write_pending(
                ticker=ticker, date="2026-05-17", decision=_decision()
            )
            await memory.resolve(
                ticker=ticker,
                date="2026-05-17",
                outcome=_outcome(),
                reflection=f"r-{ticker}",
            )
        cross = await memory.list_resolved_cross_ticker(exclude_ticker="AAPL")
        tickers = [r.ticker for r in cross]
        assert "AAPL" not in tickers
        assert set(tickers) == {"MSFT", "NVDA"}

    async def test_per_user_isolation(self):
        store = InMemoryStore()
        alice = ReflectionMemory(store, user_id="alice")
        bob = ReflectionMemory(store, user_id="bob")
        await alice.write_pending(
            ticker="AAPL", date="2026-05-17", decision=_decision()
        )
        # Bob should see no pending entries.
        assert await bob.list_pending() == []
        # Alice should see her entry.
        assert len(await alice.list_pending()) == 1


@pytest.mark.unit
class TestRenderReflectionsBlock:
    def _resolved(self, ticker: str, date: str, rating: str = "buy") -> DecisionRecord:
        return DecisionRecord(
            ticker=ticker,
            date=date,
            status="resolved",
            decision=_decision(rating),
            outcome=_outcome(),
            reflection=f"Reflection for {ticker} {date}.",
        )

    def test_empty_returns_empty_string(self):
        assert render_reflections_block(same_ticker=[], cross_ticker=[]) == ""

    def test_same_ticker_only(self):
        block = render_reflections_block(
            same_ticker=[self._resolved("AAPL", "2026-05-17")],
            cross_ticker=[],
        )
        assert "Past same-ticker decisions" in block
        assert "AAPL 2026-05-17" in block
        assert "raw +5.00%" in block
        assert "alpha +1.00%" in block

    def test_cross_ticker_only(self):
        block = render_reflections_block(
            same_ticker=[],
            cross_ticker=[self._resolved("MSFT", "2026-04-10")],
        )
        assert "Past cross-ticker reflections" in block
        assert "MSFT 2026-04-10" in block

    def test_both_sections_present(self):
        block = render_reflections_block(
            same_ticker=[self._resolved("AAPL", "2026-05-17")],
            cross_ticker=[self._resolved("MSFT", "2026-04-10")],
        )
        assert "Past same-ticker" in block
        assert "Past cross-ticker" in block
        assert block.index("same-ticker") < block.index("cross-ticker")


@pytest.mark.unit
class TestTryBuildReflectionMemory:
    """Gating factory used by both reflection bookend nodes — short-circuits
    on any of: no store, reflection disabled, no user_id."""

    def test_returns_none_when_store_is_none(self):
        config = {"configurable": {"user_id": "alice"}}
        assert try_build_reflection_memory(config, None) is None

    def test_returns_none_when_reflection_disabled(self):
        store = InMemoryStore()
        config = {
            "configurable": {"user_id": "alice", "reflection_enabled": False},
        }
        assert try_build_reflection_memory(config, store) is None

    def test_returns_none_when_user_id_unresolvable(self, monkeypatch):
        monkeypatch.delenv("MEMORY_DEBUG_USER_ID", raising=False)
        store = InMemoryStore()
        config = {"configurable": {}}
        assert try_build_reflection_memory(config, store) is None

    def test_returns_memory_when_everything_wired(self):
        store = InMemoryStore()
        config = {"configurable": {"user_id": "alice"}}
        memory = try_build_reflection_memory(config, store)
        assert memory is not None
        assert isinstance(memory, ReflectionMemory)
        assert memory.namespace == ("memories", "alice", "decisions")

    def test_uses_memory_debug_user_id_fallback(self, monkeypatch):
        monkeypatch.setenv("MEMORY_DEBUG_USER_ID", "debug-alex")
        store = InMemoryStore()
        config = {"configurable": {}}
        memory = try_build_reflection_memory(config, store)
        assert memory is not None
        assert memory.namespace == ("memories", "debug-alex", "decisions")
