"""Tests for the council graph wiring and judge node (v4)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from muffin_agent.agents.personas_council import (
    CouncilSynthesisOutput,
    build_council_graph,
    council_judge_node,
)
from muffin_agent.agents.personas_council.council_graph import PERSONA_BUILDERS


@pytest.mark.unit
class TestPersonaBuildersWiring:
    """v4 wiring: the council imports persona factories directly, no registry."""

    def test_persona_builders_list_has_all_13(self):
        assert len(PERSONA_BUILDERS) == 13
        slugs = {slug for slug, _ in PERSONA_BUILDERS}
        assert len(slugs) == 13  # all unique

    def test_persona_builders_are_callable(self):
        for _, builder in PERSONA_BUILDERS:
            assert callable(builder)


@pytest.mark.unit
@pytest.mark.asyncio
class TestCouncilGraphWiring:
    """Compile-time topology checks. MCP is mocked so tests don't hit the network."""

    async def test_graph_compiles_with_all_13_personas(self):
        # Mock the MCP fetch each persona's _build_data_collection_agent uses.
        # Mock the underlying MCP client so MultiServerMCPClient.get_tools()
        # returns nothing without touching the network. Patching the deep
        # entry point catches all persona modules at once (no per-file patch).
        mock_client = AsyncMock()
        mock_client.get_tools = AsyncMock(return_value=[])
        with patch(
            "muffin_agent.agents.data_collection.utils.MultiServerMCPClient",
            return_value=mock_client,
        ):
            graph = await build_council_graph()
        nodes = list(graph.get_graph().nodes)
        # 13 persona slugs + council_judge (+ __start__/__end__).
        assert "council_judge" in nodes
        for slug, _ in PERSONA_BUILDERS:
            assert slug in nodes
        # v4: NO shared persona_data_collection node — each persona owns its
        # own data fetch inside its subgraph.
        assert "persona_data_collection" not in nodes

    async def test_graph_compiles_with_specialists(self):
        # Mock the underlying MCP client so MultiServerMCPClient.get_tools()
        # returns nothing without touching the network. Patching the deep
        # entry point catches all persona modules at once (no per-file patch).
        mock_client = AsyncMock()
        mock_client.get_tools = AsyncMock(return_value=[])
        with patch(
            "muffin_agent.agents.data_collection.utils.MultiServerMCPClient",
            return_value=mock_client,
        ):
            graph = await build_council_graph(
                {"configurable": {"include_specialists": True}}
            )
        nodes = list(graph.get_graph().nodes)
        for slug in (
            "technicals",
            "sentiment",
            "fundamentals",
            "growth",
            "valuation",
            "news_sentiment",
        ):
            assert slug in nodes

    async def test_include_specialists_via_configurable(self):
        # The langgraph factory only receives `config`, so the flag must also be
        # readable from config["configurable"]["include_specialists"].
        mock_client = AsyncMock()
        mock_client.get_tools = AsyncMock(return_value=[])
        with patch(
            "muffin_agent.agents.data_collection.utils.MultiServerMCPClient",
            return_value=mock_client,
        ):
            graph = await build_council_graph(
                {"configurable": {"include_specialists": True}}
            )
        nodes = list(graph.get_graph().nodes)
        assert "technicals" in nodes
        assert "news_sentiment" in nodes


@pytest.mark.unit
@pytest.mark.asyncio
class TestCouncilJudgeNode:
    async def test_returns_fallback_on_no_signals(self):
        result = await council_judge_node({"ticker": "ZAB"}, {})
        synth = result["council_synthesis"]
        assert synth["ticker"] == "ZAB"
        assert synth["consensus_rating"] == "hold"
        assert synth["weighted_confidence"] == 0.0
        assert set(synth["vote_breakdown"]) == {
            "strong_sell",
            "sell",
            "hold",
            "buy",
            "strong_buy",
        }

    async def test_pre_aggregates_signals_and_calls_llm(self):
        signals = [
            {
                "agent_id": "warren_buffett",
                "signal": "strong_buy",
                "confidence": 0.9,
                "reasoning": "Wide moat",
                "evidence": {},
            },
            {
                "agent_id": "ben_graham",
                "signal": "buy",
                "confidence": 0.7,
                "reasoning": "Cheap",
                "evidence": {},
            },
            {
                "agent_id": "cathie_wood",
                "signal": "hold",
                "confidence": 0.5,
                "reasoning": "Mixed",
                "evidence": {},
            },
            {
                "agent_id": "michael_burry",
                "signal": "sell",
                "confidence": 0.6,
                "reasoning": "Overvalued",
                "evidence": {},
            },
        ]
        fake = CouncilSynthesisOutput(
            ticker="ZAB",
            consensus_rating="buy",
            weighted_confidence=0.72,
            vote_breakdown={
                "strong_sell": [],
                "sell": ["michael_burry"],
                "hold": ["cathie_wood"],
                "buy": ["ben_graham"],
                "strong_buy": ["warren_buffett"],
            },
            bull_case_synthesis="Quality compounder.",
            bear_case_synthesis="Valuation is rich.",
            dissent_summary="Burry sees overvaluation.",
            key_uncertainties=["Multiple compression risk"],
            reasoning="Majority constructive with one dissenter.",
        )
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=fake)
        with patch(
            "muffin_agent.agents.personas_council.judge.ModelConfiguration.get_chat_model_for_role",
            return_value=mock_llm,
        ):
            result = await council_judge_node(
                {"ticker": "ZAB", "persona_signals": signals}, {}
            )
        synth = result["council_synthesis"]
        assert synth["consensus_rating"] == "buy"
        assert synth["vote_breakdown"]["strong_buy"] == ["warren_buffett"]
        assert mock_llm.ainvoke.await_count == 1
        sent = mock_llm.ainvoke.call_args.args[0]
        system_prompt = sent[0].content
        assert "warren_buffett" in system_prompt
        assert "Council Judge" in system_prompt
