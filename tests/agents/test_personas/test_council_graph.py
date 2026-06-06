"""Tests for the council graph wiring, single-persona graph, and judge node."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from muffin_agent.agents.personas import (
    PERSONA_REGISTRY,
    CouncilSynthesisOutput,
    build_council_graph,
    build_single_persona_graph,
    council_judge_node,
)


@pytest.mark.unit
class TestCouncilGraphWiring:
    def test_graph_compiles(self):
        graph = build_council_graph()
        nodes = list(graph.get_graph().nodes)
        # __start__, __end__, persona_data_collection, council_judge, and 13 personas
        assert "persona_data_collection" in nodes
        assert "council_judge" in nodes
        for slug in PERSONA_REGISTRY:
            assert slug in nodes

    def test_module_level_graph_exists(self):
        from muffin_agent.agents.personas.council_graph import graph

        # Should be a CompiledStateGraph
        assert graph is not None
        assert hasattr(graph, "ainvoke")


@pytest.mark.unit
class TestSinglePersonaGraph:
    def test_compiles_for_each_persona(self):
        for slug in PERSONA_REGISTRY:
            g = build_single_persona_graph(slug)
            nodes = list(g.get_graph().nodes)
            assert "persona_data_collection" in nodes
            assert slug in nodes

    def test_unknown_slug_raises(self):
        with pytest.raises(KeyError, match="Unknown persona slug"):
            build_single_persona_graph("not_a_real_persona")


@pytest.mark.unit
@pytest.mark.asyncio
class TestCouncilJudgeNode:
    async def test_returns_fallback_on_no_signals(self):
        result = await council_judge_node({"ticker": "ZAB"}, {})
        synth = result["council_synthesis"]
        assert synth["ticker"] == "ZAB"
        assert synth["consensus_rating"] == "hold"
        assert synth["weighted_confidence"] == 0.0
        # Vote breakdown should still have the 5 expected keys
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
            "muffin_agent.agents.personas.judge.ModelConfiguration.get_chat_model_for_role",
            return_value=mock_llm,
        ):
            result = await council_judge_node(
                {"ticker": "ZAB", "persona_signals": signals}, {}
            )
        synth = result["council_synthesis"]
        assert synth["consensus_rating"] == "buy"
        assert synth["vote_breakdown"]["strong_buy"] == ["warren_buffett"]
        assert mock_llm.ainvoke.await_count == 1
        # System prompt should reference the personas
        sent = mock_llm.ainvoke.call_args.args[0]
        system_prompt = sent[0].content
        assert "warren_buffett" in system_prompt
        assert "Council Judge" in system_prompt
