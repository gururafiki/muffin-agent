"""End-to-end tests for ``build_conference_graph``."""

from __future__ import annotations

import operator
from typing import Annotated, Any
from unittest.mock import patch

import pytest
from langgraph.graph import END, START
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

from muffin_agent.multi_agent import (
    AlternatingModerator,
    LLMParticipant,
    MaxRoundsTerminator,
    RoundRobinModerator,
    StructuredOutputJudge,
    Turn,
    build_conference_graph,
)
from muffin_agent.multi_agent import judges as judges_mod
from muffin_agent.multi_agent import participants as participants_mod

from .conftest import ai, fake_model_config_seq

# ── Patching helpers ─────────────────────────────────────────────────────────


def _patch_participants(cfg):
    return patch.object(
        participants_mod.ModelConfiguration,
        "from_runnable_config",
        return_value=cfg,
    )


def _patch_judges(cfg):
    return patch.object(
        judges_mod.ModelConfiguration,
        "from_runnable_config",
        return_value=cfg,
    )


# ── Topology tests ───────────────────────────────────────────────────────────


@pytest.mark.unit
class TestBuildConferenceGraphTopology:
    def test_validates_non_empty_participants(self):
        with pytest.raises(ValueError, match="at least one participant"):
            build_conference_graph(
                participants=[],
                moderator=RoundRobinModerator(["x"]),
                terminator=MaxRoundsTerminator(1, 1),
            )

    def test_no_judge_includes_dispatch_and_participants_only(self):
        participants = [
            LLMParticipant("alpha", "multi_agent/_transcript.jinja"),
            LLMParticipant("beta", "multi_agent/_transcript.jinja"),
        ]
        graph: CompiledStateGraph = build_conference_graph(
            participants=participants,
            moderator=RoundRobinModerator(["alpha", "beta"]),
            terminator=MaxRoundsTerminator(1, 2),
        )
        nodes = set(graph.get_graph().nodes.keys())
        assert {"dispatch", "alpha", "beta"} <= nodes
        assert "judge" not in nodes

    def test_with_judge_adds_judge_node(self):
        class _V(BaseModel):
            x: str

        participants = [LLMParticipant("solo", "multi_agent/_transcript.jinja")]
        graph = build_conference_graph(
            participants=participants,
            moderator=RoundRobinModerator(["solo"]),
            terminator=MaxRoundsTerminator(1, 1),
            judge=StructuredOutputJudge("judge", "multi_agent/_transcript.jinja", _V),
        )
        nodes = set(graph.get_graph().nodes.keys())
        assert {"dispatch", "solo", "judge"} <= nodes


# ── End-to-end execution tests ───────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
class TestConferenceExecution:
    async def test_round_robin_three_speakers_one_round(self):
        participants = [
            LLMParticipant("alpha", "multi_agent/_transcript.jinja"),
            LLMParticipant("beta", "multi_agent/_transcript.jinja"),
            LLMParticipant("gamma", "multi_agent/_transcript.jinja"),
        ]
        graph = build_conference_graph(
            participants=participants,
            moderator=RoundRobinModerator(["alpha", "beta", "gamma"]),
            terminator=MaxRoundsTerminator(max_rounds=1, num_participants=3),
        )

        cfg, _ = fake_model_config_seq(ai("a-says"), ai("b-says"), ai("g-says"))
        with _patch_participants(cfg):
            result = await graph.ainvoke({})

        transcript = result["transcript"]
        assert [t["speaker"] for t in transcript] == ["alpha", "beta", "gamma"]
        assert [t["content"] for t in transcript] == ["a-says", "b-says", "g-says"]
        assert all(t["round"] == 1 for t in transcript)

    async def test_round_robin_two_rounds_six_turns(self):
        participants = [
            LLMParticipant("alpha", "multi_agent/_transcript.jinja"),
            LLMParticipant("beta", "multi_agent/_transcript.jinja"),
            LLMParticipant("gamma", "multi_agent/_transcript.jinja"),
        ]
        graph = build_conference_graph(
            participants=participants,
            moderator=RoundRobinModerator(["alpha", "beta", "gamma"]),
            terminator=MaxRoundsTerminator(max_rounds=2, num_participants=3),
        )

        cfg, _ = fake_model_config_seq(
            ai("a1"), ai("b1"), ai("g1"), ai("a2"), ai("b2"), ai("g2")
        )
        with _patch_participants(cfg):
            result = await graph.ainvoke({})

        transcript = result["transcript"]
        speakers = [t["speaker"] for t in transcript]
        rounds = [t["round"] for t in transcript]
        contents = [t["content"] for t in transcript]
        assert speakers == ["alpha", "beta", "gamma", "alpha", "beta", "gamma"]
        assert rounds == [1, 1, 1, 2, 2, 2]
        assert contents == ["a1", "b1", "g1", "a2", "b2", "g2"]

    async def test_alternating_moderator_two_speakers(self):
        participants = [
            LLMParticipant("bull", "multi_agent/_transcript.jinja"),
            LLMParticipant("bear", "multi_agent/_transcript.jinja"),
        ]
        graph = build_conference_graph(
            participants=participants,
            moderator=AlternatingModerator("bull", "bear"),
            terminator=MaxRoundsTerminator(max_rounds=2, num_participants=2),
        )

        cfg, _ = fake_model_config_seq(
            ai("bull1"), ai("bear1"), ai("bull2"), ai("bear2")
        )
        with _patch_participants(cfg):
            result = await graph.ainvoke({})

        assert [t["speaker"] for t in result["transcript"]] == [
            "bull", "bear", "bull", "bear"
        ]

    async def test_judge_runs_once_and_populates_verdict(self):
        class _Verdict(BaseModel):
            decision: str
            confidence: float

        participants = [
            LLMParticipant("solo", "multi_agent/_transcript.jinja"),
        ]
        graph = build_conference_graph(
            participants=participants,
            moderator=RoundRobinModerator(["solo"]),
            terminator=MaxRoundsTerminator(max_rounds=1, num_participants=1),
            judge=StructuredOutputJudge(
                "judge", "multi_agent/_transcript.jinja", _Verdict
            ),
        )

        cfg, _ = fake_model_config_seq(
            ai("solo-says"),
            _Verdict(decision="buy", confidence=0.8),
        )
        with _patch_participants(cfg), _patch_judges(cfg):
            result = await graph.ainvoke({})

        assert len(result["transcript"]) == 1
        assert result["verdict"] == {"decision": "buy", "confidence": 0.8}


# ── State-schema customisation tests ─────────────────────────────────────────


class _ParentState(TypedDict, total=False):
    """Parent-graph state with renamed transcript field + extra context."""

    topic: str
    risk_debate_transcript: Annotated[list[Turn], operator.add]
    next_speaker: str | None
    ticker: str


@pytest.mark.unit
@pytest.mark.asyncio
class TestConferenceWithCustomStateSchema:
    async def test_transcript_field_rename(self):
        participants = [
            LLMParticipant("alpha", "multi_agent/_transcript.jinja"),
            LLMParticipant("beta", "multi_agent/_transcript.jinja"),
        ]
        graph = build_conference_graph(
            participants=participants,
            moderator=RoundRobinModerator(["alpha", "beta"]),
            terminator=MaxRoundsTerminator(max_rounds=1, num_participants=2),
            state_schema=_ParentState,
            transcript_field="risk_debate_transcript",
        )

        cfg, fakes = fake_model_config_seq(ai("alpha-says"), ai("beta-says"))
        with _patch_participants(cfg):
            result = await graph.ainvoke({"ticker": "AAPL", "topic": "evaluate"})

        # Transcript landed under the renamed field.
        assert "risk_debate_transcript" in result
        assert "transcript" not in result  # not in this state schema
        turns: list[Turn] = result["risk_debate_transcript"]
        assert [t["speaker"] for t in turns] == ["alpha", "beta"]

        # The second participant should see the first's turn (via state normalisation).
        # fakes[1] is the LLM called for beta's turn — its system prompt should
        # contain alpha's content.
        system_msg = fakes[1].invocations[0][0]
        assert "alpha: alpha-says" in system_msg.content


# ── Routing-through-empty-conference sanity ─────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
class TestImmediateTermination:
    async def test_zero_max_rounds_stops_at_dispatch(self):
        """``max_rounds=0`` — dispatch immediately terminates, no participants run."""
        participants = [
            LLMParticipant("alpha", "multi_agent/_transcript.jinja"),
        ]
        graph = build_conference_graph(
            participants=participants,
            moderator=RoundRobinModerator(["alpha"]),
            terminator=MaxRoundsTerminator(max_rounds=0, num_participants=1),
        )

        cfg, fakes = fake_model_config_seq(ai("should-not-run"))
        with _patch_participants(cfg):
            result = await graph.ainvoke({})

        # No participant invocations recorded.
        assert fakes == [] or all(not f.invocations for f in fakes)
        # Transcript is empty (or missing — TypedDict total=False).
        assert not result.get("transcript")


# Suppress unused-import warning for END/START re-export hygiene.
_ = (END, START, Any)
