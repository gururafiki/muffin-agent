"""End-to-end tests for ``build_conference_graph``."""

from __future__ import annotations

import operator
from typing import Annotated, Any
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

from muffin_agent.multi_agent import (
    AgentParticipant,
    AlternatingModerator,
    LLMParticipant,
    MaxRoundsTerminator,
    RoundRobinModerator,
    StructuredOutputJudge,
    build_conference_graph,
)
from muffin_agent.multi_agent import judges as judges_mod
from muffin_agent.multi_agent import participants as participants_mod

from .conftest import (
    ai,
    build_counter_stub_agent,
    build_echo_stub_agent,
    build_recording_stub_agent,
    fake_model_config_seq,
)

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

    def test_agent_participant_adds_subgraph_node(self):
        agent = build_echo_stub_agent(response_text="agent reply")
        participants = [
            LLMParticipant("alpha", "multi_agent/_transcript.jinja"),
            AgentParticipant("bob", agent),
        ]
        graph = build_conference_graph(
            participants=participants,
            moderator=RoundRobinModerator(["alpha", "bob"]),
            terminator=MaxRoundsTerminator(1, 2),
        )
        nodes = set(graph.get_graph().nodes.keys())
        assert {"dispatch", "alpha", "bob"} <= nodes


# ── LLM-only conference execution ────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
class TestLLMOnlyConference:
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

        messages: list[BaseMessage] = result["messages"]
        assert [m.name for m in messages] == ["alpha", "beta", "gamma"]
        assert [m.content for m in messages] == ["a-says", "b-says", "g-says"]
        # Each message has a unique id (uuid).
        ids = [m.id for m in messages]
        assert len(set(ids)) == 3 and all(isinstance(i, str) and i for i in ids)
        # Cursors track each LLM participant's last-id.
        cursors = result["agent_cursors"]
        assert cursors["alpha"] == ids[0]
        assert cursors["beta"] == ids[1]
        assert cursors["gamma"] == ids[2]

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

        messages = result["messages"]
        speakers = [m.name for m in messages]
        contents = [m.content for m in messages]
        assert speakers == ["alpha", "beta", "gamma", "alpha", "beta", "gamma"]
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

        assert [m.name for m in result["messages"]] == ["bull", "bear", "bull", "bear"]

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

        assert len(result["messages"]) == 1
        assert result["verdict"] == {"decision": "buy", "confidence": 0.8}


# ── AgentParticipant tests ───────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
class TestAgentParticipant:
    async def test_agent_emits_one_named_aimessage_per_turn(self):
        """End-to-end: a single agent participant's turn lands as one tagged AI."""
        agent = build_echo_stub_agent(response_text="agent says hi")
        participants = [AgentParticipant("solo", agent)]
        graph = build_conference_graph(
            participants=participants,
            moderator=RoundRobinModerator(["solo"]),
            terminator=MaxRoundsTerminator(max_rounds=1, num_participants=1),
        )

        result = await graph.ainvoke({})
        messages: list[BaseMessage] = result["messages"]
        # Exactly one message; it's the agent's response, tagged with name.
        assert len(messages) == 1
        assert isinstance(messages[0], AIMessage)
        assert messages[0].content == "agent says hi"
        assert messages[0].name == "solo"
        # Cursor advanced.
        assert result["agent_cursors"]["solo"] == messages[0].id

    async def test_no_intermediate_messages_leak_to_parent(self):
        """Agent's internal tool calls / intermediate AIs are filtered."""
        # Use a stub agent that emits TWO AIMessages (simulating intermediate
        # + final). The framework's extract should only surface the LAST.

        class _ChattyAgentState(TypedDict, total=False):
            messages: Annotated[list[BaseMessage], add_messages]

        async def _chatty_node(state: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG001
            return {
                "messages": [
                    AIMessage(content="thinking out loud..."),
                    AIMessage(content="here is my answer"),
                ]
            }

        from langgraph.graph import END, START, StateGraph

        builder: StateGraph = StateGraph(_ChattyAgentState)
        builder.add_node("agent", _chatty_node)
        builder.add_edge(START, "agent")
        builder.add_edge("agent", END)
        chatty_agent = builder.compile(checkpointer=InMemorySaver())

        participants = [AgentParticipant("chatty", chatty_agent)]
        graph = build_conference_graph(
            participants=participants,
            moderator=RoundRobinModerator(["chatty"]),
            terminator=MaxRoundsTerminator(max_rounds=1, num_participants=1),
        )

        result = await graph.ainvoke({})
        messages = result["messages"]
        # Only the LAST AIMessage (the final answer) survives.
        assert len(messages) == 1
        assert messages[0].content == "here is my answer"
        assert messages[0].name == "chatty"

    async def test_cursor_advances_so_agent_sees_only_new_messages(self):
        """Across two rounds, the agent's second invocation receives only the
        message added since its first turn — not its own prior turn."""
        recording: list[list[BaseMessage]] = []
        agent = build_recording_stub_agent(recording, response_text="bob's turn")

        participants = [
            LLMParticipant("alice", "multi_agent/_transcript.jinja"),
            AgentParticipant("bob", agent),
        ]
        graph = build_conference_graph(
            participants=participants,
            moderator=RoundRobinModerator(["alice", "bob"]),
            terminator=MaxRoundsTerminator(max_rounds=2, num_participants=2),
        )

        cfg, _ = fake_model_config_seq(ai("alice-1"), ai("alice-2"))
        with _patch_participants(cfg):
            result = await graph.ainvoke({})

        # Two bob invocations recorded.
        assert len(recording) == 2

        # First bob invocation: input = [HumanMessage("[alice]: alice-1"),
        #                                HumanMessage("Take your turn now.")]
        first_input = recording[0]
        assert len(first_input) == 2
        assert isinstance(first_input[0], HumanMessage)
        assert first_input[0].content == "[alice]: alice-1"
        assert isinstance(first_input[1], HumanMessage)
        assert first_input[1].content == "Take your turn now."

        # Second bob invocation: input contains ONLY alice-2 (bob's own prior
        # turn is in his checkpointer thread, not re-fed) + take-your-turn.
        second_input = recording[1]
        assert len(second_input) == 2
        assert second_input[0].content == "[alice]: alice-2"
        assert second_input[1].content == "Take your turn now."

        # Parent messages: 4 AIMessages total, alternating alice/bob.
        msgs = result["messages"]
        assert [m.name for m in msgs] == ["alice", "bob", "alice", "bob"]

    async def test_persistent_agent_state_across_invocations(self):
        """Verify LangGraph's per-thread subgraph persistence carries agent state.

        The counter-agent is compiled with ``checkpointer=True`` (the langgraph
        sentinel for per-thread subgraph persistence). The conference is
        compiled with an ``InMemorySaver()`` — required for the subgraph's
        per-thread mechanism to engage. Across three turns the agent's
        ``invocation_count`` increments via its own checkpoint.
        """
        agent = build_counter_stub_agent(prefix="counter says")
        participants = [AgentParticipant("counter", agent)]
        graph = build_conference_graph(
            participants=participants,
            moderator=RoundRobinModerator(["counter"]),
            terminator=MaxRoundsTerminator(max_rounds=3, num_participants=1),
            checkpointer=InMemorySaver(),
        )

        result = await graph.ainvoke(
            {},
            config={"configurable": {"thread_id": "test-persistence"}},
        )

        msgs = result["messages"]
        assert len(msgs) == 3
        contents = [m.content for m in msgs]
        assert contents == [
            "counter says #1",
            "counter says #2",
            "counter says #3",
        ]


# ── Mixed-participant conference ─────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
class TestMixedConference:
    async def test_llm_and_agent_participants_interleave(self):
        agent = build_echo_stub_agent(response_text="agent replies")
        participants = [
            LLMParticipant("alice", "multi_agent/_transcript.jinja"),
            AgentParticipant("bob", agent),
        ]
        graph = build_conference_graph(
            participants=participants,
            moderator=RoundRobinModerator(["alice", "bob"]),
            terminator=MaxRoundsTerminator(max_rounds=2, num_participants=2),
        )

        cfg, _ = fake_model_config_seq(ai("alice-1"), ai("alice-2"))
        with _patch_participants(cfg):
            result = await graph.ainvoke({})

        msgs = result["messages"]
        assert [m.name for m in msgs] == ["alice", "bob", "alice", "bob"]
        assert [m.content for m in msgs] == [
            "alice-1",
            "agent replies",
            "alice-2",
            "agent replies",
        ]
        # Every message has a unique id.
        ids = [m.id for m in msgs]
        assert len(set(ids)) == 4


# ── State-schema customisation ───────────────────────────────────────────────


class _ParentState(TypedDict, total=False):
    """Parent-graph state with renamed messages field + extra context."""

    topic: str
    risk_debate_messages: Annotated[list[BaseMessage], add_messages]
    next_speaker: str | None
    agent_cursors: dict[str, str]
    ticker: str


@pytest.mark.unit
@pytest.mark.asyncio
class TestConferenceWithCustomStateSchema:
    async def test_messages_field_rename(self):
        participants = [
            LLMParticipant("alpha", "multi_agent/_transcript.jinja"),
            LLMParticipant("beta", "multi_agent/_transcript.jinja"),
        ]
        graph = build_conference_graph(
            participants=participants,
            moderator=RoundRobinModerator(["alpha", "beta"]),
            terminator=MaxRoundsTerminator(max_rounds=1, num_participants=2),
            state_schema=_ParentState,
            messages_field="risk_debate_messages",
        )

        cfg, fakes = fake_model_config_seq(ai("alpha-says"), ai("beta-says"))
        with _patch_participants(cfg):
            result = await graph.ainvoke({"ticker": "AAPL", "topic": "evaluate"})

        # Messages landed under the renamed field.
        assert "risk_debate_messages" in result
        assert "messages" not in result  # not in this state schema
        msgs: list[BaseMessage] = result["risk_debate_messages"]
        assert [m.name for m in msgs] == ["alpha", "beta"]

        # Beta's system prompt should reference alpha's rendered turn.
        beta_system = fakes[1].invocations[0][0]
        assert "alpha: alpha-says" in beta_system.content


class _EchoParentState(TypedDict, total=False):
    """Parent state sharing an ``operator.add`` reducer channel the conference
    does NOT own — the shape that doubled the trading_decision debate turns."""

    shared_accumulator: Annotated[list[str], operator.add]
    debate_messages: Annotated[list[BaseMessage], add_messages]
    debate_agent_cursors: dict[str, str]
    debate_next_speaker: str | None


class _EchoOutputSchema(TypedDict, total=False):
    """Restricts the conference's emissions to its own conference-owned fields."""

    debate_messages: Annotated[list[BaseMessage], add_messages]
    debate_agent_cursors: dict[str, str]
    debate_next_speaker: str | None


def _echo_conference(*, output_schema: type | None) -> CompiledStateGraph:
    return build_conference_graph(
        participants=[
            LLMParticipant("alpha", "multi_agent/_transcript.jinja"),
            LLMParticipant("beta", "multi_agent/_transcript.jinja"),
        ],
        moderator=AlternatingModerator("alpha", "beta"),
        terminator=MaxRoundsTerminator(max_rounds=1, num_participants=2),
        state_schema=_EchoParentState,
        output_schema=output_schema,
        messages_field="debate_messages",
        agent_cursors_field="debate_agent_cursors",
        next_speaker_field="debate_next_speaker",
    )


def _parent_with_conference(conference: CompiledStateGraph) -> CompiledStateGraph:
    """A parent graph: seed the shared accumulator, then run the conference."""

    async def seed(state: dict) -> dict:  # noqa: ARG001
        return {"shared_accumulator": ["seeded"]}

    parent: StateGraph = StateGraph(_EchoParentState)
    parent.add_node("seed", seed)  # type: ignore[type-var]
    parent.add_node("conference", conference)
    parent.add_edge(START, "seed")
    parent.add_edge("seed", "conference")
    parent.add_edge("conference", END)
    return parent.compile()


@pytest.mark.unit
@pytest.mark.asyncio
class TestConferenceOutputSchema:
    """``output_schema`` restricts the subgraph's emissions to parent state.

    Without it, a conference compiled against a parent schema echoes the
    parent's own reducer-channel value back through its final state, and the
    parent's reducer re-applies it (doubling). This is what silently doubled
    the trading_decision Bull/Bear turns when the risk-debate conference
    shared ``TradingDecisionState``'s ``operator.add`` channels.
    """

    async def test_output_schema_prevents_reducer_echo(self):
        compiled = _parent_with_conference(
            _echo_conference(output_schema=_EchoOutputSchema)
        )
        cfg, _ = fake_model_config_seq(ai("alpha-says"), ai("beta-says"))
        with _patch_participants(cfg):
            result = await compiled.ainvoke({})

        # The conference produced exactly its two turns...
        assert [m.name for m in result["debate_messages"]] == ["alpha", "beta"]
        # ...and did NOT echo the parent's reducer channel back.
        assert result["shared_accumulator"] == ["seeded"]

    async def test_without_output_schema_echoes_and_doubles(self):
        # Documents the bug the parameter fixes: no output_schema → the
        # subgraph emits the parent's shared channel back → parent doubles it.
        compiled = _parent_with_conference(_echo_conference(output_schema=None))
        cfg, _ = fake_model_config_seq(ai("alpha-says"), ai("beta-says"))
        with _patch_participants(cfg):
            result = await compiled.ainvoke({})

        assert result["shared_accumulator"] == ["seeded", "seeded"]


@pytest.mark.unit
@pytest.mark.asyncio
class TestImmediateTermination:
    async def test_zero_max_rounds_stops_at_dispatch(self):
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

        assert fakes == [] or all(not f.invocations for f in fakes)
        assert not result.get("messages")
