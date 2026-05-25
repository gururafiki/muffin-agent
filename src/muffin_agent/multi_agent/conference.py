"""``build_conference_graph`` — multi-agent conference subgraph builder.

Generates a compiled ``StateGraph`` with one named node per participant
plus a pure-Python ``dispatch`` node that handles speaker selection and
termination. Routing is graph-level (conditional edges), not via
``Command(goto=...)``.

Two participant kinds, two node shapes:

* :class:`LLMParticipant` / :class:`LLMMessageParticipant` — Python adapter
  node (``_make_participant_node``) that calls ``speak()`` and writes one
  ``AIMessage(content, name=speaker, id=...)`` to shared ``messages``.
* :class:`AgentParticipant` — a thin ``prep → agent → extract`` subgraph
  (``_build_agent_subgraph``) added as a parent-graph node. The agent
  itself is a native LangGraph node inside the subgraph. ``prep`` slices
  shared messages by per-agent cursor; ``extract`` tags the agent's
  final AIMessage and updates the cursor.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .judges import Judge
from .moderators import Moderator
from .participants import AgentParticipant, Participant
from .state import ConferenceState
from .terminators import Terminator


def build_conference_graph(
    *,
    participants: Sequence[Participant | AgentParticipant],
    moderator: Moderator,
    terminator: Terminator,
    judge: Judge | None = None,
    state_schema: type = ConferenceState,
    messages_field: str = "messages",
    next_speaker_field: str = "next_speaker",
    agent_cursors_field: str = "agent_cursors",
    verdict_field: str = "verdict",
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Compile a multi-agent conference subgraph.

    Each participant becomes a distinct named graph node — LLM participants
    as a single Python adapter, ``AgentParticipant`` as a 3-node wrapper
    subgraph (``prep → agent → extract``). The ``dispatch`` node calls
    ``terminator.should_stop`` first; if true, it clears
    ``next_speaker_field`` and routes to the judge (if configured) or END.
    Otherwise it calls ``moderator.next_speaker`` and routes to the named
    participant.

    The state schema MUST declare:

    * ``messages_field``: ``Annotated[list[BaseMessage], add_messages]``
    * ``next_speaker_field``: ``str | None``
    * ``agent_cursors_field``: ``dict[str, str]`` (only required when at
      least one ``AgentParticipant`` is in the lineup)
    * ``verdict_field``: ``dict[str, Any] | None`` (only required when a
      judge is configured)

    :class:`ConferenceState` (the default) declares all four under the
    default names. Callers passing a wider schema (e.g. a parent graph's
    state) must either include matching field names or override the
    ``*_field`` parameters.

    Args:
        participants: ordered list. Mix LLMParticipant / LLMMessageParticipant
            / AgentParticipant freely.
        moderator: turn-routing policy.
        terminator: stop-decision policy.
        judge: optional post-conference synthesiser; runs once if set.
        state_schema: TypedDict the subgraph compiles against.
        messages_field: state key holding the shared ``list[BaseMessage]``.
        next_speaker_field: state key for routing (written by ``dispatch``).
        agent_cursors_field: state key for per-agent last-seen-id tracking.
        verdict_field: state key for the judge's verdict (written by ``judge``).
        checkpointer: optional checkpointer for standalone use.
    """
    if not participants:
        raise ValueError("Conference must have at least one participant.")

    builder: StateGraph = StateGraph(state_schema)

    builder.add_node(
        "dispatch",
        _make_dispatch_node(
            moderator,
            terminator,
            messages_field=messages_field,
            next_speaker_field=next_speaker_field,
        ),
    )
    for participant in participants:
        if isinstance(participant, AgentParticipant):
            node: Any = _build_agent_subgraph(
                participant,
                parent_state_schema=state_schema,
                messages_field=messages_field,
                agent_cursors_field=agent_cursors_field,
            )
        else:
            node = _make_participant_node(
                participant,
                messages_field=messages_field,
                agent_cursors_field=agent_cursors_field,
            )
        builder.add_node(participant.name, node)
        builder.add_edge(participant.name, "dispatch")

    end_target: str = END
    if judge is not None:
        builder.add_node(
            "judge",
            _make_judge_node(
                judge,
                messages_field=messages_field,
                verdict_field=verdict_field,
            ),
        )
        builder.add_edge("judge", END)
        end_target = "judge"

    builder.add_edge(START, "dispatch")

    targets: dict[Hashable, str] = {p.name: p.name for p in participants}
    targets["__end__"] = end_target

    def _route(state: dict[str, Any]) -> str:
        nxt = state.get(next_speaker_field)
        return nxt if nxt is not None else "__end__"

    builder.add_conditional_edges("dispatch", _route, targets)

    return builder.compile(checkpointer=checkpointer)


def _normalize_state(
    state: dict[str, Any], messages_field: str
) -> dict[str, Any]:
    """Return a state view with canonical ``messages`` key.

    When ``messages_field == "messages"`` this is a no-op; otherwise
    we shallow-copy and overlay the canonical key. Participants and
    judges always read ``state["messages"]`` regardless of where the
    parent stores the underlying list.
    """
    if messages_field == "messages":
        return state
    return {**state, "messages": state.get(messages_field) or []}


def _make_dispatch_node(
    moderator: Moderator,
    terminator: Terminator,
    *,
    messages_field: str,
    next_speaker_field: str,
):
    async def dispatch(state: dict[str, Any], config) -> dict[str, Any]:  # noqa: ARG001
        normalized = _normalize_state(state, messages_field)
        stop, _reason = terminator.should_stop(normalized)
        if stop:
            return {next_speaker_field: None}
        return {next_speaker_field: moderator.next_speaker(normalized)}

    return dispatch


def _make_participant_node(
    participant: Participant,
    *,
    messages_field: str,
    agent_cursors_field: str,
):
    async def node(state: dict[str, Any], config) -> dict[str, Any]:
        normalized = _normalize_state(state, messages_field)
        content = await participant.speak(normalized, config)
        ai_id = str(uuid4())
        ai_msg = AIMessage(content=content, name=participant.name, id=ai_id)
        cursors = dict(state.get(agent_cursors_field) or {})
        cursors[participant.name] = ai_id
        return {
            messages_field: [ai_msg],
            agent_cursors_field: cursors,
        }

    return node


def _make_judge_node(
    judge: Judge,
    *,
    messages_field: str,
    verdict_field: str,
):
    async def node(state: dict[str, Any], config) -> dict[str, Any]:
        normalized = _normalize_state(state, messages_field)
        verdict = await judge.adjudicate(normalized, config)
        return {verdict_field: verdict}

    return node


def _build_agent_subgraph(
    participant: AgentParticipant,
    *,
    parent_state_schema: type,
    messages_field: str,
    agent_cursors_field: str,
) -> CompiledStateGraph:
    """Compile the ``prep → agent → extract`` subgraph for one AgentParticipant.

    The subgraph state schema is identical to the parent's so all
    conference fields (and any domain fields the parent declares like
    ``ticker`` / ``query``) flow through unchanged. ``prep`` and
    ``extract`` operate on the parent's ``messages_field``; the
    intermediate ``RemoveMessage`` entries they emit are local to the
    subgraph's state and don't propagate to parent (parent's reducer
    applies to the subgraph's FINAL state, not intermediate deltas).

    The compiled muffin agent is added as the subgraph's middle node
    (``add_node(name, agent)``) — native LangGraph composition. When the
    parent invokes this subgraph multiple times across the conference
    run, LangGraph's subgraph namespace mechanism gives the agent's own
    checkpointer a stable thread namespace, so the agent's internal state
    (tool results, ReAct scratch) persists across invocations.

    If LangGraph's automatic namespacing turns out NOT to give
    cross-invocation continuation in practice, the fallback is to call
    ``await participant.agent.ainvoke(...)`` directly from inside the
    ``prep`` node with an explicit derived ``thread_id`` — same public
    API on ``AgentParticipant``, different internal mechanism.
    """
    sub: StateGraph = StateGraph(parent_state_schema)

    async def prep(state: dict[str, Any], config) -> dict[str, Any]:  # noqa: ARG001
        all_msgs: list[BaseMessage] = state.get(messages_field) or []
        cursors: dict[str, str] = state.get(agent_cursors_field) or {}
        last_id = cursors.get(participant.name)
        if last_id is None:
            new_msgs = list(all_msgs)
        else:
            idx_after = len(all_msgs)
            for i, m in enumerate(all_msgs):
                if m.id == last_id:
                    idx_after = i + 1
                    break
            new_msgs = list(all_msgs[idx_after:])

        agent_input: list[BaseMessage] = [
            HumanMessage(
                content=f"[{m.name}]: {m.content}",
                name=m.name,
                id=m.id,
            )
            if isinstance(m, AIMessage) and m.name and m.name != participant.name
            else m
            for m in new_msgs
        ]
        agent_input.append(HumanMessage(participant.user_prompt))

        return {
            messages_field: [
                *(RemoveMessage(id=m.id) for m in all_msgs if m.id),
                *agent_input,
            ],
        }

    async def extract(state: dict[str, Any], config) -> dict[str, Any]:  # noqa: ARG001
        msgs: list[BaseMessage] = state.get(messages_field) or []
        last_ai = next(
            (m for m in reversed(msgs) if isinstance(m, AIMessage)),
            None,
        )
        if last_ai is None:
            return {}

        tagged_id = last_ai.id or str(uuid4())
        tagged = AIMessage(
            content=last_ai.content,
            name=participant.name,
            id=tagged_id,
        )

        cursors = dict(state.get(agent_cursors_field) or {})
        cursors[participant.name] = tagged_id

        return {
            messages_field: [
                *(RemoveMessage(id=m.id) for m in msgs if m.id),
                tagged,
            ],
            agent_cursors_field: cursors,
        }

    # mypy 2 cannot infer the NodeInputT type variable from our
    # `dict[str, Any]` state args (state is parameterized by the parent's
    # schema, which is dynamic). Runtime behaviour is correct — LangGraph
    # passes state matching the parent state schema as a dict regardless.
    sub.add_node("prep", prep)  # type: ignore[type-var]
    sub.add_node("agent", participant.agent)
    sub.add_node("extract", extract)  # type: ignore[type-var]
    sub.add_edge(START, "prep")
    sub.add_edge("prep", "agent")
    sub.add_edge("agent", "extract")
    sub.add_edge("extract", END)

    return sub.compile()
