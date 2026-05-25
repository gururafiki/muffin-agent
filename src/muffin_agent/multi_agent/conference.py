"""``build_conference_graph`` — multi-agent conference subgraph builder.

Generates a compiled ``StateGraph`` with one named node per participant
plus a pure-Python ``dispatch`` node that handles speaker selection and
termination. Routing is graph-level (conditional edges), not via
``Command(goto=...)`` — matches the rest of muffin-agent's
trading_decision style.

Topology::

    START -> dispatch ─┬─> participant_1 ──┐
                       ├─> participant_2 ──┤  (looped back to dispatch
                       ├─> participant_N ──┤   after each turn)
                       └─> judge -> END    (when terminator says stop)
                       └─> END             (no judge configured)
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .judges import Judge
from .moderators import Moderator
from .participants import Participant
from .state import ConferenceState, Turn
from .terminators import Terminator


def build_conference_graph(
    *,
    participants: Sequence[Participant],
    moderator: Moderator,
    terminator: Terminator,
    judge: Judge | None = None,
    state_schema: type = ConferenceState,
    transcript_field: str = "transcript",
    next_speaker_field: str = "next_speaker",
    verdict_field: str = "verdict",
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Compile a multi-agent conference subgraph.

    Each participant becomes a distinct named graph node so traces remain
    self-describing (the speaker name shows up directly). The ``dispatch``
    node calls ``terminator.should_stop`` first; if true, it clears
    ``next_speaker_field`` and routes to the judge (if configured) or END.
    Otherwise it calls ``moderator.next_speaker`` and routes to the named
    participant.

    The state schema MUST declare:

    * ``transcript_field``: ``Annotated[list[Turn], operator.add]``
    * ``next_speaker_field``: ``str | None``
    * ``verdict_field``: ``dict[str, Any] | None`` (only when a judge is
      configured)

    :class:`ConferenceState` (the default) declares all three under the
    default names. Callers passing a wider schema (e.g. a parent graph's
    state) must either include matching field names or override the
    ``*_field`` parameters.

    Args:
        participants: ordered list (preserves trace observability — each
            becomes a named node via ``add_node(name, ...)``).
        moderator: turn-routing policy. Built-ins: :class:`RoundRobinModerator`,
            :class:`AlternatingModerator`. Always returns a participant name.
        terminator: stop-decision policy. Built-in: :class:`MaxRoundsTerminator`.
        judge: optional post-conference synthesiser; runs once if set.
        state_schema: TypedDict the subgraph compiles against.
        transcript_field: state key holding the shared ``list[Turn]``.
        next_speaker_field: state key for routing (written by ``dispatch``).
        verdict_field: state key for the judge's verdict (written by ``judge``).
        checkpointer: optional checkpointer for standalone use; usually
            ``None`` when the conference is added to a parent graph that
            owns its own checkpointing.
    """
    if not participants:
        raise ValueError("Conference must have at least one participant.")
    num_participants = len(participants)

    builder: StateGraph = StateGraph(state_schema)

    builder.add_node(
        "dispatch",
        _make_dispatch_node(
            moderator,
            terminator,
            transcript_field=transcript_field,
            next_speaker_field=next_speaker_field,
        ),
    )
    for participant in participants:
        builder.add_node(
            participant.name,
            _make_participant_node(
                participant,
                transcript_field=transcript_field,
                num_participants=num_participants,
            ),
        )
        builder.add_edge(participant.name, "dispatch")

    end_target: str = END
    if judge is not None:
        builder.add_node(
            "judge",
            _make_judge_node(
                judge,
                transcript_field=transcript_field,
                verdict_field=verdict_field,
            ),
        )
        builder.add_edge("judge", END)
        end_target = "judge"

    builder.add_edge(START, "dispatch")

    # Route from dispatch by `next_speaker_field`. Sentinel "__end__" maps
    # to either the judge node (if configured) or END.
    targets: dict[Hashable, str] = {p.name: p.name for p in participants}
    targets["__end__"] = end_target

    def _route(state: dict[str, Any]) -> str:
        nxt = state.get(next_speaker_field)
        return nxt if nxt is not None else "__end__"

    builder.add_conditional_edges("dispatch", _route, targets)

    return builder.compile(checkpointer=checkpointer)


def _normalize_state(
    state: dict[str, Any], transcript_field: str
) -> dict[str, Any]:
    """Return a state view with canonical ``transcript`` key.

    When ``transcript_field == "transcript"`` this is a no-op; otherwise
    we shallow-copy and overlay the canonical key. Participants and
    judges always read ``state["transcript"]`` regardless of where the
    parent stores the underlying list.
    """
    if transcript_field == "transcript":
        return state
    return {**state, "transcript": state.get(transcript_field) or []}


def _make_dispatch_node(
    moderator: Moderator,
    terminator: Terminator,
    *,
    transcript_field: str,
    next_speaker_field: str,
):
    async def dispatch(state: dict[str, Any], config) -> dict[str, Any]:  # noqa: ARG001
        normalized = _normalize_state(state, transcript_field)
        stop, _reason = terminator.should_stop(normalized)
        if stop:
            return {next_speaker_field: None}
        return {next_speaker_field: moderator.next_speaker(normalized)}

    return dispatch


def _make_participant_node(
    participant: Participant,
    *,
    transcript_field: str,
    num_participants: int,
):
    async def node(state: dict[str, Any], config) -> dict[str, Any]:
        normalized = _normalize_state(state, transcript_field)
        prior_turns: list[Turn] = normalized.get("transcript") or []
        round_num = (len(prior_turns) // num_participants) + 1
        content = await participant.speak(normalized, config)
        turn: Turn = {
            "speaker": participant.name,
            "content": content,
            "round": round_num,
        }
        return {transcript_field: [turn]}

    return node


def _make_judge_node(
    judge: Judge,
    *,
    transcript_field: str,
    verdict_field: str,
):
    async def node(state: dict[str, Any], config) -> dict[str, Any]:
        normalized = _normalize_state(state, transcript_field)
        verdict = await judge.adjudicate(normalized, config)
        return {verdict_field: verdict}

    return node
