# Multi-Agent Conference Framework

Generic, reusable subgraph builder for **putting N agents with different system prompts in a conversation**. Each participant takes turns producing one message; a moderator decides who speaks next; a terminator decides when to stop; an optional judge synthesises a final verdict. Lives at `src/muffin_agent/multi_agent/`.

This guide covers what the framework provides, the four pluggable Protocols, the three Participant kinds (including `AgentParticipant` which wraps a compiled muffin agent with per-thread persistence), the shared-messages state shape, and the current production consumer (`trading_decision`'s risk debate).

For the per-file architecture reference, see the [multi_agent section in CLAUDE.md](../CLAUDE.md).

---

## What you get

| Abstraction | Role | Built-in implementations |
|---|---|---|
| `Participant` | One speaker; produces one message per turn | `LLMParticipant`, `LLMMessageParticipant`, `AgentParticipant` |
| `Moderator` | Picks the next speaker each turn | `RoundRobinModerator(speaker_order)`, `AlternatingModerator(speaker_a, speaker_b)` |
| `Terminator` | Decides when to end the conference | `MaxRoundsTerminator(max_rounds, num_participants)` |
| `Judge` (optional) | Post-conference synthesiser; runs once after termination | `StructuredOutputJudge(name, system_prompt_template, output_schema)` returns `result.model_dump()` |

All four are `@runtime_checkable` Protocols — bring your own implementation when the built-ins don't fit (e.g. an LLM-driven moderator, a consensus-detecting terminator).

The single public entry point is `build_conference_graph(...)`:

```python
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
) -> CompiledStateGraph: ...
```

The `*_field` parameters let the conference embed in a parent state schema that uses renamed fields (see `trading_decision` using `risk_debate_messages` / `risk_debate_agent_cursors`).

---

## State shape — messages-only

```python
class ConferenceState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    next_speaker: str | None
    agent_cursors: dict[str, str]
    verdict: dict[str, Any] | None
```

One shared `messages` reducer is the single source of truth. Each turn appends exactly one `AIMessage(content, name=<speaker>, id=<uuid>)`. The `name` attribute is set explicitly by the framework wrapper so chronological rendering and per-agent cursor tracking work portably across LLM providers (Anthropic ignores `name` over the wire; content-prefix tagging is the portable channel inside the agent — see `LLMMessageParticipant` / `AgentParticipant` below).

Per-agent cursor (`agent_cursors[speaker_name] → last seen message id`) is the mechanism that lets each `AgentParticipant` invocation receive only the messages added since it last spoke — rather than the full shared `messages` list every turn. LLMParticipants are stateless across turns so they always see the full rendered transcript.

**Why messages-only** (vs. per-speaker lists like the legacy risk-debate `risk_aggressive_responses` + `risk_conservative_responses` + `risk_neutral_responses`): one list interleaves naturally with `add_messages`, per-speaker rendering is derivable at read time, and downstream synthesis nodes get a clean `list[BaseMessage]` instead of having to interleave N lists manually.

---

## The three Participant kinds

### 1. `LLMParticipant` (Option α — prompt-text rendering)

The prior conversation is rendered chronologically into the system prompt as text. The LLM call sees `[SystemMessage(role + transcript), HumanMessage(user_prompt)]` — two messages, stateless.

Best for short conferences and for prompts that already reference `{{ transcript }}` in their Jinja templates. Zero-prompt-churn migration path for code that previously used per-speaker reducer lists.

```python
from muffin_agent.multi_agent import LLMParticipant

bull = LLMParticipant(
    name="bull",
    system_prompt_template="debate/bull.jinja",
    llm_role="reasoner",                       # one of orchestrator / collector / reasoner
    user_prompt="Take your turn now.",         # appended after the rendered prompt
)
```

Template vars made available:
- `transcript` — chronological text rendering of `state["messages"]`
- `last_opposing_message` — most recent `BaseMessage` by a non-self speaker, or `None` on the opening turn
- every other key in the conference state (so templates can read domain fields like `ticker`, `query`, `investment_judge`, etc.)

### 2. `LLMMessageParticipant` (Option β — message-thread rendering)

The prior conversation is forwarded as a `BaseMessage` thread. The LLM call sees `[SystemMessage(role only), HumanMessage("[other]: …"), AIMessage(self prior), …, HumanMessage(user_prompt)]`. The system prompt stays stable across turns (just the role description), enabling prompt-cache reuse on providers that support it.

Best for long conferences (5+ rounds) where prompt-cache hits matter. Other speakers' AIMessages are converted to `HumanMessage(f"[{name}]: {content}")` so the agent's chat-completions endpoint sees the expected alternating user/assistant pattern.

```python
from muffin_agent.multi_agent import LLMMessageParticipant

bear = LLMMessageParticipant(
    name="bear",
    system_prompt_template="debate/bear-role-only.jinja",   # no {{ transcript }} reference
)
```

### 3. `AgentParticipant` (wraps a compiled muffin agent)

Wraps a compiled muffin agent (ReAct or deep) so it can participate in a conference with **full access to tools, sub-agents, skills, memory, and middleware**. The agent's internal ReAct loop runs to completion per turn; only the agent's final `AIMessage` propagates to parent conference state (tool calls + intermediate messages stay private).

**Two prerequisites** for the per-thread persistence machinery to engage:

1. **The agent MUST be built with `MuffinAgentBuilder(...).with_checkpointer(True)`** — `True` is the LangGraph sentinel (the `Checkpointer = None | bool | BaseCheckpointSaver` type) that enables per-thread persistence for the subgraph when used as a parent-graph node. Without it, the agent's state resets every turn.
2. **The parent graph or conference MUST be compiled with a real checkpointer instance** (e.g. `InMemorySaver()`, `SqliteSaver`, `PostgresSaver`). LangGraph derives the agent's per-thread namespace from the parent's `thread_id` + the subgraph's node path.

See the [official LangGraph subgraph docs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs) for the per-thread persistence pattern.

```python
from langgraph.checkpoint.memory import InMemorySaver
from muffin_agent.multi_agent import AgentParticipant
from muffin_agent.utils.agent_builder import MuffinAgentBuilder

# 1. Build the agent with checkpointer=True (the sentinel)
research_agent = (
    MuffinAgentBuilder(model, name="bull_with_research")
    .with_system_prompt_template("debate/bull-with-research.jinja")
    .with_tool(web_search)                         # gives the agent tools per-turn
    .with_tool(equity_fundamentals_lookup)
    .with_checkpointer(True)                       # ← the sentinel
    .build_react_agent()
)

# 2. Wire it as a conference participant
bull = AgentParticipant(name="bull_with_research", agent=research_agent)
```

The framework wraps `AgentParticipant` in a small `prep → agent → extract` subgraph at build time:

- `prep` — slices `state["messages"]` by `state["agent_cursors"][name]` (only new since this agent's last turn) and converts other-speaker `AIMessage`s to `HumanMessage(f"[{speaker}]: {content}", name=speaker)`. Appends `HumanMessage(user_prompt)` to nudge the agent.
- `agent` — the user's compiled agent, added as a graph node. Runs its full ReAct loop. Its own per-thread checkpoint loads prior state and persists post-run.
- `extract` — finds the agent's last `AIMessage`, tags it with `name=participant.name`, clears the subgraph's messages buffer (via `RemoveMessage`), updates `agent_cursors[name]` with the new message id. Only the tagged AIMessage propagates to parent conference state.

**Caveat** (from the LangGraph docs): per-thread subgraphs do NOT support parallel tool calls — if an LLM has access to an AgentParticipant as a tool and calls it in parallel, the per-thread namespace collides. Doesn't apply to the conference framework because moderators serialise turns by design (RoundRobin / Alternating both pick exactly one speaker per turn).

---

## Topology

```
                  ┌──────────────────────────────────────────┐
                  │                                          │
START → dispatch ─┼─→ alice (LLMParticipant adapter)   ──────┤
                  │                                          │
                  ├─→ bob (LLMMessageParticipant adapter) ───┤
                  │                                          │
                  └─→ carol (AgentParticipant subgraph)  ────┘
                            └ prep → agent → extract
                  ┌─────────────────────────────────────────┐
                  └─→ judge (if configured) → END           │
                  │                                         │
                  └─→ END (no judge)                        ↓
```

`dispatch` is a pure-Python node (no LLM). It:
1. Calls `terminator.should_stop(state)`. If `True`, sets `next_speaker = None`.
2. Otherwise calls `moderator.next_speaker(state)` and writes that name.

A conditional edge from `dispatch` routes by `state["next_speaker"]`:
- A participant name → that participant's node runs, appends one AIMessage to `messages`, loops back to `dispatch`.
- `None` (terminator fired) → routes to the judge node (if configured) or directly to END.

Each participant node loops back to `dispatch` after running.

---

## Per-agent persistence — how it actually works

When you build the agent with `checkpointer=True` and the parent graph has a checkpointer instance, LangGraph's per-thread mechanism engages:

- Parent thread_id: `"conf:run42"`
- Agent's per-thread namespace (derived automatically): `("conf:run42", "bull_with_research", "agent")`
- Turn 1: no prior checkpoint → agent starts fresh → runs → saves final state
- Turn 2: agent loads prior checkpoint → new input merged via `add_messages` (id-deduped) → runs from START with combined state → saves
- Turn 3: same — counter or any custom state field keeps incrementing

The framework's `prep` step passes ONLY new messages (since this agent's last cursor); the agent's `add_messages` reducer merges with its prior thread state via id-dedup so prior turns aren't duplicated.

### Verified-working worked example

`tests/multi_agent/test_conference.py::test_persistent_agent_state_across_invocations` runs a stub agent with a custom `invocation_count: int` state field. Across three turns the counter shows `#1 → #2 → #3`, proving per-thread persistence works:

```python
class _StubAgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    invocation_count: int

async def _node(state):
    count = (state.get("invocation_count") or 0) + 1
    return {
        "messages": [AIMessage(content=f"counter says #{count}")],
        "invocation_count": count,
    }

builder = StateGraph(_StubAgentState)
builder.add_node("agent", _node)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)
agent = builder.compile(checkpointer=True)            # ← the sentinel

graph = build_conference_graph(
    participants=[AgentParticipant("counter", agent)],
    moderator=RoundRobinModerator(["counter"]),
    terminator=MaxRoundsTerminator(max_rounds=3, num_participants=1),
    checkpointer=InMemorySaver(),                     # ← parent checkpointer
)

result = await graph.ainvoke(
    {},
    config={"configurable": {"thread_id": "test-persistence"}},
)
# result["messages"][i].content for i in 0..2:
#   "counter says #1", "counter says #2", "counter says #3"
```

---

## Worked example: mixed-participant 3-way debate

```python
from langgraph.checkpoint.memory import InMemorySaver
from muffin_agent.multi_agent import (
    build_conference_graph,
    LLMParticipant,
    AgentParticipant,
    RoundRobinModerator,
    MaxRoundsTerminator,
    StructuredOutputJudge,
)
from pydantic import BaseModel

# Tool-using participant — needs the checkpointer sentinel
bull_agent = (
    MuffinAgentBuilder(model, name="bull")
    .with_system_prompt_template("debate/bull.jinja")
    .with_tool(web_search)
    .with_checkpointer(True)
    .build_react_agent()
)

# Two plain LLM participants (cheaper, no tools)
participants = [
    LLMParticipant("conservative", "debate/conservative.jinja"),
    AgentParticipant("bull", bull_agent),
    LLMParticipant("moderator_summary", "debate/moderator.jinja"),
]


class DebateVerdict(BaseModel):
    winner: Literal["bull", "conservative", "tie"]
    confidence: float
    rationale: str


graph = build_conference_graph(
    participants=participants,
    moderator=RoundRobinModerator([p.name for p in participants]),
    terminator=MaxRoundsTerminator(max_rounds=3, num_participants=3),
    judge=StructuredOutputJudge(
        name="judge",
        system_prompt_template="debate/judge.jinja",
        output_schema=DebateVerdict,
    ),
    checkpointer=InMemorySaver(),    # required when any AgentParticipant is in the lineup
)

result = await graph.ainvoke(
    {"ticker": "AAPL", "topic": "Q4 outlook"},
    config={"configurable": {"thread_id": "debate-aapl-q4"}},
)
# result["messages"] = 9 AIMessages (3 participants × 3 rounds), name-tagged
# result["verdict"] = {"winner": "...", "confidence": 0.7, "rationale": "..."}
```

---

## Current production consumer — `trading_decision` risk debate

The 3-way Aggressive / Conservative / Neutral risk debate inside [`trading_decision/`](trading-decision.md) is wired through the conference framework. The relevant code lives in `agents/trading_decision/graph.py:_build_risk_debate_subgraph(max_rounds)`:

```python
def _build_risk_debate_subgraph(max_rounds: int) -> CompiledStateGraph:
    participants = [
        LLMParticipant(
            name="aggressive_debator",
            system_prompt_template="trading_decision/risk_debate/aggressive.jinja",
            user_prompt="Make your argument now.",
        ),
        LLMParticipant(
            name="conservative_debator",
            system_prompt_template="trading_decision/risk_debate/conservative.jinja",
            user_prompt="Make your argument now.",
        ),
        LLMParticipant(
            name="neutral_debator",
            system_prompt_template="trading_decision/risk_debate/neutral.jinja",
            user_prompt="Make your argument now.",
        ),
    ]
    return build_conference_graph(
        participants=participants,
        moderator=RoundRobinModerator(
            speaker_order=list(_RISK_DEBATE_PARTICIPANT_NAMES)
        ),
        terminator=MaxRoundsTerminator(
            max_rounds=max_rounds,
            num_participants=len(participants),
        ),
        state_schema=TradingDecisionState,
        messages_field="risk_debate_messages",
        agent_cursors_field="risk_debate_agent_cursors",
    )
```

From the parent graph's perspective the result is a single `risk_debate` node: `trader → risk_debate → portfolio_manager` is a straight edge chain. The Portfolio Manager reads `state["risk_debate_messages"]` and passes it through `format_risk_history(messages)` (a thin wrapper over `multi_agent.render_messages_chronological`) into the PM prompt as `{{ transcript }}`.

The Bull/Bear investment debate is still bespoke today (uses per-speaker `Annotated[list[str], operator.add]` reducers + a hand-written `_route_investment_debate` router). Migration to the conference framework via `AlternatingModerator("bull_researcher", "bear_researcher")` is a roadmap item.

---

## Where to look in the code

| Concern | File |
|---|---|
| Public entry point + per-participant subgraph wrap | [src/muffin_agent/multi_agent/conference.py](../src/muffin_agent/multi_agent/conference.py) |
| `Participant` Protocol + 3 concrete implementations | [src/muffin_agent/multi_agent/participants.py](../src/muffin_agent/multi_agent/participants.py) |
| Moderators (round-robin, alternating) | [src/muffin_agent/multi_agent/moderators.py](../src/muffin_agent/multi_agent/moderators.py) |
| Terminators (max-rounds) | [src/muffin_agent/multi_agent/terminators.py](../src/muffin_agent/multi_agent/terminators.py) |
| Judges (structured-output) | [src/muffin_agent/multi_agent/judges.py](../src/muffin_agent/multi_agent/judges.py) |
| State types (`ConferenceState`) | [src/muffin_agent/multi_agent/state.py](../src/muffin_agent/multi_agent/state.py) |
| Message formatters (`render_messages_chronological`, `last_opposing_message`) | [src/muffin_agent/multi_agent/_formatters.py](../src/muffin_agent/multi_agent/_formatters.py) |
| Public re-exports | [src/muffin_agent/multi_agent/__init__.py](../src/muffin_agent/multi_agent/__init__.py) |
| Optional shared transcript Jinja partial | [src/muffin_agent/prompts/multi_agent/_transcript.jinja](../src/muffin_agent/prompts/multi_agent/_transcript.jinja) |
| Production wiring example (risk_debate) | [src/muffin_agent/agents/trading_decision/graph.py](../src/muffin_agent/agents/trading_decision/graph.py) (`_build_risk_debate_subgraph`) |
| Tests (44 cases — Participant kinds, moderators, terminators, judges, mixed conferences, persistence smoke test) | [tests/multi_agent/](../tests/multi_agent/) |

---

## Deferred / not built today

- **LLM-driven `Moderator`** — picks next speaker by reasoning over the transcript. The Protocol is the same; implement `next_speaker(state) -> str` as an LLM call. Build when a real use case lands (e.g. a Manager agent deciding who speaks next based on what the discussion needs).
- **Consensus-detecting `Terminator`** — early stop when participants converge. Implement `should_stop(state)` as an LLM call that reads recent messages. Same Protocol; bring your own.
- **Stream-back participant** — observe a streaming agent's intermediate state (e.g. for a UI panel). The current `AgentParticipant` only surfaces the final AIMessage; a streaming variant would need a per-turn callback hook.
