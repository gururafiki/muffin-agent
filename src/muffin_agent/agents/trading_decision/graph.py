"""Top-level graph builders for the trading_decision module.

Three composable graphs share the same ``TradingDecisionState`` schema so
callers can opt into the depth they need:

* :func:`build_investment_debate_graph` — analysts → Bull/Bear debate →
  Investment Judge. Smallest useful slice when you just want a
  structured directional view.
* :func:`build_investment_thesis_graph` — adds the Trader (operational
  translation: entry/stop/take-profit/sizing/time_horizon).
* :func:`build_trading_decision_graph` — full pipeline including the
  reflector_resolve / decision_writeback reflection bookends and the
  3-way risk debate + Portfolio Manager.

All builders are **async** because each starts by building four
compiled analyst ReAct agents (one per perspective: market /
fundamentals / news / social) that are added directly as parent-graph
nodes via ``add_node(name, agent, input_schema=AnalystInput)``.
The agent build is amortised to graph-construction time; per-call
invocation is just LLM + tool work.

Routing is **graph-level** — non-analyst nodes return state-update
dicts only, never ``Command(goto=...)``. The bull/bear debate uses a
hand-written conditional-edge router; the 3-way risk debate is wired
through ``muffin_agent.multi_agent.build_conference_graph`` which
generates a self-contained subgraph (round-robin moderator + max-rounds
terminator + no judge — the Portfolio Manager runs in the parent graph
and reads the conference transcript directly).

Retry layering: every LLM-call node carries a ``RetryPolicy`` so
LangGraph retries the whole node on exception (one layer above
LangChain's ``with_retry`` inside each node).
"""

from __future__ import annotations

from functools import partial

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import RetryPolicy

from ...multi_agent import (
    AlternatingModerator,
    LLMParticipant,
    MaxRoundsTerminator,
    RoundRobinModerator,
    build_conference_graph,
)
from ...utils.observability import instrument_graph
from .analysts import (
    build_fundamentals_analyst_agent,
    build_market_analyst_agent,
    build_news_analyst_agent,
    build_social_analyst_agent,
)
from .config import TradingDecisionConfiguration
from .portfolio_manager import portfolio_manager_node
from .reflection import (
    decision_writeback_node,
    reflector_resolve_node,
)
from .researchers import investment_judge_node
from .state import (
    AnalystInput,
    InvestmentDebateOutput,
    RiskDebateOutput,
    TradingDecisionState,
)
from .tools import OutcomesFetcher
from .trader import trader_node

# Standard retry for every LLM-call node. Second layer on top of
# LangChain's ``with_retry`` inside each node body.
_LLM_RETRY = RetryPolicy(max_attempts=2)

_ANALYST_NAMES: tuple[str, str, str, str] = (
    "market_analyst",
    "fundamentals_analyst",
    "news_analyst",
    "social_analyst",
)

_INVESTMENT_DEBATE_PARTICIPANT_NAMES: tuple[str, str] = (
    "bull_researcher",
    "bear_researcher",
)

_RISK_DEBATE_PARTICIPANT_NAMES: tuple[str, str, str] = (
    "aggressive_debator",
    "conservative_debator",
    "neutral_debator",
)


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _add_analyst_nodes(graph: StateGraph, config: RunnableConfig) -> None:
    """Build the 4 analyst ReAct agents and add them as parent-graph nodes.

    Each analyst is a compiled ReAct agent with a custom ``AgentState``
    extension that declares shared keys with ``TradingDecisionState``
    (``ticker``, ``decision_date``, ``<role>_report``) via
    ``OmitFromSchema`` annotations. We pass the explicit ``AnalystInput``
    field schema (``ticker`` + ``decision_date``) — NOT
    ``agent.input_schema`` (a property-less ``RootModel`` that would map
    ``{}`` and raise at coercion) — so each analyst receives exactly its
    declared inputs and the four run isolated in parallel.
    """
    market_agent = await build_market_analyst_agent(config)
    fundamentals_agent = await build_fundamentals_analyst_agent(config)
    news_agent = await build_news_analyst_agent(config)
    social_agent = await build_social_analyst_agent(config)

    graph.add_node(
        "market_analyst",
        market_agent,
        input_schema=AnalystInput,
        retry_policy=_LLM_RETRY,
    )
    graph.add_node(
        "fundamentals_analyst",
        fundamentals_agent,
        input_schema=AnalystInput,
        retry_policy=_LLM_RETRY,
    )
    graph.add_node(
        "news_analyst",
        news_agent,
        input_schema=AnalystInput,
        retry_policy=_LLM_RETRY,
    )
    graph.add_node(
        "social_analyst",
        social_agent,
        input_schema=AnalystInput,
        retry_policy=_LLM_RETRY,
    )


def _wire_analysts_to(graph: StateGraph, *, from_node: str, to_node: str) -> None:
    """Fan out from ``from_node`` to the 4 analysts; barrier into ``to_node``.

    LangGraph fires ``to_node`` only after every incoming edge resolves —
    so the 4 analysts run in parallel and the next stage runs once all
    4 have produced their report.
    """
    for analyst in _ANALYST_NAMES:
        graph.add_edge(from_node, analyst)
        graph.add_edge(analyst, to_node)


def _build_investment_debate_subgraph(
    max_rounds: int,
) -> CompiledStateGraph:
    """Compile the Bull/Bear investment-debate conference subgraph.

    Two-speaker alternation (Bull opens each round; the ``AlternatingModerator``
    reproduces the legacy ``_route_investment_debate`` turn order exactly) with
    a ``max_rounds × 2`` turn cap. No in-conference judge — the Investment
    Judge runs in the parent graph and consumes ``investment_debate_messages``.

    ``output_schema=InvestmentDebateOutput`` restricts what the compiled
    subgraph emits to the parent, so it never echoes the parent's other
    reducer channels (``tool_runs``, and — because this subgraph runs BEFORE
    the risk debate — the risk-debate channels too) back through its final
    state. See ``build_conference_graph``'s ``output_schema`` docstring.
    """
    participants = [
        LLMParticipant(
            name="bull_researcher",
            system_prompt_template="trading_decision/investment_debate/bull.jinja",
            user_prompt="Make your argument now.",
        ),
        LLMParticipant(
            name="bear_researcher",
            system_prompt_template="trading_decision/investment_debate/bear.jinja",
            user_prompt="Make your argument now.",
        ),
    ]
    return build_conference_graph(
        participants=participants,
        moderator=AlternatingModerator(
            speaker_a=_INVESTMENT_DEBATE_PARTICIPANT_NAMES[0],
            speaker_b=_INVESTMENT_DEBATE_PARTICIPANT_NAMES[1],
        ),
        terminator=MaxRoundsTerminator(
            max_rounds=max_rounds,
            num_participants=len(participants),
        ),
        state_schema=TradingDecisionState,
        output_schema=InvestmentDebateOutput,
        messages_field="investment_debate_messages",
        agent_cursors_field="investment_debate_agent_cursors",
        next_speaker_field="investment_next_speaker",
    )


def _build_risk_debate_subgraph(
    max_rounds: int,
) -> CompiledStateGraph:
    """Compile the 3-way risk-debate conference subgraph.

    Round-robin moderator (Aggressive → Conservative → Neutral) with a
    hard ``max_rounds × 3`` turn cap. No in-conference judge — the
    Portfolio Manager runs in the parent graph and consumes the
    ``risk_debate_transcript`` directly.

    ``output_schema=RiskDebateOutput`` restricts the subgraph's emissions to
    its own conference channels so it never echoes the parent's ``operator.add``
    reducer channels (e.g. ``tool_runs``) back through write-back.
    """
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
        output_schema=RiskDebateOutput,
        messages_field="risk_debate_messages",
        agent_cursors_field="risk_debate_agent_cursors",
    )


# ── Builders ──────────────────────────────────────────────────────────────────


async def build_investment_debate_graph(
    config: RunnableConfig,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> CompiledStateGraph:
    """Compile analysts → Bull/Bear debate → Investment Judge sub-graph."""
    cfg = TradingDecisionConfiguration.from_runnable_config(config)
    graph: StateGraph = StateGraph(TradingDecisionState)
    await _add_analyst_nodes(graph, config)
    graph.add_node(
        "investment_debate",
        _build_investment_debate_subgraph(max_rounds=cfg.max_investment_debate_rounds),
    )
    graph.add_node("investment_judge", investment_judge_node, retry_policy=_LLM_RETRY)

    # START → analysts (parallel) → investment_debate (barrier)
    for analyst in _ANALYST_NAMES:
        graph.add_edge(START, analyst)
        graph.add_edge(analyst, "investment_debate")

    graph.add_edge("investment_debate", "investment_judge")
    graph.add_edge("investment_judge", END)

    return graph.compile(checkpointer=checkpointer, store=store)


async def build_investment_thesis_graph(
    config: RunnableConfig,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> CompiledStateGraph:
    """Compile analysts → debate → Judge → Trader sub-graph."""
    cfg = TradingDecisionConfiguration.from_runnable_config(config)
    graph: StateGraph = StateGraph(TradingDecisionState)
    await _add_analyst_nodes(graph, config)
    graph.add_node(
        "investment_debate",
        _build_investment_debate_subgraph(max_rounds=cfg.max_investment_debate_rounds),
    )
    graph.add_node("investment_judge", investment_judge_node, retry_policy=_LLM_RETRY)
    graph.add_node("trader", trader_node, retry_policy=_LLM_RETRY)

    for analyst in _ANALYST_NAMES:
        graph.add_edge(START, analyst)
        graph.add_edge(analyst, "investment_debate")

    graph.add_edge("investment_debate", "investment_judge")
    graph.add_edge("investment_judge", "trader")
    graph.add_edge("trader", END)

    return graph.compile(checkpointer=checkpointer, store=store)


async def build_trading_decision_graph(
    config: RunnableConfig,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
    *,
    outcomes_fetcher: OutcomesFetcher | None = None,
) -> CompiledStateGraph:
    """Compile the full pipeline.

    Topology::

        START → reflector_resolve
                  │
                  ├─→ market_analyst ─┐
                  ├─→ fundamentals_analyst ─┤
                  ├─→ news_analyst ──┤   (parallel; implicit barrier)
                  └─→ social_analyst ─┘
                                     ▼
                    investment_debate (conference subgraph)
                       └── bull_researcher
                       └── bear_researcher
                       (alternating × N rounds via build_conference_graph)
                                     │
                                     ▼
                              investment_judge
                                     │
                                     ▼
                                  trader
                                     │
                                     ▼
                    risk_debate (conference subgraph)
                       └── aggressive_debator
                       └── conservative_debator
                       └── neutral_debator
                       (round-robin × M rounds via build_conference_graph)
                                     │
                                     ▼
                             portfolio_manager
                                     │
                                     ▼
                         decision_writeback → END

    Args:
        config: Active ``RunnableConfig`` — required to build the four
            analyst ReAct agents at graph-construction time. Also read for
            ``max_risk_debate_rounds`` to size the risk-debate subgraph.
        checkpointer: Optional LangGraph checkpointer for graph state.
        store: Optional ``BaseStore`` for reflection memory + tool-result
            caching. When ``None``, reflection bookends degrade to no-ops.
        outcomes_fetcher: Optional ``OutcomesFetcher`` for realised-return
            data (defaults to :func:`fetch_decision_outcome`).
    """
    cfg = TradingDecisionConfiguration.from_runnable_config(config)
    graph: StateGraph = StateGraph(TradingDecisionState)

    graph.add_node(
        "reflector_resolve",
        partial(reflector_resolve_node, store=store, outcomes_fetcher=outcomes_fetcher),
        retry_policy=_LLM_RETRY,
    )
    await _add_analyst_nodes(graph, config)
    graph.add_node(
        "investment_debate",
        _build_investment_debate_subgraph(max_rounds=cfg.max_investment_debate_rounds),
    )
    graph.add_node("investment_judge", investment_judge_node, retry_policy=_LLM_RETRY)
    graph.add_node("trader", trader_node, retry_policy=_LLM_RETRY)
    graph.add_node(
        "risk_debate",
        _build_risk_debate_subgraph(max_rounds=cfg.max_risk_debate_rounds),
    )
    graph.add_node("portfolio_manager", portfolio_manager_node, retry_policy=_LLM_RETRY)
    # Pure-IO node — no LLM, no retry beyond the store's own behaviour.
    graph.add_node("decision_writeback", partial(decision_writeback_node, store=store))

    graph.add_edge(START, "reflector_resolve")
    _wire_analysts_to(graph, from_node="reflector_resolve", to_node="investment_debate")
    graph.add_edge("investment_debate", "investment_judge")
    graph.add_edge("investment_judge", "trader")
    graph.add_edge("trader", "risk_debate")
    graph.add_edge("risk_debate", "portfolio_manager")
    graph.add_edge("portfolio_manager", "decision_writeback")
    graph.add_edge("decision_writeback", END)

    return graph.compile(checkpointer=checkpointer, store=store)


async def make_graph(config: RunnableConfig | None = None) -> CompiledStateGraph:
    """LangGraph Platform graph factory (config-only); registered in ``langgraph.json``.

    Returns the full trading-decision pipeline (:func:`build_trading_decision_graph`).
    The Platform's factory protocol only accepts parameters typed ``RunnableConfig`` /
    ``ServerRuntime`` (a ``BaseStore`` parameter is rejected) and injects its managed
    checkpointer/store into the returned graph. So the deployed entrypoint is this thin
    config-only factory; :func:`build_trading_decision_graph` keeps its ``store`` /
    ``checkpointer`` parameters for CLI / programmatic callers. Mirrors
    :func:`muffin_agent.agents.personas_council.council_graph.make_graph`.
    """
    return instrument_graph(await build_trading_decision_graph(config or {}))
