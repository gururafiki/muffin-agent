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
nodes via ``add_node(name, agent, input_schema=agent.input_schema)``.
The agent build is amortised to graph-construction time; per-call
invocation is just LLM + tool work.

Routing is **graph-level** — non-analyst nodes return state-update
dicts only, never ``Command(goto=...)``. Conditional edges (list form)
decide next nodes based on the accumulated debate-response lists.

Retry layering: every LLM-call node carries a ``RetryPolicy`` so
LangGraph retries the whole node on exception (one layer above
LangChain's ``with_retry`` inside each node).
"""

from __future__ import annotations

from functools import partial

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_config
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import RetryPolicy

from .analysts import (
    build_fundamentals_analyst_agent,
    build_market_analyst_agent,
    build_news_analyst_agent,
    build_social_analyst_agent,
)
from .config import TradingDecisionConfiguration
from .portfolio_manager import portfolio_manager_node
from .reflection import (
    OutcomesFetcher,
    decision_writeback_node,
    reflector_resolve_node,
)
from .researchers import (
    bear_researcher_node,
    bull_researcher_node,
    investment_judge_node,
)
from .risk_debate import (
    aggressive_debator_node,
    conservative_debator_node,
    neutral_debator_node,
)
from .state import TradingDecisionState
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


# ── Routing ───────────────────────────────────────────────────────────────────


def _active_config() -> RunnableConfig:
    """Pull the active LangGraph ``RunnableConfig`` (used by routers)."""
    try:
        return get_config()
    except Exception:
        return RunnableConfig(configurable={})


def _route_investment_debate(state: TradingDecisionState) -> str:
    """Pick the next investment-debate node.

    Routes ``bull_researcher`` and ``bear_researcher`` while count is below
    the round budget; routes ``investment_judge`` once exhausted.
    """
    cfg = TradingDecisionConfiguration.from_runnable_config(_active_config())
    bulls = len(state.get("investment_bull_responses") or [])
    bears = len(state.get("investment_bear_responses") or [])
    if bulls + bears >= 2 * cfg.max_investment_debate_rounds:
        return "investment_judge"
    return "bear_researcher" if bulls > bears else "bull_researcher"


_INVESTMENT_DEBATE_TARGETS: list[str] = [
    "bull_researcher",
    "bear_researcher",
    "investment_judge",
]


def _route_risk_debate(state: TradingDecisionState) -> str:
    """Pick the next risk-debate node (round-robin) or hand off to PM.

    Round-robin order: Aggressive → Conservative → Neutral. The next speaker
    is whichever role has the fewest responses so far; ties are broken in
    the canonical order. Hands off to the Portfolio Manager once each role
    has spoken ``max_risk_debate_rounds`` times.
    """
    cfg = TradingDecisionConfiguration.from_runnable_config(_active_config())
    aggressives = len(state.get("risk_aggressive_responses") or [])
    conservatives = len(state.get("risk_conservative_responses") or [])
    neutrals = len(state.get("risk_neutral_responses") or [])
    total = aggressives + conservatives + neutrals
    if total >= 3 * cfg.max_risk_debate_rounds:
        return "portfolio_manager"
    # Whichever role is behind by count goes next; canonical order on ties.
    if aggressives <= conservatives and aggressives <= neutrals:
        return "aggressive_debator"
    if conservatives <= neutrals:
        return "conservative_debator"
    return "neutral_debator"


_RISK_DEBATE_TARGETS: list[str] = [
    "aggressive_debator",
    "conservative_debator",
    "neutral_debator",
    "portfolio_manager",
]


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _add_analyst_nodes(
    graph: StateGraph, config: RunnableConfig
) -> None:
    """Build the 4 analyst ReAct agents and add them as parent-graph nodes.

    Each analyst is a compiled ReAct agent with a custom ``AgentState``
    extension that declares shared keys with ``TradingDecisionState``
    (``ticker``, ``decision_date``, ``<role>_report``) via
    ``OmitFromSchema`` annotations. We forward ``input_schema=`` from
    the compiled agent so the parent-graph composition is
    self-documenting.
    """
    market_agent = await build_market_analyst_agent(config)
    fundamentals_agent = await build_fundamentals_analyst_agent(config)
    news_agent = await build_news_analyst_agent(config)
    social_agent = await build_social_analyst_agent(config)

    graph.add_node(
        "market_analyst",
        market_agent,
        input_schema=market_agent.input_schema,
        retry_policy=_LLM_RETRY,
    )
    graph.add_node(
        "fundamentals_analyst",
        fundamentals_agent,
        input_schema=fundamentals_agent.input_schema,
        retry_policy=_LLM_RETRY,
    )
    graph.add_node(
        "news_analyst",
        news_agent,
        input_schema=news_agent.input_schema,
        retry_policy=_LLM_RETRY,
    )
    graph.add_node(
        "social_analyst",
        social_agent,
        input_schema=social_agent.input_schema,
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


# ── Builders ──────────────────────────────────────────────────────────────────


async def build_investment_debate_graph(
    config: RunnableConfig,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> CompiledStateGraph:
    """Compile analysts → Bull/Bear debate → Investment Judge sub-graph."""
    graph: StateGraph = StateGraph(TradingDecisionState)
    await _add_analyst_nodes(graph, config)
    graph.add_node("bull_researcher", bull_researcher_node, retry_policy=_LLM_RETRY)
    graph.add_node("bear_researcher", bear_researcher_node, retry_policy=_LLM_RETRY)
    graph.add_node("investment_judge", investment_judge_node, retry_policy=_LLM_RETRY)

    # START → analysts (parallel) → bull_researcher (barrier)
    for analyst in _ANALYST_NAMES:
        graph.add_edge(START, analyst)
        graph.add_edge(analyst, "bull_researcher")

    graph.add_conditional_edges(
        "bull_researcher", _route_investment_debate, _INVESTMENT_DEBATE_TARGETS
    )
    graph.add_conditional_edges(
        "bear_researcher", _route_investment_debate, _INVESTMENT_DEBATE_TARGETS
    )
    graph.add_edge("investment_judge", END)

    return graph.compile(checkpointer=checkpointer, store=store)


async def build_investment_thesis_graph(
    config: RunnableConfig,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> CompiledStateGraph:
    """Compile analysts → debate → Judge → Trader sub-graph."""
    graph: StateGraph = StateGraph(TradingDecisionState)
    await _add_analyst_nodes(graph, config)
    graph.add_node("bull_researcher", bull_researcher_node, retry_policy=_LLM_RETRY)
    graph.add_node("bear_researcher", bear_researcher_node, retry_policy=_LLM_RETRY)
    graph.add_node("investment_judge", investment_judge_node, retry_policy=_LLM_RETRY)
    graph.add_node("trader", trader_node, retry_policy=_LLM_RETRY)

    for analyst in _ANALYST_NAMES:
        graph.add_edge(START, analyst)
        graph.add_edge(analyst, "bull_researcher")

    graph.add_conditional_edges(
        "bull_researcher", _route_investment_debate, _INVESTMENT_DEBATE_TARGETS
    )
    graph.add_conditional_edges(
        "bear_researcher", _route_investment_debate, _INVESTMENT_DEBATE_TARGETS
    )
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
                            bull_researcher ⇄ bear_researcher  (N rounds)
                                     │
                                     ▼
                              investment_judge
                                     │
                                     ▼
                                  trader
                                     │
                                     ▼
                aggressive_debator → conservative_debator → neutral_debator
                                     (M rounds via round-robin)
                                     │
                                     ▼
                             portfolio_manager
                                     │
                                     ▼
                         decision_writeback → END

    Args:
        config: Active ``RunnableConfig`` — required to build the four
            analyst ReAct agents at graph-construction time.
        checkpointer: Optional LangGraph checkpointer for graph state.
        store: Optional ``BaseStore`` for reflection memory + tool-result
            caching. When ``None``, reflection bookends degrade to no-ops.
        outcomes_fetcher: Optional ``OutcomesFetcher`` for realised-return
            data (defaults to :func:`fetch_outcomes_openbb`).
    """
    graph: StateGraph = StateGraph(TradingDecisionState)

    graph.add_node(
        "reflector_resolve",
        partial(
            reflector_resolve_node, store=store, outcomes_fetcher=outcomes_fetcher
        ),
        retry_policy=_LLM_RETRY,
    )
    await _add_analyst_nodes(graph, config)
    graph.add_node("bull_researcher", bull_researcher_node, retry_policy=_LLM_RETRY)
    graph.add_node("bear_researcher", bear_researcher_node, retry_policy=_LLM_RETRY)
    graph.add_node("investment_judge", investment_judge_node, retry_policy=_LLM_RETRY)
    graph.add_node("trader", trader_node, retry_policy=_LLM_RETRY)
    graph.add_node(
        "aggressive_debator", aggressive_debator_node, retry_policy=_LLM_RETRY
    )
    graph.add_node(
        "conservative_debator", conservative_debator_node, retry_policy=_LLM_RETRY
    )
    graph.add_node("neutral_debator", neutral_debator_node, retry_policy=_LLM_RETRY)
    graph.add_node("portfolio_manager", portfolio_manager_node, retry_policy=_LLM_RETRY)
    # Pure-IO node — no LLM, no retry beyond the store's own behaviour.
    graph.add_node("decision_writeback", partial(decision_writeback_node, store=store))

    graph.add_edge(START, "reflector_resolve")
    _wire_analysts_to(graph, from_node="reflector_resolve", to_node="bull_researcher")
    graph.add_conditional_edges(
        "bull_researcher", _route_investment_debate, _INVESTMENT_DEBATE_TARGETS
    )
    graph.add_conditional_edges(
        "bear_researcher", _route_investment_debate, _INVESTMENT_DEBATE_TARGETS
    )
    graph.add_edge("investment_judge", "trader")
    graph.add_conditional_edges("trader", _route_risk_debate, _RISK_DEBATE_TARGETS)
    graph.add_conditional_edges(
        "aggressive_debator", _route_risk_debate, _RISK_DEBATE_TARGETS
    )
    graph.add_conditional_edges(
        "conservative_debator", _route_risk_debate, _RISK_DEBATE_TARGETS
    )
    graph.add_conditional_edges(
        "neutral_debator", _route_risk_debate, _RISK_DEBATE_TARGETS
    )
    graph.add_edge("portfolio_manager", "decision_writeback")
    graph.add_edge("decision_writeback", END)

    return graph.compile(checkpointer=checkpointer, store=store)
