"""Top-level graph builders for the trading_decision module.

Three composable graphs share the same ``TradingDecisionState`` schema so
callers can opt into the depth they need:

* :func:`build_investment_debate_graph` — Bull/Bear debate → Investment Judge.
  Smallest useful slice when you just want a structured directional view.
* :func:`build_investment_thesis_graph` — debate → Judge → Trader. Adds
  operational translation (entry/stop/take-profit/sizing/time_horizon).
* :func:`build_trading_decision_graph` — full pipeline including the
  reflector_resolve / decision_writeback reflection bookends and the
  3-way risk debate + Portfolio Manager.

All builders share the routing functions in this module; per-node
behaviour lives in the per-role files (``researchers/``, ``trader.py``,
``risk_debate/``, ``portfolio_manager.py``, ``reflection/``).

Routing is **graph-level** — nodes return state-update dicts only, never
``Command(goto=...)``. Conditional edges (list form) decide next nodes
based on the accumulated debate-response lists.

Retry layering: every LLM-call node carries a ``RetryPolicy`` so LangGraph
retries the whole node on exception (one layer above LangChain's
``with_retry`` inside each node).
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


# ── Builders ──────────────────────────────────────────────────────────────────


def build_investment_debate_graph(
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> CompiledStateGraph:
    """Compile the Bull/Bear debate → Investment Judge sub-graph."""
    graph: StateGraph = StateGraph(TradingDecisionState)
    graph.add_node("bull_researcher", bull_researcher_node, retry_policy=_LLM_RETRY)
    graph.add_node("bear_researcher", bear_researcher_node, retry_policy=_LLM_RETRY)
    graph.add_node("investment_judge", investment_judge_node, retry_policy=_LLM_RETRY)

    graph.add_conditional_edges(
        START, _route_investment_debate, _INVESTMENT_DEBATE_TARGETS
    )
    graph.add_conditional_edges(
        "bull_researcher", _route_investment_debate, _INVESTMENT_DEBATE_TARGETS
    )
    graph.add_conditional_edges(
        "bear_researcher", _route_investment_debate, _INVESTMENT_DEBATE_TARGETS
    )
    graph.add_edge("investment_judge", END)

    return graph.compile(checkpointer=checkpointer, store=store)


def build_investment_thesis_graph(
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> CompiledStateGraph:
    """Compile debate → Judge → Trader sub-graph."""
    graph: StateGraph = StateGraph(TradingDecisionState)
    graph.add_node("bull_researcher", bull_researcher_node, retry_policy=_LLM_RETRY)
    graph.add_node("bear_researcher", bear_researcher_node, retry_policy=_LLM_RETRY)
    graph.add_node("investment_judge", investment_judge_node, retry_policy=_LLM_RETRY)
    graph.add_node("trader", trader_node, retry_policy=_LLM_RETRY)

    graph.add_conditional_edges(
        START, _route_investment_debate, _INVESTMENT_DEBATE_TARGETS
    )
    graph.add_conditional_edges(
        "bull_researcher", _route_investment_debate, _INVESTMENT_DEBATE_TARGETS
    )
    graph.add_conditional_edges(
        "bear_researcher", _route_investment_debate, _INVESTMENT_DEBATE_TARGETS
    )
    graph.add_edge("investment_judge", "trader")
    graph.add_edge("trader", END)

    return graph.compile(checkpointer=checkpointer, store=store)


def build_trading_decision_graph(
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
    *,
    outcomes_fetcher: OutcomesFetcher | None = None,
) -> CompiledStateGraph:
    """Compile the full pipeline.

    Topology::

        START → reflector_resolve
                  │
                  ▼
        bull_researcher ⇄ bear_researcher  (N rounds via conditional edges)
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
        checkpointer: Optional LangGraph checkpointer for graph state.
        store: Optional ``BaseStore`` for reflection memory + tool-result
            caching. When ``None``, reflection bookends degrade to no-ops.
        outcomes_fetcher: Optional ``OutcomesFetcher`` for realised-return
            data (defaults to :func:`fetch_outcomes_openbb`).
    """
    graph: StateGraph = StateGraph(TradingDecisionState)

    graph.add_node(
        "reflector_resolve",
        partial(reflector_resolve_node, store=store, outcomes_fetcher=outcomes_fetcher),
        retry_policy=_LLM_RETRY,
    )
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
    # Pure-IO node — no LLM, no retry needed beyond the store's own behaviour.
    graph.add_node("decision_writeback", partial(decision_writeback_node, store=store))

    graph.add_edge(START, "reflector_resolve")
    graph.add_conditional_edges(
        "reflector_resolve", _route_investment_debate, _INVESTMENT_DEBATE_TARGETS
    )
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
