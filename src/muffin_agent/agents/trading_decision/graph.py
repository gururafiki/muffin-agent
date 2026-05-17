"""Top-level graph builders for the trading_decision module.

Three composable graphs share the same ``TradingDecisionState`` schema so
callers can opt into the depth they need:

* :func:`build_investment_debate_graph` (PR 1) — Bull/Bear debate →
  Investment Judge. Smallest useful slice when you just want a structured
  directional view.
* :func:`build_investment_thesis_graph` (PR 2) — debate → Judge → Trader.
  Adds operational translation (entry / stop / take_profit / sizing /
  time_horizon). The common "thesis + actionable instruction" surface.
* :func:`build_trading_decision_graph` (PR 3, planned) — thesis → 3-way
  risk debate → Portfolio Manager. The full pipeline with the canonical
  5-tier rating.

All three accept the same ``analysis_context`` input and produce a
superset of state keys, so swapping deeper for shallower (or vice-versa)
is a one-line CLI change.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from .conditional_logic import should_continue_investment_debate
from .nodes import (
    bear_researcher_node,
    bull_researcher_node,
    investment_judge_node,
    trader_node,
)
from .state import TradingDecisionState


def build_investment_debate_graph(
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> CompiledStateGraph:
    """Build the Bull/Bear investment debate sub-graph.

    Topology::

        START
          │
          ▼
        bull_researcher ─┐
                         │
                         ▼
               (conditional alternation by speaker tag + count)
                         │
                         ├─→ bear_researcher ─┐
                         │                    │
                         │   (loops back via conditional edges)
                         │                    │
                         ▼                    ▼
                investment_judge ◄────────────┘
                         │
                         ▼
                        END

    The conditional router (``should_continue_investment_debate``) reads
    ``count`` and the speaker tag on the most recent response. It exits
    to ``investment_judge`` once ``count >= 2 * max_investment_debate_rounds``
    (default 2 rounds = 4 total turns).

    Args:
        checkpointer: Optional LangGraph checkpoint saver for resumability.
        store: Optional shared store (used by downstream PRs).
    """
    graph: StateGraph = StateGraph(TradingDecisionState)

    graph.add_node("bull_researcher", bull_researcher_node)
    graph.add_node("bear_researcher", bear_researcher_node)
    graph.add_node("investment_judge", investment_judge_node)

    # Start by giving the Bull the floor.
    graph.add_edge(START, "bull_researcher")

    # After each debater speaks, the router decides who's next or exits to
    # the judge once the round budget is exhausted.
    graph.add_conditional_edges(
        "bull_researcher",
        should_continue_investment_debate,
        {
            "bear_researcher": "bear_researcher",
            "bull_researcher": "bull_researcher",
            "investment_judge": "investment_judge",
        },
    )
    graph.add_conditional_edges(
        "bear_researcher",
        should_continue_investment_debate,
        {
            "bear_researcher": "bear_researcher",
            "bull_researcher": "bull_researcher",
            "investment_judge": "investment_judge",
        },
    )
    graph.add_edge("investment_judge", END)

    return graph.compile(checkpointer=checkpointer, store=store)


def build_investment_thesis_graph(
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> CompiledStateGraph:
    """Build the debate-and-translate sub-graph.

    Extends :func:`build_investment_debate_graph` with a Trader node that
    converts the Investment Judge's directional view into an executable
    proposal (action, entry, stop, take-profit, position sizing, time
    horizon).

    Topology::

        START → bull_researcher ↔ bear_researcher
                       │
                       ▼
                investment_judge
                       │
                       ▼
                     trader
                       │
                       ▼
                      END

    Output state keys (in addition to those from the debate graph):
        * ``trader`` — :class:`TraderOutput` dump or error fallback dict.
    """
    graph: StateGraph = StateGraph(TradingDecisionState)

    graph.add_node("bull_researcher", bull_researcher_node)
    graph.add_node("bear_researcher", bear_researcher_node)
    graph.add_node("investment_judge", investment_judge_node)
    graph.add_node("trader", trader_node)

    graph.add_edge(START, "bull_researcher")
    graph.add_conditional_edges(
        "bull_researcher",
        should_continue_investment_debate,
        {
            "bear_researcher": "bear_researcher",
            "bull_researcher": "bull_researcher",
            "investment_judge": "investment_judge",
        },
    )
    graph.add_conditional_edges(
        "bear_researcher",
        should_continue_investment_debate,
        {
            "bear_researcher": "bear_researcher",
            "bull_researcher": "bull_researcher",
            "investment_judge": "investment_judge",
        },
    )
    graph.add_edge("investment_judge", "trader")
    graph.add_edge("trader", END)

    return graph.compile(checkpointer=checkpointer, store=store)
