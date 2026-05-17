"""Top-level graph builder for the trading_decision module.

PR 1 surface: ``build_investment_debate_graph`` runs the Bull/Bear debate
to completion and then synthesises with the Investment Judge.

PR 3 will add ``build_trading_decision_graph`` that chains:

    investment_debate → trader → risk_debate → portfolio_manager

Both graphs share the same ``TradingDecisionState`` schema so callers can
opt into the shorter or longer pipeline interchangeably.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from .conditional_logic import should_continue_investment_debate
from .nodes import bear_researcher_node, bull_researcher_node, investment_judge_node
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
