"""Per-ticker investment analysis graph.

Entry point for ``muffin analyze <TICKER>``.  Accepts a single ticker and
investment mandate, runs a 7-stage analysis pipeline, and returns a completed
``TickerAnalysisState`` with the investment thesis.

Parallel execution groups
-------------------------
Group 1 — start simultaneously from START (no inter-dependencies):

    market_regime ──────────────────────────────────────────────┐
    sector_analysis ────────────────────────────────────────────┤ (barrier)
    company_analysis ───────────────────────────────────────────┘
                                                                ↓
Group 2 — start after Group 1 barrier, run in parallel:

    forecasting ────────────────────────────────────────────────┐
    risk_assessment ────────────────────────────────────────────┘ (barrier)
                                                                ↓
Group 3 — sequential:

    valuation ──────────────────────────────────────────────────→ thesis_synthesis → END

LangGraph fires a node only when all its incoming edges have data, so barrier
synchronisation is implicit — no extra code required.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from muffin_agent.agents.investment import (
    company_analysis_node,
    forecasting_node,
    market_regime_node,
    risk_assessment_node,
    sector_analysis_node,
    thesis_synthesis_node,
    valuation_node,
)
from muffin_agent.agents.investment.state import TickerAnalysisState

_RETRY = RetryPolicy(max_attempts=2, initial_interval=5.0)
"""Retry once on transient errors (network, rate limits).

Only effective for exceptions that ``run_deep_agent_node`` propagates
(see ``TRANSIENT_ERRORS`` in ``investment.utils``).
"""


def build_investment_analysis_graph(
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Build and compile the per-ticker investment analysis graph.

    Args:
        checkpointer: Optional checkpoint saver for state persistence.
            When deployed on LangGraph Platform the server injects a
            ``PostgresSaver`` automatically via ``DATABASE_URI`` — pass
            ``None`` in that context.  For CLI usage pass
            ``InMemorySaver()`` or ``SqliteSaver`` to enable
            resume-from-failure.
    """
    graph: StateGraph = StateGraph(TickerAnalysisState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    graph.add_node("market_regime", market_regime_node, retry_policy=_RETRY)
    graph.add_node("sector_analysis", sector_analysis_node, retry_policy=_RETRY)
    graph.add_node(
        "company_analysis", company_analysis_node, retry_policy=_RETRY
    )
    graph.add_node("forecasting", forecasting_node, retry_policy=_RETRY)
    graph.add_node(
        "risk_assessment", risk_assessment_node, retry_policy=_RETRY
    )
    graph.add_node("valuation", valuation_node, retry_policy=_RETRY)
    # thesis_synthesis is pure reasoning (no MCP) — no retry needed
    graph.add_node("thesis_synthesis", thesis_synthesis_node)

    # ── Group 1: all start in parallel ───────────────────────────────────────
    graph.add_edge(START, "market_regime")
    graph.add_edge(START, "sector_analysis")
    graph.add_edge(START, "company_analysis")

    # ── Group 1 → Group 2 barrier ─────────────────────────────────────────────
    # forecasting waits for all three Group 1 nodes
    graph.add_edge("market_regime", "forecasting")
    graph.add_edge("sector_analysis", "forecasting")
    graph.add_edge("company_analysis", "forecasting")

    # risk_assessment waits for company_analysis + market_regime
    graph.add_edge("company_analysis", "risk_assessment")
    graph.add_edge("market_regime", "risk_assessment")

    # ── Group 2 → valuation barrier ───────────────────────────────────────────
    graph.add_edge("forecasting", "valuation")
    graph.add_edge("risk_assessment", "valuation")
    graph.add_edge("sector_analysis", "valuation")  # peer multiples

    # ── valuation → thesis → END ──────────────────────────────────────────────
    graph.add_edge("valuation", "thesis_synthesis")
    graph.add_edge("thesis_synthesis", END)

    return graph.compile(checkpointer=checkpointer)
