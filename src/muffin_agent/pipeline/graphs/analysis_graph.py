"""Direct-ticker analysis graph.

Entry point for ``muffin analyze <TICKER>``.  Accepts a single ticker and
runs the full 7-stage analysis pipeline, producing a completed investment
thesis as the final state.

Parallel execution groups
-------------------------
Group 1 — all three nodes start simultaneously from ``START``:

    market_regime ─────────────────────────────────────────────┐
    sector_analysis ───────────────────────────────────────────┤ (barrier)
    company_analysis ──────────────────────────────────────────┘
                                                               ↓
Group 2 — start after Group 1 barrier; run in parallel:

    forecasting ───────────────────────────────────────────────┐
    risk_assessment ───────────────────────────────────────────┘ (barrier)
                                                               ↓
Group 3 — sequential:

    valuation ─────────────────────────────────────────────────→ thesis_synthesis → END

LangGraph handles barriers automatically: a node fires only when *all* of
its incoming edges have been satisfied.

Replaceability
--------------
Pass a ``stages`` override dict to ``build_analysis_graph()`` to swap any
stage without touching graph wiring::

    from muffin_agent.pipeline.graphs.analysis_graph import build_analysis_graph
    from my_custom_stages import my_valuation_node

    graph = build_analysis_graph(stages={"valuation": my_valuation_node})
"""

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from muffin_agent.pipeline.stages import (
    company_analysis_node,
    forecasting_node,
    market_regime_node,
    risk_assessment_node,
    sector_analysis_node,
    thesis_synthesis_node,
    valuation_node,
)
from muffin_agent.pipeline.state import TickerAnalysisState

# Default stage implementations — overridable via build_analysis_graph(stages={...})
_DEFAULT_STAGES: dict[str, Callable[..., Any]] = {
    "market_regime": market_regime_node,
    "sector_analysis": sector_analysis_node,
    "company_analysis": company_analysis_node,
    "forecasting": forecasting_node,
    "risk_assessment": risk_assessment_node,
    "valuation": valuation_node,
    "thesis_synthesis": thesis_synthesis_node,
}


def build_analysis_graph(
    stages: dict[str, Callable[..., Any]] | None = None,
) -> Any:
    """Build and compile the direct-ticker analysis graph.

    Args:
        stages: Optional mapping of ``{stage_name: node_fn}`` overrides.
            Any stage not present in this dict uses the default implementation.
            Valid keys: ``market_regime``, ``sector_analysis``,
            ``company_analysis``, ``forecasting``, ``risk_assessment``,
            ``valuation``, ``thesis_synthesis``.

    Returns:
        A compiled ``CompiledStateGraph[TickerAnalysisState]``.

    Example — swap the valuation stage::

        graph = build_analysis_graph(stages={"valuation": my_dcf_node})
        result = await graph.ainvoke({"ticker": "AAPL", "query": "..."})
    """
    s = {**_DEFAULT_STAGES, **(stages or {})}

    graph: StateGraph = StateGraph(TickerAnalysisState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("market_regime", s["market_regime"])
    graph.add_node("sector_analysis", s["sector_analysis"])
    graph.add_node("company_analysis", s["company_analysis"])
    graph.add_node("forecasting", s["forecasting"])
    graph.add_node("risk_assessment", s["risk_assessment"])
    graph.add_node("valuation", s["valuation"])
    graph.add_node("thesis_synthesis", s["thesis_synthesis"])

    # ── Group 1: all start in parallel from START ─────────────────────────────
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
    # valuation waits for forecasting + risk_assessment + sector_analysis
    # (sector_analysis is already done by this point — the edge carries its
    # completed output forward so valuation can access peer multiples)
    graph.add_edge("forecasting", "valuation")
    graph.add_edge("risk_assessment", "valuation")
    graph.add_edge("sector_analysis", "valuation")

    # ── valuation → thesis ────────────────────────────────────────────────────
    graph.add_edge("valuation", "thesis_synthesis")

    # ── thesis → END ──────────────────────────────────────────────────────────
    graph.add_edge("thesis_synthesis", END)

    return graph.compile()
