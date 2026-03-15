"""Equity screening graph.

Entry point for ``muffin screen --query "..."``.  Discovers candidate tickers
from the market, evaluates each in parallel, then ranks and compares them.

Graph topology
--------------

    START
      │
      ▼
    idea_sourcing              ← find candidate tickers
      │
      ├──→ market_regime ──────┐
      └──→ sector_analysis ────┤
                               ↓  (context_ready barrier: both must complete)
                         context_ready
                               │
               (Send fan-out: one per ticker in state.tickers)
                               ↓
    ┌──────────────────────────────────────────────────────────┐
    │  build_investment_analysis_graph()  ×  N tickers          │
    │  (each runs the full 7-stage pipeline in parallel)        │
    └──────────────────────────────────────────────────────────┘
      │
      │  (fan-in: ScreeningState.theses accumulated by operator.add)
      ▼
    comparison                 ← rank and select best ideas
      │
      ▼
    END

Shared context optimisation
----------------------------
``market_regime`` and ``sector_analysis`` run once on the outer graph before
the fan-out — not repeated per ticker.  The ``context_ready`` no-op barrier
fires only when both complete.  Shared context is injected into each
``TickerAnalysisState`` when the ``Send`` objects are emitted.
"""

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from muffin_agent.agents.investment import (
    comparison_node,
    idea_sourcing_node,
    market_regime_node,
    sector_analysis_node,
)
from muffin_agent.agents.investment.state import ScreeningState, TickerAnalysisState
from muffin_agent.agents.investment_analysis import build_investment_analysis_graph


def _fan_out_tickers(state: ScreeningState) -> list[Send]:
    """Emit one Send per screened ticker, forwarding shared context."""
    return [
        Send(
            "analyze_ticker",
            TickerAnalysisState(
                ticker=ticker,
                query=state["query"],
                market_regime=state.get("market_regime", {}),
                sector_view=state.get("sector_view", {}),
                company_analysis={},
                forecast={},
                risk_assessment={},
                valuation={},
                thesis={},
            ),
        )
        for ticker in state["tickers"]
    ]


async def _analyze_ticker(
    state: TickerAnalysisState, config: RunnableConfig
) -> dict[str, Any]:
    """Run the full investment analysis pipeline for one ticker.

    Invoked once per ticker by the fan-out.  Appends the completed thesis into
    ``ScreeningState.theses`` via the ``operator.add`` reducer.
    """
    analysis_graph = build_investment_analysis_graph()
    result: TickerAnalysisState = await analysis_graph.ainvoke(state, config)
    return {"theses": [result.get("thesis", {})]}


def build_equity_screening_graph() -> StateGraph:
    """Build and compile the equity screening graph."""
    graph: StateGraph = StateGraph(ScreeningState)

    graph.add_node("idea_sourcing", idea_sourcing_node)
    graph.add_node("market_regime", market_regime_node)
    graph.add_node("sector_analysis", sector_analysis_node)
    graph.add_node("context_ready", lambda s: {})
    graph.add_node("analyze_ticker", _analyze_ticker)
    graph.add_node("comparison", comparison_node)

    graph.add_edge(START, "idea_sourcing")
    graph.add_edge("idea_sourcing", "market_regime")
    graph.add_edge("idea_sourcing", "sector_analysis")
    graph.add_edge("market_regime", "context_ready")
    graph.add_edge("sector_analysis", "context_ready")
    graph.add_conditional_edges("context_ready", _fan_out_tickers, ["analyze_ticker"])
    graph.add_edge("analyze_ticker", "comparison")
    graph.add_edge("comparison", END)

    return graph.compile()
