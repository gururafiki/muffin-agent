"""Auto-discovery screening graph.

Entry point for ``muffin screen --query "..."``.  Discovers candidate
tickers from the market, evaluates each in parallel, then ranks and compares
them to produce a final shortlist.

Graph topology
--------------

    START
      │
      ▼
    idea_sourcing           ← Stage 1: find candidate tickers
      │
      ├──→ market_regime ─────┐
      └──→ sector_analysis ───┤
                              ↓ (context_ready barrier: both must complete)
                        context_ready
                              │
                  (Send fan-out: one per ticker)
                              ↓
    ┌─────────────────────────────────────────────────────────┐
    │  per-ticker sub-graph (runs in parallel for each ticker) │
    │                                                          │
    │   market_regime ─────────────────────────────────┐      │
    │   sector_analysis ───────────────────────────────┤      │
    │   company_analysis ──────────────────────────────┘      │
    │                       ↓ (barrier)                        │
    │   forecasting ─────────────────────────────────┐        │
    │   risk_assessment ─────────────────────────────┘        │
    │                       ↓ (barrier)                        │
    │   valuation ──────────────────────────────────────→      │
    │   thesis_synthesis ───────────────────────────────→      │
    └──────────────────────────────────────────────────────────┘
      │
      │  (fan-in: theses list accumulated by operator.add reducer)
      ▼
    comparison              ← rank and select best ideas
      │
      ▼
    END

Shared context optimisation
----------------------------
``market_regime`` and ``sector_analysis`` run on the **outer graph** before
the fan-out so they execute *once* rather than once per ticker.  Both run in
**parallel** after ``idea_sourcing``; the ``context_ready`` no-op barrier node
fires only when both have completed.  Their results are injected into each
``TickerAnalysisState`` when the ``Send`` objects are created.

Replaceability
--------------
Pass ``stages`` and/or ``sourcing_node`` overrides to
``build_screening_graph()``::

    graph = build_screening_graph(
        sourcing_node=my_country_sector_sourcing_node,
        stages={"valuation": my_sum_of_parts_node},
    )
"""

from collections.abc import Callable
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from muffin_agent.pipeline.graphs.analysis_graph import build_analysis_graph
from muffin_agent.pipeline.stages import (
    comparison_node,
    idea_sourcing_node,
    market_regime_node,
    sector_analysis_node,
)
from muffin_agent.pipeline.state import PipelineState, TickerAnalysisState


def _fan_out_tickers(state: PipelineState) -> list[Send]:
    """Conditional-edge function: emit one Send per screened ticker.

    Each Send spawns an independent copy of the per-ticker sub-graph with
    the ticker's ``TickerAnalysisState``.  Shared context (``market_regime``
    and ``sector_view``) is forwarded from the outer graph so sub-graph
    workers do not need to re-fetch it.
    """
    return [
        Send(
            "analyze_ticker",
            TickerAnalysisState(
                ticker=ticker,
                query=state["query"],
                market_regime=state.get("market_regime", {}),
                sector_view=state.get("sector_view", {}),
                # remaining fields start empty; each stage node fills them in
                company_analysis={},
                forecast={},
                risk_assessment={},
                valuation={},
                thesis={},
            ),
        )
        for ticker in state["tickers"]
    ]


def build_screening_graph(
    sourcing_node: Callable[..., Any] | None = None,
    stages: dict[str, Callable[..., Any]] | None = None,
) -> Any:
    """Build and compile the auto-discovery screening graph.

    The screening graph runs idea sourcing, shared context stages, and
    per-ticker parallel analysis before aggregating results.

    Args:
        sourcing_node: Optional replacement for ``idea_sourcing_node``.
            Use this to plug in alternative sourcing strategies
            (country → sector drill-down, factor-tilt screen, watchlist, etc.).
        stages: Optional ``{stage_name: node_fn}`` overrides forwarded to
            each per-ticker ``build_analysis_graph()`` call.  Valid keys
            match those in ``analysis_graph.py``.

    Returns:
        A compiled ``CompiledStateGraph[PipelineState]``.

    Example — custom sourcing + custom valuation::

        graph = build_screening_graph(
            sourcing_node=my_em_country_sector_node,
            stages={"valuation": my_sum_of_parts_node},
        )
    """
    sourcing = sourcing_node or idea_sourcing_node
    _stages = stages  # captured by _collect_thesis closure below

    async def _collect_thesis(
        state: TickerAnalysisState, config: RunnableConfig
    ) -> dict[str, Any]:
        """Fan-in adapter: run the per-ticker sub-graph and collect the thesis.

        Wraps ``build_analysis_graph()`` so that the outer screening graph can
        invoke the full analysis pipeline as a single node, then appends the
        completed thesis into the outer ``PipelineState.theses`` list via the
        ``operator.add`` reducer.

        ``stages`` overrides are forwarded into the inner graph so custom stage
        implementations propagate correctly without mutating any module-level state.
        """
        analysis_graph = build_analysis_graph(stages=_stages)
        result: TickerAnalysisState = await analysis_graph.ainvoke(state, config)
        return {"theses": [result.get("thesis", {})]}

    outer: StateGraph = StateGraph(PipelineState)

    # ── Outer nodes ───────────────────────────────────────────────────────────
    outer.add_node("idea_sourcing", sourcing)

    # Shared context (run once before fan-out, in parallel with each other)
    outer.add_node("market_regime", market_regime_node)
    outer.add_node("sector_analysis", sector_analysis_node)

    # No-op barrier: fires only after both market_regime AND sector_analysis complete
    outer.add_node("context_ready", lambda s: {})

    # Per-ticker worker (wraps full analysis sub-graph)
    outer.add_node("analyze_ticker", _collect_thesis)

    # Comparison / ranking
    outer.add_node("comparison", comparison_node)

    # ── Outer edges ───────────────────────────────────────────────────────────
    outer.add_edge(START, "idea_sourcing")

    # market_regime and sector_analysis run in parallel after sourcing
    outer.add_edge("idea_sourcing", "market_regime")
    outer.add_edge("idea_sourcing", "sector_analysis")

    # context_ready fires only when BOTH shared-context nodes complete
    outer.add_edge("market_regime", "context_ready")
    outer.add_edge("sector_analysis", "context_ready")

    # Fan-out: one Send per ticker; shared context is injected via _fan_out_tickers
    outer.add_conditional_edges("context_ready", _fan_out_tickers, ["analyze_ticker"])

    # Fan-in: each worker appends to theses; comparison fires when all done
    outer.add_edge("analyze_ticker", "comparison")
    outer.add_edge("comparison", END)

    return outer.compile()

