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

import operator
from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import RetryPolicy, Send
from typing_extensions import TypedDict

from .investment import (
    MarketRegimeInputState,
    SectorAnalysisInputState,
    comparison_node,
    create_market_regime_agent,
    create_sector_analysis_agent,
    idea_sourcing_node,
)
from .investment.state import ScreeningState, TickerAnalysisState
from .investment_analysis import build_investment_analysis_graph

_AGENT_RETRY = RetryPolicy(max_attempts=2)


class _TickerWorkerState(TickerAnalysisState, total=False):
    """Per-ticker worker state: the analysis state plus the fan-in accumulator."""

    theses: Annotated[list[dict[str, Any]], operator.add]


class _TickerWorkerOutput(TypedDict, total=False):
    """Only the accumulator propagates back to ``ScreeningState``.

    Restricting the worker's output is mandatory: N parallel workers each carrying
    the full ``TickerAnalysisState`` back would write the parent's single-value
    ``ticker`` / ``market_regime`` / ``sector_view`` channels concurrently and raise
    ``InvalidUpdateError``.
    """

    theses: Annotated[list[dict[str, Any]], operator.add]


def _collect_thesis_node(state: _TickerWorkerState) -> dict[str, Any]:
    """Lift this ticker's thesis into the screening-level accumulator."""
    return {"theses": [state.get("thesis") or {}]}


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


async def build_equity_screening_graph(
    config: RunnableConfig | None = None,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> CompiledStateGraph:
    """Build and compile the equity screening graph.

    Async because every compiled agent — the two shared-context stages and the whole
    per-ticker analysis subgraph — is built once here rather than per invocation.
    """
    cfg: RunnableConfig = config or {}
    market_regime_agent = await create_market_regime_agent(cfg, store=store)
    sector_analysis_agent = await create_sector_analysis_agent(cfg, store=store)
    analysis_graph = await build_investment_analysis_graph(cfg, store=store)

    # The per-ticker worker: the analysis subgraph as a real node, then a pure node
    # lifting its thesis into the accumulator. Adding the subgraph directly (rather
    # than `.ainvoke`-ing it inside a function) is what gives every stage inside it a
    # checkpoint namespace of its own.
    worker: StateGraph = StateGraph(
        _TickerWorkerState, output_schema=_TickerWorkerOutput
    )
    worker.add_node("analysis", analysis_graph)
    worker.add_node("collect_thesis", _collect_thesis_node)
    worker.add_edge(START, "analysis")
    worker.add_edge("analysis", "collect_thesis")
    worker.add_edge("collect_thesis", END)

    graph: StateGraph = StateGraph(ScreeningState)

    graph.add_node("idea_sourcing", idea_sourcing_node)
    graph.add_node(
        "market_regime",
        market_regime_agent,
        input_schema=MarketRegimeInputState,
        retry_policy=_AGENT_RETRY,
    )
    graph.add_node(
        "sector_analysis",
        sector_analysis_agent,
        input_schema=SectorAnalysisInputState,
        retry_policy=_AGENT_RETRY,
    )
    graph.add_node("context_ready", lambda s: {})
    graph.add_node("analyze_ticker", worker.compile())
    graph.add_node("comparison", comparison_node)

    graph.add_edge(START, "idea_sourcing")
    graph.add_edge("idea_sourcing", "market_regime")
    graph.add_edge("idea_sourcing", "sector_analysis")
    graph.add_edge("market_regime", "context_ready")
    graph.add_edge("sector_analysis", "context_ready")
    graph.add_conditional_edges("context_ready", _fan_out_tickers, ["analyze_ticker"])
    graph.add_edge("analyze_ticker", "comparison")
    graph.add_edge("comparison", END)

    return graph.compile(checkpointer=checkpointer, store=store)
