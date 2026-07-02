"""Top-level orchestrator graph for criteria-driven analysis.

Pipeline:

    START
      │
      ▼
    ticker_classification        ← Stage 1
      │
      ├─→ criteria_definition ─┐
      └─→ valuation_methodology ┤  ← Stages 2 & 3 (parallel)
                                ▼
                          merge_criteria   ← Stage 4a (deterministic)
                                │
              (Send fan-out, one per merged criterion)
                                ▼
                       criterion_evaluation × N  ← Stage 4b
                                │
                       (operator.add fan-in)
                                ▼
                            synthesis    ← Stage 5
                                │
                                ▼
                               END

Implicit barriers (LangGraph fires a node only when all incoming edges
have data) handle the Stage 2/3 → Stage 4a synchronisation — same
pattern as ``investment_analysis.py``.

Send fan-out + ``operator.add`` reducer for criteria → evaluation
mirrors ``equity_screening.py``.
"""

from functools import partial
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Send

from muffin_agent.utils.observability import instrument_graph

from .criteria_definition_node import criteria_definition_node
from .criterion_evaluation_node import criterion_evaluation_node
from .merge_criteria import merge_criteria_node
from .state import CriteriaAnalysisState
from .synthesis import synthesis_node
from .ticker_classification import ticker_classification_node
from .valuation_methodology import valuation_methodology_node


def _fan_out_criteria(state: CriteriaAnalysisState) -> list[Send]:
    """Emit one ``Send`` per merged criterion to ``criterion_evaluation``.

    Forwards ticker, query, and the full classification payload so each
    fan-out worker has the context it needs without re-reading state.
    """
    merged: list[dict[str, Any]] = state.get("merged_criteria") or []
    classification = state.get("classification") or {}
    return [
        Send(
            "criterion_evaluation",
            {
                "ticker": state.get("ticker", ""),
                "query": state.get("query", ""),
                "criterion": criterion,
                "classification": classification,
                "criterion_evaluations": [],
            },
        )
        for criterion in merged
    ]


def build_criteria_analysis_graph(
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> CompiledStateGraph:
    """Build and compile the criteria-driven analysis orchestrator graph."""
    graph: StateGraph = StateGraph(CriteriaAnalysisState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    graph.add_node(
        "ticker_classification", partial(ticker_classification_node, store=store)
    )
    graph.add_node(
        "criteria_definition", partial(criteria_definition_node, store=store)
    )
    graph.add_node(
        "valuation_methodology", partial(valuation_methodology_node, store=store)
    )
    graph.add_node("merge_criteria", merge_criteria_node)
    graph.add_node(
        "criterion_evaluation", partial(criterion_evaluation_node, store=store)
    )
    graph.add_node("synthesis", partial(synthesis_node, store=store))

    # ── Edges ─────────────────────────────────────────────────────────────────
    graph.add_edge(START, "ticker_classification")
    graph.add_edge("ticker_classification", "criteria_definition")
    graph.add_edge("ticker_classification", "valuation_methodology")
    graph.add_edge("criteria_definition", "merge_criteria")
    graph.add_edge("valuation_methodology", "merge_criteria")
    graph.add_conditional_edges(
        "merge_criteria", _fan_out_criteria, ["criterion_evaluation"]
    )
    graph.add_edge("criterion_evaluation", "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile(checkpointer=checkpointer, store=store)


# Module-level pre-compiled graph for LangGraph Platform autodiscovery.
graph = instrument_graph(build_criteria_analysis_graph())
