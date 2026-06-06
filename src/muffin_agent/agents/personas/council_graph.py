"""Persona council graph — parallel fan-out + LLM-mediated judge synthesis.

Topology::

    START
      │
      ▼
    persona_data_collection      ← one shared deep agent fetches everything
      │
      │  (Send fan-out: one per registered persona in PERSONA_REGISTRY)
      ▼
    ┌────────────────────────────────────────────────────────┐
    │  13 persona nodes run in parallel                       │
    │  (each writes to state.persona_signals via reducer)     │
    └────────────────────────────────────────────────────────┘
      │
      │  (fan-in: persona_signals accumulated by operator.add)
      ▼
    council_judge                ← single LLM call synthesising verdicts
      │
      ▼
    END

The council graph is registered in :data:`langgraph.json` as
``"council"`` for LangGraph Platform autodiscovery.  Callers can also
import :func:`build_council_graph` and pass their own checkpointer /
store.
"""

from __future__ import annotations

import operator
from functools import partial
from typing import Annotated, Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Send
from typing_extensions import TypedDict

from ._base import PERSONA_REGISTRY
from .data_collection import persona_data_collection_node
from .judge import council_judge_node


class CouncilState(TypedDict, total=False):
    """Shared state across the council graph.

    Inputs:
        ticker: equity ticker symbol (preserves exchange suffixes)
        query: optional investment mandate / framing
        as_of_date: optional date anchor (defaults to today inside the
            data-collection agent)

    Outputs (populated by graph execution):
        data_bundle: ``PersonaDataBundle`` dump from the data-collection step
        persona_signals: accumulated 5-tier signals from all personas
            (one ``AnalystSignal.model_dump()`` dict per persona) — uses
            ``operator.add`` reducer for parallel Send fan-out
        council_synthesis: final ``CouncilSynthesisOutput`` from the judge
    """

    ticker: str
    query: str
    as_of_date: str
    data_bundle: dict[str, Any]
    persona_signals: Annotated[list[dict[str, Any]], operator.add]
    council_synthesis: dict[str, Any]


def _fanout_personas(state: CouncilState) -> list[Send]:
    """Emit one ``Send`` per registered persona, forwarding the data bundle."""
    data_bundle = state.get("data_bundle") or {}
    ticker = state.get("ticker", "")
    query = state.get("query")
    return [
        Send(
            spec.slug,
            {
                "ticker": ticker,
                "query": query,
                "data_bundle": data_bundle,
            },
        )
        for spec in PERSONA_REGISTRY.values()
    ]


def build_council_graph(
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> CompiledStateGraph:
    """Build and compile the persona council graph.

    Args:
        checkpointer: Optional ``BaseCheckpointSaver`` for resumable runs
            (LangGraph Platform injects a Postgres-backed one automatically).
        store: Optional ``BaseStore`` for the underlying
            ``ToolResultCacheMiddleware`` that caches MCP data-fetches
            across runs.

    Returns:
        Compiled state graph ready for ``ainvoke``.
    """
    graph: StateGraph = StateGraph(CouncilState)

    # Data-collection step (shared by every persona)
    graph.add_node(
        "persona_data_collection",
        partial(persona_data_collection_node, store=store),
    )

    # One node per persona (each self-registers in PERSONA_REGISTRY on
    # personas package import).  Node names match persona slugs so the
    # ``Send`` fan-out can target them by slug.
    for spec in PERSONA_REGISTRY.values():
        graph.add_node(spec.slug, spec.node)

    # Council judge synthesises 13 signals into one consensus rating
    graph.add_node("council_judge", council_judge_node)

    # Topology
    graph.add_edge(START, "persona_data_collection")
    graph.add_conditional_edges(
        "persona_data_collection",
        _fanout_personas,
        [spec.slug for spec in PERSONA_REGISTRY.values()],
    )
    for spec in PERSONA_REGISTRY.values():
        graph.add_edge(spec.slug, "council_judge")
    graph.add_edge("council_judge", END)

    return graph.compile(checkpointer=checkpointer, store=store)


# Module-level compiled graph for LangGraph Platform autodiscovery via
# ``langgraph.json``.  Lacks a real checkpointer / store — those are
# injected by the platform at request time.
graph = build_council_graph()
