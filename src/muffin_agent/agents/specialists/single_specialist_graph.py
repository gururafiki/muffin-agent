"""Standalone single-specialist graph — powers ``muffin technicals`` / ``sentiment``.

Topology mirrors the single-persona graph::

    START → persona_data_collection → <specialist> → END

The specialist nodes share the persona input/output state contract so
the data-collection step from the personas package is reused as-is.
"""

from __future__ import annotations

import operator
from functools import partial
from typing import Annotated, Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from typing_extensions import TypedDict

from ..personas.data_collection import persona_data_collection_node
from ._base import SPECIALIST_REGISTRY


class SingleSpecialistState(TypedDict, total=False):
    """State for the single-specialist graph (same shape as SinglePersonaState)."""

    ticker: str
    query: str
    as_of_date: str
    data_bundle: dict[str, Any]
    persona_signals: Annotated[list[dict[str, Any]], operator.add]


def build_single_specialist_graph(
    specialist_slug: str,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> CompiledStateGraph:
    """Build a graph that runs data collection + one specialist.

    Args:
        specialist_slug: One of the slugs in :data:`SPECIALIST_REGISTRY`
            (currently ``"technicals"`` or ``"sentiment"``).
        checkpointer: Optional ``BaseCheckpointSaver`` for resumable runs.
        store: Optional ``BaseStore`` for cross-run cache.

    Raises:
        KeyError: when ``specialist_slug`` is not in the registry.
    """
    if specialist_slug not in SPECIALIST_REGISTRY:
        raise KeyError(
            f"Unknown specialist slug {specialist_slug!r}. "
            f"Available: {', '.join(sorted(SPECIALIST_REGISTRY))}"
        )
    spec = SPECIALIST_REGISTRY[specialist_slug]

    graph: StateGraph = StateGraph(SingleSpecialistState)
    graph.add_node(
        "persona_data_collection",
        partial(persona_data_collection_node, store=store),
    )
    graph.add_node(spec.slug, spec.node)
    graph.add_edge(START, "persona_data_collection")
    graph.add_edge("persona_data_collection", spec.slug)
    graph.add_edge(spec.slug, END)
    return graph.compile(checkpointer=checkpointer, store=store)
