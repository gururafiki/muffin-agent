"""Standalone single-persona graph — powers ``muffin persona <slug> <TICKER>``.

Topology::

    START → persona_data_collection → <chosen_persona> → END

No council, no judge — one persona's verdict in isolation.  Useful for
prompt iteration on a single persona and for the standalone CLI.
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

from ._base import PERSONA_REGISTRY
from .data_collection import persona_data_collection_node


class SinglePersonaState(TypedDict, total=False):
    """State for the single-persona graph."""

    ticker: str
    query: str
    as_of_date: str
    data_bundle: dict[str, Any]
    persona_signals: Annotated[list[dict[str, Any]], operator.add]


def build_single_persona_graph(
    persona_slug: str,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> CompiledStateGraph:
    """Build and compile a single-persona graph for ``persona_slug``.

    Args:
        persona_slug: Slug of the persona to run (must exist in
            ``PERSONA_REGISTRY``).
        checkpointer: Optional checkpointer for resumable runs.
        store: Optional ``BaseStore`` for cross-run cache.

    Returns:
        Compiled state graph: data collection → persona → END.

    Raises:
        KeyError: when *persona_slug* is not in :data:`PERSONA_REGISTRY`.
    """
    if persona_slug not in PERSONA_REGISTRY:
        raise KeyError(
            f"Unknown persona slug {persona_slug!r}. "
            f"Available: {', '.join(sorted(PERSONA_REGISTRY))}"
        )
    spec = PERSONA_REGISTRY[persona_slug]

    graph: StateGraph = StateGraph(SinglePersonaState)
    graph.add_node(
        "persona_data_collection",
        partial(persona_data_collection_node, store=store),
    )
    graph.add_node(spec.slug, spec.node)

    graph.add_edge(START, "persona_data_collection")
    graph.add_edge("persona_data_collection", spec.slug)
    graph.add_edge(spec.slug, END)

    return graph.compile(checkpointer=checkpointer, store=store)
