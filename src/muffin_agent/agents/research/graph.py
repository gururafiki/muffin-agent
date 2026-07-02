"""Top-level research pipeline graph.

::

    START
      │
      ▼
    classifier
      │
      ├─ skip_search ─→ writer ─→ END
      └─ default     ─→ researcher ─→ rerank ─→ writer ─→ END

Each node is a thin wrapper around its component (LLM call, deep
agent, or pure Python).  The pipeline is intentionally linear; the
researcher's internal loop (deep agent + tools) is the only
iterative step.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from muffin_agent.utils.observability import instrument_graph

from .nodes import classifier_node, rerank_node, researcher_node, writer_node
from .state import ResearchState


def _route_after_classifier(state: ResearchState) -> str:
    """Route to ``writer`` directly when classifier sets skip_search."""
    return "writer" if state.get("skip_search") else "researcher"


def build_research_graph(
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
    extra_tools: Sequence[BaseTool] | None = None,
    extra_sources: Sequence[str] | None = None,
) -> CompiledStateGraph:
    """Build the research pipeline.

    Args:
        checkpointer: Optional ``BaseCheckpointSaver`` for thread-level
            history.  CLI passes a ``SqliteSaver``; LangGraph Platform
            injects its own (so pass ``None`` for autodiscovery).
        store: Optional ``BaseStore`` for cross-agent tool-result
            caching and ``/memories/`` access.  CLI passes
            ``InMemoryStore``; Platform injects its own.
        extra_tools: Additional tools the researcher should register.
            Use this to plug in academic / news / finance / internal
            search tools.
        extra_sources: Additional source names that should appear in
            the classifier's ``sources_to_use`` enum.  Match the
            ``source_type`` your ``extra_tools`` will populate.
    """
    graph: StateGraph = StateGraph(ResearchState)

    extra_sources_list = list(extra_sources or [])
    graph.add_node(
        "classifier",
        partial(classifier_node, extra_sources=extra_sources_list),
    )
    graph.add_node(
        "researcher",
        partial(researcher_node, store=store, extra_tools=extra_tools),
    )
    graph.add_node("rerank", rerank_node)
    graph.add_node("writer", partial(writer_node, store=store))

    graph.add_edge(START, "classifier")
    graph.add_conditional_edges(
        "classifier",
        _route_after_classifier,
        {"researcher": "researcher", "writer": "writer"},
    )
    graph.add_edge("researcher", "rerank")
    graph.add_edge("rerank", "writer")
    graph.add_edge("writer", END)

    return graph.compile(checkpointer=checkpointer, store=store)


# Module-level pre-compiled graph for LangGraph Platform autodiscovery.
# The platform injects checkpointer + store on invocation; we compile
# with neither here so the import is side-effect-free.
graph = instrument_graph(build_research_graph())
