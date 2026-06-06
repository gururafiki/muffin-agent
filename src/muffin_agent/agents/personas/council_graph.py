"""Persona council graph — parallel fan-out + LLM-mediated judge synthesis.

Topology::

    START
      │
      │  (one edge per persona; 13 + optional 2 specialists)
      ▼
    ┌────────────────────────────────────────────────────────┐
    │  Each persona is a compiled subgraph (collect_data     │
    │  ReAct + compute_evidence + render_verdict) doing      │
    │  its OWN MCP fetches via cached_invoke / middleware    │
    │  cache.  Specialists are deterministic 2-node graphs.  │
    └────────────────────────────────────────────────────────┘
      │
      │  (fan-in barrier: persona_signals accumulated via operator.add)
      ▼
    council_judge                ← single LLM call synthesising verdicts
      │
      ▼
    END

The council graph is registered in :data:`langgraph.json` as
``"council"`` for LangGraph Platform autodiscovery.  Callers can also
import :func:`build_council_graph` and pass their own checkpointer /
store.

**v4 refactor**: this module no longer depends on
``PERSONA_REGISTRY`` / ``SPECIALIST_REGISTRY``.  Every persona and
specialist is imported by name and added directly via
``StateGraph.add_node(compiled_subgraph, input_schema=…)``.  Each persona
owns its own data fetching — there is no shared front-of-flow
``persona_data_collection_node`` step any more.
"""

from __future__ import annotations

import asyncio
import operator
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import RetryPolicy
from typing_extensions import TypedDict

from .aswath_damodaran import build_aswath_damodaran_agent
from .ben_graham import build_ben_graham_agent
from .bill_ackman import build_bill_ackman_agent
from .cathie_wood import build_cathie_wood_agent
from .charlie_munger import build_charlie_munger_agent
from .judge import council_judge_node
from .michael_burry import build_michael_burry_agent
from .mohnish_pabrai import build_mohnish_pabrai_agent
from .nassim_taleb import build_nassim_taleb_agent
from .peter_lynch import build_peter_lynch_agent
from .phil_fisher import build_phil_fisher_agent
from .rakesh_jhunjhunwala import build_rakesh_jhunjhunwala_agent
from .stanley_druckenmiller import build_stanley_druckenmiller_agent
from .warren_buffett import build_warren_buffett_agent

_LLM_RETRY = RetryPolicy(max_attempts=2)


PersonaBuilder = Callable[[RunnableConfig], Awaitable[CompiledStateGraph]]
SpecialistBuilder = Callable[[], CompiledStateGraph]


# Slug → async builder map. Hardcoded list — adding a persona = add one
# entry here + import the new ``build_*_agent`` factory above.
PERSONA_BUILDERS: list[tuple[str, PersonaBuilder]] = [
    ("warren_buffett", build_warren_buffett_agent),
    ("ben_graham", build_ben_graham_agent),
    ("cathie_wood", build_cathie_wood_agent),
    ("charlie_munger", build_charlie_munger_agent),
    ("bill_ackman", build_bill_ackman_agent),
    ("michael_burry", build_michael_burry_agent),
    ("mohnish_pabrai", build_mohnish_pabrai_agent),
    ("nassim_taleb", build_nassim_taleb_agent),
    ("peter_lynch", build_peter_lynch_agent),
    ("phil_fisher", build_phil_fisher_agent),
    ("rakesh_jhunjhunwala", build_rakesh_jhunjhunwala_agent),
    ("stanley_druckenmiller", build_stanley_druckenmiller_agent),
    ("aswath_damodaran", build_aswath_damodaran_agent),
]


class CouncilState(TypedDict, total=False):
    """Shared state across the council graph."""

    ticker: str
    query: str
    as_of_date: str
    persona_signals: Annotated[list[dict[str, Any]], operator.add]
    council_synthesis: dict[str, Any]


async def build_council_graph(
    config: RunnableConfig | None = None,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
    include_specialists: bool = False,
) -> CompiledStateGraph:
    """Build and compile the persona council graph.

    Args:
        config: Runnable config forwarded to every persona's async
            ``build_*_agent`` factory (used to materialise MCP tool
            handles at compile time).  Defaults to an empty config.
        checkpointer: Optional ``BaseCheckpointSaver`` for resumable runs.
        store: Optional ``BaseStore`` for the shared
            ``ToolResultCacheMiddleware`` cache (lets MCP tool calls
            dedupe across the 13 persona subgraphs).
        include_specialists: When True, also wires the deterministic
            ``technicals`` and ``sentiment`` specialist subgraphs into
            the council fan-in.

    Returns:
        Compiled state graph ready for ``ainvoke``.
    """
    effective_config: RunnableConfig = config or {}

    graph: StateGraph = StateGraph(CouncilState)

    # Compile every persona in parallel — speeds up cold starts.
    persona_agents = await asyncio.gather(
        *(builder(effective_config) for _, builder in PERSONA_BUILDERS)
    )
    for (slug, _builder), agent in zip(
        PERSONA_BUILDERS, persona_agents, strict=False
    ):
        graph.add_node(
            slug,
            agent,
            input_schema=agent.input_schema,
            retry_policy=_LLM_RETRY,
        )
        graph.add_edge(START, slug)
        graph.add_edge(slug, "council_judge")

    if include_specialists:
        # Import lazily so users who never enable specialists don't pay
        # the deterministic-fetch import cost.
        from ..specialists.sentiment_analysis import build_sentiment_analysis_agent
        from ..specialists.technical_analysis import build_technical_analysis_agent

        specialists: list[tuple[str, SpecialistBuilder]] = [
            ("technicals", build_technical_analysis_agent),
            ("sentiment", build_sentiment_analysis_agent),
        ]
        for slug, builder in specialists:
            agent = builder()
            graph.add_node(slug, agent, input_schema=agent.input_schema)
            graph.add_edge(START, slug)
            graph.add_edge(slug, "council_judge")

    graph.add_node("council_judge", council_judge_node)
    graph.add_edge("council_judge", END)

    return graph.compile(checkpointer=checkpointer, store=store)


_LAZY_GRAPH: CompiledStateGraph | None = None


def _bootstrap_default_graph() -> CompiledStateGraph:
    """Build the default council graph (sync wrapper for langgraph autodiscovery).

    Wires a default ``InMemoryStore`` so the per-run
    ``ToolResultCacheMiddleware`` has somewhere to land its entries.
    The persona subgraphs call ``get_tools(config, _MCP_TOOLS)`` which
    requires a reachable MCP server; this function is therefore lazy
    (built on first ``graph`` attribute access, not at import time) so
    test environments and other clients can import the module without
    needing the MCP stack up.
    """
    return asyncio.run(build_council_graph(store=InMemoryStore()))


def __getattr__(name: str) -> Any:
    """Lazy-load the module-level ``graph`` on first access."""
    if name == "graph":
        global _LAZY_GRAPH
        if _LAZY_GRAPH is None:
            _LAZY_GRAPH = _bootstrap_default_graph()
        return _LAZY_GRAPH
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
