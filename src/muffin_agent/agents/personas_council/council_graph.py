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

:func:`build_council_graph` is registered in :data:`langgraph.json` as the
``"council"`` graph **factory** — LangGraph Platform calls it (async) and
injects the managed checkpointer / store.  Each persona / specialist is a
compiled subgraph added directly via ``add_node`` and owns its own MCP data
fetching; callers can also invoke the factory directly with their own
checkpointer / store.
"""

from __future__ import annotations

import asyncio
import operator
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import RetryPolicy
from typing_extensions import TypedDict

from .judge import council_judge_node
from .personas import (
    build_aswath_damodaran_agent,
    build_ben_graham_agent,
    build_bill_ackman_agent,
    build_cathie_wood_agent,
    build_charlie_munger_agent,
    build_michael_burry_agent,
    build_mohnish_pabrai_agent,
    build_nassim_taleb_agent,
    build_peter_lynch_agent,
    build_phil_fisher_agent,
    build_rakesh_jhunjhunwala_agent,
    build_stanley_druckenmiller_agent,
    build_warren_buffett_agent,
)
from .schemas import PersonaInput
from .specialists import (
    build_fundamentals_analysis_agent,
    build_growth_analysis_agent,
    build_news_sentiment_analysis_agent,
    build_sentiment_analysis_agent,
    build_technical_analysis_agent,
    build_valuation_analysis_agent,
)

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
    store: BaseStore | None = None,
) -> CompiledStateGraph:
    """Build and compile the persona council graph.

    Registered in ``langgraph.json`` as the config-only graph factory. LangGraph
    Platform caps factory parameters at two and injects the managed checkpointer /
    store, so persistence is left to the Platform (or the caller's ``store``) and the
    specialist toggle is read from ``config`` rather than passed as an argument.

    Args:
        config: Runnable config forwarded to every persona's async ``build_*_agent``
            factory (used to materialise MCP tool handles at compile time). Also
            carries ``configurable.include_specialists`` (bool) to additionally wire
            the six specialist subgraphs (``technicals``, ``sentiment``,
            ``fundamentals``, ``growth``, ``valuation``, ``news_sentiment``) into the
            fan-in. Defaults to an empty config.
        store: Optional ``BaseStore`` for the shared ``ToolResultCacheMiddleware``
            cache (dedupes MCP tool calls across the 13 persona subgraphs). The
            Platform injects its managed store; CLI / tests may pass one explicitly.

    Returns:
        Compiled state graph ready for ``ainvoke``.
    """
    effective_config: RunnableConfig = config or {}
    include_specialists = bool(
        effective_config.get("configurable", {}).get("include_specialists", False)
    )

    graph: StateGraph = StateGraph(CouncilState)

    # Compile every persona in parallel — speeds up cold starts.
    persona_agents = await asyncio.gather(
        *(builder(effective_config) for _, builder in PERSONA_BUILDERS)
    )
    for (slug, _builder), agent in zip(PERSONA_BUILDERS, persona_agents, strict=False):
        graph.add_node(
            slug,
            agent,
            input_schema=PersonaInput,
            retry_policy=_LLM_RETRY,
        )
        graph.add_edge(START, slug)
        graph.add_edge(slug, "council_judge")

    if include_specialists:
        # Fully-deterministic specialists: sync, no-arg builders.
        sync_specialists: list[tuple[str, SpecialistBuilder]] = [
            ("technicals", build_technical_analysis_agent),
            ("sentiment", build_sentiment_analysis_agent),
        ]
        for slug, sync_builder in sync_specialists:
            agent = sync_builder()
            graph.add_node(
                slug, agent, input_schema=PersonaInput, retry_policy=_LLM_RETRY
            )
            graph.add_edge(START, slug)
            graph.add_edge(slug, "council_judge")

        # Metric-heavy specialists: persona-style async builders taking config
        # (ReAct collect_data + deterministic scoring).
        async_specialists: list[tuple[str, PersonaBuilder]] = [
            ("fundamentals", build_fundamentals_analysis_agent),
            ("growth", build_growth_analysis_agent),
            ("valuation", build_valuation_analysis_agent),
            ("news_sentiment", build_news_sentiment_analysis_agent),
        ]
        async_agents = await asyncio.gather(
            *(builder(effective_config) for _, builder in async_specialists)
        )
        for (slug, _builder), agent in zip(
            async_specialists, async_agents, strict=False
        ):
            graph.add_node(
                slug, agent, input_schema=PersonaInput, retry_policy=_LLM_RETRY
            )
            graph.add_edge(START, slug)
            graph.add_edge(slug, "council_judge")

    graph.add_node("council_judge", council_judge_node)
    graph.add_edge("council_judge", END)

    # checkpointer left to the Platform (or unused for one-shot council runs); store is
    # passed by CLI/tests or injected by the Platform.
    return graph.compile(store=store)


async def make_graph(config: RunnableConfig | None = None) -> CompiledStateGraph:
    """LangGraph Platform graph factory (config-only); registered in ``langgraph.json``.

    The Platform's factory protocol only accepts parameters typed ``RunnableConfig`` /
    ``ServerRuntime`` (a ``BaseStore`` parameter is rejected) and injects its managed
    checkpointer/store into the returned graph. So the deployed entrypoint is this thin
    config-only factory; :func:`build_council_graph` keeps its ``store`` parameter for
    CLI / programmatic callers (the within-run MCP cache shared across the personas).
    ``include_specialists`` is read from ``config["configurable"]``.
    """
    return await build_council_graph(config)
