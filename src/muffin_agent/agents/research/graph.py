"""Top-level research pipeline graph.

::

    START
      │
      ▼
    prepare ──▶ classifier ──▶ lift_classification
                                    │
                    ┌─ skip_search ─┘
                    │                └─▶ researcher_{speed|balanced|quality}
                    │                          │
                    │                          ▼
                    │                        rerank
                    │                          │
                    └──────────▶ writer ◀──────┘
                                   │
                                   ▼
                              finalize ──▶ END

**Every LLM stage is a compiled agent added via ``add_node``**, so each owns its own
``checkpoint_ns``: its transcript, tool calls and nested ``task`` sub-agents are
readable per-namespace with no capture plumbing. The pure-Python nodes (``prepare``,
``lift_classification``, ``rerank``, ``finalize``) hold the deterministic logic that
used to be wrapped around the agent calls inside function nodes.

Three researcher variants exist because ``with_model_call_limit`` bakes the
mode-driven iteration budget in at build time; ``_route_researcher`` picks one on
``state["mode"]``, which also makes the chosen depth visible in the node name.

Errors propagate (``RetryPolicy`` per agent node + the model-fallback chain) — a
failed stage fails the run, matching ``criteria_analysis`` / ``trading_decision``.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import RetryPolicy

from muffin_agent.utils.observability import instrument_graph

from .nodes import (
    build_researchers_by_mode,
    create_classifier_agent,
    create_writer_agent,
    finalize_output_node,
    lift_classification_node,
    prepare_node,
    rerank_node,
)
from .state import (
    RESEARCH_MODES,
    ClassifierInput,
    ResearcherInput,
    ResearchState,
    WriterInput,
)

_AGENT_RETRY = RetryPolicy(max_attempts=2)


def _researcher_node_name(mode: str) -> str:
    return f"researcher_{mode}"


def _route_after_classification(state: ResearchState) -> str:
    """Skip research entirely when the classifier judged no lookup is needed."""
    if state.get("skip_search"):
        return "writer"
    return _route_researcher(state)


def _route_researcher(state: ResearchState) -> str:
    """Pick the researcher variant whose baked-in iteration budget matches the mode."""
    mode = state.get("mode") or "balanced"
    if mode not in RESEARCH_MODES:
        mode = "balanced"
    return _researcher_node_name(mode)


async def build_research_graph(
    config: RunnableConfig | None = None,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
    extra_tools: Sequence[BaseTool] | None = None,
) -> CompiledStateGraph:
    """Build the research pipeline.

    Async because every stage's compiled agent is built at graph-construction time —
    amortising agent construction out of the per-request hot path, exactly like
    ``build_criteria_analysis_graph`` / ``build_trading_decision_graph``.

    Args:
        config: Runtime ``RunnableConfig``; supplies model + MCP configuration to the
            agent factories.
        checkpointer: Optional ``BaseCheckpointSaver`` for thread-level history. CLI
            passes a ``SqliteSaver``; LangGraph Platform injects its own (pass
            ``None`` for autodiscovery).
        store: Optional ``BaseStore`` for cross-agent tool-result caching and
            ``/memories/`` access.
        extra_tools: Additional tools the researcher should register. Use this to plug
            in academic / news / finance / internal search tools.

    Note:
        ``extra_sources`` is no longer a build-time argument — the classifier reads
        the permitted source list from ``state["allowed_sources"]`` at runtime, so
        callers pass it per run instead of per graph.
    """
    cfg: RunnableConfig = config or {}

    classifier_agent = await create_classifier_agent(cfg)
    researchers = await build_researchers_by_mode(
        cfg, store=store, extra_tools=extra_tools
    )
    writer_agent = await create_writer_agent(cfg)

    graph: StateGraph = StateGraph(ResearchState)

    graph.add_node("prepare", prepare_node)
    graph.add_node(
        "classifier",
        classifier_agent,
        input_schema=ClassifierInput,
        retry_policy=_AGENT_RETRY,
    )
    graph.add_node("lift_classification", lift_classification_node)
    for mode in RESEARCH_MODES:
        graph.add_node(
            _researcher_node_name(mode),
            researchers[mode],
            input_schema=ResearcherInput,
            retry_policy=_AGENT_RETRY,
        )
    graph.add_node("rerank", rerank_node)
    graph.add_node(
        "writer", writer_agent, input_schema=WriterInput, retry_policy=_AGENT_RETRY
    )
    graph.add_node("finalize", finalize_output_node)

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "classifier")
    graph.add_edge("classifier", "lift_classification")
    graph.add_conditional_edges(
        "lift_classification",
        _route_after_classification,
        ["writer", *(_researcher_node_name(m) for m in RESEARCH_MODES)],
    )
    for mode in RESEARCH_MODES:
        graph.add_edge(_researcher_node_name(mode), "rerank")
    graph.add_edge("rerank", "writer")
    graph.add_edge("writer", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer, store=store)


async def make_graph(config: RunnableConfig | None = None) -> CompiledStateGraph:
    """LangGraph Platform graph factory (config-only); registered in ``langgraph.json``.

    The Platform's factory protocol only accepts a ``RunnableConfig``, and injects its
    own managed checkpointer + store into the returned graph — mirroring
    ``criteria_analysis`` / ``council`` / ``trading_decision``.
    """
    return instrument_graph(await build_research_graph(config))
