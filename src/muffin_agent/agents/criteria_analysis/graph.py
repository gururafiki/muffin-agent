"""Top-level orchestrator graph for criteria-driven analysis.

Every stage that runs an LLM is a compiled agent added DIRECTLY to the
graph as a node (the trading-analyst / council-persona pattern) — task
context flows in via an explicit ``input_schema`` and is rendered into the
agent's system prompt; the structured response unpacks into the matching
state channel via the single-field wrapper output models.

Pipeline::

    START
      │  (conditional: flat classification keys pre-supplied?)
      ├─────────────────────────────► lift_classification  (short-circuit)
      ▼
    ticker_classification ──► lift_classification   ← Stage 1
                                    │
                    ┌───────────────┴────────────────┐
                    ▼                                 ▼
             criteria_definition            valuation_methodology  ← Stages 2 & 3
                    └───────────────┬────────────────┘
                                    ▼
                              merge_criteria    ← Stage 4a (deterministic)
                                    │
                    (Send fan-out, one per merged criterion)
                                    ▼
                       criterion_evaluation × N   ← Stage 4b (worker subgraph)
                                    │
                          (operator.add fan-in)
                                    ▼
                                synthesis    ← Stage 5
                                    │
                                    ▼
                                   END

Nodes propagate failures — each agent node carries
``RetryPolicy(max_attempts=2)`` and there are no try/except fallback dicts.
A failed stage fails the run (thread status ``error``).
"""

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import RetryPolicy, Send

from muffin_agent.utils.observability import instrument_graph

from ..criteria_definition import create_criteria_definition_agent
from .criterion_evaluation_node import build_criterion_evaluation_worker
from .merge_criteria import merge_criteria_node
from .state import (
    CriteriaAnalysisState,
    CriteriaDefinitionInput,
    SynthesisInput,
    TickerClassificationInput,
    ValuationMethodologyInput,
)
from .synthesis import create_synthesis_agent
from .ticker_classification import (
    create_ticker_classification_agent,
    lift_classification_node,
    route_classification_entry,
)
from .valuation_methodology import create_valuation_methodology_agent

_AGENT_RETRY = RetryPolicy(max_attempts=2)


def _fan_out_criteria(state: CriteriaAnalysisState) -> list[Send]:
    """Emit one ``Send`` per merged criterion to the evaluation worker.

    Forwards ticker, query, the criterion, and the full classification
    payload so each worker has the context it needs without re-reading
    state.
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
            },
        )
        for criterion in merged
    ]


async def build_criteria_analysis_graph(
    config: RunnableConfig,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> CompiledStateGraph:
    """Build and compile the criteria-driven analysis orchestrator graph.

    Async because each stage's compiled agent (and its subagents) is built
    at graph-construction time — amortising agent construction out of the
    per-request hot path, exactly like ``build_trading_decision_graph``.
    """
    ticker_classification_agent = await create_ticker_classification_agent(
        config, store=store
    )
    criteria_definition_agent = await create_criteria_definition_agent(
        config, store=store
    )
    valuation_methodology_agent = await create_valuation_methodology_agent(
        config, store=store
    )
    criterion_evaluation_worker = await build_criterion_evaluation_worker(
        config, store=store
    )
    synthesis_agent = await create_synthesis_agent(config, store=store)

    graph: StateGraph = StateGraph(CriteriaAnalysisState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    graph.add_node(
        "ticker_classification",
        ticker_classification_agent,
        input_schema=TickerClassificationInput,
        retry_policy=_AGENT_RETRY,
    )
    graph.add_node("lift_classification", lift_classification_node)
    graph.add_node(
        "criteria_definition",
        criteria_definition_agent,
        input_schema=CriteriaDefinitionInput,
        retry_policy=_AGENT_RETRY,
    )
    graph.add_node(
        "valuation_methodology",
        valuation_methodology_agent,
        input_schema=ValuationMethodologyInput,
        retry_policy=_AGENT_RETRY,
    )
    graph.add_node("merge_criteria", merge_criteria_node)
    graph.add_node("criterion_evaluation", criterion_evaluation_worker)
    graph.add_node(
        "synthesis",
        synthesis_agent,
        input_schema=SynthesisInput,
        retry_policy=_AGENT_RETRY,
    )

    # ── Edges ─────────────────────────────────────────────────────────────────
    graph.add_conditional_edges(
        START,
        route_classification_entry,
        ["ticker_classification", "lift_classification"],
    )
    graph.add_edge("ticker_classification", "lift_classification")
    graph.add_edge("lift_classification", "criteria_definition")
    graph.add_edge("lift_classification", "valuation_methodology")
    graph.add_edge("criteria_definition", "merge_criteria")
    graph.add_edge("valuation_methodology", "merge_criteria")
    graph.add_conditional_edges(
        "merge_criteria", _fan_out_criteria, ["criterion_evaluation"]
    )
    graph.add_edge("criterion_evaluation", "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile(checkpointer=checkpointer, store=store)


async def make_graph(config: RunnableConfig | None = None) -> CompiledStateGraph:
    """LangGraph Platform graph factory (config-only); registered in ``langgraph.json``.

    The Platform's factory protocol only accepts a ``RunnableConfig`` /
    ``ServerRuntime`` parameter (a ``BaseStore`` parameter is rejected) and
    injects its managed checkpointer/store into the returned graph. Mirrors
    ``trading_decision.graph.make_graph`` and
    ``personas_council.council_graph.make_graph``.
    """
    return instrument_graph(await build_criteria_analysis_graph(config or {}))
