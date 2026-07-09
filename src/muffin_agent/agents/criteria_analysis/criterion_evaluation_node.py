"""Stage 4b: per-criterion evaluation worker (Send fan-out target).

The worker is a small compiled subgraph (the council-persona shape):

    evaluate   → the ``criterion_evaluation`` deep agent as a graph node
                 (task context rendered into the system prompt; structured
                 response unpacked into the ``evaluation`` channel)
    package    → pure node that augments the evaluation with the criterion's
                 ``weight`` + ``source``, reconciles the claimed
                 ``data_sources`` against the captured ``tool_runs``
                 (anti-hallucination), emits a ``criterion_evaluated`` custom
                 stream event, and appends the evaluation to the parent
                 ``criterion_evaluations`` accumulator (``operator.add``).

One ``Send`` per merged criterion targets this subgraph.

Streaming note: ALL parallel Send workers complete inside ONE parent
superstep, so the parent's ``values``/``updates`` events show the full
scorecard only at the barrier. The ``criterion_evaluated`` custom event is
the only live per-criterion signal — clients must request
``stream_mode=["custom", ...]`` AND ``subgraphs=True`` (custom events from
subgraph nodes are namespaced and are NOT delivered without it).
"""

import logging
import operator
from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import RetryPolicy
from typing_extensions import TypedDict

from ..criterion_evaluation import create_criterion_evaluation_agent
from .state import CriterionEvaluationSendPayload

logger = logging.getLogger(__name__)

_AGENT_RETRY = RetryPolicy(max_attempts=2)

_NO_DATA_CONFIDENCE_CAP = 0.3
_NO_DATA_LIMITATION = (
    "No live data was collected for this criterion; the evaluation reflects "
    "model prior knowledge only."
)


class _CriterionWorkerState(TypedDict, total=False):
    """State of the per-criterion worker subgraph.

    ``evaluation`` is written by the agent node; ``criterion_evaluations``
    is the single-element list the ``package`` node emits, merged into the
    parent accumulator via ``operator.add``.
    """

    ticker: str
    query: str
    criterion: dict[str, Any]
    classification: dict[str, Any]
    evaluation: dict[str, Any]
    criterion_evaluations: Annotated[list[dict[str, Any]], operator.add]
    # Written by the evaluate agent's AgentCaptureMiddleware (its own +
    # nested subagents' records); the package node moves it onto the
    # evaluation dict so it rides into the parent per-criterion, not top-level.
    tool_runs: Annotated[list[dict[str, Any]], operator.add]


class _CriterionWorkerOutput(TypedDict):
    """Output schema of the worker subgraph.

    Restricts what propagates to the parent to ONLY the accumulator — the
    Send inputs (``ticker`` / ``query`` / ``criterion`` / ``classification``)
    must NOT flow back, or N parallel workers would each write the parent's
    single-value ``ticker`` channel and raise ``InvalidUpdateError``.
    """

    criterion_evaluations: Annotated[list[dict[str, Any]], operator.add]


def _reconcile_data_sources(
    evaluation: dict[str, Any], tool_runs: list[dict[str, Any]]
) -> dict[str, Any]:
    """Reconcile the LLM-claimed ``data_sources`` with captured tool runs.

    Deterministic anti-hallucination pass (mutates and returns
    ``evaluation``). Observed failure mode in prod: free/weak models
    single-shot the structured output without calling a single tool and
    fabricate plausible ``data_sources``/evidence.

    * **No tool runs** — nothing was retrieved, so every claimed source is
      fabricated: clear ``data_sources``, set ``data_collected=False``, cap
      ``confidence`` at ``_NO_DATA_CONFIDENCE_CAP`` and record a limitation.
    * **Tool runs present** — set ``data_collected=True`` and keep only
      sources whose ``subagent`` matches something that actually executed
      (record ``agent`` labels, tool names, or ``task`` args previews);
      dropped names are listed in a limitation line.
    """
    limitations = [str(item) for item in (evaluation.get("limitations") or [])]

    if not tool_runs:
        evaluation["data_collected"] = False
        evaluation["data_sources"] = []
        confidence = evaluation.get("confidence")
        if isinstance(confidence, (int, float)):
            evaluation["confidence"] = min(float(confidence), _NO_DATA_CONFIDENCE_CAP)
        if _NO_DATA_LIMITATION not in limitations:
            limitations.append(_NO_DATA_LIMITATION)
        evaluation["limitations"] = limitations
        return evaluation

    evaluation["data_collected"] = True
    haystack_parts: list[str] = []
    for record in tool_runs:
        for key in ("agent", "tool", "args_preview"):
            value = record.get(key)
            if isinstance(value, str) and value:
                haystack_parts.append(value.lower())
    haystack = " ".join(haystack_parts)

    kept: list[Any] = []
    dropped: list[str] = []
    for source in evaluation.get("data_sources") or []:
        subagent = source.get("subagent") if isinstance(source, dict) else None
        if isinstance(subagent, str) and subagent.strip():
            if subagent.strip().lower() in haystack:
                kept.append(source)
            else:
                dropped.append(subagent.strip())
        else:
            # Nameless entries can't be verified either way — keep them.
            kept.append(source)
    if dropped:
        evaluation["data_sources"] = kept
        limitations.append(
            "Dropped uncorroborated data_sources entries (no matching tool "
            "execution): " + ", ".join(sorted(set(dropped)))
        )
        evaluation["limitations"] = limitations
    return evaluation


def _emit_criterion_evaluated(evaluation: dict[str, Any]) -> None:
    """Emit the per-criterion ``custom`` stream event (see module docstring).

    No-ops outside a runnable context (direct unit-test calls); inside a run
    whose client didn't request ``stream_mode="custom"`` the writer itself is
    a no-op.
    """
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer({"type": "criterion_evaluated", "evaluation": evaluation})


def package_evaluation_node(state: _CriterionWorkerState) -> dict[str, Any]:
    """Augment the agent's evaluation and append it to the accumulator.

    Carries the criterion's ``weight`` + ``source`` onto the evaluation
    dict so the synthesis stage can build the weighted breakdown without
    rejoining against ``merged_criteria``, reconciles ``data_sources``
    against the captured ``tool_runs`` and emits the ``criterion_evaluated``
    custom stream event for live per-criterion progress.
    """
    criterion = state.get("criterion") or {}
    evaluation = dict(state.get("evaluation") or {})
    evaluation["criterion_name"] = criterion.get(
        "name", evaluation.get("criterion_name", "<unknown criterion>")
    )
    evaluation["weight"] = criterion.get("weight", 0.0)
    evaluation["source"] = criterion.get("source", "skill")
    # Attach this criterion's tool-execution records (AgentCaptureMiddleware;
    # capture is unconditional). Kept per-criterion, not top-level.
    tool_runs = state.get("tool_runs") or []
    if tool_runs:
        evaluation["tool_runs"] = tool_runs
    _reconcile_data_sources(evaluation, tool_runs)
    _emit_criterion_evaluated(evaluation)
    return {"criterion_evaluations": [evaluation]}


async def build_criterion_evaluation_worker(
    config: RunnableConfig,
    *,
    store: BaseStore | None = None,
) -> CompiledStateGraph:
    """Build and compile the per-criterion worker subgraph."""
    agent = await create_criterion_evaluation_agent(config)

    worker: StateGraph = StateGraph(
        _CriterionWorkerState,
        input_schema=CriterionEvaluationSendPayload,
        output_schema=_CriterionWorkerOutput,
    )
    worker.add_node(
        "evaluate",
        agent,
        input_schema=CriterionEvaluationSendPayload,
        retry_policy=_AGENT_RETRY,
    )
    worker.add_node("package", package_evaluation_node)
    worker.add_edge(START, "evaluate")
    worker.add_edge("evaluate", "package")
    worker.add_edge("package", END)
    return worker.compile(store=store)
