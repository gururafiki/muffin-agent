"""Stage 4b: per-criterion evaluation worker (Send fan-out target).

The worker is a small compiled subgraph (the council-persona shape):

    evaluate   → the ``criterion_evaluation`` deep agent as a graph node
                 (task context rendered into the system prompt; structured
                 response unpacked into the ``evaluation`` channel)
    package    → pure node that augments the evaluation with the criterion's
                 ``weight`` + ``source`` and appends it to the parent
                 ``criterion_evaluations`` accumulator (``operator.add``).

One ``Send`` per merged criterion targets this subgraph.
"""

import logging
import operator
from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import RetryPolicy
from typing_extensions import TypedDict

from ..criterion_evaluation import create_criterion_evaluation_agent
from .state import CriterionEvaluationSendPayload

logger = logging.getLogger(__name__)

_AGENT_RETRY = RetryPolicy(max_attempts=2)


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


class _CriterionWorkerOutput(TypedDict):
    """Output schema of the worker subgraph.

    Restricts what propagates to the parent to ONLY the accumulator — the
    Send inputs (``ticker`` / ``query`` / ``criterion`` / ``classification``)
    must NOT flow back, or N parallel workers would each write the parent's
    single-value ``ticker`` channel and raise ``InvalidUpdateError``.
    """

    criterion_evaluations: Annotated[list[dict[str, Any]], operator.add]


def package_evaluation_node(state: _CriterionWorkerState) -> dict[str, Any]:
    """Augment the agent's evaluation and append it to the accumulator.

    Carries the criterion's ``weight`` + ``source`` onto the evaluation
    dict so the synthesis stage can build the weighted breakdown without
    rejoining against ``merged_criteria``.
    """
    criterion = state.get("criterion") or {}
    evaluation = dict(state.get("evaluation") or {})
    evaluation["criterion_name"] = criterion.get(
        "name", evaluation.get("criterion_name", "<unknown criterion>")
    )
    evaluation["weight"] = criterion.get("weight", 0.0)
    evaluation["source"] = criterion.get("source", "skill")
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
