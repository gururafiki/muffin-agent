"""Stage 4b: per-criterion evaluation node (Send fan-out target).

Receives one criterion + ticker context per ``Send`` from
``merge_criteria_node`` and runs the existing
``create_criterion_evaluation_agent`` against it.  Results flow into
``criterion_evaluations`` (operator.add accumulator).
"""

import json
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from ..criterion_evaluation import create_criterion_evaluation_agent

logger = logging.getLogger(__name__)


async def criterion_evaluation_node(
    state: dict[str, Any],
    config: RunnableConfig,
    *,
    store: BaseStore | None = None,
) -> dict[str, Any]:
    """Evaluate one criterion against the ticker.

    Reads ``ticker``, ``query``, ``criterion`` (single), and
    ``classification`` from the per-Send sub-state.  Returns
    ``{"criterion_evaluations": [evaluation_dict]}`` where the dict is
    ``CriterionEvaluationOutput.model_dump()`` augmented with the input
    criterion's source tag (``skill`` or ``web``) and weight (so the
    synthesis stage can build the weighted breakdown without rejoining
    against ``merged_criteria``).
    """
    criterion = state.get("criterion") or {}
    criterion_name = criterion.get("name", "<unknown criterion>")
    weight = criterion.get("weight", 0.0)
    source = criterion.get("source", "skill")

    try:
        agent = await create_criterion_evaluation_agent(config)
        context = {
            "ticker": state.get("ticker"),
            "query": state.get("query"),
            "criterion": criterion,
            "classification": state.get("classification") or {},
        }
        result = await agent.ainvoke({"input": json.dumps(context)})
        structured = (
            result.get("structured_response") if isinstance(result, dict) else None
        )
        if structured is None:
            evaluation = {
                "criterion_name": criterion_name,
                "score": 0.0,
                "confidence": 0.0,
                "signal": "neutral",
                "sub_criteria": [],
                "evidence_summary": [],
                "reasoning": "Agent did not produce structured output.",
                "counterargument": "",
                "limitations": ["No structured output."],
                "data_sources": [],
                "weight": weight,
                "source": source,
                "error": "Agent did not produce structured output",
            }
        else:
            evaluation = structured.model_dump()
            evaluation["criterion_name"] = criterion_name
            evaluation["weight"] = weight
            evaluation["source"] = source
    except Exception:
        logger.exception("criterion_evaluation_node failed for %r", criterion_name)
        evaluation = {
            "criterion_name": criterion_name,
            "score": 0.0,
            "confidence": 0.0,
            "signal": "neutral",
            "sub_criteria": [],
            "evidence_summary": [],
            "reasoning": "",
            "counterargument": "",
            "limitations": [],
            "data_sources": [],
            "weight": weight,
            "source": source,
            "error": "Agent raised an exception",
        }

    return {"criterion_evaluations": [evaluation]}
