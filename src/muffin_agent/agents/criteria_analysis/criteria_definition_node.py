"""Stage 2: criteria definition wrapper node.

Thin LangGraph node that delegates to the existing
``create_criteria_definition_agent`` deep agent.  Reads classification
fields from flat state keys (set by Stage 1) so
``SkillFilterMiddleware[TickerClassification]`` can filter
``/skills/valuation/`` correctly.
"""

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from typing_extensions import TypedDict

from ..criteria_definition import create_criteria_definition_agent
from ._node_helpers import invoke_structured_agent


class CriteriaDefinitionInputState(TypedDict, total=False):
    """Fields read by ``criteria_definition_node`` from the outer state."""

    ticker: str
    query: str
    sector: str
    sub_sector: str
    market: str
    stock_type: str


async def criteria_definition_node(
    state: CriteriaDefinitionInputState,
    config: RunnableConfig,
    *,
    store: BaseStore | None = None,
) -> dict[str, Any]:
    """Stage 2: produce sector-specific valuation criteria from skills.

    Runs in parallel with Stage 3 (``valuation_methodology_node``).
    Output written to ``criteria_definition`` state key.
    """
    return await invoke_structured_agent(
        state=dict(state),
        config=config,
        agent_factory=create_criteria_definition_agent,
        input_state_type=CriteriaDefinitionInputState,
        state_key="criteria_definition",
        error_fallback={"criteria": []},
        store=store,
    )
