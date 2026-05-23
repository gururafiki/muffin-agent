"""Reflector — single LLM call that produces a 2-4 sentence reflection.

No agent factory, no try/except. Caller (``reflector_resolve_node``) decides
whether to swallow / propagate failures. LangChain ``with_retry`` absorbs
transient issues; LangGraph node-level ``RetryPolicy`` adds a second layer.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ....model_config import ModelConfiguration
from ....prompts import render_template


async def reflect_on_decision(
    config: RunnableConfig,
    *,
    ticker: str,
    decision_date: str,
    decision: dict[str, Any],
    outcome: dict[str, Any],
) -> str:
    """Generate a 2-4 sentence reflection for one (decision, outcome) pair."""
    cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = cfg.get_llm_for_role("reasoner")
    llm = (primary.with_fallbacks(fallbacks) if fallbacks else primary).with_retry(
        stop_after_attempt=3, wait_exponential_jitter=True
    )

    prompt = render_template(
        "trading_decision/reflection/reflector.jinja",
        ticker=ticker,
        decision_date=decision_date,
        decision=decision,
        outcome=outcome,
    )

    response = await llm.ainvoke(
        [
            SystemMessage(prompt),
            HumanMessage("Write your reflection now."),
        ]
    )
    return str(response.content).strip()
