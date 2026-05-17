"""Reflector LLM agent — produces 2–4 sentence reflections on past decisions.

Single-purpose ReAct agent (no tools, no subagents, no structured output —
just terse prose). One call per resolved decision; the reflection is
stored verbatim in the decision log and re-read by future Portfolio
Manager prompts.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from ....model_config import ModelConfiguration
from ....prompts import render_template
from ....utils.agent_builder import MuffinAgentBuilder

logger = logging.getLogger(__name__)

_REFLECTOR_TRIGGER = "Write your reflection now."


async def create_reflector_agent(
    config: RunnableConfig,
    *,
    ticker: str,
    decision_date: str,
    decision: dict[str, Any],
    outcome: dict[str, Any],
):
    """Build a per-decision Reflector agent.

    The full system prompt is rendered at build time from the specific
    decision + outcome pair so the agent only needs a trivial trigger
    message at invocation. Uses the ``reasoner`` role with no structured
    output — the entire response is the reflection prose.
    """
    configuration = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = configuration.get_llm_for_role("reasoner")
    summariser = configuration.get_summariser()

    prompt = render_template(
        "trading_decision/reflection/reflector.jinja",
        ticker=ticker,
        decision_date=decision_date,
        decision=decision,
        outcome=outcome,
    )

    builder = (
        MuffinAgentBuilder(primary, name="trading_reflector")
        .with_system_prompt(prompt)
        .with_fallback_models(*fallbacks)
    )
    if summariser is not None:
        builder = builder.with_tool_knowledge(summariser)
    return builder.build_react_agent()


def _extract_text(result: Any) -> str:
    """Pull the last AI textual response from the agent result."""
    if not isinstance(result, dict):
        return ""
    messages = result.get("messages") or []
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                return "\n".join(p for p in parts if p).strip()
    return ""


async def generate_reflection(
    *,
    config: RunnableConfig,
    ticker: str,
    decision_date: str,
    decision: dict[str, Any],
    outcome: dict[str, Any],
) -> str:
    """End-to-end helper: build the agent, invoke, return prose.

    Returns a generic deterministic fallback string on any LLM-side failure
    — reflections are best-effort; a missing reflection should not block
    the trading-decision pipeline.
    """
    try:
        agent = await create_reflector_agent(
            config,
            ticker=ticker,
            decision_date=decision_date,
            decision=decision,
            outcome=outcome,
        )
        result = await agent.ainvoke({"messages": [HumanMessage(_REFLECTOR_TRIGGER)]})
    except Exception:
        logger.debug(
            "generate_reflection failed for %s/%s", ticker, decision_date, exc_info=True
        )
        return _fallback_reflection(decision, outcome)
    text = _extract_text(result)
    return text or _fallback_reflection(decision, outcome)


def _fallback_reflection(decision: dict[str, Any], outcome: dict[str, Any]) -> str:
    """Deterministic short reflection used when the LLM call fails."""
    rating = decision.get("rating", "?")
    raw = outcome.get("raw_return_pct", 0.0)
    alpha = outcome.get("alpha_return_pct", 0.0)
    direction = (
        "positive raw return"
        if raw > 0
        else "negative raw return"
        if raw < 0
        else "flat raw return"
    )
    return (
        f"Decision rating was {rating}; outcome was {direction} of {raw:+.2f}% "
        f"with alpha {alpha:+.2f}%. (Auto-generated fallback — Reflector LLM "
        "unavailable; treat as low-signal.)"
    )
