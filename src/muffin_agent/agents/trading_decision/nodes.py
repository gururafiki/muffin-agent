"""LangGraph node wrappers around the trading_decision agents.

Each node:

1. Reads ``analysis_context`` and the relevant debate sub-state.
2. Builds a per-turn agent (the per-turn system prompt is rendered inside
   the agent factory from the current ``debate_history`` and
   ``opposing_last``).
3. Invokes the agent with a trivial trigger ``HumanMessage``.
4. Captures the LLM's response, prepends the speaker tag, and writes back
   the updated ``InvestmentDebateState``.

The judge node additionally extracts ``structured_response`` into
``InvestmentJudgeOutput``.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from .conditional_logic import BEAR_TAG, BULL_TAG
from .researchers import (
    create_bear_researcher_agent,
    create_bull_researcher_agent,
    create_investment_judge_agent,
)
from .schemas import AnalysisContext, InvestmentJudgeOutput, TraderOutput
from .state import InvestmentDebateState, TradingDecisionState
from .trader import create_trader_agent

logger = logging.getLogger(__name__)

# Trivial trigger message — the per-turn system prompt holds all real
# context, so the human message just unblocks the LLM to produce a reply.
_TRIGGER = "Make your argument now."
_JUDGE_TRIGGER = "Synthesise the debate now."
_TRADER_TRIGGER = "Produce the trade instruction now."


def _context(state: TradingDecisionState) -> AnalysisContext:
    """Extract and normalise the analysis context.

    Accepts either a Pydantic instance (typical when invoked
    programmatically) or a plain dict (typical when invoked via
    ``graph.ainvoke({"analysis_context": {...}})``).
    """
    raw: Any = state.get("analysis_context")
    if raw is None:
        raise ValueError("trading_decision graph requires `analysis_context` in state.")
    if isinstance(raw, AnalysisContext):
        return raw
    return AnalysisContext.model_validate(raw)


def _context_vars(ctx: AnalysisContext) -> dict[str, Any]:
    """Dump the context fields the prompt templates expect."""
    return ctx.model_dump(exclude={"ticker", "query"})


def _debate(state: TradingDecisionState) -> InvestmentDebateState:
    """Return the current investment-debate sub-state, defaulting empty."""
    raw: dict[str, Any] = dict(state.get("investment_debate") or {})
    return InvestmentDebateState(
        history=str(raw.get("history", "")),
        bull_history=str(raw.get("bull_history", "")),
        bear_history=str(raw.get("bear_history", "")),
        current_response=str(raw.get("current_response", "")),
        judge_decision=str(raw.get("judge_decision", "")),
        count=int(raw.get("count", 0)),
    )


def _extract_text(result: Any) -> str:
    """Pull the LLM's last textual response from a ReAct agent result."""
    if not isinstance(result, dict):
        return ""
    messages = result.get("messages") or []
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                # Anthropic / multimodal content blocks — concat text parts.
                parts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                return "\n".join(p for p in parts if p).strip()
    return ""


async def bull_researcher_node(
    state: TradingDecisionState, config: RunnableConfig
) -> dict[str, Any]:
    """Run one Bull Researcher turn and append it to the debate state."""
    ctx = _context(state)
    debate = _debate(state)

    agent = await create_bull_researcher_agent(
        config,
        ticker=ctx.ticker,
        query=ctx.query,
        context_vars=_context_vars(ctx),
        debate_history=debate["history"],
        opposing_last=debate["current_response"]
        if debate["current_response"].startswith(BEAR_TAG)
        else "",
    )

    try:
        result = await agent.ainvoke({"messages": [HumanMessage(_TRIGGER)]})
    except Exception:
        logger.exception("bull_researcher_node failed")
        argument = f"{BULL_TAG} (failed to produce an argument this turn.)"
    else:
        text = _extract_text(result) or "(empty argument)"
        argument = f"{BULL_TAG} {text}"

    updated: InvestmentDebateState = {
        "history": (debate["history"] + "\n\n" + argument).strip(),
        "bull_history": (debate["bull_history"] + "\n\n" + argument).strip(),
        "bear_history": debate["bear_history"],
        "current_response": argument,
        "judge_decision": debate["judge_decision"],
        "count": debate["count"] + 1,
    }
    return {"investment_debate": updated}


async def bear_researcher_node(
    state: TradingDecisionState, config: RunnableConfig
) -> dict[str, Any]:
    """Run one Bear Researcher turn and append it to the debate state."""
    ctx = _context(state)
    debate = _debate(state)

    agent = await create_bear_researcher_agent(
        config,
        ticker=ctx.ticker,
        query=ctx.query,
        context_vars=_context_vars(ctx),
        debate_history=debate["history"],
        opposing_last=debate["current_response"]
        if debate["current_response"].startswith(BULL_TAG)
        else "",
    )

    try:
        result = await agent.ainvoke({"messages": [HumanMessage(_TRIGGER)]})
    except Exception:
        logger.exception("bear_researcher_node failed")
        argument = f"{BEAR_TAG} (failed to produce an argument this turn.)"
    else:
        text = _extract_text(result) or "(empty argument)"
        argument = f"{BEAR_TAG} {text}"

    updated: InvestmentDebateState = {
        "history": (debate["history"] + "\n\n" + argument).strip(),
        "bull_history": debate["bull_history"],
        "bear_history": (debate["bear_history"] + "\n\n" + argument).strip(),
        "current_response": argument,
        "judge_decision": debate["judge_decision"],
        "count": debate["count"] + 1,
    }
    return {"investment_debate": updated}


async def investment_judge_node(
    state: TradingDecisionState, config: RunnableConfig
) -> dict[str, Any]:
    """Synthesise the completed debate into an ``InvestmentJudgeOutput``."""
    ctx = _context(state)
    debate = _debate(state)

    fallback = {
        "signal": "hold",
        "conviction": 0.0,
        "winning_side": "balanced",
        "error": "Investment Judge did not produce structured output.",
    }

    if not debate["history"].strip():
        return {
            "investment_judge": {
                **fallback,
                "error": "Investment Judge had no debate history to synthesise.",
            }
        }

    agent = await create_investment_judge_agent(
        config,
        ticker=ctx.ticker,
        query=ctx.query,
        context_vars=_context_vars(ctx),
        debate_history=debate["history"],
    )

    try:
        result = await agent.ainvoke({"messages": [HumanMessage(_JUDGE_TRIGGER)]})
    except Exception:
        logger.exception("investment_judge_node failed")
        return {"investment_judge": {**fallback, "error": "Judge agent raised."}}

    structured = result.get("structured_response") if isinstance(result, dict) else None
    if isinstance(structured, InvestmentJudgeOutput):
        payload = structured.model_dump()
    elif structured is not None:
        try:
            payload = InvestmentJudgeOutput.model_validate(structured).model_dump()
        except Exception:
            logger.exception("investment_judge structured response did not validate")
            return {"investment_judge": fallback}
    else:
        raw = _extract_text(result)
        return {"investment_judge": {**fallback, "raw_output": raw}}

    debate_update: InvestmentDebateState = {
        "history": debate["history"],
        "bull_history": debate["bull_history"],
        "bear_history": debate["bear_history"],
        "current_response": debate["current_response"],
        "judge_decision": payload.get("summary", ""),
        "count": debate["count"],
    }
    return {
        "investment_judge": payload,
        "investment_debate": debate_update,
    }


def _is_error_payload(payload: Any) -> bool:
    """Return True if a structured-output payload is an error fallback dict."""
    return isinstance(payload, dict) and "error" in payload


async def trader_node(
    state: TradingDecisionState, config: RunnableConfig
) -> dict[str, Any]:
    """Translate the Investment Judge's signal into an executable instruction.

    Reads ``state["investment_judge"]`` (set by ``investment_judge_node``).
    Skips the LLM call entirely when the judge produced an error fallback —
    a Trader call on a missing thesis is just compounding noise.
    """
    ctx = _context(state)
    judge_payload: Any = state.get("investment_judge")

    fallback: dict[str, Any] = {
        "action": "hold",
        "position_sizing": "0% of NAV (no actionable thesis)",
        "time_horizon": "n/a",
        "reasoning": "Trader skipped: Investment Judge did not produce a valid thesis.",
    }

    if judge_payload is None or _is_error_payload(judge_payload):
        return {"trader": {**fallback, "error": "Missing or errored judge output."}}

    agent = await create_trader_agent(
        config,
        ticker=ctx.ticker,
        query=ctx.query,
        context_vars=_context_vars(ctx),
        investment_judge=judge_payload,
    )

    try:
        result = await agent.ainvoke({"messages": [HumanMessage(_TRADER_TRIGGER)]})
    except Exception:
        logger.exception("trader_node failed")
        return {"trader": {**fallback, "error": "Trader agent raised."}}

    structured = result.get("structured_response") if isinstance(result, dict) else None
    if isinstance(structured, TraderOutput):
        payload = structured.model_dump()
    elif structured is not None:
        try:
            payload = TraderOutput.model_validate(structured).model_dump()
        except Exception:
            logger.exception("trader structured response did not validate")
            return {
                "trader": {**fallback, "error": "Trader response failed validation."}
            }
    else:
        raw = _extract_text(result)
        return {
            "trader": {
                **fallback,
                "error": "Trader did not produce structured output.",
                "raw_output": raw,
            }
        }

    return {"trader": payload}
