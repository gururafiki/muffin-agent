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
from langgraph.store.base import BaseStore

from ...utils.memory_config import MemoryConfiguration
from .conditional_logic import (
    AGGRESSIVE_TAG,
    BEAR_TAG,
    BULL_TAG,
    CONSERVATIVE_TAG,
    NEUTRAL_TAG,
)
from .portfolio_manager import create_portfolio_manager_agent
from .reflection import (
    OutcomesFetcher,
    ReflectionMemory,
    fetch_outcomes_openbb,
    generate_reflection,
    render_reflections_block,
)
from .researchers import (
    create_bear_researcher_agent,
    create_bull_researcher_agent,
    create_investment_judge_agent,
)
from .risk_debate import (
    create_aggressive_debator_agent,
    create_conservative_debator_agent,
    create_neutral_debator_agent,
)
from .schemas import (
    AnalysisContext,
    InvestmentJudgeOutput,
    PortfolioDecisionOutput,
    TraderOutput,
)
from .state import InvestmentDebateState, RiskDebateState, TradingDecisionState
from .trader import create_trader_agent

logger = logging.getLogger(__name__)

# Trivial trigger message — the per-turn system prompt holds all real
# context, so the human message just unblocks the LLM to produce a reply.
_TRIGGER = "Make your argument now."
_JUDGE_TRIGGER = "Synthesise the debate now."
_TRADER_TRIGGER = "Produce the trade instruction now."
_PORTFOLIO_MANAGER_TRIGGER = "Produce the portfolio decision now."


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


def _risk_debate(state: TradingDecisionState) -> RiskDebateState:
    """Return the current risk-debate sub-state, defaulting empty."""
    raw: dict[str, Any] = dict(state.get("risk_debate") or {})
    return RiskDebateState(
        history=str(raw.get("history", "")),
        aggressive_history=str(raw.get("aggressive_history", "")),
        conservative_history=str(raw.get("conservative_history", "")),
        neutral_history=str(raw.get("neutral_history", "")),
        current_aggressive_response=str(raw.get("current_aggressive_response", "")),
        current_conservative_response=str(raw.get("current_conservative_response", "")),
        current_neutral_response=str(raw.get("current_neutral_response", "")),
        latest_speaker=raw.get("latest_speaker", ""),
        judge_decision=str(raw.get("judge_decision", "")),
        count=int(raw.get("count", 0)),
    )


def _last_opposing_argument(debate: RiskDebateState, exclude_role: str) -> str:
    """Return the most recent opposing argument from a risk debater's perspective.

    ``exclude_role`` is the current speaker's role ("aggressive" /
    "conservative" / "neutral"); we return whichever of the other two
    spoke most recently based on ``latest_speaker``.
    """
    latest = str(debate.get("latest_speaker") or "")
    if not latest or latest == "Portfolio Manager":
        return ""
    role = latest.lower()
    if role == exclude_role:
        return ""
    if role == "aggressive":
        return debate.get("current_aggressive_response", "")
    if role == "conservative":
        return debate.get("current_conservative_response", "")
    if role == "neutral":
        return debate.get("current_neutral_response", "")
    return ""


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


# ── PR 3: Risk Debate + Portfolio Manager ────────────────────────────────────


def _ensure_judge_and_trader(
    state: TradingDecisionState,
) -> tuple[dict | None, dict | None]:
    """Pull validated judge and trader payloads, or ``(None, None)`` if missing."""
    judge: Any = state.get("investment_judge")
    trader: Any = state.get("trader")
    if not isinstance(judge, dict) or _is_error_payload(judge):
        return None, None
    if not isinstance(trader, dict) or _is_error_payload(trader):
        return None, None
    return judge, trader


def _risk_debate_skip_update(role_tag: str, debate: RiskDebateState) -> RiskDebateState:
    """Build a debate update for a skipped risk-debater turn (judge/trader missing)."""
    argument = f"{role_tag} (skipped — Investment Judge or Trader output unavailable.)"
    speaker_label = role_tag.split(" ")[0]  # "Aggressive" / "Conservative" / "Neutral"
    field_map = {
        "Aggressive": "current_aggressive_response",
        "Conservative": "current_conservative_response",
        "Neutral": "current_neutral_response",
    }
    history_map = {
        "Aggressive": "aggressive_history",
        "Conservative": "conservative_history",
        "Neutral": "neutral_history",
    }
    update: RiskDebateState = {
        "history": (debate["history"] + "\n\n" + argument).strip(),
        "aggressive_history": debate["aggressive_history"],
        "conservative_history": debate["conservative_history"],
        "neutral_history": debate["neutral_history"],
        "current_aggressive_response": debate["current_aggressive_response"],
        "current_conservative_response": debate["current_conservative_response"],
        "current_neutral_response": debate["current_neutral_response"],
        "latest_speaker": speaker_label,  # type: ignore[typeddict-item]
        "judge_decision": debate["judge_decision"],
        "count": debate["count"] + 1,
    }
    update[history_map[speaker_label]] = (  # type: ignore[literal-required]
        debate[history_map[speaker_label]] + "\n\n" + argument  # type: ignore[literal-required]
    ).strip()
    update[field_map[speaker_label]] = argument  # type: ignore[literal-required]
    return update


async def _run_risk_debator(
    *,
    state: TradingDecisionState,
    config: RunnableConfig,
    role: str,
    tag: str,
    factory,
) -> dict[str, Any]:
    """Shared body for the three risk-debate node wrappers.

    Factored out because Aggressive / Conservative / Neutral have identical
    structure — only the prompt template, speaker tag, and history field
    differ.
    """
    ctx = _context(state)
    debate = _risk_debate(state)
    judge, trader = _ensure_judge_and_trader(state)

    if judge is None or trader is None:
        return {"risk_debate": _risk_debate_skip_update(tag, debate)}

    agent = await factory(
        config,
        ticker=ctx.ticker,
        query=ctx.query,
        context_vars=_context_vars(ctx),
        investment_judge=judge,
        trader=trader,
        debate_history=debate["history"],
        opposing_last=_last_opposing_argument(debate, exclude_role=role),
    )

    try:
        result = await agent.ainvoke({"messages": [HumanMessage(_TRIGGER)]})
    except Exception:
        logger.exception("%s_debator_node failed", role)
        argument = f"{tag} (failed to produce an argument this turn.)"
    else:
        text = _extract_text(result) or "(empty argument)"
        argument = f"{tag} {text}"

    speaker_label = role.capitalize()
    field_map = {
        "aggressive": "current_aggressive_response",
        "conservative": "current_conservative_response",
        "neutral": "current_neutral_response",
    }
    history_map = {
        "aggressive": "aggressive_history",
        "conservative": "conservative_history",
        "neutral": "neutral_history",
    }

    update: RiskDebateState = {
        "history": (debate["history"] + "\n\n" + argument).strip(),
        "aggressive_history": debate["aggressive_history"],
        "conservative_history": debate["conservative_history"],
        "neutral_history": debate["neutral_history"],
        "current_aggressive_response": debate["current_aggressive_response"],
        "current_conservative_response": debate["current_conservative_response"],
        "current_neutral_response": debate["current_neutral_response"],
        "latest_speaker": speaker_label,  # type: ignore[typeddict-item]
        "judge_decision": debate["judge_decision"],
        "count": debate["count"] + 1,
    }
    update[history_map[role]] = (  # type: ignore[literal-required]
        debate[history_map[role]] + "\n\n" + argument  # type: ignore[literal-required]
    ).strip()
    update[field_map[role]] = argument  # type: ignore[literal-required]
    return {"risk_debate": update}


async def aggressive_debator_node(
    state: TradingDecisionState, config: RunnableConfig
) -> dict[str, Any]:
    """Run one Aggressive Risk Debator turn."""
    return await _run_risk_debator(
        state=state,
        config=config,
        role="aggressive",
        tag=AGGRESSIVE_TAG,
        factory=create_aggressive_debator_agent,
    )


async def conservative_debator_node(
    state: TradingDecisionState, config: RunnableConfig
) -> dict[str, Any]:
    """Run one Conservative Risk Debator turn."""
    return await _run_risk_debator(
        state=state,
        config=config,
        role="conservative",
        tag=CONSERVATIVE_TAG,
        factory=create_conservative_debator_agent,
    )


async def neutral_debator_node(
    state: TradingDecisionState, config: RunnableConfig
) -> dict[str, Any]:
    """Run one Neutral Risk Debator turn."""
    return await _run_risk_debator(
        state=state,
        config=config,
        role="neutral",
        tag=NEUTRAL_TAG,
        factory=create_neutral_debator_agent,
    )


async def portfolio_manager_node(
    state: TradingDecisionState, config: RunnableConfig
) -> dict[str, Any]:
    """Synthesise judge + trader + risk-debate into the canonical decision."""
    ctx = _context(state)
    debate = _risk_debate(state)
    judge, trader = _ensure_judge_and_trader(state)

    fallback: dict[str, Any] = {
        "rating": "hold",
        "executive_summary": "Portfolio Manager skipped: upstream outputs missing.",
        "investment_thesis": (
            "No actionable thesis — Judge or Trader did not produce structured output."
        ),
        "time_horizon": "n/a",
        "position_sizing": "0% of NAV",
        "key_risks_remaining": [],
        "confidence": 0.0,
        "incorporates_past_lessons": False,
    }

    if judge is None or trader is None:
        return {
            "portfolio_decision": {
                **fallback,
                "error": "Missing or errored Investment Judge / Trader output.",
            }
        }
    if not debate["history"].strip():
        return {
            "portfolio_decision": {
                **fallback,
                "error": "No risk debate transcript to synthesise.",
            }
        }

    past_reflections = str(state.get("past_reflections") or "")

    agent = await create_portfolio_manager_agent(
        config,
        ticker=ctx.ticker,
        query=ctx.query,
        context_vars=_context_vars(ctx),
        investment_judge=judge,
        trader=trader,
        risk_debate_history=debate["history"],
        past_reflections=past_reflections,
    )

    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(_PORTFOLIO_MANAGER_TRIGGER)]}
        )
    except Exception:
        logger.exception("portfolio_manager_node failed")
        return {
            "portfolio_decision": {
                **fallback,
                "error": "Portfolio Manager agent raised.",
            }
        }

    structured = result.get("structured_response") if isinstance(result, dict) else None
    if isinstance(structured, PortfolioDecisionOutput):
        payload = structured.model_dump()
    elif structured is not None:
        try:
            payload = PortfolioDecisionOutput.model_validate(structured).model_dump()
        except Exception:
            logger.exception("portfolio_manager structured response did not validate")
            return {
                "portfolio_decision": {
                    **fallback,
                    "error": "Portfolio Manager response failed validation.",
                }
            }
    else:
        raw = _extract_text(result)
        return {
            "portfolio_decision": {
                **fallback,
                "error": "Portfolio Manager did not produce structured output.",
                "raw_output": raw,
            }
        }

    risk_update: RiskDebateState = {
        "history": debate["history"],
        "aggressive_history": debate["aggressive_history"],
        "conservative_history": debate["conservative_history"],
        "neutral_history": debate["neutral_history"],
        "current_aggressive_response": debate["current_aggressive_response"],
        "current_conservative_response": debate["current_conservative_response"],
        "current_neutral_response": debate["current_neutral_response"],
        "latest_speaker": "Portfolio Manager",
        "judge_decision": payload.get("executive_summary", ""),
        "count": debate["count"],
    }
    return {
        "portfolio_decision": payload,
        "risk_debate": risk_update,
    }


# ── PR 4: Reflection memory nodes ────────────────────────────────────────────


def _resolve_user_id(config: RunnableConfig) -> str | None:
    """Return the active per-user namespace key, or ``None`` if unresolvable.

    Checks (in order): ``configurable.user_id``, then
    :envvar:`MEMORY_DEBUG_USER_ID` via :class:`MemoryConfiguration`. Unlike
    :func:`utils.backends._memories_namespace`, this returns ``None`` rather
    than raising so the trading-decision pipeline can degrade gracefully
    when memory is unavailable.
    """
    configurable = dict(config.get("configurable") or {})
    user_id = configurable.get("user_id")
    if isinstance(user_id, str) and user_id:
        return user_id
    try:
        debug = MemoryConfiguration.from_runnable_config(config).memory_debug_user_id
    except Exception:
        return None
    return debug or None


def _resolve_decision_date(config: RunnableConfig, state: TradingDecisionState) -> str:
    """Return the date this decision is being made on (``YYYY-MM-DD``).

    Priority: state["decision_date"] (set by a prior node) → configurable
    override → today UTC. Pinned via ``configurable.decision_date`` in
    tests so the writeback key is deterministic.
    """
    existing = state.get("decision_date")
    if isinstance(existing, str) and existing:
        return existing
    configurable = dict(config.get("configurable") or {})
    override = configurable.get("decision_date")
    if isinstance(override, str) and override:
        return override
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y-%m-%d")


def _reflection_disabled(config: RunnableConfig) -> bool:
    """Return True when ``configurable.reflection_enabled`` was set to False."""
    configurable = dict(config.get("configurable") or {})
    return configurable.get("reflection_enabled") is False


def _reflection_knobs(config: RunnableConfig) -> dict[str, Any]:
    """Return per-run tunable knobs for the reflection layer."""
    configurable = dict(config.get("configurable") or {})
    return {
        "holding_days": int(configurable.get("reflection_holding_days", 5)),
        "benchmark": str(configurable.get("reflection_benchmark", "SPY")),
        "max_same_ticker": int(configurable.get("reflection_max_same_ticker", 5)),
        "max_cross_ticker": int(configurable.get("reflection_max_cross_ticker", 3)),
    }


async def reflector_resolve_node(
    state: TradingDecisionState,
    config: RunnableConfig,
    *,
    store: BaseStore | None = None,
    outcomes_fetcher: OutcomesFetcher | None = None,
) -> dict[str, Any]:
    """Resolve pending decisions and inject past reflections into the prompt.

    Runs at the START of the trading-decision pipeline. Always returns at
    least a ``decision_date`` update — even when reflection is disabled or
    the store is unavailable — so downstream nodes have a deterministic
    date to write back with.

    Steps (gated on store + user_id + reflection_enabled):

    1. For every ``pending`` entry, call the outcomes fetcher. If it
       returns ``None``, leave the entry pending (try again next run).
    2. For resolved entries, call the Reflector LLM to produce a 2–4
       sentence reflection and persist it.
    3. Render the latest *N* same-ticker + *M* cross-ticker reflections
       as a Markdown block and write to ``state["past_reflections"]``.
    """
    ctx = _context(state)
    decision_date = _resolve_decision_date(config, state)
    base_update: dict[str, Any] = {
        "decision_date": decision_date,
        "past_reflections": "",
        "resolved_decisions": [],
    }

    if _reflection_disabled(config) or store is None:
        return base_update

    user_id = _resolve_user_id(config)
    if user_id is None:
        return base_update

    try:
        memory = ReflectionMemory(store, user_id)
    except ValueError:
        return base_update

    fetcher: OutcomesFetcher = outcomes_fetcher or fetch_outcomes_openbb
    knobs = _reflection_knobs(config)

    pending = await memory.list_pending()
    resolved_records: list[dict[str, Any]] = []
    for record in pending:
        try:
            outcome = await fetcher(
                config=config,
                ticker=record.ticker,
                decision_date=record.date,
                holding_days=knobs["holding_days"],
                benchmark=knobs["benchmark"],
                decision_action=record.decision.get("rating"),
            )
        except Exception:
            logger.exception(
                "Outcomes fetcher raised for %s/%s — leaving pending",
                record.ticker,
                record.date,
            )
            continue
        if outcome is None:
            continue
        reflection = await generate_reflection(
            config=config,
            ticker=record.ticker,
            decision_date=record.date,
            decision=record.decision,
            outcome=outcome.model_dump(),
        )
        await memory.resolve(
            ticker=record.ticker,
            date=record.date,
            outcome=outcome,
            reflection=reflection,
        )
        resolved_records.append(
            {
                "ticker": record.ticker,
                "date": record.date,
                "outcome": outcome.model_dump(),
                "reflection": reflection,
            }
        )

    same_ticker = await memory.list_resolved_for_ticker(
        ctx.ticker, limit=knobs["max_same_ticker"]
    )
    cross_ticker = await memory.list_resolved_cross_ticker(
        exclude_ticker=ctx.ticker, limit=knobs["max_cross_ticker"]
    )
    block = render_reflections_block(same_ticker=same_ticker, cross_ticker=cross_ticker)

    return {
        "decision_date": decision_date,
        "past_reflections": block,
        "resolved_decisions": resolved_records,
    }


async def decision_writeback_node(
    state: TradingDecisionState,
    config: RunnableConfig,
    *,
    store: BaseStore | None = None,
) -> dict[str, Any]:
    """Persist this run's Portfolio Manager decision as a pending entry.

    Runs at the END of the trading-decision pipeline. Idempotent — writes
    overwrite any prior entry with the same (ticker, date) key. Skips
    when:

    * Reflection is disabled.
    * The store is not configured.
    * No ``user_id`` is resolvable.
    * The decision payload carries an ``error`` key (nothing useful to
      learn from).
    """
    if _reflection_disabled(config) or store is None:
        return {}
    user_id = _resolve_user_id(config)
    if user_id is None:
        return {}
    decision: Any = state.get("portfolio_decision")
    if not isinstance(decision, dict) or "error" in decision:
        return {}

    ctx = _context(state)
    decision_date = _resolve_decision_date(config, state)
    try:
        memory = ReflectionMemory(store, user_id)
    except ValueError:
        return {}
    await memory.write_pending(ticker=ctx.ticker, date=decision_date, decision=decision)
    return {}
