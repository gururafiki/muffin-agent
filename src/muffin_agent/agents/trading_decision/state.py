"""Flat state schema for the trading_decision module.

The package is fully self-contained — it does NOT depend on
``agents/investment/`` outputs. Inputs are caller-supplied (``ticker``,
``decision_date``, optional ``query`` / ``narrative``); the four analyst
agents fill in the analysis (``market_report``, ``fundamentals_report``,
``news_report``, ``sentiment_report``) before the Bull/Bear/Judge/
Trader/PM downstream nodes run.

Both debates are wired through ``muffin_agent.multi_agent.build_conference_graph``:
the Bull/Bear investment debate accumulates name-tagged ``AIMessage``
instances into ``investment_debate_messages`` (speakers ``bull_researcher``
/ ``bear_researcher``), and the 3-way risk debate
(Aggressive/Conservative/Neutral) into ``risk_debate_messages``. Each has
its own cursor + routing fields so the two conference subgraphs never
share a channel. Both subgraphs are compiled with a restricted
``output_schema`` (``InvestmentDebateOutput`` / ``RiskDebateOutput``) so
they only emit their own conference-owned channels — never the parent's
``operator.add`` reducer channels, which would otherwise be echoed and
doubled on write-back.

Structured outputs (``investment_judge``, ``trader``, ``portfolio_decision``)
live in their own top-level dict fields populated by the synthesis/judge
nodes via ``Pydantic.model_dump()``.

Reflection fields (``past_reflections``, ``resolved_decisions``) are
populated by the reflection bookends.
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from muffin_agent.middlewares.agent_capture.records import merge_tool_runs


class AnalystInput(TypedDict, total=False):
    """Input contract for the four analyst nodes (market/fundamentals/news/social).

    Each analyst reads exactly ``ticker`` + ``decision_date``. Passed via
    ``graph.add_node(name, agent, input_schema=AnalystInput)`` — NOT
    ``agent.input_schema`` (a property-less ``RootModel`` that maps ``{}`` and
    raises at coercion). A real field-based schema also isolates each analyst to
    just these inputs, so the four parallel analysts never share state.
    """

    ticker: str
    decision_date: str


class InvestmentDebateOutput(TypedDict, total=False):
    """Output schema restricting the Bull/Bear conference subgraph's emissions.

    Passed as ``output_schema=`` to ``build_conference_graph`` so the compiled
    subgraph emits ONLY its own conference-owned channels back to the parent
    graph. Without this restriction the subgraph — compiled against the full
    ``TradingDecisionState`` — would echo the parent's other reducer channels
    (e.g. ``tool_runs``) through its final state, and the parent's reducer
    would re-apply them. See ``multi_agent.build_conference_graph``'s
    ``output_schema`` docstring.
    """

    investment_debate_messages: Annotated[list[BaseMessage], add_messages]
    investment_debate_agent_cursors: dict[str, str]
    investment_next_speaker: str | None


class RiskDebateOutput(TypedDict, total=False):
    """Output schema restricting the risk-debate conference subgraph's emissions.

    Counterpart to :class:`InvestmentDebateOutput` for the 3-way risk debate.
    """

    risk_debate_messages: Annotated[list[BaseMessage], add_messages]
    risk_debate_agent_cursors: dict[str, str]
    next_speaker: str | None


class TradingDecisionState(TypedDict, total=False):
    """Top-level state for ``build_trading_decision_graph`` and its variants.

    All fields are optional (``total=False``); each node reads only the
    slice it needs. Per-role node files declare narrower ``<Role>InputState``
    and ``<Role>OutputState`` TypedDicts to document their precise contract.
    """

    # ── Inputs (from caller) ────────────────────────────────────────────────
    ticker: str
    """Stock ticker symbol (e.g. ``AAPL``). Required. Read by every
    analyst agent and forwarded into Bull/Bear/Judge/Trader/PM prompts."""

    decision_date: str
    """``YYYY-MM-DD`` for the decision. Set by ``reflector_resolve_node``
    if absent (defaults to today). Doubles as the storage key for the
    reflection bookend (``decision_writeback_node``)."""

    query: str
    """Optional user framing question / mandate (e.g. ``"long-term hold
    candidate"``)."""

    narrative: str
    """Optional caller-supplied prior context / research notes. Rendered
    into the downstream Bull/Bear/Judge/Trader/PM prompts alongside the
    analyst reports."""

    # ── Analyst outputs (set by the 4 analyst agents) ──────────────────────
    market_report: str
    """Markdown technical-analysis report from ``market_analyst``."""

    fundamentals_report: str
    """Markdown fundamentals report from ``fundamentals_analyst``."""

    news_report: str
    """Markdown news + macro report from ``news_analyst``."""

    sentiment_report: str
    """Markdown social-sentiment report from ``social_analyst``."""

    # ── Investment debate (wired via multi_agent.build_conference_graph) ────
    investment_debate_messages: Annotated[list[BaseMessage], add_messages]
    """Name-tagged ``AIMessage`` instances accumulated by the Bull/Bear
    debate conference subgraph. One ``AIMessage(content, name=<speaker>)``
    per turn (``bull_researcher`` / ``bear_researcher``). The Investment
    Judge reads this as its synthesis input via :func:`format_debate_history`."""

    investment_debate_agent_cursors: dict[str, str]
    """Per-agent last-seen message id for the Bull/Bear conference. Unused
    today (both debaters are ``LLMParticipant``) — kept for a future
    mix of LLM + Agent participants."""

    investment_next_speaker: str | None
    """Bull/Bear conference routing field — written by the conference's
    ``dispatch`` node, consumed by its conditional edge. Distinct from the
    risk debate's ``next_speaker`` so the two conference subgraphs never
    share a routing channel."""

    investment_judge: dict[str, Any]
    """``InvestmentJudgeOutput.model_dump()`` — set by the judge node."""

    # ── Trader ─────────────────────────────────────────────────────────────
    trader: dict[str, Any]
    """``TraderOutput.model_dump()`` — set by the trader node."""

    # ── Risk debate (wired via multi_agent.build_conference_graph) ─────────
    risk_debate_messages: Annotated[list[BaseMessage], add_messages]
    """Name-tagged ``AIMessage`` instances accumulated by the risk-debate
    conference subgraph. One ``AIMessage(content, name=<speaker>)`` per
    turn. The Portfolio Manager reads this as the synthesis input via
    :func:`format_risk_history`."""

    risk_debate_agent_cursors: dict[str, str]
    """Per-agent last-seen message id (populated by the multi_agent
    framework when AgentParticipants are wired into the risk debate).
    Unused today because all risk debaters are ``LLMParticipant`` — kept
    for future migration to a mix of LLM + Agent participants."""

    next_speaker: str | None
    """Conference-subgraph routing field — written by the ``dispatch``
    node, consumed by the conference's conditional edge. Set to ``None``
    when the terminator fires (signalling end-of-conference)."""

    # ── Portfolio decision (canonical artifact) ────────────────────────────
    portfolio_decision: dict[str, Any]
    """``PortfolioDecisionOutput.model_dump()`` — set by the PM node.
    **Canonical final artifact** for downstream consumers."""

    # ── Reflection memory ──────────────────────────────────────────────────
    past_reflections: str
    """Pre-rendered Markdown block of past same-ticker + cross-ticker
    reflections, injected into the Portfolio Manager prompt."""

    resolved_decisions: list[dict[str, Any]]
    """Observability: list of decisions the resolver resolved this run."""

    # ── Tool-execution capture (declaring the channel opts this graph in) ────
    tool_runs: Annotated[list[dict[str, Any]], merge_tool_runs]
    """Tool-execution records captured by ``AgentCaptureMiddleware``. The four
    analyst agents are added as parent-graph nodes (no ``output_schema``
    restriction), so their records merge up here automatically; the downstream
    Bull/Bear/Judge/Trader/PM nodes make no tool calls."""
