"""Pydantic schemas for the trading_decision module.

Defines the generic ``AnalysisContext`` envelope (input to every agent in
the module) and the structured output schemas for the three PR 1 agents:
Bull Researcher, Bear Researcher, and Investment Judge.

Output schemas for the Trader, Risk Debators, and Portfolio Manager are
added in subsequent PRs.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ── Shared enums ──────────────────────────────────────────────────────────────

InvestmentSignal = Literal[
    "strong_sell",
    "sell",
    "hold",
    "buy",
    "strong_buy",
]
"""5-tier conviction scale. Matches ``CriteriaAnalysisSynthesis.signal`` so the
two pipelines share one rating vocabulary."""


TraderAction = Literal["sell", "hold", "buy"]
"""3-tier execution action. The Trader collapses the 5-tier Judge signal to a
3-tier directional instruction (strong_buy / buy → ``"buy"``; hold → ``"hold"``;
sell / strong_sell → ``"sell"``). Magnitude of conviction is expressed through
``position_sizing`` rather than through the action vocabulary, matching how
trading desks separate direction from sizing."""


# ── Analysis context envelope ─────────────────────────────────────────────────


class AnalysisContext(BaseModel):
    """Generic envelope for analysis context fed to any trading_decision agent.

    All structured fields are optional. Callers can construct contexts from:

    * Muffin's ``investment_analysis`` pipeline (sets the structured fields).
    * Free-form research notes (sets only ``narrative``).
    * A custom upstream pipeline (mixes structured and free-form).

    Agents read this envelope and adapt their reasoning to whatever fields are
    populated — prompts use Jinja2 conditionals so a missing field is silently
    skipped rather than producing an "unknown" placeholder.
    """

    ticker: str
    query: str | None = None
    """Original investment mandate or analysis focus."""

    # Structured analysis outputs (when available)
    market_regime: dict[str, Any] | None = None
    """``MarketRegimeOutput.model_dump()`` from muffin's investment pipeline."""

    sector_view: dict[str, Any] | None = None
    """``SectorViewOutput.model_dump()`` from muffin's investment pipeline."""

    company_analysis: dict[str, Any] | None = None
    """``CompanyAnalysisOutput.model_dump()`` from muffin's investment pipeline."""

    forecast: dict[str, Any] | None = None
    """``ForecastOutput.model_dump()`` from muffin's investment pipeline."""

    risk_assessment: dict[str, Any] | None = None
    """``RiskAssessmentOutput.model_dump()`` from muffin's investment pipeline."""

    valuation: dict[str, Any] | None = None
    """``ValuationOutput.model_dump()`` from muffin's investment pipeline."""

    # Free-form context — always available as a fallback
    narrative: str | None = None
    """Markdown blob with research notes. Always usable; the only required
    context when calling from outside the investment_analysis pipeline."""

    additional_context: dict[str, Any] = Field(default_factory=dict)
    """Caller-supplied extras (e.g. per-user constraints, portfolio context)."""

    @classmethod
    def from_narrative(
        cls, ticker: str, narrative: str, **extras: Any
    ) -> AnalysisContext:
        """Build a context from a single Markdown blob.

        Convenience for ad-hoc callers — useful for CLI, tests, and any path
        that does not run muffin's full investment_analysis pipeline.
        """
        return cls(
            ticker=ticker,
            narrative=narrative,
            query=extras.pop("query", None),
            additional_context=extras,
        )


# ── PR 1 output schemas ───────────────────────────────────────────────────────


class InvestmentJudgeOutput(BaseModel):
    """Final synthesis of a Bull vs Bear debate.

    Produced by the Investment Judge after the debate exits. Captures both
    the directional view (signal) and the synthesised bull / bear cases plus
    a structured monitoring checklist for ongoing thesis tracking.
    """

    signal: InvestmentSignal
    conviction: float = Field(ge=0.0, le=1.0)
    """0.0–1.0 strength of the signal. Independent of direction."""

    summary: str
    """3–5 sentence top-line investment case in plain prose."""

    bull_case: str
    """Synthesised strongest bull argument from the debate transcript."""

    bear_case: str
    """Synthesised strongest bear argument from the debate transcript."""

    key_catalysts: list[str] = Field(default_factory=list)
    """Near-term conviction-building events (earnings, regulatory, macro)."""

    key_risks: list[str] = Field(default_factory=list)
    """Specific conviction-destroyers to monitor."""

    monitoring_checklist: list[str] = Field(default_factory=list)
    """Ongoing thesis-drift indicators that warrant re-evaluation."""

    winning_side: Literal["bull", "bear", "balanced"]
    """Which side of the debate carried the stronger argument. ``balanced``
    only when the evidence is genuinely two-sided — the prompt instructs the
    judge to commit to a side whenever possible."""

    reasoning: str
    """Why this signal and which side won. Cites specific debate points."""


# ── PR 2 output schemas ───────────────────────────────────────────────────────


class TraderOutput(BaseModel):
    """Operational translation of the Investment Judge's directional view.

    Consumed downstream by the Risk Debate (PR 3) and ultimately by the
    Portfolio Manager. Every field is sized and stop-aware — the Trader's
    job is to turn a thesis into something a desk can act on.
    """

    action: TraderAction
    """3-tier instruction. Hold means *no change* — not "small position"."""

    reasoning: str
    """2–4 sentences justifying the action. Cites the specific Judge fields
    (signal, conviction, key_catalysts, key_risks) that drove the call."""

    entry_price: float | None = None
    """Quote-currency entry target. ``None`` when holding, or when a market
    order is appropriate (e.g. on a strong-conviction directional move with
    no clear technical level)."""

    stop_loss: float | None = None
    """Quote-currency stop level for the position. Anchor to the
    ``risk_assessment.ex_ante_stop_level`` when available, otherwise to a
    recent swing low (long) or swing high (short). ``None`` when holding."""

    take_profit: float | None = None
    """Optional quote-currency profit target. Anchor between consensus PT and
    the relevant scenario NAV from ``forecast`` / ``valuation``. ``None`` if
    no specific target is warranted (e.g. open-ended trend follow)."""

    position_sizing: str
    """Concrete sizing instruction (e.g. ``"2–3% of NAV starter, scale to 5%
    on Q1 earnings beat"``). Plain prose, but always anchored to a percent of
    NAV. Vague sizes ("medium") are not acceptable."""

    time_horizon: str
    """Expected holding period (e.g. ``"3–6 months"``). Anchored to the
    Judge's ``key_catalysts`` and ``monitoring_checklist``."""
