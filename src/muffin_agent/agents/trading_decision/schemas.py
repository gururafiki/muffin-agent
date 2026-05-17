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

    @classmethod
    def from_investment_analysis_state(
        cls,
        state: dict[str, Any],
        *,
        ticker: str | None = None,
        query: str | None = None,
        narrative: str | None = None,
    ) -> AnalysisContext:
        """Build a context from a ``TickerAnalysisState`` dict.

        Maps the six structured outputs of ``build_investment_analysis_graph``
        (``market_regime``, ``sector_view``, ``company_analysis``,
        ``forecast``, ``risk_assessment``, ``valuation``) onto the
        equivalent ``AnalysisContext`` fields. Missing keys are silently
        dropped to ``None`` so the adapter works on partial states
        (e.g. when the upstream pipeline interrupted before
        ``thesis_synthesis``).

        Args:
            state: A dict shaped like ``TickerAnalysisState`` — typically
                produced by ``build_investment_analysis_graph().ainvoke(...)``
                or its JSON-serialised equivalent on disk.
            ticker: Override the ticker read from ``state["ticker"]``.
                Required only when *state* lacks a ticker key (unusual).
            query: Override the query read from ``state["query"]``. Useful
                when composing a sub-pipeline whose query should differ.
            narrative: Optional free-form notes appended alongside the
                structured fields. Lets callers mix upstream analysis with
                ad-hoc context in a single envelope.

        Raises:
            ValueError: When neither *state* nor *ticker* provides a ticker.
        """
        resolved_ticker = ticker if ticker is not None else state.get("ticker")
        if not isinstance(resolved_ticker, str) or not resolved_ticker:
            raise ValueError(
                "from_investment_analysis_state requires a non-empty ticker "
                "(either in state['ticker'] or as the `ticker=` kwarg)."
            )

        def _section(key: str) -> dict[str, Any] | None:
            value = state.get(key)
            return value if isinstance(value, dict) and value else None

        return cls(
            ticker=resolved_ticker,
            query=query if query is not None else state.get("query"),
            market_regime=_section("market_regime"),
            sector_view=_section("sector_view"),
            company_analysis=_section("company_analysis"),
            forecast=_section("forecast"),
            risk_assessment=_section("risk_assessment"),
            valuation=_section("valuation"),
            narrative=narrative,
            additional_context={},
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


# ── PR 3 output schemas ───────────────────────────────────────────────────────


class PortfolioDecisionOutput(BaseModel):
    """Final canonical output of the trading-decision pipeline.

    Produced by the Portfolio Manager after the 3-way risk debate. Consumes
    the Investment Judge's directional thesis, the Trader's operational
    proposal, and the Aggressive / Conservative / Neutral risk debate
    transcript. This is the artifact downstream callers (CLI, UI, future
    reflection memory) should treat as canonical.
    """

    rating: InvestmentSignal
    """5-tier rating — consistent with ``InvestmentJudgeOutput.signal`` and
    ``CriteriaAnalysisSynthesis.signal`` so all muffin pipelines share one
    rating vocabulary. The Portfolio Manager may revise the Judge's signal
    based on the risk debate (e.g. downgrade ``strong_buy`` to ``buy`` if
    Conservative landed a serious objection)."""

    executive_summary: str
    """2–4 sentences capturing the decision: rating, position size, top-line
    rationale, and primary risk. Designed to fit a one-line read."""

    investment_thesis: str
    """Detailed reasoning (4–8 sentences). Cites specific evidence from the
    Judge, Trader, and risk debate. Optionally references prior reflections
    when PR 4 wires the reflection-memory injection."""

    price_target: float | None = None
    """Quote-currency 12-month price target. Anchor to ``valuation`` (DCF
    base / scenario NAV) and analyst consensus. ``None`` only when the
    underlying data does not support a specific level."""

    stop_loss: float | None = None
    """Final stop level. May tighten the Trader's stop if the risk debate
    surfaced new downside catalysts. ``None`` when ``rating == "hold"`` and
    no existing position exists."""

    time_horizon: str
    """Expected holding period (e.g. ``"3–6 months"``)."""

    position_sizing: str
    """Final sizing instruction (e.g. ``"2% of NAV starter, scale to 4% on
    Q1 beat"``). May tighten the Trader's sizing if Conservative argued
    persuasively for smaller position."""

    key_risks_remaining: list[str] = Field(default_factory=list)
    """Risks the debate identified but the decision still accepts. These
    are the things that will be monitored, not eliminated."""

    confidence: float = Field(ge=0.0, le=1.0)
    """Overall confidence in the decision (0.0–1.0). Floored by the lower of
    the Judge's conviction and the analytical confidence implied by the
    analysis context's confidence fields."""

    incorporates_past_lessons: bool = False
    """Set ``True`` when the Portfolio Manager prompt was given past
    reflections via the reflection-memory pipeline AND the PM actually
    referenced them in ``investment_thesis``. The agent decides — the
    presence of context doesn't force the flag."""


# ── PR 4: Reflection memory schemas ──────────────────────────────────────────


class Outcome(BaseModel):
    """Realised price-performance outcome for a past decision.

    Computed by ``fetch_outcomes`` from price data over the holding window
    that began on the decision date. Used by the Reflector LLM to grade the
    decision and by future Portfolio Manager prompts to learn from realised
    rather than predicted returns.
    """

    raw_return_pct: float
    """Total return over the holding window, in percent (e.g. ``5.3``)."""

    alpha_return_pct: float
    """Return in excess of the benchmark over the same window, in percent."""

    holding_days: int = Field(ge=1)
    """Actual trading days realised. May be less than requested when the
    benchmark or the ticker has insufficient data near ``decision_date``."""

    benchmark: str = "SPY"
    """Ticker symbol used for the alpha computation."""

    decision_action: str | None = None
    """The ``PortfolioDecisionOutput.rating`` at decision time. Carried here
    so the Reflector can see direction without needing the original decision
    payload."""


class DecisionRecord(BaseModel):
    """A single decision lifecycle entry in the reflection memory store.

    Created in the ``pending`` state at the end of every trading-decision
    run; transitioned to ``resolved`` on the next same-ticker run once
    ``fetch_outcomes`` returns a real outcome and the Reflector LLM produces
    a 2–4 sentence reflection.
    """

    ticker: str
    date: str
    """Decision date in ``YYYY-MM-DD`` format. Forms the storage key
    together with ``ticker``."""

    status: Literal["pending", "resolved"]
    decision: dict[str, Any]
    """``PortfolioDecisionOutput.model_dump()`` snapshot at decision time."""

    outcome: Outcome | None = None
    """``Outcome.model_dump()`` once resolved; ``None`` while pending."""

    reflection: str | None = None
    """2–4 sentence prose reflection produced by the Reflector LLM. ``None``
    while pending."""
