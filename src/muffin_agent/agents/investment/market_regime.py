"""Stage 2: Market Regime & Top-Down Context."""

from typing import Any, Literal

from deepagents import CompiledSubAgent, create_deep_agent
from langchain.agents.structured_output import AutoStrategy
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from muffin_agent.agents.data_collection import (
    create_currency_commodities_data_collection_agent,
    create_economy_macro_data_collection_agent,
    create_etf_index_data_collection_agent,
    create_fama_french_data_collection_agent,
    create_fixed_income_data_collection_agent,
)
from muffin_agent.agents.data_validation import create_data_validation_agent
from muffin_agent.agents.investment.state import TickerAnalysisState
from muffin_agent.config import Configuration
from muffin_agent.prompts import render_template
from muffin_agent.sandbox import get_backend

# ── Context schema ────────────────────────────────────────────────────────────


class MarketRegimeContext(TypedDict, total=False):
    """Context passed to the market regime agent.

    All fields are optional; at least one should be provided.

    Context modes:
        (a) ticker — agent derives sector/style via ``etf_equity_exposure``
        (b) sector / industry / country — explicit context, no ticker lookup
        (c) query only — investment mandate narrows geographic/style focus
    """

    ticker: str
    query: str
    sector: str
    industry: str
    country: str


# ── Output schema ─────────────────────────────────────────────────────────────


class DimensionDetail(BaseModel):
    """Assessment of a single macro regime dimension."""

    label: str
    """Vocabulary label, e.g. 'expanding', 'risk_off', 'neutral'."""

    score: float
    """0.0–1.0 scale; higher = more extreme positive end of the dimension."""

    direction: str
    """Trend direction: 'improving' | 'stable' | 'deteriorating'."""

    key_indicators: str
    """2-3 data points with values and sources supporting the assessment."""


class RegimeDimensions(BaseModel):
    """The four macro dimensions that together define the regime."""

    growth_cycle: DimensionDetail
    inflation_regime: DimensionDetail
    monetary_policy: DimensionDetail
    liquidity_risk_appetite: DimensionDetail


class FactorTilt(BaseModel):
    """Regime-implied tilt for a single Fama-French factor."""

    tilt: Literal["tailwind", "neutral", "headwind"]
    rationale: str
    """One sentence citing specific data supporting the tilt."""


class FactorAssessment(BaseModel):
    """Factor tilts implied by the current macro regime."""

    value: FactorTilt
    quality: FactorTilt
    momentum: FactorTilt
    size: FactorTilt


class YieldCurve(BaseModel):
    """Yield curve and credit spread snapshot."""

    slope_10y2y_bps: float
    """10Y minus 2Y Treasury yield spread in basis points."""

    shape: Literal["normal", "flat", "inverted", "humped"]
    trend: Literal["steepening", "flattening", "stable"]
    credit_spread_ig_bps: float
    """Investment-grade OAS spread in basis points."""

    credit_spread_hy_bps: float
    """High-yield OAS spread in basis points."""


class RecommendedPositioning(BaseModel):
    """Regime-implied portfolio positioning guidance."""

    beta_range: str
    """E.g. '0.7-1.0'."""

    gross_exposure: str
    """Guidance relative to normal, e.g. 'reduce 10-15%'."""

    net_exposure: str
    """Guidance relative to normal, e.g. 'cautiously net long 40-50%'."""

    sector_tilts: str
    """Favoured and avoided sectors."""

    style_tilts: str
    """Factor/style preferences, e.g. 'quality over growth'."""


class TickerImpact(BaseModel):
    """How the current macro regime specifically affects a given ticker."""

    ticker: str
    regime_sensitivity: Literal["favorable", "neutral", "adverse"]
    rationale: str
    """2-3 sentences tying the regime to this stock's sector and style."""

    specific_risks: list[str]
    specific_tailwinds: list[str]


class DataSource(BaseModel):
    """Record of data collected from one subagent."""

    subagent: str
    data_retrieved: str
    period: str


class MarketRegimeOutput(BaseModel):
    """Structured output produced by the market regime deep agent."""

    regime_label: str
    """3-6 word plain-English label, e.g. 'Late-cycle stagflationary pressure'."""

    as_of_date: str
    """Analysis date in YYYY-MM-DD format."""

    confidence: float
    """0.0–1.0 reflecting data quality and completeness."""

    dimensions: RegimeDimensions
    factor_assessment: FactorAssessment
    yield_curve: YieldCurve
    macro_summary: str
    """3-4 sentence narrative covering regime dynamics and directional outlook."""

    key_risks: list[str]
    """2-4 macro tail risks."""

    recommended_positioning: RecommendedPositioning
    ticker_impact: TickerImpact | None = None
    """Present only when a ticker was provided in the task."""

    data_sources: list[DataSource] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    """Data gaps or uncertainties that reduce confidence."""


# ── Subagent builder ──────────────────────────────────────────────────────────


async def _build_macro_subagents(config: Configuration) -> list[CompiledSubAgent]:
    """Build the macro-focused subagents for market regime analysis.

    Return 5 data collection subagents + 1 data validation subagent, covering
    all macro and regime-relevant data sources.  Excludes company-specific
    subagents (equity-fundamentals, equity-price, etc.) that are irrelevant to
    market-wide regime classification.
    """
    economy_macro_agent = await create_economy_macro_data_collection_agent(config)
    fixed_income_agent = await create_fixed_income_data_collection_agent(config)
    fama_french_agent = await create_fama_french_data_collection_agent(config)
    currency_commodities_agent = (
        await create_currency_commodities_data_collection_agent(config)
    )
    etf_index_agent = await create_etf_index_data_collection_agent(config)
    validation_agent = await create_data_validation_agent(config)

    return [
        CompiledSubAgent(
            name="economy-macro",
            description=(
                "Retrieves macroeconomic data: real GDP growth, CPI/PCE, "
                "unemployment, payrolls, PMI, FOMC minutes (policy tone), "
                "FRED series, UMich sentiment, SLOOS, CLI, money measures. "
                "Primary source for the growth cycle and inflation dimensions."
            ),
            runnable=economy_macro_agent,
        ),
        CompiledSubAgent(
            name="fixed-income",
            description=(
                "Retrieves rates data: Treasury yield curve (3M, 2Y, 5Y, 10Y, 30Y), "
                "EFFR/SOFR, TIPS breakevens (5Y, 10Y), IG/HY credit spreads, "
                "corporate bond yields, and rate forecasts. Primary source for "
                "monetary policy stance and yield curve shape analysis."
            ),
            runnable=fixed_income_agent,
        ),
        CompiledSubAgent(
            name="fama-french",
            description=(
                "Retrieves Fama-French factor data: 5-factor returns "
                "(Mkt-RF, SMB, HML, RMW, CMA) and MOM with trailing 1M/3M/12M "
                "cumulative returns; US portfolio returns by size/value quintile; "
                "breakpoints. Market-wide only — no ticker. Use to assess the "
                "factor regime and calibrate tilt recommendations."
            ),
            runnable=fama_french_agent,
        ),
        CompiledSubAgent(
            name="currency-commodities",
            description=(
                "Retrieves currency and commodity data: FX rates (EUR/USD, "
                "USD/JPY, USD/CNY) as dollar-strength proxies, WTI/Brent crude, "
                "gold (risk sentiment), copper (growth proxy), EIA energy outlook. "
                "Use for commodity inflation signals and risk-appetite context."
            ),
            runnable=currency_commodities_agent,
        ),
        CompiledSubAgent(
            name="etf-index",
            description=(
                "Retrieves ETF and index data: S&P 500 multiples (P/E, P/B, "
                "EV/EBITDA), sector ETF performance (1M, 3M), index snapshots. "
                "If ticker provided, call `etf_equity_exposure` to identify "
                "sector ETFs and infer sector/style. Use for market valuation, "
                "sector rotation, and liquidity signals."
            ),
            runnable=etf_index_agent,
        ),
        CompiledSubAgent(
            name="data-validation",
            description=(
                "Validates collected data against a criterion. Checks "
                "sufficiency, relevance, temporal validity, and consistency. "
                "Returns per-dimension scores (0-1), overall confidence/"
                "relevance scores, identified gaps, and a recommendation "
                "(proceed/collect_more_data/insufficient_data). Use after "
                "data collection, before analysis. Pass the criterion, "
                "analysis date, and all collected data in the task instruction."
            ),
            runnable=validation_agent,
        ),
    ]


# ── Agent factory ─────────────────────────────────────────────────────────────


async def create_market_regime_agent(config: Configuration):
    """Build the market regime deep agent.

    Create a deep agent that collects macro and fixed-income data, validates
    it, classifies the current regime across 4 dimensions (growth cycle,
    inflation, monetary policy, liquidity/risk appetite), and produces factor
    tilt and positioning guidance.

    ``get_backend`` discovers or creates a sandbox container per conversation
    by ``thread_id`` metadata for Python computations (yield curve slope,
    factor Z-scores, composite indicators).

    ``response_format=AutoStrategy(MarketRegimeOutput)`` instructs the agent
    to call a structured output tool as its final act, returning a validated
    ``MarketRegimeOutput`` instance in ``result["structured_response"]``
    instead of free-form text.
    """
    subagents = await _build_macro_subagents(config)
    prompt = render_template("market_regime.jinja")
    llm = config.get_llm()

    return create_deep_agent(
        model=llm,
        system_prompt=prompt,
        subagents=subagents,
        backend=get_backend,
        response_format=AutoStrategy(schema=MarketRegimeOutput),
    )


# ── Context and node helpers ──────────────────────────────────────────────────


def _build_task_description(context: MarketRegimeContext) -> str:
    """Build the task description string from a MarketRegimeContext.

    Converts the typed context into a task string that the deep agent receives
    as its user message.  Only includes keys that are present in the context.
    """
    parts: list[str] = ["Classify the current macro and market regime."]

    if ticker := context.get("ticker"):
        parts.append(f"Ticker: {ticker}")
    if sector := context.get("sector"):
        parts.append(f"Sector: {sector}")
    if industry := context.get("industry"):
        parts.append(f"Industry: {industry}")
    if country := context.get("country"):
        parts.append(f"Country/region focus: {country}")
    if query := context.get("query"):
        parts.append(f"Investment mandate: {query}")

    if context.get("ticker"):
        parts.append(
            "Populate the ticker_impact field with the regime impact on "
            f"{context['ticker']}."
        )
    else:
        parts.append("Omit the ticker_impact field (no ticker provided).")

    return "\n".join(parts)


async def market_regime_node(
    state: TickerAnalysisState, config: RunnableConfig
) -> dict[str, Any]:
    """Stage 2: Market Regime & Top-Down Context.

    Classifies the current macro and liquidity regime, identifies factor
    tailwinds and headwinds, and sets the top-down frame within which the
    individual company analysis will be interpreted.

    Runs in **parallel** with ``sector_analysis_node`` and
    ``company_analysis_node`` (Group 1).  Its output flows into
    ``forecasting_node`` and ``risk_assessment_node`` (both Group 2) as
    contextual input for macro assumptions and stress-scenario design.

    Also used as a shared context node in the equity screening graph
    (``ScreeningState``) where it runs once before the per-ticker fan-out.
    Accepts both ``TickerAnalysisState`` (with ``ticker``) and
    ``ScreeningState`` (query-only); reads fields via ``.get()`` so missing
    keys degrade gracefully.

    Context is passed to the agent via ``MarketRegimeContext``:
        (a) ticker — agent derives sector/style via ``etf_equity_exposure``
        (b) sector / industry / country — passed explicitly
        (c) query only — investment mandate narrows geographic/style focus

    Outputs (state update):
        market_regime: ``MarketRegimeOutput.model_dump()`` dict, or an error
        dict ``{"regime_label": "unknown", "error": ..., "raw_output": ...}``
        if the agent fails to return structured output.
    """
    configuration = Configuration.from_runnable_config(config)
    agent = await create_market_regime_agent(configuration)

    # Build typed context from whatever is available in state
    context: MarketRegimeContext = {}
    if ticker := state.get("ticker"):  # type: ignore[union-attr]
        context["ticker"] = ticker
    if query := state.get("query"):  # type: ignore[union-attr]
        context["query"] = query
    for field in ("sector", "industry", "country"):
        if val := state.get(field):  # type: ignore[union-attr]
            context[field] = val  # type: ignore[literal-required]

    task = _build_task_description(context)
    result = await agent.ainvoke({"input": task})

    structured: MarketRegimeOutput | None = (
        result.get("structured_response") if isinstance(result, dict) else None
    )
    if structured is None:
        return {
            "market_regime": {
                "regime_label": "unknown",
                "error": "Agent did not produce structured output",
                "raw_output": (
                    result.get("output", "")
                    if isinstance(result, dict)
                    else str(result)
                ),
            }
        }

    return {"market_regime": structured.model_dump()}
