"""Stage 6: Forecasting & Scenario Modeling."""

from typing import Any, Literal

from deepagents import CompiledSubAgent, create_deep_agent
from langchain.agents.structured_output import AutoStrategy
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from muffin_agent.agents.data_collection import (
    create_currency_commodities_data_collection_agent,
    create_economy_macro_data_collection_agent,
    create_equity_estimates_data_collection_agent,
    create_equity_fundamentals_data_collection_agent,
)
from muffin_agent.agents.investment.schemas import DataSource
from muffin_agent.agents.investment.utils import run_deep_agent_node
from muffin_agent.agents.middleware import ToolResultCacheMiddleware
from muffin_agent.agents.subagents import build_validation_subagent
from muffin_agent.config import Configuration
from muffin_agent.prompts import render_template
from muffin_agent.sandbox import get_backend
from muffin_agent.tools.profitability import (
    compute_accruals_ratio,
    compute_revenue_cagr,
)
from muffin_agent.tools.projections import (
    compute_sensitivity,
    project_three_year_financials,
)

# ── Input state schema ─────────────────────────────────────────────────────────


class ForecastingInputState(TypedDict, total=False):
    """Input state schema for ``forecasting_node``.

    Documents which state fields the node reads.  All fields optional; the node
    supports multiple context modes depending on which upstream steps have run.

    Context modes:
        (a) Full pipeline  — ticker + query + company_analysis + market_regime
            Standard Group 2 run after all Group 1 nodes complete.  Uses
            ``company_analysis.financial_history`` as the historical baseline
            and ``market_regime`` macro context for scenario assumptions.
        (b) Ticker + query — agent collects all historical financials fresh
            from equity-fundamentals without relying on prior pipeline state.
        (c) Query only     — thematic forward model without a specific ticker;
            equity-estimates and equity-fundamentals are skipped or generic.
    """

    ticker: str
    query: str
    company_analysis: dict[str, Any]
    """Output of ``company_analysis_node``.

    Must contain ``financial_history`` (5-year time series), ``company_signal``
    (used for scenario probability anchoring), ``financial_quality``, and
    ``key_risks``.
    """

    market_regime: dict[str, Any]
    """Output of ``market_regime_node``.

    Provides macro assumptions: GDP growth label, monetary policy stance,
    inflation regime, and ``key_risks`` for bear-case grounding.
    """


# ── Output schema ─────────────────────────────────────────────────────────────


class YearlyProjection(BaseModel):
    """Financial projections for a single forward year."""

    year: int
    """Calendar year, e.g. 2026."""

    revenue: float | None = None
    """Projected revenue in reporting currency; null if baseline unavailable."""

    revenue_growth_pct: float | None = None
    """Year-over-year revenue growth rate (%)."""

    ebitda: float | None = None
    """Projected EBITDA in reporting currency."""

    ebitda_margin_pct: float | None = None
    """EBITDA / revenue (%)."""

    ebit: float | None = None
    """Projected EBIT (EBITDA minus D&A) in reporting currency."""

    ebit_margin_pct: float | None = None
    """EBIT / revenue (%)."""

    eps: float | None = None
    """Diluted EPS in reporting currency; null if diluted share count unavailable."""

    fcf: float | None = None
    """Free cash flow = net income + D&A - capex in reporting currency."""

    fcf_margin_pct: float | None = None
    """FCF / revenue (%)."""

    # ── Balance sheet projections ──

    net_debt: float | None = None
    """Net debt = total debt - cash; null if balance sheet baseline unavailable."""

    total_debt: float | None = None
    """Total debt in reporting currency."""

    cash: float | None = None
    """Cash and equivalents in reporting currency."""

    working_capital: float | None = None
    """Net working capital = current assets - current liabilities."""

    total_assets: float | None = None
    """Total assets in reporting currency."""

    shareholders_equity: float | None = None
    """Total shareholders' equity in reporting currency."""


class Scenario(BaseModel):
    """A single forward-looking scenario (bull, base, or bear)."""

    label: Literal["bull", "base", "bear"]
    """Scenario type."""

    probability: float
    """Analyst-assigned probability (0.0–1.0).

    The three scenario probabilities should sum to approximately 1.0 (±0.02).
    Anchored defaults by ``company_signal``:
    - ``pass``  → base=0.60, bull=0.25, bear=0.15
    - ``watch`` → base=0.50, bull=0.25, bear=0.25
    - ``fail``  → base=0.40, bull=0.25, bear=0.35
    """

    revenue_cagr_3y_pct: float | None = None
    """3-year revenue CAGR vs the LTM baseline (%)."""

    ebitda_margin_exit_pct: float | None = None
    """EBITDA margin in Year+3 (the exit-year margin for terminal value work)."""

    eps_cagr_3y_pct: float | None = None
    """3-year EPS CAGR vs LTM (%; null if diluted shares unavailable)."""

    key_assumptions: list[str]
    """3-5 specific driver assumptions that define this scenario.

    Each item should be quantified where possible, e.g.
    'Revenue CAGR +12% driven by AI product attach rate expansion'.
    """

    probability_rationale: str
    """1-2 sentences explaining why the probability matches or deviates from
    the ``company_signal``-based anchor."""

    narrative: str
    """2-3 sentences on the primary macro and company-level drivers that
    distinguish this scenario from the base case."""

    projections: list[YearlyProjection]
    """Forward projections for Year+1, Year+2, Year+3 (in that order)."""


class ConsensusAnchor(BaseModel):
    """Analyst consensus data used to anchor the base-case scenario."""

    as_of_date: str
    """Date the consensus data was retrieved (YYYY-MM-DD)."""

    num_analysts: int | None = None
    """Number of analysts contributing to the consensus."""

    eps_year1: float | None = None
    """Consensus mean EPS estimate for Year+1."""

    eps_year2: float | None = None
    """Consensus mean EPS estimate for Year+2."""

    revenue_year1: float | None = None
    """Consensus mean revenue estimate for Year+1 in reporting currency."""

    ebitda_year1: float | None = None
    """Consensus mean EBITDA estimate for Year+1 in reporting currency."""

    price_target_mean: float | None = None
    """Mean analyst 12-month price target."""

    price_target_low: float | None = None
    """Lowest analyst price target."""

    price_target_high: float | None = None
    """Highest analyst price target."""

    revision_trend_3m: Literal["upward", "flat", "downward"]
    """Direction of consensus EPS revisions over the past 3 months.

    'upward' = >+5% change; 'downward' = <-5% change; 'flat' = within ±5%.
    """

    surprise_history: str
    """2-3 sentences on the company's historical EPS beat/miss pattern,
    including typical beat magnitude and cadence (e.g. 'beats in 7 of last
    8 quarters by an average of 4%; one miss in Q3 2023 due to FX headwinds')."""


class SensitivityDriver(BaseModel):
    """Impact of a single assumption change on Year+1 EPS and FCF."""

    driver: str
    """Description of the assumption change, e.g. 'Revenue growth +1pp vs base'."""

    eps_impact_pct: float | None = None
    """% change in Year+1 EPS from this assumption change; null if EPS unavailable."""

    fcf_impact_pct: float | None = None
    """% change in Year+1 FCF from this assumption change."""


class ForecastOutput(BaseModel):
    """Structured output produced by the forecasting deep agent."""

    ticker: str
    """Equity ticker symbol analysed."""

    company_name: str
    """Full legal company name."""

    currency: str
    """Reporting currency ISO code (e.g. 'USD', 'EUR')."""

    forecast_as_of_date: str
    """Date the forecast was produced (YYYY-MM-DD)."""

    base_case: Scenario
    """Base-case scenario: consensus-anchored, mean-reverting margins."""

    bull_case: Scenario
    """Bull-case scenario: named upside catalysts with quantified assumptions."""

    bear_case: Scenario
    """Bear-case scenario: named downside risks with quantified assumptions."""

    consensus_anchoring: ConsensusAnchor
    """Analyst consensus data used to calibrate the base case."""

    revision_momentum: Literal["upward", "flat", "downward"]
    """Top-level consensus EPS revision direction over the past 3 months.

    Mirrors ``consensus_anchoring.revision_trend_3m`` for quick downstream
    access by ``valuation_node`` and ``thesis_synthesis_node``.
    """

    sensitivity_table: list[SensitivityDriver] = Field(default_factory=list)
    """3-5 key assumption sensitivities with Year+1 EPS and FCF impacts."""

    earnings_quality_flags: list[str] = Field(default_factory=list)
    """Earnings quality concerns derived from sandbox computations.

    Examples: 'High accruals ratio (0.18) — earnings may not be fully
    cash-backed', 'EPS growth partially driven by share buybacks',
    'Low FCF conversion (52%) — earnings quality concern'.
    Empty list if no quality concerns identified.
    """

    modeling_notes: str
    """2-3 sentences on: key modeling assumptions (tax rate, D&A rate, capex
    intensity), data quality, and the single event most likely to shift the
    base case materially."""

    confidence: float = Field(ge=0.0, le=1.0)
    """Overall model confidence 0.0–1.0.

    Start at 1.0; reduce by ~0.15 per missing primary subagent result,
    ~0.10 per major data gap (e.g. no consensus data, no financial_history)."""

    data_sources: list[DataSource] = Field(default_factory=list)
    """One entry per subagent consulted: name, what was retrieved, period."""

    limitations: list[str] = Field(default_factory=list)
    """Data gaps that reduce model confidence.

    Examples: 'EPS projections omitted — diluted share count unavailable',
    'No consensus data for this ticker (likely small-cap)',
    'company_analysis.financial_history absent — historical calibration
    based solely on equity-fundamentals (3 years available)'.
    """


# ── Subagent builder ──────────────────────────────────────────────────────────


async def _build_forecasting_subagents(config: Configuration) -> list[CompiledSubAgent]:
    """Build the focused subagent set for forecasting & scenario modeling.

    Return 4 data collection subagents + 1 data validation subagent.
    Excludes macro/market-structure subagents not relevant to forward modeling.
    """
    estimates_agent = await create_equity_estimates_data_collection_agent(config)
    fundamentals_agent = await create_equity_fundamentals_data_collection_agent(config)
    macro_agent = await create_economy_macro_data_collection_agent(config)
    currency_commodities_agent = (
        await create_currency_commodities_data_collection_agent(config)
    )
    validation_subagent = await build_validation_subagent(config)

    return [
        CompiledSubAgent(
            name="equity-estimates",
            description=(
                "Retrieves analyst estimates and forward-looking data: "
                "consensus EPS/revenue/EBITDA estimates for Year+1 and Year+2, "
                "price targets (mean, high, low), forward P/E and EV/EBITDA, "
                "forward sales, analyst rating breakdown (buy/hold/sell), and "
                "estimate revision history (historical consensus EPS over time "
                "for computing 3-month revision momentum). Primary source for "
                "consensus anchoring, revision_trend_3m, and surprise_history."
            ),
            runnable=estimates_agent,
        ),
        CompiledSubAgent(
            name="equity-fundamentals",
            description=(
                "Retrieves fundamental financial data needed for model "
                "calibration: income statement (diluted shares outstanding, "
                "depreciation & amortisation, effective tax rate, historical "
                "EPS), cash flow statement (capital expenditures, operating "
                "cash flow, total assets for accruals ratio). Use to obtain "
                "diluted share count (required for EPS projections) and "
                "to validate or extend the financial_history from "
                "company_analysis. Also provides historical_eps for "
                "multi-year EPS calibration."
            ),
            runnable=fundamentals_agent,
        ),
        CompiledSubAgent(
            name="economy-macro",
            description=(
                "Retrieves macroeconomic data: GDP growth forecast (1-2 year "
                "horizon) as a top-down revenue growth anchor, current 10Y "
                "Treasury rate as a WACC proxy, CPI trend for pricing power "
                "context, and interest rate path for capital structure "
                "assumptions. Use to calibrate macro-sensitive revenue growth "
                "assumptions and to apply a macroeconomic cross-check on the "
                "base-case revenue CAGR."
            ),
            runnable=macro_agent,
        ),
        CompiledSubAgent(
            name="currency-commodities",
            description=(
                "Retrieves currency, commodity, and crypto data: FX rates "
                "and history (major/emerging pairs), commodity spot prices "
                "(WTI, Brent, gold, copper, natural gas), EIA energy outlooks, "
                "and crypto price history. Use for FX exposure assessment, "
                "commodity input cost analysis, energy sector context, and "
                "Risk dimension scoring (commodity/FX headwinds). In "
                "forecasting context: quantify FX translation impact on "
                "multi-national revenue and estimate commodity cost assumptions "
                "for the bear-case margin compression scenario."
            ),
            runnable=currency_commodities_agent,
        ),
        validation_subagent,
    ]


# ── Agent factory ─────────────────────────────────────────────────────────────


_PROBABILITY_ANCHORS: dict[str, tuple[float, float, float]] = {
    "pass": (0.60, 0.25, 0.15),
    "watch": (0.50, 0.25, 0.25),
    "fail": (0.40, 0.25, 0.35),
}
"""company_signal → (base, bull, bear) probability anchors."""


async def create_forecasting_agent(
    config: Configuration,
    company_signal: str | None = None,
):
    """Build the forecasting deep agent.

    Create a deep agent that builds a 3-year forward financial model with
    bull, base, and bear scenarios anchored to analyst consensus.  Uses a
    sandbox backend for all numeric computations (historical calibration,
    scenario projection arithmetic, sensitivity table, accruals ratio).

    Args:
        config: Application configuration.
        company_signal: Triage gate signal from company_analysis_node
            ('pass', 'watch', or 'fail').  Used to set scenario probability
            anchors in the prompt template.  Defaults to 'pass' if absent.
    """
    subagents = await _build_forecasting_subagents(config)
    base_p, bull_p, bear_p = _PROBABILITY_ANCHORS.get(
        company_signal or "pass", _PROBABILITY_ANCHORS["pass"]
    )
    prompt = render_template(
        "investment/forecasting.jinja",
        base_probability=base_p,
        bull_probability=bull_p,
        bear_probability=bear_p,
        company_signal=company_signal or "pass",
    )
    llm = config.get_llm()

    return create_deep_agent(
        model=llm,
        system_prompt=prompt,
        subagents=subagents,
        tools=[
            project_three_year_financials,
            compute_sensitivity,
            compute_accruals_ratio,
            compute_revenue_cagr,
        ],
        middleware=[
            ToolResultCacheMiddleware(
                cacheable_tools=frozenset({
                    "project_three_year_financials",
                    "compute_sensitivity",
                    "compute_accruals_ratio",
                    "compute_revenue_cagr",
                })
            ),
        ],
        backend=get_backend,
        response_format=AutoStrategy(schema=ForecastOutput),
    )


# ── Node ──────────────────────────────────────────────────────────────────────


async def forecasting_node(
    state: ForecastingInputState, config: RunnableConfig
) -> dict[str, Any]:
    """Stage 6: Forecasting & Scenario Modeling.

    Builds a 3-year forward financial model with bull / base / bear scenarios
    anchored to analyst consensus and calibrated against the 5-year historical
    financial time series from ``company_analysis_node``.

    Runs in **parallel** with ``risk_assessment_node`` (Group 2) after all
    Group 1 nodes complete.  Reads ``ForecastingInputState`` fields (ticker,
    query, company_analysis, market_regime) and writes ``forecast`` to state.

    The node runs the full modeling workflow regardless of
    ``company_analysis.company_signal`` — forecasting data is valuable for
    both long and short investment theses.
    """
    company_signal = (
        state.get("company_analysis", {}).get("company_signal")
        if isinstance(state.get("company_analysis"), dict)
        else None
    )

    async def _factory(cfg: Configuration):  # noqa: E501
        return await create_forecasting_agent(cfg, company_signal=company_signal)

    return await run_deep_agent_node(
        state=state,
        config=config,
        agent_factory=_factory,
        input_state_type=ForecastingInputState,
        state_key="forecast",
    )
