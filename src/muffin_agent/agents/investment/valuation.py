"""Stage 7: Valuation & Relative Value."""

from typing import Any, Literal

from deepagents import CompiledSubAgent, create_deep_agent
from langchain.agents.structured_output import AutoStrategy
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from ...middlewares import ToolResultCacheMiddleware
from ...model_config import ModelConfiguration
from ...prompts import render_template
from ...sandbox import get_backend
from ...tools.valuation import (
    compute_dcf,
    compute_multiples_value,
    compute_scenario_weighted_value,
    compute_wacc,
)
from ..data_collection import (
    create_discovery_screening_data_collection_agent,
    create_equity_estimates_data_collection_agent,
    create_equity_price_data_collection_agent,
    create_etf_index_data_collection_agent,
    create_fixed_income_data_collection_agent,
)
from ..subagents import build_validation_subagent
from .schemas import DataSource
from .utils import run_deep_agent_node

# ── Input state schema ─────────────────────────────────────────────────────────


class ValuationInputState(TypedDict, total=False):
    """Input state schema for ``valuation_node``.

    Documents which state fields the node reads.  All fields optional; the
    node runs best in the full pipeline context where ``forecast`` and
    ``risk_assessment`` are populated.

    Context modes:
        (a) Full pipeline  — forecast + risk_assessment + sector_view + market_regime
            Standard Group 3 run after all Group 2 nodes complete.  ``forecast``
            provides the 3-year FCF projections and scenario probabilities as DCF
            inputs.  ``risk_assessment.beta`` anchors the CAPM cost of equity.
            ``sector_view`` supplies peer context for multiple selection.
        (b) Partial pipeline — forecast only (risk_assessment absent)
            WACC falls back to beta=1.0 assumption; notes added to limitations.
        (c) Ticker + query only — agent collects fresh forward estimates from
            equity-estimates without relying on prior pipeline state.  DCF is
            constructed from consensus forecasts.
    """

    ticker: str
    query: str

    forecast: dict[str, Any]
    """Output of ``forecasting_node``.

    Key fields consumed:
    - ``base_case / bull_case / bear_case`` — FCF projections (year 1-3), EBITDA
      exit margins, net_debt Year+3, scenario probabilities
    - ``ticker``, ``company_name``, ``currency``
    - ``consensus_anchoring.price_target_mean / low / high``
    """

    risk_assessment: dict[str, Any]
    """Output of ``risk_assessment_node``.

    Key fields consumed: ``beta`` (WACC CAPM input), ``risk_signal`` (context).
    If absent, beta defaults to 1.0.
    """

    sector_view: dict[str, Any]
    """Output of ``sector_analysis_node``.

    Provides cycle position and peer context to inform exit-multiple selection
    and relative-value benchmarking.
    """

    market_regime: dict[str, Any]
    """Output of ``market_regime_node``.

    Provides regime label for terminal growth rate calibration and equity risk
    premium adjustment.
    """


# ── Output schema ──────────────────────────────────────────────────────────────


class DCFValue(BaseModel):
    """Intrinsic value estimates from the blended DCF model."""

    bull: float | None = None
    """Bull-case NAV per share from ``compute_dcf``."""

    base: float | None = None
    """Base-case NAV per share (blended exit-multiple + Gordon Growth)."""

    bear: float | None = None
    """Bear-case NAV per share from ``compute_dcf``."""

    nav_blended_base: float | None = None
    """Base-case blended NAV per share (same as ``base`` when methodology is
    ``blended``; included for explicitness)."""

    methodology: Literal["blended", "exit_multiple", "gordon_growth"] | None = None
    """Terminal value method(s) used.

    ``blended``: both exit-multiple and Gordon Growth averaged.
    ``exit_multiple``: only exit EV/EBITDA multiple used.
    ``gordon_growth``: only perpetuity growth formula used.
    ``None``: insufficient inputs for either method.
    """

    wacc_used: float | None = None
    """WACC applied as the discount rate (decimal, e.g. 0.089 for 8.9%)."""

    terminal_growth_rate_used: float | None = None
    """Perpetuity growth rate used in the Gordon Growth method (decimal)."""

    exit_multiple_used: float | None = None
    """EV/EBITDA exit multiple applied to terminal-year EBITDA."""


class PeerComparison(BaseModel):
    """Relative value comparison for a single valuation metric."""

    metric: Literal["ev_ebitda", "pe"]
    """Which valuation metric this comparison covers."""

    stock_current: float | None = None
    """Stock's current forward multiple (NTM)."""

    peer_median: float | None = None
    """Median multiple across the comparable peer group."""

    market_median: float | None = None
    """Broad market (S&P 500) median multiple from ``index_sp500_multiples``."""

    premium_discount_pct: float | None = None
    """Stock's current multiple vs. peer median as a percentage.

    Positive = premium; negative = discount.
    Formula: (stock_current − peer_median) / peer_median × 100.
    """

    historical_5y_avg: float | None = None
    """Stock's own 5-year median multiple computed from historical market cap
    and fundamental data (sandbox Block D)."""

    vs_own_history: Literal["premium", "discount", "inline"] | None = None
    """Whether the stock trades at a premium, discount, or inline vs. its own
    5-year median.

    ``premium``: stock_current > historical_5y_avg × 1.05.
    ``discount``: stock_current < historical_5y_avg × 0.95.
    ``inline``: within ±5%.
    ``None``: historical_5y_avg unavailable.
    """


class ValuationOutput(BaseModel):
    """Structured output produced by the valuation deep agent."""

    ticker: str
    """Equity ticker symbol analysed."""

    company_name: str
    """Full legal company name."""

    as_of_date: str
    """Date the valuation was produced (YYYY-MM-DD)."""

    currency: str
    """Reporting currency ISO code (e.g. 'USD', 'EUR')."""

    # ── Market price ──

    current_price: float | None = None
    """Current market price per share from ``equity_price_quote``.
    None if the quote was unavailable."""

    # ── Intrinsic value methods ──

    dcf_value: DCFValue | None = None
    """DCF-based intrinsic value range (bull / base / bear NAV per share).
    None if FCF projections were entirely absent."""

    ev_ebitda_value: float | None = None
    """NTM EV/EBITDA-implied per-share fair value from ``compute_multiples_value``.
    None if NTM EBITDA or peer median EV/EBITDA was unavailable."""

    pe_value: float | None = None
    """NTM P/E-implied per-share fair value from ``compute_multiples_value``.
    None if NTM EPS or peer median P/E was unavailable."""

    fcf_yield_value: float | None = None
    """FCF yield-implied per-share fair value from ``compute_multiples_value``.
    None if NTM FCF was unavailable."""

    sum_of_parts: dict[str, Any] | None = None
    """Sum-of-parts valuation breakdown.  Always None in v1 (not implemented)."""

    # ── Analyst targets ──

    analyst_target_median: float | None = None
    """Mean analyst 12-month price target from
    ``equity_estimates_price_target_consensus``."""

    analyst_target_range: list[float] | None = None
    """[low, high] analyst price target range.  None if fewer than 2 analysts."""

    # ── Scenario-weighted NAV & upside / downside ──

    probability_weighted_nav: float | None = None
    """Probability-weighted NAV = bull × p_bull + base × p_base + bear × p_bear.
    From ``compute_scenario_weighted_value``."""

    upside_base: float | None = None
    """% upside from current price to base-case DCF NAV.  Positive = upside."""

    upside_bull: float | None = None
    """% upside from current price to bull-case DCF NAV."""

    downside_bear: float | None = None
    """% change from current price to bear-case DCF NAV.  Typically negative."""

    risk_reward_ratio: float | None = None
    """Absolute ratio |upside_base / downside_bear|.

    Higher is better (more base upside per unit of bear downside).
    None when downside_bear is zero or when current_price is unavailable.
    """

    # ── Relative value ──

    relative_value: list[PeerComparison] = Field(default_factory=list)
    """Two entries: EV/EBITDA and P/E relative-value comparisons.

    Each entry shows stock_current vs. peer_median, market_median, and the
    stock's own 5-year historical average.
    """

    # ── WACC ──

    wacc: float | None = None
    """WACC applied as the discount rate (decimal).  From ``compute_wacc``."""

    # ── Actionable outputs ──

    valuation_signal: Literal["cheap", "fairly_valued", "expensive"]
    """Composite valuation signal for downstream use.

    ``cheap``: probability_weighted_nav > current_price × 1.20 AND no
        significant peer premium (premium_discount_pct < 20%).
    ``expensive``: probability_weighted_nav < current_price × 0.90 OR both
        P/E and EV/EBITDA > 20% premium to peers with no quality justification.
    ``fairly_valued``: neither criterion clearly met.
    """

    key_valuation_drivers: list[str] = Field(default_factory=list)
    """3-5 specific, quantified statements describing the primary value drivers.

    Examples: 'Base DCF NAV of $185 implies 22% upside at current $152 price
    (WACC 8.9%)', 'Forward EV/EBITDA of 12× vs peer median 14× — 14% discount
    to peers despite higher quality profile'.
    """

    # ── Standard metadata ──

    confidence: float = Field(ge=0.0, le=1.0)
    """Overall valuation confidence 0.0–1.0.

    Start at 1.0; reduce by ~0.15 if equity-price quote unavailable, ~0.15 if
    forecast FCFs all null, ~0.10 per missing primary data source (no peer data,
    no analyst estimates).
    """

    data_sources: list[DataSource] = Field(default_factory=list)
    """One entry per subagent consulted: name, what was retrieved, period covered."""

    limitations: list[str] = Field(default_factory=list)
    """Data gaps or assumption caveats that reduce confidence.

    Examples: 'Historical own-multiples unavailable — 5-year EV/EBITDA
    comparison omitted', 'Beta unavailable — WACC computed with beta=1.0
    assumption', 'No peer comparison data — multiples-based values omitted'.
    """


# ── Subagent builder ───────────────────────────────────────────────────────────


async def _build_valuation_subagents(config: RunnableConfig) -> list[CompiledSubAgent]:
    """Build the focused subagent set for valuation & relative value.

    Return 5 data collection subagents + 1 data validation subagent.
    """
    price_agent = await create_equity_price_data_collection_agent(config)
    estimates_agent = await create_equity_estimates_data_collection_agent(config)
    etf_agent = await create_etf_index_data_collection_agent(config)
    screening_agent = await create_discovery_screening_data_collection_agent(config)
    fixed_income_agent = await create_fixed_income_data_collection_agent(config)
    validation_subagent = await build_validation_subagent(config)

    return [
        CompiledSubAgent(
            name="equity-price",
            description=(
                "Retrieves current market price data: ``equity_price_quote`` for "
                "the current bid/ask midpoint (current_price — MANDATORY for "
                "upside/downside computation and capital-structure weight derivation), "
                "``equity_historical_market_cap`` for the 5-year monthly market cap "
                "history needed to compute the stock's own historical EV/EBITDA and "
                "P/E multiples in Block D sandbox, and ``equity_price_performance`` "
                "for trailing return context."
            ),
            runnable=price_agent,
        ),
        CompiledSubAgent(
            name="equity-estimates",
            description=(
                "Retrieves analyst consensus forward estimates: "
                "``equity_estimates_forward_ebitda`` (NTM EBITDA for EV/EBITDA "
                "multiples valuation), ``equity_estimates_forward_eps`` (NTM EPS for "
                "P/E valuation), ``equity_estimates_forward_pe`` (consensus forward "
                "P/E for cross-check), and ``equity_estimates_price_target_consensus`` "
                "(analyst price target mean, low, high and number of analysts). "
                "Primary source for analyst_target_median and NTM metrics in Block C."
            ),
            runnable=estimates_agent,
        ),
        CompiledSubAgent(
            name="etf-index",
            description=(
                "Retrieves broad market and sector-level valuation benchmarks: "
                "``index_sp500_multiples`` for S&P 500 forward P/E and EV/EBITDA "
                "(market_median comparisons), and ``etf_sectors`` for sector-level "
                "multiples. Used to populate ``market_median`` in relative_value and "
                "to provide a floor/ceiling for peer-multiple selection when "
                "discovery-screening data is thin."
            ),
            runnable=etf_agent,
        ),
        CompiledSubAgent(
            name="discovery-screening",
            description=(
                "Retrieves peer group valuation multiples: ``equity_compare_peers`` "
                "for a list of comparable companies with their individual EV/EBITDA "
                "and P/E ratios (used to compute peer_median in Block C and "
                "relative_value), and ``equity_compare_groups`` for industry-group "
                "summary multiples as a broader benchmark. Primary source for "
                "exit_ebitda_multiple (peer median) passed to ``compute_dcf`` and "
                "for the peer_median fields in PeerComparison."
            ),
            runnable=screening_agent,
        ),
        CompiledSubAgent(
            name="fixed-income",
            description=(
                "Retrieves the current risk-free rate for WACC computation: "
                "``fixedincome_rate_effr`` (Federal Funds Rate / SOFR — preferred "
                "for short-term rate context) or "
                "``fixedincome_government_treasury_rates`` "
                "(10-year Treasury yield — standard for equity DCF discount rates). "
                "Used as risk_free_rate input to ``compute_wacc``. Also provides "
                "credit spread context for estimating cost_of_debt."
            ),
            runnable=fixed_income_agent,
        ),
        validation_subagent,
    ]


# ── Agent factory ──────────────────────────────────────────────────────────────


async def create_valuation_agent(
    config: RunnableConfig,
    store: BaseStore | None = None,
) -> Any:
    """Build the valuation deep agent.

    Create a deep agent that derives intrinsic value via a blended DCF model
    (exit-multiple + Gordon Growth terminal values), NTM-multiples-based fair
    values (EV/EBITDA, P/E, FCF yield), and benchmarks the stock against its
    peer group and own 5-year history.

    Args:
        config: Application configuration.
        store: Shared ``BaseStore`` for cross-agent tool result caching.
    """
    subagents = await _build_valuation_subagents(config)
    prompt = render_template("investment/valuation.jinja")
    model_config = ModelConfiguration.from_runnable_config(config)
    llm = model_config.get_llm()

    return create_deep_agent(
        model=llm,
        system_prompt=prompt,
        subagents=subagents,
        tools=[
            compute_wacc,
            compute_dcf,
            compute_multiples_value,
            compute_scenario_weighted_value,
        ],
        backend=get_backend,
        store=store,
        middleware=[
            ToolResultCacheMiddleware(
                cacheable_tools=frozenset(
                    {
                        "compute_wacc",
                        "compute_dcf",
                        "compute_multiples_value",
                        "compute_scenario_weighted_value",
                    }
                ),
            ),
        ],
        response_format=AutoStrategy(schema=ValuationOutput),
    )


# ── Node ───────────────────────────────────────────────────────────────────────


async def valuation_node(
    state: ValuationInputState,
    config: RunnableConfig,
    *,
    store: BaseStore | None = None,
) -> dict[str, Any]:
    """Stage 7: Valuation & Relative Value.

    Derives an intrinsic-value range via blended DCF (exit-multiple and Gordon
    Growth terminal values averaged), NTM EV/EBITDA multiples, NTM P/E
    multiples, and FCF yield.  Benchmarks the stock against its peer group and
    own 5-year history.  Produces a probability-weighted NAV, upside/downside
    metrics, and a ``valuation_signal`` for downstream use.

    Runs **sequentially** after both ``forecasting_node`` and
    ``risk_assessment_node`` complete (Group 3, first node).  Reads
    ``ValuationInputState`` fields (ticker, query, forecast, risk_assessment,
    sector_view, market_regime) and writes ``valuation`` to state.

    Its output flows into ``thesis_synthesis_node`` which uses the
    ``valuation_signal``, ``probability_weighted_nav``, ``upside_base``,
    ``downside_bear``, ``risk_reward_ratio``, and ``relative_value`` to
    formulate the final investment thesis and conviction score.
    """
    return await run_deep_agent_node(
        state=state,
        config=config,
        agent_factory=create_valuation_agent,
        input_state_type=ValuationInputState,
        state_key="valuation",
        error_fallback={"valuation_signal": "fairly_valued"},
        store=store,
    )
