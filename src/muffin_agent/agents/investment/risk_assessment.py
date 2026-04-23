"""Stage 8: Risk & Downside / Stress Testing."""

from typing import Any, Literal

from deepagents import CompiledSubAgent
from langchain.agents.structured_output import AutoStrategy
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from ...model_config import ModelConfiguration
from ...tools.risk import (
    compute_beta,
    compute_max_drawdown,
    compute_sharpe_sortino,
    compute_var_cvar,
)
from ...utils.agent_builder import MuffinAgentBuilder
from ..data_collection import (
    create_economy_macro_data_collection_agent,
    create_equity_ownership_data_collection_agent,
    create_equity_price_data_collection_agent,
    create_fama_french_data_collection_agent,
    create_fixed_income_data_collection_agent,
    create_options_data_collection_agent,
)
from ..subagents import build_validation_subagent
from .schemas import DataSource
from .utils import run_deep_agent_node

# ── Input state schema ─────────────────────────────────────────────────────────


class RiskAssessmentInputState(TypedDict, total=False):
    """Input state schema for ``risk_assessment_node``.

    Documents which state fields the node reads.  All fields optional; the node
    supports multiple context modes depending on which upstream steps have run.

    Context modes:
        (a) Full pipeline  — ticker + query + company_analysis + market_regime
            Standard Group 2 run after all Group 1 nodes complete.  Uses
            ``company_analysis.key_risks`` for the idiosyncratic stress scenario
            and ``market_regime.key_risks`` for the 3 regime-derived scenarios.
        (b) Ticker + query — full per-stock risk profile without upstream context;
            agent derives stress scenarios from factor data and options-implied risk.
        (c) Query only     — sector/thematic risk assessment without a specific ticker;
            equity-price is skipped; factor and options analysis uses sector ETFs.
    """

    ticker: str
    query: str

    company_analysis: dict[str, Any]
    """Output of ``company_analysis_node``.

    Used fields: ``key_risks`` (idiosyncratic scenario source), ``company_signal``
    (context), ``financial_quality.leverage_ratio`` (credit risk context).
    """

    market_regime: dict[str, Any]
    """Output of ``market_regime_node``.

    Used fields: ``key_risks`` (regime-derived stress scenario source),
    ``factor_assessment`` (regime-factor cross-check), ``recommended_positioning``
    (beta/exposure context).
    """


# ── Output schema ──────────────────────────────────────────────────────────────


class FactorLoadings(BaseModel):
    """FF5 + Momentum factor loadings from multi-factor OLS regression."""

    beta: float | None = None
    """Market (Mkt-RF) factor loading from FF regression."""

    smb: float | None = None
    """Size factor loading (positive = small-cap tilt)."""

    hml: float | None = None
    """Value factor loading (positive = value tilt)."""

    rmw: float | None = None
    """Profitability factor loading (positive = quality/high-profitability tilt)."""

    cma: float | None = None
    """Investment factor loading (positive = conservative-investment tilt)."""

    umd: float | None = None
    """Momentum factor loading (positive = momentum tilt).  None if UMD unavailable."""

    alpha_annualized: float | None = None
    """Annualised Jensen's alpha from the factor regression (decimal)."""

    r_squared: float | None = None
    """Coefficient of determination of the factor regression (0.0–1.0)."""

    regression_period: str | None = None
    """Date range used for the regression, e.g. '2023-01 to 2025-01 (monthly)'."""


class ImpliedVolatilityTermStructure(BaseModel):
    """Options-derived implied volatility term structure and skew."""

    iv_30d_pct: float | None = None
    """30-day at-the-money implied volatility (%)."""

    iv_60d_pct: float | None = None
    """60-day ATM IV (%)."""

    iv_90d_pct: float | None = None
    """90-day ATM IV (%)."""

    put_call_skew_25d: float | None = None
    """25-delta put IV minus 25-delta call IV (percentage points).

    Positive = puts more expensive than calls = bearish / tail-risk skew.
    """

    term_slope: Literal["normal", "flat", "inverted"] | None = None
    """IV term structure slope.

    ``normal`` = iv_30d < iv_90d (market calm near-term).
    ``inverted`` = iv_30d > iv_90d + 3pp (elevated near-term fear).
    ``flat`` = otherwise.
    """


class ShortInterestMetrics(BaseModel):
    """Short interest and crowding metrics."""

    short_interest_pct: float | None = None
    """Short interest as a percentage of total float."""

    days_to_cover: float | None = None
    """Short interest divided by average daily trading volume."""

    short_volume_ratio: float | None = None
    """Short volume as a fraction of total daily volume (0.0–1.0)."""

    crowding_signal: Literal["low", "moderate", "high"]
    """Crowding classification.

    ``high``: short_interest_pct > 20% OR days_to_cover > 10.
    ``moderate``: short_interest_pct 10–20% OR days_to_cover 5–10.
    ``low``: otherwise.
    """


class StressScenario(BaseModel):
    """A single stress scenario with estimated stock-level P&L impact."""

    name: str
    """Short, descriptive scenario name (e.g. 'GFC 2008 analog')."""

    scenario_type: Literal["macro", "historical", "idiosyncratic"]
    """Scenario origin type."""

    description: str
    """1–2 sentences describing the scenario and its trigger."""

    market_return_assumed_pct: float | None = None
    """Assumed broad market return (%).  None for pure idiosyncratic scenarios."""

    estimated_stock_return_pct: float
    """Estimated stock return (%) under this scenario.  Negative = loss."""

    estimated_dollar_impact_per_share: float | None = None
    """Estimated P&L per share = current_price × (estimated_stock_return_pct / 100)."""

    methodology: str
    """Brief methodology note, e.g. 'beta-scaled from market drawdown' or
    'analyst-estimated earnings-miss reaction, no market move assumed'."""


class RiskAssessmentOutput(BaseModel):
    """Structured output produced by the risk assessment deep agent."""

    ticker: str
    """Equity ticker symbol analysed."""

    company_name: str
    """Full legal company name."""

    as_of_date: str
    """Date the assessment was produced (YYYY-MM-DD)."""

    # ── Statistical risk ──

    beta: float | None = None
    """Trailing 2-year weekly CAPM beta vs broad market (from ``compute_beta``)."""

    annualized_vol_pct: float | None = None
    """Annualised historical volatility in % (stddev of weekly returns × √52 × 100)."""

    max_drawdown_1y_pct: float | None = None
    """Maximum drawdown over the trailing 1-year window (negative %, e.g. -28.5).
    Computed by ``compute_max_drawdown`` on the last 52 weekly closes."""

    var_95_1m_pct: float | None = None
    """Parametric 95% 1-month Value at Risk as a positive percentage.
    Computed by ``compute_var_cvar``."""

    cvar_95_1m_pct: float | None = None
    """Parametric 95% 1-month Expected Shortfall (CVaR) as a positive percentage.
    Always ≥ var_95_1m_pct under the normal distribution assumption."""

    sharpe_ratio: float | None = None
    """Annualised Sharpe ratio (trailing 2-year weekly,
    computed by ``compute_sharpe_sortino``)."""

    sortino_ratio: float | None = None
    """Annualised Sortino ratio (trailing 2-year weekly,
    computed by ``compute_sharpe_sortino``)."""

    # ── Factor risk ──

    factor_loadings: FactorLoadings | None = None
    """FF5 + Momentum factor loadings from sandbox OLS regression.  None if
    fama-french data was unavailable or regression failed."""

    # ── Market-implied risk ──

    implied_volatility: ImpliedVolatilityTermStructure | None = None
    """Options-derived IV term structure and 25-delta skew.  None if options
    data was unavailable."""

    # ── Short interest / crowding ──

    short_interest: ShortInterestMetrics
    """Short interest, days-to-cover, short volume ratio,
    and crowding classification."""

    # ── Stress testing ──

    stress_scenarios: list[StressScenario] = Field(default_factory=list)
    """6 stress scenarios: 2 fixed historical analogs (GFC 2008, COVID 2020),
    3 regime-derived from market_regime.key_risks, 1 idiosyncratic from
    company_analysis.key_risks."""

    # ── Actionable outputs ──

    ex_ante_stop_level: float | None = None
    """Suggested pre-defined stop-loss price level.

    Derived from current_price × (1 − 2 × var_95_1m_pct / 100), cross-checked
    against bear stress scenario implied price.  None if current_price unavailable.
    """

    stop_methodology: str | None = None
    """Brief description of how the stop level was derived."""

    risk_signal: Literal["acceptable", "elevated", "unacceptable"]
    """Composite risk signal for downstream use.

    ``unacceptable``: var_95_1m > 20%, OR max_drawdown_1y < −60%, OR
        (crowding = 'high' AND short_interest_pct > 25%), OR worst scenario < −70%.
    ``elevated``: var_95_1m 12–20%, OR max_drawdown_1y −40% to −60%, OR
        crowding = 'high', OR worst scenario −50% to −70%.
    ``acceptable``: none of the above criteria met.
    """

    # ── Standard metadata ──

    risk_flags: list[str] = Field(default_factory=list)
    """Non-empty list of specific risk flags.

    Examples: 'beta > 2.0: high systematic risk',
    'Inverted IV term structure: elevated near-term fear',
    'Crowding risk elevated: short interest > 20% of float'.
    """

    confidence: float = Field(ge=0.0, le=1.0)
    """Overall assessment confidence 0.0–1.0.

    Start at 1.0; reduce by ~0.15 per missing primary subagent (equity-price,
    options, fama-french), ~0.10 per major data gap.
    """

    data_sources: list[DataSource] = Field(default_factory=list)
    """One entry per subagent consulted: name, what was retrieved, period covered."""

    limitations: list[str] = Field(default_factory=list)
    """Data gaps or model assumptions that reduce confidence.

    Examples: 'Options data unavailable — IV term structure omitted',
    'UMD (momentum) factor absent from fama-french data — 5-factor regression used',
    'Short interest data not available for this ticker'.
    """


# ── Subagent builder ───────────────────────────────────────────────────────────


async def _build_risk_assessment_subagents(
    config: RunnableConfig,
) -> list[CompiledSubAgent]:
    """Build the focused subagent set for risk & downside assessment.

    Return 6 data collection subagents + 1 data validation subagent.
    """
    price_agent = await create_equity_price_data_collection_agent(config)
    options_agent = await create_options_data_collection_agent(config)
    fama_french_agent = await create_fama_french_data_collection_agent(config)
    ownership_agent = await create_equity_ownership_data_collection_agent(config)
    fixed_income_agent = await create_fixed_income_data_collection_agent(config)
    macro_agent = await create_economy_macro_data_collection_agent(config)
    validation_subagent = await build_validation_subagent(config)

    return [
        CompiledSubAgent(
            name="equity-price",
            description=(
                "Retrieves historical price data: 2-year weekly OHLCV for the stock "
                "AND for the broad market benchmark (SPY / S&P 500 index) aligned to "
                "the same dates — both series are required for beta calculation. Also "
                "fetches current bid/ask quote (current_price for VaR dollar amount "
                "and stop-loss derivation) and 1-year price performance. Primary "
                "source for beta, annualized vol, max drawdown, Sharpe ratio, "
                "Sortino ratio, "
                "VaR, and CVaR computations in Block A."
            ),
            runnable=price_agent,
        ),
        CompiledSubAgent(
            name="options",
            description=(
                "Retrieves options market data: full implied volatility surface "
                "(derivatives_options_surface) across expirations and strikes; "
                "options chains for the nearest 30-day, 60-day, and 90-day expirations "
                "(derivatives_options_chains) to extract ATM IV at each tenor and "
                "25-delta put/call strikes for skew calculation. Primary source for "
                "implied_volatility term structure and put_call_skew_25d."
            ),
            runnable=options_agent,
        ),
        CompiledSubAgent(
            name="fama-french",
            description=(
                "Retrieves Fama-French academic factor data: monthly returns for the "
                "5-factor model (Mkt-RF, SMB, HML, RMW, CMA) and Momentum (UMD) "
                "covering at least 2 years to align with the stock price history. "
                "Also retrieves the risk-free rate (Rf) series included in the FF "
                "dataset for computing excess returns in the regression. Primary "
                "source for the FF5+UMD multi-factor regression in Block B. "
                "Does NOT require "
                "a stock ticker — provides market-wide factor returns."
            ),
            runnable=fama_french_agent,
        ),
        CompiledSubAgent(
            name="equity-ownership",
            description=(
                "Retrieves short interest and crowding data: latest short interest "
                "level and days-to-cover (equity_shorts_short_interest), recent "
                "daily short volume ratio (equity_shorts_short_volume), "
                "fails-to-deliver as an optional crowding confirmation signal "
                "(equity_shorts_fails_to_deliver), "
                "and share statistics for float verification "
                "(equity_ownership_share_statistics). Primary source for "
                "short_interest_pct, days_to_cover, short_volume_ratio, and the "
                "crowding_signal classification."
            ),
            runnable=ownership_agent,
        ),
        CompiledSubAgent(
            name="fixed-income",
            description=(
                "Retrieves fixed income and rates data: current 10-year Treasury "
                "yield and EFFR/SOFR as the risk-free rate input for Sharpe and "
                "Sortino ratio computation; IG credit spread (OAS) and HY credit "
                "spread for stress scenario calibration and credit risk context. "
                "Also provides yield curve context for the regime-derived stress "
                "scenarios."
            ),
            runnable=fixed_income_agent,
        ),
        CompiledSubAgent(
            name="economy-macro",
            description=(
                "Retrieves macroeconomic data for stress scenario calibration: "
                "composite leading indicator (CLI) for macro scenario direction; "
                "equity risk premium estimate if available; GDP growth context. "
                "Use primarily when market_regime upstream state is absent and "
                "stress scenarios must be "
                "derived from raw macro data rather than from market_regime.key_risks."
            ),
            runnable=macro_agent,
        ),
        validation_subagent,
    ]


# ── Agent factory ──────────────────────────────────────────────────────────────


async def create_risk_assessment_agent(
    config: RunnableConfig,
    store: BaseStore | None = None,
) -> Any:
    """Build the risk assessment deep agent.

    Create a deep agent that quantifies systematic risk (beta, FF5+UMD factor
    loadings), statistical risk (vol, max drawdown, Sharpe, Sortino, VaR, CVaR),
    market-implied tail risk (IV term structure, put/call skew), and short-interest
    crowding, then stress-tests the thesis under 6 scenarios.

    Args:
        config: Application configuration.
        store: Shared ``BaseStore`` for cross-agent tool result caching.
    """
    subagents = await _build_risk_assessment_subagents(config)
    llm = ModelConfiguration.from_runnable_config(config).get_llm()

    builder = (
        MuffinAgentBuilder(llm, name="risk_assessment")
        .with_system_prompt_template("investment/risk_assessment.jinja")
        .with_sandbox()
        .with_short_term_memory()
        .with_persistent_memory()
        .with_subagents(subagents)
        .with_response_format(AutoStrategy(schema=RiskAssessmentOutput))
        .with_store(store)
    )
    for tool in (
        compute_beta,
        compute_var_cvar,
        compute_sharpe_sortino,
        compute_max_drawdown,
    ):
        builder = builder.with_tool(tool)
    return builder.build_deep_agent()


# ── Node ───────────────────────────────────────────────────────────────────────


async def risk_assessment_node(
    state: RiskAssessmentInputState,
    config: RunnableConfig,
    *,
    store: BaseStore | None = None,
) -> dict[str, Any]:
    """Stage 8: Risk & Downside / Stress Testing.

    Quantifies idiosyncratic and systematic risk for an equity position.
    Computes CAPM beta, FF5+Momentum factor loadings, annualised volatility,
    max drawdown, Sharpe/Sortino ratios, parametric VaR/CVaR, options-implied
    tail risk (IV term structure + 25-delta skew), and short-interest crowding.
    Produces 6 stress scenarios (2 historical analogs, 3 regime-derived, 1
    idiosyncratic) and an ex-ante stop-loss level.

    Runs in **parallel** with ``forecasting_node`` (Group 2) after all Group 1
    nodes complete.  Reads ``RiskAssessmentInputState`` fields (ticker, query,
    company_analysis, market_regime) and writes ``risk_assessment`` to state.

    Its output flows into ``valuation_node`` (Group 3) to inform the discount
    rate (via CAPM cost of equity) and downside price target.
    """
    return await run_deep_agent_node(
        state=state,
        config=config,
        agent_factory=create_risk_assessment_agent,
        input_state_type=RiskAssessmentInputState,
        state_key="risk_assessment",
        error_fallback={"risk_signal": "unacceptable"},
        store=store,
    )
