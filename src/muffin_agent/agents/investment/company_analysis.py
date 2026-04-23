"""Stage 4-5: Company Analysis — Business Quality & Fundamental Deep Dive."""

from typing import Any, Literal

from deepagents import CompiledSubAgent
from langchain.agents.structured_output import AutoStrategy
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from ...model_config import ModelConfiguration
from ...tools.credit_risk import (
    compute_altman_z_score,
    compute_interest_coverage,
    compute_net_debt_to_ebitda,
)
from ...tools.profitability import (
    compute_fcf_conversion,
    compute_revenue_cagr,
    compute_roic,
)
from ...utils.agent_builder import MuffinAgentBuilder
from ..data_collection import (
    create_discovery_screening_data_collection_agent,
    create_equity_fundamentals_data_collection_agent,
    create_equity_ownership_data_collection_agent,
    create_news_data_collection_agent,
    create_regulatory_filings_data_collection_agent,
)
from ..subagents import build_validation_subagent
from .schemas import DataSource
from .utils import run_deep_agent_node

# ── Input state schema ─────────────────────────────────────────────────────────


class CompanyAnalysisInputState(TypedDict, total=False):
    """Input state schema for ``company_analysis_node``.

    Documents which state fields the node reads.  All fields optional; both
    should be present for a full company-specific analysis.

    Context modes:
        (a) ticker + query — standard per-ticker analysis (Group 1 parallel run)
        (b) ticker only    — minimal mode, no investment-mandate context
        (c) query only     — thematic quality screen without a specific ticker
    """

    ticker: str
    query: str


# ── Output schema ─────────────────────────────────────────────────────────────


class MoatAssessment(BaseModel):
    """Competitive moat assessment for the company."""

    width: Literal["wide", "narrow", "none", "negative"]
    """Moat width: 'wide' = durable >15% ROIC premium vs peers for 5+ years;
    'narrow' = ROIC above WACC but less durable; 'none' = ROIC ≈ WACC;
    'negative' = ROIC < WACC, structural value destruction."""

    sources: list[
        Literal[
            "network_effects",
            "cost_advantage",
            "switching_costs",
            "intangible_assets",
            "efficient_scale",
            "none",
        ]
    ]
    """Primary moat sources identified; use ['none'] if no moat exists."""

    trend: Literal["expanding", "stable", "eroding"]
    """Whether the moat is widening, holding, or narrowing over the last 3 years."""

    confidence: float
    """Analyst confidence in the moat assessment (0.0 = no data; 1.0 = very high)."""

    rationale: str
    """2-3 sentences citing specific data: ROIC trend, peer ROIC premium,
    and one concrete structural source (e.g. patent count, switching cost evidence)."""


class ManagementQuality(BaseModel):
    """Management quality and capital allocation assessment."""

    track_record: Literal["strong", "adequate", "weak"]
    """'strong' = consistent EPS/FCF growth, beat guidance ≥70% of quarters;
    'weak' = serial misses, credibility issues, or recent major write-downs."""

    capital_allocation_quality: Literal["excellent", "good", "fair", "poor"]
    """'excellent' = ROIC-accretive M&A, buybacks below intrinsic value,
    disciplined capex; 'poor' = value-destructive acquisitions, excessive
    dilution, or trophy spending."""

    insider_alignment: Literal["high", "moderate", "low"]
    """'high' = insiders own ≥3% of shares (non-founder) or ≥10% (founder-led),
    recent open-market purchases; 'low' = heavy insider selling or minimal ownership."""

    key_concerns: list[str]
    """Specific governance or management red flags (empty list if none)."""

    summary: str
    """2-3 sentences on CEO tenure, compensation structure, and standout capital
    allocation decisions (M&A, buybacks, dividend policy) with specific examples."""


class FinancialQuality(BaseModel):
    """Financial quality snapshot — latest fiscal year or trailing twelve months."""

    revenue_cagr_3y_pct: float | None = None
    """3-year compound annual revenue growth rate in %; null if history unavailable."""

    gross_margin_pct: float | None = None
    """Gross profit / revenue (%); null if unavailable."""

    operating_margin_pct: float | None = None
    """EBIT / revenue (%); null if unavailable."""

    net_margin_pct: float | None = None
    """Net income / revenue (%); null if unavailable."""

    roic_pct: float | None = None
    """Return on invested capital: NOPAT / (equity + debt - cash) (%);
    sandbox-computed."""

    roe_pct: float | None = None
    """Return on equity: net income / total equity (%); null if unavailable."""

    fcf_conversion_pct: float | None = None
    """Free cash flow / net income (%); sandbox-computed. >80% = high quality."""

    net_debt_to_ebitda: float | None = None
    """(Total debt - cash) / EBITDA; sandbox-computed.

    >6x triggers 'distressed' flag.
    """

    interest_coverage: float | None = None
    """EBIT / interest expense; sandbox-computed. <2x is high risk."""

    quality_signal: Literal["high", "adequate", "low", "distressed"]
    """'high': ROIC>15%, FCF conversion>80%, net_debt/EBITDA<2x;
    'distressed': ROIC<WACC (~8%), negative FCF, or net_debt/EBITDA>6x."""

    trend: Literal["improving", "stable", "deteriorating"]
    """3-year direction of operating margin and ROIC: 'improving' = ≥2pp expansion."""


class FinancialHistory(BaseModel):
    """Up to 10-year financial time series for downstream forecasting_node modeling."""

    years: list[int]
    """Fiscal years in ascending order, e.g. [2014, ..., 2023]."""

    revenue: list[float | None]
    """Annual revenue in reporting currency (same order as years)."""

    gross_profit: list[float | None]
    """Annual gross profit (same order as years)."""

    ebit: list[float | None]
    """Annual operating income / EBIT (same order as years)."""

    ebitda: list[float | None]
    """Annual EBITDA = EBIT + D&A (same order as years).

    Computed in sandbox from income statement EBIT plus depreciation &
    amortisation from the cash flow statement.  Used by ``forecasting_node``
    for EBITDA margin calibration."""

    net_income: list[float | None]
    """Annual net income (same order as years)."""

    fcf: list[float | None]
    """Annual free cash flow = operating cash flow - capex (same order as years)."""

    capex: list[float | None]
    """Annual capital expenditures (positive values; same order as years)."""

    total_debt: list[float | None]
    """Total debt (short + long-term) at fiscal year-end (same order as years)."""

    cash_and_equivalents: list[float | None]
    """Cash and short-term investments at fiscal year-end (same order as years)."""

    working_capital: list[float | None]
    """Net working capital = current assets - current liabilities (same order as years).

    Used by ``forecasting_node`` Block F for NWC/revenue ratio calibration."""

    total_assets: list[float | None]
    """Total assets at fiscal year-end (same order as years)."""

    shareholders_equity: list[float | None]
    """Total shareholders' equity at fiscal year-end (same order as years)."""

    currency: str
    """Reporting currency ISO code, e.g. 'USD', 'EUR'."""

    quality_narrative: str
    """3-4 sentences synthesising the key financial trends visible in the time series:
    revenue growth trajectory, margin evolution, FCF conversion quality, and
    leverage direction.  Used by forecasting_node as qualitative context."""


class CompanyAnalysisOutput(BaseModel):
    """Structured output produced by the company analysis deep agent."""

    ticker: str
    """Equity ticker symbol analysed."""

    company_name: str
    """Full company name."""

    business_description: str
    """2-3 sentences: what the company does, primary revenue sources, and
    geographic footprint.  Based on MD&A / 10-K business overview."""

    moat_assessment: MoatAssessment

    management_quality: ManagementQuality

    esg_flags: list[str]
    """Material ESG issues identified (e.g. 'Active SEC litigation on emissions
    reporting', 'High employee turnover flagged in proxy').  Empty if none."""

    esg_signal: Literal["green", "amber", "red"]
    """'green': ESG score ≥60 (Refinitiv/MSCI), no material controversies;
    'red': ESG score <40 OR active mandate-exclusion-level controversy."""

    financial_quality: FinancialQuality

    capital_allocation_summary: str
    """2-3 sentences covering: buyback history and yield, dividend policy,
    largest M&A deals (value, strategic rationale, integration outcome)."""

    key_risks: list[str]
    """3-5 company-specific (idiosyncratic) risks not captured by sector/macro
    nodes.  Examples: customer concentration, patent cliff, regulatory overhang."""

    financial_history: FinancialHistory

    company_signal: Literal["pass", "watch", "fail"]
    """Triage gate:
    'pass'  — meets all Step 4 must-haves (intelligible business, moat ≥ 'none',
              management ≥ 'adequate', ESG ≠ 'red', financials ≠ 'distressed');
    'watch' — borderline on one dimension; acceptable with extra monitoring;
    'fail'  — blocks further analysis: negative moat + declining financials,
              critical governance failure, ESG exclusion trigger, or distressed
              balance sheet without credible turnaround catalyst."""

    quality_summary: str
    """3-4 sentence narrative covering: business quality verdict, moat source and
    durability, key financial quality signal, and the triage gate outcome with
    the primary reason."""

    confidence: float = Field(ge=0.0, le=1.0)
    """Overall assessment confidence 0.0–1.0.

    Start at 1.0; reduce by ~0.15 per missing primary subagent result,
    ~0.10 per major data gap within a source."""

    data_sources: list[DataSource] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    """Data gaps (e.g. ESG score unavailable, <5 years of financial history)
    that reduce confidence in one or more assessment dimensions."""


# ── Subagent builder ──────────────────────────────────────────────────────────


async def _build_company_analysis_subagents(
    config: RunnableConfig,
) -> list[CompiledSubAgent]:
    """Build the company-focused subagent set for business quality analysis.

    Return 5 data collection subagents + 1 data validation subagent covering
    all financial statements and ratios, management and ownership data, SEC
    filings, recent news, and peer benchmarks for moat comparison.  Excludes
    macro, rates, ETF, options, and Fama-French subagents irrelevant to
    company-specific fundamental analysis.
    """
    fundamentals_agent = await create_equity_fundamentals_data_collection_agent(config)
    ownership_agent = await create_equity_ownership_data_collection_agent(config)
    regulatory_filings_agent = await create_regulatory_filings_data_collection_agent(
        config
    )
    news_agent = await create_news_data_collection_agent(config)
    discovery_screening_agent = await create_discovery_screening_data_collection_agent(
        config
    )
    validation_subagent = await build_validation_subagent(config)

    return [
        CompiledSubAgent(
            name="equity-fundamentals",
            description=(
                "Primary source for all company financial data. Retrieves: "
                "income statement (`equity_fundamental_income`), balance sheet "
                "(`equity_fundamental_balance`), cash flow statement "
                "(`equity_fundamental_cash`), key ratios "
                "(`equity_fundamental_ratios`, `equity_fundamental_metrics`), "
                "ESG score (`equity_fundamental_esg_score`), management roster "
                "(`equity_fundamental_management`), management compensation "
                "(`equity_fundamental_management_compensation`), MD&A section "
                "(`equity_fundamental_management_discussion_analysis`), "
                "earnings call transcripts (`equity_fundamental_transcript`), "
                "revenue by segment (`equity_fundamental_revenue_per_segment`) "
                "and geography (`equity_fundamental_revenue_per_geography`), "
                "historical EPS (`equity_fundamental_historical_eps`), and "
                "dividend yield (`equity_fundamental_trailing_dividend_yield`). "
                "Use for all financial quality, moat ROIC, and ESG dimensions."
            ),
            runnable=fundamentals_agent,
        ),
        CompiledSubAgent(
            name="equity-ownership",
            description=(
                "Retrieves ownership and insider alignment data: major holders "
                "(`equity_ownership_major_holders`), insider trading activity "
                "(`equity_ownership_insider_trading`), institutional ownership "
                "breakdown (`equity_ownership_institutional`), 13F filings "
                "(`equity_ownership_form_13f`), and share statistics "
                "(`equity_ownership_share_statistics`). Primary source for "
                "insider alignment, institutional conviction signals, and "
                "share-count trends (dilution or buyback evidence)."
            ),
            runnable=ownership_agent,
        ),
        CompiledSubAgent(
            name="regulatory-filings",
            description=(
                "Retrieves SEC filing content for deep-read analysis: "
                "filing headers list (`regulators_sec_filing_headers`) to "
                "identify recent 10-K, 10-Q, 8-K filings; full-text filing "
                "content (`regulators_sec_htm_file`) for MD&A risk factors, "
                "related-party transactions, going-concern language, and "
                "legal proceedings; symbol-to-CIK mapping "
                "(`regulators_sec_symbol_map`). Primary source for governance "
                "red flags, undisclosed litigation, and off-balance-sheet risks."
            ),
            runnable=regulatory_filings_agent,
        ),
        CompiledSubAgent(
            name="news",
            description=(
                "Retrieves recent company-specific news (`news_company`): "
                "ESG controversies, management changes, activist investor "
                "activity, product recalls, regulatory actions, earnings "
                "surprises, and strategic announcements. Primary source for "
                "ESG signal calibration and management credibility events "
                "within the last 12 months."
            ),
            runnable=news_agent,
        ),
        CompiledSubAgent(
            name="discovery-screening",
            description=(
                "Retrieves peer group data for moat benchmarking: "
                "`equity_compare_peers` for 5-10 comparable tickers, "
                "`equity_compare_groups` for sector/industry group median "
                "ROIC, ROE, gross margin, and valuation multiples, "
                "`equity_profile` for company description and market cap. "
                "Primary source for the ROIC peer premium calculation that "
                "anchors moat width assessment."
            ),
            runnable=discovery_screening_agent,
        ),
        validation_subagent,
    ]


# ── Agent factory ─────────────────────────────────────────────────────────────


async def create_company_analysis_agent(
    config: RunnableConfig,
    store: BaseStore | None = None,
):
    """Build the company analysis deep agent.

    Create a deep agent that assesses business quality across four dimensions
    (moat, management/capital allocation, ESG/governance, financial quality),
    builds a 5-year financial history time series, and produces a triage gate
    signal (pass/watch/fail) for the investment process Step 4-5 gate.

    ``get_backend`` discovers or creates a sandbox container per conversation
    by ``thread_id`` metadata for Python computations (ROIC, FCF conversion,
    net debt/EBITDA, revenue CAGR, interest coverage, peer ROIC premium).

    ``response_format=AutoStrategy(CompanyAnalysisOutput)`` instructs the
    agent to call a structured output tool as its final act, returning a
    validated ``CompanyAnalysisOutput`` instance in
    ``result["structured_response"]`` instead of free-form text.
    """
    subagents = await _build_company_analysis_subagents(config)
    llm = ModelConfiguration.from_runnable_config(config).get_llm()

    builder = (
        MuffinAgentBuilder(llm, name="company_analysis")
        .with_system_prompt_template("investment/company_analysis.jinja")
        .with_sandbox()
        .with_short_term_memory()
        .with_persistent_memory()
        .with_subagents(subagents)
        .with_response_format(AutoStrategy(schema=CompanyAnalysisOutput))
        .with_store(store)
    )
    for tool in (
        compute_roic,
        compute_fcf_conversion,
        compute_net_debt_to_ebitda,
        compute_interest_coverage,
        compute_revenue_cagr,
        compute_altman_z_score,
    ):
        builder = builder.with_tool(tool)
    return builder.build_deep_agent()


# ── Node ──────────────────────────────────────────────────────────────────────


async def company_analysis_node(
    state: CompanyAnalysisInputState,
    config: RunnableConfig,
    *,
    store: BaseStore | None = None,
) -> dict[str, Any]:
    """Stage 4-5: Company Analysis — Business Quality & Fundamental Deep Dive.

    Evaluates the quality of the business across four dimensions: competitive
    moat (width, sources, ROIC peer premium), management quality and capital
    allocation discipline, ESG and governance triage, and historical financial
    quality (margins, ROIC, FCF conversion, leverage).  Also constructs a
    5-year financial history time series for use by ``forecasting_node``.

    Runs in **parallel** with ``market_regime_node`` and
    ``sector_analysis_node`` (Group 1).  Its output is the primary input for
    both ``forecasting_node`` and ``risk_assessment_node`` (Group 2).

    Input state fields (``CompanyAnalysisInputState``):
        ticker — equity ticker symbol; should be present for company analysis
        query  — investment mandate context used to focus the analysis

    Outputs (state update):
        company_analysis: ``CompanyAnalysisOutput.model_dump()`` dict, or an
        error dict ``{"error": ..., "raw_output": ...}`` if the agent fails
        to return structured output.
    """
    return await run_deep_agent_node(
        state=state,
        config=config,
        agent_factory=create_company_analysis_agent,
        input_state_type=CompanyAnalysisInputState,
        state_key="company_analysis",
        store=store,
    )
