"""Stage 3: Sector / Industry & Thematic View."""

from typing import Any, Literal

from deepagents import CompiledSubAgent
from langchain.agents.structured_output import AutoStrategy
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from ...model_config import ModelConfiguration
from ...tools.sector import (
    compute_peer_dispersion,
    compute_sector_relative_performance,
)
from ...utils.agent_builder import MuffinAgentBuilder
from ..data_collection import (
    create_discovery_screening_data_collection_agent,
    create_equity_estimates_data_collection_agent,
    create_etf_index_data_collection_agent,
    create_news_data_collection_agent,
    create_regulatory_filings_data_collection_agent,
)
from ..investment.schemas import DataSource
from ..investment.utils import run_deep_agent_node
from ..subagents import build_validation_subagent

# ── Input state schema ─────────────────────────────────────────────────────────


class SectorAnalysisInputState(TypedDict, total=False):
    """Input state schema for ``sector_analysis_node``.

    Documents which state fields the node reads.  All fields are optional;
    at least one should be present.

    Context modes:
        (a) ticker — agent calls ``etf_equity_exposure`` to derive sector/industry
        (b) sector / industry — explicit context, no ticker lookup needed
        (c) query only — thematic scan; agent infers sector focus from the mandate
    """

    ticker: str
    query: str
    sector: str
    industry: str


# ── Output schema ─────────────────────────────────────────────────────────────


class CompetitiveAssessment(BaseModel):
    """Porter's Five Forces competitive structure assessment."""

    rivalry_intensity: Literal["low", "moderate", "high", "very_high"]
    """Intensity of competition among existing players."""

    barriers_to_entry: Literal["low", "moderate", "high"]
    """Height of structural barriers (capital, patents, network effects, licensing)."""

    supplier_power: Literal["low", "moderate", "high"]
    """Bargaining power of key input suppliers."""

    buyer_power: Literal["low", "moderate", "high"]
    """Bargaining power of customers over pricing and terms."""

    threat_of_substitutes: Literal["low", "moderate", "high"]
    """Threat from alternative products or technology disruption."""

    overall_attractiveness: Literal["unattractive", "moderate", "attractive"]
    """Composite attractiveness: 'attractive' if ≥3 forces are favorable."""

    summary: str
    """2-3 sentences synthesising all five forces and their net effect on margins."""


class CyclePosition(BaseModel):
    """Sector cycle position assessment."""

    label: Literal[
        "early_expansion", "mid_expansion", "late_cycle", "contraction", "recovery"
    ]
    """Current phase of the sector business/demand cycle."""

    direction: Literal["accelerating", "stable", "decelerating"]
    """Whether the cycle is gaining momentum, plateauing, or losing steam."""

    key_indicators: str
    """2-3 data points with values and sources supporting the cycle assessment."""


class ThematicDriver(BaseModel):
    """A single secular theme affecting the sector."""

    theme: str
    """Short descriptive name, e.g. 'AI infrastructure buildout', 'GLP-1 disruption'."""

    direction: Literal["tailwind", "headwind", "neutral"]
    """Whether this theme helps, hurts, or has mixed impact on the sector."""

    time_horizon: Literal["near_term", "medium_term", "long_term"]
    """Expected horizon: near_term (≤12M), medium_term (1-3Y), long_term (>3Y)."""

    rationale: str
    """1 sentence citing a specific news headline, regulatory item, or data point."""


class SectorValuation(BaseModel):
    """Sector valuation snapshot relative to the S&P 500 and historical levels."""

    pe_ratio: float | None = None
    """Sector P/E ratio (trailing or forward); null if unavailable."""

    ev_ebitda: float | None = None
    """Sector EV/EBITDA multiple; null if unavailable."""

    pe_vs_sp500_pct: float | None = None
    """Premium (+) or discount (-) to S&P 500 P/E in percent; null if unavailable."""

    pe_vs_5y_avg_pct: float | None = None
    """Premium (+) or discount (-) to sector's own 5-year average P/E.

    Null if 5-year history is unavailable.
    """

    valuation_signal: Literal["expensive", "fairly_valued", "cheap"]
    """Summary signal.

    'expensive' if >+20% vs S&P 500 or 5Y avg; 'cheap' if <-20%; else
    'fairly_valued'.
    """


class RegulatoryBackdrop(BaseModel):
    """Regulatory and legislative environment assessment."""

    risk_level: Literal["low", "moderate", "elevated", "high"]
    """'high' = enforcement ongoing or near-term adverse legislation likely."""

    key_items: list[str]
    """Specific bills, enforcement actions, or regulatory changes identified."""

    summary: str
    """1-2 sentences on the net regulatory direction and its timing."""


class SectorViewOutput(BaseModel):
    """Structured output produced by the sector analysis deep agent."""

    sector: str
    """GICS sector name, e.g. 'Information Technology'."""

    industry: str
    """GICS industry or sub-industry name, e.g. 'Semiconductors'."""

    cycle_position: CyclePosition
    competitive_assessment: CompetitiveAssessment
    thematic_drivers: list[ThematicDriver]
    """3–5 secular themes; each scored for direction and time horizon."""

    sector_valuation: SectorValuation
    regulatory_backdrop: RegulatoryBackdrop

    peer_tickers: list[str]
    """5–10 comparable ticker symbols identified by discovery-screening."""

    alpha_opportunity: Literal["high", "moderate", "low"]
    """Dispersion-based alpha signal: 'high' if peer return std dev >25%."""

    alpha_rationale: str
    """1-2 sentences citing peer dispersion metric and spread between winners/losers."""

    sector_signal: Literal["favorable", "neutral", "cautious"]
    """Composite sector attractiveness: 'favorable' = good cycle + alpha + competitive
    dynamics; 'cautious' = late cycle, low dispersion, or elevated regulatory risk."""

    sector_summary: str
    """3-4 sentence narrative covering cycle stage, competitive dynamics, thematic
    backdrop, and alpha outlook."""

    confidence: float = Field(ge=0.0, le=1.0)
    """Overall assessment confidence 0.0–1.0.

    Start at 1.0; reduce by ~0.15 per missing primary subagent result,
    ~0.10 per major data gap within a source."""

    data_sources: list[DataSource] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    """Data gaps or uncertainties that reduce confidence in the assessment."""


# ── Subagent builder ──────────────────────────────────────────────────────────


async def _build_sector_subagents(config: RunnableConfig) -> list[CompiledSubAgent]:
    """Build the sector-focused subagent set for sector/industry analysis.

    Return 4 data collection subagents + 1 data validation subagent, covering
    ETF/index data, peer comparisons, thematic news, and regulatory signals.
    Excludes macro and company-specific subagents irrelevant to sector-level
    competitive and structural analysis.
    """
    etf_index_agent = await create_etf_index_data_collection_agent(config)
    discovery_screening_agent = await create_discovery_screening_data_collection_agent(
        config
    )
    estimates_agent = await create_equity_estimates_data_collection_agent(config)
    news_agent = await create_news_data_collection_agent(config)
    regulatory_filings_agent = await create_regulatory_filings_data_collection_agent(
        config
    )
    validation_subagent = await build_validation_subagent(config)

    return [
        CompiledSubAgent(
            name="etf-index",
            description=(
                "Retrieves ETF and index data for sector analysis: sector ETF "
                "price performance (1M, 3M, 12M), `etf_equity_exposure` to "
                "identify sector/industry/ETF from a ticker, S&P 500 multiples "
                "(P/E, EV/EBITDA) from `index_sp500_multiples`, index sector "
                "weights, and ETF holdings/info. Primary source for sector "
                "identification, relative performance, and valuation benchmarks."
            ),
            runnable=etf_index_agent,
        ),
        CompiledSubAgent(
            name="discovery-screening",
            description=(
                "Retrieves peer comparison and screening data: `equity_compare_peers` "
                "for comparable ticker lists, `equity_compare_groups` for "
                "sector/industry group valuation stats (median P/E, EV/EBITDA), "
                "`equity_profile` for company descriptions and market caps, and "
                "`equity_screener` for filtered peer lists. Primary source for "
                "peer tickers, dispersion inputs, and sector valuation multiples."
            ),
            runnable=discovery_screening_agent,
        ),
        CompiledSubAgent(
            name="equity-estimates",
            description=(
                "Retrieves analyst estimates data for sector cycle assessment: "
                "consensus EPS/revenue estimates and revision history for key "
                "sector peers. Use `equity_estimates_historical` to compute "
                "earnings revision breadth (% of companies with upward vs "
                "downward revisions) as a leading cycle position indicator."
            ),
            runnable=estimates_agent,
        ),
        CompiledSubAgent(
            name="news",
            description=(
                "Retrieves news and sentiment data: `news_world` for sector-level "
                "and thematic macro headlines (capex cycles, technology transitions, "
                "geopolitical impacts, regulatory shifts), `news_company` for "
                "key competitor news. Primary source for thematic driver "
                "identification and competitive dynamics signals."
            ),
            runnable=news_agent,
        ),
        CompiledSubAgent(
            name="regulatory-filings",
            description=(
                "Retrieves regulatory and legislative data: `uscongress_bills` "
                "for pending legislation affecting the sector, "
                "`regulators_sec_rss_litigation` for SEC enforcement actions and "
                "litigation patterns in the sector. Primary source for regulatory "
                "risk level and specific legislative/enforcement items."
            ),
            runnable=regulatory_filings_agent,
        ),
        validation_subagent,
    ]


# ── Agent factory ─────────────────────────────────────────────────────────────


async def create_sector_analysis_agent(
    config: RunnableConfig,
    store: BaseStore | None = None,
):
    """Build the sector analysis deep agent.

    Create a deep agent that identifies the sector/industry for a ticker (or
    uses explicit sector context), collects ETF performance, peer comparison,
    thematic news, and regulatory data, then scores the sector across six
    dimensions: cycle position, Porter's Five Forces competitive structure,
    thematic drivers, relative valuation, regulatory backdrop, and alpha
    opportunity (peer dispersion).

    ``get_backend`` discovers or creates a sandbox container per conversation
    by ``thread_id`` metadata for Python computations (sector relative
    performance, valuation premium/discount, peer return dispersion).

    ``response_format=AutoStrategy(SectorViewOutput)`` instructs the agent
    to call a structured output tool as its final act, returning a validated
    ``SectorViewOutput`` instance in ``result["structured_response"]``
    instead of free-form text.
    """
    subagents = await _build_sector_subagents(config)
    llm = ModelConfiguration.from_runnable_config(config).get_llm()

    builder = (
        MuffinAgentBuilder(llm, name="sector_analysis")
        .with_system_prompt_template("investment/sector_analysis.jinja")
        .with_sandbox()
        .with_short_term_memory()
        .with_persistent_memory()
        .with_subagents(subagents)
        .with_response_format(AutoStrategy(schema=SectorViewOutput))
        .with_store(store)
    )
    for tool in (compute_sector_relative_performance, compute_peer_dispersion):
        builder = builder.with_tool(tool)
    return builder.build_deep_agent()


# ── Node ──────────────────────────────────────────────────────────────────────


async def sector_analysis_node(
    state: SectorAnalysisInputState,
    config: RunnableConfig,
    *,
    store: BaseStore | None = None,
) -> dict[str, Any]:
    """Stage 3: Sector / Industry & Thematic View.

    Assesses the attractiveness of the ticker's sector and industry: cycle
    position, Porter's Five Forces competitive structure, thematic tailwinds
    and headwinds, sector-relative valuation, regulatory/legislative backdrop,
    and alpha opportunity (peer return dispersion).

    Runs in **parallel** with ``market_regime_node`` and
    ``company_analysis_node`` (Group 1).  Its output flows into
    ``valuation_node`` (Group 3) for peer-relative valuation benchmarks and
    into ``thesis_synthesis_node`` for the sector attractiveness narrative.

    In ``screening_graph`` this node runs **once** on the outer graph before
    the per-ticker fan-out when all candidates share a sector; for multi-sector
    screens the outer graph may run it once per sector or skip it when each
    ticker worker handles sector identification independently.

    Input state fields (``SectorAnalysisInputState``):
        (a) ticker — agent calls ``etf_equity_exposure`` to derive sector/industry
        (b) sector / industry — passed explicitly (screening graph pre-fanout)
        (c) query only — thematic scan; agent infers sector from the mandate

    Outputs (state update):
        sector_view: ``SectorViewOutput.model_dump()`` dict, or an error dict
        ``{"sector": "unknown", "error": ..., "raw_output": ...}`` if the
        agent fails to return structured output.
    """
    return await run_deep_agent_node(
        state=state,
        config=config,
        agent_factory=create_sector_analysis_agent,
        input_state_type=SectorAnalysisInputState,
        state_key="sector_view",
        error_fallback={"sector": "unknown"},
        store=store,
    )
