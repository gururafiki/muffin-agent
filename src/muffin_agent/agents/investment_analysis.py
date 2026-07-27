"""Per-ticker investment analysis graph.

Entry point for ``muffin analyze <TICKER>``.  Accepts a single ticker and
investment mandate, runs a 7-stage analysis pipeline, and returns a completed
``TickerAnalysisState`` with the investment thesis.

Parallel execution groups
-------------------------
Group 1 — start simultaneously from START (no inter-dependencies):

    market_regime ──────────────────────────────────────────────┐
    sector_analysis ────────────────────────────────────────────┤ (barrier)
    company_analysis ───────────────────────────────────────────┘
                                                                ↓
Group 2 — start after Group 1 barrier, run in parallel:

    forecasting ────────────────────────────────────────────────┐
    risk_assessment ────────────────────────────────────────────┘ (barrier)
                                                                ↓
Group 3 — sequential:

    valuation ──────────────────────────────────────────────────→ thesis_synthesis → END

LangGraph fires a node only when all its incoming edges have data, so barrier
synchronisation is implicit — no extra code required.

Every stage is a **compiled deep agent added directly via ``add_node``**, so each owns
its own ``checkpoint_ns`` and its transcript, tool calls and nested subagents are
readable per-namespace. Errors propagate (``RetryPolicy`` per node + the model-fallback
chain); the previous ``run_deep_agent_node`` wrapper swallowed every failure into an
error dict, which hid a live config-type bug for as long as it existed.
"""

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import RetryPolicy

from .investment import thesis_synthesis_node
from .investment.company_analysis import (
    CompanyAnalysisInputState,
    create_company_analysis_agent,
)
from .investment.forecasting import ForecastingInputState, create_forecasting_agent
from .investment.market_regime import (
    MarketRegimeInputState,
    create_market_regime_agent,
)
from .investment.risk_assessment import (
    RiskAssessmentInputState,
    create_risk_assessment_agent,
)
from .investment.sector_analysis import (
    SectorAnalysisInputState,
    create_sector_analysis_agent,
)
from .investment.state import TickerAnalysisState
from .investment.valuation import ValuationInputState, create_valuation_agent

_AGENT_RETRY = RetryPolicy(max_attempts=2)


async def build_investment_analysis_graph(
    config: RunnableConfig | None = None,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> CompiledStateGraph:
    """Build and compile the per-ticker investment analysis graph.

    Async because each stage's compiled deep agent (and its subagents) is built at
    graph-construction time, amortising agent construction out of the per-request hot
    path — the same shape as ``build_criteria_analysis_graph``.
    """
    cfg: RunnableConfig = config or {}
    market_regime_agent = await create_market_regime_agent(cfg, store=store)
    sector_analysis_agent = await create_sector_analysis_agent(cfg, store=store)
    company_analysis_agent = await create_company_analysis_agent(cfg, store=store)
    forecasting_agent = await create_forecasting_agent(cfg, store=store)
    risk_assessment_agent = await create_risk_assessment_agent(cfg, store=store)
    valuation_agent = await create_valuation_agent(cfg, store=store)

    graph: StateGraph = StateGraph(TickerAnalysisState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    graph.add_node(
        "market_regime",
        market_regime_agent,
        input_schema=MarketRegimeInputState,
        retry_policy=_AGENT_RETRY,
    )
    graph.add_node(
        "sector_analysis",
        sector_analysis_agent,
        input_schema=SectorAnalysisInputState,
        retry_policy=_AGENT_RETRY,
    )
    graph.add_node(
        "company_analysis",
        company_analysis_agent,
        input_schema=CompanyAnalysisInputState,
        retry_policy=_AGENT_RETRY,
    )
    graph.add_node(
        "forecasting",
        forecasting_agent,
        input_schema=ForecastingInputState,
        retry_policy=_AGENT_RETRY,
    )
    graph.add_node(
        "risk_assessment",
        risk_assessment_agent,
        input_schema=RiskAssessmentInputState,
        retry_policy=_AGENT_RETRY,
    )
    graph.add_node(
        "valuation",
        valuation_agent,
        input_schema=ValuationInputState,
        retry_policy=_AGENT_RETRY,
    )
    graph.add_node("thesis_synthesis", thesis_synthesis_node)

    # ── Group 1: all start in parallel ───────────────────────────────────────
    graph.add_edge(START, "market_regime")
    graph.add_edge(START, "sector_analysis")
    graph.add_edge(START, "company_analysis")

    # ── Group 1 → Group 2 barrier ─────────────────────────────────────────────
    # forecasting waits for all three Group 1 nodes
    graph.add_edge("market_regime", "forecasting")
    graph.add_edge("sector_analysis", "forecasting")
    graph.add_edge("company_analysis", "forecasting")

    # risk_assessment waits for company_analysis + market_regime
    graph.add_edge("company_analysis", "risk_assessment")
    graph.add_edge("market_regime", "risk_assessment")

    # ── Group 2 → valuation barrier ───────────────────────────────────────────
    graph.add_edge("forecasting", "valuation")
    graph.add_edge("risk_assessment", "valuation")
    graph.add_edge("sector_analysis", "valuation")  # peer multiples

    # ── valuation → thesis → END ──────────────────────────────────────────────
    graph.add_edge("valuation", "thesis_synthesis")
    graph.add_edge("thesis_synthesis", END)

    return graph.compile(checkpointer=checkpointer, store=store)
