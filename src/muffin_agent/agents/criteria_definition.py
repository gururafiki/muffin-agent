"""Criteria definition agent.

Deep agent that classifies a ticker (sector, market type, growth vs value)
and loads the matching valuation skill(s) to produce sector-specific
evaluation criteria with target ranges and methodology guidance.
"""

from typing import Literal, NotRequired

from deepagents import CompiledSubAgent, create_deep_agent
from langchain.agents import AgentState
from langchain.agents.structured_output import AutoStrategy
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from ..middlewares import SkillFilterMiddleware, ToolResultCacheMiddleware
from ..model_config import ModelConfiguration
from ..prompts import render_template
from ..utils.backends import get_skills_backend
from .data_collection import (
    create_discovery_screening_data_collection_agent,
    create_economy_macro_data_collection_agent,
    create_equity_fundamentals_data_collection_agent,
    create_etf_index_data_collection_agent,
)
from .investment.schemas import DataSource
from .subagents import build_validation_subagent

# ── Classification schema ────────────────────────────────────────────────────


class TickerClassification(AgentState):
    """Flat classification state for valuation skill filtering.

    Extends ``AgentState`` so the framework merges these fields into the
    agent's state.  ``SkillFilterMiddleware[TickerClassification]`` derives
    category keys automatically from the extra fields.
    """

    sector: NotRequired[str]
    sub_sector: NotRequired[str]
    market: NotRequired[str]
    stock_type: NotRequired[str]


# ── Output schema ─────────────────────────────────────────────────────────────


class ValuationCriterion(BaseModel):
    """A single valuation criterion with target range and guidance."""

    name: str
    """Metric name, e.g. 'Price-to-Book Ratio'."""

    target_range: str
    """Expected range, e.g. '0.8-2.0x'."""

    weight: float
    """Relative importance 0.0-1.0; weights across criteria should sum to ~1.0."""

    assessment_guidance: str
    """What constitutes strong vs weak performance on this metric."""

    data_requirements: list[str]
    """Which data sources or subagents are needed to evaluate this criterion."""


class CriteriaDefinitionOutput(BaseModel):
    """Structured output produced by the criteria definition deep agent."""

    ticker: str
    """Equity ticker symbol that was classified."""

    sector: str
    """Identified sector, e.g. 'Financial Services - Banking'."""

    market_type: Literal["developed", "emerging"]
    """Developed or emerging market classification."""

    stock_type: Literal["value", "growth"]
    """Value or growth stock classification."""

    classification_rationale: str
    """2-4 sentences explaining why this classification was chosen."""

    primary_valuation_method: str
    """Primary valuation approach, e.g. 'P/B + P/E Dual-metric'."""

    criteria: list[ValuationCriterion]
    """5-8 valuation criteria with targets and guidance."""

    screening_questions: list[str]
    """Critical questions to answer before committing to a valuation."""

    valuation_errors_to_avoid: list[str]
    """Sector-specific pitfalls to watch for."""

    confidence: float
    """0.0-1.0 reflecting classification certainty."""

    data_sources: list[DataSource] = Field(default_factory=list)
    """Data sources used for classification."""

    limitations: list[str] = Field(default_factory=list)
    """Data gaps or uncertainties."""


# ── Subagent builder ──────────────────────────────────────────────────────────


async def _build_criteria_subagents(
    config: RunnableConfig,
) -> list[CompiledSubAgent]:
    """Build the classification-focused subagents.

    Return 4 data collection subagents + 1 data validation subagent covering
    the data needed to classify a ticker by sector, market type, and stock type.
    """
    etf_index_agent = await create_etf_index_data_collection_agent(config)
    equity_fundamentals_agent = (
        await create_equity_fundamentals_data_collection_agent(config)
    )
    discovery_screening_agent = (
        await create_discovery_screening_data_collection_agent(config)
    )
    economy_macro_agent = await create_economy_macro_data_collection_agent(config)
    validation_subagent = await build_validation_subagent(config)

    return [
        CompiledSubAgent(
            name="etf-index",
            description=(
                "Retrieves ETF and index data. Call `etf_equity_exposure` to "
                "determine the ticker's sector, industry, and style classification "
                "(growth vs value). Also provides sector ETF performance and "
                "index snapshots for market context."
            ),
            runnable=etf_index_agent,
        ),
        CompiledSubAgent(
            name="equity-fundamentals",
            description=(
                "Retrieves fundamental financial data: ROE, ROA, operating margin, "
                "revenue growth rate, P/E, P/B, FCF yield, ROIC, debt ratios. "
                "Use to classify the stock as value (high margins, low growth, "
                "cheap multiples) or growth (high revenue growth, expanding margins, "
                "premium valuation)."
            ),
            runnable=equity_fundamentals_agent,
        ),
        CompiledSubAgent(
            name="discovery-screening",
            description=(
                "Retrieves peer group and sector screening data. Use to identify "
                "comparable companies, sector median valuations, and where the "
                "ticker sits relative to peers. Helps confirm sector classification."
            ),
            runnable=discovery_screening_agent,
        ),
        CompiledSubAgent(
            name="economy-macro",
            description=(
                "Retrieves macroeconomic context: GDP growth, interest rates, "
                "inflation data. Use to determine if the ticker operates in a "
                "developed or emerging market context based on its primary "
                "revenue geography."
            ),
            runnable=economy_macro_agent,
        ),
        validation_subagent,
    ]


# ── Agent factory ─────────────────────────────────────────────────────────────


async def create_criteria_definition_agent(
    config: RunnableConfig,
    store: BaseStore | None = None,
):
    """Build the criteria definition deep agent.

    Create a deep agent that classifies a ticker by sector, market type
    (developed/emerging), and stock type (value/growth), then loads the
    matching valuation skill(s) to produce sector-specific evaluation
    criteria.

    Uses ``get_skills_backend`` to route ``/skills/`` reads to the local
    filesystem while code execution goes through the sandbox.

    ``response_format=AutoStrategy(CriteriaDefinitionOutput)`` ensures
    the agent returns a validated Pydantic model.
    """
    subagents = await _build_criteria_subagents(config)
    prompt = render_template("criteria_definition.jinja")
    model_config = ModelConfiguration.from_runnable_config(config)
    llm = model_config.get_llm()

    return create_deep_agent(
        model=llm,
        system_prompt=prompt,
        subagents=subagents,
        backend=get_skills_backend,
        skills=["/skills/valuation/"],
        store=store,
        middleware=[
            ToolResultCacheMiddleware(),
            SkillFilterMiddleware[TickerClassification](),
        ],
        response_format=AutoStrategy(schema=CriteriaDefinitionOutput),
    )
