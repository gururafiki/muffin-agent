"""Pipeline stage nodes — one async node function per investment stage."""

from muffin_agent.pipeline.stages.company_analysis import company_analysis_node
from muffin_agent.pipeline.stages.comparison import comparison_node
from muffin_agent.pipeline.stages.forecasting import forecasting_node
from muffin_agent.pipeline.stages.idea_sourcing import idea_sourcing_node
from muffin_agent.pipeline.stages.market_regime import market_regime_node
from muffin_agent.pipeline.stages.risk_assessment import risk_assessment_node
from muffin_agent.pipeline.stages.sector_analysis import sector_analysis_node
from muffin_agent.pipeline.stages.thesis_synthesis import thesis_synthesis_node
from muffin_agent.pipeline.stages.valuation import valuation_node

__all__ = [
    "comparison_node",
    "company_analysis_node",
    "forecasting_node",
    "idea_sourcing_node",
    "market_regime_node",
    "risk_assessment_node",
    "sector_analysis_node",
    "thesis_synthesis_node",
    "valuation_node",
]
