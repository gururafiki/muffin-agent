"""Investment workflow stage nodes."""

from muffin_agent.agents.investment.company_analysis import company_analysis_node
from muffin_agent.agents.investment.comparison import comparison_node
from muffin_agent.agents.investment.forecasting import forecasting_node
from muffin_agent.agents.investment.idea_sourcing import idea_sourcing_node
from muffin_agent.agents.investment.market_regime import market_regime_node
from muffin_agent.agents.investment.risk_assessment import risk_assessment_node
from muffin_agent.agents.investment.sector_analysis import sector_analysis_node
from muffin_agent.agents.investment.thesis_synthesis import thesis_synthesis_node
from muffin_agent.agents.investment.valuation import valuation_node

__all__ = [
    "company_analysis_node",
    "comparison_node",
    "forecasting_node",
    "idea_sourcing_node",
    "market_regime_node",
    "risk_assessment_node",
    "sector_analysis_node",
    "thesis_synthesis_node",
    "valuation_node",
]
