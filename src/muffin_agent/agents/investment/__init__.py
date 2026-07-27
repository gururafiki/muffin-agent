"""Investment workflow stages.

The six LLM stages are exposed as **agent factories**, not node functions — each is
added to its graph as a compiled deep agent via ``add_node`` so it owns its own
``checkpoint_ns`` and its transcript / tool calls / nested subagents are readable
per-namespace. ``comparison``, ``idea_sourcing`` and ``thesis_synthesis`` remain
plain nodes.
"""

from muffin_agent.agents.investment.company_analysis import (
    CompanyAnalysisInputState,
    create_company_analysis_agent,
)
from muffin_agent.agents.investment.comparison import comparison_node
from muffin_agent.agents.investment.forecasting import (
    ForecastingInputState,
    create_forecasting_agent,
)
from muffin_agent.agents.investment.idea_sourcing import idea_sourcing_node
from muffin_agent.agents.investment.market_regime import (
    MarketRegimeInputState,
    create_market_regime_agent,
)
from muffin_agent.agents.investment.risk_assessment import (
    RiskAssessmentInputState,
    create_risk_assessment_agent,
)
from muffin_agent.agents.investment.sector_analysis import (
    SectorAnalysisInputState,
    create_sector_analysis_agent,
)
from muffin_agent.agents.investment.thesis_synthesis import thesis_synthesis_node
from muffin_agent.agents.investment.valuation import (
    ValuationInputState,
    create_valuation_agent,
)

__all__ = [
    "CompanyAnalysisInputState",
    "ForecastingInputState",
    "MarketRegimeInputState",
    "RiskAssessmentInputState",
    "SectorAnalysisInputState",
    "ValuationInputState",
    "comparison_node",
    "create_company_analysis_agent",
    "create_forecasting_agent",
    "create_market_regime_agent",
    "create_risk_assessment_agent",
    "create_sector_analysis_agent",
    "create_valuation_agent",
    "idea_sourcing_node",
    "thesis_synthesis_node",
]
