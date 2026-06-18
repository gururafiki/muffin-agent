"""Analyst layer for ``trading_decision``.

Four compiled ReAct agents that produce free-text prose reports
consumed by the Bull/Bear/Judge/Trader/PM downstream nodes:

* :func:`build_market_analyst_agent` — technical analysis (price /
  indicators / momentum).
* :func:`build_fundamentals_analyst_agent` — financial statements
  + ratios.
* :func:`build_news_analyst_agent` — company news, macro backdrop,
  insider activity.
* :func:`build_social_analyst_agent` — retail / social sentiment.

Each factory returns a ``CompiledStateGraph`` ready to be added
directly to a parent graph via ``add_node(name, agent,
input_schema=AnalystInput)`` — an explicit ``{ticker, decision_date}``
field schema, NOT ``agent.input_schema`` (a property-less ``RootModel``
that maps ``{}`` and raises at coercion). The parent state must declare
``ticker`` and ``decision_date`` (read by every analyst) plus the
analyst's output report field.
"""

from .fundamentals_analyst import (
    FundamentalsAnalystOutput,
    FundamentalsAnalystState,
    build_fundamentals_analyst_agent,
)
from .market_analyst import (
    MarketAnalystOutput,
    MarketAnalystState,
    build_market_analyst_agent,
)
from .news_analyst import (
    NewsAnalystOutput,
    NewsAnalystState,
    build_news_analyst_agent,
)
from .social_analyst import (
    SocialAnalystOutput,
    SocialAnalystState,
    build_social_analyst_agent,
)

__all__ = [
    # Factories
    "build_fundamentals_analyst_agent",
    "build_market_analyst_agent",
    "build_news_analyst_agent",
    "build_social_analyst_agent",
    # Output Pydantic models
    "FundamentalsAnalystOutput",
    "MarketAnalystOutput",
    "NewsAnalystOutput",
    "SocialAnalystOutput",
    # State schemas (extending AgentState)
    "FundamentalsAnalystState",
    "MarketAnalystState",
    "NewsAnalystState",
    "SocialAnalystState",
]
