"""Data collection agents."""

from .economy_macro import create_economy_macro_data_collection_agent
from .equity_estimates import create_equity_estimates_data_collection_agent
from .equity_fundamentals import create_equity_fundamentals_data_collection_agent
from .equity_ownership import create_equity_ownership_data_collection_agent
from .equity_price import create_equity_price_data_collection_agent
from .etf_index import create_etf_index_data_collection_agent
from .fixed_income import create_fixed_income_data_collection_agent
from .news import create_news_data_collection_agent
from .options import create_options_data_collection_agent

__all__ = [
    "create_economy_macro_data_collection_agent",
    "create_equity_estimates_data_collection_agent",
    "create_equity_fundamentals_data_collection_agent",
    "create_equity_ownership_data_collection_agent",
    "create_equity_price_data_collection_agent",
    "create_etf_index_data_collection_agent",
    "create_fixed_income_data_collection_agent",
    "create_news_data_collection_agent",
    "create_options_data_collection_agent",
]