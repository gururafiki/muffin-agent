"""Data collection agents."""

from .currency_commodities import (
    create_currency_commodities_data_collection_agent,
)
from .discovery_screening import create_discovery_screening_data_collection_agent
from .economy_macro import create_economy_macro_data_collection_agent
from .equity_estimates import create_equity_estimates_data_collection_agent
from .equity_fundamentals import create_equity_fundamentals_data_collection_agent
from .equity_ownership import create_equity_ownership_data_collection_agent
from .equity_price import create_equity_price_data_collection_agent
from .etf_index import create_etf_index_data_collection_agent
from .fama_french import create_fama_french_data_collection_agent
from .fixed_income import create_fixed_income_data_collection_agent
from .news import create_news_data_collection_agent
from .options import create_options_data_collection_agent
from .regulatory_filings import create_regulatory_filings_data_collection_agent

__all__ = [
    "create_currency_commodities_data_collection_agent",
    "create_discovery_screening_data_collection_agent",
    "create_economy_macro_data_collection_agent",
    "create_equity_estimates_data_collection_agent",
    "create_equity_fundamentals_data_collection_agent",
    "create_equity_ownership_data_collection_agent",
    "create_equity_price_data_collection_agent",
    "create_etf_index_data_collection_agent",
    "create_fama_french_data_collection_agent",
    "create_fixed_income_data_collection_agent",
    "create_news_data_collection_agent",
    "create_options_data_collection_agent",
    "create_regulatory_filings_data_collection_agent",
]
