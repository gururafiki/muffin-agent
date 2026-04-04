"""Web search and crawling configuration.

Provides ``WebConfiguration`` for connecting to self-hosted SearxNG (search)
and Firecrawl (scraping/crawling) services.
"""

from pydantic import Field

from .utils.base_config import BaseConfiguration


class WebConfiguration(BaseConfiguration):
    """Configuration for self-hosted web search and crawling services.

    Resolved from environment variables (uppercase field names) with
    fallback to ``RunnableConfig["configurable"]`` and then defaults.

    Environment variables:
    - ``SEARXNG_URL`` — SearxNG base URL (no trailing slash).
    - ``FIRECRAWL_URL`` — Firecrawl API base URL (no trailing slash).
    - ``FIRECRAWL_API_KEY`` — API key sent as ``Authorization: Bearer``.
      Use any non-empty string when ``USE_DB_AUTHENTICATION=false`` (default
      for self-hosted).
    """

    searxng_url: str = Field(
        default="http://127.0.0.1:8888",
        description=(
            "SearxNG base URL. Defaults to localhost:8888 for local CLI usage. "
            "Auto-configured to http://searxng:8080 in docker-compose."
        ),
    )
    firecrawl_url: str = Field(
        default="http://127.0.0.1:3002",
        description=(
            "Firecrawl API base URL. Defaults to localhost:3002 for local CLI usage. "
            "Auto-configured to http://firecrawl-api:3002 in docker-compose."
        ),
    )
    firecrawl_api_key: str = Field(
        default="local",
        description=(
            "Firecrawl API key. Any non-empty value works when the server runs "
            "with USE_DB_AUTHENTICATION=false (the default for self-hosted)."
        ),
    )
