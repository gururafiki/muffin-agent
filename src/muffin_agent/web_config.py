"""Web search configuration.

Provides ``WebConfiguration`` for connecting to the self-hosted SearxNG
meta-search service.  Firecrawl is accessed via its MCP server
(``McpConfiguration.firecrawl_mcp_url``) and does not require a URL here.
"""

from pydantic import Field

from .utils.base_config import BaseConfiguration


class WebConfiguration(BaseConfiguration):
    """Configuration for the self-hosted SearxNG search service.

    Resolved from environment variables (uppercase field names) with
    fallback to ``RunnableConfig["configurable"]`` and then defaults.

    Environment variables:
    - ``SEARXNG_URL`` — SearxNG base URL (no trailing slash).
    """

    searxng_url: str = Field(
        default="http://127.0.0.1:8888",
        description=(
            "SearxNG base URL. Defaults to localhost:8888 for local CLI usage. "
            "Auto-configured to http://searxng:8080 in docker-compose."
        ),
    )
