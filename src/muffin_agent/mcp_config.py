"""Configuration and LLM provider management."""

from langchain_mcp_adapters.sessions import Connection
from pydantic import Field

from .utils.base_config import BaseConfiguration


class McpConfiguration(BaseConfiguration):
    """MCP server connection configuration."""

    openbb_mcp_url: str = Field(
        default="http://127.0.0.1:8001/mcp",
        description="OpenBB MCP server URL",
    )
    firecrawl_mcp_url: str = Field(
        default="http://127.0.0.1:3000/mcp",
        description="Firecrawl MCP server URL",
    )

    def get_mcp_connections(self) -> dict[str, Connection]:
        """Get MCP server connections for MultiServerMCPClient."""
        return {
            "openbb": {
                "url": self.openbb_mcp_url,
                "transport": "streamable_http",
            },
            "firecrawl": {
                "url": self.firecrawl_mcp_url,
                "transport": "streamable_http",
            },
        }
