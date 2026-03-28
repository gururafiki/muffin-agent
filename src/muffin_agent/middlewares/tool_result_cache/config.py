"""Configuration for the tool result cache middleware."""

from pydantic import Field

from ...utils.base_config import BaseConfiguration


class ToolResultCacheConfiguration(BaseConfiguration):
    """Settings for tool output schema discovery.

    Controls which Python packages are scanned when resolving
    ``extras["output_schema"]`` for cached tool results.
    """

    tool_schema_packages: list[str] = Field(
        default=["muffin_agent.tools", "muffin_agent.middlewares.store_access"],
        description="Dotted module paths to scan for tool output schemas.",
    )
