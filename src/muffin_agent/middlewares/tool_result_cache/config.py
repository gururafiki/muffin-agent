"""Configuration for the tool result cache middleware."""

from pydantic import Field

from ...utils.base_config import BaseConfiguration


class ToolResultCacheConfiguration(BaseConfiguration):
    """Settings for tool output schema discovery and cache filtering.

    Controls which Python packages are scanned when resolving
    ``extras["output_schema"]`` for cached tool results, and which
    tool-result content patterns should be excluded from the cache.
    """

    tool_schema_packages: list[str] = Field(
        default=["muffin_agent.tools", "muffin_agent.middlewares.store_access"],
        description="Dotted module paths to scan for tool output schemas.",
    )
    non_cacheable_patterns: list[str] = Field(
        default=["tool result too large"],
        description=(
            "Case-insensitive substrings matched against string tool-result content. "
            "Results containing any of these are excluded from the cache. "
            "Default catches FilesystemMiddleware offload messages — those embed an "
            "ephemeral /large_tool_results/<tool_call_id> path that becomes stale on "
            "cache hit. Extend via env var TOOL_RESULT_CACHE_NON_CACHEABLE_PATTERNS."
        ),
    )
