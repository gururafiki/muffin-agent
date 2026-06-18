"""Tool result cache middleware — store-backed cross-agent deduplication.

Public API:

* :class:`ToolResultCacheMiddleware` — intercepts LLM-driven tool calls
  for caching.
* :func:`cached_invoke` — high-level direct invocation with cache HIT/MISS
  for non-LLM callers (deterministic graph nodes).
* :func:`cache_lookup`, :func:`cache_store` — low-level primitives used by
  both the middleware and ``cached_invoke``.
* :func:`get_args_hash`, :func:`is_error_content` — foundational helpers.
"""

from .cache import (
    cache_lookup,
    cache_store,
    cached_invoke,
    get_args_hash,
    is_content_cacheable,
    is_error_content,
)
from .middleware import ToolResultCacheMiddleware

__all__ = [
    "ToolResultCacheMiddleware",
    "cache_lookup",
    "cache_store",
    "cached_invoke",
    "get_args_hash",
    "is_content_cacheable",
    "is_error_content",
]
