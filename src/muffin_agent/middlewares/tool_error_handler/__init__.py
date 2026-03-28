"""Tool error handler middleware — catches and deduplicates permanent failures."""

from .middleware import (
    PERMANENT_ERROR_PATTERNS,
    ToolErrorHandlerMiddleware,
    get_cache_key,
    is_permanent_error,
)

__all__ = [
    "ToolErrorHandlerMiddleware",
    "is_permanent_error",
    "get_cache_key",
    "PERMANENT_ERROR_PATTERNS",
]
