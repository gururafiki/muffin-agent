"""Configuration for ``ToolTelemetryMiddleware``."""

from __future__ import annotations

from muffin_agent.utils.base_config import BaseConfiguration


class ToolTelemetryConfiguration(BaseConfiguration):
    """Per-run switch for tool-execution telemetry capture.

    Env ``TOOL_TELEMETRY_ENABLED`` or ``configurable.tool_telemetry_enabled``.
    Off by default so graphs that don't render telemetry pay nothing (the
    middleware is a no-op). The muffin-ui criteria_analysis page sends
    ``tool_telemetry_enabled: true`` by default; other graphs opt in via
    config. The flag propagates ambiently to every nested agent (LangGraph
    keeps ``configurable`` in context for the whole run).
    """

    tool_telemetry_enabled: bool = False
