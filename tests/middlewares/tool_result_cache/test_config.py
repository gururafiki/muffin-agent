"""Tests for ToolResultCacheConfiguration."""

import pytest

from muffin_agent.middlewares.tool_result_cache.config import (
    ToolResultCacheConfiguration,
)


@pytest.mark.unit
class TestToolResultCacheConfiguration:
    def test_default_packages(self):
        config = ToolResultCacheConfiguration()
        assert config.tool_schema_packages == [
            "muffin_agent.tools",
            "muffin_agent.middlewares.store_access",
        ]

    def test_from_runnable_config_custom(self):
        config = ToolResultCacheConfiguration.from_runnable_config(
            {
                "configurable": {
                    "tool_schema_packages": [
                        "muffin_agent.tools",
                        "custom_tools",
                    ],
                },
            }
        )
        assert config.tool_schema_packages == [
            "muffin_agent.tools",
            "custom_tools",
        ]

    def test_from_empty_config(self):
        config = ToolResultCacheConfiguration.from_runnable_config({"configurable": {}})
        assert config.tool_schema_packages == [
            "muffin_agent.tools",
            "muffin_agent.middlewares.store_access",
        ]
