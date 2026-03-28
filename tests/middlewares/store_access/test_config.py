"""Tests for StoreConfiguration."""

import pytest

from muffin_agent.middlewares.store_access.config import StoreConfiguration


@pytest.mark.unit
class TestStoreConfiguration:
    def test_default_unrestricted(self):
        config = StoreConfiguration()
        assert config.store_allowed_namespaces is None

    def test_from_runnable_config(self):
        config = StoreConfiguration.from_runnable_config(
            {"configurable": {"store_allowed_namespaces": ["cache", "computed"]}}
        )
        assert config.store_allowed_namespaces == ["cache", "computed"]

    def test_from_empty_config(self):
        config = StoreConfiguration.from_runnable_config({"configurable": {}})
        assert config.store_allowed_namespaces is None
