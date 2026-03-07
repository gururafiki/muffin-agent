"""Tests for Configuration and get_llm()."""

from unittest.mock import MagicMock, patch

import pytest

from muffin_agent.config import Configuration


@pytest.mark.unit
class TestGetLlmRetries:
    """Test that get_llm forwards max_retries to LLM constructors."""

    def test_default_max_retries_is_6(self):
        config = Configuration(llm_provider="openai", openai_api_key="test-key")
        assert config.llm_max_retries == 6

    def test_openai_gets_max_retries(self):
        config = Configuration(
            llm_provider="openai",
            openai_api_key="test-key",
            llm_max_retries=5,
        )
        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            config.get_llm()
        _, kwargs = mock_cls.call_args
        assert kwargs["max_retries"] == 5

    def test_anthropic_gets_max_retries(self):
        config = Configuration(
            llm_provider="anthropic",
            anthropic_api_key="test-key",
            llm_max_retries=2,
        )
        with patch("langchain_anthropic.ChatAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            config.get_llm()
        _, kwargs = mock_cls.call_args
        assert kwargs["max_retries"] == 2

    def test_env_var_overrides_default(self, monkeypatch):
        monkeypatch.setenv("LLM_MAX_RETRIES", "7")
        config = Configuration.from_runnable_config({"configurable": {}})
        assert config.llm_max_retries == 7
