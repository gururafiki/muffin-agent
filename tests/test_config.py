"""Tests for Configuration and get_llm()."""

from unittest.mock import MagicMock, patch

import pytest

from muffin_agent.model_config import ModelConfiguration


@pytest.mark.unit
class TestGetLlmRetries:
    """Test that get_llm forwards SDK-level retries to LLM constructors."""

    def test_default_sdk_retries_is_6(self):
        config = ModelConfiguration(llm_provider="openai", openai_api_key="test-key")
        assert config.llm_sdk_retries == 6

    def test_openai_gets_sdk_retries(self):
        config = ModelConfiguration(
            llm_provider="openai",
            openai_api_key="test-key",
            llm_sdk_retries=5,
        )
        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            config.get_llm()
        _, kwargs = mock_cls.call_args
        assert kwargs["max_retries"] == 5

    def test_anthropic_gets_sdk_retries(self):
        config = ModelConfiguration(
            llm_provider="anthropic",
            anthropic_api_key="test-key",
            llm_sdk_retries=2,
        )
        with patch("langchain_anthropic.ChatAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            config.get_llm()
        _, kwargs = mock_cls.call_args
        assert kwargs["max_retries"] == 2

    def test_env_var_overrides_default(self, monkeypatch):
        monkeypatch.setenv("LLM_SDK_RETRIES", "7")
        config = ModelConfiguration.from_runnable_config({"configurable": {}})
        assert config.llm_sdk_retries == 7


@pytest.mark.unit
class TestOpenRouterProvider:
    """Test the openrouter provider branch in get_llm."""

    def test_openrouter_instantiates_chat_openrouter(self):
        config = ModelConfiguration(
            llm_provider="openrouter",
            openrouter_api_key="or-test-key",
            model="nvidia/nemotron-test:free",
        )
        with patch("langchain_openrouter.ChatOpenRouter") as mock_cls:
            mock_cls.return_value = MagicMock()
            config.get_llm()
        _, kwargs = mock_cls.call_args
        assert kwargs["model"] == "nvidia/nemotron-test:free"
        assert kwargs["openrouter_api_key"] == "or-test-key"
        assert kwargs["max_retries"] == 6

    def test_openrouter_missing_key_raises(self):
        config = ModelConfiguration(llm_provider="openrouter", openrouter_api_key=None)
        with pytest.raises(ValueError, match="OpenRouter API key not configured"):
            config.get_llm()


@pytest.mark.unit
class TestRoleModelChains:
    """Test the per-role model chain helper."""

    def test_empty_chain_falls_back_to_single_get_llm(self):
        config = ModelConfiguration(llm_provider="openai", openai_api_key="test")
        with patch.object(
            ModelConfiguration, "get_llm", return_value=MagicMock()
        ) as mock:
            chain = config.get_llm_for_role("orchestrator")
        assert len(chain) == 1
        mock.assert_called_once()

    def test_populated_chain_returns_one_model_per_entry(self):
        config = ModelConfiguration(
            llm_provider="openai",
            openai_api_key="test",
            orchestrator_models=["model-a", "model-b", "model-c"],
        )
        sentinels = [MagicMock(name=f"m{i}") for i in range(3)]
        with patch.object(ModelConfiguration, "get_llm", side_effect=sentinels) as mock:
            chain = config.get_llm_for_role("orchestrator")
        assert chain == sentinels
        # Each entry passed via the `model=` kwarg.
        assert [c.kwargs["model"] for c in mock.call_args_list] == [
            "model-a",
            "model-b",
            "model-c",
        ]

    def test_each_role_reads_its_own_chain(self):
        config = ModelConfiguration(
            llm_provider="openai",
            openai_api_key="test",
            orchestrator_models=["orch-1"],
            collector_models=["coll-1", "coll-2"],
            reasoner_models=["reas-1"],
        )
        with patch.object(
            ModelConfiguration, "get_llm", return_value=MagicMock()
        ) as mock:
            config.get_llm_for_role("collector")
        assert [c.kwargs["model"] for c in mock.call_args_list] == ["coll-1", "coll-2"]

    def test_csv_env_var_populates_role_chain(self, monkeypatch):
        monkeypatch.setenv(
            "ORCHESTRATOR_MODELS",
            "anthropic/claude-sonnet-4-6, anthropic/claude-haiku-4-5",
        )
        config = ModelConfiguration.from_runnable_config({"configurable": {}})
        assert config.orchestrator_models == [
            "anthropic/claude-sonnet-4-6",
            "anthropic/claude-haiku-4-5",
        ]

    def test_configurable_dict_can_supply_list_directly(self, monkeypatch):
        # Clear env vars so the configurable wins; .env may set them locally.
        monkeypatch.delenv("REASONER_MODELS", raising=False)
        config = ModelConfiguration.from_runnable_config(
            {"configurable": {"reasoner_models": ["a", "b"]}}
        )
        assert config.reasoner_models == ["a", "b"]


@pytest.mark.unit
class TestSummariserModel:
    """Test the optional tool-failure summariser model."""

    def test_returns_none_when_unset(self):
        config = ModelConfiguration(llm_provider="openai", openai_api_key="test")
        assert config.get_summariser() is None

    def test_returns_chat_model_when_set(self):
        config = ModelConfiguration(
            llm_provider="openai",
            openai_api_key="test",
            summariser_model="haiku-cheap",
        )
        sentinel = MagicMock(name="summariser")
        with patch.object(
            ModelConfiguration, "get_llm", return_value=sentinel
        ) as mock:
            result = config.get_summariser()
        assert result is sentinel
        mock.assert_called_once()
        assert mock.call_args.kwargs["model"] == "haiku-cheap"

    def test_env_var_populates_summariser_model(self, monkeypatch):
        monkeypatch.setenv("SUMMARISER_MODEL", "anthropic/claude-haiku-4-5")
        config = ModelConfiguration.from_runnable_config({"configurable": {}})
        assert config.summariser_model == "anthropic/claude-haiku-4-5"
