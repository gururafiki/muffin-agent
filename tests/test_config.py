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
        with patch.object(ModelConfiguration, "get_llm", return_value=sentinel) as mock:
            result = config.get_summariser()
        assert result is sentinel
        mock.assert_called_once()
        assert mock.call_args.kwargs["model"] == "haiku-cheap"

    def test_env_var_populates_summariser_model(self, monkeypatch):
        monkeypatch.setenv("SUMMARISER_MODEL", "anthropic/claude-haiku-4-5")
        config = ModelConfiguration.from_runnable_config({"configurable": {}})
        assert config.summariser_model == "anthropic/claude-haiku-4-5"


@pytest.mark.unit
class TestGetChatModelForRole:
    """``ModelConfiguration.get_chat_model_for_role`` collapses the
    primary + fallbacks + with_retry boilerplate that downstream nodes
    used to repeat 9 times across ``trading_decision/``."""

    def test_returns_retry_wrapped_model_when_no_fallbacks(self):
        primary = MagicMock(name="primary-model")
        primary.with_retry.return_value = MagicMock(name="retry-wrapped")
        fake_cfg = MagicMock()
        fake_cfg.get_llm_for_role.return_value = [primary]

        with patch.object(
            ModelConfiguration, "from_runnable_config", return_value=fake_cfg
        ):
            result = ModelConfiguration.get_chat_model_for_role({}, "reasoner")

        # No fallbacks → ``with_fallbacks`` is never called; ``with_retry``
        # is applied directly to the primary.
        primary.with_fallbacks.assert_not_called()
        primary.with_retry.assert_called_once_with(
            stop_after_attempt=3, wait_exponential_jitter=True
        )
        assert result is primary.with_retry.return_value

    def test_wraps_with_fallbacks_when_present(self):
        primary = MagicMock(name="primary-model")
        fallback_a = MagicMock(name="fallback-a")
        fallback_b = MagicMock(name="fallback-b")
        with_fb = MagicMock(name="with-fallbacks")
        with_fb.with_retry.return_value = MagicMock(name="retry-wrapped")
        primary.with_fallbacks.return_value = with_fb

        fake_cfg = MagicMock()
        fake_cfg.get_llm_for_role.return_value = [primary, fallback_a, fallback_b]

        with patch.object(
            ModelConfiguration, "from_runnable_config", return_value=fake_cfg
        ):
            result = ModelConfiguration.get_chat_model_for_role({}, "reasoner")

        primary.with_fallbacks.assert_called_once_with([fallback_a, fallback_b])
        with_fb.with_retry.assert_called_once_with(
            stop_after_attempt=3, wait_exponential_jitter=True
        )
        assert result is with_fb.with_retry.return_value

    def test_stop_after_attempt_override(self):
        primary = MagicMock(name="primary-model")
        fake_cfg = MagicMock()
        fake_cfg.get_llm_for_role.return_value = [primary]

        with patch.object(
            ModelConfiguration, "from_runnable_config", return_value=fake_cfg
        ):
            ModelConfiguration.get_chat_model_for_role(
                {}, "collector", stop_after_attempt=5
            )

        primary.with_retry.assert_called_once_with(
            stop_after_attempt=5, wait_exponential_jitter=True
        )

    def test_schema_applied_before_fallbacks_and_retry(self):
        """When schema=Schema is set, with_structured_output runs on the
        primary AND each fallback BEFORE composing with_fallbacks + with_retry.

        Order matters: RunnableRetry does not proxy with_structured_output
        (the proxy magic is on RunnableBinding, not RunnableBindingBase).
        Calling .with_structured_output() AFTER .with_retry() raises
        AttributeError at runtime — this test guards that ordering.
        """

        class Schema:
            pass

        # Each model exposes with_structured_output → structured_<n>; the
        # structured_<n> mocks then chain through with_fallbacks → with_retry.
        primary = MagicMock(name="primary")
        fallback = MagicMock(name="fallback")
        structured_primary = MagicMock(name="structured-primary")
        structured_fallback = MagicMock(name="structured-fallback")
        with_fb = MagicMock(name="with-fallbacks")
        with_fb.with_retry.return_value = MagicMock(name="retry-wrapped")
        primary.with_structured_output.return_value = structured_primary
        fallback.with_structured_output.return_value = structured_fallback
        structured_primary.with_fallbacks.return_value = with_fb

        fake_cfg = MagicMock()
        fake_cfg.get_llm_for_role.return_value = [primary, fallback]

        with patch.object(
            ModelConfiguration, "from_runnable_config", return_value=fake_cfg
        ):
            result = ModelConfiguration.get_chat_model_for_role(
                {}, "reasoner", schema=Schema
            )

        # Structured output applied to BOTH primary and fallback BEFORE
        # any fallback/retry composition.
        primary.with_structured_output.assert_called_once_with(Schema)
        fallback.with_structured_output.assert_called_once_with(Schema)
        # The structured runnables are then composed with_fallbacks.
        structured_primary.with_fallbacks.assert_called_once_with([structured_fallback])
        # And with_retry is the outermost wrapper.
        with_fb.with_retry.assert_called_once_with(
            stop_after_attempt=3, wait_exponential_jitter=True
        )
        # Primary should NEVER have had with_structured_output called on
        # the post-fallbacks result — that's the broken-order trap.
        primary.with_fallbacks.assert_not_called()
        primary.with_retry.assert_not_called()
        assert result is with_fb.with_retry.return_value

    def test_schema_no_fallbacks_skips_with_fallbacks(self):
        """Single-model + schema: with_structured_output on primary,
        then with_retry directly (no with_fallbacks call)."""

        class Schema:
            pass

        primary = MagicMock(name="primary")
        structured = MagicMock(name="structured")
        primary.with_structured_output.return_value = structured

        fake_cfg = MagicMock()
        fake_cfg.get_llm_for_role.return_value = [primary]

        with patch.object(
            ModelConfiguration, "from_runnable_config", return_value=fake_cfg
        ):
            ModelConfiguration.get_chat_model_for_role({}, "reasoner", schema=Schema)

        primary.with_structured_output.assert_called_once_with(Schema)
        # No fallbacks → no with_fallbacks call.
        structured.with_fallbacks.assert_not_called()
        # Retry is applied directly to the structured runnable.
        structured.with_retry.assert_called_once_with(
            stop_after_attempt=3, wait_exponential_jitter=True
        )


@pytest.mark.unit
@pytest.mark.asyncio
class TestGetChatModelForRoleRegression:
    """Integration-style regression test that catches the chain-order trap.

    The existing FakeLLM stub in test_config.py uses MagicMock chains, which
    silently dispatch any ``with_*`` method. This test uses real LangChain
    Runnable composition (no MagicMock stubs) to verify the actual chain
    object survives ``with_structured_output`` and produces a typed Schema
    instance at runtime — guarding against the LangChain-version regression
    of ``RunnableRetry.with_structured_output`` raising AttributeError.
    """

    async def test_real_chain_with_schema_produces_schema_instance(self):
        from langchain_core.language_models.fake_chat_models import (
            GenericFakeChatModel,
        )
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda
        from langchain_core.runnables.retry import RunnableRetry
        from pydantic import BaseModel

        class Schema(BaseModel):
            answer: str

        # GenericFakeChatModel doesn't implement with_structured_output —
        # monkey-patch a fake that returns a RunnableLambda yielding the schema.
        def fake_structured(self, schema_cls, **kwargs):  # noqa: ARG001
            return RunnableLambda(lambda _: schema_cls(answer="ok"))

        original = GenericFakeChatModel.with_structured_output
        GenericFakeChatModel.with_structured_output = fake_structured
        try:
            primary = GenericFakeChatModel(messages=iter([AIMessage(content="p")]))
            fallback = GenericFakeChatModel(messages=iter([AIMessage(content="f")]))

            fake_cfg = MagicMock()
            fake_cfg.get_llm_for_role.return_value = [primary, fallback]

            with patch.object(
                ModelConfiguration,
                "from_runnable_config",
                return_value=fake_cfg,
            ):
                llm = ModelConfiguration.get_chat_model_for_role(
                    {}, "reasoner", schema=Schema
                )

            # RunnableRetry must remain the outermost wrapper — if structured
            # output had been applied AFTER with_retry, this would have raised
            # AttributeError during the helper call.
            assert isinstance(llm, RunnableRetry), (
                f"expected RunnableRetry outermost, got {type(llm).__name__}"
            )
            # And the chain actually produces a Schema instance at runtime.
            result = await llm.ainvoke({"q": "anything"})
            assert isinstance(result, Schema)
            assert result.answer == "ok"
        finally:
            GenericFakeChatModel.with_structured_output = original
