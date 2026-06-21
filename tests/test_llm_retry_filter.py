"""Unit tests for the LLM retry filter's handling of OpenRouter streaming errors."""

import pytest

from muffin_agent.utils.agent_builder import _should_retry_llm_call


@pytest.mark.unit
class TestOpenRouterStreamingRetry:
    """langchain-openrouter raises a bare ValueError mid-stream for upstream errors.

    The type-based transient filter misses it, so deployed council runs died on a
    single ``... Upstream idle timeout exceeded (code: 504)``. These guard the
    message-based branch added to _should_retry_llm_call.
    """

    def test_idle_timeout_504_is_retried(self):
        exc = ValueError(
            "OpenRouter API returned an error during streaming: "
            "Upstream idle timeout exceeded (code: 504)"
        )
        assert _should_retry_llm_call(exc) is True

    @pytest.mark.parametrize("code", ["500", "502", "503", "504", "429"])
    def test_transient_upstream_codes_are_retried(self, code):
        exc = ValueError(
            "OpenRouter API returned an error during streaming: "
            f"upstream error (code: {code})"
        )
        assert _should_retry_llm_call(exc) is True

    def test_permanent_openrouter_error_not_retried(self):
        exc = ValueError(
            "OpenRouter API returned an error during streaming: "
            "invalid model specified (code: 400)"
        )
        assert _should_retry_llm_call(exc) is False

    def test_unrelated_valueerror_not_retried(self):
        assert _should_retry_llm_call(ValueError("some application bug")) is False
