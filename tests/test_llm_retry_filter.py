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


@pytest.mark.unit
class TestTypedStatusCodeRetry:
    """openrouter SDK raises typed errors carrying an HTTP status_code (e.g.
    TooManyRequestsResponseError = 429), which the type/message branches miss."""

    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_transient_status_codes_retried(self, status):
        exc = type("FakeSDKError", (Exception,), {"status_code": status})("upstream")
        assert _should_retry_llm_call(exc) is True

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_permanent_status_codes_not_retried(self, status):
        exc = type("FakeSDKError", (Exception,), {"status_code": status})("bad request")
        assert _should_retry_llm_call(exc) is False

    def test_free_models_per_min_429_retried(self):
        # the exact deployed failure shape
        exc = type("TooManyRequestsResponseError", (Exception,), {"status_code": 429})(
            "Rate limit exceeded: free-models-per-min. "
        )
        assert _should_retry_llm_call(exc) is True
