"""Unit tests for the shared LLM rate limiter (OpenRouter free-tier throttle)."""

import pytest

from muffin_agent.model_config import ModelConfiguration, _shared_rate_limiter


@pytest.mark.unit
class TestSharedRateLimiter:
    """The council fans out ~50 parallel calls; OpenRouter free allows 20/min. A single
    process-wide limiter shared by every chat model keeps the combined rate capped."""

    def test_unset_means_no_limiter(self):
        cfg = ModelConfiguration(llm_provider="openrouter", openrouter_api_key="x")
        assert cfg.get_llm().rate_limiter is None

    def test_applied_and_shared_across_models(self):
        cfg = ModelConfiguration(
            llm_provider="openrouter",
            openrouter_api_key="x",
            llm_requests_per_second=0.3,
        )
        m1 = cfg.get_llm()
        m2 = cfg.get_llm(model="other/model:free")
        # one process-wide instance so all personas draw from the same bucket
        assert m1.rate_limiter is m2.rate_limiter
        assert m1.rate_limiter is _shared_rate_limiter(0.3)
        assert m1.rate_limiter.requests_per_second == 0.3

    def test_blank_env_value_coerces_to_none(self):
        # an empty LLM_REQUESTS_PER_SECOND (e.g. unset GitHub var) must not crash
        cfg = ModelConfiguration.from_runnable_config(
            {"configurable": {"llm_requests_per_second": ""}}
        )
        assert cfg.llm_requests_per_second is None

    def test_role_chain_models_share_the_limiter(self):
        cfg = ModelConfiguration(
            llm_provider="openrouter",
            openrouter_api_key="x",
            llm_requests_per_second=0.3,
            collector_models=["a/b:free", "c/d:free"],
        )
        models = cfg.get_llm_for_role("collector")
        assert len(models) == 2
        assert models[0].rate_limiter is models[1].rate_limiter
        assert models[0].rate_limiter is _shared_rate_limiter(0.3)


@pytest.mark.unit
class TestChainPerProfileRateLimiter:
    """The council path (`get_llm_for_role` → per-profile `_build_from_profile`)
    honours each profile's own `requests_per_second`. The deployed chain throttles
    BOTH the Ollama primary and the OpenRouter fallback so the ~19-way fan-out
    doesn't overwhelm Ollama Cloud with concurrent requests."""

    def test_ollama_primary_throttled_separately_from_openrouter_fallback(self):
        cfg = ModelConfiguration(
            openrouter_api_key="x",
            llm_chain=[
                {"provider": "ollama", "model": "m", "requests_per_second": 0.5},
                {
                    "provider": "openrouter",
                    "model": "x/y:free",
                    "requests_per_second": 0.3,
                },
            ],
        )
        primary, fallback = cfg.get_llm_for_role("collector")
        # each profile gets its OWN shared limiter keyed by its rate
        assert primary.rate_limiter is _shared_rate_limiter(0.5)
        assert fallback.rate_limiter is _shared_rate_limiter(0.3)
        assert primary.rate_limiter is not fallback.rate_limiter

    def test_ollama_profile_without_rps_is_unthrottled(self):
        # Regression: an Ollama primary with no `requests_per_second` gets NO
        # limiter — the misconfiguration that let the council flood Ollama Cloud
        # with "too many concurrent requests" (429) on the AMZN run.
        cfg = ModelConfiguration(
            llm_chain=[{"provider": "ollama", "model": "m"}],
        )
        (primary,) = cfg.get_llm_for_role("collector")
        assert primary.rate_limiter is None
