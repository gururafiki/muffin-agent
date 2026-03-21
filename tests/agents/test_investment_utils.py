"""Tests for investment node utilities."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from muffin_agent.agents.investment.utils import TRANSIENT_ERRORS, run_deep_agent_node


class _FakeInputState:
    __annotations__ = {"ticker": str, "query": str}


@pytest.fixture
def _node_kwargs():
    """Common kwargs for run_deep_agent_node."""
    return {
        "state": {"ticker": "AAPL", "query": "analyze"},
        "config": {"configurable": {}},
        "agent_factory": AsyncMock(),
        "input_state_type": _FakeInputState,
        "state_key": "market_regime",
        "error_fallback": {"regime_label": "unknown"},
    }


@pytest.mark.unit
class TestTransientErrorPropagation:
    """Transient errors propagate so RetryPolicy can retry the node."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc_class",
        [ConnectionError, TimeoutError, httpx.ConnectError, httpx.ReadTimeout],
    )
    async def test_transient_errors_propagate(self, exc_class, _node_kwargs):
        _node_kwargs["agent_factory"] = AsyncMock(side_effect=exc_class("transient"))

        with pytest.raises(exc_class):
            await run_deep_agent_node(**_node_kwargs)

    @pytest.mark.asyncio
    async def test_non_transient_errors_return_fallback(self, _node_kwargs):
        _node_kwargs["agent_factory"] = AsyncMock(
            side_effect=ValueError("bad schema")
        )

        result = await run_deep_agent_node(**_node_kwargs)

        assert result["market_regime"]["error"] == "Agent raised an exception"
        assert result["market_regime"]["regime_label"] == "unknown"


@pytest.mark.unit
class TestTransientErrorTuple:
    """Verify TRANSIENT_ERRORS contains expected exception types."""

    def test_contains_connection_error(self):
        assert ConnectionError in TRANSIENT_ERRORS

    def test_contains_timeout_error(self):
        assert TimeoutError in TRANSIENT_ERRORS

    def test_contains_httpx_network_error(self):
        assert httpx.NetworkError in TRANSIENT_ERRORS

    def test_contains_httpx_timeout(self):
        assert httpx.TimeoutException in TRANSIENT_ERRORS


@pytest.mark.unit
class TestStructuredOutputExtraction:
    """Verify structured output extraction from agent result."""

    @pytest.mark.asyncio
    async def test_successful_structured_output(self, _node_kwargs):
        mock_structured = MagicMock()
        mock_structured.model_dump.return_value = {"regime_label": "expansion"}

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"structured_response": mock_structured}
        _node_kwargs["agent_factory"] = AsyncMock(return_value=mock_agent)

        with patch(
            "muffin_agent.agents.investment.utils.Configuration"
        ) as mock_config_cls:
            mock_config_cls.from_runnable_config.return_value = AsyncMock()
            result = await run_deep_agent_node(**_node_kwargs)

        assert result["market_regime"]["regime_label"] == "expansion"

    @pytest.mark.asyncio
    async def test_missing_structured_output_returns_fallback(self, _node_kwargs):
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"output": "some text"}
        _node_kwargs["agent_factory"] = AsyncMock(return_value=mock_agent)

        with patch(
            "muffin_agent.agents.investment.utils.Configuration"
        ) as mock_config_cls:
            mock_config_cls.from_runnable_config.return_value = AsyncMock()
            result = await run_deep_agent_node(**_node_kwargs)

        assert "error" in result["market_regime"]
        assert result["market_regime"]["regime_label"] == "unknown"
        assert result["market_regime"]["raw_output"] == "some text"
