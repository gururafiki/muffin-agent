"""Tests for the equity price data collection agent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from muffin_agent.agents.data_collection.equity_price import MCP_TOOLS
from muffin_agent.agents.data_collection.utils import get_tools
from muffin_agent.prompts import render_template


@pytest.mark.unit
class TestMCPTools:
    """Test MCP tool allowlist."""

    def test_mcp_tools_count(self):
        assert len(MCP_TOOLS) == 5

    def test_mcp_tools_have_expected_prefixes(self):
        valid_prefixes = ("equity_price_", "equity_historical_")
        for tool_name in MCP_TOOLS:
            assert any(tool_name.startswith(p) for p in valid_prefixes), (
                f"Unexpected tool: {tool_name}"
            )

    def test_mcp_tools_sorted(self):
        assert MCP_TOOLS == sorted(MCP_TOOLS)


@pytest.mark.unit
class TestGetTools:
    """Test tool loading and filtering."""

    @pytest.mark.asyncio
    async def test_filters_to_allowed_tools(self):
        mock_tool_allowed = MagicMock()
        mock_tool_allowed.name = "equity_price_quote"

        mock_tool_other = MagicMock()
        mock_tool_other.name = "equity_fundamental_income"

        mock_client = AsyncMock()
        mock_client.get_tools = AsyncMock(
            return_value=[mock_tool_allowed, mock_tool_other]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        config = MagicMock()
        config.get_mcp_connections.return_value = {
            "openbb": {
                "url": "http://localhost:8001/mcp",
                "transport": "streamable_http",
            }
        }

        with patch(
            "muffin_agent.agents.data_collection.utils.MultiServerMCPClient",
            return_value=mock_client,
        ):
            tools = await get_tools(config, MCP_TOOLS)

        assert len(tools) == 1
        assert tools[0].name == "equity_price_quote"

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_matching_tools(self):
        mock_tool = MagicMock()
        mock_tool.name = "economy_gdp_real"

        mock_client = AsyncMock()
        mock_client.get_tools = AsyncMock(return_value=[mock_tool])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        config = MagicMock()
        config.get_mcp_connections.return_value = {}

        with patch(
            "muffin_agent.agents.data_collection.utils.MultiServerMCPClient",
            return_value=mock_client,
        ):
            tools = await get_tools(config, MCP_TOOLS)

        assert tools == []


@pytest.mark.unit
class TestPromptTemplate:
    """Test prompt template rendering."""

    def test_equity_price_template_renders(self):
        result = render_template("equity_price.jinja")
        assert "equity price" in result.lower()
        assert len(result) > 100
