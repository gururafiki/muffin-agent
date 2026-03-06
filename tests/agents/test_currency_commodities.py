"""Tests for the currency, commodity, and crypto data collection agent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from muffin_agent.agents.data_collection.currency_commodities import MCP_TOOLS
from muffin_agent.agents.data_collection.utils import get_tools
from muffin_agent.prompts import render_template


@pytest.mark.unit
class TestMCPTools:
    """Test MCP_TOOLS list."""

    def test_tool_count(self):
        assert len(MCP_TOOLS) == 9

    def test_tools_sorted(self):
        assert MCP_TOOLS == sorted(MCP_TOOLS)

    def test_tool_prefixes(self):
        valid_prefixes = ("commodity_", "crypto_", "currency_")
        for tool in MCP_TOOLS:
            assert tool.startswith(valid_prefixes), f"Unexpected tool prefix: {tool}"


@pytest.mark.unit
class TestGetTools:
    """Test tool loading and filtering."""

    @pytest.mark.asyncio
    async def test_filters_to_allowed_tools(self):
        mock_tool_allowed = MagicMock()
        mock_tool_allowed.name = "commodity_price_spot"
        mock_tool_other = MagicMock()
        mock_tool_other.name = "equity_price_historical"

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
        assert tools[0].name == "commodity_price_spot"

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

    def test_template_renders(self):
        result = render_template("currency_commodities.jinja")
        assert "currency" in result.lower()
        assert len(result) > 100

    def test_template_contains_tool_names(self):
        result = render_template("currency_commodities.jinja")
        assert "commodity_price_spot" in result
        assert "currency_snapshots" in result
        assert "crypto_price_historical" in result
