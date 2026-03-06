"""Tests for the discovery and screening data collection agent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from muffin_agent.agents.data_collection.discovery_screening import MCP_TOOLS
from muffin_agent.prompts import render_template


@pytest.mark.unit
class TestMCPTools:
    """Test MCP_TOOLS list."""

    def test_tool_count(self):
        assert len(MCP_TOOLS) == 23

    def test_tools_sorted(self):
        assert MCP_TOOLS == sorted(MCP_TOOLS)

    def test_tool_prefix(self):
        for tool in MCP_TOOLS:
            assert tool.startswith("equity_"), f"Unexpected tool prefix: {tool}"


@pytest.mark.unit
class TestGetTools:
    """Test tool filtering via get_tools."""

    @pytest.mark.asyncio
    async def test_filters_to_matching_tools(self):
        mock_tool = MagicMock()
        mock_tool.name = "equity_screener"
        non_matching = MagicMock()
        non_matching.name = "fixedincome_rate_sofr"

        with patch(
            "muffin_agent.agents.data_collection.utils.MultiServerMCPClient"
        ) as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get_tools = AsyncMock(return_value=[mock_tool, non_matching])
            mock_client_cls.return_value = mock_client

            from muffin_agent.agents.data_collection.utils import get_tools

            config = MagicMock()
            config.get_mcp_connections.return_value = {}
            result = await get_tools(config, ["equity_screener"])

        assert result == [mock_tool]

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_match(self):
        mock_tool = MagicMock()
        mock_tool.name = "other_tool"

        with patch(
            "muffin_agent.agents.data_collection.utils.MultiServerMCPClient"
        ) as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get_tools = AsyncMock(return_value=[mock_tool])
            mock_client_cls.return_value = mock_client

            from muffin_agent.agents.data_collection.utils import get_tools

            config = MagicMock()
            config.get_mcp_connections.return_value = {}
            result = await get_tools(config, ["equity_screener"])

        assert result == []


@pytest.mark.unit
class TestPromptTemplate:
    """Test prompt template rendering."""

    def test_template_renders(self):
        result = render_template("discovery_screening.jinja")
        assert "discovery" in result.lower() or "screening" in result.lower()
        assert len(result) > 100

    def test_template_contains_tool_names(self):
        result = render_template("discovery_screening.jinja")
        assert "equity_screener" in result
        assert "equity_compare_peers" in result
        assert "equity_profile" in result

    def test_template_notes_symbol_required_tools(self):
        result = render_template("discovery_screening.jinja")
        assert "ticker" in result.lower()
