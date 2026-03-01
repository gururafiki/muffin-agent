"""Tests for the stock evaluation agent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from muffin_agent.prompts import render_template


@pytest.mark.unit
class TestPromptTemplate:
    """Test prompt template rendering."""

    def test_stock_evaluation_template_renders(self):
        result = render_template("stock_evaluation.jinja")
        assert "stock evaluation" in result.lower()
        assert "equity-fundamentals" in result
        assert "equity-price" in result
        assert len(result) > 100

    def test_template_contains_workflow_steps(self):
        result = render_template("stock_evaluation.jinja")
        assert "Plan Data Collection" in result
        assert "Collect Data" in result
        assert "Validate Collected Data" in result
        assert "Analyze" in result
        assert "Reflect" in result

    def test_template_contains_validation_criteria(self):
        result = render_template("stock_evaluation.jinja")
        assert "Sufficiency" in result
        assert "Relevance" in result
        assert "Temporal correctness" in result
        assert "Completeness" in result

    def test_template_contains_output_format(self):
        result = render_template("stock_evaluation.jinja")
        assert "Score" in result
        assert "Reasoning" in result
        assert "Data Used" in result


@pytest.mark.unit
class TestCreateStockEvaluationAgent:
    """Test agent creation."""

    @pytest.mark.asyncio
    async def test_creates_agent_with_subagents(self):
        mock_fundamentals_agent = MagicMock()
        mock_price_agent = MagicMock()
        mock_estimates_agent = MagicMock()
        mock_ownership_agent = MagicMock()
        mock_news_agent = MagicMock()
        mock_options_agent = MagicMock()

        config = MagicMock()
        config.get_llm.return_value = MagicMock()

        with (
            patch(
                "muffin_agent.agents.stock_evaluation"
                ".create_equity_fundamentals_data_collection_agent",
                new_callable=AsyncMock,
                return_value=mock_fundamentals_agent,
            ),
            patch(
                "muffin_agent.agents.stock_evaluation"
                ".create_equity_price_data_collection_agent",
                new_callable=AsyncMock,
                return_value=mock_price_agent,
            ),
            patch(
                "muffin_agent.agents.stock_evaluation"
                ".create_equity_estimates_data_collection_agent",
                new_callable=AsyncMock,
                return_value=mock_estimates_agent,
            ),
            patch(
                "muffin_agent.agents.stock_evaluation"
                ".create_equity_ownership_data_collection_agent",
                new_callable=AsyncMock,
                return_value=mock_ownership_agent,
            ),
            patch(
                "muffin_agent.agents.stock_evaluation"
                ".create_news_data_collection_agent",
                new_callable=AsyncMock,
                return_value=mock_news_agent,
            ),
            patch(
                "muffin_agent.agents.stock_evaluation"
                ".create_options_data_collection_agent",
                new_callable=AsyncMock,
                return_value=mock_options_agent,
            ),
            patch(
                "muffin_agent.agents.stock_evaluation.create_deep_agent"
            ) as mock_create,
        ):
            mock_create.return_value = MagicMock()

            from muffin_agent.agents.stock_evaluation import (
                create_stock_evaluation_agent,
            )

            agent = await create_stock_evaluation_agent(config)

            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args
            assert call_kwargs.kwargs["model"] == config.get_llm.return_value
            subagents = call_kwargs.kwargs["subagents"]
            assert len(subagents) == 6
            assert subagents[0]["name"] == "equity-fundamentals"
            assert subagents[0]["runnable"] is mock_fundamentals_agent
            assert subagents[1]["name"] == "equity-price"
            assert subagents[1]["runnable"] is mock_price_agent
            assert subagents[2]["name"] == "equity-estimates"
            assert subagents[2]["runnable"] is mock_estimates_agent
            assert subagents[3]["name"] == "equity-ownership"
            assert subagents[3]["runnable"] is mock_ownership_agent
            assert subagents[4]["name"] == "news"
            assert subagents[4]["runnable"] is mock_news_agent
            assert subagents[5]["name"] == "options"
            assert subagents[5]["runnable"] is mock_options_agent
            assert agent is mock_create.return_value
