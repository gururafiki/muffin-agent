"""Tests for the data validation agent."""

from unittest.mock import MagicMock, patch

import pytest

from muffin_agent.prompts import render_template


@pytest.mark.unit
class TestPromptTemplate:
    """Test data validation prompt template rendering."""

    def test_data_validation_template_renders(self):
        result = render_template("data_validation.jinja")
        assert "data quality" in result.lower()
        assert len(result) > 100

    def test_template_contains_validation_dimensions(self):
        result = render_template("data_validation.jinja")
        assert "Sufficiency" in result
        assert "Relevance" in result
        assert "Temporal Validity" in result
        assert "Consistency" in result

    def test_template_contains_scoring_anchors(self):
        result = render_template("data_validation.jinja")
        assert "0.8" in result
        assert "0.5" in result

    def test_template_contains_output_sections(self):
        result = render_template("data_validation.jinja")
        assert "Data Validation Report" in result
        assert "Scores" in result
        assert "Gaps" in result
        assert "Issues" in result
        assert "Recommendation" in result

    def test_template_contains_recommendations(self):
        result = render_template("data_validation.jinja")
        assert "proceed" in result
        assert "collect_more_data" in result
        assert "insufficient_data" in result

    def test_template_contains_safety_instructions(self):
        result = render_template("data_validation.jinja")
        assert "Never" in result or "NEVER" in result
        assert "fabricat" in result.lower()

    def test_template_contains_confidence_formula(self):
        result = render_template("data_validation.jinja")
        assert "Overall Confidence" in result
        assert "Overall Relevance" in result
        assert "0.35" in result
        assert "0.30" in result


@pytest.mark.unit
class TestCreateDataValidationAgent:
    """Test agent creation."""

    @pytest.mark.asyncio
    async def test_creates_agent_without_tools(self):
        config = MagicMock()
        config.get_llm.return_value = MagicMock()

        with patch(
            "muffin_agent.agents.data_validation.create_agent"
        ) as mock_create:
            mock_create.return_value = MagicMock()

            from muffin_agent.agents.data_validation import (
                create_data_validation_agent,
            )

            agent = await create_data_validation_agent(config)

            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args
            assert call_kwargs.kwargs["model"] == config.get_llm.return_value
            assert "system_prompt" in call_kwargs.kwargs
            assert "tools" not in call_kwargs.kwargs
            assert agent is mock_create.return_value
