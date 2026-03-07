"""Tests for the criterion evaluation agent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from muffin_agent.prompts import render_template


@pytest.mark.unit
class TestPromptTemplate:
    """Test criterion evaluation prompt template rendering."""

    def test_criterion_evaluation_template_renders(self):
        result = render_template("criterion_evaluation.jinja")
        assert "criterion evaluation" in result.lower()
        assert len(result) > 100

    def test_template_contains_workflow_steps(self):
        result = render_template("criterion_evaluation.jinja")
        assert "Analyze the Criterion" in result
        assert "Collect Data" in result
        assert "Validate Data" in result
        assert "Evaluate the Criterion" in result
        assert "Reflect" in result

    def test_template_contains_subagent_selection_guide(self):
        result = render_template("criterion_evaluation.jinja")
        assert "Subagent Selection Guide" in result
        assert "Profitability" in result
        assert "Valuation" in result
        assert "Macro Sensitivity" in result

    def test_template_contains_output_format(self):
        result = render_template("criterion_evaluation.jinja")
        assert "CRITERION_EVALUATION_START" in result
        assert "CRITERION_EVALUATION_END" in result
        assert "score:" in result
        assert "confidence:" in result
        assert "signal:" in result

    def test_template_contains_cot_instructions(self):
        result = render_template("criterion_evaluation.jinja")
        assert "sub-criteria" in result.lower()
        assert "formula" in result.lower()
        assert "benchmark" in result.lower()

    def test_template_contains_reflection_checks(self):
        result = render_template("criterion_evaluation.jinja")
        assert "Score-evidence consistency" in result
        assert "Confirmation bias" in result
        assert "Anchoring bias" in result
        assert "COUNTERARGUMENT" in result

    def test_template_contains_grounding_constraints(self):
        result = render_template("criterion_evaluation.jinja")
        assert "NEVER" in result
        assert "fabricat" in result.lower()

    def test_template_contains_iteration_limit(self):
        result = render_template("criterion_evaluation.jinja")
        assert "at most 2" in result

    def test_template_delegates_validation_to_subagent(self):
        result = render_template("criterion_evaluation.jinja")
        assert "data-validation" in result
        assert "proceed" in result
        assert "collect_more_data" in result
        assert "insufficient_data" in result

    def test_template_contains_signal_mapping(self):
        result = render_template("criterion_evaluation.jinja")
        assert "strong_positive" in result
        assert "strong_negative" in result
        assert "neutral" in result


@pytest.mark.unit
class TestCreateCriterionEvaluationAgent:
    """Test agent creation."""

    @pytest.mark.asyncio
    async def test_creates_agent_with_subagents(self):
        mock_currency_commodities_agent = MagicMock()
        mock_discovery_screening_agent = MagicMock()
        mock_economy_macro_agent = MagicMock()
        mock_etf_index_agent = MagicMock()
        mock_fixed_income_agent = MagicMock()
        mock_fundamentals_agent = MagicMock()
        mock_price_agent = MagicMock()
        mock_estimates_agent = MagicMock()
        mock_ownership_agent = MagicMock()
        mock_news_agent = MagicMock()
        mock_options_agent = MagicMock()
        mock_regulatory_filings_agent = MagicMock()
        mock_fama_french_agent = MagicMock()
        mock_validation_agent = MagicMock()

        config = MagicMock()
        config.get_llm.return_value = MagicMock()

        with (
            patch(
                "muffin_agent.agents.subagents"
                ".create_currency_commodities_data_collection_agent",
                new_callable=AsyncMock,
                return_value=mock_currency_commodities_agent,
            ),
            patch(
                "muffin_agent.agents.subagents"
                ".create_discovery_screening_data_collection_agent",
                new_callable=AsyncMock,
                return_value=mock_discovery_screening_agent,
            ),
            patch(
                "muffin_agent.agents.subagents"
                ".create_economy_macro_data_collection_agent",
                new_callable=AsyncMock,
                return_value=mock_economy_macro_agent,
            ),
            patch(
                "muffin_agent.agents.subagents"
                ".create_equity_fundamentals_data_collection_agent",
                new_callable=AsyncMock,
                return_value=mock_fundamentals_agent,
            ),
            patch(
                "muffin_agent.agents.subagents"
                ".create_equity_price_data_collection_agent",
                new_callable=AsyncMock,
                return_value=mock_price_agent,
            ),
            patch(
                "muffin_agent.agents.subagents"
                ".create_equity_estimates_data_collection_agent",
                new_callable=AsyncMock,
                return_value=mock_estimates_agent,
            ),
            patch(
                "muffin_agent.agents.subagents"
                ".create_equity_ownership_data_collection_agent",
                new_callable=AsyncMock,
                return_value=mock_ownership_agent,
            ),
            patch(
                "muffin_agent.agents.subagents.create_news_data_collection_agent",
                new_callable=AsyncMock,
                return_value=mock_news_agent,
            ),
            patch(
                "muffin_agent.agents.subagents.create_options_data_collection_agent",
                new_callable=AsyncMock,
                return_value=mock_options_agent,
            ),
            patch(
                "muffin_agent.agents.subagents.create_etf_index_data_collection_agent",
                new_callable=AsyncMock,
                return_value=mock_etf_index_agent,
            ),
            patch(
                "muffin_agent.agents.subagents"
                ".create_fixed_income_data_collection_agent",
                new_callable=AsyncMock,
                return_value=mock_fixed_income_agent,
            ),
            patch(
                "muffin_agent.agents.subagents"
                ".create_regulatory_filings_data_collection_agent",
                new_callable=AsyncMock,
                return_value=mock_regulatory_filings_agent,
            ),
            patch(
                "muffin_agent.agents.subagents"
                ".create_fama_french_data_collection_agent",
                new_callable=AsyncMock,
                return_value=mock_fama_french_agent,
            ),
            patch(
                "muffin_agent.agents.subagents.create_data_validation_agent",
                new_callable=AsyncMock,
                return_value=mock_validation_agent,
            ),
            patch(
                "muffin_agent.agents.criterion_evaluation.create_deep_agent"
            ) as mock_create,
        ):
            mock_create.return_value = MagicMock()

            from muffin_agent.agents.criterion_evaluation import (
                create_criterion_evaluation_agent,
            )

            agent = await create_criterion_evaluation_agent(config)

            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args
            assert call_kwargs.kwargs["model"] == config.get_llm.return_value
            subagents = call_kwargs.kwargs["subagents"]
            assert len(subagents) == 14
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
            assert subagents[6]["name"] == "economy-macro"
            assert subagents[6]["runnable"] is mock_economy_macro_agent
            assert subagents[7]["name"] == "fixed-income"
            assert subagents[7]["runnable"] is mock_fixed_income_agent
            assert subagents[8]["name"] == "etf-index"
            assert subagents[8]["runnable"] is mock_etf_index_agent
            assert subagents[9]["name"] == "discovery-screening"
            assert subagents[9]["runnable"] is mock_discovery_screening_agent
            assert subagents[10]["name"] == "currency-commodities"
            assert subagents[10]["runnable"] is mock_currency_commodities_agent
            assert subagents[11]["name"] == "regulatory-filings"
            assert subagents[11]["runnable"] is mock_regulatory_filings_agent
            assert subagents[12]["name"] == "fama-french"
            assert subagents[12]["runnable"] is mock_fama_french_agent
            assert subagents[13]["name"] == "data-validation"
            assert subagents[13]["runnable"] is mock_validation_agent
            assert agent is mock_create.return_value
