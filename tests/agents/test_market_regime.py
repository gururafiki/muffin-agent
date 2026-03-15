"""Tests for the market regime agent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from muffin_agent.agents.investment.market_regime import (
    MarketRegimeContext,
    MarketRegimeOutput,
    _build_task_description,
)
from muffin_agent.prompts import render_template

# ── Minimal valid output dict ──────────────────────────────────────────────────

_VALID_REGIME_DICT = {
    "regime_label": "Goldilocks late-cycle",
    "as_of_date": "2026-03-15",
    "confidence": 0.8,
    "dimensions": {
        "growth_cycle": {
            "label": "slowing",
            "score": 0.45,
            "direction": "deteriorating",
            "key_indicators": "Real GDP Q4 2025 +2.1% (BEA); CLI -0.3 trend (OECD)",
        },
        "inflation_regime": {
            "label": "moderate",
            "score": 0.5,
            "direction": "falling",
            "key_indicators": "CPI YoY 2.8% (BLS Jan 2026); Core PCE 2.6% (BEA)",
        },
        "monetary_policy": {
            "label": "neutral",
            "score": 0.5,
            "direction": "stable",
            "key_indicators": "EFFR 4.33% (Fed Jan 2026); FOMC on hold",
        },
        "liquidity_risk_appetite": {
            "label": "cautiously_risk_on",
            "score": 0.65,
            "direction": "stable",
            "key_indicators": "IG OAS 90bps; HY OAS 320bps; S&P P/E 21x",
        },
    },
    "factor_assessment": {
        "value": {"tilt": "neutral", "rationale": "HML trailing 12M flat at +1.2%"},
        "quality": {
            "tilt": "tailwind",
            "rationale": "RMW +3.5% YTD; late-cycle favours defensives",
        },
        "momentum": {
            "tilt": "tailwind",
            "rationale": "MOM +8% trailing 6M in trending market",
        },
        "size": {
            "tilt": "headwind",
            "rationale": "SMB -2.1% YTD; credit conditions favour large caps",
        },
    },
    "yield_curve": {
        "slope_10y2y_bps": 25,
        "shape": "normal",
        "trend": "steepening",
        "credit_spread_ig_bps": 90,
        "credit_spread_hy_bps": 320,
    },
    "macro_summary": "The US economy is in a late-cycle slowdown with moderating inflation.",  # noqa: E501
    "key_risks": ["Reacceleration of inflation", "Hard landing", "Geopolitical shock"],
    "recommended_positioning": {
        "beta_range": "0.7-1.0",
        "gross_exposure": "Maintain normal",
        "net_exposure": "Cautiously net long 40-50%",
        "sector_tilts": "Favour healthcare, energy; reduce consumer discretionary",
        "style_tilts": "Quality over growth; value vs. growth neutral",
    },
    "data_sources": [
        {
            "subagent": "economy-macro",
            "data_retrieved": "GDP, CPI, unemployment",
            "period": "2025-2026",
        }
    ],
    "limitations": [],
}


# ── Prompt template tests ──────────────────────────────────────────────────────


@pytest.mark.unit
class TestPromptTemplate:
    """Test market regime prompt template rendering."""

    def test_template_renders(self):
        result = render_template("market_regime.jinja")
        assert len(result) > 100

    def test_template_contains_subagents(self):
        result = render_template("market_regime.jinja")
        assert "economy-macro" in result
        assert "fixed-income" in result
        assert "fama-french" in result
        assert "currency-commodities" in result
        assert "etf-index" in result
        assert "data-validation" in result

    def test_template_contains_workflow_steps(self):
        result = render_template("market_regime.jinja")
        assert "Parse Context" in result
        assert "Collect Macro Data" in result
        assert "Validate Data" in result
        assert "Classify Regime" in result
        assert "Reflect" in result

    def test_template_contains_four_dimensions(self):
        result = render_template("market_regime.jinja")
        assert "Growth" in result
        assert "Inflation" in result
        assert "Monetary" in result
        assert "Liquidity" in result

    def test_template_contains_dimension_labels(self):
        result = render_template("market_regime.jinja")
        assert "expanding" in result
        assert "contracting" in result
        assert "high_rising" in result
        assert "deflationary" in result
        assert "aggressively_tightening" in result
        assert "risk_on" in result
        assert "crisis" in result

    def test_template_uses_structured_output_tool(self):
        result = render_template("market_regime.jinja")
        assert "structured output tool" in result
        assert "MARKET_REGIME_JSON_START" not in result
        assert "MARKET_REGIME_JSON_END" not in result

    def test_template_contains_output_keys(self):
        result = render_template("market_regime.jinja")
        assert "regime_label" in result
        assert "factor_assessment" in result
        assert "yield_curve" in result
        assert "recommended_positioning" in result
        assert "ticker_impact" in result

    def test_template_contains_grounding_constraint(self):
        result = render_template("market_regime.jinja")
        assert "NEVER" in result
        assert "fabricat" in result.lower()

    def test_template_contains_validation_loop(self):
        result = render_template("market_regime.jinja")
        assert "data-validation" in result
        assert "proceed" in result
        assert "collect_more_data" in result
        assert "insufficient_data" in result

    def test_template_contains_reflection_checks(self):
        result = render_template("market_regime.jinja")
        assert "Internal consistency" in result
        assert "Factor consistency" in result

    def test_template_contains_factor_guidance(self):
        result = render_template("market_regime.jinja")
        assert "tailwind" in result
        assert "headwind" in result
        assert "Value" in result
        assert "Quality" in result
        assert "Momentum" in result
        assert "Size" in result

    def test_template_sandbox_is_mandatory(self):
        result = render_template("market_regime.jinja")
        assert "MANDATORY" in result
        assert "execute_python" in result

    def test_template_contains_sandbox_computations(self):
        result = render_template("market_regime.jinja")
        assert "slope_10y2y_bps" in result
        assert "cpi_3m_annualised_pct" in result
        assert "ff_zscores" in result
        assert "policy_rate_distance_bps" in result
        assert "ig_oas_pctile" in result
        assert "copper_mom_pct" in result

    def test_template_contains_adverse_regime_guidance(self):
        result = render_template("market_regime.jinja")
        assert "adverse regime" in result.lower()
        assert "idiosyncratic alpha" in result

    def test_template_contains_investment_process_framing(self):
        result = render_template("market_regime.jinja")
        assert "top-down context" in result


# ── _build_task_description tests ─────────────────────────────────────────────


@pytest.mark.unit
class TestBuildTaskDescription:
    """Test task description construction from context."""

    def test_ticker_context(self):
        context: MarketRegimeContext = {"ticker": "AAPL", "query": "Quality tech stock"}
        result = _build_task_description(context)
        assert "AAPL" in result
        assert "Quality tech stock" in result
        assert "ticker_impact" in result

    def test_query_only_context(self):
        context: MarketRegimeContext = {"query": "Find defensive value stocks"}
        result = _build_task_description(context)
        assert "Find defensive value stocks" in result
        assert "Omit the ticker_impact field" in result
        assert "Ticker:" not in result

    def test_sector_industry_country_context(self):
        context: MarketRegimeContext = {
            "query": "European bank stocks",
            "sector": "Financials",
            "industry": "Banks",
            "country": "Europe",
        }
        result = _build_task_description(context)
        assert "Financials" in result
        assert "Banks" in result
        assert "Europe" in result
        assert "European bank stocks" in result

    def test_minimal_empty_context(self):
        result = _build_task_description({})
        assert "regime" in result.lower()

    def test_ticker_triggers_ticker_impact_instruction(self):
        context: MarketRegimeContext = {"ticker": "MSFT"}
        result = _build_task_description(context)
        assert "ticker_impact" in result
        assert "MSFT" in result

    def test_no_ticker_no_ticker_impact_instruction(self):
        context: MarketRegimeContext = {"query": "defensive stocks"}
        result = _build_task_description(context)
        assert "Omit the ticker_impact field" in result
        assert "Populate the ticker_impact" not in result


# ── MarketRegimeOutput Pydantic model tests ────────────────────────────────────


@pytest.mark.unit
class TestMarketRegimeOutputModel:
    """Test Pydantic model validation for MarketRegimeOutput."""

    def test_valid_full_output_validates(self):
        output = MarketRegimeOutput(**_VALID_REGIME_DICT)
        assert output.regime_label == "Goldilocks late-cycle"
        assert output.confidence == 0.8
        assert output.dimensions.growth_cycle.label == "slowing"
        assert output.factor_assessment.quality.tilt == "tailwind"
        assert output.yield_curve.shape == "normal"

    def test_ticker_impact_is_optional(self):
        output = MarketRegimeOutput(**_VALID_REGIME_DICT)
        assert output.ticker_impact is None

    def test_ticker_impact_populates_when_provided(self):
        data = {
            **_VALID_REGIME_DICT,
            "ticker_impact": {
                "ticker": "AAPL",
                "regime_sensitivity": "neutral",
                "rationale": "Tech sector neutral in late-cycle slowdown.",
                "specific_risks": ["Rate sensitivity"],
                "specific_tailwinds": ["Quality premium"],
            },
        }
        output = MarketRegimeOutput(**data)
        assert output.ticker_impact is not None
        assert output.ticker_impact.ticker == "AAPL"
        assert output.ticker_impact.regime_sensitivity == "neutral"

    def test_invalid_factor_tilt_raises_validation_error(self):
        data = {
            **_VALID_REGIME_DICT,
            "factor_assessment": {
                **_VALID_REGIME_DICT["factor_assessment"],
                "value": {"tilt": "bullish", "rationale": "bad value"},
            },
        }
        with pytest.raises(ValidationError):
            MarketRegimeOutput(**data)

    def test_invalid_yield_curve_shape_raises_validation_error(self):
        data = {
            **_VALID_REGIME_DICT,
            "yield_curve": {
                **_VALID_REGIME_DICT["yield_curve"],
                "shape": "steep",
            },
        }
        with pytest.raises(ValidationError):
            MarketRegimeOutput(**data)

    def test_model_dump_serialises_correctly(self):
        output = MarketRegimeOutput(**_VALID_REGIME_DICT)
        dumped = output.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["regime_label"] == "Goldilocks late-cycle"
        assert dumped["dimensions"]["growth_cycle"]["label"] == "slowing"
        assert dumped["ticker_impact"] is None
        assert isinstance(dumped["data_sources"], list)
        assert isinstance(dumped["limitations"], list)


# ── MarketRegimeContext TypedDict tests ────────────────────────────────────────


@pytest.mark.unit
class TestMarketRegimeContextTypedDict:
    """Test MarketRegimeContext construction in the node function."""

    @pytest.mark.asyncio
    async def test_node_builds_context_with_ticker_and_query(self):
        """Node constructs context with ticker + query when both present in state."""
        mock_structured = MagicMock(spec=MarketRegimeOutput)
        mock_structured.model_dump.return_value = _VALID_REGIME_DICT
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"structured_response": mock_structured}

        with (
            patch(
                "muffin_agent.agents.investment.market_regime"
                ".create_market_regime_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.market_regime"
                ".Configuration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            from muffin_agent.agents.investment.market_regime import market_regime_node

            state = {"ticker": "AAPL", "query": "Quality tech stock"}
            await market_regime_node(state, MagicMock())  # type: ignore[arg-type]

        task_input = mock_agent.ainvoke.call_args[0][0]["input"]
        assert "AAPL" in task_input
        assert "Quality tech stock" in task_input

    @pytest.mark.asyncio
    async def test_node_omits_ticker_when_absent(self):
        """Node omits ticker key when state has no ticker."""
        mock_structured = MagicMock(spec=MarketRegimeOutput)
        mock_structured.model_dump.return_value = _VALID_REGIME_DICT
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"structured_response": mock_structured}

        with (
            patch(
                "muffin_agent.agents.investment.market_regime"
                ".create_market_regime_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.market_regime"
                ".Configuration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            from muffin_agent.agents.investment.market_regime import market_regime_node

            state = {"query": "Defensive value stocks"}
            await market_regime_node(state, MagicMock())  # type: ignore[arg-type]

        task_input = mock_agent.ainvoke.call_args[0][0]["input"]
        assert "Ticker:" not in task_input

    @pytest.mark.asyncio
    async def test_node_passes_sector_industry_country(self):
        """Sector, industry, country flow through from state to task description."""
        mock_structured = MagicMock(spec=MarketRegimeOutput)
        mock_structured.model_dump.return_value = _VALID_REGIME_DICT
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"structured_response": mock_structured}

        with (
            patch(
                "muffin_agent.agents.investment.market_regime"
                ".create_market_regime_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.market_regime"
                ".Configuration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            from muffin_agent.agents.investment.market_regime import market_regime_node

            state = {
                "sector": "Financials",
                "industry": "Banks",
                "country": "Europe",
                "query": "European bank stocks",
            }
            await market_regime_node(state, MagicMock())  # type: ignore[arg-type]

        task_input = mock_agent.ainvoke.call_args[0][0]["input"]
        assert "Financials" in task_input
        assert "Banks" in task_input
        assert "Europe" in task_input


# ── create_market_regime_agent tests ──────────────────────────────────────────


@pytest.mark.unit
class TestCreateMarketRegimeAgent:
    """Test agent factory function."""

    @pytest.mark.asyncio
    async def test_creates_agent_with_macro_subagents(self):
        mock_economy_macro = MagicMock()
        mock_fixed_income = MagicMock()
        mock_fama_french = MagicMock()
        mock_currency_commodities = MagicMock()
        mock_etf_index = MagicMock()
        mock_validation = MagicMock()

        config = MagicMock()
        config.get_llm.return_value = MagicMock()

        with (
            patch(
                "muffin_agent.agents.investment.market_regime"
                ".create_economy_macro_data_collection_agent",
                new_callable=AsyncMock,
                return_value=mock_economy_macro,
            ),
            patch(
                "muffin_agent.agents.investment.market_regime"
                ".create_fixed_income_data_collection_agent",
                new_callable=AsyncMock,
                return_value=mock_fixed_income,
            ),
            patch(
                "muffin_agent.agents.investment.market_regime"
                ".create_fama_french_data_collection_agent",
                new_callable=AsyncMock,
                return_value=mock_fama_french,
            ),
            patch(
                "muffin_agent.agents.investment.market_regime"
                ".create_currency_commodities_data_collection_agent",
                new_callable=AsyncMock,
                return_value=mock_currency_commodities,
            ),
            patch(
                "muffin_agent.agents.investment.market_regime"
                ".create_etf_index_data_collection_agent",
                new_callable=AsyncMock,
                return_value=mock_etf_index,
            ),
            patch(
                "muffin_agent.agents.investment.market_regime"
                ".create_data_validation_agent",
                new_callable=AsyncMock,
                return_value=mock_validation,
            ),
            patch(
                "muffin_agent.agents.investment.market_regime.create_deep_agent"
            ) as mock_create,
        ):
            mock_create.return_value = MagicMock()

            from muffin_agent.agents.investment.market_regime import (
                MarketRegimeOutput,
                create_market_regime_agent,
            )

            agent = await create_market_regime_agent(config)

            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args
            assert call_kwargs.kwargs["model"] == config.get_llm.return_value
            subagents = call_kwargs.kwargs["subagents"]
            assert len(subagents) == 6
            assert subagents[0]["name"] == "economy-macro"
            assert subagents[0]["runnable"] is mock_economy_macro
            assert subagents[1]["name"] == "fixed-income"
            assert subagents[1]["runnable"] is mock_fixed_income
            assert subagents[2]["name"] == "fama-french"
            assert subagents[2]["runnable"] is mock_fama_french
            assert subagents[3]["name"] == "currency-commodities"
            assert subagents[3]["runnable"] is mock_currency_commodities
            assert subagents[4]["name"] == "etf-index"
            assert subagents[4]["runnable"] is mock_etf_index
            assert subagents[5]["name"] == "data-validation"
            assert subagents[5]["runnable"] is mock_validation
            assert agent is mock_create.return_value

            # Verify response_format uses AutoStrategy with the correct schema
            response_format = call_kwargs.kwargs["response_format"]
            from langchain.agents.structured_output import AutoStrategy

            assert isinstance(response_format, AutoStrategy)
            assert response_format.schema is MarketRegimeOutput

    @pytest.mark.asyncio
    async def test_uses_backend(self):
        """Verify that get_backend is passed as sandbox backend."""
        config = MagicMock()
        config.get_llm.return_value = MagicMock()

        with (
            patch(
                "muffin_agent.agents.investment.market_regime"
                ".create_economy_macro_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.market_regime"
                ".create_fixed_income_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.market_regime"
                ".create_fama_french_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.market_regime"
                ".create_currency_commodities_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.market_regime"
                ".create_etf_index_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.market_regime"
                ".create_data_validation_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.market_regime.create_deep_agent"
            ) as mock_create,
            patch(
                "muffin_agent.agents.investment.market_regime.get_backend"
            ) as mock_backend,
        ):
            mock_create.return_value = MagicMock()

            from muffin_agent.agents.investment.market_regime import (
                create_market_regime_agent,
            )

            await create_market_regime_agent(config)

            call_kwargs = mock_create.call_args
            assert call_kwargs.kwargs["backend"] is mock_backend


# ── market_regime_node tests ───────────────────────────────────────────────────


@pytest.mark.unit
class TestMarketRegimeNode:
    """Test the LangGraph node function."""

    @pytest.mark.asyncio
    async def test_node_returns_market_regime_key(self):
        mock_structured = MagicMock(spec=MarketRegimeOutput)
        mock_structured.model_dump.return_value = _VALID_REGIME_DICT
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"structured_response": mock_structured}

        with (
            patch(
                "muffin_agent.agents.investment.market_regime"
                ".create_market_regime_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.market_regime"
                ".Configuration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            from muffin_agent.agents.investment.market_regime import market_regime_node

            state = {"ticker": "AAPL", "query": "Quality tech stock"}
            result = await market_regime_node(state, MagicMock())  # type: ignore[arg-type]

        assert "market_regime" in result
        regime = result["market_regime"]
        assert regime["regime_label"] == "Goldilocks late-cycle"
        assert "dimensions" in regime
        assert "factor_assessment" in regime
        assert "yield_curve" in regime
        assert "recommended_positioning" in regime

    @pytest.mark.asyncio
    async def test_node_passes_ticker_in_task(self):
        mock_structured = MagicMock(spec=MarketRegimeOutput)
        mock_structured.model_dump.return_value = _VALID_REGIME_DICT
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"structured_response": mock_structured}

        with (
            patch(
                "muffin_agent.agents.investment.market_regime"
                ".create_market_regime_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.market_regime"
                ".Configuration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            from muffin_agent.agents.investment.market_regime import market_regime_node

            state = {"ticker": "NVDA", "query": "AI infrastructure"}
            await market_regime_node(state, MagicMock())  # type: ignore[arg-type]

        call_args = mock_agent.ainvoke.call_args
        task_input = call_args[0][0]["input"]
        assert "NVDA" in task_input

    @pytest.mark.asyncio
    async def test_node_works_without_ticker(self):
        mock_structured = MagicMock(spec=MarketRegimeOutput)
        mock_structured.model_dump.return_value = _VALID_REGIME_DICT
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"structured_response": mock_structured}

        with (
            patch(
                "muffin_agent.agents.investment.market_regime"
                ".create_market_regime_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.market_regime"
                ".Configuration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            from muffin_agent.agents.investment.market_regime import market_regime_node

            state = {"query": "Defensive value stocks"}
            result = await market_regime_node(state, MagicMock())  # type: ignore[arg-type]

        assert "market_regime" in result

    @pytest.mark.asyncio
    async def test_node_handles_missing_structured_response(self):
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "output": "Sorry, I could not complete the analysis."
        }

        with (
            patch(
                "muffin_agent.agents.investment.market_regime"
                ".create_market_regime_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.market_regime"
                ".Configuration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            from muffin_agent.agents.investment.market_regime import market_regime_node

            state = {"query": "Tech stocks"}
            result = await market_regime_node(state, MagicMock())  # type: ignore[arg-type]

        assert "market_regime" in result
        assert result["market_regime"]["regime_label"] == "unknown"
        assert "error" in result["market_regime"]

    @pytest.mark.asyncio
    async def test_node_handles_none_structured_response(self):
        """Explicit None in structured_response key triggers fallback."""
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "structured_response": None,
            "output": "Partial analysis text.",
        }

        with (
            patch(
                "muffin_agent.agents.investment.market_regime"
                ".create_market_regime_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.market_regime"
                ".Configuration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            from muffin_agent.agents.investment.market_regime import market_regime_node

            result = await market_regime_node(
                {"query": "test"},
                MagicMock(),  # type: ignore[arg-type]
            )

        assert result["market_regime"]["regime_label"] == "unknown"
        assert "error" in result["market_regime"]
        assert result["market_regime"]["raw_output"] == "Partial analysis text."
