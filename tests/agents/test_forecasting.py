"""Tests for the forecasting investment agent (Step 6)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from deepagents import CompiledSubAgent
from pydantic import ValidationError

from muffin_agent.agents.investment.forecasting import (
    ForecastingInputState,
    ForecastOutput,
    create_forecasting_agent,
    forecasting_node,
)
from muffin_agent.prompts import render_template

# ── Minimal valid output dict ──────────────────────────────────────────────────

_YEARLY_PROJ_Y1 = {
    "year": 2026,
    "revenue": 420.0e9,
    "revenue_growth_pct": 9.6,
    "ebitda": 130.0e9,
    "ebitda_margin_pct": 31.0,
    "ebit": 120.0e9,
    "ebit_margin_pct": 28.6,
    "eps": 6.85,
    "fcf": 112.0e9,
    "fcf_margin_pct": 26.7,
}

_YEARLY_PROJ_Y2 = {
    "year": 2027,
    "revenue": 458.0e9,
    "revenue_growth_pct": 9.0,
    "ebitda": 144.0e9,
    "ebitda_margin_pct": 31.4,
    "ebit": 133.0e9,
    "ebit_margin_pct": 29.0,
    "eps": 7.60,
    "fcf": 124.0e9,
    "fcf_margin_pct": 27.1,
}

_YEARLY_PROJ_Y3 = {
    "year": 2028,
    "revenue": 495.0e9,
    "revenue_growth_pct": 8.1,
    "ebitda": 157.0e9,
    "ebitda_margin_pct": 31.7,
    "ebit": 145.0e9,
    "ebit_margin_pct": 29.3,
    "eps": 8.35,
    "fcf": 135.0e9,
    "fcf_margin_pct": 27.3,
}

_BASE_CASE = {
    "label": "base",
    "probability": 0.60,
    "revenue_cagr_3y_pct": 8.9,
    "ebitda_margin_exit_pct": 31.7,
    "eps_cagr_3y_pct": 9.2,
    "key_assumptions": [
        "Revenue CAGR 8-9% driven by Services mix shift and iPhone replacement cycle",
        "EBITDA margins stable at 31-32% reflecting operating leverage in Services",
        "Capex intensity at 2.7% of revenue (in line with 5Y historical average)",
        "Diluted share count declining ~2%/year via buybacks",
        "Effective tax rate 16% (in line with recent actuals)",
    ],
    "probability_rationale": (
        "Probability anchored to company_signal='pass' default (60%). "
        "No material deviations warranted given strong historical execution."
    ),
    "narrative": (
        "The base case reflects continued Services mix shift driving margin stability "
        "and mid-single-digit iPhone unit growth from upgrade cycle tailwinds. "
        "Revenue growth is consistent with the 3Y historical CAGR and "
        "consensus estimates."
    ),
    "projections": [_YEARLY_PROJ_Y1, _YEARLY_PROJ_Y2, _YEARLY_PROJ_Y3],
}

_BULL_CASE = {
    "label": "bull",
    "probability": 0.25,
    "revenue_cagr_3y_pct": 12.5,
    "ebitda_margin_exit_pct": 33.5,
    "eps_cagr_3y_pct": 14.0,
    "key_assumptions": [
        "AI-driven iPhone 'super cycle' accelerates upgrade rate to 22% vs 18% base",
        "Services attach rate on installed base expands from 5.2 to 6.1 "
        "products/device",
        "Gross margin expansion of +1.5pp from silicon cost efficiencies",
        "Buyback pace maintained at $90B+/year",
        "India manufacturing ramp reduces tariff exposure and opens a new "
        "growth market",
    ],
    "probability_rationale": (
        "25% probability reflecting upside optionality around AI product cycle. "
        "Bull catalysts are plausible but require multiple simultaneous tailwinds."
    ),
    "narrative": (
        "The bull case is driven by an AI-powered iPhone super-cycle and Services "
        "monetisation acceleration. Faster attach rate growth and silicon cost savings "
        "drive margin expansion beyond the base case trajectory."
    ),
    "projections": [
        {**_YEARLY_PROJ_Y1, "revenue": 435.0e9, "eps": 7.40},
        {**_YEARLY_PROJ_Y2, "revenue": 490.0e9, "eps": 8.45},
        {**_YEARLY_PROJ_Y3, "revenue": 550.0e9, "eps": 9.60},
    ],
}

_BEAR_CASE = {
    "label": "bear",
    "probability": 0.15,
    "revenue_cagr_3y_pct": 3.5,
    "ebitda_margin_exit_pct": 29.0,
    "eps_cagr_3y_pct": 2.0,
    "key_assumptions": [
        "China revenue declines 15% due to Huawei competition and "
        "geopolitical restrictions",
        "App Store take-rate cut to 17% from 30% under EU DMA enforcement",
        "Google Search default agreement terminated, removing ~$18-20B annual revenue",
        "Capex elevated at 3.5% of revenue for India manufacturing transition",
        "Services gross margin compresses 2pp from regulatory compliance costs",
    ],
    "probability_rationale": (
        "15% probability given company_signal='pass' anchor. Downside "
        "scenarios require concurrent realisation of China, regulatory, and "
        "Google risks — unlikely but plausible within a 3-year horizon given "
        "the pace of regulatory actions."
    ),
    "narrative": (
        "The bear case is driven by simultaneous China revenue pressure and App Store "
        "regulatory headwinds, consistent with market_regime key risks around "
        "geopolitical decoupling and tech platform regulation. Margin compression "
        "is amplified by elevated capex for supply chain diversification."
    ),
    "projections": [
        {**_YEARLY_PROJ_Y1, "revenue": 395.0e9, "eps": 6.10},
        {**_YEARLY_PROJ_Y2, "revenue": 405.0e9, "eps": 6.20},
        {**_YEARLY_PROJ_Y3, "revenue": 415.0e9, "eps": 6.30},
    ],
}

_CONSENSUS_ANCHOR = {
    "as_of_date": "2026-03-21",
    "num_analysts": 42,
    "eps_year1": 6.95,
    "eps_year2": 7.70,
    "revenue_year1": 422.0e9,
    "ebitda_year1": 132.0e9,
    "price_target_mean": 215.0,
    "price_target_low": 170.0,
    "price_target_high": 260.0,
    "revision_trend_3m": "flat",
    "surprise_history": (
        "Apple beat consensus EPS in 7 of the last 8 quarters by an average of 3.2%. "
        "The single miss in Q1 FY2025 was driven by China softness and one-time FX "
        "headwinds; guidance for subsequent quarters was in-line with consensus."
    ),
}

_VALID_FORECAST_DICT = {
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "currency": "USD",
    "forecast_as_of_date": "2026-03-21",
    "base_case": _BASE_CASE,
    "bull_case": _BULL_CASE,
    "bear_case": _BEAR_CASE,
    "consensus_anchoring": _CONSENSUS_ANCHOR,
    "revision_momentum": "flat",
    "sensitivity_table": [
        {
            "driver": "Revenue growth +1pp vs base (Year+1)",
            "eps_impact_pct": 1.8,
            "fcf_impact_pct": 2.1,
        },
        {
            "driver": "EBITDA margin +1pp vs base (Year+1)",
            "eps_impact_pct": 3.4,
            "fcf_impact_pct": 3.4,
        },
        {
            "driver": "Capex -10% vs base (Year+1)",
            "eps_impact_pct": None,
            "fcf_impact_pct": 1.0,
        },
    ],
    "earnings_quality_flags": [],
    "modeling_notes": (
        "Base case assumes 16% effective tax rate, D&A at 2.3% of revenue, and capex "
        "at 2.7% of revenue consistent with the 5-year historical average. The single "
        "event most likely to shift the base case is resolution of the DOJ antitrust "
        "action on the Google Search default payment (~$18-20B impact)."
    ),
    "data_sources": [
        {
            "subagent": "equity-estimates",
            "data_retrieved": (
                "Consensus EPS/revenue/EBITDA Y+1/Y+2, price targets (mean/high/low), "
                "historical consensus EPS for 6M revision history, forward PE"
            ),
            "period": "2026-03-21",
        },
        {
            "subagent": "equity-fundamentals",
            "data_retrieved": (
                "Diluted shares outstanding, D&A, effective tax rate, capex (3Y), "
                "operating cash flow, total assets"
            ),
            "period": "FY2023-FY2025",
        },
    ],
    "confidence": 0.90,
    "limitations": [],
}


# ── Prompt template tests ──────────────────────────────────────────────────────


@pytest.mark.unit
class TestPromptTemplate:
    """Verify prompt renders and contains required structural elements."""

    def test_renders(self):
        result = render_template("investment/forecasting.jinja")
        assert len(result) > 200

    def test_contains_all_subagent_names(self):
        result = render_template("investment/forecasting.jinja")
        for name in [
            "equity-estimates",
            "equity-fundamentals",
            "economy-macro",
            "currency-commodities",
            "data-validation",
        ]:
            assert name in result

    def test_contains_workflow_steps(self):
        result = render_template("investment/forecasting.jinja")
        for step in ["Step 1", "Step 2", "Validate Data", "Step 4", "Step 5"]:
            assert step in result

    def test_contains_step_labels(self):
        result = render_template("investment/forecasting.jinja")
        assert "Parse Context" in result
        assert "Collect" in result
        assert "Validate" in result
        assert "Reflect" in result

    def test_contains_scenario_labels(self):
        result = render_template("investment/forecasting.jinja")
        assert "base" in result.lower()
        assert "bull" in result.lower()
        assert "bear" in result.lower()

    def test_contains_probability_anchors(self):
        result = render_template("investment/forecasting.jinja")
        assert "company_signal" in result
        assert "0.60" in result or "60" in result
        assert "0.25" in result or "25" in result
        assert "0.15" in result or "15" in result

    def test_contains_key_output_fields(self):
        result = render_template("investment/forecasting.jinja")
        for field in [
            "base_case",
            "bull_case",
            "bear_case",
            "consensus_anchoring",
            "revision_momentum",
            "earnings_quality_flags",
            "sensitivity_table",
        ]:
            assert field in result

    def test_grounding_constraint_present(self):
        result = render_template("investment/forecasting.jinja")
        assert "NEVER" in result

    def test_sandbox_mandatory_marker(self):
        result = render_template("investment/forecasting.jinja")
        assert "MANDATORY" in result
        assert "project_three_year_financials" in result

    def test_sandbox_block_labels_present(self):
        result = render_template("investment/forecasting.jinja")
        assert "Block A" in result
        assert "Block B" in result
        assert "Block C" in result
        assert "Block D" in result
        assert "Block E" in result

    def test_consensus_calibration_instruction_present(self):
        result = render_template("investment/forecasting.jinja")
        assert "15%" in result or "±15%" in result

    def test_reflection_step_present(self):
        result = render_template("investment/forecasting.jinja")
        assert "consistency" in result.lower() or "reflect" in result.lower()

    def test_validation_loop_present(self):
        result = render_template("investment/forecasting.jinja")
        assert "data-validation" in result
        assert "proceed" in result
        assert "collect_more_data" in result
        assert "insufficient_data" in result

    def test_structured_output_tool_instruction(self):
        result = render_template("investment/forecasting.jinja")
        assert "structured output tool" in result

    def test_no_fabrication_clause(self):
        result = render_template("investment/forecasting.jinja")
        assert "fabricate" in result.lower() or "NEVER estimate" in result

    def test_eps_null_handling_mentioned(self):
        result = render_template("investment/forecasting.jinja")
        # Guidance that eps may be null if shares unavailable
        assert "null" in result.lower() or "None" in result

    def test_revision_momentum_formula_present(self):
        result = render_template("investment/forecasting.jinja")
        assert "revision" in result.lower()
        assert "upward" in result
        assert "downward" in result


# ── ForecastingInputState schema tests ────────────────────────────────────────


@pytest.mark.unit
class TestForecastingInputState:
    """Verify InputState TypedDict structure."""

    def test_all_fields_optional(self):
        # total=False: empty dict is valid
        state: ForecastingInputState = {}
        assert isinstance(state, dict)

    def test_annotations_contain_expected_keys(self):
        keys = set(ForecastingInputState.__annotations__)
        assert "ticker" in keys
        assert "query" in keys
        assert "company_analysis" in keys
        assert "market_regime" in keys

    def test_annotations_exact(self):
        keys = set(ForecastingInputState.__annotations__)
        assert keys == {"ticker", "query", "company_analysis", "market_regime"}

    def test_ticker_only_valid(self):
        state: ForecastingInputState = {"ticker": "AAPL"}
        assert state["ticker"] == "AAPL"

    def test_full_pipeline_state_valid(self):
        state: ForecastingInputState = {
            "ticker": "AAPL",
            "query": "AI compounder thesis",
            "company_analysis": {"company_signal": "pass"},
            "market_regime": {"regime_label": "Goldilocks expansion"},
        }
        assert state["ticker"] == "AAPL"
        assert state["company_analysis"]["company_signal"] == "pass"


# ── ForecastOutput Pydantic model tests ───────────────────────────────────────


@pytest.mark.unit
class TestForecastOutputModel:
    """Verify Pydantic output schema validation."""

    def test_valid_full_instance(self):
        output = ForecastOutput(**_VALID_FORECAST_DICT)
        assert output.ticker == "AAPL"
        assert output.currency == "USD"
        assert output.revision_momentum == "flat"

    def test_scenario_labels_correct(self):
        output = ForecastOutput(**_VALID_FORECAST_DICT)
        assert output.base_case.label == "base"
        assert output.bull_case.label == "bull"
        assert output.bear_case.label == "bear"

    def test_scenario_probabilities_sum_to_one(self):
        output = ForecastOutput(**_VALID_FORECAST_DICT)
        total = (
            output.base_case.probability
            + output.bull_case.probability
            + output.bear_case.probability
        )
        assert total == pytest.approx(1.0, abs=0.02)

    def test_base_case_has_three_projections(self):
        output = ForecastOutput(**_VALID_FORECAST_DICT)
        assert len(output.base_case.projections) == 3
        years = [p.year for p in output.base_case.projections]
        assert years == [2026, 2027, 2028]

    def test_yearly_projection_fields(self):
        output = ForecastOutput(**_VALID_FORECAST_DICT)
        proj = output.base_case.projections[0]
        assert proj.year == 2026
        assert proj.revenue == pytest.approx(420.0e9)
        assert proj.ebitda_margin_pct == pytest.approx(31.0)
        assert proj.eps == pytest.approx(6.85)

    def test_eps_nullable(self):
        data = {**_VALID_FORECAST_DICT}
        base = {**_BASE_CASE, "eps_cagr_3y_pct": None}
        base_projs = [
            {**_YEARLY_PROJ_Y1, "eps": None},
            {**_YEARLY_PROJ_Y2, "eps": None},
            {**_YEARLY_PROJ_Y3, "eps": None},
        ]
        base["projections"] = base_projs
        data["base_case"] = base
        output = ForecastOutput(**data)
        assert output.base_case.eps_cagr_3y_pct is None
        assert output.base_case.projections[0].eps is None
        dumped = output.model_dump(exclude_none=True)
        assert "eps" not in dumped["base_case"]["projections"][0]

    def test_consensus_anchor_fields(self):
        output = ForecastOutput(**_VALID_FORECAST_DICT)
        ca = output.consensus_anchoring
        assert ca.num_analysts == 42
        assert ca.eps_year1 == pytest.approx(6.95)
        assert ca.revision_trend_3m == "flat"
        assert isinstance(ca.surprise_history, str)

    def test_defaults_for_list_fields(self):
        data = {
            k: v
            for k, v in _VALID_FORECAST_DICT.items()
            if k
            not in (
                "data_sources",
                "limitations",
                "sensitivity_table",
                "earnings_quality_flags",
            )
        }
        output = ForecastOutput(**data)
        assert output.data_sources == []
        assert output.limitations == []
        assert output.sensitivity_table == []
        assert output.earnings_quality_flags == []

    def test_invalid_scenario_label_raises(self):
        data = {**_VALID_FORECAST_DICT}
        data["base_case"] = {**_BASE_CASE, "label": "neutral"}  # invalid
        with pytest.raises(ValidationError):
            ForecastOutput(**data)

    def test_invalid_revision_momentum_raises(self):
        data = {**_VALID_FORECAST_DICT, "revision_momentum": "rising"}  # invalid
        with pytest.raises(ValidationError):
            ForecastOutput(**data)

    def test_invalid_consensus_revision_trend_raises(self):
        data = {**_VALID_FORECAST_DICT}
        data["consensus_anchoring"] = {
            **_CONSENSUS_ANCHOR,
            "revision_trend_3m": "neutral",  # invalid
        }
        with pytest.raises(ValidationError):
            ForecastOutput(**data)

    def test_model_dump_serializable(self):
        output = ForecastOutput(**_VALID_FORECAST_DICT)
        dumped = output.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["ticker"] == "AAPL"
        assert isinstance(dumped["base_case"]["projections"], list)
        assert isinstance(dumped["sensitivity_table"], list)
        assert isinstance(dumped["earnings_quality_flags"], list)
        assert isinstance(dumped["limitations"], list)


# ── Node JSON input tests ──────────────────────────────────────────────────────


@pytest.mark.unit
class TestForecastingNodeJsonInput:
    """Verify the node serializes state context correctly."""

    @pytest.mark.asyncio
    async def test_passes_ticker_and_query_in_input(self):
        mock_structured = MagicMock(spec=ForecastOutput)
        mock_structured.model_dump.return_value = _VALID_FORECAST_DICT
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"structured_response": mock_structured}

        with (
            patch(
                "muffin_agent.agents.investment.forecasting.create_forecasting_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.forecasting"
                ".ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            await forecasting_node(
                {"ticker": "AAPL", "query": "AI compounder thesis"},
                MagicMock(),  # type: ignore[arg-type]
            )

        raw = mock_agent.ainvoke.call_args[0][0]["input"]
        ctx = json.loads(raw)
        assert ctx["ticker"] == "AAPL"
        assert ctx["query"] == "AI compounder thesis"

    @pytest.mark.asyncio
    async def test_passes_upstream_state_fields(self):
        mock_structured = MagicMock(spec=ForecastOutput)
        mock_structured.model_dump.return_value = _VALID_FORECAST_DICT
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"structured_response": mock_structured}

        company_analysis_data = {"company_signal": "pass", "financial_history": {}}
        market_regime_data = {"regime_label": "Goldilocks expansion"}

        with (
            patch(
                "muffin_agent.agents.investment.forecasting.create_forecasting_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.forecasting"
                ".ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            await forecasting_node(
                {
                    "ticker": "AAPL",
                    "query": "AI thesis",
                    "company_analysis": company_analysis_data,
                    "market_regime": market_regime_data,
                },
                MagicMock(),  # type: ignore[arg-type]
            )

        raw = mock_agent.ainvoke.call_args[0][0]["input"]
        ctx = json.loads(raw)
        assert ctx["company_analysis"]["company_signal"] == "pass"
        assert ctx["market_regime"]["regime_label"] == "Goldilocks expansion"

    @pytest.mark.asyncio
    async def test_omits_missing_state_fields(self):
        mock_structured = MagicMock(spec=ForecastOutput)
        mock_structured.model_dump.return_value = _VALID_FORECAST_DICT
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"structured_response": mock_structured}

        with (
            patch(
                "muffin_agent.agents.investment.forecasting.create_forecasting_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.forecasting"
                ".ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            await forecasting_node({"query": "tech sector"}, MagicMock())  # type: ignore[arg-type]

        raw = mock_agent.ainvoke.call_args[0][0]["input"]
        ctx = json.loads(raw)
        assert "ticker" not in ctx
        assert "company_analysis" not in ctx
        assert "market_regime" not in ctx
        assert ctx["query"] == "tech sector"

    @pytest.mark.asyncio
    async def test_excludes_non_input_state_fields(self):
        """Fields outside ForecastingInputState are not sent to the agent."""
        mock_structured = MagicMock(spec=ForecastOutput)
        mock_structured.model_dump.return_value = _VALID_FORECAST_DICT
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"structured_response": mock_structured}

        with (
            patch(
                "muffin_agent.agents.investment.forecasting.create_forecasting_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.forecasting"
                ".ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            # Simulate full TickerAnalysisState with extra fields
            state = {
                "ticker": "AAPL",
                "query": "quality tech",
                "company_analysis": {"company_signal": "pass"},
                "market_regime": {"regime_label": "Goldilocks"},
                "sector_view": {"sector": "Information Technology"},
                "risk_assessment": None,
            }
            await forecasting_node(state, MagicMock())  # type: ignore[arg-type]

        raw = mock_agent.ainvoke.call_args[0][0]["input"]
        ctx = json.loads(raw)
        assert set(ctx.keys()) <= set(ForecastingInputState.__annotations__)
        assert "sector_view" not in ctx
        assert "risk_assessment" not in ctx


# ── create_forecasting_agent tests ────────────────────────────────────────────


@pytest.mark.unit
class TestCreateForecastingAgent:
    """Verify agent factory wires subagents and response_format correctly."""

    @pytest.mark.asyncio
    async def test_creates_five_subagents(self):
        config = MagicMock()
        config.get_llm.return_value = MagicMock()

        with (
            patch(
                "muffin_agent.agents.investment.forecasting"
                ".ModelConfiguration.from_runnable_config",
                return_value=config,
            ),
            patch(
                "muffin_agent.agents.investment.forecasting"
                ".create_equity_estimates_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.forecasting"
                ".create_equity_fundamentals_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.forecasting"
                ".create_economy_macro_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.forecasting"
                ".create_currency_commodities_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.forecasting.build_validation_subagent",
                new_callable=AsyncMock,
                return_value=CompiledSubAgent(
                    name="data-validation",
                    description="mock validation",
                    runnable=MagicMock(),
                ),
            ),
            patch(
                "muffin_agent.agents.investment.forecasting.create_deep_agent"
            ) as mock_create,
        ):
            mock_create.return_value = MagicMock()
            await create_forecasting_agent(config)

            call_kwargs = mock_create.call_args.kwargs
            assert len(call_kwargs["subagents"]) == 5  # 4 data + 1 validation

    @pytest.mark.asyncio
    async def test_subagent_names_in_order(self):
        config = MagicMock()
        config.get_llm.return_value = MagicMock()

        with (
            patch(
                "muffin_agent.agents.investment.forecasting"
                ".ModelConfiguration.from_runnable_config",
                return_value=config,
            ),
            patch(
                "muffin_agent.agents.investment.forecasting"
                ".create_equity_estimates_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.forecasting"
                ".create_equity_fundamentals_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.forecasting"
                ".create_economy_macro_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.forecasting"
                ".create_currency_commodities_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.forecasting.build_validation_subagent",
                new_callable=AsyncMock,
                return_value=CompiledSubAgent(
                    name="data-validation",
                    description="mock validation",
                    runnable=MagicMock(),
                ),
            ),
            patch(
                "muffin_agent.agents.investment.forecasting.create_deep_agent"
            ) as mock_create,
        ):
            mock_create.return_value = MagicMock()
            await create_forecasting_agent(config)

            subagents = mock_create.call_args.kwargs["subagents"]
            names = [s["name"] for s in subagents]
            assert names == [
                "equity-estimates",
                "equity-fundamentals",
                "economy-macro",
                "currency-commodities",
                "data-validation",
            ]

    @pytest.mark.asyncio
    async def test_passes_get_backend(self):
        config = MagicMock()
        config.get_llm.return_value = MagicMock()

        with (
            patch(
                "muffin_agent.agents.investment.forecasting"
                ".ModelConfiguration.from_runnable_config",
                return_value=config,
            ),
            patch(
                "muffin_agent.agents.investment.forecasting"
                ".create_equity_estimates_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.forecasting"
                ".create_equity_fundamentals_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.forecasting"
                ".create_economy_macro_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.forecasting"
                ".create_currency_commodities_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.forecasting.build_validation_subagent",
                new_callable=AsyncMock,
                return_value=CompiledSubAgent(
                    name="data-validation",
                    description="mock validation",
                    runnable=MagicMock(),
                ),
            ),
            patch(
                "muffin_agent.agents.investment.forecasting.create_deep_agent"
            ) as mock_create,
            patch(
                "muffin_agent.agents.investment.forecasting.get_backend"
            ) as mock_backend,
        ):
            mock_create.return_value = MagicMock()
            await create_forecasting_agent(config)

            assert mock_create.call_args.kwargs["backend"] is mock_backend

    @pytest.mark.asyncio
    async def test_uses_auto_strategy_response_format(self):
        config = MagicMock()
        config.get_llm.return_value = MagicMock()

        with (
            patch(
                "muffin_agent.agents.investment.forecasting"
                ".ModelConfiguration.from_runnable_config",
                return_value=config,
            ),
            patch(
                "muffin_agent.agents.investment.forecasting"
                ".create_equity_estimates_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.forecasting"
                ".create_equity_fundamentals_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.forecasting"
                ".create_economy_macro_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.forecasting"
                ".create_currency_commodities_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.forecasting.build_validation_subagent",
                new_callable=AsyncMock,
                return_value=CompiledSubAgent(
                    name="data-validation",
                    description="mock validation",
                    runnable=MagicMock(),
                ),
            ),
            patch(
                "muffin_agent.agents.investment.forecasting.create_deep_agent"
            ) as mock_create,
        ):
            mock_create.return_value = MagicMock()
            await create_forecasting_agent(config)

            from langchain.agents.structured_output import AutoStrategy

            response_format = mock_create.call_args.kwargs["response_format"]
            assert isinstance(response_format, AutoStrategy)
            assert response_format.schema is ForecastOutput


# ── forecasting_node tests ────────────────────────────────────────────────────


@pytest.mark.unit
class TestForecastingNode:
    """Verify node behavior: output key, structured response, error fallback."""

    @pytest.mark.asyncio
    async def test_returns_forecast_key(self):
        mock_structured = MagicMock(spec=ForecastOutput)
        mock_structured.model_dump.return_value = _VALID_FORECAST_DICT
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"structured_response": mock_structured}

        with (
            patch(
                "muffin_agent.agents.investment.forecasting.create_forecasting_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.forecasting"
                ".ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            result = await forecasting_node(
                {"ticker": "AAPL", "query": "AI compounder"},
                MagicMock(),  # type: ignore[arg-type]
            )

        assert "forecast" in result
        fc = result["forecast"]
        assert fc["ticker"] == "AAPL"
        assert "base_case" in fc
        assert "bull_case" in fc
        assert "bear_case" in fc
        assert "consensus_anchoring" in fc

    @pytest.mark.asyncio
    async def test_error_fallback_on_missing_structured_response(self):
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "output": "Sorry, I could not produce a forecast."
        }

        with (
            patch(
                "muffin_agent.agents.investment.forecasting.create_forecasting_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.forecasting"
                ".ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            result = await forecasting_node(
                {"ticker": "AAPL", "query": "AI compounder"},
                MagicMock(),  # type: ignore[arg-type]
            )

        assert "forecast" in result
        assert "error" in result["forecast"]

    @pytest.mark.asyncio
    async def test_error_fallback_preserves_raw_output(self):
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "structured_response": None,
            "output": "Partial forecast text.",
        }

        with (
            patch(
                "muffin_agent.agents.investment.forecasting.create_forecasting_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.forecasting"
                ".ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            result = await forecasting_node(
                {"ticker": "AAPL"},
                MagicMock(),  # type: ignore[arg-type]
            )

        assert result["forecast"]["raw_output"] == "Partial forecast text."
        assert "error" in result["forecast"]

    @pytest.mark.asyncio
    async def test_runs_regardless_of_company_signal_fail(self):
        """Node runs full analysis even when company_signal='fail'."""
        mock_structured = MagicMock(spec=ForecastOutput)
        mock_structured.model_dump.return_value = _VALID_FORECAST_DICT
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"structured_response": mock_structured}

        with (
            patch(
                "muffin_agent.agents.investment.forecasting.create_forecasting_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.forecasting"
                ".ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            result = await forecasting_node(
                {
                    "ticker": "AAPL",
                    "query": "short thesis",
                    "company_analysis": {"company_signal": "fail"},
                },
                MagicMock(),  # type: ignore[arg-type]
            )

        # Agent was invoked (not short-circuited)
        assert mock_agent.ainvoke.called
        assert "forecast" in result
