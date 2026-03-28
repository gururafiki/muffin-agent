"""Tests for the sector analysis investment agent."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from deepagents import CompiledSubAgent
from pydantic import ValidationError

from muffin_agent.agents.investment.sector_analysis import (
    SectorAnalysisInputState,
    SectorViewOutput,
    create_sector_analysis_agent,
    sector_analysis_node,
)
from muffin_agent.prompts import render_template

# ── Minimal valid output dict ──────────────────────────────────────────────────

_VALID_SECTOR_DICT = {
    "sector": "Information Technology",
    "industry": "Semiconductors",
    "cycle_position": {
        "label": "mid_expansion",
        "direction": "stable",
        "key_indicators": (
            "Sector ETF (SOXX) +8.3% trailing 3M vs S&P 500 +4.1% (etf-index); "
            "ISM Manufacturing 52.1 (economy-macro); "
            "earnings revision ratio +1.4 (consensus)"
        ),
    },
    "competitive_assessment": {
        "rivalry_intensity": "high",
        "barriers_to_entry": "high",
        "supplier_power": "moderate",
        "buyer_power": "moderate",
        "threat_of_substitutes": "low",
        "overall_attractiveness": "attractive",
        "summary": (
            "Semiconductors exhibit high rivalry among 5-6 global leaders but are "
            "protected by enormous capital and IP barriers. Supplier power is moderate "
            "via ASML lithography equipment monopoly; buyer power is contained by "
            "switching costs. Net competitive structure favours incumbents with scale."
        ),
    },
    "thematic_drivers": [
        {
            "theme": "AI infrastructure buildout",
            "direction": "tailwind",
            "time_horizon": "medium_term",
            "rationale": (
                "Hyperscaler capex guidance up 40% YoY (Q4 2025 earnings calls); "
                "GPU and custom ASIC demand outpacing supply."
            ),
        },
        {
            "theme": "Inventory correction cycle",
            "direction": "headwind",
            "time_horizon": "near_term",
            "rationale": (
                "PC and smartphone end-market inventory days above 90 "
                "(industry reports); expected to normalize by mid-2026."
            ),
        },
    ],
    "sector_valuation": {
        "pe_ratio": 28.5,
        "ev_ebitda": 18.2,
        "pe_vs_sp500_pct": 15.7,
        "pe_vs_5y_avg_pct": None,
        "valuation_signal": "fairly_valued",
    },
    "regulatory_backdrop": {
        "risk_level": "moderate",
        "key_items": [
            "CHIPS Act subsidy disbursements: ongoing (2025-2030 horizon)",
            "Export controls on advanced chips to China: tightened Oct 2023, "
            "under review",
        ],
        "summary": (
            "Export controls create headwinds for China-exposed revenue but CHIPS Act "
            "subsidies support domestic fab investment. Net regulatory direction "
            "is mixed."
        ),
    },
    "peer_tickers": ["NVDA", "AMD", "INTC", "AVGO", "QCOM", "MU", "TXN"],
    "alpha_opportunity": "high",
    "alpha_rationale": (
        "Peer YTD return dispersion of 34.2% (std dev across 7 peers); NVDA +180% vs "
        "INTC -38% YTD reflects strong winner/loser dynamics driven by AI exposure."
    ),
    "sector_signal": "favorable",
    "sector_summary": (
        "Semiconductors are in mid-expansion driven by AI infrastructure demand, with "
        "high barriers to entry and incumbent pricing power insulating margins. "
        "Valuation is a slight premium to the S&P 500 but justified by AI-driven "
        "earnings upgrades. Peer dispersion is high, presenting strong stock-picking "
        "opportunity."
    ),
    "data_sources": [
        {
            "subagent": "etf-index",
            "data_retrieved": (
                "SOXX performance, S&P 500 multiples, etf_equity_exposure NVDA"
            ),
            "period": "2024-2026",
        }
    ],
    "confidence": 0.85,
    "limitations": [],
}


# ── Prompt template tests ──────────────────────────────────────────────────────


@pytest.mark.unit
class TestPromptTemplate:
    """Verify prompt renders and contains required structural elements."""

    def test_renders(self):
        result = render_template("investment/sector_analysis.jinja")
        assert len(result) > 200

    def test_contains_subagent_table(self):
        result = render_template("investment/sector_analysis.jinja")
        for name in [
            "etf-index",
            "discovery-screening",
            "news",
            "regulatory-filings",
            "data-validation",
        ]:
            assert name in result

    def test_contains_workflow_steps(self):
        result = render_template("investment/sector_analysis.jinja")
        for step in ["Step 1", "Step 2", "Validate Data", "Step 4", "Step 5"]:
            assert step in result

    def test_contains_step_labels(self):
        result = render_template("investment/sector_analysis.jinja")
        assert "Parse Context" in result
        assert "Collect" in result
        assert "Validate" in result
        assert "Score" in result
        assert "Reflect" in result

    def test_contains_five_forces(self):
        result = render_template("investment/sector_analysis.jinja")
        assert "rivalry" in result.lower()
        assert "barriers" in result.lower()
        assert "supplier" in result.lower()
        assert "buyer" in result.lower()
        assert "substitute" in result.lower()

    def test_contains_cycle_position_labels(self):
        result = render_template("investment/sector_analysis.jinja")
        assert "early_expansion" in result
        assert "mid_expansion" in result
        assert "late_cycle" in result
        assert "contraction" in result
        assert "recovery" in result

    def test_contains_sector_signal_labels(self):
        result = render_template("investment/sector_analysis.jinja")
        assert "favorable" in result
        assert "cautious" in result

    def test_contains_alpha_opportunity_labels(self):
        result = render_template("investment/sector_analysis.jinja")
        assert "alpha_opportunity" in result
        assert "dispersion" in result.lower()

    def test_contains_output_schema_keys(self):
        result = render_template("investment/sector_analysis.jinja")
        for field in [
            "sector_signal",
            "sector_summary",
            "peer_tickers",
            "thematic_drivers",
            "sector_valuation",
            "regulatory_backdrop",
        ]:
            assert field in result

    def test_grounding_constraint_present(self):
        result = render_template("investment/sector_analysis.jinja")
        assert "NEVER" in result

    def test_sandbox_mandatory_marker(self):
        result = render_template("investment/sector_analysis.jinja")
        assert "MANDATORY" in result
        assert "compute_sector_relative_performance" in result

    def test_sandbox_computations_named(self):
        result = render_template("investment/sector_analysis.jinja")
        assert "compute_sector_relative_performance" in result
        assert "pe_vs_sp500_pct" in result
        assert "peer_dispersion_pct" in result

    def test_reflection_step_present(self):
        result = render_template("investment/sector_analysis.jinja")
        assert "consistency" in result.lower() or "reflect" in result.lower()

    def test_validation_loop_present(self):
        result = render_template("investment/sector_analysis.jinja")
        assert "data-validation" in result
        assert "proceed" in result
        assert "collect_more_data" in result
        assert "insufficient_data" in result

    def test_structured_output_tool_instruction(self):
        result = render_template("investment/sector_analysis.jinja")
        assert "structured output tool" in result

    def test_no_fabrication_clause(self):
        result = render_template("investment/sector_analysis.jinja")
        assert "fabricate" in result.lower() or "fabricat" in result.lower()


# ── SectorAnalysisInputState schema tests ─────────────────────────────────────


@pytest.mark.unit
class TestSectorAnalysisInputState:
    """Verify InputState TypedDict structure."""

    def test_all_fields_optional(self):
        # total=False: empty dict is valid
        state: SectorAnalysisInputState = {}
        assert isinstance(state, dict)

    def test_annotations_contain_expected_keys(self):
        keys = set(SectorAnalysisInputState.__annotations__)
        assert keys == {"ticker", "query", "sector", "industry"}

    def test_ticker_only_valid(self):
        state: SectorAnalysisInputState = {"ticker": "AAPL"}
        assert state["ticker"] == "AAPL"

    def test_explicit_sector_industry_valid(self):
        state: SectorAnalysisInputState = {
            "sector": "Information Technology",
            "industry": "Semiconductors",
            "query": "AI chip stocks",
        }
        assert state["sector"] == "Information Technology"


# ── SectorViewOutput Pydantic model tests ─────────────────────────────────────


@pytest.mark.unit
class TestSectorViewOutputModel:
    """Verify Pydantic output schema validation."""

    def test_valid_full_instance(self):
        output = SectorViewOutput(**_VALID_SECTOR_DICT)
        assert output.sector == "Information Technology"
        assert output.industry == "Semiconductors"
        assert output.sector_signal == "favorable"
        assert output.alpha_opportunity == "high"

    def test_cycle_position_fields(self):
        output = SectorViewOutput(**_VALID_SECTOR_DICT)
        assert output.cycle_position.label == "mid_expansion"
        assert output.cycle_position.direction == "stable"

    def test_competitive_assessment_five_forces(self):
        output = SectorViewOutput(**_VALID_SECTOR_DICT)
        assert output.competitive_assessment.rivalry_intensity == "high"
        assert output.competitive_assessment.barriers_to_entry == "high"
        assert output.competitive_assessment.overall_attractiveness == "attractive"

    def test_thematic_drivers_structure(self):
        output = SectorViewOutput(**_VALID_SECTOR_DICT)
        assert len(output.thematic_drivers) == 2
        assert output.thematic_drivers[0].direction == "tailwind"
        assert output.thematic_drivers[1].time_horizon == "near_term"

    def test_sector_valuation_optional_fields_absent_when_none(self):
        output = SectorViewOutput(**_VALID_SECTOR_DICT)
        # pe_vs_5y_avg_pct is None in _VALID_SECTOR_DICT
        assert output.sector_valuation.pe_vs_5y_avg_pct is None
        dumped = output.model_dump(exclude_none=True)
        assert "pe_vs_5y_avg_pct" not in dumped

    def test_invalid_cycle_label_raises(self):
        data = {
            **_VALID_SECTOR_DICT,
            "cycle_position": {
                **_VALID_SECTOR_DICT["cycle_position"],
                "label": "boom",  # invalid
            },
        }
        with pytest.raises(ValidationError):
            SectorViewOutput(**data)

    def test_invalid_rivalry_intensity_raises(self):
        data = {
            **_VALID_SECTOR_DICT,
            "competitive_assessment": {
                **_VALID_SECTOR_DICT["competitive_assessment"],
                "rivalry_intensity": "extreme",  # invalid
            },
        }
        with pytest.raises(ValidationError):
            SectorViewOutput(**data)

    def test_invalid_sector_signal_raises(self):
        data = {**_VALID_SECTOR_DICT, "sector_signal": "bullish"}  # invalid
        with pytest.raises(ValidationError):
            SectorViewOutput(**data)

    def test_invalid_thematic_direction_raises(self):
        data = {
            **_VALID_SECTOR_DICT,
            "thematic_drivers": [
                {
                    **_VALID_SECTOR_DICT["thematic_drivers"][0],
                    "direction": "positive",  # invalid
                }
            ],
        }
        with pytest.raises(ValidationError):
            SectorViewOutput(**data)

    def test_model_dump_serializable(self):
        output = SectorViewOutput(**_VALID_SECTOR_DICT)
        dumped = output.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["sector"] == "Information Technology"
        assert dumped["cycle_position"]["label"] == "mid_expansion"
        assert isinstance(dumped["thematic_drivers"], list)
        assert isinstance(dumped["peer_tickers"], list)
        assert isinstance(dumped["data_sources"], list)
        assert isinstance(dumped["limitations"], list)

    def test_defaults_for_data_sources_and_limitations(self):
        # Omit data_sources and limitations — they should default to []
        data = {
            k: v
            for k, v in _VALID_SECTOR_DICT.items()
            if k not in ("data_sources", "limitations")
        }
        output = SectorViewOutput(**data)
        assert output.data_sources == []
        assert output.limitations == []


# ── Node JSON input tests ──────────────────────────────────────────────────────


@pytest.mark.unit
class TestSectorAnalysisNodeJsonInput:
    """Verify the node serializes state context correctly."""

    @pytest.mark.asyncio
    async def test_passes_ticker_and_query_in_input(self):
        mock_structured = MagicMock(spec=SectorViewOutput)
        mock_structured.model_dump.return_value = _VALID_SECTOR_DICT
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"structured_response": mock_structured}

        with (
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".create_sector_analysis_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            state = {"ticker": "NVDA", "query": "AI infrastructure stocks"}
            await sector_analysis_node(state, MagicMock())  # type: ignore[arg-type]

        raw = mock_agent.ainvoke.call_args[0][0]["input"]
        ctx = json.loads(raw)
        assert ctx["ticker"] == "NVDA"
        assert ctx["query"] == "AI infrastructure stocks"

    @pytest.mark.asyncio
    async def test_omits_missing_state_fields(self):
        mock_structured = MagicMock(spec=SectorViewOutput)
        mock_structured.model_dump.return_value = _VALID_SECTOR_DICT
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"structured_response": mock_structured}

        with (
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".create_sector_analysis_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            state = {"query": "semiconductor stocks"}
            await sector_analysis_node(state, MagicMock())  # type: ignore[arg-type]

        raw = mock_agent.ainvoke.call_args[0][0]["input"]
        ctx = json.loads(raw)
        assert "ticker" not in ctx
        assert ctx["query"] == "semiconductor stocks"

    @pytest.mark.asyncio
    async def test_passes_explicit_sector_industry(self):
        mock_structured = MagicMock(spec=SectorViewOutput)
        mock_structured.model_dump.return_value = _VALID_SECTOR_DICT
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"structured_response": mock_structured}

        with (
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".create_sector_analysis_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            state = {
                "sector": "Information Technology",
                "industry": "Semiconductors",
                "query": "chip stocks",
            }
            await sector_analysis_node(state, MagicMock())  # type: ignore[arg-type]

        raw = mock_agent.ainvoke.call_args[0][0]["input"]
        ctx = json.loads(raw)
        assert ctx["sector"] == "Information Technology"
        assert ctx["industry"] == "Semiconductors"
        assert "ticker" not in ctx

    @pytest.mark.asyncio
    async def test_excludes_non_input_state_fields(self):
        """Fields outside SectorAnalysisInputState are not sent to agent."""
        mock_structured = MagicMock(spec=SectorViewOutput)
        mock_structured.model_dump.return_value = _VALID_SECTOR_DICT
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"structured_response": mock_structured}

        with (
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".create_sector_analysis_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            # Simulate full TickerAnalysisState with extra fields
            state = {
                "ticker": "AAPL",
                "query": "quality tech",
                "market_regime": {"regime_label": "Goldilocks"},
                "company_analysis": None,
                "forecast": None,
            }
            await sector_analysis_node(state, MagicMock())  # type: ignore[arg-type]

        raw = mock_agent.ainvoke.call_args[0][0]["input"]
        ctx = json.loads(raw)
        assert set(ctx.keys()) <= set(SectorAnalysisInputState.__annotations__)
        assert "market_regime" not in ctx
        assert "company_analysis" not in ctx


# ── create_sector_analysis_agent tests ────────────────────────────────────────


@pytest.mark.unit
class TestCreateSectorAnalysisAgent:
    """Verify agent factory wires subagents and response_format correctly."""

    @pytest.mark.asyncio
    async def test_creates_six_subagents(self):
        config = MagicMock()
        config.get_llm.return_value = MagicMock()

        with (
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".ModelConfiguration.from_runnable_config",
                return_value=config,
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".create_etf_index_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".create_discovery_screening_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".create_equity_estimates_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".create_news_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".create_regulatory_filings_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".build_validation_subagent",
                new_callable=AsyncMock,
                return_value=CompiledSubAgent(
                    name="data-validation",
                    description="mock validation",
                    runnable=MagicMock(),
                ),
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis.create_deep_agent"
            ) as mock_create,
        ):
            mock_create.return_value = MagicMock()
            await create_sector_analysis_agent(config)

            call_kwargs = mock_create.call_args.kwargs
            assert len(call_kwargs["subagents"]) == 6  # 5 data + 1 validation

    @pytest.mark.asyncio
    async def test_subagent_names_in_order(self):
        config = MagicMock()
        config.get_llm.return_value = MagicMock()

        with (
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".ModelConfiguration.from_runnable_config",
                return_value=config,
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".create_etf_index_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".create_discovery_screening_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".create_equity_estimates_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".create_news_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".create_regulatory_filings_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".build_validation_subagent",
                new_callable=AsyncMock,
                return_value=CompiledSubAgent(
                    name="data-validation",
                    description="mock validation",
                    runnable=MagicMock(),
                ),
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis.create_deep_agent"
            ) as mock_create,
        ):
            mock_create.return_value = MagicMock()
            await create_sector_analysis_agent(config)

            subagents = mock_create.call_args.kwargs["subagents"]
            names = [s["name"] for s in subagents]
            assert names == [
                "etf-index",
                "discovery-screening",
                "equity-estimates",
                "news",
                "regulatory-filings",
                "data-validation",
            ]

    @pytest.mark.asyncio
    async def test_passes_get_backend(self):
        config = MagicMock()
        config.get_llm.return_value = MagicMock()

        with (
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".ModelConfiguration.from_runnable_config",
                return_value=config,
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".create_etf_index_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".create_discovery_screening_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".create_equity_estimates_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".create_news_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".create_regulatory_filings_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".build_validation_subagent",
                new_callable=AsyncMock,
                return_value=CompiledSubAgent(
                    name="data-validation",
                    description="mock validation",
                    runnable=MagicMock(),
                ),
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis.create_deep_agent"
            ) as mock_create,
            patch(
                "muffin_agent.agents.investment.sector_analysis.get_backend"
            ) as mock_backend,
        ):
            mock_create.return_value = MagicMock()
            await create_sector_analysis_agent(config)

            assert mock_create.call_args.kwargs["backend"] is mock_backend

    @pytest.mark.asyncio
    async def test_uses_auto_strategy_response_format(self):
        config = MagicMock()
        config.get_llm.return_value = MagicMock()

        with (
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".ModelConfiguration.from_runnable_config",
                return_value=config,
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".create_etf_index_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".create_discovery_screening_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".create_equity_estimates_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".create_news_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".create_regulatory_filings_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".build_validation_subagent",
                new_callable=AsyncMock,
                return_value=CompiledSubAgent(
                    name="data-validation",
                    description="mock validation",
                    runnable=MagicMock(),
                ),
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis.create_deep_agent"
            ) as mock_create,
        ):
            mock_create.return_value = MagicMock()
            await create_sector_analysis_agent(config)

            from langchain.agents.structured_output import AutoStrategy

            response_format = mock_create.call_args.kwargs["response_format"]
            assert isinstance(response_format, AutoStrategy)
            assert response_format.schema is SectorViewOutput


# ── sector_analysis_node tests ────────────────────────────────────────────────


@pytest.mark.unit
class TestSectorAnalysisNode:
    """Verify node behavior: output key, structured response, error fallback."""

    @pytest.mark.asyncio
    async def test_returns_sector_view_key(self):
        mock_structured = MagicMock(spec=SectorViewOutput)
        mock_structured.model_dump.return_value = _VALID_SECTOR_DICT
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"structured_response": mock_structured}

        with (
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".create_sector_analysis_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            result = await sector_analysis_node(
                {"ticker": "NVDA", "query": "AI chips"},
                MagicMock(),  # type: ignore[arg-type]
            )

        assert "sector_view" in result
        sv = result["sector_view"]
        assert sv["sector"] == "Information Technology"
        assert sv["sector_signal"] == "favorable"
        assert "cycle_position" in sv
        assert "competitive_assessment" in sv
        assert "thematic_drivers" in sv

    @pytest.mark.asyncio
    async def test_passes_ticker_in_task(self):
        mock_structured = MagicMock(spec=SectorViewOutput)
        mock_structured.model_dump.return_value = _VALID_SECTOR_DICT
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"structured_response": mock_structured}

        with (
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".create_sector_analysis_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            await sector_analysis_node(
                {"ticker": "NVDA", "query": "AI infrastructure"},
                MagicMock(),  # type: ignore[arg-type]
            )

        task_input = mock_agent.ainvoke.call_args[0][0]["input"]
        assert "NVDA" in task_input

    @pytest.mark.asyncio
    async def test_works_without_ticker(self):
        mock_structured = MagicMock(spec=SectorViewOutput)
        mock_structured.model_dump.return_value = _VALID_SECTOR_DICT
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"structured_response": mock_structured}

        with (
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".create_sector_analysis_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            result = await sector_analysis_node(
                {"sector": "Information Technology", "industry": "Semiconductors"},
                MagicMock(),  # type: ignore[arg-type]
            )

        assert "sector_view" in result

    @pytest.mark.asyncio
    async def test_error_fallback_on_missing_structured_response(self):
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "output": "Sorry, I could not complete the analysis."
        }

        with (
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".create_sector_analysis_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            result = await sector_analysis_node(
                {"query": "tech stocks"},
                MagicMock(),  # type: ignore[arg-type]
            )

        assert "sector_view" in result
        assert result["sector_view"]["sector"] == "unknown"
        assert "error" in result["sector_view"]

    @pytest.mark.asyncio
    async def test_error_fallback_preserves_raw_output(self):
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "structured_response": None,
            "output": "Partial analysis text.",
        }

        with (
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".create_sector_analysis_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.sector_analysis"
                ".ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            result = await sector_analysis_node(
                {"query": "test"},
                MagicMock(),  # type: ignore[arg-type]
            )

        assert result["sector_view"]["sector"] == "unknown"
        assert result["sector_view"]["raw_output"] == "Partial analysis text."
