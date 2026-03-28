"""Tests for the company analysis investment agent (Steps 4-5)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from deepagents import CompiledSubAgent
from pydantic import ValidationError

from muffin_agent.agents.investment.company_analysis import (
    CompanyAnalysisInputState,
    CompanyAnalysisOutput,
    company_analysis_node,
    create_company_analysis_agent,
)
from muffin_agent.prompts import render_template

# ── Minimal valid output dict ──────────────────────────────────────────────────

_VALID_COMPANY_DICT = {
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "business_description": (
        "Apple designs, manufactures, and sells consumer electronics, software, "
        "and services globally, with primary revenue from iPhone (52% of revenue), "
        "Services (22%), and Mac/iPad/Wearables (26%).  Operates in 175+ countries."
    ),
    "moat_assessment": {
        "width": "wide",
        "sources": ["switching_costs", "intangible_assets", "network_effects"],
        "trend": "stable",
        "confidence": 0.92,
        "rationale": (
            "ROIC of 34.2% vs peer median 18.7% (peer_roic_premium_pp = +15.5pp) "
            "sustained for 7 consecutive years.  Switching costs anchored by iOS "
            "ecosystem lock-in (98% iPhone retention rate per Q3 2024 transcript); "
            "intangible assets via App Store duopoly and brand equity."
        ),
    },
    "management_quality": {
        "track_record": "strong",
        "capital_allocation_quality": "excellent",
        "insider_alignment": "moderate",
        "key_concerns": [],
        "summary": (
            "CEO Tim Cook (tenure: 13 years) delivered 12% EPS CAGR vs 8% revenue CAGR "
            "2019-2023, indicating margin expansion and buyback accretion.  Capital "
            "allocation: $90B buybacks in FY2023 at an average 23x P/E; Services pivot "
            "generating 73% gross margins vs 37% for Products.  Insider "
            "ownership at 1.2% is low for a public company of this scale."
        ),
    },
    "esg_flags": [],
    "esg_signal": "green",
    "financial_quality": {
        "revenue_cagr_3y_pct": 8.2,
        "gross_margin_pct": 44.1,
        "operating_margin_pct": 29.8,
        "net_margin_pct": 25.3,
        "roic_pct": 34.2,
        "roe_pct": 147.9,
        "fcf_conversion_pct": 107.4,
        "net_debt_to_ebitda": -0.3,
        "interest_coverage": 29.6,
        "quality_signal": "high",
        "trend": "stable",
    },
    "capital_allocation_summary": (
        "Apple returned $90B via buybacks and $15B via dividends in FY2023 "
        "(dividend yield ~0.6%).  No major acquisitions > $1B since Beats in 2014; "
        "tuck-in AI/semiconductor acquisitions ($50-200M range) fund internal R&D. "
        "Net cash position of -$48B (net cash, not debt) provides defensive "
        "optionality."
    ),
    "key_risks": [
        "China revenue concentration (~19% of revenue) exposed to geopolitical "
        "decoupling and local competitor pressure from Huawei.",
        "App Store regulatory risk: EU DMA enforcement may reduce take-rate from 30% "
        "and open iOS to third-party app stores.",
        "Services growth deceleration: advertising and licensing revenue at risk from "
        "DOJ antitrust action on Google Search default agreement (~$18-20B annual).",
        "AI product cycle uncertainty: late mover in on-device LLM vs. Samsung/Google; "
        "iPhone 16 AI features adoption slower than expected.",
    ],
    "financial_history": {
        "years": [2019, 2020, 2021, 2022, 2023],
        "revenue": [260.2e9, 274.5e9, 365.8e9, 394.3e9, 383.3e9],
        "gross_profit": [98.4e9, 104.9e9, 152.8e9, 170.8e9, 169.1e9],
        "ebit": [63.9e9, 66.3e9, 108.9e9, 119.4e9, 114.3e9],
        "ebitda": [76.5e9, 77.3e9, 120.6e9, 130.5e9, 125.8e9],
        "net_income": [55.3e9, 57.4e9, 94.7e9, 99.8e9, 97.0e9],
        "fcf": [58.9e9, 73.4e9, 93.0e9, 111.4e9, 104.3e9],
        "capex": [10.5e9, 7.3e9, 11.1e9, 10.7e9, 10.9e9],
        "total_debt": [108.0e9, 112.4e9, 124.7e9, 120.1e9, 111.1e9],
        "cash_and_equivalents": [100.6e9, 90.9e9, 63.9e9, 48.3e9, 61.6e9],
        "working_capital": [57.1e9, 38.3e9, 9.4e9, -18.5e9, -2.0e9],
        "total_assets": [338.5e9, 323.9e9, 351.0e9, 352.8e9, 352.6e9],
        "shareholders_equity": [90.5e9, 65.3e9, 63.1e9, 50.7e9, 62.1e9],
        "currency": "USD",
        "quality_narrative": (
            "Revenue grew at a 10.1% CAGR 2019-2022, decelerating to -2.8% in FY2023 "
            "on iPhone unit softness and a strong USD.  Gross margins expanded "
            "steadily from 37.8% to 44.1% driven by the Services mix shift.  "
            "FCF conversion averaged 107% over 5 years, indicating minimal "
            "accruals and high earnings quality.  Leverage is structurally "
            "negative (net cash) despite $90B+ annual "
            "buybacks; balance sheet provides significant defensive capacity."
        ),
    },
    "company_signal": "pass",
    "quality_summary": (
        "Apple is a wide-moat, high-quality compounder with a durable iOS ecosystem "
        "lock-in and Services mix shift driving margin expansion.  ROIC of 34.2% "
        "represents a 15.5pp premium over the peer median, sustained for 7+ years. "
        "Financial quality is 'high': FCF conversion >100%, net cash balance sheet, "
        "29.6x interest coverage.  Triage gate: PASS — no must-have failures "
        "identified; key watch items are China concentration and App Store "
        "regulatory exposure."
    ),
    "confidence": 0.85,
    "data_sources": [
        {
            "subagent": "equity-fundamentals",
            "data_retrieved": (
                "5-year income/balance/cash-flow, ratios, ESG score, management "
                "roster, Q3+Q4 2024 transcripts, revenue by segment"
            ),
            "period": "FY2019-FY2023, Q3-Q4 2024",
        },
        {
            "subagent": "equity-ownership",
            "data_retrieved": "Major holders, insider activity (12M), share statistics",
            "period": "2024",
        },
    ],
    "limitations": [],
}


# ── Prompt template tests ──────────────────────────────────────────────────────


@pytest.mark.unit
class TestPromptTemplate:
    """Verify prompt renders and contains required structural elements."""

    def test_renders(self):
        result = render_template("investment/company_analysis.jinja")
        assert len(result) > 200

    def test_contains_all_subagent_names(self):
        result = render_template("investment/company_analysis.jinja")
        for name in [
            "equity-fundamentals",
            "equity-ownership",
            "regulatory-filings",
            "news",
            "discovery-screening",
            "data-validation",
        ]:
            assert name in result

    def test_contains_workflow_steps(self):
        result = render_template("investment/company_analysis.jinja")
        for step in ["Step 1", "Step 2", "Validate Data", "Step 4", "Step 5"]:
            assert step in result

    def test_contains_step_labels(self):
        result = render_template("investment/company_analysis.jinja")
        assert "Parse Context" in result
        assert "Collect" in result
        assert "Validate" in result
        assert "Assess" in result
        assert "Reflect" in result

    def test_contains_moat_width_labels(self):
        result = render_template("investment/company_analysis.jinja")
        for label in ["wide", "narrow", "none", "negative"]:
            assert label in result

    def test_contains_moat_sources(self):
        result = render_template("investment/company_analysis.jinja")
        for source in [
            "network_effects",
            "switching_costs",
            "intangible_assets",
            "cost_advantage",
            "efficient_scale",
        ]:
            assert source in result

    def test_contains_company_signal_labels(self):
        result = render_template("investment/company_analysis.jinja")
        assert "pass" in result
        assert "watch" in result
        assert "fail" in result

    def test_contains_esg_signal_labels(self):
        result = render_template("investment/company_analysis.jinja")
        assert "green" in result
        assert "amber" in result
        assert "red" in result

    def test_contains_financial_quality_signal_labels(self):
        result = render_template("investment/company_analysis.jinja")
        assert "distressed" in result
        assert "adequate" in result

    def test_contains_output_schema_keys(self):
        result = render_template("investment/company_analysis.jinja")
        for field in [
            "company_signal",
            "quality_summary",
            "moat_assessment",
            "financial_quality",
            "financial_history",
            "esg_signal",
            "management_quality",
            "capital_allocation_summary",
        ]:
            assert field in result

    def test_grounding_constraint_present(self):
        result = render_template("investment/company_analysis.jinja")
        assert "NEVER" in result

    def test_sandbox_mandatory_marker(self):
        result = render_template("investment/company_analysis.jinja")
        assert "MANDATORY" in result
        assert "compute_roic" in result

    def test_sandbox_computation_variables_named(self):
        result = render_template("investment/company_analysis.jinja")
        assert "compute_altman_z_score" in result
        assert "fcf_conversion_pct" in result
        assert "net_debt_to_ebitda" in result
        assert "compute_revenue_cagr" in result
        assert "interest_coverage" in result
        assert "peer_roic_premium_pp" in result

    def test_reflection_step_present(self):
        result = render_template("investment/company_analysis.jinja")
        assert "consistency" in result.lower() or "reflect" in result.lower()

    def test_validation_loop_present(self):
        result = render_template("investment/company_analysis.jinja")
        assert "data-validation" in result
        assert "proceed" in result
        assert "collect_more_data" in result
        assert "insufficient_data" in result

    def test_structured_output_tool_instruction(self):
        result = render_template("investment/company_analysis.jinja")
        assert "structured output tool" in result

    def test_no_fabrication_clause(self):
        result = render_template("investment/company_analysis.jinja")
        assert "fabricate" in result.lower() or "NEVER estimate" in result

    def test_triage_gate_logic_present(self):
        result = render_template("investment/company_analysis.jinja")
        assert (
            "must-have" in result.lower()
            or "must_have" in result.lower()
            or "gate" in result.lower()
        )

    def test_financial_history_arrays_mentioned(self):
        result = render_template("investment/company_analysis.jinja")
        assert "financial_history" in result
        assert "forecasting_node" in result


# ── CompanyAnalysisInputState schema tests ─────────────────────────────────────


@pytest.mark.unit
class TestCompanyAnalysisInputState:
    """Verify InputState TypedDict structure."""

    def test_all_fields_optional(self):
        # total=False: empty dict is valid
        state: CompanyAnalysisInputState = {}
        assert isinstance(state, dict)

    def test_annotations_contain_expected_keys(self):
        keys = set(CompanyAnalysisInputState.__annotations__)
        assert keys == {"ticker", "query"}

    def test_ticker_only_valid(self):
        state: CompanyAnalysisInputState = {"ticker": "AAPL"}
        assert state["ticker"] == "AAPL"

    def test_ticker_and_query_valid(self):
        state: CompanyAnalysisInputState = {
            "ticker": "MSFT",
            "query": "quality compounder in cloud software",
        }
        assert state["ticker"] == "MSFT"
        assert state["query"] == "quality compounder in cloud software"


# ── CompanyAnalysisOutput Pydantic model tests ─────────────────────────────────


@pytest.mark.unit
class TestCompanyAnalysisOutputModel:
    """Verify Pydantic output schema validation."""

    def test_valid_full_instance(self):
        output = CompanyAnalysisOutput(**_VALID_COMPANY_DICT)
        assert output.ticker == "AAPL"
        assert output.company_signal == "pass"
        assert output.esg_signal == "green"

    def test_moat_assessment_fields(self):
        output = CompanyAnalysisOutput(**_VALID_COMPANY_DICT)
        assert output.moat_assessment.width == "wide"
        assert output.moat_assessment.trend == "stable"
        assert output.moat_assessment.confidence == pytest.approx(0.92)
        assert "switching_costs" in output.moat_assessment.sources

    def test_management_quality_fields(self):
        output = CompanyAnalysisOutput(**_VALID_COMPANY_DICT)
        assert output.management_quality.track_record == "strong"
        assert output.management_quality.capital_allocation_quality == "excellent"
        assert output.management_quality.insider_alignment == "moderate"
        assert output.management_quality.key_concerns == []

    def test_financial_quality_fields(self):
        output = CompanyAnalysisOutput(**_VALID_COMPANY_DICT)
        fq = output.financial_quality
        assert fq.quality_signal == "high"
        assert fq.trend == "stable"
        assert fq.roic_pct == pytest.approx(34.2)
        assert fq.fcf_conversion_pct == pytest.approx(107.4)
        assert fq.net_debt_to_ebitda == pytest.approx(-0.3)

    def test_financial_quality_optional_fields_none(self):
        data = {**_VALID_COMPANY_DICT}
        data["financial_quality"] = {
            **_VALID_COMPANY_DICT["financial_quality"],
            "roic_pct": None,
            "net_debt_to_ebitda": None,
        }
        output = CompanyAnalysisOutput(**data)
        assert output.financial_quality.roic_pct is None
        assert output.financial_quality.net_debt_to_ebitda is None
        dumped = output.model_dump(exclude_none=True)
        assert "roic_pct" not in dumped["financial_quality"]

    def test_financial_history_arrays(self):
        output = CompanyAnalysisOutput(**_VALID_COMPANY_DICT)
        fh = output.financial_history
        assert len(fh.years) == 5
        assert len(fh.revenue) == 5
        assert fh.years[0] == 2019
        assert fh.currency == "USD"
        assert isinstance(fh.quality_narrative, str)
        assert len(fh.quality_narrative) > 20

    def test_defaults_for_data_sources_and_limitations(self):
        data = {
            k: v
            for k, v in _VALID_COMPANY_DICT.items()
            if k not in ("data_sources", "limitations")
        }
        output = CompanyAnalysisOutput(**data)
        assert output.data_sources == []
        assert output.limitations == []

    def test_invalid_moat_width_raises(self):
        data = {
            **_VALID_COMPANY_DICT,
            "moat_assessment": {
                **_VALID_COMPANY_DICT["moat_assessment"],
                "width": "moderate",  # invalid
            },
        }
        with pytest.raises(ValidationError):
            CompanyAnalysisOutput(**data)

    def test_invalid_moat_source_raises(self):
        data = {
            **_VALID_COMPANY_DICT,
            "moat_assessment": {
                **_VALID_COMPANY_DICT["moat_assessment"],
                "sources": ["brand_recognition"],  # invalid
            },
        }
        with pytest.raises(ValidationError):
            CompanyAnalysisOutput(**data)

    def test_invalid_company_signal_raises(self):
        data = {**_VALID_COMPANY_DICT, "company_signal": "hold"}  # invalid
        with pytest.raises(ValidationError):
            CompanyAnalysisOutput(**data)

    def test_invalid_esg_signal_raises(self):
        data = {**_VALID_COMPANY_DICT, "esg_signal": "yellow"}  # invalid
        with pytest.raises(ValidationError):
            CompanyAnalysisOutput(**data)

    def test_invalid_quality_signal_raises(self):
        data = {
            **_VALID_COMPANY_DICT,
            "financial_quality": {
                **_VALID_COMPANY_DICT["financial_quality"],
                "quality_signal": "excellent",  # invalid
            },
        }
        with pytest.raises(ValidationError):
            CompanyAnalysisOutput(**data)

    def test_invalid_management_track_record_raises(self):
        data = {
            **_VALID_COMPANY_DICT,
            "management_quality": {
                **_VALID_COMPANY_DICT["management_quality"],
                "track_record": "excellent",  # invalid
            },
        }
        with pytest.raises(ValidationError):
            CompanyAnalysisOutput(**data)

    def test_model_dump_serializable(self):
        output = CompanyAnalysisOutput(**_VALID_COMPANY_DICT)
        dumped = output.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["ticker"] == "AAPL"
        assert dumped["moat_assessment"]["width"] == "wide"
        assert isinstance(dumped["financial_history"]["years"], list)
        assert isinstance(dumped["esg_flags"], list)
        assert isinstance(dumped["key_risks"], list)
        assert isinstance(dumped["data_sources"], list)
        assert isinstance(dumped["limitations"], list)


# ── Node JSON input tests ──────────────────────────────────────────────────────


@pytest.mark.unit
class TestCompanyAnalysisNodeJsonInput:
    """Verify the node serializes state context correctly."""

    @pytest.mark.asyncio
    async def test_passes_ticker_and_query_in_input(self):
        mock_structured = MagicMock(spec=CompanyAnalysisOutput)
        mock_structured.model_dump.return_value = _VALID_COMPANY_DICT
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"structured_response": mock_structured}

        with (
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".create_company_analysis_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            state = {"ticker": "AAPL", "query": "quality compounder"}
            await company_analysis_node(state, MagicMock())  # type: ignore[arg-type]

        raw = mock_agent.ainvoke.call_args[0][0]["input"]
        ctx = json.loads(raw)
        assert ctx["ticker"] == "AAPL"
        assert ctx["query"] == "quality compounder"

    @pytest.mark.asyncio
    async def test_omits_missing_state_fields(self):
        mock_structured = MagicMock(spec=CompanyAnalysisOutput)
        mock_structured.model_dump.return_value = _VALID_COMPANY_DICT
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"structured_response": mock_structured}

        with (
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".create_company_analysis_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            state = {"query": "quality software companies"}
            await company_analysis_node(state, MagicMock())  # type: ignore[arg-type]

        raw = mock_agent.ainvoke.call_args[0][0]["input"]
        ctx = json.loads(raw)
        assert "ticker" not in ctx
        assert ctx["query"] == "quality software companies"

    @pytest.mark.asyncio
    async def test_excludes_non_input_state_fields(self):
        """Fields outside CompanyAnalysisInputState are not sent to the agent."""
        mock_structured = MagicMock(spec=CompanyAnalysisOutput)
        mock_structured.model_dump.return_value = _VALID_COMPANY_DICT
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"structured_response": mock_structured}

        with (
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".create_company_analysis_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            # Simulate full TickerAnalysisState with extra fields
            state = {
                "ticker": "AAPL",
                "query": "quality tech",
                "market_regime": {"regime_label": "Goldilocks"},
                "sector_view": {"sector": "Information Technology"},
                "forecast": None,
            }
            await company_analysis_node(state, MagicMock())  # type: ignore[arg-type]

        raw = mock_agent.ainvoke.call_args[0][0]["input"]
        ctx = json.loads(raw)
        assert set(ctx.keys()) <= set(CompanyAnalysisInputState.__annotations__)
        assert "market_regime" not in ctx
        assert "sector_view" not in ctx
        assert "forecast" not in ctx


# ── create_company_analysis_agent tests ───────────────────────────────────────


@pytest.mark.unit
class TestCreateCompanyAnalysisAgent:
    """Verify agent factory wires subagents and response_format correctly."""

    @pytest.mark.asyncio
    async def test_creates_six_subagents(self):
        config = MagicMock()
        config.get_llm.return_value = MagicMock()

        with (
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".ModelConfiguration.from_runnable_config",
                return_value=config,
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".create_equity_fundamentals_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".create_equity_ownership_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".create_regulatory_filings_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".create_news_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".create_discovery_screening_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".build_validation_subagent",
                new_callable=AsyncMock,
                return_value=CompiledSubAgent(
                    name="data-validation",
                    description="mock validation",
                    runnable=MagicMock(),
                ),
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis.create_deep_agent"
            ) as mock_create,
        ):
            mock_create.return_value = MagicMock()
            await create_company_analysis_agent(config)

            call_kwargs = mock_create.call_args.kwargs
            assert len(call_kwargs["subagents"]) == 6  # 5 data + 1 validation

    @pytest.mark.asyncio
    async def test_subagent_names_in_order(self):
        config = MagicMock()
        config.get_llm.return_value = MagicMock()

        with (
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".ModelConfiguration.from_runnable_config",
                return_value=config,
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".create_equity_fundamentals_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".create_equity_ownership_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".create_regulatory_filings_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".create_news_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".create_discovery_screening_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".build_validation_subagent",
                new_callable=AsyncMock,
                return_value=CompiledSubAgent(
                    name="data-validation",
                    description="mock validation",
                    runnable=MagicMock(),
                ),
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis.create_deep_agent"
            ) as mock_create,
        ):
            mock_create.return_value = MagicMock()
            await create_company_analysis_agent(config)

            subagents = mock_create.call_args.kwargs["subagents"]
            names = [s["name"] for s in subagents]
            assert names == [
                "equity-fundamentals",
                "equity-ownership",
                "regulatory-filings",
                "news",
                "discovery-screening",
                "data-validation",
            ]

    @pytest.mark.asyncio
    async def test_passes_get_backend(self):
        config = MagicMock()
        config.get_llm.return_value = MagicMock()

        with (
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".ModelConfiguration.from_runnable_config",
                return_value=config,
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".create_equity_fundamentals_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".create_equity_ownership_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".create_regulatory_filings_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".create_news_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".create_discovery_screening_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".build_validation_subagent",
                new_callable=AsyncMock,
                return_value=CompiledSubAgent(
                    name="data-validation",
                    description="mock validation",
                    runnable=MagicMock(),
                ),
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis.create_deep_agent"
            ) as mock_create,
            patch(
                "muffin_agent.agents.investment.company_analysis.get_backend"
            ) as mock_backend,
        ):
            mock_create.return_value = MagicMock()
            await create_company_analysis_agent(config)

            assert mock_create.call_args.kwargs["backend"] is mock_backend

    @pytest.mark.asyncio
    async def test_uses_auto_strategy_response_format(self):
        config = MagicMock()
        config.get_llm.return_value = MagicMock()

        with (
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".ModelConfiguration.from_runnable_config",
                return_value=config,
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".create_equity_fundamentals_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".create_equity_ownership_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".create_regulatory_filings_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".create_news_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".create_discovery_screening_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".build_validation_subagent",
                new_callable=AsyncMock,
                return_value=CompiledSubAgent(
                    name="data-validation",
                    description="mock validation",
                    runnable=MagicMock(),
                ),
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis.create_deep_agent"
            ) as mock_create,
        ):
            mock_create.return_value = MagicMock()
            await create_company_analysis_agent(config)

            from langchain.agents.structured_output import AutoStrategy

            response_format = mock_create.call_args.kwargs["response_format"]
            assert isinstance(response_format, AutoStrategy)
            assert response_format.schema is CompanyAnalysisOutput


# ── company_analysis_node tests ───────────────────────────────────────────────


@pytest.mark.unit
class TestCompanyAnalysisNode:
    """Verify node behavior: output key, structured response, error fallback."""

    @pytest.mark.asyncio
    async def test_returns_company_analysis_key(self):
        mock_structured = MagicMock(spec=CompanyAnalysisOutput)
        mock_structured.model_dump.return_value = _VALID_COMPANY_DICT
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"structured_response": mock_structured}

        with (
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".create_company_analysis_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            result = await company_analysis_node(
                {"ticker": "AAPL", "query": "quality compounder"},
                MagicMock(),  # type: ignore[arg-type]
            )

        assert "company_analysis" in result
        ca = result["company_analysis"]
        assert ca["ticker"] == "AAPL"
        assert ca["company_signal"] == "pass"
        assert "moat_assessment" in ca
        assert "financial_quality" in ca
        assert "financial_history" in ca

    @pytest.mark.asyncio
    async def test_passes_ticker_in_task(self):
        mock_structured = MagicMock(spec=CompanyAnalysisOutput)
        mock_structured.model_dump.return_value = _VALID_COMPANY_DICT
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"structured_response": mock_structured}

        with (
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".create_company_analysis_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            await company_analysis_node(
                {"ticker": "AAPL", "query": "quality compounder"},
                MagicMock(),  # type: ignore[arg-type]
            )

        task_input = mock_agent.ainvoke.call_args[0][0]["input"]
        assert "AAPL" in task_input

    @pytest.mark.asyncio
    async def test_works_without_ticker(self):
        mock_structured = MagicMock(spec=CompanyAnalysisOutput)
        mock_structured.model_dump.return_value = _VALID_COMPANY_DICT
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"structured_response": mock_structured}

        with (
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".create_company_analysis_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            result = await company_analysis_node(
                {"query": "quality compounders in software"},
                MagicMock(),  # type: ignore[arg-type]
            )

        assert "company_analysis" in result

    @pytest.mark.asyncio
    async def test_error_fallback_on_missing_structured_response(self):
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "output": "Sorry, I could not complete the analysis."
        }

        with (
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".create_company_analysis_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            result = await company_analysis_node(
                {"ticker": "AAPL", "query": "quality"},
                MagicMock(),  # type: ignore[arg-type]
            )

        assert "company_analysis" in result
        assert "error" in result["company_analysis"]

    @pytest.mark.asyncio
    async def test_error_fallback_preserves_raw_output(self):
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "structured_response": None,
            "output": "Partial analysis text.",
        }

        with (
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".create_company_analysis_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.company_analysis"
                ".ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            result = await company_analysis_node(
                {"ticker": "AAPL"},
                MagicMock(),  # type: ignore[arg-type]
            )

        assert result["company_analysis"]["raw_output"] == "Partial analysis text."
        assert "error" in result["company_analysis"]
