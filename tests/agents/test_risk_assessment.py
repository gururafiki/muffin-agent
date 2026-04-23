"""Tests for the risk assessment investment agent (Step 8)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from muffin_agent.agents.investment.risk_assessment import (
    RiskAssessmentInputState,
    RiskAssessmentOutput,
    StressScenario,
    create_risk_assessment_agent,
    risk_assessment_node,
)
from muffin_agent.prompts import render_template

# ── Minimal valid nested objects ──────────────────────────────────────────────

_FACTOR_LOADINGS = {
    "beta": 1.25,
    "smb": 0.18,
    "hml": -0.30,
    "rmw": 0.45,
    "cma": -0.12,
    "umd": 0.22,
    "alpha_annualized": 0.032,
    "r_squared": 0.71,
    "regression_period": "2023-01 to 2025-01 (monthly)",
}

_IV_TERM_STRUCTURE = {
    "iv_30d_pct": 28.5,
    "iv_60d_pct": 26.0,
    "iv_90d_pct": 24.5,
    "put_call_skew_25d": 3.2,
    "term_slope": "normal",
}

_SHORT_INTEREST = {
    "short_interest_pct": 4.5,
    "days_to_cover": 2.1,
    "short_volume_ratio": 0.38,
    "crowding_signal": "low",
}

_STRESS_SCENARIOS = [
    {
        "name": "GFC 2008 analog",
        "scenario_type": "historical",
        "description": "S&P 500 -56% peak-to-trough (Oct 2007 – Mar 2009).",
        "market_return_assumed_pct": -56.0,
        "estimated_stock_return_pct": -65.8,
        "estimated_dollar_impact_per_share": -123.5,
        "methodology": "S&P 500 -56% peak-to-trough, beta-scaled (beta=1.175)",
    },
    {
        "name": "COVID-19 crash Feb–Mar 2020",
        "scenario_type": "historical",
        "description": "S&P 500 -34% in 23 trading days.",
        "market_return_assumed_pct": -34.0,
        "estimated_stock_return_pct": -40.0,
        "estimated_dollar_impact_per_share": -75.0,
        "methodology": "S&P 500 -34%, beta-scaled (beta=1.175)",
    },
    {
        "name": "Rate shock +200bps",
        "scenario_type": "macro",
        "description": "Sudden Fed tightening drives market correction.",
        "market_return_assumed_pct": -15.0,
        "estimated_stock_return_pct": -17.6,
        "estimated_dollar_impact_per_share": -33.0,
        "methodology": "beta-scaled from assumed -15% market drawdown",
    },
    {
        "name": "Earnings recession",
        "scenario_type": "macro",
        "description": "Broad earnings contraction; S&P 500 -25%.",
        "market_return_assumed_pct": -25.0,
        "estimated_stock_return_pct": -29.4,
        "estimated_dollar_impact_per_share": -55.1,
        "methodology": "beta-scaled from assumed -25% market drawdown",
    },
    {
        "name": "Liquidity crunch",
        "scenario_type": "macro",
        "description": "Credit spreads widen; market sells off -20%.",
        "market_return_assumed_pct": -20.0,
        "estimated_stock_return_pct": -23.5,
        "estimated_dollar_impact_per_share": -44.1,
        "methodology": "beta-scaled from assumed -20% market drawdown",
    },
    {
        "name": "Earnings miss / guidance cut",
        "scenario_type": "idiosyncratic",
        "description": "Company misses consensus EPS by >10% and cuts guidance.",
        "market_return_assumed_pct": None,
        "estimated_stock_return_pct": -18.0,
        "estimated_dollar_impact_per_share": -33.8,
        "methodology": (
            "Company-specific; no market move assumed; based on"
            " historical earnings-miss reactions"
        ),
    },
]

_VALID_OUTPUT = {
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "as_of_date": "2026-03-27",
    "beta": 1.175,
    "annualized_vol_pct": 22.4,
    "max_drawdown_1y_pct": -28.3,
    "var_95_1m_pct": 10.8,
    "cvar_95_1m_pct": 13.6,
    "sharpe_ratio": 1.42,
    "sortino_ratio": 2.05,
    "factor_loadings": _FACTOR_LOADINGS,
    "implied_volatility": _IV_TERM_STRUCTURE,
    "short_interest": _SHORT_INTEREST,
    "stress_scenarios": _STRESS_SCENARIOS,
    "ex_ante_stop_level": 165.80,
    "stop_methodology": (
        "current_price × (1 − 2 × var_95_1m_pct / 100);"
        " cross-checked against GFC bear scenario implied price"
    ),
    "risk_signal": "acceptable",
    "risk_flags": [],
    "confidence": 0.92,
    "data_sources": [
        {
            "subagent": "equity-price",
            "data_retrieved": "2-year weekly OHLCV + SPY benchmark",
            "period": "2024-03 to 2026-03",
        }
    ],
    "limitations": [],
}


# ── TestPromptTemplate ────────────────────────────────────────────────────────


@pytest.mark.unit
class TestPromptTemplate:
    def test_renders_without_error(self):
        prompt = render_template("investment/risk_assessment.jinja")
        assert len(prompt) > 100

    def test_contains_all_subagent_names(self):
        prompt = render_template("investment/risk_assessment.jinja")
        for name in (
            "equity-price",
            "options",
            "fama-french",
            "equity-ownership",
            "fixed-income",
            "economy-macro",
            "data-validation",
        ):
            assert name in prompt, f"Subagent '{name}' missing from prompt"

    def test_contains_workflow_steps(self):
        prompt = render_template("investment/risk_assessment.jinja")
        # Validation (Step 3) is injected via _validation_step.jinja partial without
        # an explicit "Step 3" header — same convention as market_regime.jinja.
        for step in ("Step 1", "Step 2", "Step 4", "Step 5"):
            assert step in prompt

    def test_contains_key_output_fields(self):
        prompt = render_template("investment/risk_assessment.jinja")
        for field in (
            "beta",
            "annualized_vol_pct",
            "var_95_1m_pct",
            "cvar_95_1m_pct",
            "sharpe_ratio",
            "sortino_ratio",
            "factor_loadings",
            "implied_volatility",
            "stress_scenarios",
            "ex_ante_stop_level",
            "risk_signal",
            "risk_flags",
        ):
            assert field in prompt, f"Output field '{field}' missing from prompt"

    def test_contains_mandatory_compute_calls(self):
        prompt = render_template("investment/risk_assessment.jinja")
        for tool_call in (
            "compute_beta",
            "compute_sharpe_sortino",
            "compute_max_drawdown",
            "compute_var_cvar",
        ):
            assert tool_call in prompt, f"Mandatory tool call '{tool_call}' missing"

    def test_no_fabrication_clause(self):
        prompt = render_template("investment/risk_assessment.jinja")
        assert "NEVER" in prompt or "never" in prompt

    def test_contains_reflection_step(self):
        prompt = render_template("investment/risk_assessment.jinja")
        assert "Reflect" in prompt or "reflect" in prompt

    def test_contains_historical_scenario_names(self):
        prompt = render_template("investment/risk_assessment.jinja")
        assert "GFC 2008" in prompt
        assert "COVID" in prompt


# ── TestInputState ────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestInputState:
    def test_total_false_allows_empty_dict(self):
        state: RiskAssessmentInputState = {}
        assert state == {}

    def test_all_fields_optional(self):
        # Should not raise
        state: RiskAssessmentInputState = {"ticker": "MSFT"}
        assert state["ticker"] == "MSFT"

    def test_annotations_contain_expected_keys(self):
        annotations = RiskAssessmentInputState.__annotations__
        for key in ("ticker", "query", "company_analysis", "market_regime"):
            assert key in annotations

    def test_accepts_full_state(self):
        state: RiskAssessmentInputState = {
            "ticker": "NVDA",
            "query": "assess downside risk",
            "company_analysis": {"company_signal": "pass", "key_risks": []},
            "market_regime": {"key_risks": ["rate shock"], "factor_assessment": {}},
        }
        assert state["ticker"] == "NVDA"


# ── TestOutputModel ───────────────────────────────────────────────────────────


@pytest.mark.unit
class TestOutputModel:
    def test_valid_full_instance(self):
        output = RiskAssessmentOutput(**_VALID_OUTPUT)
        assert output.ticker == "AAPL"
        assert output.risk_signal == "acceptable"
        assert output.beta == pytest.approx(1.175)
        assert len(output.stress_scenarios) == 6

    def test_invalid_risk_signal_raises(self):
        bad = {**_VALID_OUTPUT, "risk_signal": "unknown_value"}
        with pytest.raises(ValidationError):
            RiskAssessmentOutput(**bad)

    def test_invalid_crowding_signal_raises(self):
        bad_si = {**_SHORT_INTEREST, "crowding_signal": "very_high"}
        bad = {**_VALID_OUTPUT, "short_interest": bad_si}
        with pytest.raises(ValidationError):
            RiskAssessmentOutput(**bad)

    def test_invalid_term_slope_raises(self):
        bad_iv = {**_IV_TERM_STRUCTURE, "term_slope": "steep"}
        bad = {**_VALID_OUTPUT, "implied_volatility": bad_iv}
        with pytest.raises(ValidationError):
            RiskAssessmentOutput(**bad)

    def test_optional_fields_can_be_none(self):
        minimal = {
            **_VALID_OUTPUT,
            "beta": None,
            "annualized_vol_pct": None,
            "max_drawdown_1y_pct": None,
            "var_95_1m_pct": None,
            "cvar_95_1m_pct": None,
            "sharpe_ratio": None,
            "sortino_ratio": None,
            "factor_loadings": None,
            "implied_volatility": None,
            "ex_ante_stop_level": None,
            "stop_methodology": None,
        }
        output = RiskAssessmentOutput(**minimal)
        assert output.beta is None
        assert output.factor_loadings is None

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            RiskAssessmentOutput(**{**_VALID_OUTPUT, "confidence": 1.5})

    def test_model_dump_serializable(self):
        import json

        output = RiskAssessmentOutput(**_VALID_OUTPUT)
        dumped = output.model_dump()
        serialized = json.dumps(dumped)  # must not raise
        assert "AAPL" in serialized

    def test_empty_risk_flags_allowed(self):
        output = RiskAssessmentOutput(**{**_VALID_OUTPUT, "risk_flags": []})
        assert output.risk_flags == []

    def test_scenario_type_literals(self):
        for stype in ("macro", "historical", "idiosyncratic"):
            s = StressScenario(
                name="test",
                scenario_type=stype,
                description="test scenario",
                estimated_stock_return_pct=-10.0,
                methodology="beta-scaled",
            )
            assert s.scenario_type == stype

    def test_invalid_scenario_type_raises(self):
        with pytest.raises(ValidationError):
            StressScenario(
                name="test",
                scenario_type="unknown",
                description="test",
                estimated_stock_return_pct=-10.0,
                methodology="x",
            )


# ── TestNodeJsonInput ─────────────────────────────────────────────────────────


@pytest.mark.unit
class TestNodeJsonInput:
    def test_state_keys_passed_as_json_input(self):
        """run_deep_agent_node serialises state keys as JSON for the agent input."""
        import json

        state = {
            "ticker": "TSLA",
            "query": "assess tail risk",
            "company_analysis": {
                "company_signal": "watch",
                "key_risks": ["competition"],
            },
        }
        context = {
            k: state[k]
            for k in RiskAssessmentInputState.__annotations__
            if state.get(k)
        }
        serialized = json.dumps(context)
        assert "TSLA" in serialized
        assert "market_regime" not in serialized  # absent from state → not included

    def test_missing_optional_state_keys_excluded(self):

        state = {"ticker": "META"}
        context = {
            k: state[k]
            for k in RiskAssessmentInputState.__annotations__
            if state.get(k)
        }
        assert "company_analysis" not in context
        assert "market_regime" not in context


# ── TestCreateAgent ───────────────────────────────────────────────────────────


@pytest.mark.unit
class TestCreateAgent:
    @pytest.mark.asyncio
    async def test_correct_subagent_count(self):
        """Agent factory should build 7 subagents (6 data + 1 validation)."""
        mock_agent = MagicMock()
        config = MagicMock()
        config.get_llm.return_value = MagicMock()

        with (
            patch(
                "muffin_agent.agents.investment.risk_assessment"
                ".ModelConfiguration.from_runnable_config",
                return_value=config,
            ),
            patch(
                "muffin_agent.utils.agent_builder.create_deep_agent",
                return_value=mock_agent,
            ) as mock_create,
            patch(
                "muffin_agent.agents.investment.risk_assessment._build_risk_assessment_subagents",
                new_callable=AsyncMock,
                return_value=[MagicMock()] * 7,
            ),
        ):
            await create_risk_assessment_agent(config)

            call_kwargs = mock_create.call_args.kwargs
            assert len(call_kwargs["subagents"]) == 7

    @pytest.mark.asyncio
    async def test_uses_auto_strategy_response_format(self):
        from langchain.agents.structured_output import AutoStrategy

        mock_agent = MagicMock()
        config = MagicMock()
        config.get_llm.return_value = MagicMock()

        with (
            patch(
                "muffin_agent.agents.investment.risk_assessment"
                ".ModelConfiguration.from_runnable_config",
                return_value=config,
            ),
            patch(
                "muffin_agent.utils.agent_builder.create_deep_agent",
                return_value=mock_agent,
            ) as mock_create,
            patch(
                "muffin_agent.agents.investment.risk_assessment._build_risk_assessment_subagents",
                new_callable=AsyncMock,
                return_value=[MagicMock()] * 7,
            ),
        ):
            await create_risk_assessment_agent(config)

            call_kwargs = mock_create.call_args.kwargs
            assert isinstance(call_kwargs["response_format"], AutoStrategy)

    @pytest.mark.asyncio
    async def test_store_forwarded_to_create_deep_agent(self):
        mock_store = MagicMock()
        mock_agent = MagicMock()
        config = MagicMock()
        config.get_llm.return_value = MagicMock()

        with (
            patch(
                "muffin_agent.agents.investment.risk_assessment"
                ".ModelConfiguration.from_runnable_config",
                return_value=config,
            ),
            patch(
                "muffin_agent.utils.agent_builder.create_deep_agent",
                return_value=mock_agent,
            ) as mock_create,
            patch(
                "muffin_agent.agents.investment.risk_assessment._build_risk_assessment_subagents",
                new_callable=AsyncMock,
                return_value=[MagicMock()] * 7,
            ),
        ):
            await create_risk_assessment_agent(config, store=mock_store)

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["store"] is mock_store

    @pytest.mark.asyncio
    async def test_risk_tools_passed(self):

        mock_agent = MagicMock()
        config = MagicMock()
        config.get_llm.return_value = MagicMock()

        with (
            patch(
                "muffin_agent.agents.investment.risk_assessment"
                ".ModelConfiguration.from_runnable_config",
                return_value=config,
            ),
            patch(
                "muffin_agent.utils.agent_builder.create_deep_agent",
                return_value=mock_agent,
            ) as mock_create,
            patch(
                "muffin_agent.agents.investment.risk_assessment._build_risk_assessment_subagents",
                new_callable=AsyncMock,
                return_value=[MagicMock()] * 7,
            ),
        ):
            await create_risk_assessment_agent(config)

            call_kwargs = mock_create.call_args.kwargs
            tool_names = {t.name for t in call_kwargs["tools"]}
            assert "compute_beta" in tool_names
            assert "compute_var_cvar" in tool_names
            assert "compute_sharpe_sortino" in tool_names
            assert "compute_max_drawdown" in tool_names


# ── TestNode ──────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestNode:
    @pytest.mark.asyncio
    async def test_returns_risk_assessment_key(self):
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "structured_response": MagicMock(model_dump=lambda: _VALID_OUTPUT)
        }

        with (
            patch(
                "muffin_agent.agents.investment.risk_assessment.create_risk_assessment_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.utils.ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            result = await risk_assessment_node(
                state={"ticker": "AAPL", "query": "risk check"},
                config=MagicMock(configurable={}),
            )

        assert "risk_assessment" in result

    @pytest.mark.asyncio
    async def test_error_fallback_contains_risk_signal_unacceptable(self):
        """When agent returns no structured response, fallback must include
        risk_signal."""
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "structured_response": None,
            "output": "agent failed",
        }

        with (
            patch(
                "muffin_agent.agents.investment.risk_assessment.create_risk_assessment_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.utils.ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            result = await risk_assessment_node(
                state={"ticker": "AAPL"},
                config=MagicMock(configurable={}),
            )

        ra = result["risk_assessment"]
        assert "error" in ra
        assert ra.get("risk_signal") == "unacceptable"

    @pytest.mark.asyncio
    async def test_exception_fallback(self):
        """Exceptions should be caught and return an error dict."""
        with (
            patch(
                "muffin_agent.agents.investment.risk_assessment.create_risk_assessment_agent",
                side_effect=RuntimeError("agent build failed"),
            ),
            patch(
                "muffin_agent.agents.investment.utils.ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            result = await risk_assessment_node(
                state={"ticker": "AAPL"},
                config=MagicMock(configurable={}),
            )

        ra = result["risk_assessment"]
        assert "error" in ra
        assert ra.get("risk_signal") == "unacceptable"

    @pytest.mark.asyncio
    async def test_store_kwarg_propagated(self):
        """Store argument must be forwarded through the node."""
        mock_store = MagicMock()
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "structured_response": MagicMock(model_dump=lambda: _VALID_OUTPUT)
        }
        captured = {}

        async def fake_factory(cfg, store=None):
            captured["store"] = store
            return mock_agent

        with (
            patch(
                "muffin_agent.agents.investment.risk_assessment.create_risk_assessment_agent",
                side_effect=fake_factory,
            ),
            patch(
                "muffin_agent.agents.investment.utils.ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            await risk_assessment_node(
                state={"ticker": "AAPL"},
                config=MagicMock(configurable={}),
                store=mock_store,
            )

        assert captured.get("store") is mock_store
