"""Unit tests for the Warren Buffett persona — v4 subgraph architecture.

Covers four layers:

1. **Composite scorers** — pure-Python functions returning typed sub-evidence.
2. **`compute_evidence_node`** — deterministic Python node composing all
   scorers into a full ``WarrenBuffettEvidence``.
3. **`render_verdict_node`** — single LLM call (mocked) with structured
   output validation.
4. **`build_warren_buffett_agent`** — compiled subgraph e2e (mocked LLMs
   + MCP tools) with ``input_schema`` / ``output_schema`` assertions.

Plus legacy-bridge tests (``warren_buffett_node`` + ``_compute_buffett_facts``)
to ensure the council & CLI keep working through phases 2-4 of the refactor.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from muffin_agent.agents.personas import PERSONA_REGISTRY
from muffin_agent.agents.personas.warren_buffett import (
    BuffettMetricsRow,
    WarrenBuffettBookValueGrowth,
    WarrenBuffettEvidence,
    WarrenBuffettFundamentals,
    WarrenBuffettManagement,
    WarrenBuffettMoat,
    WarrenBuffettPricingPower,
    WarrenBuffettRawData,
    WarrenBuffettSignal,
    WarrenBuffettState,
    _bundle_to_raw_data,
    _compute_buffett_facts,
    _score_buffett_book_value_growth,
    _score_buffett_consistency,
    _score_buffett_fundamentals,
    _score_buffett_management,
    _score_buffett_moat,
    _score_buffett_pricing_power,
    compute_evidence_node,
    render_verdict_node,
    warren_buffett_node,
)

# ── Test fixtures ─────────────────────────────────────────────────────────────


def _strong_metrics_row(**override: Any) -> BuffettMetricsRow:
    base = dict(
        return_on_equity=0.25,
        debt_to_equity=0.20,
        operating_margin=0.25,
        current_ratio=2.5,
        asset_turnover=1.1,
    )
    base.update(override)
    return BuffettMetricsRow(**base)


def _strong_state() -> dict[str, Any]:
    """A state dict representing a high-quality Buffett-style business."""
    return {
        "ticker": "ZAB",
        "as_of_date": "2025-01-31",
        "metrics_history": [_strong_metrics_row() for _ in range(8)],
        "net_income_series": [1_000, 1_200, 1_500, 1_800, 2_100, 2_400, 2_700, 3_000],
        "revenue_series": [
            10_000,
            11_000,
            12_000,
            13_000,
            14_000,
            15_000,
            16_000,
            17_000,
        ],
        "gross_margin_series": [0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.48],
        "shareholders_equity_series": [
            10_000,
            11_000,
            12_500,
            14_300,
            16_300,
            18_500,
            21_000,
            24_000,
        ],
        "outstanding_shares_series": [1_000] * 8,
        "depreciation_amortization_series": [200, 220, 240, 250, 260, 270, 280, 300],
        "capital_expenditure_series": [350, 360, 370, 380, 390, 400, 410, 420],
        "latest_issuance_or_purchase_of_equity_shares": -120,
        "latest_dividends_and_other_cash_distributions": -85,
        "market_cap": 100_000_000_000.0,
    }


def _weak_state() -> dict[str, Any]:
    """A poor-quality, financially-strained business."""
    return {
        "ticker": "ZUG",
        "as_of_date": "2025-01-31",
        "metrics_history": [
            BuffettMetricsRow(
                return_on_equity=0.03,
                debt_to_equity=2.5,
                operating_margin=0.05,
                current_ratio=0.9,
                asset_turnover=0.4,
            )
            for _ in range(8)
        ],
        "net_income_series": [500, 200, -100, 0, -200, -500, -800, -1000],
        "revenue_series": [10_000] * 8,
        "gross_margin_series": [0.20, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.12],
        "shareholders_equity_series": [
            10_000,
            9_500,
            9_000,
            8_400,
            7_800,
            7_000,
            6_000,
            5_000,
        ],
        "outstanding_shares_series": [
            1_000,
            1_050,
            1_100,
            1_200,
            1_300,
            1_400,
            1_500,
            1_700,
        ],
        "depreciation_amortization_series": [200] * 8,
        "capital_expenditure_series": [800] * 8,
        "latest_issuance_or_purchase_of_equity_shares": 800,
        "latest_dividends_and_other_cash_distributions": 0,
        "market_cap": 5_000_000_000.0,
    }


def _strong_bundle() -> dict[str, Any]:
    """A legacy PersonaDataBundle for the bridge tests."""
    return {
        "ticker": "ZAB",
        "as_of_date": "2025-01-31",
        "market_cap": 100_000_000_000.0,
        "financial_metrics": [
            {
                "return_on_equity": 0.25,
                "debt_to_equity": 0.20,
                "operating_margin": 0.25,
                "current_ratio": 2.5,
                "asset_turnover": 1.1,
            }
            for _ in range(8)
        ],
        "line_items": {
            "net_income": [1_000, 1_200, 1_500, 1_800, 2_100, 2_400, 2_700, 3_000],
            "revenue": [10_000, 11_000, 12_000, 13_000, 14_000, 15_000, 16_000, 17_000],
            "gross_margin": [0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.48],
            "shareholders_equity": [
                10_000,
                11_000,
                12_500,
                14_300,
                16_300,
                18_500,
                21_000,
                24_000,
            ],
            "outstanding_shares": [1_000] * 8,
            "depreciation_and_amortization": [200, 220, 240, 250, 260, 270, 280, 300],
            "capital_expenditure": [350, 360, 370, 380, 390, 400, 410, 420],
            "issuance_or_purchase_of_equity_shares": [
                -50,
                -60,
                -70,
                -80,
                -90,
                -100,
                -110,
                -120,
            ],
            "dividends_and_other_cash_distributions": [
                -50,
                -55,
                -60,
                -65,
                -70,
                -75,
                -80,
                -85,
            ],
        },
    }


# ── Layer 1: Composite scorers ────────────────────────────────────────────────


@pytest.mark.unit
class TestScoreBuffettFundamentals:
    def test_strong_metrics(self):
        result = _score_buffett_fundamentals(_strong_metrics_row())
        assert isinstance(result, WarrenBuffettFundamentals)
        # ROE > 20% = 3; D/E < 0.3 = 3; op_margin > 20% = 2; current > 2 = 2 → 10/10
        assert result.total_score == 10
        assert result.max_score == 10
        assert result.roe_value == pytest.approx(0.25)

    def test_weak_metrics(self):
        weak = BuffettMetricsRow(
            return_on_equity=0.05,
            debt_to_equity=2.0,
            operating_margin=0.05,
            current_ratio=0.9,
        )
        result = _score_buffett_fundamentals(weak)
        assert result.total_score == 0

    def test_none_input(self):
        result = _score_buffett_fundamentals(None)
        assert result.total_score == 0
        assert "No metrics" in result.reasoning


@pytest.mark.unit
class TestScoreBuffettConsistency:
    def test_strictly_growing(self):
        result = _score_buffett_consistency([100, 110, 120, 130, 140])
        assert result.score == 3
        assert result.strictly_growing is True
        assert result.earnings_growth_pct == pytest.approx(40.0)

    def test_zigzag(self):
        result = _score_buffett_consistency([100, 90, 110, 95, 120])
        assert result.score == 0
        assert result.strictly_growing is False

    def test_too_short(self):
        result = _score_buffett_consistency([100, 110])
        assert result.score == 0
        assert "Insufficient" in result.reasoning


@pytest.mark.unit
class TestScoreBuffettMoat:
    def test_strong_moat(self):
        metrics = [_strong_metrics_row() for _ in range(8)]
        result = _score_buffett_moat(metrics)
        assert isinstance(result, WarrenBuffettMoat)
        # ROE > 15% in 100% of periods → +2 ; turnover > 1.0 → +1 ; stability → +1
        assert result.score >= 3
        assert result.roe_consistency_pct == pytest.approx(100.0)

    def test_insufficient_history(self):
        result = _score_buffett_moat([_strong_metrics_row(), _strong_metrics_row()])
        assert result.score == 0
        assert "Insufficient" in result.reasoning

    def test_none_input(self):
        result = _score_buffett_moat(None)
        assert result.score == 0


@pytest.mark.unit
class TestScoreBuffettPricingPower:
    def test_expanding_margins(self):
        # oldest → newest, expanding (0.30 → 0.48)
        margins = [0.30, 0.32, 0.35, 0.38, 0.42, 0.45, 0.48]
        result = _score_buffett_pricing_power(margins)
        assert isinstance(result, WarrenBuffettPricingPower)
        assert result.margin_direction == "expanding"
        assert result.score >= 3

    def test_declining_margins(self):
        margins = [0.50, 0.48, 0.45, 0.42, 0.40, 0.38, 0.35]
        result = _score_buffett_pricing_power(margins)
        assert result.margin_direction == "declining"

    def test_too_short(self):
        result = _score_buffett_pricing_power([0.40, 0.42])
        assert result.score == 0


@pytest.mark.unit
class TestScoreBuffettBookValueGrowth:
    def test_growing_bvps(self):
        equity = [10_000, 11_000, 12_500, 14_300, 16_300, 18_500, 21_000, 24_000]
        shares = [1_000] * 8
        result = _score_buffett_book_value_growth(equity, shares)
        assert isinstance(result, WarrenBuffettBookValueGrowth)
        assert result.bvps_oldest == pytest.approx(10.0)
        assert result.bvps_latest == pytest.approx(24.0)
        assert result.bvps_cagr_pct is not None and result.bvps_cagr_pct > 12.0
        assert result.score >= 4

    def test_too_short(self):
        result = _score_buffett_book_value_growth([10_000, 11_000], [1_000, 1_000])
        assert result.score == 0


@pytest.mark.unit
class TestScoreBuffettManagement:
    def test_buybacks_and_dividends(self):
        result = _score_buffett_management(-1_000_000.0, -500_000.0)
        assert isinstance(result, WarrenBuffettManagement)
        assert result.score == 2

    def test_dilution_no_dividends(self):
        result = _score_buffett_management(1_000_000.0, 0.0)
        assert result.score == 0

    def test_partial(self):
        result = _score_buffett_management(-1_000.0, 0.0)
        assert result.score == 1


# ── Layer 2: compute_evidence_node ────────────────────────────────────────────


@pytest.mark.unit
class TestComputeEvidenceNode:
    def test_strong_state_high_score(self):
        update = compute_evidence_node(_strong_state())
        evidence = update["evidence"]
        assert isinstance(evidence, WarrenBuffettEvidence)
        # Quality compounder → should score most dimensions strongly
        assert evidence.total_score >= 0.6 * evidence.max_score
        assert evidence.owner_earnings is not None and evidence.owner_earnings > 0
        assert evidence.intrinsic_value is not None

    def test_weak_state_low_score(self):
        update = compute_evidence_node(_weak_state())
        evidence = update["evidence"]
        assert evidence.total_score <= 0.3 * evidence.max_score

    def test_intrinsic_and_margin_of_safety(self):
        state = _strong_state()
        # Set market_cap small enough that intrinsic > market_cap.
        state["market_cap"] = 10_000.0
        evidence = compute_evidence_node(state)["evidence"]
        assert evidence.intrinsic_value is not None
        assert evidence.intrinsic_value > 10_000
        assert evidence.margin_of_safety_pct is not None
        assert evidence.margin_of_safety_pct > 0

    def test_overvalued_yields_negative_mos(self):
        state = _strong_state()
        state["market_cap"] = 10_000_000.0  # massively above intrinsic
        evidence = compute_evidence_node(state)["evidence"]
        assert evidence.margin_of_safety_pct is not None
        assert evidence.margin_of_safety_pct < 0

    def test_evidence_pydantic_round_trip(self):
        evidence = compute_evidence_node(_strong_state())["evidence"]
        dumped = evidence.model_dump()
        reconstructed = WarrenBuffettEvidence.model_validate(dumped)
        assert reconstructed.total_score == evidence.total_score

    def test_metrics_history_accepts_dicts(self):
        """The compute node coerces dict rows into BuffettMetricsRow on the fly."""
        state = _strong_state()
        # Simulate the legacy bridge passing plain dicts
        state["metrics_history"] = [
            row.model_dump() for row in state["metrics_history"]
        ]
        evidence = compute_evidence_node(state)["evidence"]
        assert evidence.total_score >= 0

    def test_missing_data_does_not_crash(self):
        state = {
            "ticker": "ZX",
            "as_of_date": "2025-01-31",
            "metrics_history": [],
            "net_income_series": [],
            "revenue_series": [],
            "gross_margin_series": [],
            "shareholders_equity_series": [],
            "outstanding_shares_series": [],
            "depreciation_amortization_series": [],
            "capital_expenditure_series": [],
            "latest_issuance_or_purchase_of_equity_shares": None,
            "latest_dividends_and_other_cash_distributions": None,
            "market_cap": None,
        }
        evidence = compute_evidence_node(state)["evidence"]
        assert evidence.total_score == 0
        assert evidence.owner_earnings is None
        assert evidence.intrinsic_value is None


# ── Layer 3: render_verdict_node ──────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
class TestRenderVerdictNode:
    async def test_happy_path_calls_llm_with_evidence(self):
        state = _strong_state()
        state.update(compute_evidence_node(state))
        fake_signal = WarrenBuffettSignal(
            agent_id="warren_buffett",
            signal="strong_buy",
            confidence=0.92,
            reasoning="Wonderful business, ample MoS.",
            evidence=state["evidence"],
        )
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=fake_signal)
        with patch(
            "muffin_agent.agents.personas.warren_buffett.ModelConfiguration.get_chat_model_for_role",
            return_value=mock_llm,
        ):
            result = await render_verdict_node(state, {})
        sig = result["persona_signals"][0]
        assert sig["signal"] == "strong_buy"
        assert sig["confidence"] == pytest.approx(0.92)
        assert sig["agent_id"] == "warren_buffett"
        assert mock_llm.ainvoke.await_count == 1

    async def test_missing_evidence_returns_hold(self):
        result = await render_verdict_node({"ticker": "ZX", "evidence": None}, {})
        sig = result["persona_signals"][0]
        assert sig["signal"] == "hold"
        assert sig["confidence"] == 0.0

    async def test_prompt_includes_evidence_fields(self):
        state = _strong_state()
        state.update(compute_evidence_node(state))
        fake_signal = WarrenBuffettSignal(
            agent_id="warren_buffett",
            signal="buy",
            confidence=0.7,
            reasoning="Solid business.",
            evidence=state["evidence"],
        )
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=fake_signal)
        with patch(
            "muffin_agent.agents.personas.warren_buffett.ModelConfiguration.get_chat_model_for_role",
            return_value=mock_llm,
        ):
            await render_verdict_node(state, {})
        sent_messages = mock_llm.ainvoke.call_args.args[0]
        system_prompt = sent_messages[0].content
        # Granular evidence references rendered into the prompt
        assert "Warren Buffett" in system_prompt
        assert "ROE" in system_prompt
        assert "Margin of safety" in system_prompt


# ── Layer 4: Subgraph e2e ─────────────────────────────────────────────────────


def _flatten_schema_properties(json_schema: dict[str, Any]) -> set[str]:
    """Extract all property names from a LangGraph compiled-graph JSON schema.

    LangGraph wraps the user state in a ``LangGraphInput`` / ``LangGraphOutput``
    model. Some shapes inline properties at the top level; others reference
    a ``$defs`` schema. This helper unions both so callers don't have to
    care which form the compiled graph uses.
    """
    props: set[str] = set(json_schema.get("properties", {}).keys())
    for defn in json_schema.get("$defs", {}).values():
        if isinstance(defn, dict):
            props.update(defn.get("properties", {}).keys())
    return props


@pytest.mark.unit
@pytest.mark.asyncio
class TestBuildWarrenBuffettAgent:
    async def test_subgraph_exposes_input_contract(self):
        """Compiled graph's input_schema includes the council-provided fields."""
        from muffin_agent.agents.personas.warren_buffett import (
            build_warren_buffett_agent,
        )

        with patch(
            "muffin_agent.agents.personas.warren_buffett.get_tools",
            return_value=[],
        ):
            agent = await build_warren_buffett_agent({})
        input_props = _flatten_schema_properties(agent.input_schema.model_json_schema())
        # Public input contract (filtered via explicit input_schema on StateGraph)
        assert {"ticker", "as_of_date", "query"} <= input_props
        # RawData internal scratch must NOT appear in the public input schema
        assert "metrics_history" not in input_props
        assert "evidence" not in input_props

    async def test_subgraph_output_includes_persona_signals(self):
        from muffin_agent.agents.personas.warren_buffett import (
            build_warren_buffett_agent,
        )

        with patch(
            "muffin_agent.agents.personas.warren_buffett.get_tools",
            return_value=[],
        ):
            agent = await build_warren_buffett_agent({})
        output_props = _flatten_schema_properties(
            agent.output_schema.model_json_schema()
        )
        assert "persona_signals" in output_props
        # Internal scratch must NOT appear in output schema either
        assert "metrics_history" not in output_props
        assert "evidence" not in output_props


# ── Layer 5: Legacy bridge (data_bundle → v4 evidence) ────────────────────────


@pytest.mark.unit
class TestLegacyBridge:
    def test_bundle_to_raw_data(self):
        raw = _bundle_to_raw_data(_strong_bundle())
        assert isinstance(raw, WarrenBuffettRawData)
        assert len(raw.metrics_history) == 8
        assert raw.metrics_history[0].return_on_equity == pytest.approx(0.25)
        assert raw.market_cap == pytest.approx(100_000_000_000.0)
        assert raw.latest_issuance_or_purchase_of_equity_shares == pytest.approx(-120)
        assert raw.latest_dividends_and_other_cash_distributions == pytest.approx(-85)

    def test_compute_buffett_facts_uses_same_path(self):
        """Legacy ``_compute_buffett_facts`` and v4 ``compute_evidence_node`` agree."""
        bundle = _strong_bundle()
        legacy_evidence = _compute_buffett_facts(bundle)

        # Build the equivalent state directly and run the v4 path
        raw = _bundle_to_raw_data(bundle)
        v4_state = {**raw.model_dump(), "metrics_history": raw.metrics_history}
        v4_evidence = compute_evidence_node(v4_state)["evidence"]

        assert legacy_evidence.total_score == v4_evidence.total_score
        assert legacy_evidence.max_score == v4_evidence.max_score
        assert (
            legacy_evidence.fundamentals.total_score
            == v4_evidence.fundamentals.total_score
        )

    def test_compute_facts_with_realistic_bundle(self):
        """Legacy bridge runs cleanly + produces non-zero score on quality bundle."""
        evidence = _compute_buffett_facts(_strong_bundle())
        assert evidence.total_score >= 0.6 * evidence.max_score
        assert evidence.intrinsic_value is not None


# ── Registry ──────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestPersonaRegistration:
    def test_buffett_in_registry(self):
        assert "warren_buffett" in PERSONA_REGISTRY
        spec = PERSONA_REGISTRY["warren_buffett"]
        assert spec.display_name == "Warren Buffett"
        assert spec.signal_schema is WarrenBuffettSignal


@pytest.mark.unit
@pytest.mark.asyncio
class TestWarrenBuffettLegacyNode:
    async def test_data_bundle_missing_returns_hold_fallback(self):
        result = await warren_buffett_node({"ticker": "ZAB"}, {})
        assert "persona_signals" in result
        sig = result["persona_signals"][0]
        assert sig["agent_id"] == "warren_buffett"
        assert sig["signal"] == "hold"
        assert sig["confidence"] == 0.0

    async def test_data_bundle_error_returns_hold_fallback(self):
        result = await warren_buffett_node(
            {"ticker": "ZAB", "data_bundle": {"error": "MCP timeout"}}, {}
        )
        sig = result["persona_signals"][0]
        assert sig["signal"] == "hold"

    async def test_happy_path_calls_llm_via_legacy_bridge(self):
        bundle = _strong_bundle()
        fake_signal = WarrenBuffettSignal(
            agent_id="warren_buffett",
            signal="strong_buy",
            confidence=0.92,
            reasoning="Wonderful business, ample MoS.",
            evidence=_compute_buffett_facts(bundle),
        )
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=fake_signal)
        with patch(
            "muffin_agent.agents.personas.warren_buffett.ModelConfiguration.get_chat_model_for_role",
            return_value=mock_llm,
        ):
            result = await warren_buffett_node(
                {"ticker": "ZAB", "data_bundle": bundle}, {}
            )
        sig = result["persona_signals"][0]
        assert sig["signal"] == "strong_buy"
        assert sig["confidence"] == pytest.approx(0.92)
        assert mock_llm.ainvoke.await_count == 1


# ── State schema sanity ────────────────────────────────────────────────────────


@pytest.mark.unit
class TestWarrenBuffettState:
    def test_state_schema_has_expected_fields(self):
        annotations = WarrenBuffettState.__annotations__
        # Inputs the council provides
        assert "ticker" in annotations
        assert "as_of_date" in annotations
        assert "query" in annotations
        # RawData scratch fields
        assert "metrics_history" in annotations
        assert "net_income_series" in annotations
        assert "market_cap" in annotations
        # Output to the council
        assert "persona_signals" in annotations
