"""Tests for the Trader agent (PR 2).

Covers:

* ``TraderOutput`` schema validation (action enum, list defaults, optional
  price fields, round-trip JSON).
* Prompt template rendering (with and without analysis context, judge
  output always present, action mapping table present).
* ``trader_node`` state translation:
  - Happy path with structured response.
  - Skips LLM when judge output is missing or an error dict.
  - Falls back gracefully on raised exceptions and on missing structured
    output.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from muffin_agent.agents.trading_decision.nodes import trader_node
from muffin_agent.agents.trading_decision.schemas import (
    AnalysisContext,
    TraderOutput,
)
from muffin_agent.prompts import render_template

TRADER_TEMPLATE = "trading_decision/trader.jinja"


def _judge_payload(**overrides) -> dict:
    base = {
        "signal": "buy",
        "conviction": 0.72,
        "summary": "Services growth durable; iPhone refresh imminent.",
        "bull_case": "Services revenue accelerating + AI-driven device cycle.",
        "bear_case": "China demand wobble + ad-tech regulatory drag.",
        "key_catalysts": ["Q1 earnings", "WWDC announcements"],
        "key_risks": ["China demand"],
        "monitoring_checklist": ["Services growth", "iPhone units", "FX"],
        "winning_side": "bull",
        "reasoning": "Bull addressed every credible bear objection.",
    }
    base.update(overrides)
    return base


def _trader_payload(**overrides) -> dict:
    base = {
        "action": "buy",
        "reasoning": (
            "Judge signal=buy, conviction=0.72. Catalysts (Q1, WWDC) anchor a "
            "3–6 month horizon; risk_assessment stop level supports 178.50."
        ),
        "entry_price": 192.50,
        "stop_loss": 178.50,
        "take_profit": 220.00,
        "position_sizing": "2–3% of NAV starter, scale to 5% on Q1 beat",
        "time_horizon": "3–6 months",
    }
    base.update(overrides)
    return base


def _trader_result(payload: dict) -> dict:
    return {
        "messages": [AIMessage(content="(structured trader response)")],
        "structured_response": TraderOutput.model_validate(payload),
    }


def _ctx(**overrides) -> AnalysisContext:
    overrides.setdefault("ticker", "AAPL")
    overrides.setdefault("narrative", "AAPL trades at 22x P/E.")
    return AnalysisContext(**overrides)


# ── Schema tests ──────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestTraderOutput:
    def test_minimal_valid(self):
        out = TraderOutput(**_trader_payload())
        assert out.action == "buy"
        assert out.entry_price == 192.50
        assert out.stop_loss == 178.50
        assert out.take_profit == 220.00

    def test_optional_price_fields_default_to_none(self):
        out = TraderOutput(
            action="hold",
            reasoning="No actionable change.",
            position_sizing="0% of NAV",
            time_horizon="n/a",
        )
        assert out.entry_price is None
        assert out.stop_loss is None
        assert out.take_profit is None

    def test_action_must_be_in_enum(self):
        with pytest.raises(ValidationError):
            TraderOutput(**_trader_payload(action="strong_buy"))  # 5-tier not allowed

    def test_position_sizing_is_required(self):
        payload = _trader_payload()
        del payload["position_sizing"]
        with pytest.raises(ValidationError):
            TraderOutput(**payload)

    def test_time_horizon_is_required(self):
        payload = _trader_payload()
        del payload["time_horizon"]
        with pytest.raises(ValidationError):
            TraderOutput(**payload)

    def test_round_trip_json(self):
        out = TraderOutput(**_trader_payload())
        rehydrated = TraderOutput.model_validate(out.model_dump())
        assert rehydrated == out


# ── Prompt tests ──────────────────────────────────────────────────────────────


def _prompt_vars(**overrides):
    base = {
        "ticker": "AAPL",
        "query": None,
        "investment_judge": _judge_payload(),
        "market_regime": None,
        "sector_view": None,
        "company_analysis": None,
        "forecast": None,
        "risk_assessment": None,
        "valuation": None,
        "narrative": None,
        "additional_context": {},
    }
    base.update(overrides)
    return base


@pytest.mark.unit
class TestTraderPrompt:
    def test_renders_basic(self):
        result = render_template(TRADER_TEMPLATE, **_prompt_vars())
        assert "Trader" in result
        assert "AAPL" in result
        assert "TraderOutput" in result

    def test_judge_output_always_included(self):
        result = render_template(TRADER_TEMPLATE, **_prompt_vars())
        assert "Services growth durable" in result
        assert '"conviction": 0.72' in result

    def test_action_mapping_table_present(self):
        result = render_template(TRADER_TEMPLATE, **_prompt_vars())
        for tier in ("strong_buy", "buy", "hold", "sell", "strong_sell"):
            assert tier in result

    def test_sizing_anchors_to_conviction_buckets(self):
        result = render_template(TRADER_TEMPLATE, **_prompt_vars())
        # Conviction sizing table must list the boundary values.
        for boundary in ("0.8", "0.6", "0.4"):
            assert boundary in result
        assert "% of NAV" in result

    def test_renders_with_risk_assessment_context(self):
        result = render_template(
            TRADER_TEMPLATE,
            **_prompt_vars(risk_assessment={"ex_ante_stop_level": 178.5}),
        )
        assert "ex_ante_stop_level" in result
        assert "178.5" in result

    def test_renders_with_narrative(self):
        result = render_template(
            TRADER_TEMPLATE,
            **_prompt_vars(narrative="AAPL trades at 22x forward P/E."),
        )
        assert "AAPL trades at 22x forward P/E." in result

    def test_no_fabrication_instruction_present(self):
        result = render_template(TRADER_TEMPLATE, **_prompt_vars())
        assert "NEVER fabricate" in result or "never fabricate" in result.lower()


# ── Node tests ────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
class TestTraderNode:
    async def test_happy_path_writes_structured_trader(self):
        agent = AsyncMock()
        agent.ainvoke.return_value = _trader_result(_trader_payload())

        with patch(
            "muffin_agent.agents.trading_decision.nodes.create_trader_agent",
            AsyncMock(return_value=agent),
        ):
            state = {
                "analysis_context": _ctx(),
                "investment_judge": _judge_payload(),
            }
            update = await trader_node(state, {})

        trader = update["trader"]
        assert trader["action"] == "buy"
        assert trader["entry_price"] == 192.50
        assert trader["stop_loss"] == 178.50
        assert trader["position_sizing"].startswith("2–3% of NAV")
        assert "error" not in trader

    async def test_skips_when_judge_missing(self):
        factory_mock = AsyncMock()
        with patch(
            "muffin_agent.agents.trading_decision.nodes.create_trader_agent",
            factory_mock,
        ):
            state = {"analysis_context": _ctx()}  # no investment_judge
            update = await trader_node(state, {})

        assert factory_mock.await_count == 0
        assert update["trader"]["action"] == "hold"
        assert "Missing" in update["trader"]["error"]

    async def test_skips_when_judge_is_error_payload(self):
        factory_mock = AsyncMock()
        with patch(
            "muffin_agent.agents.trading_decision.nodes.create_trader_agent",
            factory_mock,
        ):
            state = {
                "analysis_context": _ctx(),
                "investment_judge": {
                    "signal": "hold",
                    "error": "Judge raised an exception.",
                },
            }
            update = await trader_node(state, {})

        assert factory_mock.await_count == 0
        assert update["trader"]["action"] == "hold"
        assert "error" in update["trader"]

    async def test_agent_exception_yields_fallback(self):
        agent = AsyncMock()
        agent.ainvoke.side_effect = RuntimeError("provider down")

        with patch(
            "muffin_agent.agents.trading_decision.nodes.create_trader_agent",
            AsyncMock(return_value=agent),
        ):
            state = {
                "analysis_context": _ctx(),
                "investment_judge": _judge_payload(),
            }
            update = await trader_node(state, {})

        assert update["trader"]["action"] == "hold"
        assert "raised" in update["trader"]["error"].lower()

    async def test_missing_structured_response_yields_fallback(self):
        agent = AsyncMock()
        agent.ainvoke.return_value = {
            "messages": [AIMessage(content="freeform trader notes, no schema")]
        }

        with patch(
            "muffin_agent.agents.trading_decision.nodes.create_trader_agent",
            AsyncMock(return_value=agent),
        ):
            state = {
                "analysis_context": _ctx(),
                "investment_judge": _judge_payload(),
            }
            update = await trader_node(state, {})

        assert update["trader"]["action"] == "hold"
        assert "raw_output" in update["trader"]
        assert "freeform" in update["trader"]["raw_output"]

    async def test_accepts_dict_structured_response(self):
        """`response_format` may return a dict rather than a Pydantic instance."""
        agent = AsyncMock()
        agent.ainvoke.return_value = {
            "messages": [AIMessage(content="...")],
            "structured_response": _trader_payload(action="sell"),
        }

        with patch(
            "muffin_agent.agents.trading_decision.nodes.create_trader_agent",
            AsyncMock(return_value=agent),
        ):
            state = {
                "analysis_context": _ctx(),
                "investment_judge": _judge_payload(),
            }
            update = await trader_node(state, {})

        assert update["trader"]["action"] == "sell"
