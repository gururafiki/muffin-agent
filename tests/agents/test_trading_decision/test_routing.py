"""Tests for the graph-level routing functions (``graph.py``).

These functions are called by LangGraph's conditional edges. They read
state directly (no Command, no node coupling) and return the literal
name of the next node.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from muffin_agent.agents.trading_decision.graph import (
    _route_investment_debate,
    _route_risk_debate,
)


def _config(**configurable) -> dict:
    return {"configurable": configurable}


@pytest.mark.unit
class TestRouteInvestmentDebate:
    def test_opening_routes_to_bull(self):
        # Empty state — bull opens.
        with patch(
            "muffin_agent.agents.trading_decision.graph._active_config",
            return_value=_config(),
        ):
            assert _route_investment_debate({}) == "bull_researcher"

    def test_after_bull_routes_to_bear(self):
        state = {"investment_bull_responses": ["a"]}
        with patch(
            "muffin_agent.agents.trading_decision.graph._active_config",
            return_value=_config(),
        ):
            assert _route_investment_debate(state) == "bear_researcher"

    def test_after_bear_routes_to_bull(self):
        state = {
            "investment_bull_responses": ["a"],
            "investment_bear_responses": ["b"],
        }
        with patch(
            "muffin_agent.agents.trading_decision.graph._active_config",
            return_value=_config(),
        ):
            assert _route_investment_debate(state) == "bull_researcher"

    def test_exit_to_judge_at_default_budget(self):
        # Default = 2 rounds = 4 turns.
        state = {
            "investment_bull_responses": ["a", "c"],
            "investment_bear_responses": ["b", "d"],
        }
        with patch(
            "muffin_agent.agents.trading_decision.graph._active_config",
            return_value=_config(),
        ):
            assert _route_investment_debate(state) == "investment_judge"

    def test_one_round_override(self):
        state = {
            "investment_bull_responses": ["a"],
            "investment_bear_responses": ["b"],
        }
        with patch(
            "muffin_agent.agents.trading_decision.graph._active_config",
            return_value=_config(max_investment_debate_rounds=1),
        ):
            assert _route_investment_debate(state) == "investment_judge"

    @pytest.mark.parametrize(
        "rounds,bulls,bears,expected",
        [
            (3, 1, 1, "bull_researcher"),
            (3, 2, 2, "bull_researcher"),
            (3, 3, 2, "bear_researcher"),
            (3, 3, 3, "investment_judge"),
        ],
    )
    def test_3_round_progression(self, rounds, bulls, bears, expected):
        state = {
            "investment_bull_responses": ["x"] * bulls,
            "investment_bear_responses": ["x"] * bears,
        }
        with patch(
            "muffin_agent.agents.trading_decision.graph._active_config",
            return_value=_config(max_investment_debate_rounds=rounds),
        ):
            assert _route_investment_debate(state) == expected


@pytest.mark.unit
class TestRouteRiskDebate:
    def test_opening_routes_to_aggressive(self):
        with patch(
            "muffin_agent.agents.trading_decision.graph._active_config",
            return_value=_config(),
        ):
            assert _route_risk_debate({}) == "aggressive_debator"

    def test_round_robin_progression_default_1_round(self):
        # Default = 1 round = 3 turns (one per persona).
        with patch(
            "muffin_agent.agents.trading_decision.graph._active_config",
            return_value=_config(),
        ):
            # After Aggressive
            state = {"risk_aggressive_responses": ["a"]}
            assert _route_risk_debate(state) == "conservative_debator"

            # After Conservative
            state["risk_conservative_responses"] = ["c"]
            assert _route_risk_debate(state) == "neutral_debator"

            # After Neutral
            state["risk_neutral_responses"] = ["n"]
            assert _route_risk_debate(state) == "portfolio_manager"

    def test_two_round_progression(self):
        # 2 rounds = 6 turns.
        with patch(
            "muffin_agent.agents.trading_decision.graph._active_config",
            return_value=_config(max_risk_debate_rounds=2),
        ):
            # After 1 round, next is Aggressive (round 2).
            state = {
                "risk_aggressive_responses": ["a1"],
                "risk_conservative_responses": ["c1"],
                "risk_neutral_responses": ["n1"],
            }
            assert _route_risk_debate(state) == "aggressive_debator"

            # Halfway through round 2.
            state["risk_aggressive_responses"].append("a2")
            assert _route_risk_debate(state) == "conservative_debator"

            state["risk_conservative_responses"].append("c2")
            assert _route_risk_debate(state) == "neutral_debator"

            state["risk_neutral_responses"].append("n2")
            assert _route_risk_debate(state) == "portfolio_manager"
