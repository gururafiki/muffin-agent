"""Tests for portfolio-level nodes: position sizing, ticker decision, reconciler."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from muffin_agent.agents.personas_council.portfolio import (
    PortfolioReconciliationOutput,
    TickerDecision,
    compute_correlation_matrix,
    portfolio_reconciler_node,
    risk_management_node,
    score_correlation_multiplier,
    score_volatility_limit,
    ticker_decision_node,
)
from muffin_agent.agents.personas_council.portfolio.executor import PortfolioOrder
from muffin_agent.agents.personas_council.portfolio.state import new_portfolio


def _uptrend_bars(n: int, base: float = 100.0) -> list[dict[str, Any]]:
    return [
        {
            "date": f"2025-{(i // 21) + 1:02d}-{(i % 21) + 1:02d}",
            "open": base * 1.001**i * 0.999,
            "high": base * 1.001**i * 1.005,
            "low": base * 1.001**i * 0.995,
            "close": base * 1.001**i,
            "volume": 1_000_000,
        }
        for i in range(n)
    ]


@pytest.mark.unit
class TestScoreVolatilityLimit:
    @pytest.mark.parametrize(
        ("vol", "expected_pct"),
        [
            (0.10, 0.25),  # <15% → 25%
            (0.20, 0.195),
            (0.40, 0.14),
            (0.60, 0.10),  # ≥50% → 10%
        ],
    )
    def test_buckets(self, vol, expected_pct):
        assert score_volatility_limit(vol) == pytest.approx(expected_pct, abs=1e-2)


@pytest.mark.unit
class TestScoreCorrelationMultiplier:
    @pytest.mark.parametrize(
        ("corr", "expected"),
        [
            (0.85, 0.70),
            (0.65, 0.85),
            (0.50, 1.00),
            (0.30, 1.05),
            (0.10, 1.10),
        ],
    )
    def test_buckets(self, corr, expected):
        assert score_correlation_multiplier(corr) == expected


@pytest.mark.unit
class TestComputeCorrelationMatrix:
    def test_two_aligned_series(self):
        # Perfectly correlated series
        m = compute_correlation_matrix(
            {
                "A": [0.01, -0.02, 0.005, 0.01, -0.005],
                "B": [0.02, -0.04, 0.01, 0.02, -0.01],
            }
        )
        assert m["A"]["B"] == pytest.approx(1.0, abs=1e-6)
        assert m["B"]["A"] == pytest.approx(1.0, abs=1e-6)

    def test_single_ticker_empty(self):
        m = compute_correlation_matrix({"A": [0.01, 0.02, 0.03, 0.04, 0.05]})
        assert m == {}

    def test_too_short_skipped(self):
        m = compute_correlation_matrix({"A": [0.01], "B": [0.02]})
        assert m == {}


@pytest.mark.unit
@pytest.mark.asyncio
class TestPositionSizingNode:
    async def test_empty_tickers_returns_empty(self):
        result = await risk_management_node({}, {})
        assert result == {"position_limits": {}}

    async def test_single_ticker_with_low_vol(self):
        p = new_portfolio(initial_cash=100_000)
        state = {
            "tickers": ["AAPL"],
            "portfolio": p.model_dump(),
            "prices_history": {"AAPL": _uptrend_bars(80, base=150)},
            "current_prices": {"AAPL": 150},
        }
        result = await risk_management_node(state, {})
        limit = result["position_limits"]["AAPL"]
        # Low vol uptrend should give roughly 25% of NAV
        assert limit["limit_pct"] > 0.10
        assert limit["limit_dollars"] > 0
        assert limit["annualized_volatility"] is not None

    async def test_missing_price_fallback(self):
        p = new_portfolio(initial_cash=100_000)
        state = {
            "tickers": ["AAPL"],
            "portfolio": p.model_dump(),
            "prices_history": {},
            "current_prices": {},
        }
        result = await risk_management_node(state, {})
        limit = result["position_limits"]["AAPL"]
        assert limit["limit_dollars"] == 0
        assert "No current price" in limit["reasoning"]


@pytest.mark.unit
@pytest.mark.asyncio
class TestTickerDecisionNode:
    async def test_no_signals_returns_hold_without_llm(self):
        result = await ticker_decision_node({"ticker": "AAPL"}, {})
        d = result["ticker_decision"]
        assert d["recommended_action"] == "hold"
        assert d["target_pct_of_nav"] == 0
        assert d["confidence"] == 0.0

    async def test_llm_called_when_signals_present(self):
        fake = TickerDecision(
            ticker="AAPL",
            recommended_action="buy",
            target_pct_of_nav=0.04,
            rating="buy",
            confidence=0.75,
            reasoning="Council favours buy.",
            signals_summary={"buy": ["warren_buffett"]},
        )
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=fake)
        with patch(
            "muffin_agent.agents.personas_council.portfolio.ticker_decision.ModelConfiguration.get_chat_model_for_role",
            return_value=mock_llm,
        ):
            result = await ticker_decision_node(
                {
                    "ticker": "AAPL",
                    "persona_signals": [
                        {
                            "agent_id": "warren_buffett",
                            "signal": "buy",
                            "confidence": 0.8,
                            "reasoning": "wide moat",
                            "evidence": {},
                        }
                    ],
                    "council_synthesis": {"consensus_rating": "buy"},
                },
                {},
            )
        assert result["ticker_decision"]["recommended_action"] == "buy"
        assert mock_llm.ainvoke.await_count == 1


@pytest.mark.unit
@pytest.mark.asyncio
class TestPortfolioReconcilerNode:
    async def test_empty_decisions_returns_no_orders(self):
        result = await portfolio_reconciler_node(
            {"portfolio": new_portfolio(initial_cash=100_000).model_dump()},
            {},
        )
        assert result["orders"] == []
        assert "No ticker decisions" in result["portfolio_notes"]

    async def test_pre_fill_hold_when_no_capacity(self):
        # Cash = 0 → no buy capacity; no position → no sell capacity
        p = new_portfolio(initial_cash=0)
        state = {
            "portfolio": p.model_dump(),
            "position_limits": {"AAPL": {"limit_dollars": 0, "remaining_dollars": 0}},
            "ticker_decisions": {
                "AAPL": {
                    "recommended_action": "buy",
                    "target_pct_of_nav": 0.04,
                    "rating": "buy",
                    "confidence": 0.7,
                    "reasoning": "council bull",
                    "signals_summary": {},
                }
            },
            "current_prices": {"AAPL": 150},
        }
        # Pre-filled → LLM should NOT be called
        result = await portfolio_reconciler_node(state, {})
        assert len(result["orders"]) == 1
        assert result["orders"][0]["action"] == "hold"
        assert result["orders"][0]["ticker"] == "AAPL"
        assert "pre-filled" in result["portfolio_notes"].lower()

    async def test_llm_called_when_capacity_exists(self):
        p = new_portfolio(initial_cash=100_000)
        fake = PortfolioReconciliationOutput(
            orders=[
                PortfolioOrder(
                    ticker="AAPL",
                    action="buy",
                    quantity=20,
                    confidence=0.75,
                    reasoning="20 shares = ~3% of NAV",
                )
            ],
            portfolio_notes="Single-position concentration; within limits.",
        )
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=fake)
        with patch(
            "muffin_agent.agents.personas_council.portfolio.portfolio_reconciler.ModelConfiguration.get_chat_model_for_role",
            return_value=mock_llm,
        ):
            result = await portfolio_reconciler_node(
                {
                    "portfolio": p.model_dump(),
                    "position_limits": {
                        "AAPL": {
                            "limit_dollars": 25_000,
                            "remaining_dollars": 25_000,
                        }
                    },
                    "ticker_decisions": {
                        "AAPL": {
                            "recommended_action": "buy",
                            "target_pct_of_nav": 0.03,
                            "rating": "buy",
                            "confidence": 0.75,
                            "reasoning": "council bull",
                            "signals_summary": {},
                        }
                    },
                    "current_prices": {"AAPL": 150},
                },
                {},
            )
        assert len(result["orders"]) == 1
        assert result["orders"][0]["action"] == "buy"
        assert result["orders"][0]["quantity"] == 20
        assert mock_llm.ainvoke.await_count == 1
