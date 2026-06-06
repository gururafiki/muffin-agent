"""E2E coverage for the v4 portfolio_decision graph.

Catches the class of bug where the council no longer produces a shared
``PersonaDataBundle`` (v3) and the per-ticker subgraph must fetch its own
OHLCV via ``cached_invoke`` to feed downstream position sizing. Before
this test landed, ``prices_history`` silently dropped to ``[]`` and
``position_sizing_node`` fell back to its hardcoded 25% volatility
default for every ticker.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from muffin_agent.agents.portfolio_decision import (
    _fetch_prices_history,
    _parse_ohlcv_response,
)

# ── Synthetic OHLCV fixture ───────────────────────────────────────────────────


def _ohlcv_bars(
    n: int = 252, base: float = 100.0, drift: float = 0.002
) -> list[dict[str, Any]]:
    """Generate a synthetic uptrend OHLCV series matching OpenBB shape."""
    return [
        {
            "date": f"2024-{(i // 21) + 1:02d}-{(i % 21) + 1:02d}",
            "open": base * (1 + drift) ** i * 0.999,
            "high": base * (1 + drift) ** i * 1.005,
            "low": base * (1 + drift) ** i * 0.995,
            "close": base * (1 + drift) ** i,
            "volume": 1_000_000 + i * 1_000,
        }
        for i in range(n)
    ]


# ── _parse_ohlcv_response — shape compatibility with OpenBB MCP outputs ──────


@pytest.mark.unit
class TestParseOhlcvResponse:
    def test_results_wrapped_dict(self):
        bars = _ohlcv_bars(5)
        raw = {"results": bars, "warnings": []}
        assert _parse_ohlcv_response(raw) == bars

    def test_json_string_payload(self):
        bars = _ohlcv_bars(5)
        raw = json.dumps({"results": bars})
        assert _parse_ohlcv_response(raw) == bars

    def test_bare_list(self):
        bars = _ohlcv_bars(5)
        assert _parse_ohlcv_response(bars) == bars

    def test_malformed_json_returns_empty(self):
        assert _parse_ohlcv_response("not-json") == []

    def test_unknown_shape_returns_empty(self):
        assert _parse_ohlcv_response(42) == []


# ── _fetch_prices_history — the path that fixes the data_bundle bug ──────────


@pytest.mark.unit
@pytest.mark.asyncio
class TestFetchPricesHistory:
    async def _patched_fetch(
        self,
        ticker: str,
        *,
        mcp_response: Any,
    ) -> list[dict[str, Any]]:
        """Run _fetch_prices_history with a mocked MCP client + store."""
        # Mock the MCP client so get_tools(config, [...]) returns one tool.
        fake_tool = MagicMock()
        fake_tool.name = "equity_price_historical"
        fake_tool.ainvoke = AsyncMock(return_value=mcp_response)

        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=[fake_tool])

        # Mock the store so cached_invoke can put + get.
        store = AsyncMock()
        store.aget = AsyncMock(return_value=None)
        store.aput = AsyncMock()

        with (
            patch(
                "muffin_agent.agents.data_collection.utils.MultiServerMCPClient",
                return_value=mock_client,
            ),
            patch(
                "muffin_agent.agents.portfolio_decision.get_store",
                return_value=store,
            ),
        ):
            return await _fetch_prices_history(ticker, {"configurable": {}})

    async def test_returns_parsed_bars_on_happy_path(self):
        bars = _ohlcv_bars(252)
        result = await self._patched_fetch("AAPL", mcp_response={"results": bars})
        assert len(result) == 252
        assert result[0]["close"] > 0
        assert result[-1]["close"] > result[0]["close"]

    async def test_empty_ticker_short_circuits(self):
        result = await self._patched_fetch("", mcp_response={"results": []})
        assert result == []

    async def test_mcp_failure_returns_empty(self):
        # Patch get_tools to raise — _fetch_prices_history should swallow.
        with patch(
            "muffin_agent.agents.portfolio_decision.get_tools",
            new=AsyncMock(side_effect=RuntimeError("MCP unreachable")),
        ):
            result = await _fetch_prices_history("AAPL", {"configurable": {}})
        assert result == []


# ── Per-ticker subgraph — verifies prices_history flows into state ───────────


@pytest.mark.unit
@pytest.mark.asyncio
class TestPerTickerSubgraphPriceFlow:
    """Regression: in v4, prices_history MUST come from cached_invoke,
    not from the deleted PersonaDataBundle. Asserts that the per-ticker
    subgraph emits a non-empty ``prices_history_for_ticker``.
    """

    async def test_run_council_populates_prices_history(self):
        from muffin_agent.agents.portfolio_decision import _build_per_ticker_subgraph

        bars = _ohlcv_bars(252)

        # Stub the council subgraph entirely — we only care about the
        # _run_council wrapper's behaviour, not the full 13-persona fan-out.
        fake_council = MagicMock()
        fake_council.ainvoke = AsyncMock(
            return_value={
                "persona_signals": [
                    {
                        "agent_id": "warren_buffett",
                        "signal": "buy",
                        "confidence": 0.75,
                        "reasoning": "stub",
                        "evidence": {},
                    }
                ],
                "council_synthesis": {
                    "ticker": "AAPL",
                    "consensus_rating": "buy",
                    "weighted_confidence": 0.75,
                    "vote_breakdown": {},
                    "bull_case_synthesis": "",
                    "bear_case_synthesis": "",
                    "dissent_summary": "",
                    "key_uncertainties": [],
                    "reasoning": "stub",
                },
            }
        )

        fake_tool = MagicMock()
        fake_tool.name = "equity_price_historical"
        fake_tool.ainvoke = AsyncMock(return_value={"results": bars})
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=[fake_tool])

        store = AsyncMock()
        store.aget = AsyncMock(return_value=None)
        store.aput = AsyncMock()

        # Mock the LLM that ticker_decision_node calls inside the subgraph
        # (otherwise the subgraph invocation tries to reach OpenAI/etc.).
        ticker_decision = AsyncMock()
        ticker_decision.ainvoke = AsyncMock(
            return_value=MagicMock(
                model_dump=lambda: {
                    "ticker": "AAPL",
                    "recommended_action": "buy",
                    "target_pct_of_nav": 0.05,
                    "rating": "buy",
                    "confidence": 0.7,
                    "reasoning": "stub",
                    "signals_summary": {},
                }
            )
        )

        with (
            patch(
                "muffin_agent.agents.portfolio_decision.build_council_graph",
                AsyncMock(return_value=fake_council),
            ),
            patch(
                "muffin_agent.agents.data_collection.utils.MultiServerMCPClient",
                return_value=mock_client,
            ),
            patch(
                "muffin_agent.agents.portfolio_decision.get_store",
                return_value=store,
            ),
            patch(
                "muffin_agent.agents.portfolio.ticker_decision.ModelConfiguration."
                "get_chat_model_for_role",
                return_value=ticker_decision,
            ),
        ):
            per_ticker = await _build_per_ticker_subgraph()
            result = await per_ticker.ainvoke({"ticker": "AAPL", "query": None})

        # The bug: this was [] in v4 before the fix. After the fix it must
        # carry the full bar series so position_sizing sees real volatility.
        assert "prices_history_for_ticker" in result
        assert len(result["prices_history_for_ticker"]) == 252
        assert result["prices_history_for_ticker"][0]["close"] > 0


# ── position_sizing sanity check: real bars ≠ fallback ────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
class TestPositionSizingWithRealBars:
    """If prices_history is populated, position_sizing must NOT fall back
    to the hardcoded 25% volatility default. The fallback exists for
    missing data — when bars are present, real vol must be computed.
    """

    async def test_real_bars_yield_non_default_vol(self):
        from muffin_agent.agents.portfolio import position_sizing_node
        from muffin_agent.portfolio.state import new_portfolio

        # A low-vol synthetic series (drift only, near-zero noise).
        bars = _ohlcv_bars(252, base=100.0, drift=0.0005)
        portfolio = new_portfolio(initial_cash=100_000)
        state = {
            "tickers": ["AAPL"],
            "portfolio": portfolio.model_dump(),
            "current_prices": {"AAPL": bars[-1]["close"]},
            "prices_history": {"AAPL": bars},
        }
        result = await position_sizing_node(state, {})
        limit = result["position_limits"]["AAPL"]
        # 0.25 is the missing-data fallback; we must see a different value.
        assert limit["annualized_volatility"] != pytest.approx(0.25, abs=1e-6)
        assert limit["annualized_volatility"] is not None
        assert limit["limit_dollars"] > 0
