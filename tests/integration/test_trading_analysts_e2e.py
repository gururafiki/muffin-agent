"""E2E integration test — the trading-decision analyst layer composition.

The four analyst ReAct agents (Market / Fundamentals / News / Social) are added
directly to the parent ``TradingDecisionState`` graph via the real
``_add_analyst_nodes`` and run **in parallel**. This test proves the
compiled-subagent composition fix for the analyst pattern (added via
``input_schema=AnalystInput``):

* each analyst maps ``ticker`` + ``decision_date`` in and writes its ``<role>_report``
  back to the parent (the auto-unpacked ``response_format`` field, ``output=False``),
* the four run isolated — no shared ``messages`` channel / cross-talk.

Driven by the schema-routed model (parallel-safe). Before the fix this raised
``ValidationError`` at the first analyst node.
"""

from __future__ import annotations

import pytest
from langgraph.graph import END, START, StateGraph

from muffin_agent.agents.trading_decision.graph import _add_analyst_nodes
from muffin_agent.agents.trading_decision.state import TradingDecisionState

from ._harness import patch_llm_by_schema, patch_mcp, patch_sandbox

pytestmark = pytest.mark.asyncio

# Each analyst's response_format schema → the report it should emit.
_ANALYST_RESPONSES = {
    "MarketAnalystOutput": {"market_report": "Market: uptrend, RSI 58."},
    "FundamentalsAnalystOutput": {"fundamentals_report": "Fundamentals: ROIC 28%."},
    "NewsAnalystOutput": {"news_report": "News: net positive."},
    "SocialAnalystOutput": {"sentiment_report": "Social: bullish chatter."},
}


async def test_four_analysts_compose_and_report(config):
    """All 4 analysts map ticker/decision_date in and propagate their reports."""
    with patch_mcp(scenario="aapl"), patch_sandbox(), patch_llm_by_schema(
        _ANALYST_RESPONSES
    ):
        graph = StateGraph(TradingDecisionState)
        await _add_analyst_nodes(graph, config)
        # Fan out to all 4 from START; barrier into END.
        for name in (
            "market_analyst",
            "fundamentals_analyst",
            "news_analyst",
            "social_analyst",
        ):
            graph.add_edge(START, name)
            graph.add_edge(name, END)
        app = graph.compile()
        result = await app.ainvoke(
            {"ticker": "AAPL", "decision_date": "2026-06-09"}, config=config
        )

    # Every analyst's report propagated to the parent state (output=False fields).
    assert result["market_report"] == "Market: uptrend, RSI 58."
    assert result["fundamentals_report"] == "Fundamentals: ROIC 28%."
    assert result["news_report"] == "News: net positive."
    assert result["sentiment_report"] == "Social: bullish chatter."
    # Inputs the analysts read are unchanged; no internal `messages` leaked into
    # the parent TypedDict (which declares none).
    assert result["ticker"] == "AAPL"
    assert "messages" not in result
