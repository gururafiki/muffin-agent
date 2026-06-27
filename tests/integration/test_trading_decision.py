"""E2E integration test — the full ``trading_decision`` deployable graph.

The ``"trading_decision"`` graph (langgraph.json) is the full pipeline:

    reflector_resolve
      → 4 analyst ReAct agents (Market / Fundamentals / News / Social, parallel)
        → Bull ⇄ Bear debate → Investment Judge → Trader
          → 3-way risk debate (conference subgraph) → Portfolio Manager
            → decision_writeback

This runs the **real** compiled graph through its registered entrypoint
(``graph.make_graph`` — the config-only Platform factory) and mocks only the
external boundaries (LLM / MCP / sandbox).

* The four analysts fan out in parallel, so we drive the graph with the
  **schema-routed** model (:func:`patch_llm_by_schema`) — it answers by the bound
  response schema, not call order, so the parallel interleaving is irrelevant.
* The downstream Judge / Trader / Portfolio Manager use direct
  ``with_structured_output`` calls — answered by class-keyed entries.
* Bull / Bear researchers and the three risk debators are free-form LLM calls —
  the schema-routed model returns placeholder prose, which is enough for an
  end-to-end completion test.
* Reflection bookends run for real but short-circuit to no-ops (no ``store`` /
  ``user_id`` on the deployed config-only entrypoint).
"""

from __future__ import annotations

import pytest

from muffin_agent.agents.trading_decision.graph import make_graph
from muffin_agent.agents.trading_decision.schemas import (
    InvestmentJudgeOutput,
    InvestmentSignal,
    PortfolioDecisionOutput,
    TraderOutput,
)

from ._harness import patch_llm_by_schema, patch_mcp, patch_sandbox

pytestmark = pytest.mark.asyncio


# Analyst response_format schemas (keyed by class name → arg dict) end each
# analyst's ReAct loop; the downstream structured nodes (keyed by class →
# instance) answer the direct with_structured_output calls.
_RESPONSES = {
    "MarketAnalystOutput": {"market_report": "Uptrend; RSI 58, above 200-DMA."},
    "FundamentalsAnalystOutput": {
        "fundamentals_report": "ROIC 28%, low leverage, FCF positive."
    },
    "NewsAnalystOutput": {"news_report": "Net-positive headline flow."},
    "SocialAnalystOutput": {"sentiment_report": "Bullish retail chatter."},
    InvestmentJudgeOutput: InvestmentJudgeOutput(
        signal="buy",
        conviction=0.7,
        summary="Solid quality compounder at a fair price.",
        bull_case="Durable moat, expanding margins.",
        bear_case="Valuation leaves little margin of safety.",
        winning_side="bull",
        reasoning="Bull case on returns outweighs valuation concern.",
    ),
    TraderOutput: TraderOutput(
        action="buy",
        reasoning="Judge signal buy with 0.7 conviction; clean technicals.",
        position_sizing="2% of NAV starter, scale to 4% on Q1 beat.",
        time_horizon="3-6 months",
    ),
    PortfolioDecisionOutput: PortfolioDecisionOutput(
        rating="buy",
        executive_summary="Initiate a 2% starter long; quality at a fair price.",
        investment_thesis="Moat + margin expansion vs. a full multiple; risk debate "
        "surfaced no thesis-breaking objection.",
        time_horizon="3-6 months",
        position_sizing="2% of NAV starter, scale to 4% on Q1 beat.",
        confidence=0.7,
    ),
}


async def test_trading_decision_full_pipeline_to_portfolio_decision(config):
    """make_graph + ainvoke: the full pipeline produces a portfolio decision."""
    with patch_mcp(scenario="aapl"), patch_sandbox(), patch_llm_by_schema(_RESPONSES):
        graph = await make_graph(config)
        result = await graph.ainvoke(
            {"ticker": "AAPL", "decision_date": "2026-06-09"},
            config=config,
        )

    # All four analysts ran end-to-end and propagated their reports.
    assert result["market_report"]
    assert result["fundamentals_report"]
    assert result["news_report"]
    assert result["sentiment_report"]

    # The debate → judge → trader chain produced its structured artifacts.
    assert result["investment_judge"]["signal"] in InvestmentSignal.__args__
    assert result["trader"]["action"] in ("sell", "hold", "buy")

    # The canonical final artifact: the Portfolio Manager's decision.
    decision = result["portfolio_decision"]
    assert decision["rating"] in InvestmentSignal.__args__
    assert 0.0 <= decision["confidence"] <= 1.0
