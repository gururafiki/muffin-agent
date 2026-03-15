"""Stage 9: Thesis Synthesis & Investment Case."""

from typing import Any

from langchain_core.runnables import RunnableConfig

from muffin_agent.agents.investment.state import TickerAnalysisState


async def thesis_synthesis_node(
    state: TickerAnalysisState, config: RunnableConfig
) -> dict[str, Any]:
    """Stage 9: Thesis Synthesis & Investment Case.

    Synthesises all prior stage outputs into a concise, structured investment
    memo.  This is a **pure-reasoning** stage: no new data is fetched.  The
    agent reads the full accumulated state and produces a final conviction
    score, signal, key catalysts, risk/reward framing, and suggested position
    parameters.

    Runs **sequentially** after ``valuation_node`` (the last Group 3 stage).

    Inputs (from state):
        ticker: Equity ticker symbol.
        query: Original investment mandate.
        market_regime: Top-down macro frame.
        sector_view: Industry and thematic context.
        company_analysis: Business quality assessment.
        forecast: Bull / base / bear financial model.
        risk_assessment: Risk metrics, factor loadings, stress scenarios.
        valuation: Intrinsic value range, upside/downside, relative value.

    Outputs (state update):
        thesis: dict containing, e.g.:
            - signal: str — "strong_buy" | "buy" | "hold" | "sell" | "strong_sell"
            - conviction: float — 0.0–1.0 overall conviction score
            - summary: str — 3–5 sentence investment case
            - bull_case: str — key upside scenario narrative
            - bear_case: str — key downside scenario narrative
            - key_catalysts: list[str] — near-term conviction-building events
            - key_risks: list[str] — conviction-destroyers to monitor
            - price_target: float — base-case 12-month price target
            - stop_level: float | None — suggested stop-loss from risk_assessment
            - suggested_position_size: str — e.g. "1–2% of NAV starter" or
              "full 3% on confirmation of catalyst"
            - monitoring_checklist: list[str] — ongoing thesis drift indicators

    Planned agent type:
        Pure-reasoning deep agent (``create_deep_agent``) with no data
        collection subagents — analogous to ``data_validation.py``.
        Optionally adds ``news`` → ``news_company`` as a single subagent for
        a last-mile catalyst freshness check.
    """
    raise NotImplementedError("thesis_synthesis_node not yet implemented")
