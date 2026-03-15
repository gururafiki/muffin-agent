"""Comparison stage: rank and compare evaluated candidates (screening_graph only)."""

from typing import Any

from langchain_core.runnables import RunnableConfig

from muffin_agent.agents.investment.state import ScreeningState


async def comparison_node(
    state: ScreeningState, config: RunnableConfig
) -> dict[str, Any]:
    """Comparison: Rank and Compare All Evaluated Candidates.

    Aggregates the ``theses`` list (one thesis dict per ticker, collected via
    the ``operator.add`` reducer from parallel workers) and produces a ranked
    shortlist with a relative attractiveness assessment.

    This stage only exists in ``screening_graph``; the ``analysis_graph``
    ends at ``thesis_synthesis_node``.

    Inputs (from state):
        theses: List of thesis dicts, one per ticker (e.g. 3–20 entries).
        query: Original investment mandate for relative ranking context.
        market_regime: Shared top-down frame for cross-asset comparison.

    Outputs (state update):
        comparison: dict containing, e.g.:
            - ranked_tickers: list[str] — tickers sorted by conviction desc.
            - ranking_rationale: list[dict] — per-ticker rank, conviction,
              signal, key differentiator vs. next-best alternative
            - top_pick: str — single best idea with one-line rationale
            - watch_list: list[str] — interesting but not yet actionable
            - pass_list: list[str] — screened out with brief reason
            - portfolio_fit_notes: str — concentration, correlation,
              diversification observations across candidates

    Planned agent type:
        Pure-reasoning deep agent (``create_deep_agent``) with no data
        collection subagents.  Receives all theses as context and performs
        structured comparison reasoning.
    """
    raise NotImplementedError("comparison_node not yet implemented")
