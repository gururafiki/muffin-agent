"""Investment process pipeline — composable, stage-based multi-agent workflow.

Two entry-point graphs are exposed:

``build_analysis_graph()``
    Direct-ticker analysis: accepts a single ticker and investment mandate,
    runs all 7 analysis stages (with parallel execution where dependencies
    allow), and returns a completed investment thesis.

``build_screening_graph()``
    Auto-discovery pipeline: screens the market for candidate tickers, runs
    ``build_analysis_graph()`` for each candidate in parallel, then ranks and
    compares the results.

Both factory functions accept a ``stages`` dict for plugging in alternative
node implementations without touching the graph wiring.
"""

from muffin_agent.pipeline.graphs import build_analysis_graph, build_screening_graph

__all__ = ["build_analysis_graph", "build_screening_graph"]
