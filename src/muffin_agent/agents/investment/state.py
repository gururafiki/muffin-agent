"""State schemas for the investment workflow agents.

- ``TickerAnalysisState`` — per-ticker state flowing through the analysis
  sub-graph.  Each parallel worker gets its own instance.
- ``ScreeningState`` — outer state for the equity screening graph.
  ``theses`` uses an ``operator.add`` reducer so parallel ticker workers can
  each append their result without overwriting one another.
"""

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict


class TickerAnalysisState(TypedDict):
    """State flowing through the per-ticker analysis sub-graph.

    Fields are written by stage nodes (in parallel for Group 1, sequentially
    for Groups 2 and 3).  Every node returns a partial dict — only the keys
    it owns.
    """

    # ── Input ────────────────────────────────────────────────────────────────
    ticker: str
    """Equity ticker symbol, e.g. ``"AAPL"``."""

    query: str
    """Investment mandate / natural-language context for this analysis run."""

    # ── Group 1 (parallel) ───────────────────────────────────────────────────
    market_regime: dict[str, Any]
    """Output of ``market_regime_node``.

    Macro/liquidity regime classification, factor tailwinds/headwinds,
    recommended gross/net/beta ranges.
    """

    sector_view: dict[str, Any]
    """Output of ``sector_analysis_node``.

    Industry structure, cycle position, sector ETF performance, thematic
    backdrop, regulatory climate.
    """

    company_analysis: dict[str, Any]
    """Output of ``company_analysis_node``.

    Business quality, moat assessment, ESG triage, management quality,
    historical 3-statement financials, capital allocation track-record.
    """

    # ── Group 2 (parallel, start after company_analysis) ─────────────────────
    forecast: dict[str, Any]
    """Output of ``forecasting_node``.

    3-statement forward model with bull / base / bear scenarios, analyst
    consensus anchoring, and key driver sensitivity table.
    """

    risk_assessment: dict[str, Any]
    """Output of ``risk_assessment_node``.

    Factor exposures (Fama-French), beta, VaR, max drawdown, implied
    volatility surface, short-interest crowding, stress-test scenarios.
    """

    # ── Group 3 (sequential) ─────────────────────────────────────────────────
    valuation: dict[str, Any]
    """Output of ``valuation_node``.

    DCF, EV/EBITDA, P/E, EV/FCF, sum-of-parts; relative value vs. peers and
    historical ranges; analyst price-target distribution.
    """

    thesis: dict[str, Any]
    """Output of ``thesis_synthesis_node``.

    Completed investment memo: conviction score, signal, key catalysts,
    risk/reward skew, suggested position sizing parameters.
    """


class ScreeningState(TypedDict):
    """Outer state for the equity screening graph (auto-discovery entry point).

    ``theses`` is the only list-accumulating field: parallel ticker workers
    each append one thesis dict, so the reducer must be ``operator.add``.
    """

    # ── Input ────────────────────────────────────────────────────────────────
    query: str
    """Investment mandate / screening objective."""

    # ── Shared context (written once before fan-out) ──────────────────────────
    tickers: list[str]
    """Candidate tickers produced by ``idea_sourcing_node``."""

    market_regime: dict[str, Any]
    """Shared macro regime context — computed once, injected into every ticker
    sub-graph so it is not re-fetched per ticker."""

    sector_view: dict[str, Any]
    """Shared sector/industry backdrop — computed once before fan-out."""

    # ── Fan-in accumulator ───────────────────────────────────────────────────
    theses: Annotated[list[dict[str, Any]], operator.add]
    """Thesis dicts collected from all parallel ticker sub-graphs.

    Each parallel worker appends ``[thesis_dict]``; the ``operator.add``
    reducer merges lists correctly across concurrent branches.
    """

    # ── Final output ─────────────────────────────────────────────────────────
    comparison: dict[str, Any]
    """Output of ``comparison_node`` — ranked candidates with conviction scores
    and recommended watch-list or allocation."""
