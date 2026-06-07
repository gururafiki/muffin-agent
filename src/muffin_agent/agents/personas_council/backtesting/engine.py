"""Backtest engine — walk-forward replay of the portfolio decision pipeline.

Two modes:

* **Mode A — ``full``**: invokes :func:`build_portfolio_decision_graph`
  at every rebalance date.  Expensive (council × N tickers per date) but
  highest fidelity.
* **Mode B — ``signals``**: invokes only the council per ticker (no
  position-sizing or reconciliation).  Useful for prompt-iteration on
  signals without paying for the full graph cost.

Rebalance frequency defaults to **monthly** — realistic for fundamental
strategies and inexpensive enough to run.  Weekly / quarterly supported
via the ``rebalance_freq`` argument (pandas offset alias).

As-of data fidelity: muffin's OpenBB MCP tools don't enforce
``as_of_date`` server-side for every endpoint.  The engine injects
``decision_date`` into the council's ``query`` so the LLM is asked to
ignore data after that date — soft constraint, documented in the
roadmap.  Use ``mode="signals"`` for cleaner reproducibility while we
harden the as-of plumbing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from ..council_graph import build_council_graph
from ..portfolio.portfolio_decision import build_portfolio_decision_graph
from ..portfolio.state import Portfolio, PortfolioValue, mark_to_market, new_portfolio
from .metrics import (
    DEFAULT_RISK_FREE_RATE,
    compute_benchmark_comparison,
    compute_max_drawdown,
    compute_returns_from_equity,
    compute_sharpe,
    compute_sortino,
    compute_total_return,
)

logger = logging.getLogger(__name__)

BacktestMode = Literal["full", "signals"]


@dataclass
class RebalanceSnapshot:
    """One row in the walk-forward log."""

    date: str
    nav: float
    cash: float
    long_value: float
    short_exposure: float
    realized_gains_total: float
    margin_used: float
    benchmark_value: float | None
    orders: list[dict[str, Any]] = field(default_factory=list)
    executed_trades: list[dict[str, Any]] = field(default_factory=list)
    council_synthesis: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Mapping ticker → council synthesis dict, only in signals mode."""


@dataclass
class BacktestResults:
    """Final summary of a backtest run."""

    history: list[RebalanceSnapshot]
    final_portfolio: dict[str, Any]
    final_value: dict[str, Any]
    metrics: dict[str, float | None]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the history list to a pandas DataFrame for analysis."""
        return pd.DataFrame(
            {
                "date": [s.date for s in self.history],
                "nav": [s.nav for s in self.history],
                "cash": [s.cash for s in self.history],
                "long_value": [s.long_value for s in self.history],
                "short_exposure": [s.short_exposure for s in self.history],
                "realized_gains_total": [s.realized_gains_total for s in self.history],
                "margin_used": [s.margin_used for s in self.history],
                "benchmark_value": [s.benchmark_value for s in self.history],
            }
        )


class BacktestEngine:
    """Walk-forward backtest engine.

    Caller supplies a ``prices_provider`` callable that returns
    ``(close, ohlcv_bars_up_to_date)`` for ``(ticker, date)``.  In
    production this wraps OpenBB MCP via the persona data-collection
    step (the council is invoked with ``decision_date=date``).  In tests
    a synthetic provider can be passed for reproducibility.
    """

    def __init__(
        self,
        *,
        start: str,
        end: str,
        tickers: list[str],
        portfolio: Portfolio | None = None,
        initial_cash: float = 100_000.0,
        mode: BacktestMode = "full",
        rebalance_freq: str = "ME",
        benchmark: str = "SPY",
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        checkpointer: BaseCheckpointSaver | None = None,
        store: BaseStore | None = None,
    ) -> None:
        """Initialise the backtest engine with the supplied configuration."""
        self.start = start
        self.end = end
        self.tickers = list(tickers)
        self.portfolio = portfolio or new_portfolio(initial_cash=initial_cash)
        self.mode = mode
        self.rebalance_freq = rebalance_freq
        self.benchmark = benchmark
        self.risk_free_rate = risk_free_rate
        self.checkpointer = checkpointer
        self.store = store
        self.history: list[RebalanceSnapshot] = []

    def _rebalance_dates(self) -> list[str]:
        idx = pd.date_range(start=self.start, end=self.end, freq=self.rebalance_freq)
        return [d.strftime("%Y-%m-%d") for d in idx]

    async def _run_full_mode_step(
        self,
        date: str,
        current_prices: dict[str, float],
        config: RunnableConfig,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict]]:
        """Invoke the portfolio decision graph for a single rebalance date.

        Returns ``(orders, executed_trades, council_synthesis_by_ticker)``.
        """
        graph = await build_portfolio_decision_graph(
            config=config, checkpointer=self.checkpointer, store=self.store
        )
        result = await graph.ainvoke(
            {
                "tickers": self.tickers,
                "query": f"Decision date: {date}. Ignore data after this date.",
                "portfolio": self.portfolio.model_dump(),
                "current_prices": current_prices,
            },
            config,
        )
        orders = result.get("orders", [])
        executed = result.get("executed_trades", [])
        # Update the portfolio with the graph's final portfolio state
        new_portfolio_dump = result.get("portfolio")
        if new_portfolio_dump:
            self.portfolio = Portfolio.model_validate(new_portfolio_dump)
        # Council syntheses aren't surfaced in the full graph result by default
        return orders, executed, {}

    async def _run_signals_mode_step(
        self,
        date: str,
        current_prices: dict[str, float],
        config: RunnableConfig,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict]]:
        """Signals-only mode: run council per ticker, skip sizing/reconciliation.

        Returns empty orders/trades + per-ticker council synthesis.  Used
        to test signal quality without exercising the trade-execution
        machinery.
        """
        council = await build_council_graph(config, store=self.store)
        synth_by_ticker: dict[str, dict[str, Any]] = {}
        for ticker in self.tickers:
            result = await council.ainvoke(
                {
                    "ticker": ticker,
                    "query": (f"Decision date: {date}. Ignore data after this date."),
                },
                config,
            )
            synth_by_ticker[ticker] = result.get("council_synthesis") or {}
        return [], [], synth_by_ticker

    async def run(
        self,
        *,
        prices_provider: Any,
        benchmark_provider: Any | None = None,
        config: RunnableConfig | None = None,
    ) -> BacktestResults:
        """Execute the walk-forward backtest.

        Args:
            prices_provider: Async callable
                ``(date) -> dict[ticker, current_price]`` returning
                close prices for *date*.
            benchmark_provider: Optional async callable
                ``(date) -> float | None`` returning the benchmark close
                at *date* (default tracker: SPY).
            config: ``RunnableConfig`` (e.g. callbacks / configurable
                fields) forwarded to the inner graph invocations.

        Returns:
            :class:`BacktestResults` with the per-rebalance history,
            final portfolio state, and summary metrics.
        """
        cfg: RunnableConfig = config or {"configurable": {}}
        dates = self._rebalance_dates()
        if not dates:
            raise ValueError(
                f"No rebalance dates in range {self.start} → {self.end} "
                f"with freq {self.rebalance_freq!r}"
            )

        for date in dates:
            current_prices: dict[str, float] = await prices_provider(date) or {}
            benchmark_value: float | None = None
            if benchmark_provider is not None:
                benchmark_value = await benchmark_provider(date)

            orders: list[dict[str, Any]]
            executed: list[dict[str, Any]]
            council_synth: dict[str, dict[str, Any]]
            if self.mode == "full":
                orders, executed, council_synth = await self._run_full_mode_step(
                    date, current_prices, cfg
                )
            else:
                orders, executed, council_synth = await self._run_signals_mode_step(
                    date, current_prices, cfg
                )

            value = mark_to_market(self.portfolio, current_prices)
            self.history.append(
                RebalanceSnapshot(
                    date=date,
                    nav=value.nav,
                    cash=value.cash,
                    long_value=value.long_value,
                    short_exposure=value.short_exposure,
                    realized_gains_total=value.realized_gains_total,
                    margin_used=value.margin_used,
                    benchmark_value=benchmark_value,
                    orders=orders,
                    executed_trades=executed,
                    council_synthesis=council_synth,
                )
            )

        return self._build_results()

    def _build_results(self) -> BacktestResults:
        equity_curve = [s.nav for s in self.history]
        bench_curve = [
            s.benchmark_value for s in self.history if s.benchmark_value is not None
        ]
        period_returns = compute_returns_from_equity(equity_curve)
        # Annualisation: monthly → 12, weekly → 52, etc.  Default = monthly.
        per_year = {
            "M": 12,
            "ME": 12,
            "MS": 12,
            "W": 52,
            "W-FRI": 52,
            "Q": 4,
            "QE": 4,
            "Y": 1,
            "YE": 1,
        }.get(self.rebalance_freq, 12)
        sharpe = compute_sharpe(period_returns, self.risk_free_rate, frequency=per_year)
        sortino = compute_sortino(
            period_returns, self.risk_free_rate, frequency=per_year
        )
        max_dd = compute_max_drawdown(equity_curve)
        total_ret = compute_total_return(equity_curve)
        bench_metrics = (
            compute_benchmark_comparison(equity_curve, bench_curve)
            if bench_curve
            else {}
        )
        metrics: dict[str, float | None] = {
            "total_return": total_ret,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "rebalance_count": float(len(self.history)),
        }
        metrics.update(bench_metrics)

        final_value = (
            mark_to_market(
                self.portfolio,
                {s.date: 0.0 for s in []},  # final mark-to-market deferred
            )
            if not self.history
            else PortfolioValue(
                cash=self.history[-1].cash,
                long_value=self.history[-1].long_value,
                short_exposure=self.history[-1].short_exposure,
                realized_gains_total=self.history[-1].realized_gains_total,
                nav=self.history[-1].nav,
                margin_used=self.history[-1].margin_used,
            )
        )
        return BacktestResults(
            history=self.history,
            final_portfolio=self.portfolio.model_dump(),
            final_value=final_value.model_dump(),
            metrics=metrics,
        )


# ── Synthetic provider (test fixture) ────────────────────────────────────────


def synthetic_price_provider(
    base_prices: dict[str, float],
    daily_drift: float = 0.001,
):
    """Return a deterministic ``prices_provider`` for tests.

    Generates a ``current_prices`` dict that compounds *base_prices* by
    ``daily_drift`` each calendar day past 2025-01-01.  Useful for
    unit-testing the engine without hitting MCP.
    """

    async def _provider(date: str) -> dict[str, float]:
        days = (pd.Timestamp(date) - pd.Timestamp("2025-01-01")).days
        drift = (1 + daily_drift) ** max(days, 0)
        return {ticker: price * drift for ticker, price in base_prices.items()}

    return _provider
