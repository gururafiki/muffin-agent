"""Tests for the backtest engine — walk-forward loop + results aggregation."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from muffin_agent.agents.personas_council.backtesting import (
    BacktestEngine,
    BacktestResults,
    synthetic_price_provider,
)


@pytest.mark.unit
class TestBacktestEngineConstruction:
    def test_default_initial_cash(self):
        e = BacktestEngine(
            start="2025-01-01", end="2025-03-31", tickers=["AAPL"], mode="signals"
        )
        assert e.portfolio.cash == 100_000.0
        assert e.mode == "signals"
        assert e.rebalance_freq == "ME"

    def test_monthly_rebalance_dates(self):
        e = BacktestEngine(
            start="2025-01-01",
            end="2025-04-30",
            tickers=["AAPL"],
            rebalance_freq="ME",
        )
        dates = e._rebalance_dates()
        assert dates == [
            "2025-01-31",
            "2025-02-28",
            "2025-03-31",
            "2025-04-30",
        ]

    def test_weekly_rebalance_dates(self):
        e = BacktestEngine(
            start="2025-01-01",
            end="2025-01-31",
            tickers=["AAPL"],
            rebalance_freq="W-FRI",
        )
        dates = e._rebalance_dates()
        # 5 Fridays in January 2025
        assert len(dates) == 5
        for d in dates:
            assert d.startswith("2025-01")


@pytest.mark.unit
class TestSyntheticPriceProvider:
    @pytest.mark.asyncio
    async def test_returns_drifted_prices(self):
        provider = synthetic_price_provider(
            {"AAPL": 150.0, "MSFT": 300.0}, daily_drift=0.001
        )
        early = await provider("2025-01-01")
        later = await provider("2025-06-30")
        # Prices should drift up over time
        assert later["AAPL"] > early["AAPL"]
        assert later["MSFT"] > early["MSFT"]


@pytest.mark.unit
@pytest.mark.asyncio
class TestRunSignalsMode:
    async def test_run_with_no_dates_raises(self):
        # End < start → empty pd.date_range
        e = BacktestEngine(
            start="2025-12-01",
            end="2025-01-01",
            tickers=["AAPL"],
            mode="signals",
        )
        provider = synthetic_price_provider({"AAPL": 150.0})
        with pytest.raises(ValueError, match="No rebalance dates"):
            await e.run(prices_provider=provider)

    async def test_signals_mode_records_history(self):
        e = BacktestEngine(
            start="2025-01-01",
            end="2025-04-30",
            tickers=["AAPL"],
            mode="signals",
            initial_cash=100_000.0,
        )
        prices_provider = synthetic_price_provider({"AAPL": 150.0})

        # Mock the council graph so we don't hit MCP
        fake_graph = AsyncMock()
        fake_graph.ainvoke = AsyncMock(
            return_value={"council_synthesis": {"consensus_rating": "buy"}}
        )
        with patch(
            "muffin_agent.agents.personas_council.backtesting.engine.build_council_graph",
            return_value=fake_graph,
        ):
            results = await e.run(prices_provider=prices_provider)

        assert isinstance(results, BacktestResults)
        assert len(results.history) == 4  # 4 month-ends
        # Signals mode → no orders / executed
        for snap in results.history:
            assert snap.orders == []
            assert snap.executed_trades == []
            assert "AAPL" in snap.council_synthesis
        # Equity curve = cash only (no trades)
        assert all(snap.nav == 100_000.0 for snap in results.history)
        # Metrics dict has the expected fields
        assert "total_return" in results.metrics
        assert "sharpe_ratio" in results.metrics
        assert "max_drawdown" in results.metrics

    async def test_signals_mode_records_benchmark_value(self):
        e = BacktestEngine(
            start="2025-01-01",
            end="2025-03-31",
            tickers=["AAPL"],
            mode="signals",
        )
        prices_provider = synthetic_price_provider({"AAPL": 150.0})
        spy_values = iter([400.0, 410.0, 420.0])

        async def bench(date: str) -> float:
            return next(spy_values)

        fake_graph = AsyncMock()
        fake_graph.ainvoke = AsyncMock(return_value={"council_synthesis": {}})
        with patch(
            "muffin_agent.agents.personas_council.backtesting.engine.build_council_graph",
            return_value=fake_graph,
        ):
            results = await e.run(
                prices_provider=prices_provider, benchmark_provider=bench
            )

        bench_values = [snap.benchmark_value for snap in results.history]
        assert bench_values == [400.0, 410.0, 420.0]
        assert "benchmark_total_return" in results.metrics


@pytest.mark.unit
class TestBacktestResultsToDataFrame:
    @pytest.mark.asyncio
    async def test_df_columns(self):
        e = BacktestEngine(
            start="2025-01-01",
            end="2025-03-31",
            tickers=["AAPL"],
            mode="signals",
        )
        provider = synthetic_price_provider({"AAPL": 150.0})
        fake_graph = AsyncMock()
        fake_graph.ainvoke = AsyncMock(return_value={"council_synthesis": {}})
        with patch(
            "muffin_agent.agents.personas_council.backtesting.engine.build_council_graph",
            return_value=fake_graph,
        ):
            results = await e.run(prices_provider=provider)
        df = results.to_dataframe()
        for col in (
            "date",
            "nav",
            "cash",
            "long_value",
            "short_exposure",
            "margin_used",
        ):
            assert col in df.columns
