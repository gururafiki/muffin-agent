# Backtester

Walk-forward backtester for the persona council + paper-trading pipeline.  Two modes (`full` / `signals`), monthly rebalance default, deterministic performance metrics.

## Two modes

| Mode | What runs per rebalance | Cost | Use when |
|---|---|---|---|
| `full` | Entire [`portfolio_decision_graph`](../src/muffin_agent/agents/portfolio_decision.py) (council × N tickers + sizing + reconciler + execution) | High | Validating end-to-end strategy returns |
| `signals` | Council per ticker only (no sizing / execution) | Low | Iterating on persona prompts; pure signal-quality measurement |

## Programmatic API

The CLI is a configuration stub today (see "Known limitations" below).  Use the engine programmatically with a caller-supplied `prices_provider`:

```python
from muffin_agent.backtesting import BacktestEngine, synthetic_price_provider

engine = BacktestEngine(
    start="2024-01-01",
    end="2024-12-31",
    tickers=["AAPL", "MSFT"],
    initial_cash=100_000,
    mode="signals",
    rebalance_freq="ME",  # monthly (pandas offset alias)
    benchmark="SPY",
)

# In tests: synthetic provider.  In production: wrap OpenBB MCP.
prices_provider = synthetic_price_provider(
    {"AAPL": 150.0, "MSFT": 300.0}, daily_drift=0.001
)

results = await engine.run(prices_provider=prices_provider)
print(results.metrics)
print(results.to_dataframe())
```

### `BacktestResults`

* `history: list[RebalanceSnapshot]` — one entry per rebalance with `nav`, `cash`, `long_value`, `short_exposure`, `realized_gains_total`, `margin_used`, `benchmark_value`, `orders`, `executed_trades`, `council_synthesis`.
* `final_portfolio: dict` — `Portfolio.model_dump()` at the end of the run.
* `final_value: dict` — `PortfolioValue.model_dump()` of the last snapshot.
* `metrics: dict[str, float | None]` — `total_return`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `rebalance_count`, plus benchmark fields (`portfolio_total_return`, `benchmark_total_return`, `alpha`, `tracking_error`).
* `to_dataframe()` — pandas DataFrame view of `history`.

## Metrics

[`metrics.py`](../src/muffin_agent/backtesting/metrics.py) is pure-Python (statistics stdlib + math; no scipy):

* `compute_sharpe(returns, risk_free_rate, frequency)` — annualised
* `compute_sortino(returns, risk_free_rate, frequency)` — downside-deviation variant
* `compute_max_drawdown(equity_curve)` — non-positive decimal
* `compute_total_return(equity_curve)` — start → end
* `compute_returns_from_equity(equity_curve)` — per-period
* `compute_benchmark_comparison(portfolio_curve, benchmark_prices)` — alpha + tracking error

Default risk-free rate: 4.34% (matches ai-hedge-fund's upstream).

## Frequency aliases

The engine uses pandas offset aliases.  Defaults to monthly (`ME` for "month end" in pandas 3).  Supported (informally — anything pandas accepts works):

| Alias | Periods/year used for annualisation |
|---|---|
| `ME` (default) | 12 |
| `W` / `W-FRI` | 52 |
| `QE` | 4 |
| `YE` | 1 |

## OutcomesFetcher

For reflection-style backtests (where you'd want to feed past decisions back into the persona prompts via realised outcomes), the canonical hook is [`utils.outcomes.OutcomesFetcher`](../src/muffin_agent/utils/outcomes.py).  This is a re-export of the `OutcomesFetcher` Protocol + default `fetch_decision_outcome` implementation from [`trading_decision/tools.py`](../src/muffin_agent/agents/trading_decision/tools.py) — preserving backward compatibility while making the utility discoverable from a non-trading_decision-specific location.

## Known limitations

* **`prices_provider` is caller-supplied.**  The `muffin backtest` CLI currently prints the configuration and exits — wiring an MCP-backed default provider is a roadmap item.  Use the programmatic API above with a custom or synthetic provider until that lands.
* **As-of-date soft enforcement.**  The engine injects `Decision date: <date>. Ignore data after this date.` into the council's `query` so the LLM is asked to ignore future data.  This is a soft constraint — some OpenBB MCP endpoints don't enforce as-of dates server-side.  `mode="signals"` is the cleaner mode for reproducibility while we harden the as-of plumbing.

## CLI

```bash
muffin backtest AAPL,MSFT \
    --start 2024-01-01 \
    --end 2024-12-31 \
    --mode signals \
    --freq ME \
    --initial-cash 100000 \
    --benchmark SPY
```

Today this prints the configuration and exits with code 1 (price-provider wiring is the roadmap item).  Use the programmatic API in the meantime.
