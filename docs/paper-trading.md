# Paper-Trading Pipeline

Multi-ticker portfolio decision graph that composes the [persona council](personas.md) with a deterministic position-sizing layer and an LLM-mediated portfolio reconciler.  Produces concrete share-count orders + an updated portfolio snapshot.

**Independent of [`trading_decision`](trading-decision.md)** — uses the persona council, not Bull / Bear / Judge / Trader.

## CLI

```bash
# Dry-run on a few tickers using a fresh portfolio
muffin trade AAPL,MSFT,NVDA --initial-cash 100000

# Persist the new state to ~/.muffin/portfolios/<name>.json
muffin trade AAPL,MSFT --portfolio my-port --apply

# Custom investment mandate
muffin trade AAPL,MSFT -q "Long-only quality bias"
```

## Architecture

```
START
  │
  ▼
[Send fan-out × N tickers]
  │
  ▼
┌────────────────────────────────────────────────────────────┐
│  Per ticker:                                                │
│    council_graph (13 persona subgraphs in parallel → judge)│
│    → ticker_decision_node (LLM consolidator)               │
│  Writes {ticker: ticker_decision} via merge reducer        │
└────────────────────────────────────────────────────────────┘
  │
  │  (barrier: ticker_decisions accumulated across workers)
  ▼
[position_sizing_node]      ← deterministic vol × correlation
  ↓
[portfolio_reconciler_node] ← hybrid deterministic + LLM
  ↓
[execute_orders]            ← portfolio.executor.apply_orders
  ↓
END
```

### Per-ticker subgraph

The inner subgraph (compiled once, reused via `Send`) runs the full council + a per-ticker consolidator:

* [`council_graph`](../src/muffin_agent/agents/personas/council_graph.py) — 13 personas + judge synthesis
* [`ticker_decision_node`](../src/muffin_agent/agents/portfolio/ticker_decision.py) — LLM picks a per-ticker `recommended_action` + `target_pct_of_nav`

### Position sizing (deterministic, no LLM)

[`position_sizing_node`](../src/muffin_agent/agents/portfolio/position_sizing.py) ports ai-hedge-fund's `risk_manager.py` exactly:

* **Volatility bucket** → base limit pct  
  `<15%` → 25% / `15-30%` → 20%→12.5% / `30-50%` → 15%→5% / `≥50%` → 10%
* **Correlation multiplier** → adjusts base limit  
  `≥0.80` → 0.70× / `0.60-0.80` → 0.85× / `0.40-0.60` → 1.00× / `0.20-0.40` → 1.05× / `<0.20` → 1.10×
* `limit_dollars = nav × base_pct × corr_mult`
* `remaining_dollars = limit_dollars − current_value`

### Portfolio reconciler (hybrid)

[`portfolio_reconciler_node`](../src/muffin_agent/agents/portfolio/portfolio_reconciler.py):

1. **Deterministic pre-pass** — compute `allowed_actions` per ticker (max buy / sell / short / cover shares within constraints).  Pre-fill `hold` for any ticker with no valid non-hold action (saves the LLM compute).
2. **LLM call** (skipped entirely when all tickers are pre-filled) — with a compact representation of decisions + allowed actions + current prices.  Returns concrete share-count orders.

### Trade execution

[`portfolio.executor.apply_orders`](../src/muffin_agent/portfolio/executor.py) is a pure function that applies orders to the portfolio and returns `(new_portfolio, executed_trades)`.  Defensively validates legality — illegal orders are silently dropped with `skipped=True` in the trade log.

## Portfolio state

[`portfolio.state.Portfolio`](../src/muffin_agent/portfolio/state.py) is a Pydantic-immutable snapshot:

```python
class Portfolio(BaseModel):
    cash: float
    margin_requirement: float = 0.5
    margin_used: float = 0.0
    positions: dict[str, Position]            # per-ticker long/short + cost basis
    realized_gains: dict[str, RealizedGain]   # per-ticker long/short P&L
```

Mutation helpers are **pure functions** returning `(new_portfolio, executed_qty)`:

* `apply_long_buy(portfolio, ticker, qty, price)` — partial fill when cash is short
* `apply_long_sell(portfolio, ticker, qty, price)` — realises gain into `realized_gains[ticker].long`
* `apply_short_open(portfolio, ticker, qty, price)` — locks `qty × price × margin_requirement` as margin
* `apply_short_cover(portfolio, ticker, qty, price)` — releases proportional margin

`mark_to_market(portfolio, prices)` returns a [`PortfolioValue`](../src/muffin_agent/portfolio/state.py) with cash / long_value / short_exposure / realised_gains_total / NAV / margin_used.

## Persistence

Portfolios are JSON dumps under `~/.muffin/portfolios/<name>.json`.  `muffin trade --apply` writes back; without `--apply` the command is a dry-run that only prints the orders.
