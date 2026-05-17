# Trading Decision Pipeline

A composable trading-decision pipeline ported from [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents). Decoupled from `investment_analysis`: every agent accepts a generic `AnalysisContext` envelope, so callers can plug in muffin's pipeline output, free-form research notes, or any other upstream analysis source.

This document covers what the pipeline does, what each agent contributes, and the composition patterns. For the per-file architecture reference, see the relevant entries in [CLAUDE.md](../CLAUDE.md#L120).

---

## What the pipeline produces

The canonical artifact is `PortfolioDecisionOutput`:

```python
class PortfolioDecisionOutput(BaseModel):
    rating: Literal["strong_sell", "sell", "hold", "buy", "strong_buy"]
    executive_summary: str          # 2-4 sentence headline
    investment_thesis: str          # detailed reasoning
    price_target: float | None
    stop_loss: float | None
    time_horizon: str               # e.g. "3-6 months"
    position_sizing: str            # e.g. "2% NAV starter, scale to 4% on Q1 beat"
    key_risks_remaining: list[str]
    confidence: float               # 0.0-1.0
    incorporates_past_lessons: bool # set true when the PM cited reflection memory
```

The same 5-tier vocabulary (`strong_sell` … `strong_buy`) is shared with `CriteriaAnalysisSynthesis.signal` so both pipelines speak the same rating language.

---

## Pipeline shape

```
START
  ↓
[reflector_resolve]      ← resolve prior pending decisions + inject past reflections
  ↓
[Bull Researcher] ⇄ [Bear Researcher]    ← N rounds (default 2)
  ↓
[Investment Judge]       ← InvestmentJudgeOutput (signal + bull/bear cases + catalysts/risks)
  ↓
[Trader]                 ← TraderOutput (action + entry/stop/take-profit + sizing + horizon)
  ↓
[Aggressive] → [Conservative] → [Neutral]    ← M rounds (default 1)
  ↓
[Portfolio Manager]      ← PortfolioDecisionOutput (canonical final artifact)
  ↓
[decision_writeback]     ← persist current decision as pending for future reflection
  ↓
END
```

Three builders share `TradingDecisionState`; callers opt into depth:

| Builder | Topology |
|---|---|
| `build_investment_debate_graph` | Bull/Bear debate → Judge (smallest useful slice) |
| `build_investment_thesis_graph` | …+ Trader (adds operational translation) |
| `build_trading_decision_graph` | full pipeline above (canonical 5-tier decision + reflection bookends) |

---

## Why adversarial debate

The Bull and Bear researchers cite the same `AnalysisContext` but argue opposite directions. Each rebuts the other's last argument (not just restates its own thesis), which forces both sides of the case onto the record. The Investment Judge then commits to a directional signal — with explicit prompt discipline to **not** default to `hold` under uncertainty.

The 3-way risk debate works the same way one layer down. Aggressive argues for pressing the position; Conservative argues for protection; Neutral calls out where either extreme over-presses. The Portfolio Manager synthesises all three plus the Judge thesis and the Trader proposal into the canonical decision.

Per-run round counts (configurable):

* `configurable.max_investment_debate_rounds` — default `2` = Bull → Bear → Bull → Bear (allows one round of rebuttal)
* `configurable.max_risk_debate_rounds` — default `1` = Aggressive → Conservative → Neutral (one pass per persona)

---

## Outcome-driven reflection memory

The reflection layer turns the pipeline into a learning loop:

1. **Write (end of every run)** — `decision_writeback_node` persists the current `PortfolioDecisionOutput` as a `pending` record under `("memories", user_id, "decisions")` with key `f"{TICKER}:{YYYY-MM-DD}"`.
2. **Resolve (start of every run)** — `reflector_resolve_node` walks all pending entries (any ticker), calls `fetch_outcomes_openbb` for realised returns + alpha vs benchmark (default `SPY`), and runs the Reflector LLM to produce a 2–4 sentence reflection. Resolved records persist in the same store.
3. **Inject (same start)** — The most-recent `reflection_max_same_ticker` (5) + `reflection_max_cross_ticker` (3) resolved reflections are rendered as a Markdown block and injected into the Portfolio Manager prompt. The PM sets `incorporates_past_lessons=true` when it actually cites them.

Knobs:

| Configurable key | Default | Purpose |
|---|---|---|
| `reflection_enabled` | `true` | Skip the entire layer when `false` |
| `decision_date` | today UTC | Override for deterministic tests |
| `reflection_holding_days` | `5` | Trading-day window for return computation |
| `reflection_benchmark` | `"SPY"` | Alpha benchmark ticker |
| `reflection_max_same_ticker` | `5` | Same-ticker reflections injected into PM prompt |
| `reflection_max_cross_ticker` | `3` | Cross-ticker reflections injected into PM prompt |

The pipeline degrades silently when reflection infrastructure is unavailable (no store, no resolvable `user_id`, `reflection_enabled=false`, or OpenBB MCP unreachable). A failed outcome fetch leaves the entry `pending` so the next run can retry.

---

## Composition patterns

`AnalysisContext` is the abstraction that decouples the pipeline from any specific upstream analysis. All structured fields are optional; `narrative` is the always-available fallback. Prompts use Jinja conditionals so a missing field renders to nothing rather than an "unknown" placeholder.

### Pattern 1: Ad-hoc notes

```python
from muffin_agent.agents.trading_decision import (
    AnalysisContext,
    build_trading_decision_graph,
)
from langgraph.store.memory import InMemoryStore

graph = build_trading_decision_graph(store=InMemoryStore())
context = AnalysisContext.from_narrative(
    "AAPL",
    "Bull thesis: durable services growth. Bear: China demand wobble.",
    query="long-term hold candidate",
)
result = await graph.ainvoke(
    {"analysis_context": context.model_dump()},
    config={"configurable": {"user_id": "alice"}},
)
print(result["portfolio_decision"]["rating"])
```

### Pattern 2: From muffin's investment_analysis pipeline

Use the adapter `AnalysisContext.from_investment_analysis_state(state)` on the output of `build_investment_analysis_graph`. Today the pipeline cannot complete in a single invocation (`thesis_synthesis_node` is a stub) — so this pattern works against a pre-staged state JSON. Once `thesis_synthesis_node` is implemented (or the `interrupt_before` follow-up lands), you'll be able to run both graphs in sequence.

CLI:

```bash
# Step 1: produce or hand-author a TickerAnalysisState JSON.
# (Once `muffin analyze` is wired to output --json, this becomes one pipe.)
muffin decide AAPL --analysis-json state.json --query "long-term hold"

# Combine structured analysis with free-form notes.
muffin decide AAPL --analysis-json state.json \
                   --narrative "Add'l context: management transition"
```

Programmatic:

```python
from muffin_agent.agents.trading_decision import AnalysisContext

state = {                  # produced by build_investment_analysis_graph
    "ticker": "AAPL",
    "query": "long-term hold",
    "market_regime": {...},
    "sector_view": {...},
    "company_analysis": {...},
    "forecast": {...},
    "risk_assessment": {...},
    "valuation": {...},
}

context = AnalysisContext.from_investment_analysis_state(
    state,
    narrative="Additional context that didn't fit the structured fields.",
)
```

### Pattern 3: External research / custom upstream

The adapter doesn't care where the dict came from. Any caller that can populate the six structured field keys can use `from_investment_analysis_state`; any caller without structured fields can fall back to `from_narrative`.

```python
# Custom upstream — e.g. a quant screener that emits its own structured snapshot
ctx = AnalysisContext(
    ticker="AAPL",
    query="momentum + quality",
    valuation={"valuation_signal": "fairly_valued", "pe_ttm": 22.1},
    risk_assessment={"var_95_1m_pct": 6.4, "beta": 1.18},
    narrative="Quant screen flagged this ticker on Q1 EPS revision + momentum.",
    additional_context={"screener_score": 0.81, "factor": "quality"},
)
```

---

## Standalone agent factories

Every agent in the pipeline is independently invocable for use inside other graphs:

```python
from muffin_agent.agents.trading_decision import (
    create_bull_researcher_agent,
    create_bear_researcher_agent,
    create_investment_judge_agent,
    create_trader_agent,
    create_aggressive_debator_agent,
    create_conservative_debator_agent,
    create_neutral_debator_agent,
    create_portfolio_manager_agent,
    create_reflector_agent,
)
```

Each factory accepts the per-turn variables it needs (ticker / query / context_vars / debate history / opposing argument or upstream output) and returns a compiled ReAct agent. The graph node wrappers in `nodes.py` show the canonical invocation pattern for each.

---

## Persistent reflection store (follow-up)

The CLI ships with `InMemoryStore`, so reflection memory persists only within a single Python process. For cross-session persistence:

* **LangGraph Platform**: the runtime injects a managed Postgres store automatically. No changes required.
* **Self-hosted**: wire a `PostgresStore` (see [LangGraph store docs](https://langchain-ai.github.io/langgraph/reference/store/)) by passing it to `build_trading_decision_graph(store=...)`.
* **File-backed**: not currently shipped; tracked in [roadmap.md](../roadmap.md) under "Follow-up — persistent reflection store" as a candidate for a simple SQLite-backed store.

The `--from-analysis` one-shot CLI flag (runs `muffin analyze` then pipes through the adapter automatically) is also a follow-up — blocked on `thesis_synthesis_node` (which currently raises `NotImplementedError`).

---

## Where to look in the code

| Concern | File |
|---|---|
| Schemas | [src/muffin_agent/agents/trading_decision/schemas.py](../src/muffin_agent/agents/trading_decision/schemas.py) |
| State | [src/muffin_agent/agents/trading_decision/state.py](../src/muffin_agent/agents/trading_decision/state.py) |
| Routing | [src/muffin_agent/agents/trading_decision/conditional_logic.py](../src/muffin_agent/agents/trading_decision/conditional_logic.py) |
| Researchers | [src/muffin_agent/agents/trading_decision/researchers/](../src/muffin_agent/agents/trading_decision/researchers/) |
| Trader | [src/muffin_agent/agents/trading_decision/trader.py](../src/muffin_agent/agents/trading_decision/trader.py) |
| Risk debaters | [src/muffin_agent/agents/trading_decision/risk_debate/](../src/muffin_agent/agents/trading_decision/risk_debate/) |
| Portfolio Manager | [src/muffin_agent/agents/trading_decision/portfolio_manager.py](../src/muffin_agent/agents/trading_decision/portfolio_manager.py) |
| Reflection memory | [src/muffin_agent/agents/trading_decision/reflection/](../src/muffin_agent/agents/trading_decision/reflection/) |
| Node wrappers | [src/muffin_agent/agents/trading_decision/nodes.py](../src/muffin_agent/agents/trading_decision/nodes.py) |
| Graph builders | [src/muffin_agent/agents/trading_decision/graph.py](../src/muffin_agent/agents/trading_decision/graph.py) |
| Prompts | [src/muffin_agent/prompts/trading_decision/](../src/muffin_agent/prompts/trading_decision/) |
| CLI | [src/muffin_cli/main.py](../src/muffin_cli/main.py) (`decide` command) |
| Tests | [tests/agents/test_trading_decision/](../tests/agents/test_trading_decision/) (193 tests) + [tests/cli/test_decide_helpers.py](../tests/cli/test_decide_helpers.py) |
