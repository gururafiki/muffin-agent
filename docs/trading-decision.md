# Trading Decision Pipeline

A composable trading-decision pipeline ported from [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) and refactored to LangGraph-native primitives. **Fully self-contained** — fetches its own data via OpenBB MCP (price / fundamentals / news / ownership) and Firecrawl MCP (social / web) through four ReAct analyst agents that run before the Bull/Bear/Judge/Trader/PM downstream nodes. Does NOT consume outputs from any other muffin pipeline.

This guide covers what the pipeline produces, the architecture, how the compiled analyst agents are wired in directly as parent-graph nodes, the reflection loop, the CLI surface, and the future migration paths for when a role needs to grow further.

For the per-file architecture reference, see the relevant entries in [CLAUDE.md](../CLAUDE.md).

---

## What the pipeline produces

`PortfolioDecisionOutput` is the canonical artifact:

```python
class PortfolioDecisionOutput(BaseModel):
    rating: Literal["strong_sell", "sell", "hold", "buy", "strong_buy"]
    executive_summary: str          # 2–4 sentence headline
    investment_thesis: str          # detailed reasoning
    price_target: float | None
    stop_loss: float | None
    time_horizon: str               # e.g. "3–6 months"
    position_sizing: str            # e.g. "2% NAV starter, scale to 4% on Q1 beat"
    key_risks_remaining: list[str]
    confidence: float               # 0.0–1.0
    incorporates_past_lessons: bool # set true when the PM cited reflection memory
```

The same 5-tier vocabulary (`strong_sell` … `strong_buy`) is shared with `CriteriaAnalysisSynthesis.signal` so both pipelines speak one rating language.

---

## Pipeline shape

```
START
  ↓
[reflector_resolve]            ← resolve prior pending decisions + inject past reflections
  ↓
[Market]      [Fundamentals]   [News]        [Social]   ← 4 analyst ReAct agents in parallel
  └────┬──────────┬────────────┬──────────────┘
       ▼ (implicit barrier — bull_researcher fires when all 4 reports are in state)
[Bull Researcher] ⇄ [Bear Researcher]    ← N rounds (default 2)
  ↓
[Investment Judge]             ← InvestmentJudgeOutput (signal + bull/bear cases + catalysts/risks)
  ↓
[Trader]                       ← TraderOutput (action + entry/stop/take-profit + sizing + horizon)
  ↓
[Aggressive] → [Conservative] → [Neutral]    ← M rounds (default 1)
  ↓
[Portfolio Manager]            ← PortfolioDecisionOutput (canonical final artifact)
  ↓
[decision_writeback]           ← persist current decision as pending for future reflection
  ↓
END
```

Three composable async builders share `TradingDecisionState` and let callers opt into depth:

| Builder | Topology |
|---|---|
| `build_investment_debate_graph` | analysts → Bull/Bear → Judge (smallest useful slice) |
| `build_investment_thesis_graph` | …+ Trader (adds operational translation) |
| `build_trading_decision_graph` | full pipeline above (canonical 5-tier decision + reflection bookends) |

All three are **async** — each starts by building the four compiled analyst agents (so the agent construction cost is amortised to graph-build time, not per-call).

---

## The analyst layer (compiled agents added directly as parent-graph nodes)

The four analysts replace what used to be a hard dependency on muffin's `agents/investment/` outputs. Each analyst is a **compiled ReAct agent** (from `langchain.agents.create_agent`) added directly to the parent graph — no wrapping subgraph, no per-analyst prepare/extract nodes, no per-analyst middleware in the analyst file.

This works because each analyst declares its own state schema extending `langchain.agents.AgentState`, with `OmitFromSchema` annotations marking which extra fields are input-only vs output-only:

```python
# analysts/market_analyst.py — shape every analyst follows
from typing import Annotated

from langchain.agents import AgentState
from langchain.agents.middleware.types import OmitFromSchema
from pydantic import BaseModel, Field


class MarketAnalystOutput(BaseModel):
    """Field names match the state-schema output fields so the
    structured response is auto-unpacked into parent state by name."""

    market_report: str = Field(description="Markdown technical-analysis report …")


class MarketAnalystState(AgentState):
    ticker: Annotated[str, OmitFromSchema(input=False, output=True)]
    decision_date: Annotated[str, OmitFromSchema(input=False, output=True)]
    market_report: Annotated[str, OmitFromSchema(input=True, output=False)]


async def build_market_analyst_agent(config: RunnableConfig) -> CompiledStateGraph:
    cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = cfg.get_llm_for_role("collector")
    mcp_tools = await get_tools(config, [
        "equity_price_historical", "equity_price_quote",
        "equity_price_performance", "equity_historical_market_cap",
    ])

    builder = (
        MuffinAgentBuilder(primary, name="market_analyst")
        .with_fallback_models(*fallbacks)
        .with_state_schema(MarketAnalystState)
        .with_runtime_system_prompt_template("trading_decision/analysts/market.jinja")
        .with_response_format(MarketAnalystOutput)  # auto-unpack to state by name
    )
    for tool in mcp_tools:
        builder = builder.with_tool(tool, run_limit=4)
    builder = builder.with_tool(get_indicators, run_limit=10)
    return builder.build_react_agent()
```

### Three `MuffinAgentBuilder` features that make this possible

1. **`with_state_schema(schema)`** forwards a custom state schema (extending `AgentState`) to `create_agent`. Combined with `OmitFromSchema(input=…, output=…)` annotations on schema fields, this auto-derives the agent's `input_schema` and `output_schema` so LangGraph maps shared keys (by name) between parent state and the compiled agent.
2. **`with_runtime_system_prompt_template(template)`** registers a built-in middleware that re-renders the system prompt from agent state on every model call. When `with_state_schema(…)` is also set, the middleware introspects the schema's `OmitFromSchema` annotations and passes ONLY the input-eligible fields as Jinja variables (skipping reserved fields like `messages` / `structured_response`). Templates use `{% if x %}` for missing values.
3. **Auto-trigger structured-response → state unpacking** when both `with_state_schema(...)` AND `with_response_format(...)` are set — the builder wires a built-in middleware that unpacks the Pydantic response into per-field state updates (each Pydantic field maps to a same-named state field). Scales naturally to N output fields and means the parent reads them directly without cracking a `structured_response` dict.

The compiled agent is added to the parent graph with its derived input schema for self-documentation:

```python
# graph.py
await _add_analyst_nodes(graph, config)
# expands to:
market_agent = await build_market_analyst_agent(config)
graph.add_node(
    "market_analyst",
    market_agent,
    input_schema=market_agent.input_schema,  # makes the contract visible at the parent
    retry_policy=_LLM_RETRY,
)
# ...same for fundamentals_analyst, news_analyst, social_analyst...

# parallel fan-out + implicit barrier
for analyst in ("market_analyst", "fundamentals_analyst",
                "news_analyst", "social_analyst"):
    graph.add_edge("reflector_resolve", analyst)
    graph.add_edge(analyst, "bull_researcher")
```

### Tools by analyst

| Analyst | OpenBB MCP tools | Extra |
|---|---|---|
| Market | `equity_price_historical`, `equity_price_quote`, `equity_price_performance`, `equity_historical_market_cap` | `get_indicators` (local — stockstats over OpenBB OHLCV; fills the technical-indicator gap left by OpenBB MCP) |
| Fundamentals | `equity_fundamental_balance`, `equity_fundamental_income`, `equity_fundamental_cash`, `equity_fundamental_ratios`, `equity_fundamental_metrics`, `equity_fundamental_historical_eps`, `equity_fundamental_dividends` | — |
| News | `news_company`, `news_world`, `equity_ownership_insider_trading` | — |
| Social | `news_company`, `firecrawl_search` | — |

Make sure `docker compose up -d openbb-mcp firecrawl-mcp searxng` is running before invoking the pipeline.

---

## Downstream-node architecture (LangGraph-native, no agent wrappers)

The non-analyst nodes (Bull, Bear, Judge, Trader, 3 risk debators, PM, reflection bookends) are each **one async node function** with typed input/output state schemas. The function body resolves an LLM, renders a Jinja template, calls `llm.ainvoke(...)`, and returns a state-update dict. No `MuffinAgentBuilder` factories, no per-call agent rebuilds, and no `Command(goto=...)` — routing lives at the graph level via conditional edges.

```python
# researchers/bull_researcher.py — shape every downstream node follows

class BullResearcherInputState(TypedDict, total=False):
    """State keys this node reads."""
    ticker: str
    query: str
    narrative: str
    market_report: str
    fundamentals_report: str
    news_report: str
    sentiment_report: str
    investment_bull_responses: Annotated[list[str], operator.add]
    investment_bear_responses: Annotated[list[str], operator.add]


class BullResearcherOutputState(TypedDict, total=False):
    """State keys this node writes."""
    investment_bull_responses: Annotated[list[str], operator.add]


async def bull_researcher_node(
    state: BullResearcherInputState, config: RunnableConfig
) -> BullResearcherOutputState:
    bulls = state.get("investment_bull_responses") or []
    bears = state.get("investment_bear_responses") or []
    opposing_last = bears[-1] if bears else ""

    cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = cfg.get_llm_for_role("reasoner")
    llm = (primary.with_fallbacks(fallbacks) if fallbacks else primary).with_retry(
        stop_after_attempt=3, wait_exponential_jitter=True
    )

    prompt = render_template(
        "trading_decision/researchers/bull.jinja",
        ticker=state.get("ticker", ""),
        query=state.get("query"),
        narrative=state.get("narrative"),
        market_report=state.get("market_report"),
        fundamentals_report=state.get("fundamentals_report"),
        news_report=state.get("news_report"),
        sentiment_report=state.get("sentiment_report"),
        debate_history=format_debate_history(bulls, bears),
        opposing_last=opposing_last,
    )

    response = await llm.ainvoke([
        SystemMessage(prompt),
        HumanMessage("Make your argument now."),
    ])
    return {"investment_bull_responses": [str(response.content).strip()]}
```

### State uses LangGraph reducers (no sub-state structs)

`TradingDecisionState` is **flat**. Inputs (`ticker`, `decision_date`, `query`, `narrative`) and analyst outputs (`market_report`, `fundamentals_report`, `news_report`, `sentiment_report`) are top-level fields. Debate responses live in `Annotated[list[str], operator.add]` reducers per speaker (`investment_bull_responses`, `risk_aggressive_responses`, etc.). Structured outputs (`investment_judge`, `trader`, `portfolio_decision`) live in their own top-level dict fields populated via `Pydantic.model_dump()`.

### Routing lives at the graph level

`graph.py` defines `_route_investment_debate(state) -> str` and `_route_risk_debate(state) -> str` that read accumulated response lists and the per-run `TradingDecisionConfiguration` (via `langgraph.config.get_config()`). Conditional edges use the **list form** (`["bull_researcher", "bear_researcher", "investment_judge"]`) — sidesteps the `dict[Hashable, str]` mypy variance complaint.

### Two-layer retry, no fallback dicts

Every LLM call gets two retry layers and no try/except:

1. **LangChain `with_retry(stop_after_attempt=3, wait_exponential_jitter=True)`** wraps each `llm.ainvoke` — catches transient provider errors.
2. **LangGraph `RetryPolicy(max_attempts=2)`** is applied per-node via `graph.add_node("name", node_fn, retry_policy=...)` — catches anything that escapes the LLM layer.

After both layers exhaust, the exception propagates and the graph fails — the correct caller signal.

### Typed per-run knobs via `TradingDecisionConfiguration`

```python
class TradingDecisionConfiguration(BaseConfiguration):
    max_investment_debate_rounds: int = 2
    max_risk_debate_rounds: int = 1
    reflection_enabled: bool = True
    reflection_holding_days: int = 5
    reflection_benchmark: str = "SPY"
    reflection_max_same_ticker: int = 5
    reflection_max_cross_ticker: int = 3
    decision_date: str | None = None
```

Mirrors muffin's existing `MemoryConfiguration` / `McpConfiguration` / `ResearchConfiguration` pattern.

---

## Reflection memory loop

The reflection layer turns the pipeline into a learning loop:

1. **Write (end of every run)** — `decision_writeback_node` persists the current `PortfolioDecisionOutput` as a `pending` record under namespace `("memories", user_id, "decisions")` with key `f"{TICKER}:{YYYY-MM-DD}"`.
2. **Resolve (start of every run)** — `reflector_resolve_node` walks all pending entries (any ticker), calls `fetch_outcomes_openbb` for realised returns + alpha vs benchmark (default `SPY`), and calls `reflect_on_decision` (single LLM call) to produce a 2–4 sentence reflection.
3. **Inject (same start)** — The most-recent `reflection_max_same_ticker` (5) + `reflection_max_cross_ticker` (3) resolved reflections are rendered as a Markdown block and injected into the Portfolio Manager prompt. The PM sets `incorporates_past_lessons=true` when it actually cites them.

The reflection layer degrades silently when: the store is `None`, no `user_id` is resolvable, `reflection_enabled` is `False`, or `fetch_outcomes_openbb` returns `None`. These are operational unavailabilities. The reflector LLM call itself runs *without* try/except — if it fails after retries, the graph fails.

---

## CLI

```bash
# Minimal — the four analysts fetch their own data.
muffin decide AAPL

# With user framing
muffin decide AAPL --query "long-term hold candidate"

# Layer caller-provided notes alongside the analyst reports
muffin decide AAPL --narrative "Recent earnings call mentioned X..."

# Pin a decision date (deterministic testing + reflection-memory bookkeeping)
muffin decide AAPL --decision-date 2026-05-23 --invest-rounds 1 --risk-rounds 1
```

Required: OpenBB MCP and Firecrawl MCP running (`docker compose up -d openbb-mcp firecrawl-mcp searxng`).

The CLI uses an in-process `InMemoryStore` — reflection memory persists only within a single Python process. For cross-session persistence, wire a `PostgresStore` (see [LangGraph store docs](https://langchain-ai.github.io/langgraph/reference/store/)) or run on LangGraph Platform (it injects a managed store automatically).

---

## Per-role files — independently importable

Every node function is independently importable, lets external graphs satisfy the input TypedDict, and writes to the output TypedDict. The graph that owns the next-node routing decides what happens after.

```python
from muffin_agent.agents.trading_decision import (
    # Analyst factories (async, return compiled ReAct agents)
    build_market_analyst_agent,
    build_fundamentals_analyst_agent,
    build_news_analyst_agent,
    build_social_analyst_agent,
    # Downstream nodes
    bull_researcher_node,
    bear_researcher_node,
    investment_judge_node,
    trader_node,
    aggressive_debator_node,
    conservative_debator_node,
    neutral_debator_node,
    portfolio_manager_node,
    # Reflection
    reflect_on_decision,
    reflector_resolve_node,
    decision_writeback_node,
    # Local tool
    get_indicators,
)
```

The corresponding TypedDicts are also exported (`BullResearcherInputState`, `BullResearcherOutputState`, `MarketAnalystState`, `MarketAnalystOutput`, etc.) so external callers can type-narrow their parent state.

---

## Future migration paths

The current shape — analysts as compiled ReAct agents, downstream nodes as single-LLM-call functions — is the right default. When a role grows beyond that there are **two documented promotion paths**.

| Question | Path 1 (custom subgraph) | Path 2 (MuffinAgentBuilder agent) | Stay with current pattern |
|---|---|---|---|
| Tools? | Yes — needs `ToolNode` | Yes — uses `with_tool` | Analysts already do |
| Multiple LLM calls per turn (e.g. self-critique)? | Yes | No (single ReAct loop) | No |
| Wants `ToolKnowledgeMiddleware` / `ToolResultCacheMiddleware`? | Add manually | Yes (free) | N/A |
| Custom internal routing / state? | Yes | No | No |

The analyst layer already uses Path 2 internally — each analyst IS a compiled `MuffinAgentBuilder.build_react_agent()` added directly to the parent graph. The patterns below are for downstream nodes (Bull, Bear, etc.) if they ever need tools / multi-step structure.

### Path 1 — Compiled subgraph with `ToolNode`

When a role needs tools AND has its own internal state machine.

```python
# Hypothetical: trader.py after tools arrive
@lru_cache(maxsize=1)
def build_trader_subgraph() -> CompiledStateGraph:
    g = StateGraph(TraderInternalState)
    g.add_node("llm", _trader_llm_step)
    g.add_node("tools", ToolNode([compute_position_sizing]))
    g.add_edge(START, "llm")
    g.add_conditional_edges("llm", _route_trader, {"tools": "tools", END: END})
    g.add_edge("tools", "llm")
    return g.compile()
```

### Path 2 — `MuffinAgentBuilder` agent as graph node (analyst pattern)

The analyst factories already follow this pattern. Extend the same shape to a downstream node when it needs muffin's middleware stack (tool-knowledge lessons, cross-agent caching, etc.) AND can be expressed as a standard ReAct loop:

```python
# Hypothetical: trader.py promoted to a tools-enabled ReAct agent
async def build_trader_agent(config: RunnableConfig) -> CompiledStateGraph:
    cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = cfg.get_llm_for_role("reasoner")

    class TraderAgentState(AgentState):
        ticker: Annotated[str, OmitFromSchema(input=False, output=True)]
        investment_judge: Annotated[dict, OmitFromSchema(input=False, output=True)]
        trader: Annotated[dict, OmitFromSchema(input=True, output=False)]

    builder = (
        MuffinAgentBuilder(primary, name="trader")
        .with_fallback_models(*fallbacks)
        .with_state_schema(TraderAgentState)
        .with_runtime_system_prompt_template("trading_decision/trader.jinja")
        .with_response_format(TraderOutput)  # auto-unpacks to state.trader by name
        .with_tool(compute_position_sizing)
    )
    return builder.build_react_agent()

# Parent graph:
graph.add_node("trader", await build_trader_agent(config),
               input_schema=trader_agent.input_schema)
```

---

## Deviations from upstream TradingAgents

The port is substantially faithful to [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) and re-platforms several pieces onto more modern LangGraph primitives. Intentional deviations:

- **5-tier rating vocabulary**: `strong_buy / buy / hold / sell / strong_sell` (port) vs `Buy / Overweight / Hold / Underweight / Sell` (upstream). Port aligns with [`CriteriaAnalysisSynthesis.signal`](../src/muffin_agent/agents/criteria_analysis/schemas.py) so both pipelines speak one rating language.
- **`max_investment_debate_rounds=2`** by default (vs upstream `1`). Yields four bull/bear turns instead of two — more thorough debate at 2× the LLM cost. Override per-run via `--invest-rounds` or `configurable.max_investment_debate_rounds`.
- **Reflection memory is `BaseStore`-backed** (`("memories", user_id, "decisions")` namespace, `f"{TICKER}:{YYYY-MM-DD}"` key) rather than the upstream append-only markdown file at `~/.tradingagents/memory/trading_memory.md`. No built-in `memory_log_max_entries` rotation knob — retention is delegated to the store layer (`PostgresStore`, LangGraph Platform's managed store, etc.).
- **Reflection resolution is global, not per-ticker**: `reflector_resolve_node` walks ALL pending entries across tickers each run. Upstream only resolved same-ticker pending entries; the port resolves cross-ticker pending entries as well so cross-ticker lessons accumulate with real outcome figures.
- **No `output_language` switch** — English-only. Upstream supported per-run language selection for analyst/PM outputs while keeping internal debate in English. Roadmap item.
- **No JSON state-log / markdown report file saving** in the CLI. Observability is via LangSmith / LangFuse traces (see `setup_tracing` in [src/muffin_agent/utils/observability.py](../src/muffin_agent/utils/observability.py)) instead of post-run file writes.
- **No "FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**" coordination marker**. Upstream used it as a heuristic stop-signal in analyst ReAct loops; the port replaces that with structured-output schemas + per-tool `run_limit` governance, which makes the marker unnecessary.
- **`recursion_limit=100`** is set on the `muffin decide` CLI to match upstream — the full pipeline (4 analyst ReAct loops + 4-turn investment debate + 3-turn risk debate + judge / trader / PM / reflector bookends) easily exceeds LangGraph's default of 25 steps. Callers wiring `build_trading_decision_graph` from outside the CLI should set `recursion_limit` themselves on the `RunnableConfig`.

### Look-ahead-bias caveat for backtesting

Upstream enforces date capping at the dataflow layer (`load_ohlcv` filters rows `<= curr_date`, `filter_financials_by_date` strips future fiscal columns). The port pushes `decision_date` into prompts and trusts the analyst LLM to pass it as the `end_date` / `curr_date` argument on OpenBB tool calls. The local `get_indicators` tool is safe — it enforces `curr_date` server-side and only fetches OHLCV up to that date. The OpenBB MCP tools (`equity_price_*`, `equity_fundamental_*`, `news_*`, `equity_ownership_*`) do **not** have a hard server-side cap. For rigorous historical backtesting, audit the tool-call trace in LangFuse / LangSmith to confirm `end_date` / `start_date` / `as_of_date` arguments are bounded by `decision_date`.

---

## Where to look in the code

| Concern | File |
|---|---|
| Schemas (Judge / Trader / PM outputs + reflection records) | [src/muffin_agent/agents/trading_decision/schemas.py](../src/muffin_agent/agents/trading_decision/schemas.py) |
| State (flat, with reducers) | [src/muffin_agent/agents/trading_decision/state.py](../src/muffin_agent/agents/trading_decision/state.py) |
| Configuration | [src/muffin_agent/agents/trading_decision/config.py](../src/muffin_agent/agents/trading_decision/config.py) |
| Debate formatters | [src/muffin_agent/agents/trading_decision/_debate.py](../src/muffin_agent/agents/trading_decision/_debate.py) |
| Analysts (4 compiled ReAct agents) | [src/muffin_agent/agents/trading_decision/analysts/](../src/muffin_agent/agents/trading_decision/analysts/) |
| Local tool (`get_indicators`) | [src/muffin_agent/agents/trading_decision/tools.py](../src/muffin_agent/agents/trading_decision/tools.py) |
| Researchers | [src/muffin_agent/agents/trading_decision/researchers/](../src/muffin_agent/agents/trading_decision/researchers/) |
| Trader | [src/muffin_agent/agents/trading_decision/trader.py](../src/muffin_agent/agents/trading_decision/trader.py) |
| Risk debaters | [src/muffin_agent/agents/trading_decision/risk_debate/](../src/muffin_agent/agents/trading_decision/risk_debate/) |
| Portfolio Manager | [src/muffin_agent/agents/trading_decision/portfolio_manager.py](../src/muffin_agent/agents/trading_decision/portfolio_manager.py) |
| Reflection memory | [src/muffin_agent/agents/trading_decision/reflection/](../src/muffin_agent/agents/trading_decision/reflection/) |
| Graph builders + routers (async) | [src/muffin_agent/agents/trading_decision/graph.py](../src/muffin_agent/agents/trading_decision/graph.py) |
| Prompts | [src/muffin_agent/prompts/trading_decision/](../src/muffin_agent/prompts/trading_decision/) (shared `_instrument_context.jinja` partial is included by every agent prompt to preserve exchange suffixes like `.TO` / `.L` / `.HK`) |
| CLI | [src/muffin_cli/main.py](../src/muffin_cli/main.py) (`decide` command) |
| MuffinAgentBuilder extensions | [src/muffin_agent/utils/agent_builder.py](../src/muffin_agent/utils/agent_builder.py) (`with_state_schema`, `with_runtime_system_prompt_template`, plus auto-unpack on `with_response_format` + `with_state_schema`) |
| Tests | [tests/agents/test_trading_decision/](../tests/agents/test_trading_decision/) + [tests/utils/test_agent_builder_state_aware.py](../tests/utils/test_agent_builder_state_aware.py) |
