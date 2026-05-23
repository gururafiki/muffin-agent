# Trading Decision Pipeline

A composable trading-decision pipeline ported from [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) and refactored to LangGraph-native primitives. Decoupled from `investment_analysis`: every node accepts a generic `AnalysisContext`, so callers can plug in muffin's pipeline output, free-form research notes, or any custom upstream source.

This guide covers what the pipeline does, how the architecture is shaped, the composition patterns, and the future migration paths for when a role needs to grow beyond a single LLM call.

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

Three composable builders share `TradingDecisionState` and let callers opt into depth:

| Builder | Topology |
|---|---|
| `build_investment_debate_graph` | Bull/Bear → Judge (smallest useful slice) |
| `build_investment_thesis_graph` | …+ Trader (adds operational translation) |
| `build_trading_decision_graph` | full pipeline above (canonical 5-tier decision + reflection bookends) |

---

## Node architecture (LangGraph-native, no agent wrappers)

Each per-role file ships **one async node function** with typed input/output state schemas. The function body resolves an LLM, renders a Jinja template, calls `llm.ainvoke(...)`, and returns a state-update dict. There are no `MuffinAgentBuilder` factories, no per-call agent rebuilds, and no `Command(goto=...)` — routing lives at the graph level via conditional edges.

```python
# researchers/bull_researcher.py — shape every per-role file follows

class BullResearcherInputState(TypedDict, total=False):
    """State keys this node reads."""
    analysis_context: dict[str, Any]
    investment_bull_responses: Annotated[list[str], operator.add]
    investment_bear_responses: Annotated[list[str], operator.add]


class BullResearcherOutputState(TypedDict, total=False):
    """State keys this node writes."""
    investment_bull_responses: Annotated[list[str], operator.add]


async def bull_researcher_node(
    state: BullResearcherInputState, config: RunnableConfig
) -> BullResearcherOutputState:
    # State reads — explicit at the call site.
    analysis_context = state["analysis_context"]
    bulls = state.get("investment_bull_responses") or []
    bears = state.get("investment_bear_responses") or []
    opposing_last = bears[-1] if bears else ""

    # LLM resolution — muffin's existing ModelConfiguration pattern.
    cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = cfg.get_llm_for_role("reasoner")
    llm = (primary.with_fallbacks(fallbacks) if fallbacks else primary).with_retry(
        stop_after_attempt=3, wait_exponential_jitter=True
    )

    # One Jinja template per role — composes shared partials via {% include %}.
    prompt = render_template(
        "trading_decision/researchers/bull.jinja",
        ticker=analysis_context.get("ticker", ""),
        query=analysis_context.get("query"),
        debate_history=format_debate_history(bulls, bears),
        opposing_last=opposing_last,
        # ... all analysis_context fields visible at call site ...
    )

    # No try/except — failures propagate to the graph (and through retries).
    response = await llm.ainvoke([
        SystemMessage(prompt),
        HumanMessage("Make your argument now."),
    ])
    return {"investment_bull_responses": [str(response.content).strip()]}
```

### State uses LangGraph reducers (no sub-state structs)

`TradingDecisionState` is **flat**. Debate responses live in top-level `Annotated[list[str], operator.add]` fields per speaker (`investment_bull_responses`, `investment_bear_responses`, `risk_aggressive_responses`, etc.). Each node returns `{"<field>": [new_response]}` and LangGraph's reducer appends to the accumulated list. The "latest" response is `responses[-1]`; the "round count" is `len(bulls) + len(bears)`. Structured outputs (`investment_judge`, `trader`, `portfolio_decision`) live in their own top-level dict fields populated via `Pydantic.model_dump()`.

### Routing lives at the graph level

`graph.py` defines `_route_investment_debate(state) -> str` and `_route_risk_debate(state) -> str` that read accumulated response lists and the per-run `TradingDecisionConfiguration` (via `langgraph.config.get_config()`). Conditional edges use the **list form** (`["bull_researcher", "bear_researcher", "investment_judge"]`) which sidesteps the `dict[Hashable, str]` mypy variance complaint that forced the inline-dict workaround in the old code.

### Two-layer retry, no fallback dicts

Every LLM call gets two retry layers and no try/except:

1. **LangChain `with_retry(stop_after_attempt=3, wait_exponential_jitter=True)`** wraps each `llm.ainvoke` — catches transient provider errors.
2. **LangGraph `RetryPolicy(max_attempts=2)`** is applied per-node via `graph.add_node("name", node_fn, retry_policy=...)` — catches anything that escapes the LLM layer.

After both layers exhaust, the exception propagates and the graph fails. That's the correct caller signal. A deterministic failure (e.g. a bad pending reflection entry that always errors) is the user's cue to investigate or clear the offending entry from the store.

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

Mirrors muffin's existing `MemoryConfiguration` / `McpConfiguration` / `ResearchConfiguration` pattern. Read via `TradingDecisionConfiguration.from_runnable_config(config)` inside router functions and the reflection nodes.

---

## Reflection memory loop

The reflection layer turns the pipeline into a learning loop:

1. **Write (end of every run)** — `decision_writeback_node` persists the current `PortfolioDecisionOutput` as a `pending` record under namespace `("memories", user_id, "decisions")` with key `f"{TICKER}:{YYYY-MM-DD}"`.
2. **Resolve (start of every run)** — `reflector_resolve_node` walks all pending entries (any ticker), calls `fetch_outcomes_openbb` for realised returns + alpha vs benchmark (default `SPY`), and calls `reflect_on_decision` (single LLM call) to produce a 2–4 sentence reflection. Resolved records persist in the same store.
3. **Inject (same start)** — The most-recent `reflection_max_same_ticker` (5) + `reflection_max_cross_ticker` (3) resolved reflections are rendered as a Markdown block and injected into the Portfolio Manager prompt. The PM sets `incorporates_past_lessons=true` when it actually cites them.

The reflection layer degrades silently when:
- the store is `None`,
- no `user_id` is resolvable from `configurable.user_id` or `MEMORY_DEBUG_USER_ID`,
- `reflection_enabled` is `False`,
- `fetch_outcomes_openbb` returns `None` for a pending entry (then it stays pending; next run will retry).

These are operational unavailabilities, not LLM failures. The reflector LLM call itself runs *without* try/except — if it fails after retries, the graph fails (consistent with the no-try-except rule for LLM nodes).

---

## Composition patterns

`AnalysisContext` is the abstraction that decouples this pipeline from any specific upstream analysis.

### Pattern 1: Ad-hoc notes

```python
from langgraph.store.memory import InMemoryStore

from muffin_agent.agents.trading_decision import (
    AnalysisContext,
    build_trading_decision_graph,
)

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

```bash
muffin decide AAPL --analysis-json state.json --query "long-term hold"

# Combine structured analysis with free-form notes
muffin decide AAPL --analysis-json state.json \
                   --narrative "Add'l context: management transition"
```

```python
from muffin_agent.agents.trading_decision import AnalysisContext

state = {                  # produced by build_investment_analysis_graph
    "ticker": "AAPL",
    "query": "long-term hold",
    "market_regime": {...},
    "sector_view": {...},
    # ...other 4 structured outputs...
}

context = AnalysisContext.from_investment_analysis_state(
    state,
    narrative="Additional context that didn't fit the structured fields.",
)
```

### Pattern 3: External / custom upstream

Any caller with the right structured field keys can use `from_investment_analysis_state`; any caller without structured fields can fall back to `from_narrative`. The adapter doesn't care where the dict came from.

```python
ctx = AnalysisContext(
    ticker="AAPL",
    query="momentum + quality",
    valuation={"valuation_signal": "fairly_valued", "pe_ttm": 22.1},
    risk_assessment={"var_95_1m_pct": 6.4, "beta": 1.18},
    narrative="Quant screen flagged this ticker.",
    additional_context={"screener_score": 0.81, "factor": "quality"},
)
```

---

## Per-role files — independently importable

Every node function is independently importable, lets external graphs satisfy the input TypedDict, and writes to the output TypedDict. The graph that owns the next-node routing decides what happens after.

```python
from muffin_agent.agents.trading_decision import (
    bull_researcher_node,
    bear_researcher_node,
    investment_judge_node,
    trader_node,
    aggressive_debator_node,
    conservative_debator_node,
    neutral_debator_node,
    portfolio_manager_node,
    reflect_on_decision,  # pure async helper, not a node
    reflector_resolve_node,
    decision_writeback_node,
)
```

The corresponding TypedDicts are also exported (`BullResearcherInputState`, `BullResearcherOutputState`, etc.) so external callers can type-narrow their parent state.

---

## Future migration paths

The current single-LLM-call shape is right for trading_decision's reasoning-only roles. When a role grows beyond a single LLM call there are **two documented promotion paths**. Pick using the rubric below.

| Question | Path 1 (custom subgraph) | Path 2 (MuffinAgentBuilder agent) | Stay with current pattern |
|---|---|---|---|
| Tools? | Yes — needs `ToolNode` | Yes — uses `with_tool` | No |
| Multiple LLM calls per turn (e.g. self-critique)? | Yes | No (single ReAct loop) | No |
| Wants `ToolKnowledgeMiddleware` / `ToolResultCacheMiddleware`? | Add manually | Yes (free) | N/A |
| Custom internal routing / state? | Yes | No | No |
| Default for a new role in trading_decision? | | | **Yes** |

### Path 1 — Compiled subgraph with `ToolNode`

When a role needs tools AND/OR has its own internal state machine (retry loop, self-critique pass, internal CoT decomposition).

```python
# Hypothetical: trader.py after tools arrive
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode


class TraderInternalState(TypedDict, total=False):
    messages: list[BaseMessage]
    structured_response: TraderOutput | None


def _trader_llm_step(state, config):
    llm = (
        ModelConfiguration.from_runnable_config(config)
        .get_llm_for_role("reasoner")[0]
        .bind_tools([compute_position_sizing])
        .with_structured_output(TraderOutput)
    )
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def _route_trader(state) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


@lru_cache(maxsize=1)
def build_trader_subgraph() -> CompiledStateGraph:
    g = StateGraph(TraderInternalState)
    g.add_node("llm", _trader_llm_step)
    g.add_node("tools", ToolNode([compute_position_sizing]))
    g.add_edge(START, "llm")
    g.add_conditional_edges("llm", _route_trader, {"tools": "tools", END: END})
    g.add_edge("tools", "llm")
    return g.compile()


async def trader_node(state, config):
    """Parent-graph adapter: stages messages, invokes the subgraph, unpacks output."""
    subgraph = build_trader_subgraph()
    sub_state = {"messages": [HumanMessage(_build_trader_context(state))]}
    result = await subgraph.ainvoke(sub_state, config)
    structured = result["structured_response"]
    return {"trader": structured.model_dump()}
```

The subgraph's internal state is private to the role. Tools, retries, branching all stay encapsulated.

### Path 2 — `MuffinAgentBuilder` agent as graph node

When a role needs muffin's middleware stack (tool-knowledge lessons, cross-agent caching, etc.) but doesn't need a custom topology — i.e. a standard ReAct loop is enough.

```python
# Hypothetical: trader.py after we want tool-knowledge + caching
@lru_cache(maxsize=1)
def build_trader_agent() -> CompiledStateGraph:
    cfg = ModelConfiguration.from_runnable_config({"configurable": {}})
    primary, *fallbacks = cfg.get_llm_for_role("reasoner")
    summariser = cfg.get_summariser()
    builder = (
        MuffinAgentBuilder(primary, name="trader")
        .with_system_prompt_template("trading_decision/trader.jinja")
        .with_fallback_models(*fallbacks)
        .with_tool(compute_position_sizing)
        .with_response_format(AutoStrategy(TraderOutput))
    )
    if summariser is not None:
        builder = builder.with_tool_knowledge(summariser)
    return builder.build_react_agent()


# Parent graph adds the compiled agent as a node:
graph.add_node("trader", build_trader_agent())
```

The agent's state schema (`AgentState` with `messages` + `structured_response`) must be a subset of (or compatible via state mapping with) the parent graph's state.

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
| State (flat, with reducers) | [src/muffin_agent/agents/trading_decision/state.py](../src/muffin_agent/agents/trading_decision/state.py) |
| Configuration | [src/muffin_agent/agents/trading_decision/config.py](../src/muffin_agent/agents/trading_decision/config.py) |
| Debate formatters | [src/muffin_agent/agents/trading_decision/_debate.py](../src/muffin_agent/agents/trading_decision/_debate.py) |
| Researchers | [src/muffin_agent/agents/trading_decision/researchers/](../src/muffin_agent/agents/trading_decision/researchers/) |
| Trader | [src/muffin_agent/agents/trading_decision/trader.py](../src/muffin_agent/agents/trading_decision/trader.py) |
| Risk debaters | [src/muffin_agent/agents/trading_decision/risk_debate/](../src/muffin_agent/agents/trading_decision/risk_debate/) |
| Portfolio Manager | [src/muffin_agent/agents/trading_decision/portfolio_manager.py](../src/muffin_agent/agents/trading_decision/portfolio_manager.py) |
| Reflection memory | [src/muffin_agent/agents/trading_decision/reflection/](../src/muffin_agent/agents/trading_decision/reflection/) |
| Graph builders + routers | [src/muffin_agent/agents/trading_decision/graph.py](../src/muffin_agent/agents/trading_decision/graph.py) |
| Prompts (templates + 3 shared partials) | [src/muffin_agent/prompts/trading_decision/](../src/muffin_agent/prompts/trading_decision/) |
| CLI | [src/muffin_cli/main.py](../src/muffin_cli/main.py) (`decide` command) |
| Tests | [tests/agents/test_trading_decision/](../tests/agents/test_trading_decision/) (~140 tests) + [tests/cli/test_decide_helpers.py](../tests/cli/test_decide_helpers.py) |
