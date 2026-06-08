# Persona Council

A council of 13 well-known investor personas evaluates a ticker through their distinct lenses and produces a synthesised verdict. Each persona is a self-contained compiled subgraph that owns its own data collection + deterministic scoring + a single-LLM-call verdict, emitting an `AnalystSignal` with a 5-tier rating (`strong_sell` / `sell` / `hold` / `buy` / `strong_buy`).

## At a glance

| Slug | Display | Lens |
|---|---|---|
| `warren_buffett` | Warren Buffett | Quality compounders, 3-stage owner-earnings DCF, ≥20% MOS |
| `ben_graham` | Benjamin Graham | NCAV / Graham Number, ≥50% margin of safety |
| `cathie_wood` | Cathie Wood | Disruptive innovation, R&D intensity, high-growth DCF (20%/15%/25×) |
| `charlie_munger` | Charlie Munger | Mental models, ROIC consistency, predictability (0.35/0.25/0.25/0.15) |
| `bill_ackman` | Bill Ackman | Concentrated activist, business quality + identifiable catalyst |
| `michael_burry` | Michael Burry | Contrarian deep value, FCF yield ≥15%, EV/EBIT <6 |
| `mohnish_pabrai` | Mohnish Pabrai | Dhandho: 0.45 downside + 0.35 valuation + 0.20 double |
| `nassim_taleb` | Nassim Taleb | Antifragility, convexity, via negativa, tail-risk |
| `peter_lynch` | Peter Lynch | GARP, PEG, ten-bagger hunting |
| `phil_fisher` | Phil Fisher | Qualitative growth, R&D 3-15%, scuttlebutt |
| `rakesh_jhunjhunwala` | Rakesh Jhunjhunwala | EM growth, quality-tier DCF (12/15/18%), 30% MoS |
| `stanley_druckenmiller` | Stanley Druckenmiller | Macro + momentum, asymmetric R/R |
| `aswath_damodaran` | Aswath Damodaran | Academic FCFF DCF + CAPM, story-then-numbers |

## CLI

```bash
# Run a single persona
muffin persona warren_buffett AAPL

# Run the full council + LLM-mediated synthesis
muffin council AAPL

# Council with a custom investment mandate
muffin council MSFT -q "Long-only quality bias, 5-year horizon"

# Council including the 6 specialists (technicals / sentiment / fundamentals /
# growth / valuation / news_sentiment)
muffin council AAPL --include-specialists
```

Outputs are JSON — see `AnalystSignal` / `CouncilSynthesisOutput` schemas for the exact shape.

## Architecture

```
START
  │  (13 edges — one per persona, all run in parallel)
  ▼
┌────────────────────────────────────────────────────────────────────┐
│  Each persona is a compiled subgraph with 3 nodes:                  │
│                                                                     │
│    collect_data  →  compute_evidence  →  render_verdict             │
│    (ReAct,         (deterministic        (single LLM call           │
│     MCP fetch)      Python scoring        with response_format =    │
│                     via                   <Persona>Signal)          │
│                     scoring_helpers)                                │
└────────────────────────────────────────────────────────────────────┘
  │  (fan-in barrier: persona_signals accumulated via operator.add)
  ▼
council_judge             ← LLM-mediated synthesis
  ↓
END
```

There is **no shared front-of-flow data-collection step** — each persona owns its data fetching via a curated set of OpenBB MCP tools. The `ToolResultCacheMiddleware` (and the matching [`cached_invoke`](../src/muffin_agent/middlewares/tool_result_cache/cache.py) helper used by the deterministic specialists) shares cache hits across all 13 personas, so an MCP call from `warren_buffett`'s `collect_data` ReAct loop is reused by `ben_graham`'s loop later in the same council run.

## Persona internals

Each persona file (e.g. [`warren_buffett.py`](../src/muffin_agent/agents/personas_council/personas/warren_buffett.py)) ships ~600-1000 LOC:

1. **Typed sub-evidence Pydantics** — replace the generic `ScoreDetail`. For Buffett these are `WarrenBuffettFundamentals`, `WarrenBuffettMoat`, `WarrenBuffettPricingPower`, etc. — every score / value / reasoning string the verdict prompt needs is a typed field.
2. **`<Persona>Evidence`** — the full evidence Pydantic combining all sub-evidence + DCF outputs + margin of safety.
3. **`<Persona>Signal(AnalystSignal)`** — narrows `evidence` to the typed model so the council judge sees a self-describing structure.
4. **`<Persona>RawData`** — the structured response of the `collect_data` ReAct sub-agent. Field descriptions teach the LLM exactly which OpenBB MCP endpoint each value comes from.
5. **`<Persona>State(AgentState)`** — internal subgraph state; `OmitFromSchema` annotations control what flows in/out at the council boundary.
6. **`<Persona>Input` / `<Persona>Output`** — explicit `TypedDict` boundaries passed to `StateGraph(state, input_schema=..., output_schema=...)`. The council only sees `ticker` / `as_of_date` / `query` on the input side and `persona_signals` on the output side.
7. **Composite scorers** — `_score_<persona>_<aspect>(state) -> <SubEvidence>` private Python functions. Use atomic helpers from [`agents/personas_council/tools/scoring_helpers.py`](../src/muffin_agent/agents/personas_council/tools/scoring_helpers.py) (`score_roe`, `compute_owner_earnings`, `compute_buffett_3stage_dcf`, etc.) plus persona-specific aggregation logic.
8. **`compute_evidence_node(state)`** — deterministic Python node that wires all composite scorers together. Never crosses an LLM boundary.
9. **`render_verdict_node(state, config)`** — single LLM call via `ModelConfiguration.get_chat_model_for_role(config, "reasoner", schema=<Persona>Signal)`. The Jinja template receives the typed evidence Pydantic instance directly (not `.model_dump()`) so the prompt uses granular dotted-attribute access + conditional rule blocks.
10. **`_build_data_collection_agent(config)`** — compiled ReAct sub-agent wired with the persona's curated MCP tool list, `response_format=<Persona>RawData`. The `_StructuredResponseToStateMiddleware` auto-unpacks the RawData fields into the subgraph state.
11. **`build_<persona>_agent(config)`** — async factory that wires the 3 nodes into a `StateGraph(<Persona>State, input_schema=<Persona>Input, output_schema=<Persona>Output)`. Used by the council + `muffin persona <slug>` CLI.

## Council judge

The judge ([`judge.py`](../src/muffin_agent/agents/personas_council/judge.py)) consumes the 13 signals and produces a [`CouncilSynthesisOutput`](../src/muffin_agent/agents/personas_council/judge.py):

```python
class CouncilSynthesisOutput(BaseModel):
    ticker: str
    consensus_rating: InvestmentSignal       # 5-tier
    weighted_confidence: float
    vote_breakdown: dict[str, list[str]]     # rating → [persona slugs]
    bull_case_synthesis: str
    bear_case_synthesis: str
    dissent_summary: str
    key_uncertainties: list[str]
    reasoning: str
```

The judge **may override the numerical majority** when a small number of personas have very high-confidence dissent grounded in specific data the majority overlooked. See [`council_judge.jinja`](../src/muffin_agent/prompts/personas/council_judge.jinja) for the explicit override criteria.

## Valuation methodology (ai-hedge-fund parity)

The DCF / intrinsic-value layer is kept faithful to the upstream
[ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) agents — growth
assumptions are **derived from each company's own history**, not hardcoded:

| Persona | Growth | Terminal value | Discount |
|---|---|---|---|
| Warren Buffett | 3-stage: stage-1 from historical NI CAGR (clamp −5%..+15%, ×0.7, cap 8%), stage-2 = min(stage1×0.5, 4%) | Gordon Growth 2.5% on year-10 owner earnings, ×0.85 haircut | 10% |
| Bill Ackman | 6% | **15× exit multiple** on year-5 FCF | 10% |
| Rakesh Jhunjhunwala | tiered from NI CAGR (>25%→20%; >15%→×0.8; >5%→×0.9; else 5%) | quality-based **exit multiple** 18/15/12× | quality-based 12/15/18% |
| Cathie Wood | 20% | 25× exit multiple | 15% |
| Aswath Damodaran | 5-yr revenue CAGR (cap 12%, fade to 2.5%) | Gordon Growth anchored on **base FCFF** (`terminal_basis="base_fcff"`) | CAPM `rf + β×ERP` |

Owner earnings (Buffett) = `NI + D&A − maintenance capex (median of 0.85×capex / depreciation / hist-ratio×revenue) − ΔWC`.

**Note on Damodaran's terminal value:** upstream anchors the perpetuity on the
*un-grown* base FCFF, which understates the terminal value relative to a textbook
DCF. The port reproduces this for parity (`compute_damodaran_fcff_dcf(...,
terminal_basis="base_fcff")`); pass `terminal_basis="final_cf"` for the
conventional (higher) intrinsic value. See
[`agents/personas_council/tools/scoring_helpers.py`](../src/muffin_agent/agents/personas_council/tools/scoring_helpers.py).

**Deferred:** Nassim Taleb's upstream `analyze_black_swan_sentinel` (negative-news
ratio + volume-spike crisis signal) is not ported — it needs news collection added
to the Taleb data step; the volume/price-dislocation half is implicitly available
to the verdict LLM via the tail-risk + vol-regime evidence.

## Pluggability

```python
from muffin_agent.agents.personas_council.council_graph import PERSONA_BUILDERS

# PERSONA_BUILDERS is a list of (slug, async_builder) pairs — that's the
# whole "registry". Adding a persona = adding one entry here + writing
# the per-persona file + verdict + data-collection prompts.
for slug, builder in PERSONA_BUILDERS:
    agent = await builder(config)
    parent_graph.add_node(slug, agent, input_schema=agent.input_schema)
    parent_graph.add_edge(START, slug)
    parent_graph.add_edge(slug, "my_aggregator_node")
```

## Specialists

Six specialists ([`agents/personas_council/specialists/`](../src/muffin_agent/agents/personas_council/specialists/)) emit the same `AnalystSignal` contract (5-tier rating). All scoring is **deterministic** — the four metric-heavy ones use an LLM only to *extract* OpenBB fields into a typed `RawData`, never to judge. They mirror ai-hedge-fund's six specialist agents.

Fully deterministic (sync, no-arg builders, `cached_invoke` fetch → compute):

* **`technicals`** — 5-strategy ensemble (trend / mean-reversion / momentum / volatility-regime / stat-arb) over 1-year OHLCV. 2-node subgraph: `fetch_ohlcv` → `compute_technical_signal`.
* **`sentiment`** — 30/70 weighted insider + benzinga news sentiment aggregation. 3-node parallel `fetch_insider_trades` + `fetch_company_news` → `compute_sentiment_signal`.

Persona-style (async builders taking `config`, ReAct `collect_data` extraction → deterministic compute):

* **`fundamentals`** — 4-dimension majority vote (profitability / growth / financial-health / price-ratios). Logic in [`tools/fundamentals.py`](../src/muffin_agent/agents/personas_council/tools/fundamentals.py).
* **`growth`** — 5 weighted sub-scores (growth-trend 0.40 / valuation 0.25 / margins 0.15 / insider 0.10 / health 0.10). [`tools/growth.py`](../src/muffin_agent/agents/personas_council/tools/growth.py).
* **`valuation`** — weighted intrinsic-value gap from four methods (WACC-DCF scenarios 0.35 / owner earnings 0.35 / EV-EBITDA 0.20 / residual income 0.10). [`agents/personas_council/tools/valuation_signal.py`](../src/muffin_agent/agents/personas_council/tools/valuation_signal.py).
* **`news_sentiment`** — the one LLM specialist: a ReAct step fetches recent headlines and classifies each (positive/negative/neutral + confidence); aggregation is deterministic (`0.7 × avg matching-headline confidence + 0.3 × proportion`).

Enable them in the council via `build_council_graph(config, include_specialists=True)` or invoke standalone via `muffin technicals|sentiment|growth|valuation|news-sentiment <TICKER>` (the fundamentals signal is `muffin fundamentals-signal <TICKER>` — `muffin fundamentals` is the separate raw data-collection command). Their MCP fetches share the same cache as the persona ReAct loops thanks to `cached_invoke` / `ToolResultCacheMiddleware` matching the namespace + hash scheme.

## Cache sharing

`MuffinAgentBuilder.with_tool(...)` defaults to `is_cacheable=True`, which wires the `ToolResultCacheMiddleware`. Personas inherit this automatically. The specialists' deterministic `cached_invoke` helper writes to the same `("cache", tool_name)` namespace with the same `get_args_hash(args)` key, so a single OpenBB call for e.g. `equity_fundamental_metrics(AAPL, annual, 5)` from the first persona that runs is reused by the remaining 12 personas (and by any subsequent specialist) in the same council run.

See [`cache.py`](../src/muffin_agent/middlewares/tool_result_cache/cache.py) for the `cache_lookup` / `cache_store` / `cached_invoke` primitives and the parity tests in [`tests/middlewares/tool_result_cache/test_cache.py`](../tests/middlewares/tool_result_cache/test_cache.py).
