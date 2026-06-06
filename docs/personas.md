# Persona Council

A council of 13 well-known investor personas evaluates a ticker through their distinct lenses and produces a synthesised verdict.  Each persona is a single-LLM-call node that consumes precomputed deterministic facts (ported from [ai-hedge-fund](https://github.com/virattt/ai-hedge-fund)) and emits an `AnalystSignal` with a 5-tier rating (`strong_sell` / `sell` / `hold` / `buy` / `strong_buy`).

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
```

Outputs are JSON — see `AnalystSignal` / `CouncilSynthesisOutput` schemas for the exact shape.

## Architecture

```
START
  ↓
[persona_data_collection]   ← one deep-agent run gathers PersonaDataBundle
  ↓
[Send fan-out × 13 personas]
  │
  ▼
┌────────────────────────────────────────────────────────────┐
│  Each persona is a single LLM call:                         │
│    1. Compute facts via tools/scoring_helpers.py            │
│    2. Render prompt extending personas/_persona_base.jinja  │
│    3. LLM → <Persona>Signal (5-tier rating + evidence)      │
└────────────────────────────────────────────────────────────┘
  │
  │  (fan-in: persona_signals via operator.add reducer)
  ▼
[council_judge]             ← LLM-mediated synthesis
  ↓
END
```

The data collection step (`persona_data_collection_node`) runs **once** and produces a [`PersonaDataBundle`](../src/muffin_agent/agents/personas/data.py) with 28 line items + ~16 financial metrics + market cap + 5y market-cap history + 1y insider trades + 1y company news + 1y OHLCV.  All 13 personas read the same bundle — no per-persona MCP overhead.

## Council judge

The judge ([`judge.py`](../src/muffin_agent/agents/personas/judge.py)) consumes the 13 signals and produces a [`CouncilSynthesisOutput`](../src/muffin_agent/agents/personas/judge.py):

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

The judge **may override the numerical majority** when a small number of personas have very high-confidence dissent grounded in specific data the majority overlooked.  See [`council_judge.jinja`](../src/muffin_agent/prompts/personas/council_judge.jinja) for the explicit override criteria.

## Persona node template

Each persona file (e.g. [`warren_buffett.py`](../src/muffin_agent/agents/personas/warren_buffett.py)) is ~150-250 LOC with five sections:

1. **Evidence model** — `<Persona>Evidence(BaseModel)` with typed sub-scores
2. **Signal model** — `<Persona>Signal(AnalystSignal)` with `agent_id: Literal[<slug>]`
3. **Fact computer** — `_compute_<slug>_facts(data_bundle) -> <Persona>Evidence` (pure-Python deterministic scoring via `tools/scoring_helpers.py`)
4. **Node** — `<slug>_node(state, config) -> {persona_signals: [...]}` (single LLM call)
5. **Registry entry** — `PERSONA_SPEC = register_persona(PersonaSpec(...))`

Adding a new persona is a 150-line file + a Jinja prompt + an import in [`personas/__init__.py`](../src/muffin_agent/agents/personas/__init__.py).

## Pluggability

The `PERSONA_REGISTRY` dict exposes every persona by slug:

```python
from muffin_agent.agents.personas import PERSONA_REGISTRY
from langgraph.types import Send

def fanout_personas(state):
    return [
        Send(spec.slug, {
            "ticker": state["ticker"],
            "data_bundle": state["data_bundle"],
        })
        for spec in PERSONA_REGISTRY.values()
    ]

# Wire into any external LangGraph
graph.add_conditional_edges("my_node", fanout_personas, list(PERSONA_REGISTRY))
```

This is how the [paper-trading pipeline](paper-trading.md) reuses the council per ticker without re-implementing the data-collection step.
