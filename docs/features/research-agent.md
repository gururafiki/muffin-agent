# Research Agent

**Domain-agnostic Perplexity-style deep research agent.**

Lives at [`src/muffin_agent/agents/research/`](../../src/muffin_agent/agents/research/). Inspired by [Vane (Perplexica)](https://github.com/ItzCrazyKns/Vane).

The MVP supersedes the original draft FRD; that file is preserved at [`docs/features/_archive/research-agent-frd.md`](_archive/research-agent-frd.md) for historical context.

## Why

Two driving use cases:

1. **Standalone**: `muffin research "..."` for ad-hoc deep research with cited answers.
2. **Subagent**: callable from other agents (investment, criteria-analysis, custom workflows) to gather web evidence without each agent reimplementing search/scrape/cite.

Pluggability is a core requirement: different callers will hand it different tool sets (web by default; web + academic for science; web + finance for investment; web + internal docs for product research).

## Architecture

```
START
  │
  ▼
classifier   ← LLM call (collector role, structured output)
  │
  ├── skip_search=True ─→ writer ─→ END
  └── default ──────────→ researcher ─→ rerank ─→ writer ─→ END
```

| Node | Type | Role | Output |
|------|------|------|--------|
| `classifier` | ReAct agent (no tools) | collector | `ResearchClassification` — `standalone_query` (coref-resolved), `task_type`, `mode_hint`, `sources_to_use`, `skip_search`, `rationale`. |
| `researcher` | Deep agent (tools + skills) | orchestrator | `ResearchEvidenceFindings.evidence_chunks` — list of `EvidenceChunk` (title/url/snippet/content/source_type). |
| `rerank` | Pure Python (no LLM) | — | `reranked_evidence` — embedding cosine ≥ threshold, URL-deduped, top-K. |
| `writer` | ReAct agent (no tools) | orchestrator | `ResearchOutput` — markdown answer with inline `[N]`, key findings, sources, confidence, missing info, follow-ups. |

**Why a deep agent for the researcher (and not a sub-graph)?** Less code, universal middleware applies, skills + filesystem work natively, the LLM is good at deciding when it has enough evidence. The migration trigger to swap to a sub-graph is: (a) we want to stream a research-trail UI panel, OR (b) LangSmith traces show the LLM mis-sequencing search/scrape decisions. The state contract (`state["query"] → state["evidence"]`) is the only public surface, so the swap is non-breaking.

## Components

### State — `state.py`

`ResearchState(AgentState)` carries everything through the pipeline:

- Input: `query`, `chat_history`, `allowed_sources`, `mode_override`, `task_type_override`, `system_instructions`.
- Lifted by classifier: `standalone_query`, `task_type`, `mode`, `sources_to_use`, `skip_search`, `classification` (full dict).
- Accumulator: `evidence: Annotated[list[dict], operator.add]` (researcher writes once).
- Outputs: `reranked_evidence`, `output`.

`ResearchClassificationFilterState(AgentState)` is the minimal schema fed to `SkillFilterMiddleware[…]` — only `mode` + `task_type`, the two filtering dimensions.

### Schemas — `schemas.py`

- `ResearchClassification` — classifier output.
- `EvidenceChunk` — `{title, url, snippet, content, source_type, retrieved_at, relevance}`.
- `ResearchEvidenceFindings` — `{evidence_chunks, notes}`.
- `Source` — `{n, title, url}` (citation slot).
- `ResearchOutput` — the public contract.

### Configuration — `config.py`

`ResearchConfiguration(BaseConfiguration)` exposes:

| Field | Env var | Default | Purpose |
|-------|---------|---------|---------|
| `embedding_model` | `EMBEDDING_MODEL` | `text-embedding-3-small` | Rerank embedding model. |
| `embedding_base_url` | `EMBEDDING_BASE_URL` | `None` | OpenAI-compatible override (OpenRouter, vLLM, LM Studio, Ollama). |
| `embedding_api_key` | `EMBEDDING_API_KEY` | `None` | Falls back to `OPENAI_API_KEY`. |
| `rerank_threshold` | `RERANK_THRESHOLD` | `0.5` | Cosine cutoff (Vane). |
| `rerank_top_k` | `RERANK_TOP_K` | `20` | Max chunks after rerank. |
| `research_default_mode` | `RESEARCH_DEFAULT_MODE` | `balanced` | speed / balanced / quality. |
| `research_default_sources` | `RESEARCH_DEFAULT_SOURCES` | `["web"]` | csv-parsed. |
| `max_search_results` | `MAX_SEARCH_RESULTS` | `8` | Per `firecrawl_search` call. |
| `research_iter_speed` | `RESEARCH_ITER_SPEED` | `2` | Researcher LLM-call budget (speed). |
| `research_iter_balanced` | `RESEARCH_ITER_BALANCED` | `6` | (balanced). |
| `research_iter_quality` | `RESEARCH_ITER_QUALITY` | `25` | (quality). |

Firecrawl base URL is **not** duplicated here — `McpConfiguration.firecrawl_mcp_url` covers it.

### Tools

The researcher loads two tools from the existing Firecrawl MCP via `get_tools(config, ["firecrawl_search", "firecrawl_scrape"])`. No new HTTP wrappers; no `httpx` dep added.

Caller-supplied `extra_tools` are appended via `with_tool(tool, is_cacheable=True)`.

### Skills — `/skills/research/`

10 SKILL.md files at MVP. `SkillFilterMiddleware[ResearchClassificationFilterState]` filters by `mode` + `task_type` (lifted flat keys); universal skills (no metadata) always match.

```
src/muffin_agent/skills/research/
├── modes/
│   ├── speed/SKILL.md       # metadata: { mode: speed }
│   ├── balanced/SKILL.md    # metadata: { mode: balanced }
│   └── quality/SKILL.md     # metadata: { mode: quality }
├── task_types/
│   ├── research_report/SKILL.md
│   ├── comparison/SKILL.md
│   ├── how_to/SKILL.md
│   ├── summary/SKILL.md
│   ├── debate/SKILL.md
│   └── factual_qa/SKILL.md
└── _shared/
    └── citation_discipline/SKILL.md  # universal — always applies
```

Add a skill: drop a SKILL.md with matching `metadata` under `/skills/research/`. The researcher's `SkillsMiddleware` auto-discovers it on next run.

### Embeddings — `embeddings.py`

`compute_evidence_relevance` embeds query + each chunk via `langchain_openai.OpenAIEmbeddings(model=..., base_url=..., api_key=SecretStr(...))`. Provider flexibility:

- **OpenAI direct**: unset env vars; `OPENAI_API_KEY` is read automatically.
- **OpenRouter** (free testing model): `EMBEDDING_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2:free` + `EMBEDDING_BASE_URL=https://openrouter.ai/api/v1`. `EMBEDDING_API_KEY` is optional — falls back to `OPENAI_API_KEY` env (which OpenRouter users already populate with their OpenRouter key per the existing convention).
- **Local OpenAI-compatible servers** (vLLM, LM Studio, Ollama): same shape — set `EMBEDDING_BASE_URL` to the local endpoint.

Dedup pass: highest-relevance chunk per URL wins; content from the duplicate URL is merged (longer body kept).

## Public entrypoints

```python
from muffin_agent.agents.research import (
    build_research_graph,        # standalone graph
    build_research_subagent,     # CompiledSubAgent factory
    ResearchConfiguration,
    ResearchOutput,
)
```

### Standalone

```python
g = build_research_graph(
    checkpointer=SqliteSaver(...),       # optional
    store=InMemoryStore(),                # optional
    extra_tools=[arxiv_search],           # optional
    extra_sources=["academic"],           # optional
)
result = await g.ainvoke(
    {"query": "..."},
    config={"configurable": {"user_id": "alice"}},
)
ResearchOutput.model_validate(result["output"])
```

Module-level `graph = build_research_graph()` is registered in [`langgraph.json`](../../langgraph.json) as `research` for Platform autodiscovery.

### As a subagent

```python
from muffin_agent.agents.research import build_research_subagent
from muffin_agent.utils.agent_builder import MuffinAgentBuilder

research = await build_research_subagent(
    config,
    extra_tools=[arxiv_search, news_search],
    extra_sources=["academic", "news"],
)

agent = (
    MuffinAgentBuilder(model)
    .with_system_prompt("...")
    .with_subagents([research, ...other subagents])
    .build_deep_agent()
)
```

The parent agent's `task` tool routes research-flavoured questions to `deep-research`. The subagent returns `ResearchOutput` as its structured response.

## CLI

```
muffin research "QUERY" [--mode {speed,balanced,quality}]
                       [--sources web,academic,...]
                       [--task-type {research_report,comparison,...}]
                       [--user USER]
                       [--thread THREAD]
```

Examples:

```bash
# Default — balanced mode, web only
muffin research "Latest news on Anthropic Claude 4.7"

# Quality mode for deep coverage
muffin research "Postgres vs MySQL for OLTP" --mode quality

# skip_search path — no external lookup needed
muffin research "What is 2+2?"

# Explicit task type override
muffin research "How do I set up pgvector?" --task-type how_to --mode quality
```

The CLI writes a SQLite checkpointer to `~/.muffin/checkpoints.db` (shared with other muffin commands).

`--user` populates `configurable.user_id` for `/memories/` namespacing.

## Pluggability — adding new sources

The factory contract: `build_research_subagent(config, *, extra_tools=[BaseTool, ...], extra_sources=["academic", "news", ...])`. Source-gating is **prompt-driven** (Vane parity) — the classifier emits `sources_to_use`, and the researcher prompt tells the LLM which tools are in/out of scope.

### Academic source

1. New tool implementation: plain `@tool`-decorated function calling Semantic Scholar / arXiv.
2. Caller passes `extra_tools=[academic_search]` + `extra_sources=["academic"]`.
3. Optional `/skills/research/sources/academic/SKILL.md` describing when to call `academic_search` + credibility heuristics.

### Finance source

Caller passes existing finance MCP tools (or data-collection subagents wrapped as tools) via `extra_tools=[...]` + `extra_sources=["finance"]`. No change in `agents/research/`.

### News, discussions, internal docs

Same shape — wrap the source as a `BaseTool`, register via `extra_tools` + `extra_sources`.

## Why no `with_subagent_refinement()`?

The refinement protocol (`CollectionFindings`, `prior_call_id`, `/scratch/subagent_runs/`) is designed for orchestrators that re-issue partial gap-filling calls. Research's contract is the **complete** `ResearchOutput`, not a partial finding. Follow-ups are handled by re-invocation with a new query + the SQLite checkpointer's thread history.

## Migration path: deep agent → sub-graph

The researcher is currently a single `create_deep_agent` call (~30 LOC). The migration trigger:

- We want to **stream a research-trail UI panel** (Vane-style block streaming), OR
- LangSmith traces consistently show the LLM **mis-sequencing search/scrape decisions**.

The swap: replace `researcher_node` internals with a multi-node sub-graph (plan → search → scrape → decide_continue → consolidate). The state contract (`state["query"] → state["evidence"]`) is the only public surface, so `classifier_node`, `rerank_node`, `writer_node`, the CompiledSubAgent wrapper, and all callers are unaffected.

## Roadmap

See [roadmap.md → Research Agent — follow-ups](../../roadmap.md): academic / news / finance source tools, LangSmith eval pipeline, Supabase pgvector for persistent vector cache (also serves as PostgreSQL host for multi-user checkpoints), fact-checking verifier node, source credibility scoring, streaming progress UI, multi-modal evidence, discussion source tools, Wolfram/calculator integration, conversation-aware suggestion generation.
