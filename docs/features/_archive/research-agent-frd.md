# Research Agent Feature Requirements Document (Simplified)

**Version**: 2.0  
**Date**: April 2026  
**Status**: Feature Specification for Implementation  
**Framework**: LangChain DeepAgents + LangGraph (native approach)  
**Target Architecture**: General-Purpose Research Agent (inspired by Vane/Perplexica)

---

## Executive Summary

This document specifies a self-hosted, privacy-first research agent system modeled on Vane/Perplexica's architecture.

The document presents **two implementation paths**: a simplified pure-DeepAgent approach and a hybrid approach using subagents/subgraphs for complex research workflows.

---

## 1. Architecture Overview

### 1.1 Option A: Pure DeepAgent (Simplified, Recommended for MVP)

```
User Query + Configuration (Mode, Sources, Task Profile)
         ↓
    create_deep_agent()
    ├─ Planner reasoning loop (built-in)
    ├─ Tool/Skill/Subagent calls
    ├─ Store + StateBackend memory
    └─ response_format for output schema
         ↓
    Final Answer + Citations
```

**Advantages**:
- Minimal boilerplate; leverages DeepAgents' built-in planning
- Self-contained; no custom graph wiring
- Automatic backtracking + adaptive planning
- Memory management handled by StateBackend/StoreBackend

**Limitations**:
- Less explicit control over research stages (classify → search → rerank → render)
- Harder to enforce strict stop conditions per mode

### 1.2 Option B: Hybrid (LangGraph + DeepAgents)

```
Custom LangGraph (Classifier → Planner → Executor → Reranker → Evaluator → Renderer)
         ↓
    Each stage triggers create_deep_agent() subagents or tools
         ↓
    Evidence Store (LangGraph Store + Vector Store)
         ↓
    Final Answer + Citations
```

**Advantages**:
- Explicit control over research pipeline
- Mode-specific stopping criteria (Speed/Balanced/Quality)
- Multi-stage composition with clear evidence flow
- Detailed tracing + debugging

**Limitations**:
- More complex implementation
- Requires manual state threading between stages

---

## 2. Configuration Model: Mode × Sources × Task Profile

### 2.1 Reasoning Modes (Controlled via Prompt Injection)

```python
from enum import Enum
from pydantic import BaseModel

class ResearchMode(str, Enum):
    SPEED = "speed"
    BALANCED = "balanced"
    QUALITY = "quality"

class ResearchConfig(BaseModel):
    mode: ResearchMode
    sources: list[str]  # ["web", "academic", "discussions", ...]
    task_profile: str    # "research_report", "thesis", "comparison", etc.
```

**Modes encoded in system prompt**:

```python
MODE_INSTRUCTIONS = {
    "speed": """
    Gather information quickly with 1-2 sources max.
    Prioritize most obvious answers.
    Minimize planning; answer directly.
    Time budget: <5s.
    """,
    "balanced": """
    Gather information from 3-4 reliable sources.
    Use reasoning to connect findings.
    Plan before executing.
    Time budget: 15-30s.
    """,
    "quality": """
    Exhaustively research from multiple sources (6+).
    Iteratively refine hypothesis.
    Check for contradictions and gaps.
    Synthesize comprehensive analysis.
    Time budget: 60-120s.
    """,
}
```

### 2.2 Data Sources (Composable)

```python
ENABLED_SOURCES = {
    "web": "SearXNG meta-search",
    "academic": "ArXiv, academic databases",
    "discussions": "Reddit, HN, forums",
    "news": "News aggregators, RSS",
    "financial": "SEC filings, market data APIs",
    "internal": "User uploaded docs, knowledge base",
}
```

Each source maps to a **tool** or **subagent** in the agent's toolkit.

### 2.3 Task Profiles (Pydantic Models for Output Schema)

```python
from pydantic import BaseModel, Field
from typing import Literal

class ResearchReportOutput(BaseModel):
    """Structure for research report task profile"""
    executive_summary: str = Field(description="High-level overview")
    key_findings: list[str] = Field(description="Main discoveries")
    sections: list[dict] = Field(description="Detailed sections with content")
    citations: dict[str, str] = Field(description="Claim -> source URL mapping")
    confidence: float = Field(description="0-1 confidence score")

class ThesisOutput(BaseModel):
    """Structure for thesis task profile"""
    thesis_statement: str
    supporting_evidence: list[str]
    counter_arguments: list[str]
    synthesis: str
    citations: dict[str, str]

class ComparisonOutput(BaseModel):
    """Structure for comparison task profile"""
    entities: list[str]
    dimensions: list[str]  # e.g., ["cost", "performance", "reliability"]
    comparison_table: dict  # entity -> {dimension -> value}
    summary: str
    citations: dict[str, str]
```

---

## 3. Implementation Path: Pure DeepAgent (Recommended)

### 3.1 Tool Definitions (Native LangChain Tools)

```python
from langchain.tools import tool
from langchain_community.utilities import SearxSearchWrapper
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
from datetime import datetime

@tool
async def search_web(query: str, num_results: int = 5) -> list[dict]:
    """
    Search the web via SearXNG.
    Returns: [{title, url, content, snippet}, ...]
    """
    searx = SearxSearchWrapper(searx_host="http://localhost:8888")
    results = await searx.aresults(query, num_results=num_results)
    
    evidence_list = []
    for result in results:
        try:
            # Scrape via Firecrawl
            scrape_result = requests.post(
                "http://localhost:3000/v1/scrape",
                json={
                    "url": result["url"],
                    "formats": ["markdown"],
                },
                timeout=10,
            )
            
            if scrape_result.status_code == 200:
                content = scrape_result.json().get("data", {}).get("markdown", "")
            else:
                content = result.get("snippet", "")
            
            evidence_list.append({
                "title": result.get("title"),
                "url": result["url"],
                "content": content[:2000],  # Truncate for context
                "snippet": result.get("snippet"),
                "source_type": "web",
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            # Fallback to snippet
            evidence_list.append({
                "title": result.get("title"),
                "url": result["url"],
                "content": result.get("snippet", ""),
                "source_type": "web",
                "timestamp": datetime.now().isoformat(),
            })
    
    return evidence_list

@tool
async def extract_from_document(document_content: str, query: str) -> dict:
    """
    Extract key facts from a document relevant to the query.
    Returns: {key_facts, entities, dates, claims}
    """
    from langchain.chat_models import init_chat_model
    
    model = init_chat_model("gpt-4-mini")
    
    prompt = f"""
    Analyze this document and extract facts relevant to: {query}
    
    Return JSON with:
    - key_facts: list of facts
    - entities: list of entities mentioned
    - dates: list of important dates
    - claims: list of claims/statements
    
    Document (first 3000 chars):
    {document_content[:3000]}
    """
    
    response = await model.ainvoke([{"role": "user", "content": prompt}])
    
    # Parse JSON response
    import json
    try:
        result = json.loads(response.content)
    except:
        result = {"key_facts": [response.content], "entities": [], "dates": [], "claims": []}
    
    return result

@tool
async def fetch_financial_data(entity: str, metric: str, period: str = "latest") -> dict:
    """
    Fetch financial data for an entity.
    Integrates with your existing data_collection agents.
    """
    from your_module import financial_data_collection_agent
    
    result = await financial_data_collection_agent.ainvoke({
        "entity": entity,
        "metric": metric,
        "period": period,
    })
    
    return result
```

### 3.2 Subagents for Specialized Tasks (Optional)

**Instead of tools, use subagents for complex workflows:**

```python
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model

# Subagent 1: Academic research specialist
academic_researcher = create_deep_agent(
    model=init_chat_model("gpt-4-mini"),
    tools=[search_web],  # Filtered to academic sources
    system_prompt="""
    You are an academic research specialist.
    Search for peer-reviewed papers, articles, and academic sources.
    Focus on credibility and citations.
    Summarize findings with full attribution.
    """,
)

# Subagent 2: Financial analyst
financial_analyst = create_deep_agent(
    model=init_chat_model("gpt-4-mini"),
    tools=[fetch_financial_data, search_web],
    system_prompt="""
    You are a financial analyst.
    Gather financial metrics, market data, and analyst sentiment.
    Compute ratios and trends.
    Highlight risks and opportunities.
    """,
)

# Wrap subagents as tools in the langgraph
@tool
async def research_academics(query: str) -> str:
    """Ask academic researcher specialist"""
    result = await academic_researcher.ainvoke({
        "messages": [{"role": "user", "content": query}],
    })
    return result["messages"][-1].content

@tool
async def analyze_financials(query: str) -> str:
    """Ask financial analyst specialist"""
    result = await financial_analyst.ainvoke({
        "messages": [{"role": "user", "content": query}],
    })
    return result["messages"][-1].content

# Or add as subagents to deepagents
```

### 3.3 Main Research Agent with Structured Output

```python
from pydantic import BaseModel, Field
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver

# Define output schemas
class Citation(BaseModel):
    claim: str = Field(description="The claim being cited")
    source_url: str = Field(description="Source URL")
    confidence: float = Field(description="0-1 confidence in the claim")

class ResearchOutput(BaseModel):
    summary: str = Field(description="Executive summary")
    key_findings: list[str] = Field(description="Main findings")
    detailed_analysis: str = Field(description="Comprehensive analysis with inline citations")
    citations: list[Citation] = Field(description="List of citations")
    confidence_score: float = Field(description="Overall confidence 0-1")
    missing_information: list[str] = Field(description="What we couldn't find")

# Create main research agent
research_agent = create_deep_agent(
    model=init_chat_model("gpt-4"),  # Use gpt-4 for native structured output
    tools=[
        search_web,
        extract_from_document,
        fetch_financial_data,
        research_academics,
        analyze_financials,
    ],
    system_prompt="""
    You are a comprehensive research agent.
    
    Your task:
    1. Understand the user's research question
    2. Search for relevant information using available tools
    3. Extract key facts and synthesize findings
    4. Identify gaps and contradictions
    5. Provide a comprehensive, well-cited answer
    
    Always:
    - Cite sources for all claims
    - Note confidence levels
    - Point out missing information
    - Consider multiple perspectives
    - Cross-check contradictions
    """,
    
    # Output schema for structured response
    response_format=ProviderStrategy(schema=ResearchOutput),
    
    # Memory management
    checkpointer=MemorySaver(),
    store=InMemoryStore(),
)

# Usage - TODO: consider using skills for specific mode and source isntructions instead of custom building of content.
async def conduct_research(query: str, mode: str = "balanced", sources: list[str] = None):
    """
    Conduct research on a query.
    
    Args:
        query: Research question
        mode: "speed", "balanced", or "quality"
        sources: List of enabled sources
    """
    
    mode_instruction = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["balanced"])
    sources_instruction = f"Use only these sources: {', '.join(sources or ['web'])}"
    
    result = await research_agent.ainvoke({
        "messages": [
            {
                "role": "user",
                "content": f"""
                {mode_instruction}
                {sources_instruction}
                
                Research question: {query}
                """
            }
        ]
    })
    
    # Extract structured response
    structured_output = result.get("structured_response")
    return structured_output
```

---

## 4. Alternative Implementation: Hybrid with Custom LangGraph

### 4.1 When to Use Hybrid

Use hybrid approach if you need:
- Strict mode-based control (e.g., Speed mode must stop after 1 search)
- Multi-stage pipeline with evidence aggregation
- Complex reasoning with explicit reranking
- Monitoring/debugging of each stage

### 4.2 LangGraph State with Store Backend

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.store.postgres import PostgresStore  # Or use InMemoryStore
from langgraph.checkpoint.memory import MemorySaver

class ResearchState(TypedDict):
    """Shared state for research pipeline"""
    query: str
    mode: str
    sources: list[str]
    task_profile: str
    
    # Evidence accumulation (stored in LangGraph Store)
    evidence_ids: Annotated[list, "Evidence pointers in Store"]
    
    # Planning
    plan: str
    plan_executed: int
    
    # Stopping
    should_continue: bool
    stop_reason: str
    
    # Output
    final_answer: dict  # Pydantic model instance
```

### 4.3 Using Store for Persistent Evidence

```python
from langgraph.store.postgres import PostgresStore

# Initialize persistent store
store = PostgresStore(
    conn_string="postgresql://user:password@localhost/research_db",
)

async def store_evidence(store, evidence_id: str, evidence: dict):
    """Store evidence in persistent store"""
    await store.aput(
        ("evidence",),  # Namespace
        evidence_id,    # Key
        evidence,       # Value (dict with content, url, etc.)
    )

async def retrieve_evidence(store, evidence_id: str) -> dict:
    """Retrieve evidence from store"""
    items = await store.aget(("evidence",), evidence_id)
    if items:
        return items[0].value
    return None

async def search_evidence_by_query(store, query: str, namespace_prefix: str = "evidence"):
    """
    Retrieve all evidence matching query.
    Note: LangGraph Store doesn't have semantic search by default;
    use vector store for embedding-based search.
    """
    items = await store.alist(namespace=("evidence",))
    return [item.value for item in items]
```

### 4.4 Vector Store for Semantic Search

```python
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings

# Initialize Supabase vector store
embeddings = OpenAIEmbeddings()
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase_client,
    table_name="research_evidence",
    query_name="match_evidence",
)

async def add_evidence_to_vector_store(evidence_list: list[dict]):
    """Add evidence to vector store for semantic search"""
    from langchain.schema import Document
    
    docs = [
        Document(
            page_content=e["content"],
            metadata={
                "url": e["url"],
                "source_type": e["source_type"],
                "timestamp": e["timestamp"],
                "title": e.get("title"),
            }
        )
        for e in evidence_list
    ]
    
    await vector_store.aadd_documents(docs)

async def semantic_search_evidence(query: str, k: int = 5):
    """Search evidence by semantic similarity"""
    results = await vector_store.asimilarity_search(query, k=k)
    return [
        {
            "content": doc.page_content,
            "url": doc.metadata["url"],
            "source_type": doc.metadata["source_type"],
            "similarity": doc.metadata.get("score", 0.0),
        }
        for doc in results
    ]
```

### 4.5 Hybrid Pipeline Nodes

```python
async def node_search(state: ResearchState) -> Command:
    """Execute search using tools/subagents"""
    
    # Dynamically select tools based on enabled sources
    enabled_tools = {
        "web": search_web,
        "academic": research_academics,
        "financial": analyze_financials,
        # ... map sources to tools
    }
    
    # Run tool calls
    evidence_results = []
    for source in state["sources"]:
        if source in enabled_tools:
            tool = enabled_tools[source]
            result = await tool(state["query"])
            evidence_results.extend(result)
    
    # Store evidence in both persistent store and vector store
    for i, evidence in enumerate(evidence_results):
        evidence_id = f"{state['query'][:20]}_{i}_{datetime.now().timestamp()}"
        await store_evidence(store, evidence_id, evidence)
        state["evidence_ids"].append(evidence_id)
        await add_evidence_to_vector_store([evidence])
    
    return Command(
        update={
            "evidence_ids": state["evidence_ids"],
            "plan_executed": state["plan_executed"] + 1,
        }
    )

async def node_reranker(state: ResearchState) -> Command:
    """Rerank evidence by semantic relevance"""
    
    # Retrieve from vector store
    relevant_evidence = await semantic_search_evidence(state["query"], k=10)
    
    # Sort by relevance
    sorted_evidence = sorted(
        relevant_evidence,
        key=lambda x: x.get("similarity", 0.0),
        reverse=True,
    )
    
    # Store top evidence IDs
    top_ids = [e["url"] for e in sorted_evidence[:5]]
    
    return Command(
        update={
            "evidence_ids": top_ids,  # Update with top-ranked
        }
    )

async def node_confidence_evaluator(state: ResearchState) -> Command:
    """Determine if research is sufficient"""
    
    if state["plan_executed"] >= {"speed": 1, "balanced": 3, "quality": 6}.get(state["mode"], 3):
        return Command(
            update={
                "should_continue": False,
                "stop_reason": "plan_complete",
            }
        )
    
    # Check evidence sufficiency
    if len(state["evidence_ids"]) >= {"speed": 2, "balanced": 5, "quality": 10}.get(state["mode"], 5):
        # High confidence; can stop
        return Command(
            update={
                "should_continue": False,
                "stop_reason": "sufficient_evidence",
            }
        )
    
    return Command(
        update={
            "should_continue": True,
        }
    )

async def node_renderer(state: ResearchState) -> Command:
    """Generate final answer using task profile schema"""
    
    from langchain.chat_models import init_chat_model
    
    model = init_chat_model("gpt-4")
    
    # Retrieve top evidence
    top_evidence = [
        await retrieve_evidence(store, eid)
        for eid in state["evidence_ids"][:5]
    ]
    
    evidence_text = "\n".join([
        f"[{i+1}] {e.get('title', e['url'])}\n{e['content']}"
        for i, e in enumerate(top_evidence)
    ])
    
    # Get task profile output schema
    output_schemas = {
        "research_report": ResearchReportOutput,
        "thesis": ThesisOutput,
        "comparison": ComparisonOutput,
    }
    
    output_schema = output_schemas.get(state["task_profile"], ResearchReportOutput)
    
    prompt = f"""
    Generate a {state['task_profile']} for: {state['query']}
    
    Use this evidence:
    {evidence_text}
    
    Format your response as specified in the task profile schema.
    Include inline citations [1], [2], etc.
    """
    
    # Call with structured output
    response = await model.with_structured_output(output_schema).ainvoke([
        {"role": "user", "content": prompt}
    ])
    
    return Command(
        update={
            "final_answer": response.model_dump(),
        }
    )
```

---

## 5. Memory Management: StateBackend vs StoreBackend

| Feature | StateBackend | StoreBackend |
|---|---|---|
| **Storage** | LangGraph agent state | Persistent LangGraph Store |
| **Lifetime** | Single invocation | Across invocations |
| **Use Case** | Scratch pad, temporary work | Long-term memory, knowledge base |
| **Thread-safe** | No | Yes |

**Typical pattern**:
- **StateBackend**: Store current evidence during active research
- **StoreBackend**: Store research history, previous queries, learned patterns

```python
from deepagents.backends import StateBackend, StoreBackend, CompositeBackend

# Use composite backend
composite_backend = lambda rt: CompositeBackend(
    default=StateBackend(rt),
    routes={
        "/evidence/": StoreBackend(rt),  # Persistent evidence
    }
)

research_agent = create_deep_agent(
    backend=composite_backend,
    store=PostgresStore(...),
)
```

---

## 6. Response Format: Structured Output

### 6.1 Using ProviderStrategy (Native OpenAI)

```python
from langchain.agents.structured_output import ProviderStrategy
from pydantic import BaseModel

class ResearchOutput(BaseModel):
    summary: str
    findings: list[str]
    citations: dict

agent = create_deep_agent(
    model=init_chat_model("gpt-4"),
    response_format=ProviderStrategy(schema=ResearchOutput),  # Native OpenAI
)

result = agent.invoke({...})
structured_output = result["structured_response"]  # Validated Pydantic instance
```

### 6.2 Using ToolStrategy (Non-OpenAI Models)

```python
from langchain.agents.structured_output import ToolStrategy

agent = create_deep_agent(
    model=init_chat_model("claude-3-opus"),  # Non-native
    response_format=ToolStrategy(schema=ResearchOutput),  # Tool calling
)
```

---

## 7. Configuration & Setup

### 7.1 Environment Variables

```bash
# LLM
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Search
SEARX_HOST=http://localhost:8888

# Document Scraping
FIRECRAWL_URL=http://localhost:3000

# Embeddings
EMBEDDING_MODEL=openai:text-embedding-3-small

# Vector Store (Supabase)
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJ...

# Persistent Store (Postgres)
POSTGRES_CONNECTION_STRING=postgresql://user:pass@localhost/research_db

# LangSmith
LANGSMITH_API_KEY=ls-...
```

### 7.2 Factory Function

```python
async def create_research_agent(
    mode: str = "balanced",
    sources: list[str] = None,
    use_vector_store: bool = True,
    use_persistent_store: bool = False,
):
    """Factory to create pre-configured research agent"""
    
    from deepagents import create_deep_agent
    from langgraph.store.memory import InMemoryStore
    from langgraph.store.postgres import PostgresStore
    
    # Select store backend
    if use_persistent_store:
        store = PostgresStore(conn_string=os.getenv("POSTGRES_CONNECTION_STRING"))
    else:
        store = InMemoryStore()
    
    # Create agent
    agent = create_deep_agent(
        model=init_chat_model("gpt-4"),
        tools=[search_web, extract_from_document, fetch_financial_data],
        system_prompt=f"""
        You are a research agent operating in {mode} mode.
        Enabled sources: {', '.join(sources or ['web'])}
        {MODE_INSTRUCTIONS.get(mode)}
        """,
        response_format=ProviderStrategy(schema=ResearchOutput),
        checkpointer=MemorySaver(),
        store=store,
    )
    
    return agent
```

---

## 8. Usage Examples

### 8.1 Simple Research Query

```python
agent = await create_research_agent(mode="balanced")

result = await agent.ainvoke({
    "messages": [
        {
            "role": "user",
            "content": "What are the latest developments in quantum computing?"
        }
    ]
})

output = result["structured_response"]
print(output.summary)
print(f"Confidence: {output.confidence_score}")
```

### 8.2 With Custom Configuration

```python
result = await agent.ainvoke({
    "messages": [
        {
            "role": "user",
            "content": """
            Research: Compare Python vs Go for microservices
            Mode: quality
            Sources: web, academic, code
            Task Profile: comparison
            """
        }
    ]
})
```

---

## 9. Testing & Observability

### 9.1 LangSmith Integration

```python
import langsmith

langsmith.configure(api_key=os.getenv("LANGSMITH_API_KEY"))

# Automatic tracing
result = await agent.ainvoke(
    {"messages": [...]},
    config={"run_name": "research_query_001"},
)
```

### 9.2 Evaluation Framework

```python
from langsmith import evaluate

def eval_relevance(run, example):
    """Check if answer matches query"""
    output = run.outputs["structured_response"]
    # Compare similarity
    return {"pass": similarity(output.summary, example.reference) > 0.8}

def eval_citations(run, example):
    """Check citation coverage"""
    output = run.outputs["structured_response"]
    return {"coverage": len(output.citations) > 0}
```

---

## 10. Comparison: Pure DeepAgent vs Hybrid

| Aspect | Pure DeepAgent | Hybrid LangGraph |
|---|---|---|
| **Complexity** | Minimal | Moderate |
| **Boilerplate** | Least | More |
| **Mode Control** | Soft (prompt) | Hard (explicit) |
| **Evidence Flow** | Implicit | Explicit |
| **Debugging** | Harder | Easier |
| **Extensibility** | Via tools/subagents | Via nodes/subgraphs |
| **Performance** | Faster (fewer steps) | Tunable |
| **Recommended for** | MVP, simple use cases | Production, complex workflows |

---

## 11. Integration Checklist

### Phase 1: Pure DeepAgent MVP
- [ ] Define tools (search_web, extract_from_document, etc.)
- [ ] Define Pydantic output schemas
- [ ] Create research agent with `create_deep_agent`
- [ ] Test with LangSmith
- [ ] Deploy to production

### Phase 2: Add Vector Store
- [ ] Set up Supabase + pgvector
- [ ] Implement semantic_search_evidence
- [ ] Add vector store indexing to tools
- [ ] Test relevance ranking

### Phase 3: Hybrid Approach (if needed)
- [ ] Build custom LangGraph for explicit pipeline
- [ ] Add mode-based stopping criteria
- [ ] Integrate Store + StateBackend
- [ ] Add evidence reranking node

### Phase 4: Finance Specialization
- [ ] Create subagents for financial analysis
- [ ] Add financial data tools
- [ ] Define finance-specific task profiles
- [ ] Test on financial queries

---

## 12. Known Limitations & Future Work

1. **No native multi-turn refinement**: User cannot request deeper dives on specific claims (use custom node for this)
2. **Store query limitations**: LangGraph Store doesn't support semantic queries (use vector store alongside)
3. **Mode control via prompt only**: For strict enforcement, use hybrid LangGraph approach
4. **No built-in fact-checking**: Integrate external fact-check APIs in tools
5. **Real-time data**: Add live API tools (stock prices, weather) as needed

---

## 13. File Structure

```
research-agent/
├── agent.py                      # Main agent factory
├── tools/
│   ├── web_search.py            # search_web tool
│   ├── document.py              # extract_from_document tool
│   ├── financial.py             # fetch_financial_data tool
│   └── __init__.py
├── schemas/
│   ├── output.py                # Pydantic models for response_format
│   ├── config.py                # ResearchConfig, ResearchMode
│   └── __init__.py
├── store/
│   ├── vector_store.py          # Supabase vector store setup
│   ├── persistent_store.py      # PostgreSQL store setup
│   └── __init__.py
├── hybrid/ (optional)
│   ├── graph.py                 # Custom LangGraph definition
│   ├── nodes.py                 # Graph node implementations
│   └── __init__.py
├── tests/
│   ├── test_agent.py
│   ├── test_tools.py
│   └── test_output_schemas.py
├── config.py                    # ENV var loading
├── requirements.txt
└── README.md
```

---

**Document Status**: Ready for Implementation (Simplified Edition)  
**Recommended Start**: Phase 1 Pure DeepAgent MVP  
**Estimated Effort**: 2-3 days for MVP, 1-2 weeks for production-ready with vector store
