"""LangGraph state schemas for the research agent.

``ResearchState`` is the full top-level pipeline state. It is a **plain
``TypedDict``, deliberately NOT an ``AgentState``** — the same choice
``CriteriaAnalysisState`` / ``TradingDecisionState`` / ``CouncilState`` make. Every
LLM stage here is a compiled agent added via ``add_node``, and a parent ``messages``
channel would swallow each agent's internal ReAct transcript into the pipeline's own
state: checkpoints bloat, and the per-namespace isolation that makes each stage
independently inspectable is destroyed. Without the channel, each agent's messages
stay in its own ``checkpoint_ns`` — which is exactly where a consumer reads them.

The per-node ``*Input`` TypedDicts declare exactly what each agent node reads from the
outer state. Never pass ``agent.input_schema`` to ``add_node`` — it is a property-less
``RootModel`` that maps ``{}`` and raises at coercion.

``ResearchClassificationFilterState`` is a smaller schema consumed by
``SkillFilterMiddleware[…]`` inside the researcher — its only purpose is to expose
``mode`` and ``task_type`` as flat keys so the filter can read them.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, NotRequired

from langchain.agents import AgentState
from langchain.agents.middleware.types import OmitFromSchema
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

TaskType = Literal[
    "research_report",
    "comparison",
    "how_to",
    "summary",
    "debate",
    "factual_qa",
]
ResearchMode = Literal["speed", "balanced", "quality"]

RESEARCH_MODES: tuple[ResearchMode, ...] = ("speed", "balanced", "quality")


class ResearchState(TypedDict, total=False):
    """Pipeline state for prepare → classifier → lift → researcher → rerank → writer.

    Contract for the ``evidence`` accumulator: callers must emit a ``list`` per state
    update (use ``[]`` for empty). ``operator.add`` concatenates lists. Future
    parallel-source fan-out (multiple writers of ``evidence``) must respect this
    contract to avoid silent collisions.
    """

    # ── Caller input ───────────────────────────────────────────────────
    query: str
    chat_history: list[BaseMessage]
    allowed_sources: list[str]
    mode_override: ResearchMode
    task_type_override: TaskType
    system_instructions: str

    # ── Normalised by prepare_node (pure) ──────────────────────────────
    chat_history_text: str
    today: str

    # ── Classifier output, lifted into flat keys by lift_classification ─
    classification: dict[str, Any]
    standalone_query: str
    task_type: str
    mode: str
    sources_to_use: list[str]
    skip_search: bool

    # ── Researcher accumulator ─────────────────────────────────────────
    evidence: Annotated[list[dict[str, Any]], operator.add]
    notes: str

    # ── Rerank output ──────────────────────────────────────────────────
    reranked_evidence: list[dict[str, Any]]

    # ── Final output ───────────────────────────────────────────────────
    output: dict[str, Any]


# ── Per-node input schemas (what each agent node reads from ResearchState) ────


class ClassifierInput(TypedDict, total=False):
    """Fields the classifier agent reads from the outer state."""

    query: str
    allowed_sources: list[str]
    chat_history_text: str
    today: str


class ResearcherInput(TypedDict, total=False):
    """Fields a researcher agent node reads from the outer state.

    ``mode`` and ``task_type`` feed both the input prompt and
    ``SkillFilterMiddleware`` (which reads them off the agent's own state).
    """

    standalone_query: str
    task_type: str
    mode: str
    sources_to_use: list[str]


class WriterInput(TypedDict, total=False):
    """Fields the writer agent reads from the outer state."""

    standalone_query: str
    task_type: str
    mode: str
    reranked_evidence: list[dict[str, Any]]
    skip_search: bool
    system_instructions: str
    today: str


# ── Agent-side state schemas (extend AgentState; drive the input prompt) ──────


class ClassifierAgentState(AgentState):
    """Classifier agent state. Inputs render into the first human message."""

    query: NotRequired[Annotated[str, OmitFromSchema(input=False, output=True)]]
    allowed_sources: NotRequired[
        Annotated[list[str], OmitFromSchema(input=False, output=True)]
    ]
    chat_history_text: NotRequired[
        Annotated[str, OmitFromSchema(input=False, output=True)]
    ]
    today: NotRequired[Annotated[str, OmitFromSchema(input=False, output=True)]]
    # Written by the agent; must stay in the OUTPUT schema to reach the parent.
    classification: NotRequired[
        Annotated[dict[str, Any], OmitFromSchema(input=True, output=False)]
    ]


class ResearcherAgentState(AgentState):
    """Researcher agent state.

    ``mode`` / ``task_type`` are ALSO the two filtering dimensions
    ``SkillFilterMiddleware[ResearchClassificationFilterState]`` reads.
    """

    standalone_query: NotRequired[
        Annotated[str, OmitFromSchema(input=False, output=True)]
    ]
    task_type: NotRequired[Annotated[str, OmitFromSchema(input=False, output=True)]]
    mode: NotRequired[Annotated[str, OmitFromSchema(input=False, output=True)]]
    sources_to_use: NotRequired[
        Annotated[list[str], OmitFromSchema(input=False, output=True)]
    ]
    evidence: NotRequired[
        Annotated[list[dict[str, Any]], OmitFromSchema(input=True, output=False)]
    ]
    notes: NotRequired[Annotated[str, OmitFromSchema(input=True, output=False)]]


class WriterAgentState(AgentState):
    """Writer agent state."""

    standalone_query: NotRequired[
        Annotated[str, OmitFromSchema(input=False, output=True)]
    ]
    task_type: NotRequired[Annotated[str, OmitFromSchema(input=False, output=True)]]
    mode: NotRequired[Annotated[str, OmitFromSchema(input=False, output=True)]]
    reranked_evidence: NotRequired[
        Annotated[list[dict[str, Any]], OmitFromSchema(input=False, output=True)]
    ]
    skip_search: NotRequired[Annotated[bool, OmitFromSchema(input=False, output=True)]]
    system_instructions: NotRequired[
        Annotated[str, OmitFromSchema(input=False, output=True)]
    ]
    today: NotRequired[Annotated[str, OmitFromSchema(input=False, output=True)]]
    output: NotRequired[
        Annotated[dict[str, Any], OmitFromSchema(input=True, output=False)]
    ]


class ResearchClassificationFilterState(AgentState):
    """State schema fed to ``SkillFilterMiddleware`` inside the researcher.

    The middleware reads ``mode`` and ``task_type`` from these flat keys to filter
    ``skills_metadata`` and inject context into the system prompt. Only these two
    fields are filtering dimensions — keep this schema minimal so the middleware's
    category-key derivation stays accurate.
    """

    mode: NotRequired[str]
    task_type: NotRequired[str]
