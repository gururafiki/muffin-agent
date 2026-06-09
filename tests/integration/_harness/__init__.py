"""Reusable harness for muffin E2E integration tests.

Four mock seams, lowest reasonable level:

* **LLM** — :func:`patch_llm` / :class:`ScriptedChatModel`. A *real*
  ``BaseChatModel`` subclass that replays a scripted timeline of model turns.
  Drives real ReAct loops (``bind_tools`` + ``_generate``) and the real
  structured-output paths (``response_format`` tool-call AND the direct-node
  ``with_structured_output`` path). One shared cursor = one ordered timeline
  across every node in the graph.
* **MCP tools** — :func:`patch_mcp` / :func:`build_fake_mcp_tools`. Fixture-backed
  ``StructuredTool``s; the *real* ``get_tools`` name-filter still runs.
* **Sandbox** — :func:`patch_sandbox`. In-memory ``execute_python`` stand-in.
* **Embeddings** — :func:`patch_embeddings`. Canned-vector fake for the research
  rerank step.
"""

from .mcp import build_fake_mcp_tools, patch_mcp
from .sandbox import patch_sandbox
from .scripted_model import (
    SchemaRoutedModel,
    Script,
    ScriptedChatModel,
    ScriptExhaustedError,
    final,
    patch_llm,
    patch_llm_by_schema,
    tool_turn,
)

__all__ = [
    "SchemaRoutedModel",
    "Script",
    "ScriptExhaustedError",
    "ScriptedChatModel",
    "build_fake_mcp_tools",
    "final",
    "patch_embeddings",
    "patch_llm",
    "patch_llm_by_schema",
    "patch_mcp",
    "patch_sandbox",
    "tool_turn",
]


def patch_embeddings(vectors: dict[str, list[float]] | None = None):
    """Patch research's ``OpenAIEmbeddings`` with a canned-vector fake.

    Lazy import to avoid pulling the research package for non-research tests.
    """
    from .embeddings import patch_embeddings as _impl

    return _impl(vectors)
