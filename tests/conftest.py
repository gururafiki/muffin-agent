"""Session-wide pytest configuration.

Disable LangSmith tracing for the whole test suite. ``muffin_agent.utils.base_config``
calls ``load_dotenv()`` at import, which pulls the repo-root ``.env`` (with
``LANGSMITH_TRACING=true`` + an API key) into the environment — so test runs would
otherwise upload a trace per graph/LLM run to LangSmith (and hit its rate limit,
spewing ``429`` noise).

We force the tracing flags off *here*, at the top of the root conftest, which pytest
imports before any test module imports ``muffin_agent``. ``load_dotenv`` defaults to
``override=False``, so these pre-set values win over the ``.env``. Both the current
(``LANGSMITH_*``) and legacy (``LANGCHAIN_*``) flag names are set so
``langsmith.utils.tracing_is_enabled()`` resolves to ``False`` regardless of which the
environment uses. The API key is left untouched — with tracing off it is never read.
"""

from __future__ import annotations

import os

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import pytest  # noqa: E402  (must follow the env vars above)


@pytest.fixture(autouse=True)
def _clear_mcp_tool_cache():
    """Isolate the process-wide MCP tool cache between tests.

    ``get_tools`` caches discovered tool lists for the process (see
    ``data_collection/utils.py`` — it turns a graph build's 23 MCP round trips
    into one). Nearly every agent test patches
    ``data_collection.utils.MultiServerMCPClient`` with its own fixture tools, so
    without this reset the first test to run would poison every later one with
    its tools, and call-count assertions would see zero calls.
    """
    from muffin_agent.agents.data_collection.utils import reset_mcp_tool_cache

    reset_mcp_tool_cache()
    yield
    reset_mcp_tool_cache()
