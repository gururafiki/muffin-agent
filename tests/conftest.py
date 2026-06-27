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
