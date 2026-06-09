"""Shared fixtures for E2E integration tests.

Every test under ``tests/integration/`` is auto-tagged ``@pytest.mark.integration``
(so ``pytest -m integration`` selects exactly this suite, and ``-m "not live"``
still excludes the capture test which additionally carries ``live``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langgraph.store.memory import InMemoryStore

_HERE = Path(__file__).resolve().parent


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-apply the ``integration`` marker to this package only.

    A subdirectory conftest's ``pytest_collection_modifyitems`` receives the
    *whole session's* items, so we must scope by path — otherwise every test in
    the repo would be tagged ``integration``.
    """
    for item in items:
        item_path = Path(str(getattr(item, "path", item.fspath))).resolve()
        if _HERE == item_path or _HERE in item_path.parents:
            item.add_marker(pytest.mark.integration)


@pytest.fixture
def store() -> InMemoryStore:
    """A fresh in-memory store (tool-result cache, persona/reflection memory)."""
    return InMemoryStore()


@pytest.fixture
def config() -> dict[str, Any]:
    """A minimal ``RunnableConfig`` with a thread id (no real credentials)."""
    return {"configurable": {"thread_id": "integration-test"}}
