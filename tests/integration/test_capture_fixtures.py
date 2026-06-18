"""Live fixture refresh — snapshots real MCP payloads (opt-in).

Skipped automatically unless the MCP stack is reachable, so a plain ``pytest``
run never fails on it. To refresh the committed fixtures with genuine payloads::

    docker compose up -d openbb-mcp firecrawl-mcp searxng
    .venv/bin/pytest tests/integration/test_capture_fixtures.py -m live

This is the *capture* half of the hybrid fixture strategy; the offline examples
(``test_equity_price_collector.py`` etc.) consume whatever the library holds.
"""

from __future__ import annotations

import pytest

from ._harness.capture import CAPTURE_PLAN, capture_all, mcp_reachable

pytestmark = [pytest.mark.live, pytest.mark.asyncio]


async def test_refresh_mcp_fixtures():
    config = {"configurable": {}}
    if not mcp_reachable(config):
        pytest.skip(
            "OpenBB MCP not reachable — run `docker compose up -d openbb-mcp "
            "firecrawl-mcp searxng` to capture live fixtures."
        )
    written = await capture_all(config)
    assert len(written) == len(CAPTURE_PLAN)
    for path in written:
        assert path.exists() and path.stat().st_size > 0
