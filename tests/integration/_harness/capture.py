"""Live fixture capture — snapshot REAL MCP tool outputs into the fixture library.

Hybrid sourcing (see ``docs/integration-testing.md``): the committed fixtures are
hand-authored from the OpenBB schemas so the suite runs offline, but they can be
refreshed to genuine payloads whenever the MCP stack is up::

    docker compose up -d openbb-mcp firecrawl-mcp searxng
    .venv/bin/pytest tests/integration/test_capture_fixtures.py -m live

Capture uses the REAL ``McpConfiguration`` + ``get_tools`` path (only the network
is real here — no mocks), invokes each tool with canonical args, and writes the
``ToolMessage`` content verbatim to ``fixtures/<openbb|firecrawl>/<tool>__<sc>.json``.
Adding a tool to the library = add one ``CapturePlanEntry`` here (and/or drop in a
hand-authored file).
"""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from langchain_core.runnables import RunnableConfig

from muffin_agent.agents.data_collection.utils import get_tools
from muffin_agent.mcp_config import McpConfiguration

from .fixtures import FIXTURES_DIR


@dataclass(frozen=True)
class CapturePlanEntry:
    """One tool to capture: its name, canonical args, and fixture scenario."""

    tool_name: str
    args: dict[str, Any]
    scenario: str = "aapl"


# Canonical capture plan — mirrors the committed fixture set. Extend as graphs
# add tools. Args are the minimal set that yields a representative payload.
CAPTURE_PLAN: tuple[CapturePlanEntry, ...] = (
    CapturePlanEntry("equity_price_quote", {"symbol": "AAPL"}),
    CapturePlanEntry("equity_price_historical", {"symbol": "AAPL", "interval": "1d"}),
    CapturePlanEntry("equity_price_performance", {"symbol": "AAPL"}),
    CapturePlanEntry("equity_historical_market_cap", {"symbol": "AAPL"}),
    CapturePlanEntry(
        "equity_fundamental_metrics", {"symbol": "AAPL", "period": "annual"}
    ),
    CapturePlanEntry(
        "equity_fundamental_income", {"symbol": "AAPL", "period": "annual", "limit": 3}
    ),
    CapturePlanEntry("equity_ownership_insider_trading", {"symbol": "AAPL"}),
    CapturePlanEntry("news_company", {"symbol": "AAPL"}),
    CapturePlanEntry(
        "firecrawl_search",
        {"query": "Apple Inc investor relations latest results"},
        scenario="generic",
    ),
)


def mcp_reachable(
    config: RunnableConfig | None = None, *, timeout: float = 1.0
) -> bool:
    """Whether the configured OpenBB MCP host:port accepts a TCP connection."""
    mcp = McpConfiguration.from_runnable_config(config or {})
    parsed = urlparse(mcp.openbb_mcp_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _fixture_path(tool_name: str, scenario: str) -> Path:
    subdir = "firecrawl" if tool_name.startswith("firecrawl") else "openbb"
    return FIXTURES_DIR / subdir / f"{tool_name}__{scenario}.json"


def _to_serialisable(content: Any) -> Any:
    """Normalise tool output to the on-disk fixture shape (envelope / list)."""
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"content": content}
    return content  # list (Firecrawl) or already-structured payload


async def capture_one(config: RunnableConfig, entry: CapturePlanEntry) -> Path:
    """Invoke one real MCP tool and write its output to the fixture library."""
    tools = await get_tools(config, [entry.tool_name])
    tool = next((t for t in tools if t.name == entry.tool_name), None)
    if tool is None:
        raise LookupError(
            f"MCP tool {entry.tool_name!r} not exposed by the live server — "
            "is the right MCP service running?"
        )
    content = await tool.ainvoke(entry.args)
    path = _fixture_path(entry.tool_name, entry.scenario)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_serialisable(content), indent=2) + "\n")
    return path


async def capture_all(
    config: RunnableConfig, plan: tuple[CapturePlanEntry, ...] = CAPTURE_PLAN
) -> list[Path]:
    """Capture every entry in *plan*; returns the written fixture paths."""
    written: list[Path] = []
    for entry in plan:
        written.append(await capture_one(config, entry))
    return written
