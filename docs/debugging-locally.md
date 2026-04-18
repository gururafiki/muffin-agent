# Debugging Locally

Run the agent server on the host under a Python debugger while docker compose provides infrastructure (OpenBB MCP, Firecrawl MCP stack, OpenSandbox, SearxNG, Redis, Postgres) and the chat UI. Breakpoints hit in the same files you edit; `langgraph dev` hot-reloads on save.

## Architecture

```
┌──────────────────── docker compose (infra only) ────────────────────┐
│  openbb-mcp        firecrawl-*         opensandbox-server   searxng │
│  langgraph-redis   langgraph-postgres                               │
│                                                                     │
│  agent-chat-ui ────────┐                                            │
└─────────────────────── │ ───────────────────────────────────────────┘
                         │
                         ▼  http://localhost:8123
┌──────────── host (VSCode debugpy) ──────────────────────────────────┐
│  python -m langgraph_cli dev --port 8123                            │
│    └─ src/muffin_agent/**  ←── breakpoints hit here                 │
└─────────────────────────────────────────────────────────────────────┘
```

The base [docker-compose.yml](../docker-compose.yml) describes the full production stack (including the in-docker `langgraph-api`). The dev overlay [docker-compose.dev.yml](../docker-compose.dev.yml) hides `langgraph-api` behind the `production` profile (so host port 8123 is free for `langgraph dev`), publishes host ports for the MCP servers, and swaps in a host-reachable OpenSandbox config. `agent-chat-ui` has `NEXT_PUBLIC_API_URL=http://localhost:8123` baked at build time, so no rebuild is needed.

## Prerequisites

- Docker Desktop with Compose CLI v2.x.
- Python 3.13 virtual env at `.venv/`:
  ```bash
  python3.13 -m venv .venv
  .venv/bin/pip install -e ".[dev]"
  ```
- `.env` populated — copy [.env.example](../.env.example) and fill in LLM keys. OpenBB MCP's default (`http://127.0.0.1:8001/mcp`) already matches the dev overlay's host port. Firecrawl MCP is remapped by the overlay to host port 3100 to avoid colliding with `agent-chat-ui` on 3000; the **LangGraph Dev Server (Debug)** launch config sets `FIRECRAWL_MCP_URL=http://127.0.0.1:3100/mcp` automatically. For the per-agent CLI configs, either add `FIRECRAWL_MCP_URL=http://127.0.0.1:3100/mcp` to your `.env`, or leave it unset if you aren't using Firecrawl tools.
- VSCode with the **Python** extension (provides the `debugpy` launch type).

The VSCode task launches docker compose with [docker-compose.dev.yml](../docker-compose.dev.yml) layered on top. The overlay: hides `langgraph-api` behind `profiles: [production]`, publishes `127.0.0.1:8001` (openbb-mcp) and `127.0.0.1:3100` (firecrawl-mcp) for the host client, and swaps the OpenSandbox config to [config.dev.toml](../extras/opensandbox/config.dev.toml) (`host_ip = "127.0.0.1"`). No `/etc/hosts` edits are required; the production path (no overlay) keeps `host.docker.internal` for container-to-container reach.

## One-click workflow (recommended)

1. Open the repo in VSCode.
2. Run & Debug panel → pick **LangGraph Dev Server (Debug)** → press `F5`.
3. The `Start Docker Infra` preLaunch task runs `docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --wait`. On cold boot expect ~60s (firecrawl-api has a 30s `start_period`, firecrawl-mcp has 60s).
4. After the integrated terminal prints `Server started`, open http://localhost:3000 — agent-chat-ui is already pointing at the host server on port 8123.
5. Set breakpoints anywhere in `src/muffin_agent/**`. Send a message from the chat UI; the request routes to the host-side `langgraph dev` process and your breakpoint fires.
6. Edit a file (e.g. [src/muffin_agent/agents/investment/forecasting.py](../src/muffin_agent/agents/investment/forecasting.py)), save, and `langgraph dev` reloads automatically.
7. `F5 → Stop` detaches the debugger and stops the dev server. Docker infra keeps running; `docker compose down` when you're done.

## Manual / CLI workflow

Useful when running outside VSCode or when you want `--wait-for-client` so the server blocks until a debugger attaches:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --wait
source .venv/bin/activate
python -m langgraph_cli dev --host 127.0.0.1 --port 8123 --no-browser --allow-blocking
```

The `-f docker-compose.dev.yml` overlay swaps the OpenSandbox config to [config.dev.toml](../extras/opensandbox/config.dev.toml) (`host_ip = "127.0.0.1"`) so the host-side client can reach sandboxes. Omit it for the production-profile flow.

To break inside import-time code ([graph.py](../src/muffin_agent/graph.py) runs `asyncio.run(_build_graph())` at module load):

```bash
python -m langgraph_cli dev --host 127.0.0.1 --port 8123 --no-browser \
    --debug-port 5678 --wait-for-client
```

Then attach from VSCode with:

```json
{
    "name": "Attach to langgraph dev",
    "type": "debugpy",
    "request": "attach",
    "connect": { "host": "127.0.0.1", "port": 5678 },
    "justMyCode": false
}
```

## Hot reload

`langgraph dev` watches the project and reloads on save (disable with `--no-reload`). Limitations:

- Edits to the module level of [graph.py](../src/muffin_agent/graph.py) re-run `asyncio.run(_build_graph())`, which re-fetches MCP tools. Fails loudly if infra is down.
- Dependency changes in `pyproject.toml` require a fresh `pip install -e ".[dev]"` in `.venv`, then relaunch.
- The in-memory checkpointer is reset on every reload — thread state is lost.

## Alternative — per-agent CLI debugger

Fastest path for iterating on a single agent without the full graph. 26 launch configs already exist in [.vscode/launch.json](../.vscode/launch.json), grouped as:

- **Data collection agents** (14): Fundamentals, Price, Estimates, Ownership, News, Options, Economy & Macro, Fixed Income, Discovery & Screening, ETF/Index, Currency & Commodities, Fama-French, Regulatory & Filings, Web Search.
- **Stock evaluation**: full deep agent with a custom ticker prompt.
- **Criterion evaluation**: custom ticker + criterion + query.

These run the `muffin` CLI directly — no server involved. Docker infra must still be up (healthchecks gate MCP calls).

## Production stack

When you want to verify the deployed `langgraph-api` image:

```bash
# stop any host-side langgraph dev first (port 8123 collision)
docker compose up --build
```

This brings up the full production stack including the pre-built agent server. Do **not** pass `-f docker-compose.dev.yml` — without the overlay, `opensandbox-server` uses [config.toml](../extras/opensandbox/config.toml)'s `host_ip = "host.docker.internal"` (required for the containerised `langgraph-api` to reach spawned sandboxes via Docker Desktop's internal DNS), the MCP servers are not published to host ports (no need — `langgraph-api` is on the docker network), and `langgraph-api` is not profile-gated so it starts by default.

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `Port 8123 is already allocated` | Stale `langgraph-api` container from a prior production-stack run (`docker compose up` without the dev overlay). `docker compose ps`, then `docker compose down`. |
| `Connection refused` to MCP server | Docker infra not healthy yet. `docker compose ps` — wait for all services to report `(healthy)`. firecrawl-mcp can take ~60s from cold. |
| Breakpoint not hit | Wrong debug session attached. Confirm the VSCode status bar shows **LangGraph Dev Server (Debug)** and that no other `langgraph dev` process is listening on 8123 (`lsof -i :8123`). |
| Chat UI at localhost:3000 returns network errors | The UI still has `NEXT_PUBLIC_API_URL=http://localhost:8123` baked in — make sure `langgraph dev` is actually running and listening on 8123. |
| VSCode "Starting pre-launch task" hangs | `docker compose up -d --wait` on cold boot — firecrawl services can take ~60s. Watch progress with `docker compose ps` in a separate terminal. |
| `SandboxReadyTimeoutException: Sandbox health check timed out` on first agent call (host-side `langgraph dev`) | Stack was started without the `-f docker-compose.dev.yml` overlay, so `opensandbox-server` is advertising sandboxes at `host.docker.internal:PORT` — which is not resolvable on the macOS host. Restart via the VSCode task, or manually: `docker compose down && docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --wait`. |

## Limitations & future work

- **Host-only debugging** — this guide covers roadmap options 4 and 5 (server on host + per-agent CLI). Options 1 (VSCode dev-containers) and 2 (debugpy attached to a docker container with source bind-mount) are not yet implemented; they're open as future enhancements when fully in-container development is needed.
- **`--allow-blocking`** — the launch config passes this flag because [SandboxFactory](../src/muffin_agent/sandbox/factory.py) uses the sync `opensandbox` client (blocking `socket.connect`), which `langgraph dev` would otherwise refuse under ASGI. Migrating to the async client is tracked as a future cleanup.
- **Studio UI** — optional. Pointing your browser at `https://smith.langchain.com/studio/?baseUrl=http://localhost:8123` loads a graph visualiser, but requires a LangSmith account. The `agent-chat-ui` at localhost:3000 is the primary UI for this workflow.
