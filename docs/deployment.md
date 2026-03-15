# Deployment Guide (Not suitable for production yet)

Deploy the Muffin Agent to a [LangGraph Standalone Server](https://docs.langchain.com/langsmith/deploy-standalone-server) (Self-Hosted Lite).

## Prerequisites

Complete the [Setup steps in the README](../README.md#-setup) first (install, OpenBB MCP server, OpenSandbox, environment variables).

Additionally you need:
- **Docker** and **Docker Compose**
- **LangSmith account** (free Developer plan) — [sign up](https://smith.langchain.com/)
- `LANGSMITH_API_KEY` in your `.env` — [get one here](https://smith.langchain.com/settings)

## Local Development (No Docker)

For development with hot-reload:

```bash
langgraph dev
```

Requires the OpenBB MCP server and the OpenSandbox server running on localhost, and all env vars set in `.env`.

Start OpenSandbox locally:

```bash
docker run -d --name opensandbox \
  -p 8080:8080 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  ghcr.io/alibaba/opensandbox/server:latest
```

## Docker Deployment

### 1. Build the Docker image

```bash
langgraph build -t muffin-agent:latest
```

This uses `langgraph.json` to bundle the agent into a Docker image with all dependencies.

### 2. Configure OpenBB provider API keys

Copy the OpenBB env example and fill in your keys:

```bash
cp extras/openbb/.env.example extras/openbb/.env
```

See [extras/openbb/README.md](../extras/openbb/README.md) for which providers are free and where to get keys.

### 3. Run with Docker Compose

```bash
docker compose up
```

This starts six services:
- **langgraph-api** — the agent server on port `8123`
- **agent-chat-ui** — chat interface on port `3000`
- **opensandbox-server** — OpenSandbox container manager on port `8080` (internal only); mounts the Docker socket to provision per-conversation Python execution containers
- **openbb-mcp** — OpenBB MCP server (internal only)
- **langgraph-postgres** — PostgreSQL for persistence (internal only)
- **langgraph-redis** — Redis for streaming pub-sub (internal only)

Open [http://localhost:3000](http://localhost:3000) to use the chat UI.

### 4. Verify

```bash
curl http://localhost:8123/ok
# Expected: {"ok": true}
```

## Using the Deployed Agent

### Via API

```bash
curl -X POST http://localhost:8123/runs/stream \
  -H "Content-Type: application/json" \
  -d '{
    "input": {"messages": [{"role": "user", "content": "Analyze AAPL"}]},
    "assistant_id": "stock_evaluation"
  }'
```

### Via Agent Chat UI

Open [http://localhost:3000](http://localhost:3000) — the chat UI is included in Docker Compose and pre-configured to connect to the agent server.

Alternatively, use the hosted version at [agentchat.vercel.app](https://agentchat.vercel.app/) and point it to `http://localhost:8123`.

### Via LangSmith Studio

Open [LangSmith Studio](https://smith.langchain.com/) — your deployed agent will appear under deployments.

## Architecture

```
┌──────────────┐
│ Agent Chat UI│ (:3000)
└──────┬───────┘
       │
┌──────▼─────────┐     ┌─────────────────┐     ┌──────────────────────┐
│ LangGraph API  │────▶│  OpenBB MCP     │     │  OpenSandbox Server  │
│   (:8123)      │     │   (:8001)       │     │   (:8080)            │
└───────┬────────┘     └─────────────────┘     └──────────┬───────────┘
        │                                                  │
 ┌──────┴──────┐                                sandbox containers
 │             │                                (one per conversation)
┌▼──────────┐ ┌▼───────────┐
│ PostgreSQL│ │   Redis     │
│ (persist) │ │ (streaming) │
└───────────┘ └─────────────┘
```

**Sandbox lifecycle**: Each chat conversation gets its own isolated container.
Sandboxes are discovered lazily by `thread_id` metadata — `get_backend` and
`execute_python` call the OpenSandbox API to find a running container tagged
with the current `thread_id`. If none exists, a new container is created
automatically. If the container dies mid-conversation (e.g. 1-hour timeout,
container crash), a new container is created transparently on the next call
(in-sandbox state like installed packages or written files is lost).
Containers auto-terminate after a 1-hour idle timeout.

## Production Considerations

The current Docker Compose setup is intended for **local development**. Before deploying to a production environment, address the following:

### Security
- [ ] **Database credentials** — Replace hardcoded `postgres:postgres` with strong credentials via environment variables or secrets manager
- [ ] **Redis authentication** — Enable `requirepass` on Redis
- [ ] **TLS/HTTPS** — Add a reverse proxy (nginx, Traefik, or Caddy) for HTTPS termination
- [ ] **API authentication** — The LangGraph API has no auth by default; add authentication at the proxy layer or via LangSmith API keys
- [ ] **OpenSandbox authentication** — Set `OPENSANDBOX_API_KEY` and `SANDBOX_API_KEY` in the compose environment to require auth on the sandbox server
- [ ] **Sandbox image** — Set `OPENSANDBOX_IMAGE` to a hardened custom image with only the required packages; avoid `python:3.11-slim` in production as it allows arbitrary package installation at runtime

### Reliability
- [ ] **Postgres backups** — Set up periodic `pg_dump` or use a managed database service
- [ ] **Resource limits** — Add `mem_limit` / `cpus` constraints to containers
- [ ] **Logging** — Configure centralized log collection (e.g., `docker compose logs` to a log aggregator)

### Build reproducibility
- [ ] **Pin Agent Chat UI version** — The Dockerfile clones `main` at build time; pin to a specific commit or tag for reproducible builds

## Limits & Alternatives

**Self-Hosted Lite** is free up to **1 million node executions**. After that, upgrade to Self-Hosted Enterprise (contact LangChain sales).

**Alternative — [Aegra](https://github.com/ibbybuilds/aegra)**: An open-source, Apache 2.0 licensed drop-in replacement for LangSmith Deployments. Free and unlimited. Compatible with the same LangGraph SDK, Agent Chat UI, and CopilotKit. Includes OpenTelemetry tracing (works with Langfuse).
