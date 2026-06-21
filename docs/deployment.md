# Deployment

> **The production deployment (Terraform + Ansible + Docker Swarm on Oracle Cloud Always-Free) now
> lives in [`muffin-deployment`](https://github.com/gururafiki/muffin-deployment).** This page covers
> only agent-local dev + the LangGraph server surface; for the full stack, image references, Cloudflare
> setup, and the one-command `terraform apply`, see that repo (and the umbrella
> [`muffin`](https://github.com/gururafiki/muffin)).

The Muffin agent is deployed as a [LangGraph Standalone Server](https://docs.langchain.com/langsmith/deploy-standalone-server)
image — `ghcr.io/gururafiki/muffin-agent`, built by this repo's
[`build-image.yml`](../.github/workflows/build-image.yml). `muffin-deployment` runs it as the
`langgraph-api` service behind Traefik + Cloudflare Access.

## Local development

```bash
langgraph dev
```

Requires the OpenBB MCP + Firecrawl MCP + OpenSandbox + SearxNG services on localhost and env vars in
`.env`. The local MCP/infra stack lives in
[`muffin-deployment/compose`](https://github.com/gururafiki/muffin-deployment/tree/main/compose); see
[debugging-locally.md](debugging-locally.md) for the host-debugger workflow.

## API usage

```bash
curl -X POST http://localhost:8123/runs/stream \
  -H "Content-Type: application/json" \
  -d '{"input": {"messages": [{"role": "user", "content": "What is the current AAPL price?"}]}, "assistant_id": "stock_evaluation"}'
```

Graphs: `stock_evaluation`, `criteria_analysis`, `research`, `council` (see [langgraph.json](../langgraph.json)).
Locally the Agent Chat UI ([`agent-chat-ui-docker`](https://github.com/gururafiki/agent-chat-ui-docker))
points at the server via its same-origin `/api` proxy.

## Authentication

The LangGraph Standalone Server ships with **no auth**. Two complementary layers (both configured in
`muffin-deployment`):

- **Edge — Cloudflare Access**: a Zero-Trust Access application over the chat + API hostnames. Browser
  users authenticate via email/SSO; programmatic clients use an Access **service token** (two policies:
  an email `allow` + a service-token `non_identity`).
- **Origin — LangGraph custom auth** ([`auth.py`](../auth.py), wired via the `auth` key in
  [`langgraph.json`](../langgraph.json)): set `MUFFIN_API_TOKEN` (shared bearer) and/or
  `CF_ACCESS_TEAM_DOMAIN` + `CF_ACCESS_AUD` (verifies the Cloudflare Access JWT). With nothing set it
  stays **disabled** (anonymous), so `langgraph dev` is unaffected. When the CF Access JWT is verified,
  the email becomes `configurable.user_id` → real per-user `/memories/` isolation (drop
  `MEMORY_DEBUG_USER_ID` once enabled).

## Limits & alternatives

**Self-Hosted Lite** is free up to **1 million node executions**. After that, upgrade to Self-Hosted
Enterprise (contact LangChain sales).

**Alternative — [Aegra](https://github.com/ibbybuilds/aegra)**: an open-source, Apache-2.0 drop-in
replacement for LangSmith Deployments — free, unlimited, same LangGraph SDK / Agent Chat UI surface,
with OpenTelemetry tracing (works with Langfuse).
