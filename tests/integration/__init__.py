"""End-to-end integration tests for muffin graphs.

These tests run the *real* compiled graph (real ReAct loop, real middleware
stack, real routing, real deterministic nodes) and mock **only** the external
boundaries: LLM calls, MCP tools, the sandbox, and embeddings. See
``docs/integration-testing.md`` for the full recipe and the reusable harness in
``tests/integration/_harness/``.
"""
