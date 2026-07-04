"""Tests for the root-level ``auth.py`` LangGraph auth hook.

Covers mode auto-selection, Supabase (GoTrue) HS256 verification — including the
rejection of the anon/service_role API keys, which are HS256 JWTs signed with the
same secret but without the ``aud=authenticated`` user claim — and the
``@auth.on.threads`` per-user ownership scoping.

``auth.py`` reads its configuration from the environment at import time, so each
test loads a fresh module instance via ``_load_auth`` with a controlled env.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jwt
import pytest
from langgraph_sdk import Auth

AUTH_PATH = Path(__file__).resolve().parents[1] / "auth.py"
SECRET = "unit-test-jwt-secret-0123456789abcdef0123456789abcdef"
SUPABASE_URL = "https://supabase.example.com"
_ENV_KEYS = (
    "MUFFIN_API_TOKEN",
    "CF_ACCESS_TEAM_DOMAIN",
    "CF_ACCESS_AUD",
    "SUPABASE_URL",
    "SUPABASE_JWT_SECRET",
    "MUFFIN_AUTH_OPTIONAL",
)


def _load_auth(monkeypatch: pytest.MonkeyPatch, **env: str) -> Any:
    """Import a fresh auth.py under a controlled environment."""
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    spec = importlib.util.spec_from_file_location("muffin_auth_under_test", AUTH_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mint(
    *,
    secret: str = SECRET,
    sub: str | None = "9f6d0d05-1111-4222-8333-444455556666",
    aud: str | None = "authenticated",
    iss: str | None = f"{SUPABASE_URL}/auth/v1",
    exp_delta: int = 3600,
    role: str = "authenticated",
) -> str:
    """Mint a GoTrue-shaped HS256 token; None claims are omitted."""
    now = int(time.time())
    payload: dict[str, Any] = {
        "sub": sub,
        "aud": aud,
        "iss": iss,
        "role": role,
        "iat": now,
        "exp": now + exp_delta,
    }
    return jwt.encode(
        {k: v for k, v in payload.items() if v is not None}, secret, algorithm="HS256"
    )


def _ctx(identity: str, action: str = "create") -> Any:
    """Minimal AuthContext stand-in (handlers read ctx.user.identity + ctx.action)."""
    return SimpleNamespace(user=SimpleNamespace(identity=identity), action=action)


@pytest.mark.unit
class TestModeSelection:
    @pytest.mark.asyncio
    async def test_disabled_allows_anonymous(self, monkeypatch):
        mod = _load_auth(monkeypatch)
        user = await mod.authenticate(headers={})
        assert user["identity"] == "anonymous"

    @pytest.mark.asyncio
    async def test_supabase_secret_enables_auth(self, monkeypatch):
        mod = _load_auth(monkeypatch, SUPABASE_JWT_SECRET=SECRET)
        with pytest.raises(Auth.exceptions.HTTPException):
            await mod.authenticate(headers={})


@pytest.mark.unit
class TestSupabaseMode:
    @pytest.mark.asyncio
    async def test_valid_token_yields_sub_identity(self, monkeypatch):
        mod = _load_auth(
            monkeypatch, SUPABASE_JWT_SECRET=SECRET, SUPABASE_URL=SUPABASE_URL
        )
        token = _mint()
        user = await mod.authenticate(headers={"authorization": f"Bearer {token}"})
        assert user["identity"] == "9f6d0d05-1111-4222-8333-444455556666"
        assert user["is_authenticated"] is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "token_kwargs",
        [
            pytest.param({"exp_delta": -60}, id="expired"),
            pytest.param(
                {"secret": "wrong-secret-0123456789abcdef0123456789abcdef"},
                id="wrong-secret",
            ),
            pytest.param(
                {"iss": "https://evil.example.com/auth/v1"}, id="wrong-issuer"
            ),
            # The anon / service_role API keys: same secret, but no user claims.
            pytest.param(
                {"aud": None, "sub": None, "role": "anon", "iss": "supabase"},
                id="anon-api-key",
            ),
            pytest.param(
                {"aud": None, "sub": None, "role": "service_role", "iss": "supabase"},
                id="service-role-api-key",
            ),
            pytest.param({"sub": None}, id="missing-sub"),
        ],
    )
    async def test_invalid_tokens_rejected(self, monkeypatch, token_kwargs):
        mod = _load_auth(
            monkeypatch, SUPABASE_JWT_SECRET=SECRET, SUPABASE_URL=SUPABASE_URL
        )
        token = _mint(**token_kwargs)
        with pytest.raises(Auth.exceptions.HTTPException):
            await mod.authenticate(headers={"authorization": f"Bearer {token}"})

    @pytest.mark.asyncio
    async def test_issuer_not_checked_without_supabase_url(self, monkeypatch):
        mod = _load_auth(monkeypatch, SUPABASE_JWT_SECRET=SECRET)
        token = _mint(iss="https://anything.example.com/auth/v1")
        user = await mod.authenticate(headers={"authorization": f"Bearer {token}"})
        assert user["is_authenticated"] is True

    @pytest.mark.asyncio
    async def test_shared_bearer_token_coexists(self, monkeypatch):
        mod = _load_auth(
            monkeypatch, SUPABASE_JWT_SECRET=SECRET, MUFFIN_API_TOKEN="shared-tok"
        )
        user = await mod.authenticate(headers={"authorization": "Bearer shared-tok"})
        assert user["identity"] == "api-client"


@pytest.mark.unit
class TestOptionalMode:
    """MUFFIN_AUTH_OPTIONAL=true — sign-in adds identity, absence is anonymous."""

    @pytest.mark.asyncio
    async def test_no_credentials_falls_back_to_anonymous(self, monkeypatch):
        mod = _load_auth(
            monkeypatch, SUPABASE_JWT_SECRET=SECRET, MUFFIN_AUTH_OPTIONAL="true"
        )
        user = await mod.authenticate(headers={})
        assert user["identity"] == "anonymous"

    @pytest.mark.asyncio
    async def test_valid_token_still_yields_identity(self, monkeypatch):
        mod = _load_auth(
            monkeypatch, SUPABASE_JWT_SECRET=SECRET, MUFFIN_AUTH_OPTIONAL="true"
        )
        user = await mod.authenticate(headers={"authorization": f"Bearer {_mint()}"})
        assert user["identity"] == "9f6d0d05-1111-4222-8333-444455556666"

    @pytest.mark.asyncio
    async def test_presented_but_invalid_credential_still_401s(self, monkeypatch):
        mod = _load_auth(
            monkeypatch, SUPABASE_JWT_SECRET=SECRET, MUFFIN_AUTH_OPTIONAL="true"
        )
        token = _mint(exp_delta=-60)
        with pytest.raises(Auth.exceptions.HTTPException):
            await mod.authenticate(headers={"authorization": f"Bearer {token}"})


@pytest.mark.unit
class TestThreadScoping:
    """Shared-by-default: open reads, owner-stamped creates, owner-only mutations."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("action", ["read", "search"])
    async def test_reads_are_open_to_everyone(self, monkeypatch, action):
        mod = _load_auth(monkeypatch, SUPABASE_JWT_SECRET=SECRET)
        value: dict[str, Any] = {}
        assert await mod.scope_threads(_ctx("user-uuid", action), value) is None
        assert value == {}

    @pytest.mark.asyncio
    async def test_create_stamps_owner_without_filtering(self, monkeypatch):
        mod = _load_auth(monkeypatch, SUPABASE_JWT_SECRET=SECRET)
        value: dict[str, Any] = {"metadata": {"agentId": "research"}}
        result = await mod.scope_threads(_ctx("user-uuid", "create"), value)
        assert result is None
        assert value["metadata"] == {"agentId": "research", "owner": "user-uuid"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("action", ["update", "delete"])
    async def test_mutations_are_owner_scoped(self, monkeypatch, action):
        mod = _load_auth(monkeypatch, SUPABASE_JWT_SECRET=SECRET)
        result = await mod.scope_threads(_ctx("user-uuid", action), {})
        assert result == {"owner": "user-uuid"}

    @pytest.mark.asyncio
    async def test_create_run_forces_verified_user_id(self, monkeypatch):
        mod = _load_auth(monkeypatch, SUPABASE_JWT_SECRET=SECRET)
        value: dict[str, Any] = {
            "kwargs": {"config": {"configurable": {"user_id": "spoofed"}}}
        }
        await mod.scope_threads(_ctx("user-uuid", "create_run"), value)
        assert value["kwargs"]["config"]["configurable"]["user_id"] == "user-uuid"
        assert value["metadata"]["owner"] == "user-uuid"

    @pytest.mark.asyncio
    async def test_create_run_anonymous_keeps_client_user_id(self, monkeypatch):
        mod = _load_auth(monkeypatch, SUPABASE_JWT_SECRET=SECRET)
        value: dict[str, Any] = {
            "kwargs": {"config": {"configurable": {"user_id": "local-alice"}}}
        }
        await mod.scope_threads(_ctx("anonymous", "create_run"), value)
        assert value["kwargs"]["config"]["configurable"]["user_id"] == "local-alice"
        assert value["metadata"]["owner"] == "anonymous"

    @pytest.mark.asyncio
    async def test_create_run_without_config_shape_is_left_alone(self, monkeypatch):
        mod = _load_auth(monkeypatch, SUPABASE_JWT_SECRET=SECRET)
        value: dict[str, Any] = {"kwargs": "not-a-dict"}
        await mod.scope_threads(_ctx("user-uuid", "create_run"), value)
        assert value["kwargs"] == "not-a-dict"

    @pytest.mark.asyncio
    async def test_api_client_unfiltered(self, monkeypatch):
        mod = _load_auth(monkeypatch, SUPABASE_JWT_SECRET=SECRET)
        value: dict[str, Any] = {}
        assert await mod.scope_threads(_ctx("api-client", "update"), value) is None
        assert value == {}

    @pytest.mark.asyncio
    async def test_no_filter_when_auth_disabled(self, monkeypatch):
        mod = _load_auth(monkeypatch)
        assert await mod.scope_threads(_ctx("someone", "update"), {}) is None
