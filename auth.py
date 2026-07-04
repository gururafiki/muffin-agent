"""Optional authentication for the Muffin LangGraph server.

Wired via ``langgraph.json`` (``"auth": {"path": "./auth.py:auth"}``). It gates the
default LangGraph routes (``/assistants``, ``/threads``, ``/runs``).

Four modes, auto-selected from the environment so local development keeps working:

1. **Disabled** (default — nothing configured): every request is allowed as an
   anonymous user. ``langgraph dev`` and local ``docker compose up`` are unaffected.
2. **Shared bearer token**: set ``MUFFIN_API_TOKEN``. Requests must send
   ``Authorization: Bearer <token>``. Good for a single trusted client or CI.
3. **Cloudflare Access JWT**: set ``CF_ACCESS_TEAM_DOMAIN`` (e.g.
   ``myteam.cloudflareaccess.com``) and ``CF_ACCESS_AUD`` (the Access application
   AUD tag). Requests must carry a valid ``Cf-Access-Jwt-Assertion`` header
   (injected by Cloudflare Access). The verified email becomes the user identity,
   which LangGraph exposes as ``configurable.user_id`` — enabling muffin's
   per-user memory isolation. Requires PyJWT: ``pip install "pyjwt[crypto]"``.
4. **Supabase (GoTrue) JWT**: set ``SUPABASE_JWT_SECRET`` (the self-hosted
   project's HS256 signing secret) and, recommended, ``SUPABASE_URL`` (enables
   the issuer check against ``<SUPABASE_URL>/auth/v1``). User access tokens are
   sent as ``Authorization: Bearer`` by the muffin app; the verified ``sub``
   (the Supabase user UUID) becomes the identity. The anon / service_role API
   keys are rejected here by the ``aud=authenticated`` claim check — only real
   user sessions authenticate. HS256 only needs base PyJWT (no crypto extra).

Modes 2–4 may be enabled together (any accepted credential wins).

**Per-user thread isolation**: when any auth mode is enabled, the ``@auth.on.threads``
handler stamps ``metadata.owner`` on created threads/runs and filters reads/searches
by it, so users only see their own runs (the app's Calls tab). The shared-token
identity (``api-client``) is exempt and sees everything; assistants stay unfiltered
(presets are non-secret and shared by design). Threads created before this handler
existed have no ``owner`` and are visible only to the exempt identity.
"""

from __future__ import annotations

import hmac
import os
from typing import Any

from langgraph_sdk import Auth

auth = Auth()

_API_TOKEN = os.environ.get("MUFFIN_API_TOKEN")
_CF_TEAM_DOMAIN = os.environ.get("CF_ACCESS_TEAM_DOMAIN")
_CF_AUD = os.environ.get("CF_ACCESS_AUD")
_SUPABASE_URL = os.environ.get("SUPABASE_URL")
_SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET")
_AUTH_ENABLED = bool(
    _API_TOKEN or (_CF_TEAM_DOMAIN and _CF_AUD) or _SUPABASE_JWT_SECRET
)

# Identities that bypass per-user resource scoping: the anonymous single-tenant
# mode and the shared-bearer client (trusted operator tooling).
_SCOPE_EXEMPT_IDENTITIES = frozenset({"anonymous", "api-client"})


def _header(headers: Any, name: str) -> str | None:
    """Read a header value across str/bytes-keyed mappings (case-insensitive)."""
    if not headers:
        return None
    for key in (name, name.lower(), name.encode(), name.lower().encode()):
        getter = getattr(headers, "get", None)
        val = getter(key) if getter else None
        if val is not None:
            return val.decode() if isinstance(val, (bytes, bytearray)) else str(val)
    items = getattr(headers, "items", None)
    if items:
        for k, v in items():
            kk = k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
            if kk.lower() == name.lower():
                return v.decode() if isinstance(v, (bytes, bytearray)) else str(v)
    return None


def _secure_eq(a: str, b: str) -> bool:
    """Compare two strings in constant time."""
    return hmac.compare_digest(a, b)


def _user(identity: str) -> dict[str, Any]:
    """Build a minimal authenticated-user dict for the given identity."""
    return {
        "identity": identity,
        "is_authenticated": True,
        "permissions": ["authenticated"],
    }


def _verify_cf_access(token: str) -> str | None:
    """Verify a Cloudflare Access JWT, returning the email/sub identity or None."""
    try:
        import jwt  # PyJWT
        from jwt import PyJWKClient
    except ImportError:
        return None
    issuer = f"https://{_CF_TEAM_DOMAIN}"
    try:
        signing_key = PyJWKClient(
            f"{issuer}/cdn-cgi/access/certs"
        ).get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=_CF_AUD,
            issuer=issuer,
        )
    except Exception:
        return None
    identity = claims.get("email") or claims.get("sub")
    return str(identity) if identity else None


def _verify_supabase(token: str) -> str | None:
    """Verify a Supabase (GoTrue) HS256 access token → the user UUID or None."""
    if not _SUPABASE_JWT_SECRET:
        return None
    try:
        import jwt  # PyJWT
    except ImportError:
        return None
    kwargs: dict[str, Any] = {"algorithms": ["HS256"], "audience": "authenticated"}
    if _SUPABASE_URL:
        kwargs["issuer"] = f"{_SUPABASE_URL}/auth/v1"
    try:
        claims = jwt.decode(
            token,
            _SUPABASE_JWT_SECRET,
            options={"require": ["exp", "sub"]},
            **kwargs,
        )
    except Exception:
        return None
    sub = claims.get("sub")
    return str(sub) if sub else None


@auth.authenticate
async def authenticate(headers: Any) -> dict[str, Any]:
    """Authenticate an incoming request and return the user identity."""
    # Mode 1 — auth disabled: allow anonymous (keeps local dev / Studio working).
    if not _AUTH_ENABLED:
        return _user("anonymous")

    authorization = _header(headers, "authorization")
    bearer: str | None = None
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization[7:].strip()

    # Mode 2 — shared bearer token.
    if _API_TOKEN and bearer and _secure_eq(bearer, _API_TOKEN):
        return _user("api-client")

    # Mode 4 — Supabase user JWT (HS256 fails fast on non-Supabase tokens, so
    # Cloudflare's RS256 assertions fall through to mode 3 below).
    if _SUPABASE_JWT_SECRET and bearer:
        identity = _verify_supabase(bearer)
        if identity:
            return _user(identity)

    # Mode 3 — Cloudflare Access JWT (dedicated header, or carried as a bearer token).
    if _CF_TEAM_DOMAIN and _CF_AUD:
        cf_token = _header(headers, "cf-access-jwt-assertion") or bearer
        if cf_token:
            identity = _verify_cf_access(cf_token)
            if identity:
                return _user(identity)

    raise Auth.exceptions.HTTPException(
        status_code=401, detail="Invalid or missing credentials"
    )


@auth.on.threads
async def scope_threads(
    ctx: Auth.types.AuthContext, value: dict[str, Any]
) -> dict[str, str] | None:
    """Per-user thread isolation: stamp `owner` on writes, filter reads by it.

    Applies to every thread action (create / create_run / read / search /
    update / delete). Returning the filter dict makes LangGraph reject or hide
    resources whose metadata doesn't match; returning None applies no scoping
    (single-tenant modes and the trusted shared-token client).
    """
    identity = ctx.user.identity
    if not _AUTH_ENABLED or identity in _SCOPE_EXEMPT_IDENTITIES:
        return None
    filters = {"owner": identity}
    # On create/create_run this metadata is persisted with the resource; on
    # read-style actions mutating the payload is harmless (documented pattern).
    metadata = value.setdefault("metadata", {})
    metadata.update(filters)
    return filters
