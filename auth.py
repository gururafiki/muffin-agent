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

**Optional sign-in** (``MUFFIN_AUTH_OPTIONAL=true``): requests that present NO
credential fall back to the anonymous identity instead of 401 — sign-in adds
identity on top of an already-gated perimeter (Cloudflare Access in the muffin
deployment). A credential that IS presented but fails verification still 401s
(fail loud, never silently downgrade a signed-in client).

**Read-shared, write-authenticated threads**: when any auth mode is enabled, the
``@auth.on.threads`` handler leaves reads/searches OPEN (muffin's brand — research
shared by everyone behind the Access perimeter) but requires sign-in to CREATE a
thread or start a run — anonymous callers are read-only (403 on create). A signed-in
create stamps ``metadata.owner`` for attribution and forces the run's
``configurable.user_id`` to the verified identity (single source of truth inside the
graph); ``update`` / ``delete`` require ownership. The shared-token identity
(``api-client``) is fully exempt; assistants stay unfiltered (presets are non-secret
and shared by design).
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
_AUTH_OPTIONAL = os.environ.get("MUFFIN_AUTH_OPTIONAL", "").lower() in {
    "1",
    "true",
    "yes",
}

# Identities that bypass per-user resource scoping entirely (trusted operator
# tooling). Anonymous is NOT exempt — it is scoped to its own shared pool.
_SCOPE_EXEMPT_IDENTITIES = frozenset({"api-client"})


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
    cf_mode = bool(_CF_TEAM_DOMAIN and _CF_AUD)
    cf_token = _header(headers, "cf-access-jwt-assertion")
    if cf_mode:
        token = cf_token or bearer
        if token:
            identity = _verify_cf_access(token)
            if identity:
                return _user(identity)

    # Optional sign-in: no credential we can verify was presented → anonymous
    # (the perimeter is gated elsewhere, e.g. Cloudflare Access). A
    # presented-but-invalid credential still falls through to 401 — never
    # silently downgrade. The ``Cf-Access-Jwt-Assertion`` header is forwarded on
    # EVERY request behind Cloudflare Access, so it only counts as a presented
    # credential when CF mode is actually enabled — otherwise a signed-out
    # browser (which always carries it) could never reach the anonymous path.
    cf_presented = cf_mode and bool(cf_token)
    if _AUTH_OPTIONAL and not bearer and not cf_presented:
        return _user("anonymous")

    raise Auth.exceptions.HTTPException(
        status_code=401, detail="Invalid or missing credentials"
    )


def _force_verified_user_id(value: dict[str, Any], identity: str) -> None:
    """Overwrite the run's ``configurable.user_id`` with the verified identity.

    Makes ``user_id`` the single source of truth inside the graph: every
    configurable consumer (memory namespaces, reflection, future
    ``runtime.context.user_id`` context schemas) sees the authenticated user,
    and a client cannot run as somebody else by sending a different value.
    Mutating the ``create_run`` payload in the auth handler is the documented
    way to inject server-side values; shapes are checked defensively so a
    payload without ``kwargs.config`` is simply left alone (the
    ``resolve_user_id`` chain still prefers the injected
    ``langgraph_auth_user_id`` as belt-and-braces).
    """
    kwargs = value.get("kwargs")
    if not isinstance(kwargs, dict):
        return
    config = kwargs.setdefault("config", {})
    if not isinstance(config, dict):
        return
    configurable = config.setdefault("configurable", {})
    if isinstance(configurable, dict):
        configurable["user_id"] = identity


@auth.on.threads
async def scope_threads(
    ctx: Auth.types.AuthContext, value: dict[str, Any]
) -> dict[str, str] | bool | None:
    """Read-shared, write-authenticated threads.

    Muffin's brand is *research shared by everyone* and the perimeter is
    Cloudflare Access, so read-style actions (``read`` / ``search``) are open
    to every caller — anonymous callers see all runs, including other users'.
    **Starting new work requires sign-in**: ``create`` / ``create_run`` from an
    anonymous caller are rejected (403). A signed-in create stamps
    ``metadata.owner`` for attribution and forces the run's
    ``configurable.user_id`` to the verified identity. ``update`` / ``delete``
    require ownership. Returning ``None`` = allow with no scoping (auth fully
    disabled, read actions, or the trusted shared-token client); ``False`` =
    reject.
    """
    identity = ctx.user.identity
    if not _AUTH_ENABLED or identity in _SCOPE_EXEMPT_IDENTITIES:
        return None

    action = ctx.action
    if action in ("read", "search"):
        return None

    if action in ("create", "create_run"):
        # Reads are open to all, but creating a thread / starting a run is
        # authenticated-only — anonymous callers are read-only.
        if identity == "anonymous":
            return False
        metadata = value.setdefault("metadata", {})
        metadata.setdefault("owner", identity)
        if action == "create_run":
            _force_verified_user_id(value, identity)
        return None

    # update / delete (and anything new): owner-only.
    return {"owner": identity}
