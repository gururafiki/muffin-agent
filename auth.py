"""Optional authentication for the Muffin LangGraph server.

Wired via ``langgraph.json`` (``"auth": {"path": "./auth.py:auth"}``). It gates the
default LangGraph routes (``/assistants``, ``/threads``, ``/runs``).

Three modes, auto-selected from the environment so local development keeps working:

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

Modes 2 and 3 may be enabled together (either credential is accepted).
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
_AUTH_ENABLED = bool(_API_TOKEN or (_CF_TEAM_DOMAIN and _CF_AUD))


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
