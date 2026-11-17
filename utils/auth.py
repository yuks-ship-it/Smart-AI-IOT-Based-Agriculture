"""
API Key authentication.

Every /predict/* and /health endpoint (except the root "/") requires an
X-API-Key header matching the API_KEY environment variable. This is
intentionally simple (a single shared key) rather than full OAuth/JWT,
since the typical consumers here are your own IoT devices and your own
dashboard backend, not third-party public clients. If you need per-device
keys or expiring tokens later, swap this dependency out -- every route
already depends on `verify_api_key`, so the change is centralized here.
"""
from __future__ import annotations

import os
import secrets

from fastapi import Header, HTTPException, status

from utils.logger import get_logger

logger = get_logger(__name__)

_API_KEY_ENV_VAR = "API_KEY"


def _get_configured_key() -> str | None:
    return os.environ.get(_API_KEY_ENV_VAR)


async def verify_api_key(x_api_key: str = Header(default=None, alias="X-API-Key")) -> None:
    """
    FastAPI dependency. Raises 401 if the API key is missing or wrong.

    If API_KEY is not set in the environment at all, authentication is
    skipped with a warning -- this keeps local development frictionless,
    but you MUST set API_KEY before exposing this service beyond localhost.
    """
    configured_key = _get_configured_key()

    if not configured_key:
        logger.warning(
            "API_KEY is not set in the environment. Authentication is DISABLED. "
            "Set API_KEY in your .env before deploying beyond local development."
        )
        return

    if not x_api_key or not secrets.compare_digest(x_api_key, configured_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Include it in the 'X-API-Key' header.",
        )
