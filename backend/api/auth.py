"""Authentication dependency boundary.

This module intentionally keeps authentication separate from RAG logic.
Replace the API-key dependency with the project's existing JWT/auth dependency
when that system is wired into this service.
"""

from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException, status


def require_api_key(
    x_api_key: str | None = Header(
        default=None,
        alias="X-API-Key",
    ),
) -> None:
    expected = os.getenv("RAG_API_KEY")

    # Fail closed when authentication is enabled for the production app but
    # the secret has not been configured.
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured.",
        )

    if x_api_key is None or not hmac.compare_digest(
        x_api_key,
        expected,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication is required.",
        )
