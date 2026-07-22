"""Health-check endpoint.

Available without authentication so that load balancers and CI pipelines
can confirm the service is running without needing credentials.
"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str


@router.get("/health", response_model=HealthResponse, summary="Health check")
def health() -> HealthResponse:
    """Return ``{"status": "ok"}`` when the service is running."""
    return HealthResponse(status="ok")
