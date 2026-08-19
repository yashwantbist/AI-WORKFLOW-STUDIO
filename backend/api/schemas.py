"""HTTP request/response schemas for the production RAG API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Natural-language question for the RAG service.",
    )


class AskResponse(BaseModel):
    request_id: str
    answer: str


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str
