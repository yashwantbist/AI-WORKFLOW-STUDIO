"""Operational error types exposed safely through the API layer."""

from __future__ import annotations


class RAGServiceError(Exception):
    error_code = "RAG_SERVICE_ERROR"
    status_code = 500
    public_message = "The request could not be completed."


class AuthenticationError(RAGServiceError):
    error_code = "AUTHENTICATION_REQUIRED"
    status_code = 401
    public_message = "Authentication is required."


class RateLimitError(RAGServiceError):
    error_code = "RATE_LIMITED"
    status_code = 429
    public_message = "Too many requests. Please try again later."


class DependencyUnavailableError(RAGServiceError):
    error_code = "AI_SERVICE_UNAVAILABLE"
    status_code = 503
    public_message = "The AI service is temporarily unavailable."
