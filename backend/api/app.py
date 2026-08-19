"""FastAPI application factory for the production-oriented RAG API."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from uuid import uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .auth import require_api_key
from .errors import (
    DependencyUnavailableError,
    RAGServiceError,
)
from .observability import log_event
from .schemas import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    HealthResponse,
    ReadinessResponse,
)
from .service import RAGService, ReadinessCheck


RequestIdFactory = Callable[[], str]
Clock = Callable[[], float]


def _request_id(request: Request) -> str:
    return getattr(
        request.state,
        "request_id",
        "unknown",
    )


def create_app(
    *,
    rag_service: RAGService,
    readiness: ReadinessCheck | None = None,
    auth_dependency=Depends(require_api_key),
    request_id_factory: RequestIdFactory = lambda: str(uuid4()),
    clock: Clock = perf_counter,
) -> FastAPI:
    app = FastAPI(
        title="AI Workflow Studio RAG API",
        version="1.0.0",
    )

    @app.middleware("http")
    async def request_tracing(
        request: Request,
        call_next,
    ):
        request_id = request_id_factory()
        request.state.request_id = request_id
        started = clock()

        log_event(
            "rag_request_started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        response = await call_next(request)

        elapsed_ms = (clock() - started) * 1000.0
        response.headers["X-Request-ID"] = request_id

        log_event(
            "rag_request_completed",
            request_id=request_id,
            status_code=response.status_code,
            total_latency_ms=elapsed_ms,
            method=request.method,
            path=request.url.path,
        )

        return response

    @app.exception_handler(RAGServiceError)
    async def rag_service_error_handler(
        request: Request,
        exc: RAGServiceError,
    ):
        request_id = _request_id(request)

        log_event(
            "rag_request_failed",
            request_id=request_id,
            error_code=exc.error_code,
            error_type=type(exc).__name__,
        )

        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=exc.error_code,
                message=exc.public_message,
                request_id=request_id,
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ):
        request_id = _request_id(request)

        log_event(
            "rag_request_rejected",
            request_id=request_id,
            error_code="INVALID_REQUEST",
            error_type=type(exc).__name__,
        )

        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="INVALID_REQUEST",
                message="The request body is invalid.",
                request_id=request_id,
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(
        request: Request,
        exc: Exception,
    ):
        request_id = _request_id(request)

        # Do not return exception text to the client.
        log_event(
            "rag_request_failed",
            request_id=request_id,
            error_code="INTERNAL_ERROR",
            error_type=type(exc).__name__,
        )

        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="INTERNAL_ERROR",
                message="An unexpected error occurred.",
                request_id=request_id,
            ).model_dump(),
        )

    @app.get(
        "/health",
        response_model=HealthResponse,
    )
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get(
        "/ready",
        response_model=ReadinessResponse,
    )
    async def ready() -> ReadinessResponse:
        if readiness is not None and not readiness.is_ready():
            raise DependencyUnavailableError()

        return ReadinessResponse(status="ready")

    @app.post(
        "/api/v1/ask",
        response_model=AskResponse,
        responses={
            401: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        dependencies=[auth_dependency],
    )
    async def ask(
        request: AskRequest,
        raw_request: Request,
    ) -> AskResponse:
        request_id = _request_id(raw_request)

        # HTTP concerns stop here. Retrieval, generation, and evaluation live
        # behind the service boundary.
        result = rag_service.answer(
            request.question,
            request_id=request_id,
        )

        return AskResponse(
            request_id=request_id,
            answer=result.answer,
        )

    return app
