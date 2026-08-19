from dataclasses import dataclass

import pytest
from fastapi import Depends, HTTPException
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.api.errors import (
    DependencyUnavailableError,
    RateLimitError,
)
from backend.api.service import RAGServiceResult


class FakeRAGService:
    def __init__(self, *, answer="Grounded answer.", error=None):
        self.answer_text = answer
        self.error = error
        self.calls = []

    def answer(self, question, *, request_id):
        self.calls.append(
            {
                "question": question,
                "request_id": request_id,
            }
        )

        if self.error is not None:
            raise self.error

        return RAGServiceResult(
            answer=self.answer_text,
        )


class FakeReadiness:
    def __init__(self, ready=True):
        self.ready = ready

    def is_ready(self):
        return self.ready


def allow_auth():
    return None


def deny_auth():
    raise HTTPException(
        status_code=401,
        detail="Authentication is required.",
    )


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        current = self.value
        self.value += 0.05
        return current


def client_for(
    service=None,
    *,
    readiness=None,
    auth=allow_auth,
):
    service = service or FakeRAGService()

    app = create_app(
        rag_service=service,
        readiness=readiness,
        auth_dependency=Depends(auth),
        request_id_factory=lambda: "req-test-123",
        clock=FakeClock(),
    )

    return TestClient(
        app,
        raise_server_exceptions=False,
    ), service


def test_valid_request_delegates_to_service():
    client, service = client_for()

    response = client.post(
        "/api/v1/ask",
        json={"question": "How does RAG work?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "request_id": "req-test-123",
        "answer": "Grounded answer.",
    }

    assert service.calls == [
        {
            "question": "How does RAG work?",
            "request_id": "req-test-123",
        }
    ]


def test_request_id_exists_in_body_and_header():
    client, _ = client_for()

    response = client.post(
        "/api/v1/ask",
        json={"question": "Question"},
    )

    assert response.json()["request_id"] == "req-test-123"
    assert response.headers["X-Request-ID"] == "req-test-123"


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "x" * 2001},
        {},
    ],
)
def test_invalid_question_returns_safe_validation_error(payload):
    client, service = client_for()

    response = client.post(
        "/api/v1/ask",
        json=payload,
    )

    assert response.status_code == 422
    assert response.json() == {
        "error": "INVALID_REQUEST",
        "message": "The request body is invalid.",
        "request_id": "req-test-123",
    }
    assert service.calls == []


def test_unauthenticated_request_is_rejected_before_service():
    client, service = client_for(auth=deny_auth)

    response = client.post(
        "/api/v1/ask",
        json={"question": "Question"},
    )

    assert response.status_code == 401
    assert service.calls == []


def test_dependency_failure_returns_safe_503():
    service = FakeRAGService(
        error=DependencyUnavailableError(
            "secret-provider-detail"
        )
    )
    client, _ = client_for(service)

    response = client.post(
        "/api/v1/ask",
        json={"question": "Question"},
    )

    assert response.status_code == 503

    body = response.json()

    assert body["error"] == "AI_SERVICE_UNAVAILABLE"
    assert body["message"] == (
        "The AI service is temporarily unavailable."
    )
    assert body["request_id"] == "req-test-123"
    assert "secret-provider-detail" not in response.text


def test_rate_limit_maps_to_429():
    client, _ = client_for(
        FakeRAGService(
            error=RateLimitError(
                "provider quota details"
            )
        )
    )

    response = client.post(
        "/api/v1/ask",
        json={"question": "Question"},
    )

    assert response.status_code == 429
    assert response.json()["error"] == "RATE_LIMITED"
    assert "provider quota details" not in response.text


def test_unexpected_bug_returns_safe_500_without_exception_text():
    client, _ = client_for(
        FakeRAGService(
            error=RuntimeError(
                "database-password=super-secret"
            )
        )
    )

    response = client.post(
        "/api/v1/ask",
        json={"question": "Question"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "error": "INTERNAL_ERROR",
        "message": "An unexpected error occurred.",
        "request_id": "req-test-123",
    }
    assert "super-secret" not in response.text
    assert "RuntimeError" not in response.text


def test_health_is_liveness_only_and_does_not_call_rag_service():
    client, service = client_for()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert service.calls == []


def test_ready_succeeds_when_dependencies_are_ready():
    client, _ = client_for(
        readiness=FakeReadiness(True)
    )

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_ready_returns_503_when_dependency_is_not_ready():
    client, _ = client_for(
        readiness=FakeReadiness(False)
    )

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["error"] == "AI_SERVICE_UNAVAILABLE"
    assert response.json()["request_id"] == "req-test-123"


def test_health_and_readiness_do_not_require_authentication():
    app = create_app(
        rag_service=FakeRAGService(),
        readiness=FakeReadiness(True),
        auth_dependency=Depends(deny_auth),
        request_id_factory=lambda: "req-test-123",
        clock=FakeClock(),
    )
    client = TestClient(
        app,
        raise_server_exceptions=False,
    )

    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200
    assert client.post(
        "/api/v1/ask",
        json={"question": "Question"},
    ).status_code == 401
