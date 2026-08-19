"""Safe request tracing and structured event helpers."""

from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger("ai_workflow_studio.rag_api")


def log_event(
    event: str,
    *,
    request_id: str,
    **fields: Any,
) -> None:
    """Log structured metadata without prompt or answer bodies."""

    safe_fields = {
        key: value
        for key, value in fields.items()
        if key not in {"question", "prompt", "answer", "context"}
    }

    logger.info(
        event,
        extra={
            "event": event,
            "request_id": request_id,
            **safe_fields,
        },
    )
