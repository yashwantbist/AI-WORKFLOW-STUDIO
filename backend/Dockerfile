# syntax=docker/dockerfile:1
# ──────────────────────────────────────────────────────────────────────────────
# AI Workflow Studio — Backend Dockerfile
#
# Multi-stage build:
#   1. builder  — install Python dependencies into a virtual environment
#   2. runtime  — copy the venv into a slim image, no build tools included
# ──────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies for wheels that require compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment so the runtime stage can copy just the venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Don't write .pyc files; don't buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy the virtualenv from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application source
COPY main.py .
COPY app/ ./app/

# Expose the port uvicorn will listen on
EXPOSE 8000

# Run as a non-root user for security
RUN adduser --system --no-create-home appuser
USER appuser

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
