"""AI Workflow Studio — FastAPI application entry point.

Run locally:
    uvicorn main:app --reload --port 8000

Interactive docs:
    http://localhost:8000/docs
"""

from fastapi import FastAPI

from app.api.health import router as health_router
from app.config.settings import settings

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "RAG-powered document Q&A platform built on NVIDIA NIM endpoints. "
        "See backend/README.md for architecture details."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health_router)

# Future routers will be registered here:
#   app.include_router(auth_router,      prefix="/auth")
#   app.include_router(documents_router, prefix="/documents")
#   app.include_router(retrieval_router, prefix="/retrieval")
#   app.include_router(generation_router,prefix="/generation")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
