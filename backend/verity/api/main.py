"""Verity FastAPI application.

Creates the FastAPI app, configures middleware, and includes all route
modules. The backend owns authorization, data validation, retrieval, and
model-provider access — the frontend never talks to NVIDIA directly.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from verity.api.documents import router as documents_router
from verity.api.sessions import router as sessions_router
from verity.api.ingestion import router as ingestion_router

# The provider keys live in .env at the repo root. Loading here (module
# import, before any route runs) means uvicorn needs no shell profile
# wrapper, and the backends in verity.providers can read os.environ at
# construction. override=False: explicit environment wins over .env.
_REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_REPO_ROOT / ".env", override=False)


def create_app() -> FastAPI:
    """Build the FastAPI application.

    Configures CORS so the frontend (served separately) can call the API,
    includes all route routers, and sets up database connection pooling.
    """
    app = FastAPI(
        title="Verity",
        description="Ask your documents. Answers with verifiable citations.",
        version="0.2.0",
    )

    # CORS: the frontend calls the backend API. The NVIDIA API key never
    # reaches the browser — the backend proxy is a security requirement.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health", tags=["health"])
    async def health() -> dict[str, str]:
        """Health check. Returns ok when the API is running."""
        return {"status": "ok"}

    app.include_router(documents_router)
    app.include_router(sessions_router)
    app.include_router(ingestion_router)

    return app


app = create_app()
