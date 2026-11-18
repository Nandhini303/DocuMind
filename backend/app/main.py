"""
DocuMind AI — Main FastAPI Application
---------------------------------------
Modular application setup with clean architecture, dependency injection,
static file serving, and security middleware.
"""

import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.security import SecurityHeadersMiddleware
from app.core.exceptions import DocuMindError
from app.api.routes import health_router, analyze_router

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("documind.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown routines."""
    logger.info("Initializing %s v%s in %s mode", settings.APP_NAME, settings.APP_VERSION, settings.ENVIRONMENT)
    logger.info("Active LLM Provider: %s (Model: %s)", settings.active_provider, settings.active_model)
    if not settings.is_api_key_configured:
        logger.warning(
            "WARNING: No LLM API key detected. Set ANTHROPIC_API_KEY as an environment variable."
        )
    yield
    logger.info("Shutting down %s", settings.APP_NAME)


def create_app() -> FastAPI:
    """Factory creating and configuring the FastAPI application."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        lifespan=lifespan,
    )

    # Security Headers Middleware
    app.add_middleware(SecurityHeadersMiddleware)

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static files directory
    base_dir = Path(__file__).resolve().parent.parent
    static_dir = base_dir / "static"

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Global Exception Handler for domain errors
    @app.exception_handler(DocuMindError)
    async def documind_exception_handler(request: Request, exc: DocuMindError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    # Mount API routers
    app.include_router(health_router)
    app.include_router(analyze_router)

    # Serve static frontend root
    @app.get("/", include_in_schema=False)
    async def serve_index():
        index_file = static_dir / "index.html"
        if not index_file.exists():
            return JSONResponse(
                status_code=404,
                content={"detail": "Frontend index.html not found."},
            )
        return FileResponse(str(index_file))

    # Favicon endpoint to silence browser 404 noise
    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        favicon_file = static_dir / "favicon.ico"
        if favicon_file.exists():
            return FileResponse(str(favicon_file))
        return Response(status_code=204)

    return app


app = create_app()
