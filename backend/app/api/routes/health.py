"""
Health check and telemetry endpoint.
"""

from fastapi import APIRouter
from app.core.config import settings
from app.models.schemas import HealthResponse

router = APIRouter(prefix="/api", tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Returns the server health status and API configuration state.
    Preserves exact contract expected by frontend static/index.html.
    """
    return HealthResponse(
        status="ok",
        model=settings.active_model,
        api_key_configured=settings.is_api_key_configured,
        provider=settings.active_provider,
        version=settings.APP_VERSION,
        rag_enabled=True,
    )
