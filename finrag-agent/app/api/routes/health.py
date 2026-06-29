"""
Health Check Routes
GET /health - System health status
"""
from fastapi import APIRouter, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.models.schemas import HealthResponse, OllamaStatus, VectorStoreStatus
from app.services.ollama_service import OllamaService

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
)
async def health_check(request: Request):
    """
    Check the health of all system components:
    - Ollama LLM service
    - ChromaDB vector store
    - SQLite database
    """
    # Check Ollama
    ollama_service = OllamaService()
    ollama_available, ollama_error = await ollama_service.is_available()
    models = await ollama_service.list_models() if ollama_available else []

    ollama_status = OllamaStatus(
        available=ollama_available,
        model=settings.OLLAMA_MODEL,
        embed_model=settings.OLLAMA_EMBED_MODEL,
        error=ollama_error,
    )

    # Check Vector Store
    vector_service = request.app.state.vector_service
    vs_stats = vector_service.get_collection_stats()
    vector_status = VectorStoreStatus(
        available=vs_stats.get("available", False),
        collection=settings.CHROMA_COLLECTION_NAME,
        document_count=vs_stats.get("document_count", 0),
        error=vs_stats.get("error"),
    )

    # Check Database
    db_status = {"available": False, "error": None}
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        db_status["available"] = True
    except Exception as e:
        db_status["error"] = str(e)

    # Overall status
    if ollama_available and vector_status.available and db_status["available"]:
        overall = "healthy"
    elif db_status["available"] and vector_status.available:
        overall = "degraded"  # DB and vectors OK but Ollama down
    else:
        overall = "unhealthy"

    return HealthResponse(
        status=overall,
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
        ollama=ollama_status,
        vector_store=vector_status,
        database=db_status,
    )


@router.get("/", summary="Root endpoint")
async def root():
    """FinRAG Agent root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "Financial Document RAG Agent",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
