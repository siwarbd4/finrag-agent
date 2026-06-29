"""
FinRAG Agent - Financial Document RAG System
Main FastAPI Application Entry Point
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import documents, query, health
from app.core.config import settings
from app.core.database import init_db
from app.services.vector_store import VectorStoreService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - startup and shutdown events."""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.APP_ENV}")

    # Initialize database
    await init_db()
    logger.info("Database initialized successfully")

    # Initialize vector store
    vector_service = VectorStoreService()
    await vector_service.initialize()
    logger.info("Vector store initialized successfully")

    # Store services in app state
    app.state.vector_service = vector_service

    logger.info(f"Ollama endpoint: {settings.OLLAMA_BASE_URL}")
    logger.info(f"Using model: {settings.OLLAMA_MODEL}")
    logger.info("FinRAG Agent is ready to serve requests!")

    yield

    # Shutdown
    logger.info("Shutting down FinRAG Agent...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="""
        ## FinRAG Agent - Financial Document AI Assistant

        A RAG (Retrieval-Augmented Generation) system specialized for financial documents.

        ### Features
        - 📄 **PDF Ingestion**: Upload and index financial PDFs (reports, prospectus, fund sheets, etc.)
        - 🔍 **Semantic Search**: Find relevant passages using vector similarity
        - 🤖 **AI Q&A**: Ask questions in natural language, get answers grounded in your documents
        - 🏦 **Financial Focus**: Optimized for financial terminology and document structures

        ### Workflow
        1. Upload PDFs via `/api/v1/documents/upload`
        2. Ask questions via `/api/v1/query`
        3. Get AI-powered answers based solely on your indexed documents
        """,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(health.router, prefix="/api/v1", tags=["Health"])
    app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
    app.include_router(query.router, prefix="/api/v1/query", tags=["Query"])

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
