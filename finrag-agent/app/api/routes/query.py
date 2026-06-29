"""
Query API Routes
POST / - Ask a question in natural language
"""
import json
import time

from fastapi import APIRouter, Depends, HTTPException, Request, status
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import QueryLog, get_db
from app.models.schemas import QueryRequest, QueryResponse
from app.services.ollama_service import OllamaService

router = APIRouter()


@router.post(
    "/",
    response_model=QueryResponse,
    summary="Ask a question about your financial documents",
)
async def ask_question(
    request: Request,
    query: QueryRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Submit a natural language question and get an AI-generated answer
    based ONLY on the indexed financial documents.

    **Workflow:**
    1. Embed the question using the same model as documents
    2. Perform semantic search in ChromaDB
    3. Build context from retrieved chunks
    4. Send context + question to Ollama LLM
    5. Return grounded answer with source citations
    """
    start_time = time.time()

    vector_service = request.app.state.vector_service
    ollama_service = OllamaService()

    # Check Ollama availability
    available, error = await ollama_service.is_available()
    if not available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Ollama LLM not available: {error}. "
                   f"Please start Ollama and ensure model '{settings.OLLAMA_MODEL}' is pulled.",
        )

    # Semantic search
    top_k = query.top_k or settings.TOP_K_RESULTS
    logger.info(f"Searching for: '{query.question[:80]}...' (top_k={top_k})")

    chunks = await vector_service.search(
        query=query.question,
        top_k=top_k,
        doc_ids=query.doc_ids,
    )

    # Check if we have relevant content
    warning = None
    if not chunks:
        warning = (
            "Aucun document pertinent trouvé. "
            "Vérifiez que des documents financiers sont indexés "
            "et que votre question est liée à leur contenu."
        )
        logger.warning(f"No chunks found for query: {query.question[:80]}")

    # Generate answer via Ollama
    answer, gen_time_ms = await ollama_service.generate(
        question=query.question,
        chunks=chunks,
        language=query.language or "fr",
    )

    total_time_ms = (time.time() - start_time) * 1000

    # Log the query
    query_log = QueryLog(
        question=query.question,
        answer=answer,
        sources=json.dumps([c.document_id for c in chunks]),
        processing_time_ms=total_time_ms,
        model_used=settings.OLLAMA_MODEL,
        chunks_retrieved=len(chunks),
    )
    db.add(query_log)
    await db.commit()

    logger.info(
        f"Query answered in {total_time_ms:.0f}ms "
        f"({len(chunks)} chunks, model={settings.OLLAMA_MODEL})"
    )

    return QueryResponse(
        question=query.question,
        answer=answer,
        sources=chunks,
        model_used=settings.OLLAMA_MODEL,
        chunks_retrieved=len(chunks),
        processing_time_ms=round(total_time_ms, 2),
        has_answer=bool(chunks),
        warning=warning,
    )


@router.get(
    "/history",
    summary="Get recent query history",
)
async def get_query_history(
    db: AsyncSession = Depends(get_db),
    limit: int = 20,
):
    """Return the last N queries with their answers."""
    from sqlalchemy import select
    from app.core.database import QueryLog

    stmt = (
        select(QueryLog)
        .order_by(QueryLog.created_at.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    logs = result.scalars().all()

    return {
        "total": len(logs),
        "queries": [
            {
                "id": log.id,
                "question": log.question,
                "answer": log.answer[:200] + "..." if log.answer and len(log.answer) > 200 else log.answer,
                "chunks_retrieved": log.chunks_retrieved,
                "model_used": log.model_used,
                "processing_time_ms": log.processing_time_ms,
                "created_at": log.created_at.isoformat() if log.created_at else None,
            }
            for log in logs
        ],
    }
