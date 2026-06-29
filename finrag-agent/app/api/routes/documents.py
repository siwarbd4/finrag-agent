"""
Document Management API Routes
POST /upload, GET /, GET /{id}, DELETE /{id}
"""
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import Document, get_db
from app.models.schemas import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
)
from app.services.ingestion_service import IngestionService

router = APIRouter()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a financial PDF",
)
async def upload_document(
    request: Request,
    file: UploadFile = File(..., description="PDF file to upload and index"),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a financial PDF document and start the indexing pipeline:
    1. Save file to disk
    2. Extract text (pdfplumber / pypdf)
    3. Split into chunks
    4. Generate embeddings
    5. Store in ChromaDB vector database
    """
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported",
        )

    # Validate file size (read Content-Length header or check after read)
    if file.size and file.size > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE_MB}MB",
        )

    vector_service = request.app.state.vector_service
    ingestion = IngestionService(db=db, vector_service=vector_service)

    try:
        doc = await ingestion.ingest_pdf(file)
        return DocumentUploadResponse(
            document_id=doc.id,
            filename=doc.filename,
            original_filename=doc.original_filename,
            file_size=doc.file_size,
            status=doc.status,
            message=(
                f"Document indexed successfully: "
                f"{doc.page_count} pages, {doc.chunk_count} chunks"
            ),
        )
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {str(e)}",
        )


@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List all indexed documents",
)
async def list_documents(
    db: AsyncSession = Depends(get_db),
    status_filter: str = None,
):
    """Return all indexed documents with their metadata."""
    stmt = select(Document).order_by(Document.created_at.desc())
    if status_filter:
        stmt = stmt.where(Document.status == status_filter)

    result = await db.execute(stmt)
    docs = result.scalars().all()

    return DocumentListResponse(
        total=len(docs),
        documents=[DocumentStatusResponse(**d.to_dict()) for d in docs],
    )


@router.get(
    "/{document_id}",
    response_model=DocumentStatusResponse,
    summary="Get document details",
)
async def get_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get metadata and indexing status for a specific document."""
    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )
    return DocumentStatusResponse(**doc.to_dict())


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete a document and its embeddings",
)
async def delete_document(
    document_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Delete a document from the database and remove its vectors from ChromaDB."""
    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    # Remove from vector store
    vector_service = request.app.state.vector_service
    deleted_chunks = await vector_service.delete_document_chunks(document_id)

    # Remove from DB
    await db.delete(doc)
    await db.commit()

    return DocumentDeleteResponse(
        document_id=document_id,
        message=f"Document deleted. Removed {deleted_chunks} chunks from vector store.",
    )
