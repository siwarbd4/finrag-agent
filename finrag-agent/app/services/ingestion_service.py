"""
Document Ingestion Service
Orchestrates the full PDF → Extract → Embed → Store pipeline
"""
import uuid
from pathlib import Path

from fastapi import UploadFile
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import Document
from app.services.pdf_service import PDFService
from app.services.vector_store import VectorStoreService


# Financial document type detection keywords
DOC_TYPE_KEYWORDS = {
    "rapport_annuel": ["rapport annuel", "annual report", "bilan annuel", "exercice"],
    "prospectus": ["prospectus", "offre publique", "introduction en bourse", "ipo"],
    "fiche_fonds": ["fiche fonds", "fund factsheet", "performance du fonds", "ucits", "opcvm"],
    "etats_financiers": ["états financiers", "financial statements", "bilan", "compte de résultat"],
    "rapport_trimestriel": ["rapport trimestriel", "quarterly report", "résultats trimestriels"],
    "note_information": ["note d'information", "information notice", "document de référence"],
}


def detect_doc_type(text: str) -> str:
    """Detect financial document type from content."""
    text_lower = text[:3000].lower()
    for doc_type, keywords in DOC_TYPE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return doc_type
    return "document_financier"


class IngestionService:
    """Orchestrates the complete document ingestion pipeline."""

    def __init__(self, db: AsyncSession, vector_service: VectorStoreService):
        self.db = db
        self.vector_service = vector_service
        self.pdf_service = PDFService()

    async def ingest_pdf(self, upload_file: UploadFile) -> Document:
        """
        Full ingestion pipeline:
        1. Save PDF to disk
        2. Create DB record
        3. Extract text
        4. Split into chunks
        5. Generate & store embeddings
        6. Update DB record
        """
        # 1. Save file to disk
        safe_filename = self._safe_filename(upload_file.filename)
        file_path = settings.pdf_upload_path / safe_filename

        logger.info(f"Saving uploaded file: {safe_filename}")
        content = await upload_file.read()
        file_size = len(content)

        with open(file_path, "wb") as f:
            f.write(content)

        # 2. Create DB record (status: pending)
        doc = Document(
            filename=safe_filename,
            original_filename=upload_file.filename,
            file_path=str(file_path),
            file_size=file_size,
            status="pending",
        )
        self.db.add(doc)
        await self.db.flush()   # Get the auto-generated ID
        await self.db.commit()
        await self.db.refresh(doc)
        doc_id = doc.id
        logger.info(f"Created document record ID={doc_id}")

        try:
            # 3. Extract text
            extraction = await self.pdf_service.extract(file_path)
            doc_type = detect_doc_type(extraction.text)

            # 4. Split into chunks
            chunks = self.pdf_service.split_into_chunks(
                extraction, doc_id, safe_filename
            )

            # Update doc record with extracted info
            doc.page_count = extraction.page_count
            doc.doc_type = doc_type
            doc.status = "indexing"
            await self.db.commit()

            # 5. Generate embeddings & store in ChromaDB
            added = await self.vector_service.add_chunks(chunks)

            # 6. Mark as indexed
            doc.chunk_count = added
            doc.status = "indexed"
            await self.db.commit()
            await self.db.refresh(doc)

            logger.info(
                f"Document {doc_id} indexed successfully: "
                f"{extraction.page_count} pages, {added} chunks, type={doc_type}"
            )
            return doc

        except Exception as e:
            logger.error(f"Ingestion failed for document {doc_id}: {e}")
            doc.status = "error"
            doc.error_message = str(e)
            await self.db.commit()
            raise

    @staticmethod
    def _safe_filename(original: str) -> str:
        """Generate a safe, unique filename to avoid collisions."""
        suffix = Path(original).suffix.lower() or ".pdf"
        stem = Path(original).stem
        # Sanitize stem
        safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
        uid = uuid.uuid4().hex[:8]
        return f"{safe_stem}_{uid}{suffix}"
