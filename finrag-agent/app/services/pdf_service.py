"""
PDF Processing Service
Handles PDF extraction, text cleaning, and chunking
"""
import hashlib
import re
from pathlib import Path
from typing import Optional

import pdfplumber
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from app.core.config import settings


class PDFExtractionResult:
    """Result of PDF text extraction."""

    def __init__(
        self,
        text: str,
        page_count: int,
        metadata: dict,
        pages: list[dict],
    ):
        self.text = text
        self.page_count = page_count
        self.metadata = metadata
        self.pages = pages  # [{page_num, text, char_count}]


class PDFService:
    """Service for extracting and processing PDF documents."""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    async def extract(self, file_path: str | Path) -> PDFExtractionResult:
        """
        Extract text from a PDF file.
        Tries pdfplumber first (better for tables/financial docs),
        falls back to pypdf for scanned/encrypted files.
        """
        file_path = Path(file_path)
        logger.info(f"Extracting text from: {file_path.name}")

        try:
            result = self._extract_with_pdfplumber(file_path)
            if result and len(result.text.strip()) > 100:
                logger.info(
                    f"pdfplumber extracted {len(result.text)} chars, "
                    f"{result.page_count} pages"
                )
                return result
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}, trying pypdf...")

        # Fallback to pypdf
        return self._extract_with_pypdf(file_path)

    def _extract_with_pdfplumber(self, file_path: Path) -> PDFExtractionResult:
        """Extract using pdfplumber (better for financial tables)."""
        pages = []
        full_text_parts = []

        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            metadata = {
                "title": pdf.metadata.get("Title", ""),
                "author": pdf.metadata.get("Author", ""),
                "creator": pdf.metadata.get("Creator", ""),
                "subject": pdf.metadata.get("Subject", ""),
                "producer": pdf.metadata.get("Producer", ""),
            }

            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract text
                text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""

                # Extract tables and convert to text
                tables = page.extract_tables()
                table_text = self._tables_to_text(tables)

                page_text = text
                if table_text:
                    page_text += f"\n\n[TABLEAU PAGE {page_num}]\n{table_text}"

                cleaned = self._clean_text(page_text)
                pages.append({
                    "page_num": page_num,
                    "text": cleaned,
                    "char_count": len(cleaned),
                })
                if cleaned:
                    full_text_parts.append(f"--- PAGE {page_num} ---\n{cleaned}")

        return PDFExtractionResult(
            text="\n\n".join(full_text_parts),
            page_count=page_count,
            metadata=metadata,
            pages=pages,
        )

    def _extract_with_pypdf(self, file_path: Path) -> PDFExtractionResult:
        """Fallback extraction using pypdf."""
        pages = []
        full_text_parts = []

        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            page_count = len(reader.pages)
            info = reader.metadata or {}
            metadata = {
                "title": info.get("/Title", ""),
                "author": info.get("/Author", ""),
                "creator": info.get("/Creator", ""),
                "subject": info.get("/Subject", ""),
            }

            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                cleaned = self._clean_text(text)
                pages.append({
                    "page_num": page_num,
                    "text": cleaned,
                    "char_count": len(cleaned),
                })
                if cleaned:
                    full_text_parts.append(f"--- PAGE {page_num} ---\n{cleaned}")

        return PDFExtractionResult(
            text="\n\n".join(full_text_parts),
            page_count=page_count,
            metadata=metadata,
            pages=pages,
        )

    def _tables_to_text(self, tables: list) -> str:
        """Convert extracted table data to readable text."""
        if not tables:
            return ""
        result = []
        for table in tables:
            if not table:
                continue
            rows = []
            for row in table:
                if row:
                    cells = [str(c).strip() if c is not None else "" for c in row]
                    rows.append(" | ".join(cells))
            if rows:
                result.append("\n".join(rows))
        return "\n\n".join(result)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove form feeds and special chars
        text = re.sub(r"[\f\r]", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Fix common PDF extraction issues
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        return text.strip()

    def split_into_chunks(
        self,
        extraction: PDFExtractionResult,
        document_id: int,
        filename: str,
    ) -> list[dict]:
        """
        Split extracted text into overlapping chunks for embedding.
        Returns list of chunk dicts with metadata.
        """
        chunks = self.text_splitter.split_text(extraction.text)
        logger.info(f"Split into {len(chunks)} chunks from {filename}")

        result = []
        for idx, chunk_text in enumerate(chunks):
            # Estimate page number from chunk position
            page_num = self._estimate_page(chunk_text, extraction.pages)

            result.append({
                "chunk_index": idx,
                "content": chunk_text,
                "document_id": document_id,
                "filename": filename,
                "page_num": page_num,
                "char_count": len(chunk_text),
                "chunk_id": self._make_chunk_id(document_id, idx),
            })

        return result

    def _estimate_page(self, chunk_text: str, pages: list[dict]) -> Optional[int]:
        """Try to find which page a chunk came from."""
        sample = chunk_text[:200]
        best_page = None
        best_ratio = 0.0
        for page in pages:
            page_words = set(page["text"].lower().split())
            chunk_words = set(sample.lower().split())
            if not chunk_words:
                continue
            overlap = len(page_words & chunk_words) / len(chunk_words)
            if overlap > best_ratio:
                best_ratio = overlap
                best_page = page["page_num"]
        return best_page

    @staticmethod
    def _make_chunk_id(document_id: int, chunk_index: int) -> str:
        """Generate a unique chunk ID."""
        return f"doc_{document_id}_chunk_{chunk_index}"

    @staticmethod
    def get_file_hash(file_path: str | Path) -> str:
        """Compute SHA-256 hash of a file for deduplication."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha256.update(block)
        return sha256.hexdigest()
