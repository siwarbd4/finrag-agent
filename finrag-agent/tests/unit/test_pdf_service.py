"""
Unit tests for the PDF extraction service
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.services.pdf_service import PDFService, PDFExtractionResult


class TestPDFService:
    """Tests for PDFService."""

    def setup_method(self):
        self.service = PDFService()

    def test_clean_text_removes_extra_whitespace(self):
        dirty = "Total  actif   net   :    1 234 567"
        cleaned = self.service._clean_text(dirty)
        assert "  " not in cleaned

    def test_clean_text_handles_empty(self):
        assert self.service._clean_text("") == ""
        assert self.service._clean_text(None) == ""

    def test_make_chunk_id_format(self):
        chunk_id = PDFService._make_chunk_id(42, 7)
        assert chunk_id == "doc_42_chunk_7"

    def test_get_file_hash_is_deterministic(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-1.4 test content")
        hash1 = PDFService.get_file_hash(f)
        hash2 = PDFService.get_file_hash(f)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex

    def test_split_into_chunks_basic(self):
        extraction = PDFExtractionResult(
            text="A" * 3000,
            page_count=2,
            metadata={},
            pages=[{"page_num": 1, "text": "A" * 1500, "char_count": 1500},
                   {"page_num": 2, "text": "A" * 1500, "char_count": 1500}],
        )
        chunks = self.service.split_into_chunks(extraction, doc_id=1, filename="test.pdf")
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk["document_id"] == 1
            assert chunk["filename"] == "test.pdf"
            assert "chunk_id" in chunk
            assert len(chunk["content"]) <= 1200  # chunk_size + some tolerance

    def test_tables_to_text_with_data(self):
        tables = [
            [["Actif", "Montant"], ["Trésorerie", "100 000"], ["Immobilisations", "500 000"]],
        ]
        result = self.service._tables_to_text(tables)
        assert "Actif" in result
        assert "Trésorerie" in result
        assert "100 000" in result

    def test_tables_to_text_empty(self):
        assert self.service._tables_to_text([]) == ""
        assert self.service._tables_to_text(None) == ""


class TestDocTypeDetection:
    """Tests for document type auto-detection."""

    def test_detect_rapport_annuel(self):
        from app.services.ingestion_service import detect_doc_type
        text = "Rapport Annuel 2023 - Exercice clos le 31 décembre 2023"
        assert detect_doc_type(text) == "rapport_annuel"

    def test_detect_prospectus(self):
        from app.services.ingestion_service import detect_doc_type
        text = "Prospectus d'émission d'obligations - Offre publique d'achat"
        assert detect_doc_type(text) == "prospectus"

    def test_detect_fiche_fonds(self):
        from app.services.ingestion_service import detect_doc_type
        text = "Fiche Fonds - Performance du fonds UCITS - OPCVM"
        assert detect_doc_type(text) == "fiche_fonds"

    def test_detect_unknown_returns_default(self):
        from app.services.ingestion_service import detect_doc_type
        text = "Document sans classification claire"
        assert detect_doc_type(text) == "document_financier"
