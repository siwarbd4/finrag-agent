"""
Vector Store Service using ChromaDB + sentence-transformers
Handles embeddings generation and semantic search
"""
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.models.schemas import SourceChunk


class VectorStoreService:
    """Manages the ChromaDB vector store for semantic search."""

    def __init__(self):
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection = None
        self.embed_model: Optional[SentenceTransformer] = None
        self._initialized = False

    async def initialize(self):
        """Initialize ChromaDB client and embedding model."""
        logger.info("Initializing vector store...")

        # Load embedding model
        # Using multilingual model for French/English financial docs
        model_name = "paraphrase-multilingual-mpnet-base-v2"
        logger.info(f"Loading embedding model: {model_name}")
        self.embed_model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded")

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        count = self.collection.count()
        logger.info(
            f"Vector collection '{settings.CHROMA_COLLECTION_NAME}' ready. "
            f"Current chunks: {count}"
        )
        self._initialized = True

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        if not self.embed_model:
            raise RuntimeError("Embedding model not initialized")
        embeddings = self.embed_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=len(texts) > 10,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    async def add_chunks(self, chunks: list[dict]) -> int:
        """
        Add document chunks to the vector store.
        Returns number of chunks successfully added.
        """
        if not self._initialized:
            raise RuntimeError("Vector store not initialized")

        if not chunks:
            return 0

        logger.info(f"Embedding {len(chunks)} chunks...")

        texts = [c["content"] for c in chunks]
        ids = [c["chunk_id"] for c in chunks]
        metadatas = [
            {
                "document_id": str(c["document_id"]),
                "filename": c["filename"],
                "page_num": str(c.get("page_num") or ""),
                "chunk_index": str(c["chunk_index"]),
                "char_count": str(c["char_count"]),
            }
            for c in chunks
        ]

        # Generate embeddings
        embeddings = self._embed(texts)

        # Add to ChromaDB in batches
        batch_size = 100
        added = 0
        for i in range(0, len(chunks), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]
            batch_metas = metadatas[i : i + batch_size]

            self.collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metas,
            )
            added += len(batch_ids)

        logger.info(f"Added {added} chunks to vector store")
        return added

    async def search(
        self,
        query: str,
        top_k: int = None,
        doc_ids: Optional[list[int]] = None,
    ) -> list[SourceChunk]:
        """
        Semantic search: find most relevant chunks for a query.
        Optionally filter by document IDs.
        """
        if not self._initialized:
            raise RuntimeError("Vector store not initialized")

        top_k = top_k or settings.TOP_K_RESULTS

        # Build where filter for document IDs
        where_filter = None
        if doc_ids:
            if len(doc_ids) == 1:
                where_filter = {"document_id": str(doc_ids[0])}
            else:
                where_filter = {
                    "$or": [{"document_id": str(did)} for did in doc_ids]
                }

        # Embed the query
        query_embedding = self._embed([query])[0]

        # Query ChromaDB
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        results = self.collection.query(**query_kwargs)

        # Parse results
        source_chunks = []
        if not results["ids"] or not results["ids"][0]:
            return source_chunks

        for idx, (chunk_id, doc, meta, dist) in enumerate(
            zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ):
            # Cosine distance → similarity score (1 - distance for cosine)
            similarity = 1.0 - dist

            if similarity < settings.SIMILARITY_THRESHOLD:
                continue

            source_chunks.append(
                SourceChunk(
                    document_id=int(meta.get("document_id", 0)),
                    document_name=meta.get("filename", "Unknown"),
                    page=int(meta["page_num"]) if meta.get("page_num") else None,
                    chunk_index=int(meta.get("chunk_index", idx)),
                    content=doc,
                    similarity_score=round(similarity, 4),
                )
            )

        logger.info(
            f"Search returned {len(source_chunks)} relevant chunks "
            f"(threshold: {settings.SIMILARITY_THRESHOLD})"
        )
        return source_chunks

    async def delete_document_chunks(self, document_id: int) -> int:
        """Remove all chunks belonging to a document."""
        if not self._initialized:
            raise RuntimeError("Vector store not initialized")

        # Find all chunk IDs for this document
        results = self.collection.get(
            where={"document_id": str(document_id)},
            include=[],
        )
        chunk_ids = results.get("ids", [])
        if chunk_ids:
            self.collection.delete(ids=chunk_ids)
            logger.info(f"Deleted {len(chunk_ids)} chunks for document {document_id}")

        return len(chunk_ids)

    def get_collection_stats(self) -> dict:
        """Return statistics about the vector collection."""
        if not self._initialized:
            return {"available": False, "error": "Not initialized"}
        return {
            "available": True,
            "collection": settings.CHROMA_COLLECTION_NAME,
            "document_count": self.collection.count(),
        }
