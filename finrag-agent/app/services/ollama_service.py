"""
Ollama LLM Service
Handles communication with local Ollama instance for generation
"""
import time
from typing import AsyncIterator, Optional

import httpx
from loguru import logger

from app.core.config import settings
from app.models.schemas import QueryResponse, SourceChunk


# System prompt specialized for financial document Q&A
FINANCIAL_SYSTEM_PROMPT = """Tu es un assistant financier expert et rigoureux.
Tu réponds aux questions en te basant UNIQUEMENT sur les informations contenues dans les documents fournis.
Tu n'inventes pas de données, de chiffres ou de faits.
Si l'information n'est pas présente dans les documents, tu l'indiques clairement.
Tu cites les sources (numéro de page, nom du document) quand c'est possible.
Tu réponds de façon structurée, claire et précise, en utilisant des termes financiers appropriés.
Si la question est en français, tu réponds en français. Si elle est en anglais, tu réponds en anglais."""

FINANCIAL_SYSTEM_PROMPT_EN = """You are an expert and rigorous financial assistant.
You answer questions based ONLY on the information contained in the provided documents.
You do not invent data, figures, or facts.
If the information is not in the documents, you clearly state so.
You cite sources (page number, document name) when possible.
You respond in a structured, clear, and precise manner using appropriate financial terminology.
If the question is in French, respond in French. If in English, respond in English."""


class OllamaService:
    """Service to interact with local Ollama LLM."""

    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_MODEL
        self.timeout = httpx.Timeout(120.0, connect=10.0)

    async def is_available(self) -> tuple[bool, Optional[str]]:
        """Check if Ollama is running and the model is available."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                models = [m["name"].split(":")[0] for m in data.get("models", [])]
                if self.model.split(":")[0] not in models:
                    return False, f"Model '{self.model}' not found. Available: {models}"
                return True, None
        except Exception as e:
            return False, str(e)

    def _build_rag_prompt(
        self,
        question: str,
        chunks: list[SourceChunk],
        language: str = "fr",
    ) -> str:
        """
        Build a RAG prompt from retrieved chunks.
        Constructs a grounded context for the LLM.
        """
        if not chunks:
            return f"""Aucun document pertinent n'a été trouvé pour répondre à cette question.

Question : {question}

Réponds en expliquant qu'aucune information pertinente n'a été trouvée dans les documents indexés."""

        # Build context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            page_info = f" (page {chunk.page})" if chunk.page else ""
            doc_info = f"[Source {i}: {chunk.document_name}{page_info}]"
            context_parts.append(f"{doc_info}\n{chunk.content}")

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""CONTEXTE DOCUMENTAIRE :
{context}

---

QUESTION : {question}

INSTRUCTIONS :
- Réponds en te basant UNIQUEMENT sur le contexte documentaire fourni ci-dessus
- Si l'information n'est pas dans les documents, dis-le clairement
- Cite les sources pertinentes (nom du document, numéro de page si disponible)
- Sois précis et factuel

RÉPONSE :"""

        return prompt

    async def generate(
        self,
        question: str,
        chunks: list[SourceChunk],
        language: str = "fr",
    ) -> tuple[str, float]:
        """
        Generate an answer using Ollama with RAG context.
        Returns (answer_text, processing_time_ms).
        """
        start_time = time.time()

        prompt = self._build_rag_prompt(question, chunks, language)
        system = FINANCIAL_SYSTEM_PROMPT if language == "fr" else FINANCIAL_SYSTEM_PROMPT_EN

        logger.info(f"Sending prompt to Ollama ({self.model})...")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "system": system,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,    # Low temp for factual answers
                            "top_p": 0.9,
                            "num_predict": 1024,
                        },
                    },
                )
                response.raise_for_status()
                data = response.json()
                answer = data.get("response", "").strip()
                elapsed_ms = (time.time() - start_time) * 1000

                logger.info(
                    f"Ollama response: {len(answer)} chars in {elapsed_ms:.0f}ms"
                )
                return answer, elapsed_ms

        except httpx.TimeoutException:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error("Ollama request timed out")
            return (
                "Erreur : Le modèle a mis trop de temps à répondre. "
                "Veuillez réessayer avec une question plus courte.",
                elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(f"Ollama error: {e}")
            return f"Erreur de génération : {str(e)}", elapsed_ms

    async def list_models(self) -> list[str]:
        """Return list of available Ollama models."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Could not list Ollama models: {e}")
            return []
