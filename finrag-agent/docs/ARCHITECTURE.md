# FinRAG Agent — Architecture & Choix Techniques

## Vue d'ensemble

FinRAG Agent est un système RAG (Retrieval-Augmented Generation) conçu spécifiquement pour l'analyse de documents financiers. Il permet d'interroger une base documentaire en langage naturel et obtenir des réponses ancrées dans les documents, sans hallucination.

---

## Composants principaux

### 1. API Gateway — FastAPI

**Choix :** FastAPI 0.115  
**Raisons :**
- Framework Python le plus rapide (Starlette + Uvicorn)
- Support natif `async/await` pour les opérations I/O
- Génération automatique OpenAPI/Swagger
- Validation des données via Pydantic (type safety)
- Middleware CORS intégré
- Gestion du cycle de vie avec `lifespan`

**Endpoints exposés :**

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| POST | `/api/v1/documents/upload` | Uploader et indexer un PDF |
| GET | `/api/v1/documents/` | Lister tous les documents |
| GET | `/api/v1/documents/{id}` | Détails d'un document |
| DELETE | `/api/v1/documents/{id}` | Supprimer document + vecteurs |
| POST | `/api/v1/query/` | Poser une question |
| GET | `/api/v1/query/history` | Historique des requêtes |
| GET | `/api/v1/health` | Santé du système |

---

### 2. Extraction PDF — pdfplumber + pypdf

**Stratégie :** Double moteur avec fallback automatique

```
pdfplumber (primaire)
  → Meilleur pour : tableaux financiers, colonnes, texte structuré
  → Extrait : texte + tableaux séparément
  → Problème : peut échouer sur PDF scannés ou chiffrés

pypdf (fallback)
  → Meilleur pour : PDF simples, scannés avec texte OCR embarqué
  → Plus tolérant aux PDF malformés
```

**Post-traitement :**
- Normalisation des espaces et caractères spéciaux
- Conversion des tableaux en texte structuré lisible
- Numérotation des pages dans le texte extrait
- Estimation du type de document (rapport annuel, prospectus, etc.)

---

### 3. Découpage en chunks — LangChain TextSplitter

**Algorithme :** `RecursiveCharacterTextSplitter`

**Paramètres par défaut :**
```
chunk_size = 1000 caractères
chunk_overlap = 200 caractères
séparateurs = ["\n\n", "\n", ". ", " ", ""]
```

**Raison du chevauchement :**  
Les informations financières importantes peuvent chevaucher deux paragraphes (ex: "le ratio de solvabilité est de 12,5%... conformément aux exigences Bâle III"). Le chevauchement de 200 caractères préserve ce contexte.

**Métadonnées stockées par chunk :**
- `document_id` : référence vers SQLite
- `filename` : nom du fichier source
- `page_num` : numéro de page estimé
- `chunk_index` : index dans la séquence
- `char_count` : nombre de caractères

---

### 4. Embeddings — sentence-transformers

**Modèle :** `paraphrase-multilingual-mpnet-base-v2`

**Justification :**
- Supporte 50+ langues dont **français et anglais** (crucial pour docs financiers bilingues)
- Dimensions : 768 → bonne capacité représentationnelle
- Performance : top-3 MTEB pour les tâches de retrieval multilingue
- Taille : ~280 MB (téléchargé automatiquement au premier démarrage)
- Licence : Apache 2.0 (usage commercial autorisé)
- **Gratuit, local, aucun appel API externe**

**Alternative pour Ollama embeddings :**  
Ollama peut aussi générer les embeddings (`nomic-embed-text`), mais sentence-transformers est plus rapide en batch processing et ne requiert pas Ollama pour l'indexation.

---

### 5. Base vectorielle — ChromaDB

**Choix :** ChromaDB 0.5 (mode persistant)

**Justification :**
- **Zero-config** : pas de serveur séparé, s'intègre directement dans le processus Python
- **Persistance** : les vecteurs survivent aux redémarrages
- **Cosine similarity** : métrique optimale pour les embeddings textuels normalisés
- **HNSW index** : recherche approximative rapide (logarithmique)
- **Filtres de métadonnées** : permet de restreindre la recherche à un sous-ensemble de documents

**Index utilisé :** HNSW (Hierarchical Navigable Small World)
```
Configuration : hnsw:space = cosine
Score = 1 - cosine_distance
Seuil minimum : 0.3 (configurable)
```

**Alternatives considérées :**
- Weaviate : nécessite un serveur Docker séparé
- Qdrant : excellent mais overhead de déploiement
- FAISS : rapide mais pas de persistance native ni filtres

---

### 6. LLM — Ollama

**Choix :** Ollama comme runtime local pour LLM

**Avantages :**
- **Privé** : 100% local, aucune donnée ne sort du serveur
- **Gratuit** : pas de coût par requête
- **Flexible** : un seul outil pour changer de modèle
- **RGPD-compatible** : traitement des données financières sensibles en local

**Modèles recommandés :**

| Modèle | RAM | Qualité FR | Vitesse | Usage |
|--------|-----|-----------|---------|-------|
| phi3:mini | 4 GB | Moyen | Rapide | Machines légères |
| mistral:7b | 8 GB | Excellent | Moyen | Standard |
| llama3.2:8b | 10 GB | Très bon | Moyen | Standard+ |
| mixtral:8x7b | 32 GB | Excellent | Lent | Serveurs |

**Prompt engineering :**
- Température basse (0.1) → réponses factuelles et reproductibles
- System prompt spécialisé finance → citation des sources et refus d'inventer
- Contexte RAG structuré → sources numérotées avec nom fichier et page

---

### 7. Base de données — SQLite + SQLAlchemy Async

**Choix :** SQLite pour les métadonnées documentaires

**Tables :**

```sql
-- Documents indexés
documents (
  id, filename, original_filename, file_path,
  file_size, page_count, doc_type, language,
  status, chunk_count, error_message,
  created_at, updated_at
)

-- Historique des requêtes
query_logs (
  id, question, answer, sources (JSON),
  processing_time_ms, model_used, chunks_retrieved,
  created_at
)
```

**SQLAlchemy async :** Permet les opérations DB non-bloquantes compatibles avec FastAPI async.

---

## Flux de données détaillé

### Ingestion d'un document

```
POST /api/v1/documents/upload
  │
  ├─ Validation (extension PDF, taille max)
  │
  ├─ Sauvegarde fichier : data/pdfs/{safe_filename}
  │
  ├─ INSERT documents (status='pending')
  │
  ├─ PDFService.extract()
  │    ├─ pdfplumber.open() → text + tables par page
  │    └─ (fallback) pypdf.PdfReader() → text par page
  │
  ├─ detect_doc_type() → classifie le type de document
  │
  ├─ UPDATE documents (status='indexing', page_count, doc_type)
  │
  ├─ PDFService.split_into_chunks()
  │    └─ RecursiveCharacterTextSplitter → N chunks avec métadonnées
  │
  ├─ VectorStoreService.add_chunks()
  │    ├─ SentenceTransformer.encode(texts) → embeddings [N x 768]
  │    └─ ChromaDB.collection.upsert(ids, texts, embeddings, metadatas)
  │
  └─ UPDATE documents (status='indexed', chunk_count=N)
```

### Traitement d'une requête

```
POST /api/v1/query/
  │
  ├─ OllamaService.is_available() → vérifie que Ollama tourne
  │
  ├─ VectorStoreService.search(question, top_k=5)
  │    ├─ SentenceTransformer.encode([question]) → query_embedding [768]
  │    └─ ChromaDB.query(embedding, n_results=5) → chunks + scores
  │
  ├─ Filtrage par seuil de similarité (≥ 0.3)
  │
  ├─ OllamaService.generate(question, chunks)
  │    ├─ _build_rag_prompt() → contexte structuré
  │    └─ POST /api/generate → réponse texte
  │
  ├─ INSERT query_logs (question, answer, sources, timing)
  │
  └─ QueryResponse (answer, sources[], timing, model_used)
```

---

## Considérations de performance

| Opération | Temps estimé | Optimisation possible |
|-----------|-------------|----------------------|
| Indexation (10 pages) | 5-15s | Batch embeddings, parallélisme |
| Indexation (100 pages) | 30-90s | Worker asynchrone |
| Recherche sémantique | < 500ms | Cache embeddings fréquents |
| Génération LLM (mistral) | 2-10s | Streaming response |
| Génération LLM (phi3:mini) | 1-3s | Plus rapide, moins précis |

---

## Sécurité

- Validation des types de fichiers (extension + magic bytes)
- Limite de taille de fichier (50 MB par défaut)
- Nom de fichier sanitisé (UUID suffix pour éviter collisions)
- LLM local → aucune donnée sensible transmise à des API externes
- CORS configurable par environnement

---

## Scalabilité future

Le système peut évoluer vers :
1. **Workers asynchrones** (Celery/Redis) pour l'indexation en arrière-plan
2. **PostgreSQL + pgvector** pour remplacer SQLite + ChromaDB en production
3. **Streaming** des réponses LLM (SSE/WebSocket)
4. **Multi-tenant** avec isolation par utilisateur/organisation
5. **GPU acceleration** pour les embeddings et le LLM
