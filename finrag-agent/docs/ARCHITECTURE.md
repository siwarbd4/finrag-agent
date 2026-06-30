# FinRAG Agent — Architecture & Choix Techniques

**Développé par Siwar Bouamoud**  
**Dépôt :** [github.com/siwar-bouamoud/finrag-agent](https://github.com/siwar-bouamoud/finrag-agent)

---

## Vue d'ensemble

FinRAG Agent est un système RAG (Retrieval-Augmented Generation) conçu pour l'analyse de documents financiers. Il permet d'interroger une base documentaire PDF en langage naturel et d'obtenir des réponses factuelles, ancrées uniquement dans les documents indexés, sans hallucination et sans transmission de données à des API cloud externes.

---

## Composants principaux

### 1. API Gateway — FastAPI

**Version :** FastAPI 0.115  
**Justification :**
- Framework Python asynchrone le plus rapide (Starlette + Uvicorn)
- Support natif `async/await` pour les opérations I/O non-bloquantes
- Génération automatique de documentation OpenAPI/Swagger
- Validation des données via Pydantic (type safety stricte)
- Middleware CORS intégré
- Gestion du cycle de vie avec `lifespan`

**Endpoints exposés :**

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| POST | `/api/v1/documents/upload` | Uploader et indexer un PDF |
| GET | `/api/v1/documents/` | Lister tous les documents |
| GET | `/api/v1/documents/{id}` | Détails d'un document |
| DELETE | `/api/v1/documents/{id}` | Supprimer document + vecteurs |
| POST | `/api/v1/query/` | Poser une question (RAG pipeline) |
| GET | `/api/v1/query/history` | Historique des requêtes |
| GET | `/api/v1/health` | Santé du système |

---

### 2. Extraction PDF — pdfplumber + pypdf

**Stratégie :** Double moteur avec fallback automatique

```
pdfplumber (moteur primaire)
  → Optimisé pour : tableaux financiers, colonnes, texte structuré
  → Extrait : texte + tableaux séparément avec formatage
  → Limite : peut échouer sur PDF scannés ou chiffrés

pypdf (fallback automatique)
  → Optimisé pour : PDF simples, documents avec texte OCR embarqué
  → Plus tolérant aux PDF malformés ou protégés
```

**Post-traitement :**
- Normalisation des espaces et caractères spéciaux
- Conversion des tableaux en texte structuré lisible
- Numérotation des pages dans le texte extrait
- Détection automatique du type de document (rapport annuel, prospectus, etc.)

---

### 3. Découpage en chunks — LangChain TextSplitter

**Algorithme :** `RecursiveCharacterTextSplitter`

**Paramètres :**
```
chunk_size    = 1000 caractères
chunk_overlap = 200 caractères
séparateurs   = ["\n\n", "\n", ". ", " ", ""]
```

**Raison du chevauchement de 200 caractères :**  
Les informations financières importantes chevauchent souvent deux paragraphes (ex: *"le ratio de solvabilité est de 12,5%... conformément aux exigences Bâle III"*). Le chevauchement préserve ce contexte inter-paragraphes.

**Métadonnées stockées par chunk :**
- `document_id` — référence vers SQLite
- `filename` — nom du fichier source
- `page_num` — numéro de page estimé
- `chunk_index` — index dans la séquence
- `char_count` — nombre de caractères

---

### 4. Embeddings — sentence-transformers

**Modèle :** `paraphrase-multilingual-mpnet-base-v2`

**Justification :**
- Supporte 50+ langues dont **français et anglais** (essentiel pour documents financiers bilingues)
- Dimensions : **768** — bonne capacité représentationnelle
- Performance : top-3 MTEB pour les tâches de retrieval multilingue
- Taille : ~280 MB (téléchargement automatique au premier démarrage)
- Licence : Apache 2.0 (usage commercial autorisé)
- **100% local, aucun appel API externe, gratuit**

---

### 5. Base vectorielle — ChromaDB

**Version :** ChromaDB 0.5 (mode persistant local)

**Justification :**
- **Zero-config** : pas de serveur séparé, intégré directement dans le processus Python
- **Persistance** : les vecteurs survivent aux redémarrages (stockés sur disque)
- **Cosine similarity** : métrique optimale pour les embeddings textuels normalisés
- **Index HNSW** : recherche approximative en temps logarithmique
- **Filtres de métadonnées** : restriction de la recherche à un sous-ensemble de documents

**Configuration de l'index :**
```
Algorithme : HNSW (Hierarchical Navigable Small World)
Métrique   : cosine
Score      : 1 - cosine_distance
Seuil min  : 0.3 (configurable via SIMILARITY_THRESHOLD)
```

**Alternatives évaluées et non retenues :**
- Weaviate : nécessite un serveur Docker séparé
- Qdrant : excellent mais overhead de déploiement pour un MVP
- FAISS : rapide mais pas de persistance native ni de filtres par métadonnées

---

### 6. LLM — Ollama

**Justification du choix Ollama :**
- **Confidentialité totale** : 100% local, aucune donnée financière sensible transmise à l'extérieur
- **Conformité RGPD** : traitement des données sur la machine locale uniquement
- **Gratuit** : pas de coût par requête
- **Flexibilité** : changement de modèle sans modifier le code

**Modèles recommandés :**

| Modèle | RAM | Qualité FR | Vitesse | Usage |
|--------|-----|-----------|---------|-------|
| phi3:mini | 4 GB | Moyen | Rapide | Machines légères |
| **mistral:7b** | **8 GB** | **Excellent** | **Moyen** | **Standard** ← recommandé |
| llama3.2:8b | 10 GB | Très bon | Moyen | Standard+ |
| mixtral:8x7b | 32 GB | Excellent | Lent | Serveurs |

**Paramètres de génération :**
- Température : **0.1** → réponses factuelles et reproductibles (évite les hallucinations)
- System prompt spécialisé finance → citation obligatoire des sources, refus d'inventer
- Contexte RAG structuré → sources numérotées avec nom fichier et numéro de page

---

### 7. Base de données — SQLite + SQLAlchemy Async

**Justification :** SQLite pour les métadonnées documentaires, SQLAlchemy async pour la compatibilité FastAPI.

**Schéma des tables :**

```sql
-- Documents indexés
documents (
  id                INTEGER PRIMARY KEY,
  filename          TEXT,       -- Nom sauvegardé (avec suffix UUID)
  original_filename TEXT,       -- Nom original uploadé
  file_path         TEXT,       -- Chemin sur disque
  file_size         INTEGER,    -- Taille en bytes
  page_count        INTEGER,    -- Nombre de pages
  doc_type          TEXT,       -- Type détecté (rapport_annuel, etc.)
  language          TEXT,       -- Langue détectée
  status            TEXT,       -- pending / indexing / indexed / error
  chunk_count       INTEGER,    -- Nombre de chunks générés
  error_message     TEXT,       -- Message d'erreur si status=error
  created_at        DATETIME,
  updated_at        DATETIME
)

-- Historique des requêtes
query_logs (
  id                  INTEGER PRIMARY KEY,
  question            TEXT,       -- Question posée
  answer              TEXT,       -- Réponse générée
  sources             TEXT,       -- JSON: liste des sources citées
  processing_time_ms  FLOAT,      -- Temps de traitement
  model_used          TEXT,       -- Modèle Ollama utilisé
  chunks_retrieved    INTEGER,    -- Nombre de chunks récupérés
  created_at          DATETIME
)
```

---

## Flux de données détaillé

### Ingestion d'un document

```
POST /api/v1/documents/upload
  │
  ├─ Validation (extension PDF, taille max 50 MB)
  │
  ├─ Sauvegarde fichier : data/pdfs/{uuid_filename}.pdf
  │
  ├─ INSERT documents (status='pending')
  │
  ├─ PDFService.extract()
  │    ├─ pdfplumber.open() → texte + tableaux par page
  │    └─ (fallback) pypdf.PdfReader() → texte par page
  │
  ├─ detect_doc_type() → classifie le type de document
  │
  ├─ UPDATE documents (status='indexing', page_count, doc_type)
  │
  ├─ PDFService.split_into_chunks()
  │    └─ RecursiveCharacterTextSplitter → N chunks avec métadonnées
  │
  ├─ VectorStoreService.add_chunks()
  │    ├─ SentenceTransformer.encode(texts) → embeddings [N × 768]
  │    └─ ChromaDB.collection.upsert(ids, texts, embeddings, metadatas)
  │
  └─ UPDATE documents (status='indexed', chunk_count=N)
```

### Traitement d'une requête RAG

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
  │    ├─ _build_rag_prompt() → contexte structuré avec sources numérotées
  │    └─ POST http://localhost:11434/api/generate → texte de réponse
  │
  ├─ INSERT query_logs (question, answer, sources, timing)
  │
  └─ QueryResponse (answer, sources[], timing_ms, model_used)
```

---

## Performances estimées

| Opération | Temps estimé | Notes |
|-----------|-------------|-------|
| Indexation (10 pages) | 5–15 secondes | Selon RAM et CPU |
| Indexation (100 pages) | 30–90 secondes | Batch embeddings |
| Recherche sémantique | < 500 ms | HNSW logarithmique |
| Génération LLM (mistral) | 2–10 secondes | Selon charge CPU |
| Génération LLM (phi3:mini) | 1–3 secondes | Plus rapide, moins précis |

---

## Sécurité

- Validation du type de fichier (extension + magic bytes)
- Limite de taille : 50 MB par fichier (configurable)
- Nom de fichier sanitisé avec suffix UUID (évite les collisions et injections)
- LLM 100% local → aucune donnée financière sensible transmise à des API externes
- CORS configurable par environnement

---

## Évolutivité future

Le système peut évoluer vers :

1. **Workers asynchrones** (Celery + Redis) pour indexation en arrière-plan
2. **PostgreSQL + pgvector** pour remplacer SQLite + ChromaDB en production
3. **Streaming** des réponses LLM (Server-Sent Events / WebSocket)
4. **Multi-tenant** avec isolation par utilisateur ou organisation
5. **GPU acceleration** pour accélérer embeddings et inférence LLM
6. **Interface web** React/Vue pour une utilisation sans ligne de commande

---

## Étapes d'installation résumées

```bash
# 1. Cloner
git clone https://github.com/siwar-bouamoud/finrag-agent.git
cd finrag-agent

# 2. Ollama
ollama serve
ollama pull mistral

# 3. Python
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 4. Configuration
cp .env.example .env
# Éditer OLLAMA_MODEL=mistral dans .env

# 5. Lancer
uvicorn app.main:app --reload --port 8000

# 6. Tester
open http://localhost:8000/docs
```

---

*Documentation rédigée par Siwar Bouamoud — Projet FinRAG Agent*