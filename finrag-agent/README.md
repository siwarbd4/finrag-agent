# FinRAG Agent 🏦🤖

**Agent IA RAG pour l'analyse de documents financiers**

Un microservice complet de type RAG (Retrieval-Augmented Generation) permettant d'interroger des documents financiers en langage naturel, avec un LLM local via Ollama.

---

## 📋 Table des matières

1. [Architecture](#architecture)
2. [Choix techniques](#choix-techniques)
3. [Prérequis](#prérequis)
4. [Installation rapide](#installation-rapide)
5. [Installation détaillée](#installation-détaillée)
6. [Configuration Ollama](#configuration-ollama)
7. [Utilisation de l'API](#utilisation-de-lapi)
8. [Tests](#tests)
9. [Docker](#docker)
10. [Structure du projet](#structure-du-projet)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT (HTTP)                            │
└───────────────────┬─────────────────────────┬───────────────────┘
                    │                         │
              POST /upload              POST /query
                    │                         │
┌───────────────────▼─────────────────────────▼───────────────────┐
│                    FastAPI Application                           │
│  ┌──────────────────┐         ┌──────────────────────────────┐  │
│  │  Document Route  │         │       Query Route            │  │
│  └────────┬─────────┘         └──────────────┬───────────────┘  │
│           │                                  │                   │
│  ┌────────▼─────────┐         ┌──────────────▼───────────────┐  │
│  │ Ingestion Service│         │       RAG Pipeline           │  │
│  │                  │         │  1. Embed question           │  │
│  │ 1. Save PDF      │         │  2. Semantic search          │  │
│  │ 2. Extract text  │         │  3. Build context            │  │
│  │ 3. Chunk text    │         │  4. LLM generation           │  │
│  │ 4. Embed chunks  │         └──────────┬───────────────────┘  │
│  │ 5. Store vectors │                    │                       │
│  └──────────────────┘                   │                       │
└──────┬────────────────────────────┬─────┘                       │
       │                            │                              │
┌──────▼──────────┐    ┌────────────▼──────────┐   ┌─────────────▼──┐
│   SQLite DB     │    │   ChromaDB            │   │  Ollama LLM    │
│                 │    │   (Vector Store)      │   │                │
│ - Documents     │    │                       │   │ - mistral      │
│ - Query logs    │    │ - Embeddings          │   │ - llama3.2     │
│ - Metadata      │    │ - Similarity search   │   │ - phi3         │
│                 │    │ - Cosine distance     │   │ - etc.         │
└─────────────────┘    └───────────────────────┘   └────────────────┘
```

### Pipeline d'ingestion PDF

```
PDF File
   │
   ▼
pdfplumber (primary)     pypdf (fallback)
   │                         │
   └──────────┬──────────────┘
              │
         Text + Tables
              │
         Text Cleaning
              │
    RecursiveCharacterTextSplitter
    (chunk_size=1000, overlap=200)
              │
         Chunks [ ]
              │
   SentenceTransformer.encode()
   (paraphrase-multilingual-mpnet-base-v2)
              │
         Embeddings
              │
    ChromaDB.upsert() → Cosine index
              │
         SQLite update (status=indexed)
```

### Pipeline de requête RAG

```
Question (langage naturel)
         │
SentenceTransformer.encode()
         │
ChromaDB.query() → Top-K chunks (cosine similarity)
         │
Filtre par seuil (≥ 0.3)
         │
Build RAG prompt:
  [CONTEXTE]
  Source 1: chunk_text (doc_name, page)
  Source 2: chunk_text ...
  ---
  QUESTION: ...
         │
Ollama /api/generate
(temperature=0.1, num_predict=1024)
         │
Réponse structurée + sources citées
         │
JSON Response (answer, sources, timing)
```

---

## Choix techniques

| Composant | Choix | Justification |
|-----------|-------|---------------|
| **Langage** | Python 3.11 | Ecosystème ML/AI mature, syntaxe async native |
| **Framework API** | FastAPI | Async natif, OpenAPI auto, validation Pydantic |
| **LLM local** | Ollama | Simple à installer, multi-OS, multi-modèles |
| **Modèle génération** | Mistral 7B | Excellent français, léger (4GB RAM), instruction-tuned |
| **Modèle embeddings** | paraphrase-multilingual-mpnet-base-v2 | Multilingue FR/EN, haute qualité, gratuit |
| **Vector Store** | ChromaDB | Persistant, simple, cosine similarity, open-source |
| **PDF extraction** | pdfplumber + pypdf | pdfplumber pour tables financières, pypdf en fallback |
| **Text splitting** | LangChain RecursiveCharacterTextSplitter | Overlap intelligent, préserve le contexte |
| **Base de données** | SQLite + SQLAlchemy async | Zéro configuration, parfait pour dev/MVP |
| **Conteneurisation** | Docker + docker-compose | Reproductible, isolé, prêt pour le déploiement |

### Modèles Ollama recommandés par profil machine

| Profil machine | RAM disponible | Modèle recommandé | Commande |
|----------------|---------------|-------------------|---------|
| Légère | 4 GB | `phi3:mini` | `ollama pull phi3:mini` |
| Standard | 8 GB | `mistral:7b` | `ollama pull mistral` |
| Puissante | 16 GB | `llama3.2:8b` | `ollama pull llama3.2` |
| GPU disponible | - | `mistral:latest` | `ollama pull mistral` |

---

## Prérequis

- Python **3.10+**
- [Ollama](https://ollama.ai) installé et en cours d'exécution
- 8 GB RAM minimum (4 GB pour phi3:mini)
- ~5 GB d'espace disque (modèles + données)

---

## Installation rapide

```bash
# 1. Cloner le repo
git clone https://github.com/VOTRE_USERNAME/finrag-agent.git
cd finrag-agent

# 2. Setup automatique
chmod +x scripts/setup.sh
./scripts/setup.sh

# 3. Configurer l'environnement
cp .env.example .env
# Éditer .env si nécessaire

# 4. Démarrer Ollama + télécharger le modèle
ollama serve &
ollama pull mistral

# 5. Lancer l'API
source venv/bin/activate
uvicorn app.main:app --reload

# 6. Accéder à la documentation
open http://localhost:8000/docs
```

---

## Installation détaillée

### Étape 1 — Installer Ollama

**macOS / Linux :**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows :** Télécharger l'installeur depuis [ollama.ai](https://ollama.ai)

### Étape 2 — Télécharger un modèle adapté

```bash
# Démarrer le daemon Ollama
ollama serve

# Choisir selon votre RAM :
ollama pull phi3:mini        # 4 GB RAM
ollama pull mistral          # 8 GB RAM  ← recommandé
ollama pull llama3.2         # 16 GB RAM

# Tester le modèle
ollama run mistral "Qu'est-ce qu'un bilan financier ?"
```

### Étape 3 — Vérifier le bon fonctionnement

```bash
# Tester Ollama via le script fourni
chmod +x scripts/test_ollama.sh
OLLAMA_MODEL=mistral ./scripts/test_ollama.sh
```

### Étape 4 — Installer le projet

```bash
# Créer et activer le venv
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# ou: venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Configurer l'environnement
cp .env.example .env
# Éditer .env : définir OLLAMA_MODEL selon votre choix
```

### Étape 5 — Lancer l'API

```bash
# Développement (avec rechargement automatique)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

L'API est disponible sur :
- **Swagger UI** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc
- **Health** : http://localhost:8000/api/v1/health

---

## Configuration Ollama

Le fichier `.env` contrôle toute la configuration :

```env
# Modèle de génération (mistral, llama3.2, phi3:mini...)
OLLAMA_MODEL=mistral

# URL Ollama (par défaut en local)
OLLAMA_BASE_URL=http://localhost:11434

# Paramètres RAG
CHUNK_SIZE=1000          # Taille des chunks (caractères)
CHUNK_OVERLAP=200        # Chevauchement entre chunks
TOP_K_RESULTS=5          # Nombre de chunks récupérés par requête
SIMILARITY_THRESHOLD=0.3 # Score minimum de similarité (0-1)
```

---

## Utilisation de l'API

### 1. Uploader un document PDF

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@rapport_annuel_2023.pdf"
```

**Réponse :**
```json
{
  "document_id": 1,
  "filename": "rapport_annuel_2023_a1b2c3d4.pdf",
  "original_filename": "rapport_annuel_2023.pdf",
  "file_size": 2048576,
  "status": "indexed",
  "message": "Document indexed successfully: 42 pages, 187 chunks"
}
```

### 2. Poser une question

```bash
curl -X POST http://localhost:8000/api/v1/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quel est le chiffre d affaires total pour l exercice 2023 ?",
    "language": "fr"
  }'
```

**Réponse :**
```json
{
  "question": "Quel est le chiffre d affaires total pour l exercice 2023 ?",
  "answer": "Selon le rapport annuel 2023 (page 12), le chiffre d'affaires total s'élève à 145,3 millions d'euros, en progression de 8,2% par rapport à l'exercice précédent...",
  "sources": [
    {
      "document_id": 1,
      "document_name": "rapport_annuel_2023_a1b2c3d4.pdf",
      "page": 12,
      "chunk_index": 45,
      "content": "Le chiffre d'affaires consolidé pour l'exercice 2023...",
      "similarity_score": 0.8423
    }
  ],
  "model_used": "mistral",
  "chunks_retrieved": 5,
  "processing_time_ms": 2340.5,
  "has_answer": true
}
```

### 3. Lister les documents indexés

```bash
curl http://localhost:8000/api/v1/documents/
```

### 4. Vérifier la santé du système

```bash
curl http://localhost:8000/api/v1/health
```

### 5. Restreindre la recherche à certains documents

```bash
curl -X POST http://localhost:8000/api/v1/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quels sont les ratios de solvabilité ?",
    "doc_ids": [1, 3],
    "top_k": 8
  }'
```

---

## Tests

### Tests unitaires

```bash
# Activer le venv
source venv/bin/activate

# Lancer tous les tests unitaires
pytest tests/unit/ -v

# Avec couverture
pytest tests/unit/ --cov=app --cov-report=html
```

### Tests d'intégration API

```bash
# Tester l'API (doit être lancée au préalable)
chmod +x scripts/test_api.sh

# Sans document
./scripts/test_api.sh

# Avec un PDF financier
./scripts/test_api.sh /chemin/vers/rapport_annuel.pdf
```

### Test complet Ollama

```bash
chmod +x scripts/test_ollama.sh
OLLAMA_MODEL=mistral ./scripts/test_ollama.sh
```

---

## Docker

### Démarrage complet avec Docker Compose

```bash
# Construire et démarrer tous les services
docker-compose up --build -d

# Suivre les logs
docker-compose logs -f finrag-api

# Arrêter les services
docker-compose down
```

Le service `ollama-pull` tire automatiquement le modèle au premier démarrage.

---

## Structure du projet

```
finrag-agent/
├── app/
│   ├── main.py                  # Point d'entrée FastAPI
│   ├── api/
│   │   └── routes/
│   │       ├── documents.py     # POST /upload, GET /, DELETE /{id}
│   │       ├── query.py         # POST /query, GET /history
│   │       └── health.py        # GET /health
│   ├── core/
│   │   ├── config.py            # Settings (pydantic-settings)
│   │   └── database.py          # SQLAlchemy models + init
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response schemas
│   └── services/
│       ├── pdf_service.py       # Extraction + chunking PDF
│       ├── vector_store.py      # ChromaDB + embeddings
│       ├── ollama_service.py    # Client Ollama LLM
│       └── ingestion_service.py # Pipeline orchestration
├── data/
│   ├── pdfs/                    # PDFs uploadés
│   └── vectors/                 # ChromaDB persist
├── tests/
│   ├── unit/
│   │   └── test_pdf_service.py  # Tests unitaires
│   └── integration/
│       └── test_api.py          # Tests d'intégration FastAPI
├── docs/
│   └── ARCHITECTURE.md          # Documentation architecture détaillée
├── scripts/
│   ├── setup.sh                 # Script d'installation
│   ├── test_ollama.sh           # Tests Ollama
│   └── test_api.sh              # Tests API end-to-end
├── docker/
│   └── Dockerfile               # Image Docker
├── docker-compose.yml           # Orchestration Docker
├── requirements.txt             # Dépendances Python
├── pytest.ini                   # Configuration pytest
├── .env.example                 # Template configuration
└── README.md
```

---

## Documents financiers supportés

| Type | Exemples | Détection automatique |
|------|---------|----------------------|
| Rapport annuel | Rapport annuel 2023, Annual Report | ✅ |
| Prospectus | Prospectus d'émission, IPO | ✅ |
| Fiche fonds | Fund Factsheet, UCITS, OPCVM | ✅ |
| États financiers | Bilan, Compte de résultat | ✅ |
| Rapport trimestriel | Résultats T3 2023 | ✅ |
| Note d'information | Document de référence | ✅ |
| Autres | Tout document financier PDF | ✅ (générique) |

---

## Licence

MIT License — voir [LICENSE](LICENSE)
