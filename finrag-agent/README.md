# FinRAG Agent 🏦🤖

**Agent IA RAG pour l'analyse de documents financiers**  
*Développé par Siwar Bouamoud*

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-orange.svg)](https://ollama.ai)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-purple.svg)](https://chromadb.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Description

**FinRAG Agent** est un système RAG (Retrieval-Augmented Generation) conçu spécifiquement pour l'analyse de documents financiers PDF. Il permet d'interroger une base documentaire en langage naturel et d'obtenir des réponses précises, ancrées uniquement dans les documents indexés — sans hallucination et sans aucun appel à des API cloud.

> 💡 **Exemple d'usage :** Uploader un rapport annuel de 200 pages et demander directement *"Quel est le chiffre d'affaires du S1 2023 ?"* — le système retrouve les passages pertinents et génère une réponse sourcée.

---

## 🗂️ Table des matières

1. [Architecture](#-architecture)
2. [Choix techniques](#-choix-techniques)
3. [Prérequis](#-prérequis)
4. [Installation](#-installation)
5. [Configuration Ollama](#-configuration-ollama)
6. [Utilisation de l'API](#-utilisation-de-lapi)
7. [Tests](#-tests)
8. [Docker](#-docker)
9. [Structure du projet](#-structure-du-projet)
10. [Documents supportés](#-documents-financiers-supportés)

---

## 🏗️ Architecture

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
│ - Query logs    │    │ - Embeddings (768d)   │   │ - llama3.2     │
│ - Métadonnées   │    │ - Cosine similarity   │   │ - phi3:mini    │
└─────────────────┘    └───────────────────────┘   └────────────────┘
```

### Pipeline d'ingestion PDF

```
Fichier PDF
     │
     ▼
pdfplumber (primaire)    pypdf (fallback)
     │                        │
     └──────────┬─────────────┘
                │
           Texte + Tableaux
                │
           Nettoyage texte
                │
     RecursiveCharacterTextSplitter
     (chunk_size=1000, overlap=200)
                │
           Chunks [ ]
                │
     SentenceTransformer.encode()
     (paraphrase-multilingual-mpnet-base-v2)
                │
           Embeddings [N × 768]
                │
     ChromaDB.upsert() → Index HNSW
                │
     SQLite update (status=indexed)
```

### Pipeline de requête RAG

```
Question (langage naturel)
          │
SentenceTransformer.encode()
          │
ChromaDB.query() → Top-5 chunks (cosine similarity)
          │
Filtre par seuil (≥ 0.3)
          │
Construction du prompt RAG :
  [CONTEXTE]
  Source 1: chunk_text (fichier, page)
  Source 2: ...
  QUESTION: ...
          │
Ollama /api/generate (mistral, temp=0.1)
          │
Réponse structurée + sources citées
          │
JSON Response (answer, sources, timing_ms)
```

---

## ⚙️ Choix techniques

| Composant | Choix retenu | Justification |
|-----------|-------------|---------------|
| **Langage** | Python 3.11+ | Écosystème ML/AI mature, async natif |
| **Framework API** | FastAPI 0.115 | Async natif, OpenAPI auto, validation Pydantic |
| **LLM local** | Ollama | Multi-OS, multi-modèles, 100% local |
| **Modèle génération** | Mistral 7B | Excellent français, 8 GB RAM, instruction-tuned |
| **Modèle embeddings** | paraphrase-multilingual-mpnet-base-v2 | Multilingue FR/EN, 768d, gratuit |
| **Vector Store** | ChromaDB | Persistant, cosine similarity, zéro serveur |
| **PDF extraction** | pdfplumber + pypdf | pdfplumber pour tableaux financiers, pypdf en fallback |
| **Text splitting** | LangChain RecursiveCharacterTextSplitter | Overlap intelligent, préserve le contexte |
| **Base de données** | SQLite + SQLAlchemy async | Zéro configuration, adapté MVP |
| **Conteneurisation** | Docker + docker-compose | Reproductible, isolé |

---

## 📋 Prérequis

- **Python** 3.11 ou supérieur
- **Ollama** installé et en cours d'exécution ([ollama.ai](https://ollama.ai))
- **RAM** : 8 GB minimum (4 GB si utilisation de `phi3:mini`)
- **Espace disque** : ~6 GB (modèle Mistral + dépendances)
- **Windows** : Microsoft C++ Build Tools (pour ChromaDB)

---

## 🚀 Installation

### Étape 1 — Cloner le dépôt

```bash
git clone https://github.com/siwar-bouamoud/finrag-agent.git
cd finrag-agent
```

### Étape 2 — Installer Ollama

**macOS / Linux :**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows :** Télécharger l'installeur depuis [ollama.ai/download](https://ollama.ai/download)

### Étape 3 — Télécharger le modèle

```bash
# Démarrer Ollama
ollama serve

# Choisir selon votre RAM :
ollama pull phi3:mini    # 4 GB RAM
ollama pull mistral      # 8 GB RAM  ← recommandé
ollama pull llama3.2     # 16 GB RAM

# Vérifier le fonctionnement
ollama run mistral "Qu'est-ce qu'un bilan financier ?"
```

### Étape 4 — Environnement Python

```bash
# Créer l'environnement virtuel
python -m venv venv

# Activer (Linux/macOS)
source venv/bin/activate

# Activer (Windows)
venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

> **Windows uniquement :** Si une erreur `chroma-hnswlib` apparaît, installer d'abord les [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) en sélectionnant "Desktop development with C++".

### Étape 5 — Configuration

```bash
cp .env.example .env
```

Éditer `.env` :
```env
OLLAMA_MODEL=mistral
OLLAMA_BASE_URL=http://localhost:11434
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.3
```

### Étape 6 — Lancer l'API

```bash
# Terminal 1 — Ollama (laisser ouvert)
ollama serve

# Terminal 2 — API
venv\Scripts\activate   # ou source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Accéder à la documentation interactive :
- **Swagger UI** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc
- **Health check** : http://localhost:8000/api/v1/health

---

## 🤖 Configuration Ollama

| Profil machine | RAM disponible | Modèle recommandé | Commande |
|----------------|---------------|-------------------|----------|
| Légère | 4 GB | `phi3:mini` | `ollama pull phi3:mini` |
| **Standard** | **8 GB** | **`mistral`** | **`ollama pull mistral`** |
| Puissante | 16 GB | `llama3.2` | `ollama pull llama3.2` |

---

## 📡 Utilisation de l'API

### 1. Uploader un document PDF

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@rapport_annuel_2023.pdf"
```

**Réponse :**
```json
{
  "document_id": 1,
  "filename": "rapport_annuel_2023_a1b2c3.pdf",
  "original_filename": "rapport_annuel_2023.pdf",
  "status": "indexed",
  "message": "Document indexed successfully: 42 pages, 187 chunks"
}
```

### 2. Poser une question en langage naturel

```bash
curl -X POST http://localhost:8000/api/v1/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quel est le chiffre d affaires total pour 2023 ?",
    "language": "fr"
  }'
```

**Réponse :**
```json
{
  "question": "Quel est le chiffre d affaires total pour 2023 ?",
  "answer": "Selon le rapport annuel 2023 (page 12), le chiffre d'affaires total s'élève à 145,3 millions d'euros, en progression de 8,2% par rapport à l'exercice précédent.",
  "sources": [
    {
      "document_id": 1,
      "document_name": "rapport_annuel_2023.pdf",
      "page": 12,
      "similarity_score": 0.842
    }
  ],
  "model_used": "mistral",
  "processing_time_ms": 2340,
  "has_answer": true
}
```

### 3. Lister les documents indexés

```bash
curl http://localhost:8000/api/v1/documents/
```

### 4. Restreindre la recherche à certains documents

```bash
curl -X POST http://localhost:8000/api/v1/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quels sont les ratios de solvabilité ?",
    "doc_ids": [1, 3],
    "top_k": 8
  }'
```

### 5. Supprimer un document

```bash
curl -X DELETE http://localhost:8000/api/v1/documents/1
```

---

## 🧪 Tests

```bash
# Tests unitaires
pytest tests/unit/ -v

# Tests avec couverture
pytest tests/unit/ --cov=app --cov-report=html

# Tests d'intégration (API doit être lancée)
pytest tests/integration/ -v
```

---

## 🐳 Docker

```bash
# Lancer tous les services
docker-compose up --build -d

# Suivre les logs
docker-compose logs -f finrag-api

# Arrêter
docker-compose down
```

---

## 📁 Structure du projet

```
finrag-agent/
├── app/
│   ├── main.py                   # Point d'entrée FastAPI
│   ├── api/
│   │   └── routes/
│   │       ├── documents.py      # Upload, liste, suppression
│   │       ├── query.py          # Question + historique
│   │       └── health.py         # Santé du système
│   ├── core/
│   │   ├── config.py             # Settings (.env)
│   │   └── database.py           # SQLAlchemy + SQLite
│   ├── models/
│   │   └── schemas.py            # Schémas Pydantic
│   └── services/
│       ├── pdf_service.py        # Extraction + découpage PDF
│       ├── vector_store.py       # ChromaDB + embeddings
│       ├── ollama_service.py     # Client LLM Ollama
│       └── ingestion_service.py  # Orchestration pipeline
├── data/
│   ├── pdfs/                     # PDFs uploadés
│   └── vectors/                  # ChromaDB persistant
├── tests/
│   ├── unit/
│   └── integration/
├── docs/
│   └── ARCHITECTURE.md           # Documentation architecture
├── scripts/
│   ├── setup.sh
│   ├── test_ollama.sh
│   └── test_api.sh
├── docker/
│   └── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## 📄 Documents financiers supportés

| Type | Exemples |
|------|---------|
| Rapport annuel | Rapport annuel 2023, Annual Report |
| Prospectus | Prospectus d'émission, Note d'information |
| Fiche fonds | Fund Factsheet, OPCVM, UCITS |
| États financiers | Bilan, Compte de résultat, Tableau de flux |
| Rapport trimestriel | Résultats T3 2023, Quarterly Report |
| Autres | Tout document financier PDF |

---

## 👤 Auteur

**Siwar Bouamoud**  
GitHub : [github.com/siwar-bouamoud/finrag-agent](https://github.com/siwar-bouamoud/finrag-agent)

---

## 📜 Licence

MIT License — libre d'utilisation, modification et distribution.