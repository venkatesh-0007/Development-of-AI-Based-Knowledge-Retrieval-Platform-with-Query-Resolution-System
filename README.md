# Development of AI-Based Knowledge Retrieval Platform with Query Resolution System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-orange.svg)](https://www.trychroma.com/)
[![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-green.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

An enterprise-grade, multi-agent Retrieval-Augmented Generation (RAG) platform designed to ingest multi-format organizational knowledge, perform semantic vector retrieval, resolve complex user queries with source citations, and support voice interactions via the Web Speech API.

---

## 📌 Milestone 1 Overview

Milestone 1 establishes the core architectural foundation, multi-format knowledge ingestion pipeline, vector store indexing, lightweight multi-agent orchestration, and semantic retrieval evaluation.

### Deliverables Checklist

- [x] **M1.1 Research & Technical Understanding**: Documented RAG pipelines, chunking, embeddings, vector search, multi-agent patterns, and Web Speech API ([`docs/m1.1-research.md`](docs/m1.1-research.md)).
- [x] **M1.2 System Architecture & Design**: Architecture diagram ([`docs/architecture.png`](docs/architecture.png)), technical stack selection ([`docs/tech-stack.md`](docs/tech-stack.md)), agent workflows ([`docs/system-architecture.md`](docs/system-architecture.md)), and data models ([`docs/data-models.md`](docs/data-models.md)).
- [x] **M1.3 Knowledge Base Ingestion Module**: Multi-format parsing (PDF, DOCX, TXT, CSV), boundary-aware chunking with overlap, `all-MiniLM-L6-v2` embeddings, and persistent ChromaDB indexing.
- [x] **M1.4 Retrieval Pipeline & Validation**: Top-K semantic retrieval, 5-agent query resolution orchestration, 2-domain benchmark dataset, and evaluation metrics framework ([`docs/m1.4-evaluation.md`](docs/m1.4-evaluation.md)).

---

## 🏛️ System Architecture

The platform separates ingestion from query-time retrieval to maximize throughput and search accuracy:

```text
               ┌────────────────────────────────────────────────────────┐
               │                Presentation Layer                      │
               │   • Streamlit Web UI      • Web Speech API (STT / TTS) │
               └───────────────────────────┬────────────────────────────┘
                                           │
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │                Multi-Agent Orchestrator                │
               └──────┬────────────────────┬────────────────────┬───────┘
                      │                    │                    │
                      ▼                    ▼                    ▼
     ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
     │ Query Understanding  │  │   Retrieval Agent    │  │  Response Generation │
     │        Agent         │  │   (Semantic Search)  │  │   Agent & Citations  │
     └──────────────────────┘  └──────────┬───────────┘  └──────────────────────┘
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │          ChromaDB Vector Store        │
                      │  (all-MiniLM-L6-v2 Embeddings + Meta) │
                      └───────────────────▲───────────────────┘
                                          │
                                          │ Ingestion Pipeline
                      ┌───────────────────┴───────────────────┐
                      │ PDF / DOCX / TXT / CSV Text Extraction │
                      │  Cleaning → Chunking → Embedding Gen  │
                      └───────────────────────────────────────┘
```

### Multi-Agent Roles

1. **Query Understanding Agent**: Identifies intent, filters empty/noisy input, and reformulates queries for retrieval.
2. **Clarification Agent**: Detects ambiguous queries and prompts the user for necessary context.
3. **Retrieval Agent**: Embeds user queries and executes cosine-similarity search against ChromaDB.
4. **Response Generation Agent**: Synthesizes grounded answers referencing retrieved knowledge chunks with source citations.
5. **Conversation Memory Agent**: Maintains multi-turn conversation context across interactions.

---

## 📂 Project Structure

```text
.
├── app/
│   ├── __init__.py
│   ├── main.py                     # Streamlit application entrypoint
│   ├── agents/                     # Multi-Agent subsystem
│   │   ├── __init__.py
│   │   ├── agents.py               # 5 Agent role implementations
│   │   └── orchestrator.py         # Multi-Agent query coordinator
│   ├── embeddings/                 # Vector embeddings
│   │   ├── __init__.py
│   │   └── embedder.py             # SentenceTransformers wrapper
│   ├── ingestion/                  # Multi-format document parser & chunker
│   │   ├── __init__.py
│   │   ├── chunker.py              # Boundary-aware sliding-window chunker
│   │   ├── cleaner.py              # Text normalizer and sanitization
│   │   ├── loaders.py              # PDF, DOCX, TXT, CSV extractors
│   │   └── pipeline.py             # Document-to-Vector indexing pipeline
│   ├── retrieval/                  # Semantic search pipeline
│   │   ├── __init__.py
│   │   └── pipeline.py             # Top-K retrieval execution
│   └── vectorstore/                # Vector store client
│       ├── __init__.py
│       └── chroma_store.py         # Persistent ChromaDB client
├── data/
│   ├── chroma/                     # Persistent vector database files
│   ├── uploads/                    # Local staging for uploaded files
│   └── sample_docs/                # 2-Domain benchmark dataset (ML & Networks)
│       ├── ml_overview.txt
│       ├── ml_algorithms.csv
│       ├── computer_networks.txt
│       └── network_protocols.csv
├── docs/                           # Architecture & Milestone Documentation
│   ├── architecture.png            # Visual architecture diagram
│   ├── data-models.md              # RAG data schemas & entity relationships
│   ├── m1.1-research.md            # Research findings & tech rationale
│   ├── m1.4-evaluation.md          # Benchmark evaluation protocol & results
│   ├── MILESTONE_1_STATUS.md       # Status checklist
│   ├── system-architecture.md      # In-depth system architecture
│   └── tech-stack.md               # Technology stack specifications
├── tests/                          # Automated unit & integration tests
│   ├── __init__.py
│   ├── test_chunker.py
│   ├── test_cleaner.py
│   ├── test_loaders.py
│   └── test_orchestrator.py
├── .env.example                    # Environment variable template
├── .gitignore                      # Git exclusion rules
├── requirements.txt                # Python package dependencies
└── README.md                       # Main documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10 or higher
- Git

### 2. Environment Setup

```bash
# Clone the repository
git clone <repository_url>
cd Development-of-AI-Based-Knowledge-Retrieval-Platform-with-Query-Resolution-System

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app/main.py
```

Open your browser at `http://localhost:8501` to access the knowledge retrieval platform.

### 4. Running Tests

```bash
pytest tests/ -v
```

---

## 📊 Sample Knowledge Base & Evaluation (M1.4)

The repository includes curated benchmark datasets in [`data/sample_docs/`](data/sample_docs/):
- **Domain 1 — Machine Learning**: `ml_overview.txt`, `ml_algorithms.csv`
- **Domain 2 — Computer Networks**: `computer_networks.txt`, `network_protocols.csv`

To validate retrieval accuracy across Factual, Procedural, Comparative, and Unavailable queries, consult [`docs/m1.4-evaluation.md`](docs/m1.4-evaluation.md).

---

## 🔮 Roadmap: Milestone 2 & Beyond

- [ ] FastAPI REST backend endpoints for decoupled microservice architecture.
- [ ] LLM integration for contextual answer generation with dynamic prompt grounding.
- [ ] Hybrid search (BM25 keyword + dense semantic vector retrieval) with reranking.
- [ ] Full browser Web Speech API voice capture & synthesis integration in UI.
- [ ] Advanced query analytics and knowledge gap detection.
