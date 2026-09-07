# Development of AI-Based Knowledge Retrieval Platform with Query Resolution System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-orange.svg)](https://www.trychroma.com/)
[![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-green.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

An enterprise-grade, multi-agent Retrieval-Augmented Generation (RAG) platform designed to ingest multi-format organizational knowledge, perform semantic vector retrieval, resolve complex user queries with source citations, and support voice interactions via the Web Speech API.

---

## 📁 Repository Structure

```text
Development-of-AI-Based-Knowledge-Retrieval-Platform-with-Query-Resolution-System/
├── milestone_1/                    # Milestone 1: Foundation & Knowledge Ingestion
│   ├── app/                        # Streamlit UI, Multi-Agent, Ingestion, Embeddings, Vectorstore
│   ├── data/                       # Vector store path and sample domain datasets (ML & Networks)
│   ├── docs/                       # Architecture diagrams, tech stack, data models, research, evaluation
│   ├── tests/                      # Automated unit test suite
│   ├── README.md                   # Milestone 1 specific documentation
│   ├── requirements.txt            # Dependencies
│   └── .env.example                # Environment variables template
├── .gitignore                      # Global Git ignore rules
└── README.md                       # Main project overview
```

---

## 📌 Milestone 1 Overview

Milestone 1 establishes the core architectural foundation, multi-format knowledge ingestion pipeline, vector store indexing, lightweight multi-agent orchestration, and semantic retrieval evaluation.

### Deliverables Checklist

- [x] **M1.1 Research & Technical Understanding**: Documented RAG pipelines, chunking, embeddings, vector search, multi-agent patterns, and Web Speech API ([`milestone_1/docs/m1.1-research.md`](milestone_1/docs/m1.1-research.md)).
- [x] **M1.2 System Architecture & Design**: Architecture diagram ([`milestone_1/docs/architecture.png`](milestone_1/docs/architecture.png)), technical stack selection ([`milestone_1/docs/tech-stack.md`](milestone_1/docs/tech-stack.md)), agent workflows ([`milestone_1/docs/system-architecture.md`](milestone_1/docs/system-architecture.md)), and data models ([`milestone_1/docs/data-models.md`](milestone_1/docs/data-models.md)).
- [x] **M1.3 Knowledge Base Ingestion Module**: Multi-format parsing (PDF, DOCX, TXT, CSV), boundary-aware chunking with overlap, `all-MiniLM-L6-v2` embeddings, and persistent ChromaDB indexing.
- [x] **M1.4 Retrieval Pipeline & Validation**: Top-K semantic retrieval, 5-agent query resolution orchestration, 2-domain benchmark dataset, and evaluation metrics framework ([`milestone_1/docs/m1.4-evaluation.md`](milestone_1/docs/m1.4-evaluation.md)).

---

## 🚀 Quickstart: Running Milestone 1

```bash
cd milestone_1

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit Web UI
streamlit run app/main.py

# Run Unit Tests
python3 -m unittest discover -s tests
```

For complete documentation and details, see [**`milestone_1/README.md`**](milestone_1/README.md).
