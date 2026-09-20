# Development of AI-Based Knowledge Retrieval Platform with Query Resolution System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-orange.svg)](https://www.trychroma.com/)
[![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-green.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Tests: 92/92 Passing](https://img.shields.io/badge/Tests-92%2F92%20Passing-success.svg)]()

An enterprise-grade, multi-agent Retrieval-Augmented Generation (RAG) platform designed to ingest multi-format organizational knowledge, perform semantic vector retrieval, resolve complex user queries with source citations, maintain multi-turn conversational memory, support voice interactions via the Web Speech API, and provide granular response transparency.

---

## 📁 Repository Structure by Milestones

```text
Development-of-AI-Based-Knowledge-Retrieval-Platform-with-Query-Resolution-System/
├── milestone_1/                    # Milestone 1: Foundation & Knowledge Ingestion (COMPLETED)
│   ├── app/                        # Ingestion, Embeddings, ChromaDB, Baseline Retrieval
│   ├── data/                       # Vector store path and sample domain datasets (ML & Networks)
│   ├── docs/                       # Architecture diagrams, tech stack, data models, research, evaluation
│   ├── tests/                      # Automated unit tests (11 passing)
│   └── README.md
├── milestone_2/                    # Milestone 2: Multi-Agent Query Resolution & Response (COMPLETED)
│   ├── app/                        # Query Understanding, Retrieval, Response Generation, Orchestration, UI
│   ├── data/                       # Vector store path and sample domain datasets
│   ├── docs/                       # Technical specs (M2.1, M2.2, M2.3, M2.4, Status)
│   ├── tests/                      # Automated unit tests (41 passing)
│   └── README.md
├── milestone_3/                    # Milestone 3: Clarification, Memory, Voice & Transparency (COMPLETED)
│   ├── app/                        # Clarification Agent, Conversation Memory, Voice Module, Transparency, UI
│   ├── data/                       # Vector store path and sample domain datasets
│   ├── docs/                       # Technical specs (M3.1, M3.2, M3.3, M3.4, Status)
│   ├── evaluation/                 # Benchmark scripts (Memory, Voice, Transparency, Clarification)
│   ├── tests/                      # Automated unit and integration tests (40 passing)
│   └── README.md
├── .gitignore                      # Global Git ignore rules
└── README.md                       # Main repository overview
```

---

## 📌 Milestones Roadmap & Status

| Milestone | Scope & Deliverables | Status | Link |
|---|---|:---:|---|
| **Milestone 1** | Foundation & Ingestion: Multi-format parsing (PDF, DOCX, TXT, CSV), chunking with overlap, `all-MiniLM-L6-v2` embeddings, ChromaDB, 5 agent definitions, Streamlit prototype, and 2-domain evaluation. | **Completed** | [📂 `milestone_1/`](milestone_1/) |
| **Milestone 2** | Multi-Agent Query Resolution: M2.1 Query Understanding Agent (intent classification, ambiguity detection), M2.2 Retrieval Agent (dynamic Top-K, relevance filtering), M2.3 Response Generation Agent (grounded synthesis, confidence indicators, source citations), M2.4 Orchestration Layer & interactive Streamlit UI. | **Completed** | [📂 `milestone_2/`](milestone_2/) |
| **Milestone 3** | Clarification, Memory, Voice & Transparency: M3.1 Clarification Agent (interactive ambiguity resolution dialogues), M3.2 Conversation Memory Agent (anaphora resolution, multi-turn history), M3.3 Web Speech API Voice STT/TTS with speech cleaner, M3.4 Response Transparency Panel (confidence formulas, chunk inspectors, citations). | **Completed** | [📂 `milestone_3/`](milestone_3/) |
| **Milestone 4** | Benchmarking & Enterprise Deployment: Precision/Recall/MRR evaluation, Knowledge Gap detection, Docker containerization, Query analytics. | *Planned* | `milestone_4/` |

---

## 🚀 Quickstart: Running Milestone 3

```bash
cd milestone_3

# Run Streamlit Multi-Agent Application
streamlit run app/main.py

# Run Unit & Integration Tests (40 passing)
python3 -m unittest discover -s tests -v

# Run Benchmark & Evaluation Suite
python3 evaluation/evaluate_m3_memory_and_transparency.py
```

