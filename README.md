# Development of AI-Based Knowledge Retrieval Platform with Query Resolution System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-orange.svg)](https://www.trychroma.com/)
[![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-green.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Tests: 109/109 Passing](https://img.shields.io/badge/Tests-109%2F109%20Passing-success.svg)]()

An enterprise-grade, multi-agent Retrieval-Augmented Generation (RAG) platform designed to ingest multi-format organizational knowledge, perform semantic vector retrieval, resolve complex user queries with source citations, maintain multi-turn conversational memory, support voice interactions via the Web Speech API, provide granular response transparency, track query analytics, and automatically detect knowledge base gaps.

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
├── milestone_4/                    # Milestone 4: Analytics, Gap Detection, Testing & Optimization (COMPLETED)
│   ├── app/                        # Query Analytics, Gap Detection Engine, Optimized Multi-Agent Platform, UI
│   ├── data/                       # Sample datasets for 3 Domains (ML, Networks, Cybersecurity), Chroma, SQLite
│   ├── demo/                       # Automated terminal showcase script (run_final_demonstration.py)
│   ├── docs/                       # Specifications (M4.1–M4.4), Project Report, and Demonstration Guide
│   ├── evaluation/                 # Retrieval optimization & before-vs-after comparative benchmarks
│   ├── tests/                      # Automated unit and integration tests (17 passing)
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
| **Milestone 4** | Query Analytics, Gap Detection & Optimization: M4.1 Query Analytics and Knowledge Gap Detection module with SQLite telemetry and interactive dashboard, M4.2 End-to-end testing across 3 domains (ML, Networks, Cybersecurity), M4.3 Hybrid retrieval, prompt, agent routing & voice optimization (100% Recall@3, 1.000 MRR), M4.4 Technical documentation, final project report, and demonstration script. | **Completed** | [📂 `milestone_4/`](milestone_4/) |

---

## 🚀 Quickstart: Running Milestone 4 (Final Complete Platform)

### 1. Launch the Interactive Web Application
```bash
cd milestone_4
streamlit run app/main.py
```

### 2. Run the Multi-Domain Automated Demonstration Script
```bash
cd milestone_4
python3 demo/run_final_demonstration.py
```

### 3. Run the Milestone 4 Automated Test Suite (17 Passing)
```bash
cd milestone_4
python3 -m unittest discover -s tests -v
```

### 4. Run Quantitative Optimization Benchmarks
```bash
cd milestone_4

# Retrieval Parameter Tuning Benchmark
python3 evaluation/evaluate_retrieval_optimization.py

# Before-vs-After Multi-Agent System Benchmark
python3 evaluation/benchmark_before_after_m4.py
```
