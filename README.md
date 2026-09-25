# Development of AI-Based Knowledge Retrieval Platform with Query Resolution System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-orange.svg)](https://www.trychroma.com/)
[![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-green.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Tests: 161/161 Passing](https://img.shields.io/badge/Tests-161%2F161%20Passing-success.svg)]()
[![Validation: Milestone 4 Validated](https://img.shields.io/badge/Validation-Milestone%204%20Validated-brightgreen.svg)]()

An enterprise-grade, multi-agent Retrieval-Augmented Generation (RAG) platform designed to ingest multi-format organizational knowledge, perform hybrid semantic vector and lexical retrieval, resolve complex user queries with transparent source citations, maintain multi-turn conversational memory, support voice interactions via the Web Speech API, track operational query analytics in persistent SQLite storage, and automatically discover knowledge base gaps through deterministic thematic clustering.

---

## 📁 Repository Structure by Milestones

```text
Development-of-AI-Based-Knowledge-Retrieval-Platform-with-Query-Resolution-System/
├── milestone_1/                    # Milestone 1: Foundation & Knowledge Ingestion (COMPLETED)
│   ├── app/                        # Ingestion, Embeddings, ChromaDB, Baseline Retrieval
│   ├── data/                       # Vector store path and sample domain datasets (ML & Networks)
│   ├── docs/                       # Architecture diagrams, tech stack, data models, research
│   └── tests/                      # Automated unit tests (11 passing)
├── milestone_2/                    # Milestone 2: Multi-Agent Query Resolution & Response (COMPLETED)
│   ├── app/                        # Query Understanding, Retrieval, Response Generation, Orchestration, UI
│   ├── data/                       # Vector store path and sample domain datasets
│   ├── docs/                       # Technical specs (M2.1, M2.2, M2.3, M2.4, Status)
│   └── tests/                      # Automated unit tests (41 passing)
├── milestone_3/                    # Milestone 3: Clarification, Memory, Voice & Transparency (COMPLETED)
│   ├── app/                        # Clarification Agent, Conversation Memory, Voice Module, Transparency, UI
│   ├── data/                       # Vector store path and sample domain datasets
│   ├── docs/                       # Technical specs (M3.1, M3.2, M3.3, M3.4, Status)
│   ├── evaluation/                 # Benchmark scripts (Memory, Voice, Transparency, Clarification)
│   └── tests/                      # Automated unit and integration tests (47 passing)
├── milestone_4/                    # Milestone 4: Analytics, Gap Detection, Testing & Optimization (COMPLETED)
│   ├── app/                        # Query Analytics, Gap Detection, Centralized Confidence, Multi-Agent UI
│   ├── data/                       # Datasets for 3 Domains (ML, Networks, Cybersecurity), SQLite Analytics DB
│   ├── demo/                       # Automated terminal showcase script
│   ├── docs/                       # Specifications (M4.1–M4.4), Audit, Architecture, Demo Script, Final Reports
│   ├── evaluation/                 # Programmatic evaluation benchmarks (Recall, MRR, Latency, Routing)
│   └── tests/                      # 10 Automated test suites (62 passing, 12 subtests)
├── FINAL_REPORT.md                 # Complete 27-section final project report
├── .gitignore                      # Global Git ignore rules
└── README.md                       # Main repository overview
```

---

## 📌 Milestones Roadmap & Validated Status

| Milestone | Scope & Deliverables | Status | Tests | Validated Performance | Link |
|---|---|:---:|:---:|:---:|---|
| **Milestone 1** | Foundation & Ingestion: Multi-format parsing (PDF, TXT, CSV), recursive character chunking, `all-MiniLM-L6-v2` dense embeddings, ChromaDB, prototype UI. | **Completed** | 11/11 | Functional parsing & vector store | [📂 `milestone_1/`](milestone_1/) |
| **Milestone 2** | Multi-Agent Query Resolution: M2.1 Query Understanding Agent, M2.2 Retrieval Agent, M2.3 Response Generation Agent with grounded citations, M2.4 Orchestration Layer & Streamlit UI. | **Completed** | 41/41 | Multi-agent execution | [📂 `milestone_2/`](milestone_2/) |
| **Milestone 3** | Clarification, Memory, Voice & Transparency: M3.1 Clarification Agent (interactive ambiguity resolution), M3.2 Conversation Memory (anaphora resolution), M3.3 Web Speech API Voice STT/TTS, M3.4 Transparency Panel. | **Completed** | 47/47 | Multi-turn memory & voice cleaning | [📂 `milestone_3/`](milestone_3/) |
| **Milestone 4** | Query Analytics, Gap Detection & Optimization: M4.1 Persistent SQLite query analytics & 6-condition gap detection, M4.2 3-Domain testing (ML, Networks, Cyber), M4.3 Hybrid retrieval optimization, prompt & routing calibration, M4.4 Final technical documentation & demo. | **Completed** | 62/62 | **Recall@3: 100.0%, MRR: 1.000, Routing: 100.0%, Latency: 1.42 ms** | [📂 `milestone_4/`](milestone_4/) |

**Total Platform Automated Tests:** **161 / 161 PASSING** (0 failures, 0 errors).

---

## 🔬 Programmatic Milestone 4 Benchmark Results

*All metrics are calculated programmatically from real execution against canonical ground-truth datasets without hardcoding.*

| Evaluation Metric | Milestone 3 Baseline | Milestone 4 Optimized | Measured Net Gain | Raw Data Source |
|---|---|---|---|---|
| **Retrieval Recall@3** | 0.0% | **100.0%** | **+100.0%** | `evaluation/before_after_results.json` |
| **Retrieval Precision@3** | 0.0% | **95.0%** | **+95.0%** | `evaluation/before_after_results.json` |
| **Mean Reciprocal Rank (MRR)** | 0.000 | **1.000** | **+1.000** | `evaluation/before_after_results.json` |
| **Average Grounded Confidence** | 0.000 | **0.495** | **+0.495** | `evaluation/before_after_results.json` |
| **Average Pipeline Latency** | 1.09 ms | **1.08 ms** | -0.01 ms | `evaluation/before_after_results.json` |
| **Unanswered Query Rate** | 100.0% | **7.1%** | **-92.9%** | `evaluation/before_after_results.json` |
| **Query Routing Accuracy** | 92.9% | **100.0%** | **+7.1%** | `evaluation/routing_results.json` |
| **Voice Speech Sanitization** | PASS | **PASS** | Complete coverage | `evaluation/before_after_results.json` |

---

## 🚀 Quickstart: Running Milestone 4 (Final Platform)

### 1. Launch the Interactive Web Application
```bash
streamlit run milestone_4/app/main.py
```
Open `http://localhost:8501` to access the full multi-domain chat interface, voice interaction, transparency inspector, and interactive analytics dashboard.

### 2. Run the Full Test Suite
```bash
# Run Milestone 4 test suite via pytest
pytest -q milestone_4/tests

# Run cross-milestone regression suite (161 tests)
python3.12 -m unittest discover -s milestone_1/tests
python3.12 -m unittest discover -s milestone_2/tests
python3.12 -m unittest discover -s milestone_3/tests
python3.12 -m unittest discover -s milestone_4/tests
```

### 3. Run Programmatic Benchmarks
```bash
# Systematic retrieval parameter tuning (Chunking, Overlap, Thresholds, Top-K)
python3.12 milestone_4/evaluation/benchmark_retrieval.py

# Query intent and routing accuracy benchmark
python3.12 milestone_4/evaluation/benchmark_routing.py

# Component and end-to-end latency benchmark
python3.12 milestone_4/evaluation/benchmark_latency.py

# Genuine M3 Baseline vs M4 Optimized comparative benchmark
python3.12 milestone_4/evaluation/benchmark_before_after_m4.py
```

### 4. Run the Terminal Final Demonstration
```bash
python3.12 milestone_4/demo/run_final_demonstration.py
```

---

## 📖 Key Documentation Links
- **[Final Project Report](FINAL_REPORT.md)**: Full 27-section comprehensive technical report.
- **[Implementation Audit](milestone_4/docs/M4_IMPLEMENTATION_AUDIT.md)**: Detailed audit of repository findings and corrective remediation.
- **[Final Demonstration Script](milestone_4/docs/FINAL_DEMO_SCRIPT.md)**: Step-by-step walkthrough covering all 10 demonstration scenarios.
- **[Final Validation Report](milestone_4/docs/FINAL_VALIDATION_REPORT.md)**: Complete test counts and programmatic evaluation summaries.
- **[Final Architecture](milestone_4/docs/final-architecture.md)**: Detailed multi-agent and storage architecture.
- **[Optimization Results](milestone_4/docs/optimization-results.md)**: Raw measured benchmark data.
