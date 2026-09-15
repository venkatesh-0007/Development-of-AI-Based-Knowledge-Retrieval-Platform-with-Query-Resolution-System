# Milestone 2: Multi-Agent Query Resolution & Response Generation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status: Milestone 2 Completed](https://img.shields.io/badge/Milestone%202-Completed-brightgreen.svg)]()
[![Tests: 24/24 Passing](https://img.shields.io/badge/Tests-24%2F24%20Passing-success.svg)]()

Milestone 2 implements an end-to-end multi-agent Query Resolution and Grounded Response Generation platform. It features specialized AI agents for intent classification, semantic vector search, confidence filtering, grounded answer synthesis with source attribution, and sequential orchestration.

---

## 📌 Deliverables & Architecture

```text
               ┌────────────────────────────────────────────────────────┐
               │                Presentation Layer                      │
               │         • Streamlit Interactive Web App                │
               └───────────────────────────┬────────────────────────────┘
                                           │
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │        Multi-Agent Orchestrator (M2.4)                 │
               └──────┬────────────────────┬────────────────────┬───────┘
                      │                    │                    │
                      ▼                    ▼                    ▼
     ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
     │ Query Understanding  │  │   Retrieval Agent    │  │  Response Generation │
     │     Agent (M2.1)     │  │        (M2.2)        │  │     Agent (M2.3)     │
     │ • Intent Classifier  │  │ • Semantic Search    │  │ • Grounded Synthesis │
     │ • Ambiguity Detector │  │ • Confidence Filter  │  │ • Confidence Meter   │
     │ • Entity Extractor   │  │ • Top-K Ranking      │  │ • Source Attribution │
     └──────────────────────┘  └──────────┬───────────┘  └──────────────────────┘
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │          ChromaDB Vector Store        │
                      │  (all-MiniLM-L6-v2 Embeddings + Meta) │
                      └───────────────────────────────────────┘
```

### Key Modules

1. **M2.1 Query Understanding Agent** ([`docs/m2.1-query-understanding.md`](docs/m2.1-query-understanding.md))
   - Classifies queries into `factual`, `procedural`, `comparative`, and `ambiguous`.
   - Flags underspecified queries and routes to clarification.
   - Extracts domain keywords and reformulates queries for retrieval.

2. **M2.2 Retrieval Agent** ([`docs/m2.2-retrieval.md`](docs/m2.2-retrieval.md))
   - Executes dense vector similarity search against ChromaDB using Sentence Transformers.
   - Dynamically selects Top-K (Top-3 for factual, Top-5 for procedural/comparative).
   - Applies configurable confidence threshold (`0.45`) to filter out noise.

3. **M2.3 Response Generation Agent** ([`docs/m2.3-response-generation.md`](docs/m2.3-response-generation.md))
   - Synthesizes grounded responses strictly using retrieved evidence.
   - Formats outputs according to intent (Definitions, Numbered Steps, Comparative Tables).
   - Computes application-level confidence score & categorical indicator (`HIGH`, `MEDIUM`, `LOW`, `NONE`).
   - Attaches structured source attributions with relevance scores and chunk snippets.

4. **M2.4 Multi-Agent Orchestration Layer** ([`docs/m2.4-orchestration.md`](docs/m2.4-orchestration.md))
   - Coordinates sequential data flow between agents.
   - Immediate clarification bypass for ambiguous queries.
   - Full latency profiling and exception handling.

---

## 🚀 Quickstart & Usage

### 1. Setup & Installation

```bash
cd milestone_2

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Streamlit Application

```bash
streamlit run app/main.py
```

### 3. Run the Automated Test Suite

```bash
python3 -m unittest discover -s tests -v
```

---

## 📂 Directory Structure

```text
milestone_2/
├── app/
│   ├── __init__.py
│   ├── main.py                     # Streamlit interactive application
│   ├── agents/                     # Multi-Agent Subsystem
│   │   ├── __init__.py
│   │   ├── models.py               # Data schemas & enums
│   │   ├── query_understanding.py  # M2.1 Query Understanding Agent
│   │   ├── retrieval.py            # M2.2 Retrieval Agent
│   │   ├── response_generation.py  # M2.3 Response Generation Agent
│   │   ├── clarification.py        # Clarification Handler
│   │   └── orchestrator.py         # M2.4 Multi-Agent Orchestrator
│   ├── embeddings/                 # Sentence Transformers embedder
│   ├── ingestion/                  # Multi-format document parser
│   └── vectorstore/                # ChromaDB client
├── data/
│   ├── chroma/.gitkeep             # Vector database storage path
│   ├── uploads/.gitkeep            # Upload directory
│   └── sample_docs/                # Benchmark datasets (ML & Networks)
├── docs/                           # Technical documentation & benchmarks
│   ├── m2.1-query-understanding.md
│   ├── m2.2-retrieval.md
│   ├── m2.3-response-generation.md
│   ├── m2.4-orchestration.md
│   └── MILESTONE_2_STATUS.md
├── tests/                          # Automated unit tests (24 passing)
│   ├── test_query_understanding.py
│   ├── test_retrieval.py
│   ├── test_response_generation.py
│   └── test_orchestrator.py
├── requirements.txt
├── .env.example
└── README.md
```
