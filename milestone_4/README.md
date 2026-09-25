# Milestone 4: Query Analytics, Knowledge Gap Detection, Multi-Domain Testing & Optimization

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status: Milestone 4 Completed](https://img.shields.io/badge/Milestone%204-Completed-brightgreen.svg)]()
[![Tests: 17/17 Passing](https://img.shields.io/badge/Tests-17%2F17%20Passing-success.svg)]()
[![Recall@3: 100%](https://img.shields.io/badge/Recall%403-100%25-success.svg)]()
[![MRR: 1.000](https://img.shields.io/badge/MRR-1.000-success.svg)]()

Milestone 4 completes the **AI-Based Knowledge Retrieval Platform with Query Resolution System**. It introduces enterprise **Query Analytics and Automated Knowledge Gap Detection**, conducts end-to-end testing across **three distinct knowledge domains** (Machine Learning, Computer Networks, and Cybersecurity), optimizes retrieval quality and voice interaction reliability, and provides full technical documentation and demonstration tooling.

---

## 📌 Milestone 4 Architecture & Flow

```text
               User Query (Text / Web Speech STT)
                               │
                               ▼
                 ┌───────────────────────────┐
                 │ Conversation Memory Agent │ (M3.2/M4.2)
                 │ • Anaphora Resolution     │
                 │ • Domain-Switch Isolation │
                 └─────────────┬─────────────┘
                               │ Rewritten Standalone Query
                               ▼
                 ┌───────────────────────────┐
                 │ Query Understanding Agent │ (M2.1/M4.2)
                 │ • Multi-Domain Detection  │
                 │ • Intent Classification   │
                 └─────────────┬─────────────┘
                               │
                   ┌───────────┴───────────┐
                   │                       │
          [Needs Clarification]     [Clear / Coherent]
                   │                       │
                   ▼                       ▼
        ┌──────────────────────┐  ┌──────────────────────┐
        │  Clarification Agent │  │   Retrieval Agent    │ (M4.3)
        │  • Polysemy Handling │  │ • Hybrid Dense-Lex.  │
        │  • Targeted Options  │  │ • Domain Affinity    │
        │  • Refinement Engine │  │ • Dynamic Top-K      │
        └──────────┬───────────┘  └──────────┬───────────┘
                   │ User Response           │
                   ▼                         ▼
        ┌──────────────────────┐  ┌──────────────────────┐
        │  Query Refinement &  │  │ Response Generation  │ (M4.3)
        │  State Transition    │  │ • Grounded Synthesis │
        │  (Re-enters Pipeline)│  │ • Transparency Panel │
        └──────────────────────┘  │ • Speech Sanitizer   │
                                  └──────────┬───────────┘
                                             │
                                             ▼
                             ┌──────────────────────────────┐
                             │ Analytics & Gap Detection    │ (M4.1)
                             │ • Independent SQLite Store   │
                             │ • Jaccard Semantic Clusters  │
                             │ • Severity & Remediation     │
                             │ • Interactive UI Dashboard   │
                             └──────────────────────────────┘
```

---

## 🎯 Deliverables & Key Modules

1. **M4.1 Query Analytics & Knowledge Gap Detection** ([`docs/m4.1-query-analytics-and-knowledge-gap-detection.md`](docs/m4.1-query-analytics-and-knowledge-gap-detection.md))
   - Independent persistent SQLite telemetry store (`data/analytics/analytics.db`).
   - Telemetry logging: timestamps, queries, domains, routing targets, retrieval scores, confidence, clarification flags, latency, and status (`RESOLVED`, `LOW_CONFIDENCE`, `UNANSWERED`, `CLARIFICATION_REQUESTED`).
   - Semantic clustering of low-confidence and unanswered queries to detect thematic knowledge gaps.
   - Empirical severity scoring ($S = \text{freq} \times (1 - \bar{C}) + 1.5 \times N_{\text{unanswered}}$) with targeted document ingestion recommendations.
   - Interactive Streamlit Analytics Dashboard with KPI metrics, domain charts, filterable audit trail, and CSV/JSON export.

2. **M4.2 End-to-End Testing Across 3 Domains** ([`docs/m4.2-end-to-end-multi-domain-testing.md`](docs/m4.2-end-to-end-multi-domain-testing.md))
   - Three distinct benchmark domains:
     - **Domain 1**: Machine Learning & AI (`ml_overview.txt`, `ml_algorithms.csv`, `deep_learning_and_transformers.txt`)
     - **Domain 2**: Computer Networks & Cloud (`computer_networks.txt`, `network_protocols.csv`, `cloud_distributed_systems.txt`)
     - **Domain 3**: Cybersecurity & InfoSec (`cybersecurity_fundamentals.txt`, `cryptography_and_zero_trust.csv`, `incident_response_and_threats.txt`)
   - 17 automated tests verifying factual, procedural, comparative, ambiguous, multi-turn, cross-domain switching, voice, and transparency behaviors.

3. **M4.3 System Optimization & Evaluation Benchmarks** ([`docs/m4.3-retrieval-prompt-routing-voice-optimization.md`](docs/m4.3-retrieval-prompt-routing-voice-optimization.md))
   - Hybrid dense-lexical scoring ($0.55 \times S_{\text{vector}} + 0.45 \times S_{\text{lexical}}$).
   - Chunking tuning: 800 chars / 150 overlap achieves **100% Recall@3** and **1.000 MRR**.
   - Dynamic Top-$K$ scaling and domain affinity scoring boost ($+0.05$).
   - VoiceModule natural speech sanitization and cross-browser resilience.
   - Before-vs-After benchmark suite measuring latency (-33%), recall (+55.6%), and confidence (+0.268).

4. **M4.4 Technical Documentation & Demonstration** ([`docs/MILESTONE_4_PROJECT_REPORT.md`](docs/MILESTONE_4_PROJECT_REPORT.md) & [`docs/FINAL_DEMONSTRATION_GUIDE.md`](docs/FINAL_DEMONSTRATION_GUIDE.md))
   - Comprehensive final project report covering problem statement, system architecture, methodology, empirical results, limitations, and future roadmap.
   - Standalone CLI automated demonstration script (`demo/run_final_demonstration.py`).

---

## 🚀 Quickstart & Usage

### 1. Run the Interactive Streamlit Web Application

```bash
cd milestone_4

# Launch Streamlit App
streamlit run app/main.py
```

### 2. Run the Automated Test Suite (17 Tests)

```bash
python3 -m unittest discover -s tests -v
```

### 3. Run the Retrieval Parameter Optimization Benchmark

```bash
python3 evaluation/evaluate_retrieval_optimization.py
```

### 4. Run the Before-vs-After Optimization Benchmark

```bash
python3 evaluation/benchmark_before_after_m4.py
```

### 5. Run the Final Multi-Domain Demonstration Script

```bash
python3 demo/run_final_demonstration.py
```

---

## 📂 Directory Structure

```text
milestone_4/
├── app/
│   ├── __init__.py
│   ├── main.py                     # Streamlit app (Multi-Domain Chat & Analytics Dashboard)
│   ├── agents/                     # Multi-Agent Subsystem
│   │   ├── __init__.py
│   │   ├── models.py               # Telemetry, Transparency, and Agent Schemas
│   │   ├── query_understanding.py  # Intent & Domain Classification Agent
│   │   ├── retrieval.py            # Hybrid Dense-Lexical Retrieval Agent
│   │   ├── response_generation.py  # Grounded Synthesis & Transparency Agent
│   │   ├── clarification.py        # Context-Aware Clarification Agent
│   │   ├── memory.py               # Conversation Memory & Domain-Switch Isolation Agent
│   │   ├── voice.py                # Multimodal Voice & Speech Sanitizer Module
│   │   └── orchestrator.py         # Multi-Agent Orchestrator with Analytics Tracking
│   ├── analytics/                  # M4.1 Query Analytics & Gap Detection
│   │   ├── __init__.py
│   │   ├── models.py               # Telemetry and Gap Report Schemas
│   │   ├── storage.py              # Independent SQLite Storage Backend
│   │   ├── gap_detector.py         # Thematic Keyword/N-Gram Clustering Engine
│   │   └── tracker.py              # Orchestration Middleware Tracker
│   ├── components/                 # Reusable Streamlit UI Components
│   │   ├── __init__.py
│   │   ├── analytics_dashboard.py  # Analytics KPI & Gap Radar Component
│   │   ├── transparency_panel.py   # Response Transparency Component
│   │   └── voice_interface.py      # Web Speech API STT/TTS Component
│   ├── embeddings/                 # Sentence Transformers with 384-d Fallback
│   ├── ingestion/                  # Multi-Format Pipeline (PDF, DOCX, TXT, CSV)
│   └── vectorstore/                # ChromaDB Vector Store with Cosine Fallback
├── data/
│   ├── analytics/                  # SQLite telemetry database
│   ├── chroma/                     # ChromaDB persistent index
│   └── sample_docs/                # 9 Benchmark Documents across 3 Domains
├── demo/
│   └── run_final_demonstration.py  # Automated Terminal Showcase
├── docs/                           # Technical Specifications & Project Report
│   ├── m4.1-query-analytics-and-knowledge-gap-detection.md
│   ├── m4.2-end-to-end-multi-domain-testing.md
│   ├── m4.3-retrieval-prompt-routing-voice-optimization.md
│   ├── m4.4-technical-architecture-and-data-models.md
│   ├── MILESTONE_4_PROJECT_REPORT.md
│   └── FINAL_DEMONSTRATION_GUIDE.md
├── evaluation/                     # Quantitative Benchmarks
│   ├── evaluate_retrieval_optimization.py
│   └── benchmark_before_after_m4.py
├── tests/                          # 17 Automated Unit & Integration Tests
│   ├── __init__.py
│   ├── test_analytics.py
│   ├── test_gap_detection.py
│   ├── test_multi_domain.py
│   └── test_end_to_end_m4.py
├── requirements.txt
├── .env.example
└── README.md
```
