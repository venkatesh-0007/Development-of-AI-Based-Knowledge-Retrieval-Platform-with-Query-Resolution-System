# Milestone 3: Advanced Clarification, Conversation Memory, Voice & Transparency

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status: M3.1 Completed](https://img.shields.io/badge/M3.1-Completed%20%26%20Hardened-brightgreen.svg)]()
[![Tests: 24/24 Passing](https://img.shields.io/badge/Tests-24%2F24%20Passing-success.svg)]()

Milestone 3 enhances the AI-Based Knowledge Retrieval Platform with conversational capabilities, intelligent query clarification, voice interaction, and response transparency.

---

## 📌 Deliverables & Architecture

```text
               User Query (Text / Voice)
                          │
                          ▼
            ┌───────────────────────────┐
            │ Query Understanding Agent │
            └─────────────┬─────────────┘
                          │
                          ▼
            ┌───────────────────────────┐
            │ Clarification Agent (M3.1)│
            │ • Context-Aware Polysemy  │
            │ • Antecedent Pronoun Check│
            │ • Multi-Part Analyzer     │
            │ • Targeted Follow-ups     │
            │ • Hardened State Tracker  │
            │ • Query Refinement Engine │
            └─────────────┬─────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
     [Needs Clarification]     [Clear / Coherent]
              │                       │
              ▼                       ▼
   ┌──────────────────────┐  ┌──────────────────────┐
   │ Clarification Dialog │  │   Retrieval Agent    │
   │ • Reason & Follow-up │  │   (Dense Search &    │
   │ • Targeted Options   │  │    Dynamic Top-K)    │
   └──────────┬───────────┘  └──────────┬───────────┘
              │ User Response           │
              ▼                         ▼
   ┌──────────────────────┐  ┌──────────────────────┐
   │ Query Refinement &   │  │ Response Generation  │
   │ State Transition     │  │ (Grounded Synthesis  │
   │ (PENDING -> RESOLVED)│  │  & Source Citations) │
   └──────────┬───────────┘  └──────────────────────┘
              │
              └──────────────► Re-enters M2 Pipeline (Retrieval -> Response Generation)
```

### Key Modules

1. **M3.1 Clarification Agent** ([`docs/m3.1-clarification-agent.md`](docs/m3.1-clarification-agent.md))
   - Identifies genuinely ambiguous or incomplete queries while avoiding false positives on clear queries.
   - Decomposes multi-part queries and distinguishes coherent sub-questions from conflicting ones.
   - Generates targeted follow-up questions asking strictly for missing information.
   - Hardens state management with strict `PENDING` -> `RESOLVED` lifecycle and input validation.
   - Fuses original context and clarification into a refined standalone query and re-enters the M2 pipeline.

---

## 🚀 Quickstart & Usage

### 1. Run the Streamlit Application

```bash
cd milestone_3

# Run Streamlit Web Application
streamlit run app/main.py
```

### 2. Run the Automated Test Suite

```bash
python3 -m unittest discover -s tests -v
```

---

## 📂 Directory Structure

```text
milestone_3/
├── app/
│   ├── __init__.py
│   ├── main.py                     # Streamlit interactive app with clarification dialogues
│   ├── agents/                     # Multi-Agent Subsystem
│   │   ├── __init__.py
│   │   ├── models.py               # Schemas (ClarificationRequest, ClarificationType, etc.)
│   │   ├── clarification.py        # M3.1 Clarification Agent
│   │   ├── query_understanding.py  # Query Understanding Agent
│   │   ├── retrieval.py            # Retrieval Agent
│   │   ├── response_generation.py  # Response Generation Agent
│   │   └── orchestrator.py         # Multi-Agent Orchestrator with clarification loop
│   ├── embeddings/                 # Sentence Transformers embedder
│   ├── ingestion/                  # Multi-format document parser
│   └── vectorstore/                # ChromaDB vector store
├── data/
│   ├── chroma/                     # Vector store storage path
│   ├── uploads/                    # Upload directory
│   └── sample_docs/                # Benchmark datasets (ML & Networks)
├── docs/                           # Technical documentation
│   ├── m3.1-clarification-agent.md
│   └── MILESTONE_3_STATUS.md
├── tests/                          # Automated unit and integration tests (24 passing)
│   ├── __init__.py
│   ├── test_clarification.py
│   └── test_end_to_end_clarification.py
├── requirements.txt
├── .env.example
└── README.md
```
