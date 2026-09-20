# Milestone 3: Advanced Clarification, Conversation Memory, Voice & Transparency

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status: Milestone 3 Completed](https://img.shields.io/badge/Milestone%203-Completed-brightgreen.svg)]()
[![Tests: 40/40 Passing](https://img.shields.io/badge/Tests-40%2F40%20Passing-success.svg)]()

Milestone 3 completes the AI-Based Knowledge Retrieval Platform with multi-turn conversational memory, intelligent clarification dialogues, multimodal voice input & text-to-speech, and deep response transparency.

---

## 📌 Milestone 3 Subsystems & Architecture

```text
                User Query (Text / Web Speech STT)
                              │
                              ▼
                ┌───────────────────────────┐
                │ Conversation Memory Agent │ (M3.2)
                │ • Anaphora / Pronoun Res. │
                │ • Context vs Topic Switch │
                │ • Sliding Window History  │
                └─────────────┬─────────────┘
                              │ Rewritten Standalone Query
                              ▼
                ┌───────────────────────────┐
                │ Query Understanding Agent │
                └─────────────┬─────────────┘
                              │
                              ▼
                ┌───────────────────────────┐
                │ Clarification Agent (M3.1)│
                │ • Context-Aware Polysemy  │
                │ • Multi-Part Analyzer     │
                │ • Targeted Follow-up Qs   │
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
       │ • Interactive Options│  │   (Dense Search &    │
       │ • Freeform Refinement│  │    Dynamic Top-K)    │
       └──────────┬───────────┘  └──────────┬───────────┘
                  │ User Response           │
                  ▼                         ▼
       ┌──────────────────────┐  ┌──────────────────────┐
       │ Query Refinement &   │  │ Response Generation  │
       │ State Transition     │  │ • Grounded Synthesis │
       │ (PENDING -> RESOLVED)│  │ • Transparency Panel │ (M3.4)
       └──────────┬───────────┘  │ • Speech Sanitizer   │ (M3.3)
                  │              └──────────┬───────────┘
                  │                         │
                  └──────────────► Re-enters Retrieval Pipeline
```

---

## 🎯 Deliverables & Modules

1. **M3.1 Clarification Agent** ([`docs/m3.1-clarification-agent.md`](docs/m3.1-clarification-agent.md))
   - Identifies ambiguous, incomplete, or multi-domain queries without false positives.
   - Decomposes multi-part queries into sub-questions.
   - Generates targeted follow-up questions and relevant options.
   - Hardened `PENDING` -> `RESOLVED` lifecycle.

2. **M3.2 Conversation Memory Agent** ([`docs/m3.2-conversation-memory-agent.md`](docs/m3.2-conversation-memory-agent.md))
   - Multi-turn conversation turn buffer (`ConversationSession`, `ConversationTurn`).
   - Anaphora and pronoun resolution (`"What are its advantages?"` -> `"What are the advantages of TCP?"`).
   - Context continuation vs context switching detection.
   - Sliding window pruning.

3. **M3.3 Voice Input & Text-to-Speech** ([`docs/m3.3-voice-and-tts-module.md`](docs/m3.3-voice-and-tts-module.md))
   - HTML5 Web Speech API SpeechRecognition microphone with live audio waveforms.
   - Browser-native speech synthesis with Play/Pause/Resume/Stop controls.
   - `VoiceModule` speech cleaner stripping markdown, code blocks, links, and citation markers.

4. **M3.4 Response Transparency Panel** ([`docs/m3.4-response-transparency-panel.md`](docs/m3.4-response-transparency-panel.md))
   - Interactive evidence inspection panel in Streamlit UI.
   - Empirical confidence calculation: $0.70 \times \text{Top-Score} + 0.30 \times \text{Avg-Top-}K$.
   - Ranked chunk inspector with progress bars and citation verification.

---

## 🚀 Quickstart & Usage

### 1. Run the Streamlit Multi-Agent Application

```bash
cd milestone_3

# Run Streamlit Web Application
streamlit run app/main.py
```

### 2. Run the Automated Test Suite

```bash
python3 -m unittest discover -s tests -v
```

### 3. Run the Comprehensive Evaluation Benchmark

```bash
python3 evaluation/evaluate_m3_memory_and_transparency.py
```

---

## 📂 Directory Structure

```text
milestone_3/
├── app/
│   ├── __init__.py
│   ├── main.py                     # Streamlit app (Memory, Voice, Clarification, Transparency)
│   ├── agents/                     # Multi-Agent Subsystem
│   │   ├── __init__.py
│   │   ├── models.py               # Schemas (Memory, Voice, Transparency, Clarification)
│   │   ├── clarification.py        # M3.1 Clarification Agent
│   │   ├── memory.py               # M3.2 Conversation Memory Agent
│   │   ├── voice.py                # M3.3 Voice & TTS Module
│   │   ├── query_understanding.py  # Query Understanding Agent
│   │   ├── retrieval.py            # Retrieval Agent
│   │   ├── response_generation.py  # Response Generation Agent
│   │   └── orchestrator.py         # Multi-Agent Orchestrator
│   ├── components/                 # Reusable Streamlit UI Components
│   │   ├── __init__.py
│   │   ├── voice_interface.py      # Web Speech API STT/TTS component
│   │   └── transparency_panel.py   # Response Transparency Panel
│   ├── embeddings/                 # Embedder
│   ├── ingestion/                  # Multi-format document ingestion pipeline
│   └── vectorstore/                # ChromaDB vector store
├── data/
│   ├── chroma/                     # ChromaDB persistent storage
│   ├── uploads/                    # Uploads directory
│   └── sample_docs/                # Benchmark datasets (ML & Networks)
├── docs/                           # Technical documentation
│   ├── m3.1-clarification-agent.md
│   ├── m3.2-conversation-memory-agent.md
│   ├── m3.3-voice-and-tts-module.md
│   ├── m3.4-response-transparency-panel.md
│   └── MILESTONE_3_STATUS.md
├── evaluation/
│   └── evaluate_m3_memory_and_transparency.py
├── tests/                          # 40 Automated unit and integration tests
│   ├── __init__.py
│   ├── test_clarification.py
│   ├── test_end_to_end_clarification.py
│   ├── test_memory.py
│   ├── test_voice.py
│   ├── test_transparency.py
│   └── test_end_to_end_m3.py
├── requirements.txt
├── .env.example
└── README.md
```

