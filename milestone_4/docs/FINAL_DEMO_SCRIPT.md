# Milestone 4 — Final Demonstration Script

This script provides a step-by-step walkthrough to showcase the AI-Based Knowledge Retrieval Platform across multiple loaded domains, voice interactions, multi-turn memory, transparency inspectability, and persistent query analytics.

---

## Pre-requisites: Launching the Application
Execute the Streamlit web application from the terminal:
```bash
streamlit run milestone_4/app/main.py
```
Open your browser at `http://localhost:8501`.

---

## Step 1: Load Knowledge Base Domains
1. Navigate to the **Knowledge Management / Ingestion** section in the sidebar.
2. Verify all three canonical domains are loaded:
   - **Machine Learning / AI**: `ml_fundamentals.txt`, `deep_learning_and_nlp.txt`
   - **Computer Networks / Cloud**: `computer_networks.txt`, `network_protocols.csv`, `cloud_distributed_systems.txt`
   - **Cybersecurity**: `cybersecurity_fundamentals.txt`, `cryptography_and_zero_trust.csv`, `incident_response_and_threats.txt`
3. Click **Ingest & Re-Index Documents**. Verify confirmation showing documents chunked and indexed.

---

## Step 2: Factual Query Demonstration
- **Input Query**: `What is TCP?`
- **Expected System Behavior**:
  - Query classified as `FACTUAL`, domain: `computer_networks`.
  - Retrieved chunk: `computer_networks.txt` (Chunk describing Transmission Control Protocol, 3-way handshake, reliability).
  - Grounded answer generated with inline citations `[computer_networks.txt, Chunk 1]`.
  - Confidence Badge: **HIGH** (`>= 0.70`).
  - Open **Transparency Panel**: Inspect similarity scores, chunk text, and source attribution.

---

## Step 3: Procedural Query Demonstration
- **Input Query**: `How does DNS resolution work?`
- **Expected System Behavior**:
  - Query classified as `PROCEDURAL`, domain: `computer_networks`.
  - Resolution path: `Query Understanding -> Retrieval -> Step-oriented response`.
  - Answer formatted into sequential numbered steps (Client Cache -> Recursive Resolver -> Root Server -> TLD Server -> Authoritative Server).
  - Confidence: **HIGH**.

---

## Step 4: Comparative Query Demonstration
- **Input Query**: `Compare TCP and UDP.`
- **Expected System Behavior**:
  - Query classified as `COMPARATIVE`, domain: `computer_networks`.
  - Resolution path: `Query Understanding -> Multi-document retrieval -> Comparison response`.
  - Answer synthesizes comparison across reliability, connection state, overhead, and packet header structures.

---

## Step 5: Ambiguous Query & Clarification Demonstration
- **Input Query**: `Explain tree.`
- **Expected System Behavior**:
  - Ambiguity detected: `tree` is a polysemous concept spanning ML (Decision Trees), Networks (Spanning Tree Protocol), and Security (Merkle Trees).
  - Status: `clarification_requested`.
  - An interactive clarification card appears with options:
    1. Decision Trees (Machine Learning / Classification)
    2. Spanning Tree Protocol (Networking / STP)
    3. Binary Search Tree / Merkle Tree (Data Structures / Security)
- **User Action**: Click **Decision Trees (Machine Learning)**.
- **Result**: Clarification resolves; system retrieves from `ml_fundamentals.txt` and presents a detailed overview of Decision Trees.

---

## Step 6: Multi-Turn Memory & Pronominal Coreference
- **Turn 1 Input**: `What is TCP?` -> System answers with TCP definition.
- **Turn 2 Input**: `What are its advantages?`
- **Expected System Behavior**:
  - Conversation Memory Agent resolves pronoun "its" to antecedent entity "TCP".
  - Effective query generated: `What are the advantages of TCP?`.
  - Answer details connection-oriented reliability, ordered delivery, and flow control.
  - Transparency panel marks `is_multi_turn: True` and displays rewritten query.

---

## Step 7: Cross-Domain Context Switching
Demonstrate clean transitions without stale context contamination:
1. **Query 1 (ML)**: `What is Supervised Learning?` -> Domain: `machine_learning`.
2. **Query 2 (Networks)**: `Explain BGP routing.` -> Domain switches to `computer_networks`.
3. **Query 3 (Cybersecurity)**: `Define the CIA Triad.` -> Domain switches to `cybersecurity`.
- Verify transparency panel reflects zero chunk leakage from prior domains.

---

## Step 8: Unknown Query & Knowledge Gap Logging
- **Input Query**: `What is quantum annealing?`
- **Expected System Behavior**:
  - Query executed; retrieval fails to find relevant chunks meeting the 0.35 threshold.
  - Grounded generator outputs an honest, non-hallucinated response:
    > "I do not have sufficient evidence in the knowledge base to answer this query."
  - Status logged as `UNANSWERED`.
  - Centralized confidence: `0.0` (`NONE`).
  - Recorded in SQLite as an active knowledge gap candidate.

---

## Step 9: Voice Interaction Demonstration
1. Click the **Microphone Icon** (or trigger voice test input).
2. Speak: `"Explain AES encryption"`
3. **Speech-to-Text (STT)**: Transcribes speech, removes background artifacts, normalizes input.
4. **Resolution**: System retrieves AES details from `cryptography_and_zero_trust.csv`.
5. **Text-to-Speech (TTS)**: Click the **Listen (TTS)** button.
   - Speech sanitizer strips code blocks, URLs, and citation brackets, reading a natural, fluent spoken answer.

---

## Step 10: Persistent Analytics Dashboard & Knowledge Gaps
1. Switch to the **Analytics Dashboard** tab in the UI.
2. Inspect the **7 KPI Cards**:
   - Total Queries
   - Answered Queries
   - Unanswered Queries
   - Low-Confidence Queries
   - Clarification Requests
   - Average Confidence Score
   - Average Latency
3. Inspect **Distribution Visualizations**:
   - Query Volume over time
   - Confidence distribution (HIGH / MED / LOW / NONE)
   - Query Type breakdown
   - Domain distribution
4. Inspect the **Knowledge Gap Radar**:
   - Review detected gap for `quantum annealing` showing frequency, zero confidence, and recommended action: "Add canonical documentation covering Quantum Annealing".
5. Test **Interactive Multi-Dimensional Filtering**:
   - Filter by domain (`computer_networks`), status (`RESOLVED`), or confidence (`HIGH`).
6. Click **Export Analytics (CSV / JSON)** to download persistent audit logs.
