# System Architecture

## 1. System Overview

The proposed system is a Retrieval-Augmented Generation (RAG) based multi-agent knowledge assistant. It accepts user queries through text or voice, retrieves relevant information from an indexed knowledge base, and generates grounded responses using an LLM.

The architecture separates knowledge-base ingestion from query-time retrieval. This allows documents to be processed once and searched efficiently across subsequent user queries.

## 2. Architecture Layers

### Layer 1 — Presentation

- User Interface
- Text query input
- Voice input
- Results and response interface
- Text-to-Speech output

### Layer 2 — API and Orchestration

- Backend/API layer
- Multi-Agent Orchestrator

The API layer receives requests and the orchestrator determines which agents should participate in resolving the query.

### Layer 3 — AI Agent Layer

- Query Understanding Agent
- Retrieval Agent
- Response Generation Agent
- Clarification Agent
- Conversation Memory Agent

### Layer 4 — Knowledge and RAG Layer

- Document upload and storage
- Text extraction
- Cleaning and normalization
- Chunking
- Embedding generation
- Vector database
- Semantic search

### Layer 5 — AI and Data Services

- LLM/Foundation Model
- Document metadata database
- Vector database
- Document storage

## 3. Knowledge Base Ingestion Flow

```text
PDF / DOCX / TXT / CSV
          ↓
Document Upload
          ↓
Document Storage
          ↓
Text Extraction
          ↓
Cleaning & Normalization
          ↓
Chunking
          ↓
Embedding Generation
          ↓
Vector Database
          +
Document / Chunk Metadata
```

Each chunk is stored with metadata such as document ID, file name, source type, page or row information where applicable, chunk index, and embedding.

## 4. Query-Time RAG Flow

```text
User Query
    ↓
Backend API
    ↓
Multi-Agent Orchestrator
    ↓
Query Understanding Agent
    ↓
Query Clear?
   /     \
 NO       YES
 ↓         ↓
Clarification   Retrieval Agent
Agent              ↓
 ↓             Vector Database
User               ↓
               Relevant Chunks
                    ↓
           Response Generation Agent
                    ↓
                   LLM
                    ↓
          Response + Source Citations
                    ↓
       Confidence / Retrieval Transparency
                    ↓
             Results Interface
```

The Conversation Memory Agent supplies relevant previous conversation context when required.

## 5. Agent Responsibilities

| Agent | Responsibility |
|---|---|
| Query Understanding Agent | Identifies intent, entities, query type, and ambiguity; can reformulate the query for retrieval. |
| Retrieval Agent | Converts the query into a searchable representation and retrieves the most relevant chunks from the vector database. |
| Response Generation Agent | Combines the retrieved context with the user query and conversation context to produce a grounded response. |
| Clarification Agent | Detects insufficient or ambiguous queries and asks the user for the missing information. |
| Conversation Memory Agent | Stores and retrieves relevant conversation context to support multi-turn interactions. |

## 6. Multi-Agent Orchestration

The orchestrator acts as the control layer between the API and the agents.

A typical interaction is:

1. Receive the user query.
2. Obtain relevant conversation context.
3. Analyze the query.
4. If the query is ambiguous, route it to the Clarification Agent.
5. If the query is clear, route it to the Retrieval Agent.
6. Retrieve the highest-relevance knowledge chunks.
7. Pass retrieved context to the Response Generation Agent.
8. Generate a grounded response using the LLM.
9. Attach source citations and retrieval/confidence information.
10. Return the result to the user.

## 7. Voice Interaction

The Web Speech API is planned for browser-based voice interaction:

```text
Voice Input
    ↓
Speech-to-Text
    ↓
User Query
    ↓
RAG / Multi-Agent Pipeline
    ↓
Generated Response
    ↓
Text-to-Speech
    ↓
Voice Output
```

## 8. Design Decisions

### RAG instead of LLM-only generation

The system uses retrieval to ground responses in the uploaded knowledge base and reduce unsupported answers.

### Vector search

Semantic vector search is used because relevant documents may use different wording from the user's query.

### Chunk-based indexing

Documents are divided into smaller chunks so retrieval can return focused context rather than entire documents.

### Metadata-aware retrieval

Document and chunk metadata are stored with embeddings so retrieved information can be traced back to its source.

### Modular agents

Each agent has a focused responsibility, making the architecture easier to test, replace, and extend in later milestones.

## 9. Future Extensions

The architecture can later include:

- Hybrid keyword + semantic retrieval
- Reranking models
- Advanced confidence scoring
- Conversation summarization
- Knowledge-gap detection
- Automatic knowledge-base updates
- Query analytics
- Streaming responses
- Authentication and authorization
- Production document storage
