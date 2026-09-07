# Technology Stack

## Proposed Stack

| Layer / Component | Technology | Purpose |
|---|---|---|
| Programming Language | Python | Core application and RAG implementation |
| Backend API | FastAPI | REST API and service layer |
| UI / Prototype | Streamlit | Rapid knowledge-base upload and query interface |
| RAG Framework | LangChain | RAG pipeline and component integration |
| PDF Processing | PyMuPDF | PDF text extraction |
| DOCX Processing | python-docx | DOCX text extraction |
| TXT Processing | Python standard library | Plain-text ingestion |
| CSV Processing | Pandas | CSV parsing and tabular document processing |
| Embeddings | Sentence Transformers | Generate semantic vector representations |
| Initial Embedding Model | all-MiniLM-L6-v2 | Lightweight baseline embedding model |
| Vector Database | ChromaDB | Store embeddings and perform vector similarity search |
| LLM / Foundation Model | Configurable | Generate grounded natural-language responses |
| Voice | Web Speech API | Browser speech-to-text and text-to-speech |
| Testing | Pytest | Unit and pipeline testing |
| Version Control | Git + GitHub | Source control and milestone submission |

## Technology Selection Rationale

### Python

Python provides a mature ecosystem for document processing, embeddings, vector databases, LLM applications, and experimentation.

### FastAPI

FastAPI provides a lightweight, typed backend API that can later support a separate web or mobile frontend.

### LangChain

LangChain can simplify integration between document loaders, chunking, embeddings, retrieval, and generation while allowing individual components to remain replaceable.

### Sentence Transformers

Sentence Transformers provides efficient local embedding generation and is suitable for establishing a reproducible baseline during retrieval evaluation.

### ChromaDB

ChromaDB provides a simple vector-store implementation suitable for local development and Milestone 1 experimentation.

### Web Speech API

The Web Speech API provides browser-side speech recognition and speech synthesis without requiring a separate speech service for the initial prototype.

## Configuration

LLM provider, model, embedding model, and vector-store paths should be configurable through environment variables rather than hard-coded in source code.

## Future Technology Changes

The stack may be updated in later milestones based on retrieval accuracy, latency, scalability, deployment requirements, and mentor/project requirements.
