# Data Models

The following schemas define the core entities required by the RAG pipeline.

## 1. Document

Represents an uploaded source document.

```json
{
  "document_id": "doc_001",
  "filename": "machine_learning.pdf",
  "file_type": "pdf",
  "domain": "machine_learning",
  "file_size": 102400,
  "uploaded_at": "2026-08-28T10:00:00",
  "status": "indexed"
}
```

## 2. Chunk

Represents a searchable segment extracted from a document.

```json
{
  "chunk_id": "chunk_001",
  "document_id": "doc_001",
  "chunk_index": 0,
  "text": "Example chunk text...",
  "metadata": {
    "page": 1,
    "source": "machine_learning.pdf"
  }
}
```

## 3. Embedding

Represents the vector generated from a chunk.

```json
{
  "embedding_id": "emb_001",
  "chunk_id": "chunk_001",
  "model": "all-MiniLM-L6-v2",
  "dimension": 384
}
```

The actual vector is stored by the vector database.

## 4. Query

Represents a user request.

```json
{
  "query_id": "query_001",
  "session_id": "session_001",
  "text": "What is overfitting?",
  "query_type": "factual",
  "created_at": "2026-08-28T10:05:00"
}
```

## 5. Retrieval Result

Represents a chunk returned by semantic search.

```json
{
  "query_id": "query_001",
  "chunk_id": "chunk_001",
  "rank": 1,
  "similarity_score": 0.87,
  "document_id": "doc_001"
}
```

## 6. Response

Represents the final generated answer.

```json
{
  "response_id": "response_001",
  "query_id": "query_001",
  "answer": "Overfitting occurs when...",
  "sources": [
    {
      "document_id": "doc_001",
      "chunk_id": "chunk_001"
    }
  ],
  "confidence": 0.84
}
```

## Entity Relationships

```text
Document
   │
   └── 1 : Many ──> Chunk
                         │
                         └── 1 : 1 ──> Embedding

Query
   │
   └── 1 : Many ──> Retrieval Result
                         │
                         └── references ──> Chunk

Query
   │
   └── 1 : 1 ──> Response
                         │
                         └── references ──> Source Chunks
```
