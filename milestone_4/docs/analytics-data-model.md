# Analytics Data Model Specification

## 1. Relational Schema (SQLite)

### Table: `query_logs`
```sql
CREATE TABLE query_logs (
    query_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    query_text TEXT NOT NULL,
    effective_query TEXT NOT NULL,
    domain TEXT NOT NULL,
    query_type TEXT NOT NULL,
    route_target TEXT NOT NULL,
    top_score REAL NOT NULL,
    avg_score REAL NOT NULL,
    confidence_score REAL NOT NULL,
    confidence_level TEXT NOT NULL,
    has_sufficient_evidence INTEGER NOT NULL,
    requires_clarification INTEGER NOT NULL,
    status TEXT NOT NULL,
    retrieved_chunk_ids TEXT NOT NULL,
    sources TEXT NOT NULL,
    clarification_id TEXT,
    response_text TEXT,
    latency_ms REAL NOT NULL,
    session_id TEXT,
    input_modality TEXT NOT NULL,
    resolution_path TEXT DEFAULT '',
    agent_sequence TEXT DEFAULT '[]',
    source_count INTEGER DEFAULT 0,
    error TEXT
);

CREATE INDEX idx_query_logs_timestamp ON query_logs(timestamp);
CREATE INDEX idx_query_logs_domain ON query_logs(domain);
CREATE INDEX idx_query_logs_status ON query_logs(status);
CREATE INDEX idx_query_logs_confidence ON query_logs(confidence_level);
```

### Table: `knowledge_gaps`
```sql
CREATE TABLE knowledge_gaps (
    gap_id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    domain TEXT NOT NULL,
    severity TEXT NOT NULL,
    frequency INTEGER NOT NULL,
    average_confidence REAL NOT NULL,
    unanswered_count INTEGER NOT NULL,
    low_confidence_count INTEGER DEFAULT 0,
    average_retrieval_score REAL DEFAULT 0.0,
    domains_involved TEXT DEFAULT '[]',
    normalized_query TEXT DEFAULT '',
    sample_queries TEXT NOT NULL,
    suggested_actions TEXT NOT NULL,
    first_detected TEXT NOT NULL,
    last_detected TEXT NOT NULL
);
```

---

## 2. Python Data Models (`milestone_4/app/analytics/models.py`)

### `ResolutionStatus` Enum
- `RESOLVED`: Successfully retrieved with MEDIUM or HIGH confidence.
- `LOW_CONFIDENCE`: Retrieved chunks scored below configured threshold or confidence level is LOW.
- `UNANSWERED`: Zero relevant documents or confidence level is NONE / insufficient evidence.
- `CLARIFICATION_REQUESTED`: Ambiguity triggered clarification inquiry.
- `CLARIFICATION_RESOLVED`: Ambiguity resolved after user response.
- `ERROR`: Pipeline execution failure.

### `GapSeverity` Enum
- `CRITICAL`: Frequency >= 5 or unanswered >= 3.
- `HIGH`: Frequency >= 3 or low-confidence >= 3.
- `MEDIUM`: Frequency >= 2.
- `LOW`: Single isolated occurrence.
