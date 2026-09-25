# Knowledge Gap Detection System (M4.1)

## 1. Overview and Core Philosophy
Knowledge gap detection identifies systemic blind spots in the ingested knowledge bases. Rather than treating individual failed queries in isolation, the detector aggregates, normalizes, and groups recurring failed or weak inquiries to provide actionable recommendations for knowledge base curators.

---

## 2. Six Trigger Conditions for Knowledge Gaps
A query entry is classified as a gap candidate if any of the following 6 conditions are met:
1. **No Relevant Documents Retrieved**: Grounded synthesis identifies that zero evidence chunks meet threshold criteria.
2. **Retrieved Document Count is Zero**: Vector and lexical search returns an empty result set.
3. **Top Similarity Score is Sub-Threshold**: Top similarity score is strictly below the configured cutoff (`< 0.35`).
4. **Confidence is LOW or NONE**: Centralized confidence calculation produces `LOW` or `NONE`.
5. **Repeated Low-Confidence Topic**: Multiple queries on the same concept consistently produce marginal scores.
6. **Repeated Unanswered Query/Theme**: The same missing concept appears across multiple user sessions.

---

## 3. Clustering Methodology: Lexical & Thematic Token Clustering
To remain fully deterministic, robust, and transparent, the detection engine employs **Lexical and Thematic Jaccard Token Clustering with Porter-Style Suffix Normalization**.

> **Note on Methodology:** This clustering is explicitly documented as lexical/thematic token similarity rather than opaque high-dimensional vector clustering.

### Algorithm Steps:
1. **Query Text Normalization**:
   - Strip punctuation, question formatting, and lowercases text.
   - Filter generic stop-words (`what`, `is`, `explain`, `tell`, `how`, `does`, `the`).
   - Apply token suffix normalization (`ing`, `ed`, `es`, `s`).
2. **Thematic Fingerprinting**:
   - Extract core noun/technical tokens (e.g., `quantum`, `annealing`).
   - Group queries sharing >= 50% Jaccard token overlap into the same thematic cluster.
3. **Statistical Aggregation**:
   - `frequency`: Total occurrences of the cluster.
   - `average_confidence`: Mean confidence score across instances.
   - `average_retrieval_score`: Mean top retrieval score across instances.
   - `domains_involved`: Set of knowledge domains associated with the queries.
   - `unanswered_count`: Number of queries where evidence was completely absent.
   - `low_confidence_count`: Number of queries where evidence was marginal.
   - `first_detected` & `last_detected`: UTC timestamps tracking first and most recent appearance.
4. **Actionable Suggestions Generation**:
   - Produces targeted guidance for documentation ingestion (e.g., "Add canonical documentation covering Quantum Annealing to the Machine Learning domain").
