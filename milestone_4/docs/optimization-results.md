# Milestone 4 — Complete Optimization Benchmark Results

**Source:** Programmatically measured from runtime execution  
**Raw Results Path:** `milestone_4/evaluation/`  
**Evaluation Dates:** September 2026  

---

## 1. Before vs After System Benchmark (MS3 Baseline vs MS4 Optimized)

*Data source: `milestone_4/evaluation/before_after_results.json`*

| Metric | MS3 Baseline (Dense Vector, chunk=500, thresh=0.45) | MS4 Optimized (Hybrid, chunk=800, overlap=150, thresh=0.35) | Measured Improvement |
|---|---|---|---|
| **Retrieval Recall@3** | 0.0% | **100.0%** | **+100.0%** |
| **Retrieval Precision@3** | 0.0% | **95.0%** | **+95.0%** |
| **Mean Reciprocal Rank (MRR)** | 0.000 | **1.000** | **+1.000** |
| **Average Grounded Confidence** | 0.000 | **0.495** | **+0.495** |
| **Average Pipeline Latency** | 1.09 ms | **1.08 ms** | -0.01 ms |
| **Unanswered Query Rate** | 100.0% | **7.1%** | **-92.9%** |
| **Query Routing Accuracy** | 92.9% | **92.9%** (100% on labeled set) | Maintained / Enhanced |
| **Voice Speech Sanitization** | PASS | **PASS** | Complete coverage |

---

## 2. Granular Latency Profiling by Pipeline Component

*Data source: `milestone_4/evaluation/latency_results.json`*

| Pipeline Stage | Mean (ms) | Median (ms) | P95 (ms) | Max (ms) |
|---|---|---|---|---|
| **Memory Contextualization** | 0.04 | 0.01 | 0.23 | 0.34 |
| **Query Understanding** | 0.11 | 0.11 | 0.14 | 0.14 |
| **Query Embedding Vectorization** | 0.05 | 0.05 | 0.06 | 0.06 |
| **Hybrid Retrieval & Re-ranking** | 0.91 | 0.81 | 1.54 | 1.64 |
| **Response Generation & Citations** | 0.07 | 0.06 | 0.13 | 0.16 |
| **Speech Sanitization (TTS)** | 0.05 | 0.05 | 0.08 | 0.09 |
| **Memory Turn Recording** | 0.00 | 0.00 | 0.00 | 0.01 |
| **Analytics Telemetry (SQLite)** | 0.36 | 0.36 | 0.40 | 0.40 |
| **Full End-to-End Orchestration** | **1.42** | **1.36** | **2.04** | **2.12** |

---

## 3. Query Intent Routing Accuracy Benchmark

*Data source: `milestone_4/evaluation/routing_results.json`*

- **Total Test Cases**: 20 labeled queries across 3 knowledge domains
- **Overall Intent Classification Accuracy**: **100.0%** (20/20)
- **Route Target Routing Accuracy**: **100.0%** (20/20)
- **Domain Classification Accuracy**: **100.0%** (20/20)

### Breakdown by Intent Category:
- **FACTUAL**: 100.0% (4/4) -> Retrieval -> Response Generation
- **PROCEDURAL**: 100.0% (4/4) -> Retrieval -> Step-oriented response
- **COMPARATIVE**: 100.0% (4/4) -> Multi-document retrieval -> Comparison response
- **AMBIGUOUS**: 100.0% (4/4) -> Clarification Agent inquiry
- **MULTI_PART**: 100.0% (4/4) -> Decomposition -> Retrieval -> Synthesis
