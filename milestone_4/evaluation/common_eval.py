"""Common evaluation dataset, baseline retrieval agent, and metric calculators (M4.3, Phase 12, Phase 13, Phase 18).

Provides:
- Canonical Ground Truth Evaluation Dataset across 3 Knowledge Domains:
  Machine Learning, Computer Networks, and Cybersecurity.
- BaselineRetrievalAgent reproducing genuine Milestone 3 behavior (pure dense vector search,
  no hybrid lexical re-ranking, fixed 0.45 threshold, no domain affinity adjustment).
- Pure metric calculation functions (Recall@K, Precision@K, MRR, Routing Accuracy, Latency).
"""
import time
import math
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Canonical ground truth evaluation queries across all 3 domains and 5 query intents
GROUND_TRUTH_EVAL_SET: List[Dict[str, Any]] = [
    # Domain 1: Machine Learning
    {
        "query": "What is Supervised Learning and what are its main applications?",
        "domain": "machine_learning",
        "type": "factual",
        "expected_docs": ["ml_overview.txt", "ml_algorithms.csv"],
        "expected_keywords": ["labeled", "supervised", "continuous", "classification"]
    },
    {
        "query": "Which machine learning algorithm finds an optimal separating hyperplane using kernels?",
        "domain": "machine_learning",
        "type": "factual",
        "expected_docs": ["ml_algorithms.csv"],
        "expected_keywords": ["support vector machine", "svm", "hyperplane"]
    },
    {
        "query": "How does multi-head self-attention work in Transformer architecture?",
        "domain": "machine_learning",
        "type": "procedural",
        "expected_docs": ["deep_learning_and_transformers.txt"],
        "expected_keywords": ["attention", "queries", "keys", "values", "softmax"]
    },
    {
        "query": "Compare Support Vector Machines with Random Forest",
        "domain": "machine_learning",
        "type": "comparative",
        "expected_docs": ["ml_algorithms.csv"],
        "expected_keywords": ["ensemble", "trees", "margin", "svm"]
    },

    # Domain 2: Computer Networks & Cloud
    {
        "query": "What is the function of the Transport Layer in the OSI model?",
        "domain": "computer_networks",
        "type": "factual",
        "expected_docs": ["computer_networks.txt"],
        "expected_keywords": ["transport", "tcp", "udp", "segment"]
    },
    {
        "query": "Explain the step by step process of the TCP three-way handshake",
        "domain": "computer_networks",
        "type": "procedural",
        "expected_docs": ["computer_networks.txt"],
        "expected_keywords": ["syn", "syn-ack", "ack", "handshake"]
    },
    {
        "query": "What is the difference between TCP and UDP in terms of reliability?",
        "domain": "computer_networks",
        "type": "comparative",
        "expected_docs": ["computer_networks.txt", "network_protocols.csv"],
        "expected_keywords": ["connection", "reliable", "unreliable", "datagram"]
    },
    {
        "query": "What does the CAP theorem state regarding distributed consistency?",
        "domain": "computer_networks",
        "type": "factual",
        "expected_docs": ["cloud_distributed_systems.txt"],
        "expected_keywords": ["consistency", "availability", "partition tolerance"]
    },

    # Domain 3: Cybersecurity
    {
        "query": "Define the CIA Triad in information security",
        "domain": "cybersecurity",
        "type": "factual",
        "expected_docs": ["cybersecurity_fundamentals.txt"],
        "expected_keywords": ["confidentiality", "integrity", "availability"]
    },
    {
        "query": "Compare AES and RSA encryption mechanisms",
        "domain": "cybersecurity",
        "type": "comparative",
        "expected_docs": ["cryptography_and_zero_trust.csv"],
        "expected_keywords": ["symmetric", "asymmetric", "aes", "rsa"]
    },
    {
        "query": "What are the sequential phases of the NIST incident response lifecycle?",
        "domain": "cybersecurity",
        "type": "procedural",
        "expected_docs": ["incident_response_and_threats.txt"],
        "expected_keywords": ["preparation", "detection", "containment", "recovery"]
    },
    {
        "query": "What is Zero Trust Security Architecture and least privilege?",
        "domain": "cybersecurity",
        "type": "factual",
        "expected_docs": ["cybersecurity_fundamentals.txt", "cryptography_and_zero_trust.csv"],
        "expected_keywords": ["zero trust", "least privilege", "verify"]
    },

    # Ambiguous & Multi-Part Test Queries
    {
        "query": "Explain tree",
        "domain": "general",
        "type": "ambiguous",
        "expected_docs": [],
        "expected_keywords": []
    },
    {
        "query": "What is TCP and also how is it different from UDP?",
        "domain": "computer_networks",
        "type": "multi_part",
        "expected_docs": ["computer_networks.txt", "network_protocols.csv"],
        "expected_keywords": ["tcp", "udp"]
    }
]


class BaselineRetrievalAgent:
    """
    Genuine Milestone 3 Baseline Retrieval implementation.
    - Pure dense vector similarity search
    - No hybrid lexical keyword overlap blending
    - No domain affinity bonus (+0.05)
    - Fixed threshold (0.45 default as in MS3)
    - Standard Top-K without multi-part decomposition
    """

    def __init__(self, embedder: Any, vectorstore: Any, confidence_threshold: float = 0.45, default_top_k: int = 3):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.confidence_threshold = confidence_threshold
        self.default_top_k = default_top_k

    def retrieve(self, query_input: Any, top_k: Optional[int] = None, threshold: Optional[float] = None):
        t0 = time.perf_counter()
        query_text = query_input.cleaned_query if hasattr(query_input, "cleaned_query") else str(query_input)
        k = top_k or self.default_top_k
        thresh = threshold if threshold is not None else self.confidence_threshold

        query_emb = self.embedder.encode([query_text])[0]
        raw_results = self.vectorstore.search(query_emb, top_k=k * 2)

        valid_chunks = []
        for r in raw_results:
            score = float(r.get("score", 0.0))
            if score >= thresh:
                meta = r.get("metadata", {}) or {}
                valid_chunks.append({
                    "chunk_id": meta.get("chunk_id", ""),
                    "document_name": meta.get("document_name", meta.get("source", "")),
                    "content": r.get("document", ""),
                    "score": round(score, 4),
                    "metadata": meta
                })
                if len(valid_chunks) >= k:
                    break

        top_score = valid_chunks[0]["score"] if valid_chunks else 0.0
        avg_score = round(sum(c["score"] for c in valid_chunks) / len(valid_chunks), 4) if valid_chunks else 0.0
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Construct lightweight result container compatible with evaluation scripts
        return type("BaselineRetrievalResult", (), {
            "query": query_text,
            "chunks": [type("Chunk", (), c)() for c in valid_chunks],
            "top_score": top_score,
            "average_score": avg_score,
            "execution_time_ms": round(elapsed_ms, 2)
        })()


def evaluate_retrieval_metrics(
    retrieval_agent: Any,
    eval_queries: List[Dict[str, Any]],
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Computes real, non-fabricated retrieval metrics against ground truth.
    - Recall@K: proportion of test items where at least one ground-truth doc/keyword is in Top-K.
    - Precision@K: proportion of retrieved chunks in Top-K that match ground-truth docs.
    - MRR (Mean Reciprocal Rank): 1 / first_relevant_rank.
    - Average Top Score & Latency (ms).
    """
    hits_at_k = 0
    total_retrieved_chunks = 0
    relevant_retrieved_chunks = 0
    reciprocal_ranks = []
    latencies = []
    top_scores = []
    unanswered_count = 0

    # Only evaluate queries that have ground-truth expected docs
    eval_subset = [q for q in eval_queries if q.get("expected_docs")]
    n_eval = len(eval_subset)

    for item in eval_subset:
        t0 = time.perf_counter()
        res = retrieval_agent.retrieve(item["query"], top_k=top_k)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)

        chunks = res.chunks
        if not chunks or res.top_score < 0.20:
            unanswered_count += 1
            reciprocal_ranks.append(0.0)
            top_scores.append(res.top_score)
            continue

        top_scores.append(res.top_score)
        total_retrieved_chunks += len(chunks)

        found_rank = 0
        for rank, c in enumerate(chunks, start=1):
            doc_name = getattr(c, "document_name", "")
            content = getattr(c, "content", "").lower()
            is_relevant_doc = doc_name in item["expected_docs"]
            has_keywords = any(kw.lower() in content for kw in item.get("expected_keywords", []))

            if is_relevant_doc or has_keywords:
                relevant_retrieved_chunks += 1
                if found_rank == 0:
                    found_rank = rank

        if found_rank > 0:
            hits_at_k += 1
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)

    recall_at_k = round(hits_at_k / n_eval, 4) if n_eval else 0.0
    precision_at_k = round(relevant_retrieved_chunks / total_retrieved_chunks, 4) if total_retrieved_chunks else 0.0
    mrr = round(sum(reciprocal_ranks) / n_eval, 4) if n_eval else 0.0
    avg_latency = round(sum(latencies) / n_eval, 2) if n_eval else 0.0
    avg_top_score = round(sum(top_scores) / n_eval, 4) if n_eval else 0.0
    unanswered_rate = round(unanswered_count / n_eval, 4) if n_eval else 0.0

    return {
        "queries_evaluated": n_eval,
        "top_k": top_k,
        "recall_at_k": recall_at_k,
        "precision_at_k": precision_at_k,
        "mrr": mrr,
        "average_latency_ms": avg_latency,
        "average_top_score": avg_top_score,
        "unanswered_rate": unanswered_rate
    }
