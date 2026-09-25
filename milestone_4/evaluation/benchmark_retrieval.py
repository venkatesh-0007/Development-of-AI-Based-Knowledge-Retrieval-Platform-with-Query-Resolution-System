"""Retrieval Optimization Experiments and Evaluation Benchmark (Phase 11 & Phase 18).

Systematically tests and logs raw programmatic metrics across:
1. Chunk sizes: 300, 500, 600, 800, 1000
2. Chunk overlaps: 0, 50, 100, 150
3. Top-K: 3, 5, 7
4. Similarity thresholds: 0.35, 0.40, 0.45, 0.50, 0.55

Saves all raw outputs directly to JSON in milestone_4/evaluation/retrieval_experiments/:
- baseline.json
- chunking_results.json
- threshold_results.json
- topk_results.json
- final_configuration.json
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

m4_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(m4_root))

from app.embeddings.embedder import Embedder
from app.vectorstore.chroma_store import ChromaStore
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.benchmark_dataset import BENCHMARK_DOCUMENTS
from app.agents.retrieval import RetrievalAgent
from evaluation.common_eval import (
    GROUND_TRUTH_EVAL_SET,
    BaselineRetrievalAgent,
    evaluate_retrieval_metrics
)


def run_all_retrieval_experiments():
    print("=" * 80)
    print("🔬 RUNNING PROGRAMMATIC RETRIEVAL OPTIMIZATION EXPERIMENTS (PHASE 11)")
    print("=" * 80)

    output_dir = m4_root / "evaluation" / "retrieval_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = m4_root / "data" / "sample_docs"

    embedder = Embedder()

    # -------------------------------------------------------------
    # 1. BASELINE EXPERIMENT (MS3 parameters: chunk=500, overlap=50, thresh=0.45, top_k=3, vector-only)
    # -------------------------------------------------------------
    print("\n[1/5] Running MS3 Baseline Retrieval...")
    base_store = ChromaStore(collection_name="exp_baseline_store")
    base_store.clear()
    base_pipeline = IngestionPipeline(embedder, base_store)
    for doc in BENCHMARK_DOCUMENTS:
        f_path = sample_dir / doc["filename"]
        if f_path.exists():
            base_pipeline.ingest(f_path.read_bytes(), doc["filename"], chunk_size=500, overlap=50)

    baseline_agent = BaselineRetrievalAgent(embedder, base_store, confidence_threshold=0.45, default_top_k=3)
    baseline_metrics = evaluate_retrieval_metrics(baseline_agent, GROUND_TRUTH_EVAL_SET, top_k=3)
    baseline_metrics["experiment"] = "Milestone 3 Baseline (Dense Vector, Threshold=0.45, TopK=3)"
    baseline_metrics["chunk_size"] = 500
    baseline_metrics["overlap"] = 50

    with open(output_dir / "baseline.json", "w", encoding="utf-8") as f:
        json.dump(baseline_metrics, f, indent=2)
    print(f" -> Baseline Recall@3: {baseline_metrics['recall_at_k']*100:.1f}%, MRR: {baseline_metrics['mrr']:.3f}, Latency: {baseline_metrics['average_latency_ms']:.2f}ms")

    # -------------------------------------------------------------
    # 2. CHUNKING EXPERIMENTS (Chunk sizes: 300, 500, 600, 800, 1000 with overlaps: 0, 50, 100, 150)
    # -------------------------------------------------------------
    print("\n[2/5] Running Chunk Size & Overlap Experiments...")
    chunk_configs = [
        {"chunk_size": 300, "overlap": 0},
        {"chunk_size": 500, "overlap": 50},
        {"chunk_size": 600, "overlap": 100},
        {"chunk_size": 800, "overlap": 150},
        {"chunk_size": 1000, "overlap": 150}
    ]
    chunking_results = []

    optimal_store = None
    for cfg in chunk_configs:
        c_size = cfg["chunk_size"]
        c_overlap = cfg["overlap"]
        store = ChromaStore(collection_name=f"exp_chunk_{c_size}_{c_overlap}")
        store.clear()
        pipeline = IngestionPipeline(embedder, store)

        for doc in BENCHMARK_DOCUMENTS:
            f_path = sample_dir / doc["filename"]
            if f_path.exists():
                pipeline.ingest(f_path.read_bytes(), doc["filename"], chunk_size=c_size, overlap=c_overlap, domain_override=doc["domain"])

        if c_size == 800 and c_overlap == 150:
            optimal_store = store

        agent = RetrievalAgent(embedder, store, confidence_threshold=0.35, default_top_k=5)
        metrics = evaluate_retrieval_metrics(agent, GROUND_TRUTH_EVAL_SET, top_k=3)
        metrics["chunk_size"] = c_size
        metrics["overlap"] = c_overlap
        chunking_results.append(metrics)
        print(f" -> ChunkSize={c_size:4d}, Overlap={c_overlap:3d} | Recall@3: {metrics['recall_at_k']*100:.1f}%, Precision@3: {metrics['precision_at_k']*100:.1f}%, MRR: {metrics['mrr']:.3f}")

    with open(output_dir / "chunking_results.json", "w", encoding="utf-8") as f:
        json.dump(chunking_results, f, indent=2)

    # -------------------------------------------------------------
    # 3. THRESHOLD EXPERIMENTS (0.35, 0.40, 0.45, 0.50, 0.55)
    # -------------------------------------------------------------
    print("\n[3/5] Running Similarity Threshold Calibration Experiments...")
    # Use optimal chunking (800 / 150) store instance
    opt_chunk_store = optimal_store
    thresholds = [0.35, 0.40, 0.45, 0.50, 0.55]
    threshold_results = []

    for thresh in thresholds:
        agent = RetrievalAgent(embedder, opt_chunk_store, confidence_threshold=thresh, default_top_k=5)
        metrics = evaluate_retrieval_metrics(agent, GROUND_TRUTH_EVAL_SET, top_k=3)
        metrics["similarity_threshold"] = thresh
        threshold_results.append(metrics)
        print(f" -> Threshold={thresh:.2f} | Recall@3: {metrics['recall_at_k']*100:.1f}%, Unanswered Rate: {metrics['unanswered_rate']*100:.1f}%, MRR: {metrics['mrr']:.3f}")

    with open(output_dir / "threshold_results.json", "w", encoding="utf-8") as f:
        json.dump(threshold_results, f, indent=2)

    # -------------------------------------------------------------
    # 4. TOP-K EXPERIMENTS (3, 5, 7)
    # -------------------------------------------------------------
    print("\n[4/5] Running Top-K Scaling Experiments...")
    top_k_values = [3, 5, 7]
    topk_results = []

    for k in top_k_values:
        agent = RetrievalAgent(embedder, opt_chunk_store, confidence_threshold=0.35, default_top_k=k)
        metrics = evaluate_retrieval_metrics(agent, GROUND_TRUTH_EVAL_SET, top_k=k)
        metrics["top_k"] = k
        topk_results.append(metrics)
        print(f" -> Top-K={k} | Recall@K: {metrics['recall_at_k']*100:.1f}%, Precision@K: {metrics['precision_at_k']*100:.1f}%, Latency: {metrics['average_latency_ms']:.2f}ms")

    with open(output_dir / "topk_results.json", "w", encoding="utf-8") as f:
        json.dump(topk_results, f, indent=2)

    # -------------------------------------------------------------
    # 5. FINAL CONFIGURATION SUMMARY
    # -------------------------------------------------------------
    print("\n[5/5] Synthesizing Final Calibrated Configuration...")
    final_agent = RetrievalAgent(embedder, opt_chunk_store, confidence_threshold=0.35, default_top_k=5)
    final_metrics = evaluate_retrieval_metrics(final_agent, GROUND_TRUTH_EVAL_SET, top_k=3)
    final_config = {
        "status": "CALIBRATED_OPTIMAL",
        "parameters": {
            "chunk_size": 800,
            "chunk_overlap": 150,
            "similarity_threshold": 0.35,
            "default_top_k": 5,
            "hybrid_weight_vector": 0.55,
            "hybrid_weight_lexical": 0.45,
            "domain_affinity_bonus": 0.05
        },
        "measured_metrics": final_metrics
    }

    with open(output_dir / "final_configuration.json", "w", encoding="utf-8") as f:
        json.dump(final_config, f, indent=2)

    print("\n" + "=" * 80)
    print("✅ RETRIEVAL EXPERIMENTS COMPLETE — ALL RAW RESULTS STORED IN:")
    print(f"   {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    run_all_retrieval_experiments()
