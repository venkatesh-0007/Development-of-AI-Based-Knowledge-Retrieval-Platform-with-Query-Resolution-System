"""Component and End-to-End Latency Benchmark for Milestone 4 (Phase 18).

Measures real, programmatic execution latency across every individual pipeline stage:
1. Conversation Memory retrieval & contextualization
2. Query Understanding (entity extraction, intent, domain classification)
3. Embedding vector generation
4. Hybrid Retrieval & Re-ranking
5. Response Generation, citation formatting & TTS sanitization
6. Memory turn recording
7. Analytics telemetry persistence (SQLite)
8. End-to-End Pipeline Execution

Calculates Mean, Min, Max, P50, P90, P95, P99 and saves raw JSON to:
milestone_4/evaluation/latency_results.json
"""
import os
import sys
import time
import json
import statistics
from pathlib import Path
from typing import List, Dict, Any

m4_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(m4_root))

from app.embeddings.embedder import Embedder
from app.vectorstore.chroma_store import ChromaStore
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.benchmark_dataset import BENCHMARK_DOCUMENTS
from app.agents.retrieval import RetrievalAgent
from app.agents.query_understanding import QueryUnderstandingAgent
from app.agents.clarification import ClarificationAgent
from app.agents.response_generation import ResponseGenerationAgent
from app.agents.memory import ConversationMemoryAgent
from app.agents.voice import VoiceModule
from app.analytics.storage import AnalyticsStorage
from app.analytics.tracker import AnalyticsTracker
from app.agents.orchestrator import MultiAgentOrchestrator
from evaluation.common_eval import GROUND_TRUTH_EVAL_SET


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    k = (len(data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c < len(data):
        return data[f] + (k - f) * (data[c] - data[f])
    return data[f]


def run_latency_benchmark():
    print("=" * 80)
    print("⏱️ RUNNING PROGRAMMATIC COMPONENT & PIPELINE LATENCY BENCHMARK (PHASE 18)")
    print("=" * 80)

    sample_dir = m4_root / "data" / "sample_docs"
    db_path = m4_root / "data" / "analytics" / "latency_benchmark.db"
    if db_path.exists():
        db_path.unlink()

    embedder = Embedder()
    store = ChromaStore(collection_name="latency_bench_store")
    store.clear()
    pipeline = IngestionPipeline(embedder, store)

    print("Populating store with canonical benchmark documents...")
    for doc in BENCHMARK_DOCUMENTS:
        f_path = sample_dir / doc["filename"]
        if f_path.exists():
            pipeline.ingest(f_path.read_bytes(), doc["filename"], chunk_size=800, overlap=150, domain_override=doc["domain"])

    clarification_agent = ClarificationAgent()
    query_understanding_agent = QueryUnderstandingAgent(clarification_agent=clarification_agent)
    retrieval_agent = RetrievalAgent(embedder, store, confidence_threshold=0.35, default_top_k=5)
    response_generation_agent = ResponseGenerationAgent()
    memory_agent = ConversationMemoryAgent()
    analytics_storage = AnalyticsStorage(db_path=str(db_path))
    analytics_tracker = AnalyticsTracker(storage=analytics_storage)
    voice_module = VoiceModule()

    orchestrator = MultiAgentOrchestrator(
        retrieval_agent=retrieval_agent,
        query_understanding_agent=query_understanding_agent,
        response_generation_agent=response_generation_agent,
        clarification_agent=clarification_agent,
        memory_agent=memory_agent,
        analytics_tracker=analytics_tracker
    )

    stage_timings = {
        "memory_contextualization_ms": [],
        "query_understanding_ms": [],
        "query_embedding_ms": [],
        "hybrid_retrieval_ms": [],
        "response_generation_ms": [],
        "speech_sanitization_ms": [],
        "memory_recording_ms": [],
        "analytics_persistence_ms": [],
        "end_to_end_orchestration_ms": []
    }

    warmup_queries = ["What is TCP?", "Explain Supervised Learning"]
    for w in warmup_queries:
        orchestrator.run(w)

    session_id = "latency_test_session"
    print(f"Executing {len(GROUND_TRUTH_EVAL_SET)} canonical test queries with granular instrumentation...")

    for item in GROUND_TRUTH_EVAL_SET:
        query = item["query"]

        # Step 1: Memory Contextualization
        t0 = time.perf_counter()
        session = memory_agent.get_or_create_session(session_id)
        effective_query = memory_agent.contextualize_query(query, session)
        stage_timings["memory_contextualization_ms"].append((time.perf_counter() - t0) * 1000)

        # Step 2: Query Understanding
        t0 = time.perf_counter()
        qa = query_understanding_agent.analyze(effective_query, active_domain=session.active_domain)
        stage_timings["query_understanding_ms"].append((time.perf_counter() - t0) * 1000)

        # Step 3: Query Embedding
        t0 = time.perf_counter()
        _ = embedder.encode([effective_query])
        stage_timings["query_embedding_ms"].append((time.perf_counter() - t0) * 1000)

        # Step 4: Hybrid Retrieval
        t0 = time.perf_counter()
        retrieval_result = retrieval_agent.retrieve(query_input=qa, top_k=5)
        stage_timings["hybrid_retrieval_ms"].append((time.perf_counter() - t0) * 1000)

        # Step 5: Response Generation
        t0 = time.perf_counter()
        agent_response = response_generation_agent.generate(query_analysis=qa, retrieval_result=retrieval_result)
        stage_timings["response_generation_ms"].append((time.perf_counter() - t0) * 1000)

        # Step 6: Speech Sanitization
        t0 = time.perf_counter()
        _ = voice_module.clean_for_speech(agent_response.answer)
        stage_timings["speech_sanitization_ms"].append((time.perf_counter() - t0) * 1000)

        # Step 7: Memory Recording
        t0 = time.perf_counter()
        from app.agents.models import ConversationTurn
        turn = ConversationTurn(
            query=query,
            effective_query=effective_query,
            response=agent_response.answer,
            confidence_score=agent_response.confidence_score,
            sources=agent_response.sources,
            key_entities=qa.entities,
            transparency=agent_response.transparency,
            domain=qa.domain
        )
        memory_agent.record_turn(session, turn)
        stage_timings["memory_recording_ms"].append((time.perf_counter() - t0) * 1000)

        # Step 8: End-to-End Orchestrator Run (including Analytics Telemetry)
        t0 = time.perf_counter()
        orch_res = orchestrator.run(query, session_id="isolated_timing_session")
        stage_timings["end_to_end_orchestration_ms"].append((time.perf_counter() - t0) * 1000)

        # Step 9: Analytics Telemetry Persistence
        t0 = time.perf_counter()
        analytics_tracker.record_orchestration(orch_res)
        stage_timings["analytics_persistence_ms"].append((time.perf_counter() - t0) * 1000)

    # Compute Statistics
    stats = {}
    for stage, vals in stage_timings.items():
        sorted_vals = sorted(vals)
        stats[stage] = {
            "mean_ms": round(statistics.mean(vals), 3),
            "median_ms": round(statistics.median(vals), 3),
            "min_ms": round(min(vals), 3),
            "max_ms": round(max(vals), 3),
            "p50_ms": round(percentile(sorted_vals, 50), 3),
            "p90_ms": round(percentile(sorted_vals, 90), 3),
            "p95_ms": round(percentile(sorted_vals, 95), 3),
            "p99_ms": round(percentile(sorted_vals, 99), 3),
            "sample_count": len(vals)
        }

    output_path = m4_root / "evaluation" / "latency_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 80)
    print(f"{'Pipeline Stage':<35} | {'Mean (ms)':<10} | {'Median':<10} | {'P95 (ms)':<10} | {'Max (ms)':<10}")
    print("-" * 80)
    for stage, s in stats.items():
        display_name = stage.replace("_ms", "").replace("_", " ").title()
        print(f"{display_name:<35} | {s['mean_ms']:<10.2f} | {s['median_ms']:<10.2f} | {s['p95_ms']:<10.2f} | {s['max_ms']:<10.2f}")
    print("=" * 80)
    print(f"✅ Programmatic latency benchmark complete. Raw results saved to:\n   {output_path}\n")

    return stats


if __name__ == "__main__":
    run_latency_benchmark()
