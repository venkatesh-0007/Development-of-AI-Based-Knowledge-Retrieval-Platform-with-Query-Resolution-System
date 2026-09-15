"""Evaluation script for Milestone 2.4 End-to-End Multi-Agent Pipeline."""
import sys
import time
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "milestone_2"))

from app.embeddings.embedder import Embedder
from app.vectorstore.chroma_store import ChromaStore
from app.ingestion.pipeline import IngestionPipeline
from app.agents.query_understanding import QueryUnderstandingAgent
from app.agents.retrieval import RetrievalAgent
from app.agents.response_generation import ResponseGenerationAgent
from app.agents.clarification import ClarificationAgent
from app.agents.orchestrator import MultiAgentOrchestrator
from app.agents.models import QueryType, ConfidenceLevel

E2E_SCENARIOS = [
    {"query": "What is supervised learning?", "category": "Factual (In-Domain)", "expect_clarification": False, "expect_evidence": True},
    {"query": "What is TCP?", "category": "Factual (In-Domain)", "expect_clarification": False, "expect_evidence": True},
    {"query": "How does gradient descent work?", "category": "Procedural (In-Domain)", "expect_clarification": False, "expect_evidence": True},
    {"query": "How does the DNS resolution process work?", "category": "Procedural (In-Domain)", "expect_clarification": False, "expect_evidence": True},
    {"query": "Compare classification and regression.", "category": "Comparative (In-Domain)", "expect_clarification": False, "expect_evidence": True},
    {"query": "Compare TCP and UDP.", "category": "Comparative (In-Domain)", "expect_clarification": False, "expect_evidence": True},
    {"query": "Tell me about it.", "category": "Ambiguous (Vague)", "expect_clarification": True, "expect_evidence": False},
    {"query": "Explain this.", "category": "Ambiguous (Vague)", "expect_clarification": True, "expect_evidence": False},
    {"query": "What is quantum computing?", "category": "Unavailable (Out-of-Domain)", "expect_clarification": False, "expect_evidence": False},
]

def run_evaluation():
    print("=" * 80)
    print("Evaluating Milestone 2.4 Multi-Agent End-to-End Resolution Pipeline")
    print("=" * 80)

    embedder = Embedder()
    store = ChromaStore(path="data/chroma", collection_name="m2_e2e_eval_collection")
    ingestion = IngestionPipeline(embedder, store)

    sample_dir = root_dir / "milestone_2" / "data" / "sample_docs"
    for doc in sample_dir.glob("*.*"):
        if doc.suffix.lower() in [".txt", ".csv"]:
            ingestion.ingest(doc.read_bytes(), doc.name, chunk_size=800, overlap=100)

    orchestrator = MultiAgentOrchestrator(
        retrieval_agent=RetrievalAgent(embedder, store, confidence_threshold=0.45),
        query_understanding_agent=QueryUnderstandingAgent(),
        response_generation_agent=ResponseGenerationAgent(),
        clarification_agent=ClarificationAgent()
    )

    total = len(E2E_SCENARIOS)
    passed = 0
    latencies = []

    print(f"\n{'Query':<38} | {'Category':<24} | {'Conf Level':<10} | {'Latency':<8} | {'Status'}")
    print("-" * 92)

    for item in E2E_SCENARIOS:
        q = item["query"]
        res = orchestrator.run(q)
        latencies.append(res.execution_time_ms)

        is_clarif_ok = (res.response.requires_clarification == item["expect_clarification"])
        is_evid_ok = (res.response.has_sufficient_evidence == item["expect_evidence"])
        is_ok = is_clarif_ok and is_evid_ok and res.status in ["success", "clarification_requested"]

        if is_ok:
            passed += 1

        short_q = (q[:35] + "...") if len(q) > 37 else q
        status_str = "✅ PASS" if is_ok else "❌ FAIL"
        conf_str = res.response.confidence_level.value
        print(f"{short_q:<38} | {item['category']:<24} | {conf_str:<10} | {res.execution_time_ms:<6.1f}ms | {status_str}")

    avg_lat = sum(latencies) / len(latencies)
    print("-" * 92)
    print(f"End-to-End Success Rate:     {passed / total * 100:.1f}% ({passed}/{total})")
    print(f"Average Pipeline Latency:    {avg_lat:.2f} ms")
    print("=" * 80)

if __name__ == "__main__":
    run_evaluation()
