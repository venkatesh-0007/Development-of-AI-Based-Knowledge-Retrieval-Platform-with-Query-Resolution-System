"""End-to-End multi-agent pipeline evaluation script."""
import sys
import time
from pathlib import Path

milestone_2_root = Path(__file__).resolve().parent.parent
if str(milestone_2_root) not in sys.path:
    sys.path.insert(0, str(milestone_2_root))

from app.embeddings.embedder import Embedder
from app.vectorstore.chroma_store import ChromaStore
from app.ingestion.pipeline import IngestionPipeline
from app.agents.query_understanding import QueryUnderstandingAgent
from app.agents.retrieval import RetrievalAgent
from app.agents.response_generation import ResponseGenerationAgent
from app.agents.clarification import ClarificationAgent
from app.agents.orchestrator import MultiAgentOrchestrator
from app.agents.models import ConfidenceLevel, QueryType

BENCHMARKS = [
    {"query": "What is supervised learning?", "expected_status": "success", "expected_conf": ConfidenceLevel.HIGH},
    {"query": "How does DNS resolution work?", "expected_status": "success", "expected_conf": ConfidenceLevel.HIGH},
    {"query": "Compare TCP vs UDP", "expected_status": "success", "expected_conf": ConfidenceLevel.HIGH},
    {"query": "Tell me about it.", "expected_status": "clarification_requested", "expected_conf": ConfidenceLevel.NONE},
    {"query": "What is quantum computing?", "expected_status": "success", "expected_conf": ConfidenceLevel.NONE},
]

def run_evaluation():
    print("=" * 65)
    print("Evaluating Milestone 2 Multi-Agent End-to-End Resolution Pipeline")
    print("=" * 65)

    embedder = Embedder()
    store = ChromaStore(path="data/chroma", collection_name="m2_e2e_eval")
    ingestion = IngestionPipeline(embedder, store)

    sample_dir = milestone_2_root / "data" / "sample_docs"
    for sample in sample_dir.glob("*.*"):
        if sample.suffix.lower() in [".txt", ".csv"]:
            ingestion.ingest(sample.read_bytes(), sample.name, chunk_size=800, overlap=100)

    query_agent = QueryUnderstandingAgent()
    retrieval_agent = RetrievalAgent(embedder, store, confidence_threshold=0.45)
    response_agent = ResponseGenerationAgent()
    clarification_agent = ClarificationAgent()

    orchestrator = MultiAgentOrchestrator(
        retrieval_agent=retrieval_agent,
        query_understanding_agent=query_agent,
        response_generation_agent=response_agent,
        clarification_agent=clarification_agent
    )

    print(f"\n{'Query':<32} | {'Status':<23} | {'Confidence':<10} | {'Latency':<9} | {'Result'}")
    print("-" * 85)

    for item in BENCHMARKS:
        q = item["query"]
        res = orchestrator.run(q)
        resp = res.response

        status_ok = res.status == item["expected_status"]
        conf_ok = resp.confidence_level == item["expected_conf"]
        passed = status_ok and conf_ok

        result_str = "✅ PASS" if passed else "❌ FAIL"
        short_q = (q[:29] + "...") if len(q) > 32 else q
        print(f"{short_q:<32} | {res.status:<23} | {resp.confidence_level.value:<10} | {res.execution_time_ms:<6.1f} ms | {result_str}")

    print("=" * 85)

if __name__ == "__main__":
    run_evaluation()
