"""Evaluation script for Milestone 2.2 Retrieval Agent."""
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
from app.agents.models import QueryType

BENCHMARK_RETRIEVAL = [
    {"query": "What is supervised learning?", "expected_doc": "ml_overview.txt", "type": QueryType.FACTUAL, "min_score": 0.60},
    {"query": "How does gradient descent work?", "expected_doc": "ml_overview.txt", "type": QueryType.PROCEDURAL, "min_score": 0.55},
    {"query": "Compare TCP and UDP.", "expected_doc": "computer_networks.txt", "type": QueryType.COMPARATIVE, "min_score": 0.60},
    {"query": "What is the primary use case of Random Forest?", "expected_doc": "ml_algorithms.csv", "type": QueryType.FACTUAL, "min_score": 0.50},
    {"query": "Which port does HTTPS use?", "expected_doc": "network_protocols.csv", "type": QueryType.FACTUAL, "min_score": 0.65},
    {"query": "What is quantum computing?", "expected_doc": None, "type": QueryType.FACTUAL, "min_score": 0.0},
]

def run_evaluation():
    print("=" * 65)
    print("Evaluating Milestone 2.2 Retrieval Agent (Dynamic Top-K & Filtering)")
    print("=" * 65)

    embedder = Embedder()
    store = ChromaStore(path="data/chroma", collection_name="m2_eval_collection")
    ingestion = IngestionPipeline(embedder, store)
    
    # Ingest benchmark documents
    sample_dir = milestone_2_root / "data" / "sample_docs"
    for sample in sample_dir.glob("*.*"):
        if sample.suffix.lower() in [".txt", ".csv"]:
            ingestion.ingest(sample.read_bytes(), sample.name, chunk_size=800, overlap=100)

    query_agent = QueryUnderstandingAgent()
    retrieval_agent = RetrievalAgent(embedder, store, confidence_threshold=0.45)

    print(f"\n{'Query':<35} | {'Type':<11} | {'Top-K':<5} | {'Top Score':<9} | {'Evidence':<8} | {'Status'}")
    print("-" * 80)

    for item in BENCHMARK_RETRIEVAL:
        q = item["query"]
        qa = query_agent.analyze(q)
        res = retrieval_agent.retrieve(qa)

        expected_doc = item["expected_doc"]
        if expected_doc is None:
            passed = not res.has_sufficient_evidence
        else:
            sources = [c.document_name for c in res.chunks]
            passed = expected_doc in sources and res.has_sufficient_evidence

        status_str = "✅ PASS" if passed else "❌ FAIL"
        short_q = (q[:32] + "...") if len(q) > 35 else q
        print(f"{short_q:<35} | {qa.query_type.value:<11} | {len(res.chunks):<5} | {res.top_score:<9.3f} | {str(res.has_sufficient_evidence):<8} | {status_str}")

    print("=" * 80)

if __name__ == "__main__":
    run_evaluation()
