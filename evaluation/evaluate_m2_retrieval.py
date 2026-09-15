"""Evaluation script for Milestone 2.2 Retrieval Agent."""
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "milestone_2"))

from app.embeddings.embedder import Embedder
from app.vectorstore.chroma_store import ChromaStore
from app.ingestion.pipeline import IngestionPipeline
from app.agents.query_understanding import QueryUnderstandingAgent
from app.agents.retrieval import RetrievalAgent

RETRIEVAL_TEST_QUERIES = [
    # In-domain factual
    {"query": "What is supervised learning?", "should_have_evidence": True},
    {"query": "What is TCP?", "should_have_evidence": True},
    # In-domain procedural
    {"query": "How does gradient descent work?", "should_have_evidence": True},
    {"query": "How does the DNS resolution process work?", "should_have_evidence": True},
    # In-domain comparative
    {"query": "Compare TCP vs UDP protocols", "should_have_evidence": True},
    {"query": "What are the differences between supervised and unsupervised learning?", "should_have_evidence": True},
    # Out-of-domain unavailable queries (Must filter out and report no sufficient evidence)
    {"query": "What is quantum computing?", "should_have_evidence": False},
    {"query": "How does blockchain proof of work mining work?", "should_have_evidence": False},
]

def run_evaluation():
    print("=" * 70)
    print("Evaluating Milestone 2.2 Retrieval Agent (Ranking & Threshold Filtering)")
    print("=" * 70)

    embedder = Embedder()
    store = ChromaStore(path="data/chroma", collection_name="m2_eval_collection")
    ingestion = IngestionPipeline(embedder, store)

    # Ingest sample docs
    sample_dir = root_dir / "milestone_2" / "data" / "sample_docs"
    for doc in sample_dir.glob("*.*"):
        if doc.suffix.lower() in [".txt", ".csv"]:
            ingestion.ingest(doc.read_bytes(), doc.name, chunk_size=800, overlap=100)

    query_agent = QueryUnderstandingAgent()
    retrieval_agent = RetrievalAgent(embedder, store, confidence_threshold=0.45)

    total = len(RETRIEVAL_TEST_QUERIES)
    correct = 0

    print(f"\n{'Query':<42} | {'Top Score':<10} | {'Retrieved':<10} | {'Filtered':<10} | {'Status'}")
    print("-" * 85)

    for item in RETRIEVAL_TEST_QUERIES:
        q = item["query"]
        expected_ev = item["should_have_evidence"]

        qa = query_agent.analyze(q)
        res = retrieval_agent.retrieve(qa)

        is_ok = (res.has_sufficient_evidence == expected_ev)
        if is_ok:
            correct += 1

        short_q = (q[:38] + "...") if len(q) > 40 else q
        status_str = "✅ PASS" if is_ok else "❌ FAIL"
        print(f"{short_q:<42} | {res.top_score:<10.3f} | {len(res.chunks):<10} | {res.filtered_count:<10} | {status_str}")

    print("-" * 85)
    print(f"Retrieval Evidence & Filtering Accuracy: {correct / total * 100:.1f}% ({correct}/{total})")
    print("=" * 70)

if __name__ == "__main__":
    run_evaluation()
