"""Evaluation script for Milestone 1 Retrieval Accuracy."""
import sys
import time
from pathlib import Path

# Add milestone_1 root to sys.path
milestone_1_root = Path(__file__).resolve().parent.parent
if str(milestone_1_root) not in sys.path:
    sys.path.insert(0, str(milestone_1_root))

from app.embeddings.embedder import Embedder
from app.vectorstore.chroma_store import ChromaStore
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.pipeline import RetrievalPipeline

BENCHMARK_QUERIES = [
    # Domain 1: Machine Learning
    {"domain": "ML", "type": "factual", "query": "What is supervised learning?", "expected_file": "ml_overview.txt"},
    {"domain": "ML", "type": "factual", "query": "What is overfitting?", "expected_file": "ml_overview.txt"},
    {"domain": "ML", "type": "procedural", "query": "How does gradient descent optimization work?", "expected_file": "ml_overview.txt"},
    {"domain": "ML", "type": "comparative", "query": "Compare supervised and unsupervised learning", "expected_file": "ml_overview.txt"},
    {"domain": "ML", "type": "factual", "query": "What is the primary use case of Random Forest?", "expected_file": "ml_algorithms.csv"},
    # Domain 2: Computer Networks
    {"domain": "Networks", "type": "factual", "query": "What are the layers of the OSI model?", "expected_file": "computer_networks.txt"},
    {"domain": "Networks", "type": "procedural", "query": "How does the DNS resolution process work?", "expected_file": "computer_networks.txt"},
    {"domain": "Networks", "type": "comparative", "query": "Compare TCP vs UDP protocols", "expected_file": "computer_networks.txt"},
    {"domain": "Networks", "type": "factual", "query": "Which port does HTTPS use?", "expected_file": "network_protocols.csv"},
    {"domain": "Networks", "type": "factual", "query": "What transport layer protocol does DNS use?", "expected_file": "network_protocols.csv"},
]

def run_evaluation():
    print("=" * 60)
    print("Evaluating Milestone 1 Retrieval Accuracy")
    print("=" * 60)

    embedder = Embedder()
    store = ChromaStore(path="data/chroma", collection_name="m1_eval_collection")
    ingestion = IngestionPipeline(embedder, store)
    retrieval = RetrievalPipeline(embedder, store)

    # Ingest sample docs
    sample_dir = milestone_1_root / "data" / "sample_docs"
    for doc in sample_dir.glob("*.*"):
        if doc.suffix.lower() in [".txt", ".csv"]:
            ingestion.ingest(doc.read_bytes(), doc.name, chunk_size=800, overlap=100)

    top_1_hits = 0
    top_3_hits = 0
    top_5_hits = 0
    total = len(BENCHMARK_QUERIES)

    print(f"\nRunning {total} benchmark queries across ML & Networks domains...\n")
    print(f"{'Domain':<10} | {'Type':<12} | {'Top-1':<6} | {'Top-3':<6} | {'Top-5':<6} | {'Query'}")
    print("-" * 75)

    for item in BENCHMARK_QUERIES:
        query = item["query"]
        expected = item["expected_file"]
        
        results = retrieval.retrieve(query, top_k=5)
        sources = [r.get("metadata", {}).get("source", "") for r in results]

        in_top_1 = expected in sources[:1]
        in_top_3 = expected in sources[:3]
        in_top_5 = expected in sources[:5]

        if in_top_1: top_1_hits += 1
        if in_top_3: top_3_hits += 1
        if in_top_5: top_5_hits += 1

        print(f"{item['domain']:<10} | {item['type']:<12} | {'✓' if in_top_1 else '✗':<6} | {'✓' if in_top_3 else '✗':<6} | {'✓' if in_top_5 else '✗':<6} | {query}")

    print("-" * 75)
    print(f"Top-1 Accuracy: {top_1_hits / total * 100:.1f}% ({top_1_hits}/{total})")
    print(f"Top-3 Accuracy: {top_3_hits / total * 100:.1f}% ({top_3_hits}/{total})")
    print(f"Top-5 Accuracy: {top_5_hits / total * 100:.1f}% ({top_5_hits}/{total})")
    print("=" * 60)

if __name__ == "__main__":
    run_evaluation()
