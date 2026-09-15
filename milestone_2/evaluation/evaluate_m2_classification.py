"""Evaluation script for Milestone 2.1 Query Understanding Classifier."""
import sys
import time
from pathlib import Path

milestone_2_root = Path(__file__).resolve().parent.parent
if str(milestone_2_root) not in sys.path:
    sys.path.insert(0, str(milestone_2_root))

from app.agents.query_understanding import QueryUnderstandingAgent
from app.agents.models import QueryType, RoutingTarget

BENCHMARK_SET = [
    # Machine Learning Domain
    {"query": "What is supervised learning?", "expected_type": QueryType.FACTUAL, "expected_route": RoutingTarget.RETRIEVAL},
    {"query": "What is regression?", "expected_type": QueryType.FACTUAL, "expected_route": RoutingTarget.RETRIEVAL},
    {"query": "How does gradient descent work?", "expected_type": QueryType.PROCEDURAL, "expected_route": RoutingTarget.RETRIEVAL},
    {"query": "How can a model be trained?", "expected_type": QueryType.PROCEDURAL, "expected_route": RoutingTarget.RETRIEVAL},
    {"query": "Compare classification and regression.", "expected_type": QueryType.COMPARATIVE, "expected_route": RoutingTarget.RETRIEVAL},
    {"query": "What is the difference between supervised and unsupervised learning?", "expected_type": QueryType.COMPARATIVE, "expected_route": RoutingTarget.RETRIEVAL},
    {"query": "Tell me about it.", "expected_type": QueryType.AMBIGUOUS, "expected_route": RoutingTarget.CLARIFICATION},
    {"query": "Explain this.", "expected_type": QueryType.AMBIGUOUS, "expected_route": RoutingTarget.CLARIFICATION},
    {"query": "SVM", "expected_type": QueryType.FACTUAL, "expected_route": RoutingTarget.RETRIEVAL},
    {"query": "What is overfitting in machine learning models?", "expected_type": QueryType.FACTUAL, "expected_route": RoutingTarget.RETRIEVAL},
    # Computer Networks Domain
    {"query": "What is TCP?", "expected_type": QueryType.FACTUAL, "expected_route": RoutingTarget.RETRIEVAL},
    {"query": "What is HTTP?", "expected_type": QueryType.FACTUAL, "expected_route": RoutingTarget.RETRIEVAL},
    {"query": "How does TCP establish a connection?", "expected_type": QueryType.PROCEDURAL, "expected_route": RoutingTarget.RETRIEVAL},
    {"query": "How do I configure a network connection?", "expected_type": QueryType.PROCEDURAL, "expected_route": RoutingTarget.RETRIEVAL},
    {"query": "Compare TCP and UDP.", "expected_type": QueryType.COMPARATIVE, "expected_route": RoutingTarget.RETRIEVAL},
    {"query": "What is the difference between HTTP and HTTPS?", "expected_type": QueryType.COMPARATIVE, "expected_route": RoutingTarget.RETRIEVAL},
    {"query": "What about that?", "expected_type": QueryType.AMBIGUOUS, "expected_route": RoutingTarget.CLARIFICATION},
    {"query": "How does it work?", "expected_type": QueryType.AMBIGUOUS, "expected_route": RoutingTarget.CLARIFICATION},
    {"query": "Which port does HTTPS use?", "expected_type": QueryType.FACTUAL, "expected_route": RoutingTarget.RETRIEVAL},
    {"query": "What is quantum computing?", "expected_type": QueryType.FACTUAL, "expected_route": RoutingTarget.RETRIEVAL},
]

def run_evaluation():
    print("=" * 65)
    print("Evaluating Milestone 2.1 Query Understanding Classifier")
    print("=" * 65)

    agent = QueryUnderstandingAgent()
    total = len(BENCHMARK_SET)
    correct = 0
    routing_correct = 0
    total_time_ms = 0.0

    class_counts = {t: 0 for t in QueryType}
    class_correct = {t: 0 for t in QueryType}

    print(f"\n{'Query':<42} | {'Expected':<12} | {'Predicted':<12} | {'Status'}")
    print("-" * 75)

    for item in BENCHMARK_SET:
        q = item["query"]
        expected_t = item["expected_type"]
        expected_r = item["expected_route"]

        t0 = time.perf_counter()
        res = agent.analyze(q)
        t1 = time.perf_counter()
        total_time_ms += (t1 - t0) * 1000

        class_counts[expected_t] += 1
        is_type_ok = res.query_type == expected_t
        is_route_ok = res.route_to == expected_r

        if is_type_ok:
            correct += 1
            class_correct[expected_t] += 1
        if is_route_ok:
            routing_correct += 1

        status_str = "✅ PASS" if (is_type_ok and is_route_ok) else "❌ FAIL"
        short_q = (q[:38] + "...") if len(q) > 40 else q
        print(f"{short_q:<42} | {expected_t.value:<12} | {res.query_type.value:<12} | {status_str}")

    avg_latency = total_time_ms / total

    print("-" * 75)
    print(f"Total Benchmark Queries:        {total}")
    print(f"Overall Classification Accuracy: {correct / total * 100:.1f}% ({correct}/{total})")
    print(f"Deterministic Routing Accuracy:  {routing_correct / total * 100:.1f}% ({routing_correct}/{total})")
    print(f"Average Classifier Latency:      {avg_latency:.3f} ms / query")
    print("\nPer-Class Breakdown:")
    for t in QueryType:
        n = class_counts[t]
        c = class_correct[t]
        pct = (c / n * 100) if n > 0 else 0.0
        print(f"  - {t.value.upper():<12}: {pct:.1f}% ({c}/{n})")
    print("=" * 65)

if __name__ == "__main__":
    run_evaluation()
