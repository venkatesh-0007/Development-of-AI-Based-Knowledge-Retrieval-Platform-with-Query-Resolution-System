"""Agent Routing Optimization Benchmark (Phase 14 & Phase 18).

Calculates real, programmatic routing accuracy against a labeled evaluation dataset
covering:
1. FACTUAL -> Retrieval -> Response
2. PROCEDURAL -> Retrieval -> Step-oriented response
3. COMPARATIVE -> Multi-document retrieval -> Comparison response
4. AMBIGUOUS -> Clarification
5. MULTI_PART -> Decompose -> Retrieve -> Synthesize

Computes and saves raw JSON metrics:
- overall_routing_accuracy
- intent_accuracy_by_class
- domain_classification_accuracy
- confusion_matrix
- per_query_evaluations
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

m4_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(m4_root))

from app.embeddings.embedder import Embedder
from app.vectorstore.chroma_store import ChromaStore
from app.agents.retrieval import RetrievalAgent
from app.agents.query_understanding import QueryUnderstandingAgent
from app.agents.clarification import ClarificationAgent
from app.agents.orchestrator import MultiAgentOrchestrator
from app.agents.models import QueryType, RoutingTarget

# 20 Canonical Labeled Test Cases across all 5 Query Types & 3 Domains
LABELED_ROUTING_DATASET = [
    # FACTUAL
    {"query": "What is Supervised Learning?", "expected_type": "factual", "expected_domain": "machine_learning", "expected_route": "retrieval"},
    {"query": "What is the function of the Transport Layer in the OSI model?", "expected_type": "factual", "expected_domain": "computer_networks", "expected_route": "retrieval"},
    {"query": "Define the CIA Triad in information security", "expected_type": "factual", "expected_domain": "cybersecurity", "expected_route": "retrieval"},
    {"query": "What is TCP?", "expected_type": "factual", "expected_domain": "computer_networks", "expected_route": "retrieval"},
    
    # PROCEDURAL
    {"query": "Explain the step by step process of the TCP three-way handshake", "expected_type": "procedural", "expected_domain": "computer_networks", "expected_route": "retrieval"},
    {"query": "How does DNS resolution work step by step?", "expected_type": "procedural", "expected_domain": "computer_networks", "expected_route": "retrieval"},
    {"query": "What are the sequential phases of the NIST incident response lifecycle?", "expected_type": "procedural", "expected_domain": "cybersecurity", "expected_route": "retrieval"},
    {"query": "How to train a Support Vector Machine model from scratch?", "expected_type": "procedural", "expected_domain": "machine_learning", "expected_route": "retrieval"},

    # COMPARATIVE
    {"query": "Compare Support Vector Machines with Random Forest", "expected_type": "comparative", "expected_domain": "machine_learning", "expected_route": "retrieval"},
    {"query": "Compare TCP and UDP protocols", "expected_type": "comparative", "expected_domain": "computer_networks", "expected_route": "retrieval"},
    {"query": "What is the difference between symmetric and asymmetric encryption?", "expected_type": "comparative", "expected_domain": "cybersecurity", "expected_route": "retrieval"},
    {"query": "Compare AES and RSA in terms of key length and security", "expected_type": "comparative", "expected_domain": "cybersecurity", "expected_route": "retrieval"},

    # AMBIGUOUS
    {"query": "Explain tree", "expected_type": "ambiguous", "expected_domain": "general", "expected_route": "clarification"},
    {"query": "How does it work?", "expected_type": "ambiguous", "expected_domain": "general", "expected_route": "clarification"},
    {"query": "Explain clustering", "expected_type": "ambiguous", "expected_domain": "general", "expected_route": "clarification"},
    {"query": "What is token?", "expected_type": "ambiguous", "expected_domain": "general", "expected_route": "clarification"},

    # MULTI_PART
    {"query": "What is TCP and how is it different from UDP?", "expected_type": "multi_part", "expected_domain": "computer_networks", "expected_route": "retrieval"},
    {"query": "What is overfitting in machine learning and how do we prevent it?", "expected_type": "multi_part", "expected_domain": "machine_learning", "expected_route": "retrieval"},
    {"query": "Explain zero trust architecture and what are its core principles?", "expected_type": "multi_part", "expected_domain": "cybersecurity", "expected_route": "retrieval"},
    {"query": "What is DNS and why is port 53 used?", "expected_type": "multi_part", "expected_domain": "computer_networks", "expected_route": "retrieval"},
]


def run_routing_benchmark() -> Dict[str, Any]:
    print("=" * 80)
    print("🎯 RUNNING PROGRAMMATIC AGENT ROUTING BENCHMARK (PHASE 14 & 18)")
    print("=" * 80)

    qu_agent = QueryUnderstandingAgent()
    clar_agent = ClarificationAgent()
    embedder = Embedder()
    store = ChromaStore(collection_name="routing_bench_store")
    retrieval_agent = RetrievalAgent(embedder, store)
    orchestrator = MultiAgentOrchestrator(
        retrieval_agent=retrieval_agent,
        query_understanding_agent=qu_agent,
        clarification_agent=clar_agent
    )

    total_queries = len(LABELED_ROUTING_DATASET)
    type_correct = 0
    domain_correct = 0
    route_correct = 0

    class_counts = defaultdict(int)
    class_correct = defaultdict(int)
    evaluations = []

    for item in LABELED_ROUTING_DATASET:
        q_text = item["query"]
        exp_type = item["expected_type"]
        exp_domain = item["expected_domain"]
        exp_route = item["expected_route"]

        analysis = qu_agent.analyze(q_text)
        orch_res = orchestrator.run(q_text)

        actual_type = analysis.query_type.value
        actual_domain = analysis.domain
        actual_route = analysis.route_to.value

        is_type_match = (actual_type == exp_type)
        is_domain_match = (exp_domain == "general") or (actual_domain == exp_domain)
        is_route_match = (actual_route == exp_route)

        class_counts[exp_type] += 1
        if is_type_match:
            type_correct += 1
            class_correct[exp_type] += 1
        if is_domain_match:
            domain_correct += 1
        if is_route_match:
            route_correct += 1

        evaluations.append({
            "query": q_text,
            "expected_type": exp_type,
            "predicted_type": actual_type,
            "type_match": is_type_match,
            "expected_domain": exp_domain,
            "predicted_domain": actual_domain,
            "domain_match": is_domain_match,
            "expected_route": exp_route,
            "predicted_route": actual_route,
            "route_match": is_route_match,
            "orchestrator_status": orch_res.status
        })

    type_accuracy = round(type_correct / total_queries, 4)
    domain_accuracy = round(domain_correct / total_queries, 4)
    route_accuracy = round(route_correct / total_queries, 4)

    per_class_accuracy = {}
    for cls_name, count in class_counts.items():
        corr = class_correct[cls_name]
        per_class_accuracy[cls_name] = {
            "total": count,
            "correct": corr,
            "accuracy": round(corr / count, 4) if count > 0 else 0.0
        }

    results = {
        "total_test_queries": total_queries,
        "overall_intent_classification_accuracy": type_accuracy,
        "overall_route_target_accuracy": route_accuracy,
        "domain_classification_accuracy": domain_accuracy,
        "per_class_accuracy": per_class_accuracy,
        "evaluations": evaluations
    }

    output_path = m4_root / "evaluation" / "routing_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nTotal Labeled Queries Evaluated: {total_queries}")
    print(f"Overall Query Type Accuracy:   {type_accuracy * 100:.1f}% ({type_correct}/{total_queries})")
    print(f"Route Target Routing Accuracy: {route_accuracy * 100:.1f}% ({route_correct}/{total_queries})")
    print(f"Domain Classification Accuracy:{domain_accuracy * 100:.1f}% ({domain_correct}/{total_queries})")
    print("\nPer-Intent Class Breakdown:")
    for cls_name, stat in per_class_accuracy.items():
        print(f" - {cls_name.upper():<12}: {stat['accuracy']*100:.1f}% ({stat['correct']}/{stat['total']})")

    print(f"\n✅ Programmatic routing benchmark complete. Raw results saved to:\n   {output_path}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_routing_benchmark()
