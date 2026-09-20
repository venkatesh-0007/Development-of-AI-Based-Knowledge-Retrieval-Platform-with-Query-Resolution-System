"""Benchmark & Evaluation Script for Milestone 3 (Clarification, Memory, Voice, and Transparency)."""
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add milestone_3 root to path
m3_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(m3_root))

from app.agents.models import (
    QueryType,
    RoutingTarget,
    ConfidenceLevel,
    RetrievalChunk,
    RetrievalResult
)
from app.agents.query_understanding import QueryUnderstandingAgent
from app.agents.clarification import ClarificationAgent
from app.agents.response_generation import ResponseGenerationAgent
from app.agents.memory import ConversationMemoryAgent
from app.agents.voice import VoiceModule
from app.agents.orchestrator import MultiAgentOrchestrator

class MockEvaluatorKB:
    """Mock KB for deterministic evaluation of M3 components."""
    def __init__(self):
        self.chunks = {
            "tcp": [
                RetrievalChunk(
                    chunk_id="chk_tcp_1",
                    document_name="tcp_networking.txt",
                    content="Transmission Control Protocol (TCP) guarantees reliable and ordered stream delivery across IP networks.",
                    score=0.95,
                    rank=1
                ),
                RetrievalChunk(
                    chunk_id="chk_tcp_2",
                    document_name="tcp_networking.txt",
                    content="TCP performs flow control and congestion avoidance using sliding window mechanisms.",
                    score=0.88,
                    rank=2
                )
            ],
            "decision tree": [
                RetrievalChunk(
                    chunk_id="chk_dt_1",
                    document_name="ml_decision_trees.txt",
                    content="Decision trees split dataset features using criteria like Gini impurity or information gain entropy.",
                    score=0.92,
                    rank=1
                )
            ],
            "svm": [
                RetrievalChunk(
                    chunk_id="chk_svm_1",
                    document_name="ml_svm.txt",
                    content="Support Vector Machines use kernel functions to map data into high-dimensional space to find separating hyperplanes.",
                    score=0.91,
                    rank=1
                )
            ]
        }

    def retrieve(self, query_input, top_k=None, threshold=None) -> RetrievalResult:
        query_text = query_input.query if hasattr(query_input, "query") else str(query_input)
        q = query_text.lower()
        matched = []
        for key, chks in self.chunks.items():
            if key in q:
                matched.extend(chks)
        
        if not matched:
            matched = [
                RetrievalChunk(
                    chunk_id="chk_generic",
                    document_name="generic.txt",
                    content=f"General context for query: {query_text}",
                    score=0.60,
                    rank=1
                )
            ]

        top_score = matched[0].similarity_score if matched else 0.0
        return RetrievalResult(
            query=query_text,
            query_type=getattr(query_input, "query_type", QueryType.FACTUAL),
            chunks=matched,
            top_score=top_score,
            total_retrieved=len(matched),
            filtered_count=0,
            has_sufficient_evidence=bool(matched),
            confidence_threshold=threshold or 0.45
        )

def run_evaluation():
    print("=" * 70)
    print("🚀 MILESTONE 3 COMPREHENSIVE BENCHMARK & EVALUATION SUITE")
    print("=" * 70)

    clarification_agent = ClarificationAgent()
    query_agent = QueryUnderstandingAgent(clarification_agent=clarification_agent)
    retrieval_agent = MockEvaluatorKB()
    response_agent = ResponseGenerationAgent()
    memory_agent = ConversationMemoryAgent()
    voice_module = VoiceModule()

    orchestrator = MultiAgentOrchestrator(
        retrieval_agent=retrieval_agent,
        query_understanding_agent=query_agent,
        response_generation_agent=response_agent,
        clarification_agent=clarification_agent,
        memory_agent=memory_agent
    )

    # 1. Evaluate Conversation Memory & Anaphora Resolution
    print("\n--- [1/4] Evaluating Conversation Memory & Anaphora Resolution ---")
    memory_test_cases = [
        {
            "turn_1": "What is Transmission Control Protocol (TCP)?",
            "followup": "How does it ensure reliable delivery?",
            "expected_entity": "Transmission Control Protocol (TCP)",
            "expected_resolved_keywords": ["transmission control protocol (tcp)", "reliable delivery"]
        },
        {
            "turn_1": "Explain Decision Tree algorithms in machine learning.",
            "followup": "What are its main splitting criteria?",
            "expected_entity": "Decision Tree",
            "expected_resolved_keywords": ["decision tree", "splitting criteria"]
        }
    ]

    memory_passed = 0
    for idx, tc in enumerate(memory_test_cases, 1):
        session_id = f"eval_mem_{idx}"
        # Turn 1
        res1 = orchestrator.run(tc["turn_1"], session_id=session_id)
        # Turn 2
        res2 = orchestrator.run(tc["followup"], session_id=session_id)
        
        session = memory_agent.get_or_create_session(session_id)
        last_turn = session.turns[-1]
        
        has_entity = any(kw.lower() in last_turn.effective_query.lower() for kw in tc["expected_resolved_keywords"])
        if has_entity and res2.status == "success":
            memory_passed += 1
            print(f"  ✅ Memory Case {idx}: Successfully resolved '{tc['followup']}' -> '{last_turn.effective_query}'")
        else:
            print(f"  ❌ Memory Case {idx}: Failed resolution -> '{last_turn.effective_query}'")

    print(f"  Memory Resolution Score: {memory_passed}/{len(memory_test_cases)} ({memory_passed/len(memory_test_cases)*100:.1f}%)")

    # 2. Evaluate Clarification Agent (M3.1)
    print("\n--- [2/4] Evaluating Clarification Agent Disambiguation ---")
    clar_test_cases = [
        ("Explain tree", True, "Decision Trees in Machine Learning"),
        ("How does it work?", True, "I am asking about TCP protocol"),
        ("What is Support Vector Machines (SVM)?", False, None)
    ]
    clar_passed = 0
    for query, expects_clar, resolve_choice in clar_test_cases:
        res = orchestrator.run(query)
        if expects_clar:
            if res.status == "clarification_requested" and res.clarification_request:
                # Resolve it
                res_resolved = orchestrator.resolve_clarification(
                    res.clarification_request.clarification_id,
                    user_response=resolve_choice
                )
                if res_resolved.status == "clarified_success":
                    clar_passed += 1
                    print(f"  ✅ Clarification Triggered & Resolved: '{query}' -> '{res_resolved.response.refined_query}'")
                else:
                    print(f"  ❌ Clarification Resolution failed for: '{query}'")
            else:
                print(f"  ❌ Clarification was expected but not triggered for: '{query}'")
        else:
            if res.status == "success":
                clar_passed += 1
                print(f"  ✅ Direct Retrieval correctly bypassed clarification: '{query}'")
            else:
                print(f"  ❌ Clarification triggered incorrectly for clear query: '{query}'")

    print(f"  Clarification Pipeline Score: {clar_passed}/{len(clar_test_cases)} ({clar_passed/len(clar_test_cases)*100:.1f}%)")

    # 3. Evaluate Voice Module (M3.3)
    print("\n--- [3/4] Evaluating Voice & Speech Sanitizer ---")
    voice_sample = "### Grounded Answer:\n**TCP** is reliable [1].\n```python\nsocket.connect()\n```\nSee [RFC 793](https://tools.ietf.org)."
    cleaned = voice_module.clean_for_speech(voice_sample)
    voice_passed = (
        "###" not in cleaned and
        "[1]" not in cleaned and
        "socket.connect" not in cleaned and
        "https://" not in cleaned and
        "**" not in cleaned
    )
    if voice_passed:
        print(f"  ✅ Speech Sanitizer stripped markdown, code blocks, links, and citations cleanly.")
    else:
        print(f"  ❌ Speech Sanitizer left markdown artifacts: {cleaned}")

    # 4. Evaluate Response Transparency Panel (M3.4)
    print("\n--- [4/4] Evaluating Response Transparency & Confidence Formula ---")
    trans_res = orchestrator.run("What is Support Vector Machines (SVM)?")
    trans_payload = trans_res.response.transparency
    trans_passed = (
        trans_payload is not None and
        len(trans_payload.retrieved_chunks) > 0 and
        len(trans_payload.citations) > 0 and
        trans_payload.score_breakdown.combined_confidence > 0.0
    )
    if trans_passed:
        sb = trans_payload.score_breakdown
        print(f"  ✅ Transparency Payload verified:")
        print(f"     - Top Chunk Score: {sb.top_chunk_score:.3f}")
        print(f"     - Avg Top-K Score: {sb.avg_top_k_score:.3f}")
        print(f"     - Combined Confidence: {sb.combined_confidence:.3f} ({sb.confidence_level.value.upper()})")
        print(f"     - Citations Count: {len(trans_payload.citations)}")
    else:
        print(f"  ❌ Transparency payload incomplete or missing.")

    print("\n" + "=" * 70)
    print("🎉 MILESTONE 3 EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_evaluation()
