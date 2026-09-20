"""End-to-End Integration Tests for Milestone 3 (Full Pipeline)."""
import unittest
from typing import List, Dict, Any

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

class MockRetrievalAgent:
    """Mock retrieval agent returning deterministic chunks for end-to-end testing."""
    def __init__(self):
        self.doc_kb = {
            "tcp": [
                RetrievalChunk(
                    chunk_id="chunk_tcp_1",
                    document_name="networking_protocols.txt",
                    content="Transmission Control Protocol (TCP) is a connection-oriented protocol providing reliable, ordered delivery of byte streams.",
                    score=0.94,
                    rank=1
                ),
                RetrievalChunk(
                    chunk_id="chunk_tcp_2",
                    document_name="networking_protocols.txt",
                    content="TCP uses a three-way handshake (SYN, SYN-ACK, ACK) to establish a reliable connection before data transfer.",
                    score=0.89,
                    rank=2
                )
            ],
            "tree": [
                RetrievalChunk(
                    chunk_id="chunk_tree_1",
                    document_name="ml_decision_trees.txt",
                    content="Decision Tree is a non-parametric supervised learning method used for classification and regression.",
                    score=0.91,
                    rank=1
                )
            ],
            "svm": [
                RetrievalChunk(
                    chunk_id="chunk_svm_1",
                    document_name="ml_algorithms.txt",
                    content="Support Vector Machines (SVM) find the optimal hyperplane maximizing the margin between different classes.",
                    score=0.93,
                    rank=1
                )
            ]
        }

    def retrieve(self, query_input, top_k=None, threshold=None) -> RetrievalResult:
        query_text = query_input.query if hasattr(query_input, "query") else str(query_input)
        q = query_text.lower()
        matched = []
        if "tcp" in q or "transmission control protocol" in q:
            matched = self.doc_kb["tcp"]
        elif "decision tree" in q or "tree" in q:
            matched = self.doc_kb["tree"]
        elif "svm" in q or "support vector" in q:
            matched = self.doc_kb["svm"]
        else:
            matched = [
                RetrievalChunk(
                    chunk_id="chunk_default",
                    document_name="general_knowledge.txt",
                    content=f"General information related to {query_text}",
                    score=0.75,
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

class TestMilestone3EndToEnd(unittest.TestCase):
    def setUp(self):
        self.clarification_agent = ClarificationAgent()
        self.query_understanding = QueryUnderstandingAgent(clarification_agent=self.clarification_agent)
        self.retrieval_agent = MockRetrievalAgent()
        self.response_generation = ResponseGenerationAgent()
        self.memory_agent = ConversationMemoryAgent()
        self.voice_module = VoiceModule()

        self.orchestrator = MultiAgentOrchestrator(
            retrieval_agent=self.retrieval_agent,
            query_understanding_agent=self.query_understanding,
            response_generation_agent=self.response_generation,
            clarification_agent=self.clarification_agent,
            memory_agent=self.memory_agent
        )

    def test_multi_turn_conversation_and_anaphora_resolution(self):
        session_id = "session_e2e_1"

        # Turn 1: Ask about TCP
        res1 = self.orchestrator.run(
            query="What is Transmission Control Protocol (TCP)?",
            session_id=session_id
        )
        self.assertEqual(res1.status, "success")
        self.assertIn("Transmission Control Protocol (TCP)", res1.response.answer)
        self.assertIsNotNone(res1.response.transparency)
        self.assertIsNotNone(res1.response.spoken_text)

        # Turn 2: Follow-up with pronoun "How does it establish a connection?"
        res2 = self.orchestrator.run(
            query="How does it establish a connection?",
            session_id=session_id
        )
        self.assertEqual(res2.status, "success")
        # Ensure memory rewritten effective query was passed to response
        self.assertIn("TCP", res2.response.answer)

        # Verify session has 2 turns
        session = self.memory_agent.get_or_create_session(session_id)
        self.assertEqual(len(session.turns), 2)
        self.assertEqual(session.turns[0].query, "What is Transmission Control Protocol (TCP)?")
        self.assertIn("Transmission Control Protocol (TCP)", session.turns[1].effective_query)

    def test_clarification_dialogue_in_session(self):
        session_id = "session_e2e_2"

        # Turn 1: Ambiguous polysemous query "Explain tree"
        res1 = self.orchestrator.run(
            query="Explain tree",
            session_id=session_id
        )
        self.assertEqual(res1.status, "clarification_requested")
        self.assertIsNotNone(res1.clarification_request)
        req_id = res1.clarification_request.clarification_id

        # Turn 2: User resolves clarification by selecting "Decision Trees in Machine Learning"
        res2 = self.orchestrator.resolve_clarification(
            clarification_id=req_id,
            user_response="Decision Trees in Machine Learning",
            session_id=session_id
        )
        self.assertEqual(res2.status, "clarified_success")
        self.assertTrue(res2.response.is_clarified_resolution)
        self.assertIn("Decision Tree", res2.response.answer)
        self.assertIsNotNone(res2.response.transparency)

        # Ensure turn is recorded in memory
        session = self.memory_agent.get_or_create_session(session_id)
        self.assertEqual(len(session.turns), 1)
        self.assertIn("Decision Trees", session.turns[0].effective_query)

    def test_transparency_and_voice_synthesis_integration(self):
        res = self.orchestrator.run("What is Support Vector Machines (SVM)?")
        self.assertEqual(res.status, "success")

        # Verify Transparency Payload
        trans = res.response.transparency
        self.assertIsNotNone(trans)
        self.assertGreaterEqual(len(trans.retrieved_chunks), 1)
        self.assertGreaterEqual(len(trans.citations), 1)
        self.assertEqual(trans.score_breakdown.confidence_level, ConfidenceLevel.HIGH)
        self.assertAlmostEqual(trans.score_breakdown.top_chunk_score, 0.93, places=2)

        # Verify Spoken Text (clean of citations/code blocks)
        spoken = res.response.spoken_text
        self.assertIsNotNone(spoken)
        self.assertNotIn("[1]", spoken)
        self.assertNotIn("```", spoken)

if __name__ == "__main__":
    unittest.main()
