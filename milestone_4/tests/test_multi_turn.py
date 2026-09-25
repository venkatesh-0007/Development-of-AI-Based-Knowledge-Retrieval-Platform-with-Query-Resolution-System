"""End-to-End multi-turn conversational memory tests (Phase 9 & Phase 16).

Validates:
1. Multi-turn anaphora resolution ("What is TCP?" -> "What are its advantages?")
2. Multi-step pronoun resolution across turns
3. Context preservation without leaking irrelevant prior domain facts
4. History rolling window constraints (max_history_turns)
5. Effective query transcription and transparency recording
"""
import sys
import unittest
from pathlib import Path

m4_root = str(Path(__file__).resolve().parent.parent)
if m4_root not in sys.path:
    sys.path.insert(0, m4_root)

from app.embeddings.embedder import Embedder
from app.vectorstore.chroma_store import ChromaStore
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.benchmark_dataset import load_canonical_benchmark_documents
from app.agents.query_understanding import QueryUnderstandingAgent
from app.agents.retrieval import RetrievalAgent
from app.agents.response_generation import ResponseGenerationAgent
from app.agents.clarification import ClarificationAgent
from app.agents.memory import ConversationMemoryAgent
from app.agents.orchestrator import MultiAgentOrchestrator
from app.agents.models import ConversationTurn


class TestMultiTurnMemory(unittest.TestCase):
    """Verifies multi-turn conversational context tracking and pronoun resolution."""

    @classmethod
    def setUpClass(cls):
        cls.embedder = Embedder()
        cls.store = ChromaStore(collection_name="test_multi_turn_suite")
        cls.store.clear()
        cls.pipeline = IngestionPipeline(cls.embedder, cls.store)
        load_canonical_benchmark_documents(cls.pipeline)

        cls.clarification_agent = ClarificationAgent()
        cls.query_agent = QueryUnderstandingAgent(clarification_agent=cls.clarification_agent)
        cls.retrieval_agent = RetrievalAgent(cls.embedder, cls.store, confidence_threshold=0.35)
        cls.response_agent = ResponseGenerationAgent()
        cls.memory_agent = ConversationMemoryAgent(max_history_turns=4)

        cls.orchestrator = MultiAgentOrchestrator(
            retrieval_agent=cls.retrieval_agent,
            query_understanding_agent=cls.query_agent,
            response_generation_agent=cls.response_agent,
            clarification_agent=cls.clarification_agent,
            memory_agent=cls.memory_agent
        )

    def test_multi_turn_anaphora_tcp_flow(self):
        """User asks 'What is TCP?' then 'What are its advantages?' - second query resolves 'its'."""
        session_id = "sess_tcp_anaphora"

        # Turn 1
        res1 = self.orchestrator.run("What is TCP?", session_id=session_id)
        self.assertEqual(res1.status, "success")
        self.assertIn("TCP", res1.response.answer)

        # Turn 2: Follow-up with 'its'
        res2 = self.orchestrator.run("What are its advantages?", session_id=session_id)
        self.assertEqual(res2.status, "success")
        self.assertIsNotNone(res2.response.transparency)
        rewritten = res2.response.transparency.anaphora_rewritten_query or ""
        self.assertTrue("TCP" in rewritten or "TCP" in res2.response.answer)

        # Turn 3: Second follow-up
        res3 = self.orchestrator.run("How does it ensure reliable delivery?", session_id=session_id)
        self.assertEqual(res3.status, "success")
        rewritten3 = res3.response.transparency.anaphora_rewritten_query or ""
        self.assertTrue("TCP" in rewritten3 or "handshake" in res3.response.answer.lower() or "reliable" in res3.response.answer.lower())

    def test_multi_turn_across_domains_with_reset(self):
        """Switch from ML to Cybersecurity and verify pronoun binds to the latest active entity."""
        session_id = "sess_ml_to_cyber"

        # ML turn
        t1 = self.orchestrator.run("What is Random Forest in machine learning?", session_id=session_id)
        self.assertEqual(t1.status, "success")

        # ML follow-up
        t2 = self.orchestrator.run("How does it prevent overfitting?", session_id=session_id)
        self.assertEqual(t2.status, "success")
        self.assertTrue("Random Forest" in (t2.response.transparency.anaphora_rewritten_query or t2.response.answer))

        # Switch to Cybersecurity
        t3 = self.orchestrator.run("What is AES encryption mechanism?", session_id=session_id)
        self.assertEqual(t3.status, "success")

        # Cyber follow-up: 'it' should bind to AES, NOT Random Forest
        t4 = self.orchestrator.run("What is its primary function and key length?", session_id=session_id)
        self.assertEqual(t4.status, "success")
        self.assertFalse("Random Forest" in t4.response.answer, "Old ML domain leaked into cybersecurity turn")
        self.assertTrue(any("cryptography_and_zero_trust.csv" in s or "cybersecurity_fundamentals.txt" in s for s in t4.response.sources))

    def test_rolling_window_limits_memory_size(self):
        """Verify memory maintains only rolling window up to max_history_turns."""
        session_id = "sess_rolling_window"
        session = self.memory_agent.get_or_create_session(session_id)

        for i in range(8):
            turn = ConversationTurn(
                query=f"Query {i}",
                effective_query=f"Query {i}",
                response=f"Answer {i}",
                confidence_score=0.8,
                sources=["test.txt"],
                key_entities=[f"Entity_{i}"]
            )
            self.memory_agent.record_turn(session, turn)

        # Configured max_history_turns is 4
        self.assertEqual(len(session.turns), 4)
        self.assertEqual(session.turns[0].query, "Query 4")
        self.assertEqual(session.turns[-1].query, "Query 7")


if __name__ == "__main__":
    unittest.main()
