"""End-to-End tests for Cross-Domain Context Switching and Domain Isolation (Phase 9 & Phase 16).

Verifies:
- Sequential queries across: Machine Learning -> Computer Networks -> Cybersecurity
- Domain switching updates the session's active_domain accurately
- Memory history from an earlier domain does not contaminate retrieval in subsequent domains
- Information isolation across domains is strictly maintained
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


class TestContextSwitching(unittest.TestCase):
    """Tests multi-turn sessions transitioning across distinct domains."""

    @classmethod
    def setUpClass(cls):
        cls.embedder = Embedder()
        cls.store = ChromaStore(collection_name="test_context_switch_suite")
        cls.store.clear()
        cls.pipeline = IngestionPipeline(cls.embedder, cls.store)
        load_canonical_benchmark_documents(cls.pipeline)

        cls.clarification_agent = ClarificationAgent()
        cls.query_agent = QueryUnderstandingAgent(clarification_agent=cls.clarification_agent)
        cls.retrieval_agent = RetrievalAgent(cls.embedder, cls.store, confidence_threshold=0.35)
        cls.response_agent = ResponseGenerationAgent()
        cls.memory_agent = ConversationMemoryAgent()

        cls.orchestrator = MultiAgentOrchestrator(
            retrieval_agent=cls.retrieval_agent,
            query_understanding_agent=cls.query_agent,
            response_generation_agent=cls.response_agent,
            clarification_agent=cls.clarification_agent,
            memory_agent=cls.memory_agent
        )

    def test_three_way_domain_transition(self):
        """Execute ML -> Network -> Cybersecurity in a single session without cross-contamination."""
        session_id = "sess_3way_switch"

        # Turn 1: Machine Learning
        res_ml = self.orchestrator.run("What is Supervised Learning and regression?", session_id=session_id)
        self.assertEqual(res_ml.status, "success")
        self.assertTrue(any("ml_" in s for s in res_ml.response.sources))
        session = self.memory_agent.get_or_create_session(session_id)
        self.assertEqual(session.active_domain, "machine_learning")

        # Turn 2: Switch to Computer Networks
        res_net = self.orchestrator.run("What is TCP and the OSI 7-layer model?", session_id=session_id)
        self.assertEqual(res_net.status, "success")
        # Ensure ML sources did not contaminate Network query
        self.assertTrue(any("network" in s.lower() or "protocol" in s.lower() for s in res_net.response.sources))
        self.assertFalse(any("ml_" in s for s in res_net.response.sources), "ML source contaminated network query")
        session = self.memory_agent.get_or_create_session(session_id)
        self.assertEqual(session.active_domain, "computer_networks")

        # Turn 3: Switch to Cybersecurity
        res_sec = self.orchestrator.run("What is Zero Trust Security Architecture?", session_id=session_id)
        self.assertEqual(res_sec.status, "success")
        self.assertTrue(any("cyber" in s.lower() or "crypto" in s.lower() for s in res_sec.response.sources))
        self.assertFalse(any("ml_" in s for s in res_sec.response.sources), "ML source contaminated security query")
        self.assertFalse(any("network_protocols.csv" in s for s in res_sec.response.sources), "Network source contaminated security query")
        session = self.memory_agent.get_or_create_session(session_id)
        self.assertEqual(session.active_domain, "cybersecurity")

    def test_domain_targeted_follow_up_with_pronoun(self):
        """Test follow-up pronoun in domain 1, switch to domain 2 with follow-up."""
        session_id = "sess_domain_follow_ups"

        # Turn 1: ML
        self.orchestrator.run("What is Support Vector Machine (SVM)?", session_id=session_id)
        # Follow-up in ML
        t2 = self.orchestrator.run("What are its key characteristics?", session_id=session_id)
        self.assertIn("Support Vector Machine", t2.response.transparency.anaphora_rewritten_query or t2.response.answer)

        # Turn 3: Switch to Networks
        t3 = self.orchestrator.run("What is TCP protocol in networking?", session_id=session_id)
        self.assertEqual(t3.status, "success")
        # Follow-up in Networks
        t4 = self.orchestrator.run("What are its advantages and reliability mechanisms?", session_id=session_id)
        self.assertIn("TCP", t4.response.transparency.anaphora_rewritten_query or t4.response.answer)


if __name__ == "__main__":
    unittest.main()
