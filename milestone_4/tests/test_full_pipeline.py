"""Comprehensive end-to-end integration test across all query types, modalities, and agents (Phase 9).

Validates all Phase 9 requirements:
1. Factual: "What is TCP?"
2. Procedural: "How does the TCP three-way handshake work?"
3. Comparative: "Compare TCP and UDP."
4. Ambiguous: "Explain tree." -> clarification requested -> user disambiguates -> resolution
5. Multi-part: "What is TCP and how is it different from UDP?"
6. Multi-turn: Turn 1 + Turn 2 anaphora follow-up
7. Context switching: ML -> Network -> Cybersecurity
8. Unknown query: Concepts outside knowledge base produce insufficient evidence, no hallucinations
9. Transparency: Payload verification
10. Analytics: Every query logs telemetry entry into persistent storage
"""
import os
import sys
import unittest
import tempfile
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
from app.agents.models import QueryType, ConfidenceLevel
from app.analytics.storage import AnalyticsStorage
from app.analytics.tracker import AnalyticsTracker
from app.analytics.models import ResolutionStatus


class TestFullPipeline(unittest.TestCase):
    """End-to-end test suite for the complete Milestone 4 pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.analytics_db = os.path.join(cls.temp_dir.name, "pipeline_analytics.db")
        cls.analytics_storage = AnalyticsStorage(db_path=cls.analytics_db)
        cls.analytics_tracker = AnalyticsTracker(cls.analytics_storage, low_confidence_threshold=0.45)

        cls.embedder = Embedder()
        cls.store = ChromaStore(collection_name="test_full_pipeline_store")
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
            memory_agent=cls.memory_agent,
            analytics_tracker=cls.analytics_tracker
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_01_factual_query(self):
        """1. Factual query: What is TCP?"""
        res = self.orchestrator.run("What is TCP?")
        self.assertEqual(res.status, "success")
        self.assertEqual(res.response.query_analysis.query_type, QueryType.FACTUAL)
        self.assertTrue(res.response.has_sufficient_evidence)
        self.assertGreater(res.response.confidence_score, 0.40)
        self.assertIn("TCP", res.response.answer)
        self.assertTrue(any("computer_networks.txt" in s for s in res.response.sources))

    def test_02_procedural_query(self):
        """2. Procedural query: How does the TCP three-way handshake work?"""
        res = self.orchestrator.run("Explain the step by step process of the TCP three-way handshake")
        self.assertEqual(res.status, "success")
        self.assertEqual(res.response.query_analysis.query_type, QueryType.PROCEDURAL)
        self.assertIn("Step", res.response.answer)
        self.assertIn("SYN", res.response.answer)

    def test_03_comparative_query(self):
        """3. Comparative query: Compare TCP and UDP."""
        res = self.orchestrator.run("Compare TCP and UDP protocols")
        self.assertEqual(res.status, "success")
        self.assertEqual(res.response.query_analysis.query_type, QueryType.COMPARATIVE)
        self.assertTrue(res.response.has_sufficient_evidence)
        self.assertIn("Comparative", res.response.answer)

    def test_04_ambiguous_query_and_clarification_resolution(self):
        """4. Ambiguous query: Explain tree -> disambiguate -> resolution."""
        res = self.orchestrator.run("Explain tree")
        self.assertEqual(res.status, "clarification_requested")
        self.assertIsNotNone(res.clarification_request)

        # Resolve clarification
        resolved = self.orchestrator.resolve_clarification(
            clarification_id=res.clarification_request.clarification_id,
            user_response="Decision Trees (Machine Learning / Classification)"
        )
        self.assertEqual(resolved.status, "clarified_success")
        self.assertIn("Decision Trees", resolved.response.refined_query)
        self.assertTrue(resolved.response.has_sufficient_evidence)

    def test_05_multi_part_query(self):
        """5. Multi-part query: What is TCP and how is it different from UDP?"""
        res = self.orchestrator.run("What is TCP and also how is it different from UDP?")
        self.assertEqual(res.status, "success")
        self.assertEqual(res.response.query_analysis.query_type, QueryType.MULTI_PART)
        self.assertTrue(res.response.has_sufficient_evidence)

    def test_06_multi_turn_anaphora(self):
        """6. Multi-turn anaphora follow-up resolution."""
        session_id = "sess_full_pipeline_multi_turn"
        t1 = self.orchestrator.run("What is Support Vector Machine (SVM)?", session_id=session_id)
        self.assertEqual(t1.status, "success")

        t2 = self.orchestrator.run("What are its key characteristics?", session_id=session_id)
        self.assertEqual(t2.status, "success")
        self.assertTrue("Support Vector Machine" in (t2.response.transparency.anaphora_rewritten_query or t2.response.answer))

    def test_07_context_switching_across_domains(self):
        """7. Context switching across ML -> Computer Networks -> Cybersecurity."""
        session_id = "sess_pipeline_context_switch"

        # ML
        r1 = self.orchestrator.run("What is supervised machine learning?", session_id=session_id)
        self.assertEqual(r1.response.query_analysis.domain, "machine_learning")

        # Network
        r2 = self.orchestrator.run("What is TCP in computer networks?", session_id=session_id)
        self.assertEqual(r2.response.query_analysis.domain, "computer_networks")

        # Cybersecurity
        r3 = self.orchestrator.run("What is Zero Trust Security Architecture?", session_id=session_id)
        self.assertEqual(r3.response.query_analysis.domain, "cybersecurity")

    def test_08_unknown_query_insufficient_evidence(self):
        """8. Unknown query: concepts outside knowledge base produce insufficient evidence, no hallucinations."""
        res = self.orchestrator.run("What is quantum annealing hardware architecture?")
        self.assertFalse(res.response.has_sufficient_evidence)
        self.assertIn(res.response.confidence_level, [ConfidenceLevel.LOW, ConfidenceLevel.NONE])
        self.assertIn("No sufficiently relevant information was found", res.response.answer)

    def test_09_transparency_and_analytics_logging(self):
        """9 & 10. Transparency payload and persistent analytics telemetry validation."""
        initial_count = self.analytics_storage.get_summary().total_queries

        query_text = "What is AES symmetric encryption and key length?"
        res = self.orchestrator.run(query_text)
        self.assertEqual(res.status, "success")
        self.assertIsNotNone(res.response.transparency)

        # Telemetry verification
        summary = self.analytics_storage.get_summary()
        self.assertEqual(summary.total_queries, initial_count + 1)
        self.assertGreater(summary.avg_confidence, 0.0)
        self.assertGreater(summary.avg_latency_ms, 0.0)

        # Audit query list
        queries = self.analytics_storage.get_queries(limit=1)
        self.assertEqual(len(queries), 1)
        latest = queries[0]
        self.assertEqual(latest.query_text, query_text)
        self.assertEqual(latest.status, ResolutionStatus.RESOLVED)


if __name__ == "__main__":
    unittest.main()
