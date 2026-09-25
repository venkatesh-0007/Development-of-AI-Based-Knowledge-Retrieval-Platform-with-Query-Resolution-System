"""Comprehensive End-to-End Integration Test for Milestone 4 (M4.1 - M4.4)."""
import os
import sys
import unittest
import tempfile
from pathlib import Path

# Add milestone_4 root to sys.path
m4_root = str(Path(__file__).resolve().parent.parent)
if m4_root not in sys.path:
    sys.path.insert(0, m4_root)

from app.embeddings.embedder import Embedder
from app.vectorstore.chroma_store import ChromaStore
from app.ingestion.pipeline import IngestionPipeline
from app.agents.query_understanding import QueryUnderstandingAgent
from app.agents.retrieval import RetrievalAgent
from app.agents.response_generation import ResponseGenerationAgent
from app.agents.clarification import ClarificationAgent
from app.agents.memory import ConversationMemoryAgent
from app.agents.voice import VoiceModule
from app.agents.orchestrator import MultiAgentOrchestrator
from app.analytics.storage import AnalyticsStorage
from app.analytics.gap_detector import KnowledgeGapDetector
from app.analytics.tracker import AnalyticsTracker
from app.analytics.models import ResolutionStatus

class TestMilestone4CompletePipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "e2e_analytics.db")
        self.analytics_storage = AnalyticsStorage(db_path=self.db_path)
        self.gap_detector = KnowledgeGapDetector(self.analytics_storage, confidence_threshold=0.45)
        self.analytics_tracker = AnalyticsTracker(self.analytics_storage, low_confidence_threshold=0.45)

        self.embedder = Embedder()
        self.store = ChromaStore(collection_name="e2e_chroma")
        self.store.clear()
        self.pipeline = IngestionPipeline(self.embedder, self.store)

        m4_root_path = Path(__file__).resolve().parent.parent
        sample_dir = m4_root_path / "data" / "sample_docs"

        for f in sample_dir.glob("*.*"):
            self.pipeline.ingest(f.read_bytes(), f.name, chunk_size=800, overlap=150)

        self.clarification_agent = ClarificationAgent()
        self.query_agent = QueryUnderstandingAgent(clarification_agent=self.clarification_agent)
        self.retrieval_agent = RetrievalAgent(self.embedder, self.store, confidence_threshold=0.35)
        self.response_agent = ResponseGenerationAgent()
        self.memory_agent = ConversationMemoryAgent()
        self.voice_module = VoiceModule()

        self.orchestrator = MultiAgentOrchestrator(
            retrieval_agent=self.retrieval_agent,
            query_understanding_agent=self.query_agent,
            response_generation_agent=self.response_agent,
            clarification_agent=self.clarification_agent,
            memory_agent=self.memory_agent,
            analytics_tracker=self.analytics_tracker
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_full_pipeline_with_analytics_and_gap_detection(self):
        res1 = self.orchestrator.run("What is the function of the Transport Layer in the OSI model?")
        self.assertEqual(res1.status, "success")
        
        logs = self.analytics_storage.get_queries(limit=10)
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0].query_text, "What is the function of the Transport Layer in the OSI model?")
        self.assertEqual(logs[0].status, ResolutionStatus.RESOLVED)

        res_clar = self.orchestrator.run("Explain tree")
        self.assertEqual(res_clar.status, "clarification_requested")
        
        logs_after_clar = self.analytics_storage.get_queries(limit=10)
        self.assertEqual(len(logs_after_clar), 2)
        self.assertEqual(logs_after_clar[0].status, ResolutionStatus.CLARIFICATION_REQUESTED)

        res_resolved = self.orchestrator.resolve_clarification(
            clarification_id=res_clar.clarification_request.clarification_id,
            user_response="Decision Trees (Machine Learning / Classification)"
        )
        self.assertEqual(res_resolved.status, "clarified_success")
        
        logs_after_res = self.analytics_storage.get_queries(limit=10)
        self.assertEqual(len(logs_after_res), 3)
        self.assertEqual(logs_after_res[0].status, ResolutionStatus.CLARIFICATION_RESOLVED)

        out_of_domain = [
            "What is Quantum Key Distribution in post-quantum cryptography?",
            "How does Quantum Key Distribution guarantee unconditional security?"
        ]
        for q in out_of_domain:
            self.orchestrator.run(q)

        gaps = self.gap_detector.detect_gaps()
        self.assertGreaterEqual(len(gaps), 1)
        self.assertTrue(any("Quantum" in g.topic or "Key" in g.topic or "Distribution" in g.topic for g in gaps))

        summary = self.analytics_storage.get_summary()
        self.assertEqual(summary.total_queries, 5)
        self.assertGreater(summary.resolution_rate, 0.0)
        self.assertGreaterEqual(summary.active_gaps_count, 1)

if __name__ == "__main__":
    unittest.main()
