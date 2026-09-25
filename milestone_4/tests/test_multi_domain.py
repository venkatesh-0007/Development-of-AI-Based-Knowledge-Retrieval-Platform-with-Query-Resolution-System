"""End-to-End System Testing across 3 Knowledge Base Domains (M4.2)."""
import sys
import unittest
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
from app.agents.models import QueryType, ConfidenceLevel

class TestMultiDomainPlatform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedder = Embedder()
        cls.store = ChromaStore(collection_name="test_multi_domain_suite")
        cls.store.clear()
        cls.pipeline = IngestionPipeline(cls.embedder, cls.store)

        m4_root_path = Path(__file__).resolve().parent.parent
        sample_dir = m4_root_path / "data" / "sample_docs"

        for f in sample_dir.glob("*.*"):
            cls.pipeline.ingest(f.read_bytes(), f.name, chunk_size=800, overlap=150)

        cls.clarification_agent = ClarificationAgent()
        cls.query_agent = QueryUnderstandingAgent(clarification_agent=cls.clarification_agent)
        cls.retrieval_agent = RetrievalAgent(cls.embedder, cls.store, confidence_threshold=0.35)
        cls.response_agent = ResponseGenerationAgent()
        cls.memory_agent = ConversationMemoryAgent()
        cls.voice_module = VoiceModule()

        cls.orchestrator = MultiAgentOrchestrator(
            retrieval_agent=cls.retrieval_agent,
            query_understanding_agent=cls.query_agent,
            response_generation_agent=cls.response_agent,
            clarification_agent=cls.clarification_agent,
            memory_agent=cls.memory_agent
        )

    def test_01_factual_query_domain_1_machine_learning(self):
        res = self.orchestrator.run("What is Supervised Learning and what are its applications?")
        self.assertEqual(res.status, "success")
        self.assertGreater(res.response.confidence_score, 0.40)
        self.assertTrue(res.response.has_sufficient_evidence)
        self.assertTrue(any("ml_overview.txt" in s or "ml_algorithms.csv" in s for s in res.response.sources))
        self.assertIn("Supervised", res.response.answer)

    def test_02_factual_query_domain_2_computer_networks(self):
        res = self.orchestrator.run("What is the function of the Transport Layer in the OSI model?")
        self.assertEqual(res.status, "success")
        self.assertGreater(res.response.confidence_score, 0.40)
        self.assertTrue(any("computer_networks.txt" in s for s in res.response.sources))
        self.assertIn("Transport", res.response.answer)

    def test_03_factual_query_domain_3_cybersecurity(self):
        res = self.orchestrator.run("What are the three pillars of the CIA Triad in cybersecurity?")
        self.assertEqual(res.status, "success")
        self.assertGreater(res.response.confidence_score, 0.40)
        self.assertTrue(any("cybersecurity_fundamentals.txt" in s for s in res.response.sources))
        self.assertIn("Confidentiality", res.response.answer)

    def test_04_procedural_query_handshake(self):
        res = self.orchestrator.run("Explain the step by step process of the TCP three-way handshake")
        self.assertEqual(res.status, "success")
        self.assertEqual(res.response.query_analysis.query_type, QueryType.PROCEDURAL)
        self.assertTrue(res.response.has_sufficient_evidence)
        self.assertIn("Step", res.response.answer)
        self.assertIn("SYN", res.response.answer)

    def test_05_comparative_query_encryption(self):
        res = self.orchestrator.run("Compare AES symmetric encryption with RSA asymmetric encryption")
        self.assertEqual(res.status, "success")
        self.assertEqual(res.response.query_analysis.query_type, QueryType.COMPARATIVE)
        self.assertTrue(any("cryptography_and_zero_trust.csv" in s or "cybersecurity_fundamentals.txt" in s for s in res.response.sources))

    def test_06_ambiguous_query_triggers_clarification(self):
        res = self.orchestrator.run("Explain tree")
        self.assertEqual(res.status, "clarification_requested")
        self.assertIsNotNone(res.clarification_request)
        self.assertTrue(len(res.clarification_request.suggested_options) >= 2)

        resolved_res = self.orchestrator.resolve_clarification(
            clarification_id=res.clarification_request.clarification_id,
            user_response="Decision Trees (Machine Learning / Classification)"
        )
        self.assertEqual(resolved_res.status, "clarified_success")
        self.assertIn("Decision Trees", resolved_res.response.refined_query)
        self.assertTrue(resolved_res.response.has_sufficient_evidence)

    def test_07_multi_turn_anaphora_resolution(self):
        session_id = "test_memory_session_turn"
        t1 = self.orchestrator.run("What is Support Vector Machines (SVM)?", session_id=session_id)
        self.assertEqual(t1.status, "success")

        t2 = self.orchestrator.run("What are its key characteristics and how does it use kernels?", session_id=session_id)
        self.assertEqual(t2.status, "success")
        self.assertTrue(t2.response.transparency is not None)

    def test_08_cross_domain_context_switching_isolation(self):
        session_id = "test_domain_switch_session"
        t1 = self.orchestrator.run("What is TCP?", session_id=session_id)
        self.assertIn("TCP", t1.response.answer)

        t2 = self.orchestrator.run("What is Zero Trust Security Architecture?", session_id=session_id)
        self.assertEqual(t2.status, "success")
        self.assertIn("Zero Trust", t2.response.answer)
        self.assertTrue(any("cybersecurity_fundamentals.txt" in s for s in t2.response.sources))

    def test_09_voice_sanitization_integrity(self):
        raw_text = "### Heading\nHere is `code` and **bold** text [1]. Link: [IETF](https://ietf.org)"
        cleaned = self.voice_module.clean_for_speech(raw_text)
        self.assertNotIn("###", cleaned)
        self.assertNotIn("`code`", cleaned)
        self.assertNotIn("**", cleaned)
        self.assertNotIn("[1]", cleaned)
        self.assertNotIn("https://", cleaned)
        self.assertIn("Heading", cleaned)
        self.assertIn("bold", cleaned)

    def test_10_response_transparency_payload_verification(self):
        res = self.orchestrator.run("What is the CAP theorem in distributed systems?")
        self.assertEqual(res.status, "success")
        t_payload = res.response.transparency
        self.assertIsNotNone(t_payload)
        self.assertGreater(len(t_payload.ranked_chunks), 0)
        self.assertGreater(t_payload.score_breakdown.combined_score, 0.0)
        expected_combined = round(
            0.70 * t_payload.score_breakdown.top_chunk_score + 0.30 * t_payload.score_breakdown.avg_top_k_score,
            4
        )
        self.assertAlmostEqual(t_payload.score_breakdown.combined_score, expected_combined, places=3)

if __name__ == "__main__":
    unittest.main()
