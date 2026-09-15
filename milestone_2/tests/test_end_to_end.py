"""End-to-End integration tests for Milestone 2 Multi-Agent Query Resolution System."""
import unittest
import sys
from pathlib import Path

# Ensure milestone_2 root is in sys.path
milestone_2_root = str(Path(__file__).resolve().parent.parent)
if milestone_2_root not in sys.path:
    sys.path.insert(0, milestone_2_root)

from app.embeddings.embedder import Embedder
from app.vectorstore.chroma_store import ChromaStore
from app.ingestion.pipeline import IngestionPipeline
from app.agents.models import QueryType, RoutingTarget, ConfidenceLevel
from app.agents.query_understanding import QueryUnderstandingAgent
from app.agents.retrieval import RetrievalAgent
from app.agents.response_generation import ResponseGenerationAgent
from app.agents.clarification import ClarificationAgent
from app.agents.orchestrator import MultiAgentOrchestrator

class TestEndToEndPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up in-memory/test Chroma store and index sample domain documents."""
        cls.embedder = Embedder()
        # Use isolated test collection
        cls.store = ChromaStore(path="data/chroma", collection_name="test_e2e_collection")
        cls.ingestion = IngestionPipeline(cls.embedder, cls.store)
        
        # Load sample docs
        sample_dir = Path(milestone_2_root) / "data" / "sample_docs"
        if sample_dir.exists():
            for sample_file in sample_dir.glob("*.*"):
                if sample_file.suffix.lower() in [".txt", ".csv"]:
                    cls.ingestion.ingest(sample_file.read_bytes(), sample_file.name, chunk_size=800, overlap=100)

        cls.query_agent = QueryUnderstandingAgent()
        cls.retrieval_agent = RetrievalAgent(cls.embedder, cls.store, confidence_threshold=0.45)
        cls.response_agent = ResponseGenerationAgent()
        cls.clarification_agent = ClarificationAgent()
        
        cls.orchestrator = MultiAgentOrchestrator(
            retrieval_agent=cls.retrieval_agent,
            query_understanding_agent=cls.query_agent,
            response_generation_agent=cls.response_agent,
            clarification_agent=cls.clarification_agent
        )

    # --- 1. FACTUAL END-TO-END TESTS ---
    def test_e2e_factual_supervised_learning(self):
        res = self.orchestrator.run("What is supervised learning?")
        self.assertEqual(res.status, "success")
        self.assertEqual(res.response.query_analysis.query_type, QueryType.FACTUAL)
        self.assertTrue(res.response.has_sufficient_evidence)
        self.assertIn("supervised", res.response.answer.lower())
        self.assertGreaterEqual(len(res.response.sources), 1)

    def test_e2e_factual_tcp(self):
        res = self.orchestrator.run("What is TCP?")
        self.assertEqual(res.status, "success")
        self.assertEqual(res.response.query_analysis.query_type, QueryType.FACTUAL)
        self.assertTrue(res.response.has_sufficient_evidence)
        self.assertIn("tcp", res.response.answer.lower())

    # --- 2. PROCEDURAL END-TO-END TESTS ---
    def test_e2e_procedural_gradient_descent(self):
        res = self.orchestrator.run("How does gradient descent work?")
        self.assertEqual(res.status, "success")
        self.assertEqual(res.response.query_analysis.query_type, QueryType.PROCEDURAL)
        self.assertTrue(res.response.has_sufficient_evidence)
        self.assertIn("Procedure & Implementation Steps", res.response.answer)

    def test_e2e_procedural_dns_resolution(self):
        res = self.orchestrator.run("How does the DNS resolution process work?")
        self.assertEqual(res.status, "success")
        self.assertEqual(res.response.query_analysis.query_type, QueryType.PROCEDURAL)
        self.assertTrue(res.response.has_sufficient_evidence)

    # --- 3. COMPARATIVE END-TO-END TESTS ---
    def test_e2e_comparative_classification_regression(self):
        res = self.orchestrator.run("Compare classification and regression.")
        self.assertEqual(res.status, "success")
        self.assertEqual(res.response.query_analysis.query_type, QueryType.COMPARATIVE)
        self.assertTrue(res.response.has_sufficient_evidence)

    def test_e2e_comparative_tcp_udp(self):
        res = self.orchestrator.run("Compare TCP and UDP.")
        self.assertEqual(res.status, "success")
        self.assertEqual(res.response.query_analysis.query_type, QueryType.COMPARATIVE)
        self.assertTrue(res.response.has_sufficient_evidence)
        self.assertIn("Comparative Analysis", res.response.answer)

    # --- 4. AMBIGUOUS END-TO-END TESTS ---
    def test_e2e_ambiguous_tell_me_about_it(self):
        res = self.orchestrator.run("Tell me about it.")
        self.assertEqual(res.status, "clarification_requested")
        self.assertTrue(res.response.requires_clarification)
        self.assertIsNone(res.retrieval_result) # Bypasses retrieval

    def test_e2e_ambiguous_explain_this(self):
        res = self.orchestrator.run("Explain this.")
        self.assertEqual(res.status, "clarification_requested")
        self.assertTrue(res.response.requires_clarification)
        self.assertIsNone(res.retrieval_result)

    # --- 5. UNAVAILABLE INFORMATION (NO HALLUCINATION) ---
    def test_e2e_unavailable_quantum_computing(self):
        # Query classification MUST be factual, but retrieval must find no evidence
        res = self.orchestrator.run("What is quantum computing?")
        self.assertEqual(res.status, "success")
        self.assertEqual(res.response.query_analysis.query_type, QueryType.FACTUAL)
        self.assertFalse(res.response.has_sufficient_evidence)
        self.assertEqual(res.response.confidence_level, ConfidenceLevel.NONE)
        self.assertIn("No sufficiently relevant information was found", res.response.answer)

if __name__ == "__main__":
    unittest.main()
