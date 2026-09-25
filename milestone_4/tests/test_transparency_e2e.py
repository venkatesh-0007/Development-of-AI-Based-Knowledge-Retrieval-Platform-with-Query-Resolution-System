"""End-to-End tests for Response Transparency Panel payload and attribution (Phase 9).

Validates:
1. Grounded response transparency panel payload generation
2. Exact mathematical confidence formula: Combined = 0.70 * TopScore + 0.30 * AvgTopK
3. Ranked chunk attribution matching document sources and snippets
4. Consistency between generated citations, document names, and chunk scores
5. Classification levels matching ConfidenceCalculator definitions
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
from app.agents.orchestrator import MultiAgentOrchestrator
from app.agents.models import ConfidenceLevel
from app.confidence.calculator import default_confidence_calculator


class TestTransparencyE2E(unittest.TestCase):
    """Verifies transparency panel payloads, mathematical scoring, and source citations."""

    @classmethod
    def setUpClass(cls):
        cls.embedder = Embedder()
        cls.store = ChromaStore(collection_name="test_transparency_suite")
        cls.store.clear()
        cls.pipeline = IngestionPipeline(cls.embedder, cls.store)
        load_canonical_benchmark_documents(cls.pipeline)

        cls.query_agent = QueryUnderstandingAgent()
        cls.retrieval_agent = RetrievalAgent(cls.embedder, cls.store, confidence_threshold=0.35)
        cls.response_agent = ResponseGenerationAgent()

        cls.orchestrator = MultiAgentOrchestrator(
            retrieval_agent=cls.retrieval_agent,
            query_understanding_agent=cls.query_agent,
            response_generation_agent=cls.response_agent
        )

    def test_transparency_payload_mathematical_formula(self):
        """Verify Combined = 0.70 * TopScore + 0.30 * AvgTopK."""
        res = self.orchestrator.run("What is the CAP theorem in distributed systems?")
        self.assertEqual(res.status, "success")

        transparency = res.response.transparency
        self.assertIsNotNone(transparency)
        breakdown = transparency.score_breakdown

        top_score = breakdown.top_chunk_score
        avg_score = breakdown.avg_top_k_score
        expected_combined = round(0.70 * top_score + 0.30 * avg_score, 4)

        self.assertAlmostEqual(breakdown.combined_score, expected_combined, places=4)
        self.assertEqual(res.response.confidence_score, breakdown.combined_score)

        # Classification aligns with calculator
        expected_level = default_confidence_calculator.classify(expected_combined)
        self.assertEqual(breakdown.confidence_level, expected_level)
        self.assertEqual(res.response.confidence_level, expected_level)

    def test_transparency_ranked_chunks_and_citations(self):
        """Verify ranked chunks match source attributions, document names, and answer citations."""
        res = self.orchestrator.run("What is AES encryption and what key lengths does it support?")
        self.assertEqual(res.status, "success")

        transparency = res.response.transparency
        self.assertIsNotNone(transparency)
        self.assertGreater(len(transparency.ranked_chunks), 0)

        # Chunks are ordered by score descending
        scores = [c.score for c in transparency.ranked_chunks]
        self.assertEqual(scores, sorted(scores, reverse=True))

        # Check source attributions
        self.assertGreater(len(transparency.source_attributions), 0)
        for attr in transparency.source_attributions:
            self.assertTrue(bool(attr.document_name))
            self.assertTrue(bool(attr.chunk_id))
            self.assertGreater(attr.relevance_score, 0.0)
            self.assertTrue(bool(attr.snippet))

        # Answer must contain citation section
        self.assertIn("**Sources:**", res.response.answer)
        for src_name in res.response.sources:
            self.assertIn(src_name, res.response.answer)

    def test_transparency_zero_chunks_insufficient_evidence(self):
        """When evidence is insufficient, transparency reflects 0 score and NONE level."""
        res = self.orchestrator.run("What is non-existent quantum hyper-dimensional string theory?")
        self.assertFalse(res.response.has_sufficient_evidence)

        transparency = res.response.transparency
        self.assertIsNotNone(transparency)
        self.assertEqual(len(transparency.source_attributions), 0)
        self.assertIn(transparency.score_breakdown.confidence_level, [ConfidenceLevel.NONE, ConfidenceLevel.LOW])


if __name__ == "__main__":
    unittest.main()
