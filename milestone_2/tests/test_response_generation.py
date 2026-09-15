import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.models import (
    QueryType,
    QueryAnalysisResult,
    RoutingTarget,
    RetrievalResult,
    RetrievalChunk,
    ConfidenceLevel
)
from app.agents.response_generation import ResponseGenerationAgent

class TestResponseGenerationAgent(unittest.TestCase):
    def setUp(self):
        self.agent = ResponseGenerationAgent()

    def _create_sample_qa(self, qtype=QueryType.FACTUAL, query="What is overfitting?"):
        return QueryAnalysisResult(
            query=query,
            cleaned_query=query,
            query_type=qtype,
            confidence=0.9,
            route_to=RoutingTarget.RETRIEVAL,
            reasoning="Sample classification",
            entities=["overfitting"],
            requires_clarification=False,
            reformulated_query=query
        )

    def test_generate_factual_response(self):
        qa = self._create_sample_qa(QueryType.FACTUAL, "What is overfitting?")
        chunks = [
            RetrievalChunk(
                chunk_id="chunk-1",
                content="Overfitting occurs when a model learns the training data too well, failing to generalize to unseen test data.",
                metadata={"source": "ml_overview.txt"},
                similarity_score=0.88,
                rank=1
            )
        ]
        retrieval = RetrievalResult(
            query=qa.query,
            query_type=QueryType.FACTUAL,
            chunks=chunks,
            top_score=0.88,
            total_retrieved=1,
            filtered_count=0,
            has_sufficient_evidence=True,
            confidence_threshold=0.45
        )

        response = self.agent.generate(qa, retrieval)
        
        self.assertTrue(response.has_sufficient_evidence)
        self.assertEqual(response.confidence_level, ConfidenceLevel.HIGH)
        self.assertIn("Overfitting occurs when", response.answer)
        self.assertEqual(len(response.sources), 1)
        self.assertEqual(response.sources[0].document_name, "ml_overview.txt")

    def test_generate_procedural_response(self):
        qa = self._create_sample_qa(QueryType.PROCEDURAL, "How does DNS resolution work?")
        chunks = [
            RetrievalChunk(
                chunk_id="chunk-2",
                content="1. Client checks local browser cache.\n2. Query forwarded to recursive resolver.\n3. Root server returns TLD IP.",
                metadata={"source": "computer_networks.txt"},
                similarity_score=0.84,
                rank=1
            )
        ]
        retrieval = RetrievalResult(
            query=qa.query,
            query_type=QueryType.PROCEDURAL,
            chunks=chunks,
            top_score=0.84,
            total_retrieved=1,
            filtered_count=0,
            has_sufficient_evidence=True,
            confidence_threshold=0.45
        )

        response = self.agent.generate(qa, retrieval)
        
        self.assertIn("Procedure & Implementation Steps:", response.answer)
        self.assertIn("Client checks local browser cache", response.answer)

    def test_generate_comparative_response(self):
        qa = self._create_sample_qa(QueryType.COMPARATIVE, "Compare TCP vs UDP")
        chunks = [
            RetrievalChunk(
                chunk_id="chunk-3",
                content="TCP vs UDP Comparison:\nTCP is connection-oriented and reliable.\nUDP is connectionless and fast.",
                metadata={"source": "computer_networks.txt"},
                similarity_score=0.91,
                rank=1
            )
        ]
        retrieval = RetrievalResult(
            query=qa.query,
            query_type=QueryType.COMPARATIVE,
            chunks=chunks,
            top_score=0.91,
            total_retrieved=1,
            filtered_count=0,
            has_sufficient_evidence=True,
            confidence_threshold=0.45
        )

        response = self.agent.generate(qa, retrieval)
        
        self.assertIn("Comparative Analysis:", response.answer)
        self.assertIn("TCP is connection-oriented", response.answer)

    def test_generate_no_evidence_fallback(self):
        qa = self._create_sample_qa(QueryType.FACTUAL, "What is quantum gravity?")
        retrieval = RetrievalResult(
            query=qa.query,
            query_type=QueryType.FACTUAL,
            chunks=[],
            top_score=0.0,
            total_retrieved=0,
            filtered_count=0,
            has_sufficient_evidence=False,
            confidence_threshold=0.45
        )

        response = self.agent.generate(qa, retrieval)
        
        self.assertFalse(response.has_sufficient_evidence)
        self.assertEqual(response.confidence_level, ConfidenceLevel.NONE)
        self.assertIn("No sufficiently relevant information was found in the knowledge base", response.answer)
        self.assertEqual(len(response.sources), 0)

if __name__ == "__main__":
    unittest.main()
