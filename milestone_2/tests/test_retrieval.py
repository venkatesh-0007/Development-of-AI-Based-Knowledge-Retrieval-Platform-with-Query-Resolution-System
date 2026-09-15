import unittest
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.models import QueryType, QueryAnalysisResult, RoutingTarget
from app.agents.retrieval import RetrievalAgent

class TestRetrievalAgent(unittest.TestCase):
    def setUp(self):
        self.mock_embedder = MagicMock()
        self.mock_embedder.encode.return_value = [[0.1] * 384]
        self.mock_vectorstore = MagicMock()
        self.agent = RetrievalAgent(self.mock_embedder, self.mock_vectorstore, confidence_threshold=0.45)

    def test_retrieve_with_sufficient_results(self):
        self.mock_vectorstore.search.return_value = [
            {"content": "Overfitting occurs when...", "metadata": {"source": "ml.txt", "file_hash": "abc", "chunk_index": 0}, "score": 0.88},
            {"content": "Regularization prevents overfitting...", "metadata": {"source": "ml.txt", "file_hash": "abc", "chunk_index": 1}, "score": 0.72}
        ]
        
        qa = QueryAnalysisResult(
            query="What is overfitting?",
            cleaned_query="What is overfitting?",
            query_type=QueryType.FACTUAL,
            confidence=0.9,
            route_to=RoutingTarget.RETRIEVAL,
            reasoning="Factual question",
            entities=["overfitting"],
            requires_clarification=False,
            reformulated_query="What is overfitting"
        )

        result = self.agent.retrieve(qa, top_k=2)
        
        self.assertEqual(len(result.chunks), 2)
        self.assertTrue(result.has_sufficient_evidence)
        self.assertAlmostEqual(result.top_score, 0.88)
        self.assertEqual(result.chunks[0].rank, 1)
        self.assertEqual(result.chunks[0].metadata["source"], "ml.txt")

    def test_retrieve_filters_low_confidence_chunks(self):
        # 1 high score chunk, 1 below threshold chunk
        self.mock_vectorstore.search.return_value = [
            {"content": "TCP is reliable...", "metadata": {"source": "net.txt", "file_hash": "def", "chunk_index": 0}, "score": 0.82},
            {"content": "Random unrelated text...", "metadata": {"source": "unrelated.txt", "file_hash": "xyz", "chunk_index": 0}, "score": 0.30}
        ]

        result = self.agent.retrieve("Tell me about TCP", threshold=0.50)
        
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.filtered_count, 1)
        self.assertEqual(result.chunks[0].similarity_score, 0.82)

    def test_retrieve_empty_when_no_chunks_meet_threshold(self):
        self.mock_vectorstore.search.return_value = [
            {"content": "Unrelated chunk...", "metadata": {"source": "doc.txt"}, "score": 0.25}
        ]

        result = self.agent.retrieve("What is quantum computing?", threshold=0.60)
        
        self.assertEqual(len(result.chunks), 0)
        self.assertFalse(result.has_sufficient_evidence)
        self.assertEqual(result.top_score, 0.0)

    def test_retrieve_empty_query_string(self):
        result = self.agent.retrieve("")
        self.assertEqual(len(result.chunks), 0)
        self.assertFalse(result.has_sufficient_evidence)

if __name__ == "__main__":
    unittest.main()
