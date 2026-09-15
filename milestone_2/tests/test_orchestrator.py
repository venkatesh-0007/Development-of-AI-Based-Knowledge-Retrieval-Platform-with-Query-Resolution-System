import unittest
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.models import (
    QueryType,
    RoutingTarget,
    ConfidenceLevel,
    QueryAnalysisResult,
    RetrievalResult,
    RetrievalChunk,
    AgentResponse
)
from app.agents.orchestrator import MultiAgentOrchestrator

class TestMultiAgentOrchestrator(unittest.TestCase):
    def setUp(self):
        self.mock_retrieval_agent = MagicMock()
        self.mock_query_agent = MagicMock()
        self.mock_response_agent = MagicMock()
        self.mock_clarification_agent = MagicMock()

        self.orchestrator = MultiAgentOrchestrator(
            retrieval_agent=self.mock_retrieval_agent,
            query_understanding_agent=self.mock_query_agent,
            response_generation_agent=self.mock_response_agent,
            clarification_agent=self.mock_clarification_agent
        )

    def test_end_to_end_successful_flow(self):
        # 1. Query Understanding output
        qa = QueryAnalysisResult(
            query="Compare TCP vs UDP",
            cleaned_query="Compare TCP vs UDP",
            query_type=QueryType.COMPARATIVE,
            confidence=0.95,
            route_to=RoutingTarget.RETRIEVAL,
            reasoning="Comparative query",
            entities=["TCP", "UDP"],
            requires_clarification=False,
            reformulated_query="Compare TCP vs UDP"
        )
        self.mock_query_agent.analyze.return_value = qa

        # 2. Retrieval Agent output
        retrieval = RetrievalResult(
            query=qa.query,
            query_type=QueryType.COMPARATIVE,
            chunks=[
                RetrievalChunk(
                    chunk_id="chunk-1",
                    content="TCP is reliable, UDP is connectionless.",
                    metadata={"source": "networks.txt"},
                    similarity_score=0.92,
                    rank=1
                )
            ],
            top_score=0.92,
            total_retrieved=1,
            filtered_count=0,
            has_sufficient_evidence=True,
            confidence_threshold=0.45
        )
        self.mock_retrieval_agent.retrieve.return_value = retrieval

        # 3. Response Generation Agent output
        expected_resp = AgentResponse(
            answer="**Comparative Analysis:** TCP is reliable, UDP is connectionless.",
            confidence_score=0.92,
            confidence_level=ConfidenceLevel.HIGH,
            sources=[],
            query_analysis=qa,
            has_sufficient_evidence=True,
            requires_clarification=False
        )
        self.mock_response_agent.generate.return_value = expected_resp

        # Run orchestrator
        res = self.orchestrator.run("Compare TCP vs UDP")

        self.assertEqual(res.status, "success")
        self.assertGreater(res.execution_time_ms, 0)
        self.assertEqual(res.response.answer, expected_resp.answer)
        self.mock_retrieval_agent.retrieve.assert_called_once()
        self.mock_response_agent.generate.assert_called_once()
        self.mock_clarification_agent.handle.assert_not_called()

    def test_ambiguous_query_bypasses_retrieval(self):
        qa = QueryAnalysisResult(
            query="how to fix it?",
            cleaned_query="how to fix it?",
            query_type=QueryType.AMBIGUOUS,
            confidence=0.88,
            route_to=RoutingTarget.CLARIFICATION,
            reasoning="Vague pronouns without subject",
            entities=[],
            requires_clarification=True,
            reformulated_query="how to fix it?"
        )
        self.mock_query_agent.analyze.return_value = qa

        clarification_resp = AgentResponse(
            answer="Your query is ambiguous. Please specify the technology.",
            confidence_score=0.88,
            confidence_level=ConfidenceLevel.NONE,
            sources=[],
            query_analysis=qa,
            has_sufficient_evidence=False,
            requires_clarification=True
        )
        self.mock_clarification_agent.handle.return_value = clarification_resp

        res = self.orchestrator.run("how to fix it?")

        self.assertEqual(res.status, "clarification_requested")
        self.assertTrue(res.response.requires_clarification)
        self.mock_retrieval_agent.retrieve.assert_not_called()
        self.mock_response_agent.generate.assert_not_called()
        self.mock_clarification_agent.handle.assert_called_once()

    def test_orchestrator_error_handling(self):
        self.mock_query_agent.analyze.side_effect = RuntimeError("Database connection timed out")

        res = self.orchestrator.run("What is SGD?")

        self.assertEqual(res.status, "error")
        self.assertIn("Database connection timed out", res.error)
        self.assertIn("An unexpected error occurred", res.response.answer)

if __name__ == "__main__":
    unittest.main()
