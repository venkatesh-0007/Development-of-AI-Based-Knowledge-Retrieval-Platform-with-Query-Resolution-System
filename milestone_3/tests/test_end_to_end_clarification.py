"""End-to-End integration tests for MultiAgent Clarification Pipeline (M3.1).

Verifies:
- Ambiguous query triggers clarification
- User clarification resolution re-enters the M2 pipeline (Classification -> Retrieval -> Grounded Generation)
- Clear queries bypass clarification directly to grounded answers
- Invalid IDs and empty responses return structured error responses
- Multi-part query resolution with focus selection
"""
import sys
from pathlib import Path

# Add milestone_3 root to sys.path
milestone_3_root = str(Path(__file__).resolve().parent.parent)
if milestone_3_root not in sys.path:
    sys.path.insert(0, milestone_3_root)

import unittest
from unittest.mock import MagicMock

from app.agents.clarification import ClarificationAgent
from app.agents.query_understanding import QueryUnderstandingAgent
from app.agents.retrieval import RetrievalAgent
from app.agents.response_generation import ResponseGenerationAgent
from app.agents.orchestrator import MultiAgentOrchestrator
from app.agents.models import (
    QueryType,
    RoutingTarget,
    ConfidenceLevel,
    ClarificationStatus
)

class TestEndToEndClarification(unittest.TestCase):
    """Integration test suite testing the end-to-end multi-agent clarification loop."""

    def setUp(self):
        # Mock Embedder and Vector Store
        self.mock_embedder = MagicMock()
        self.mock_embedder.encode.return_value = [[0.1, 0.2, 0.3]]

        self.mock_vectorstore = MagicMock()
        self.mock_vectorstore.search.return_value = [
            {
                "content": "Decision Trees are non-parametric supervised learning models used for classification and regression tasks.",
                "score": 0.88,
                "metadata": {
                    "document_name": "ml_algorithms.csv",
                    "chunk_id": "chunk-dt-1",
                    "page": 1,
                    "section": "Decision Trees"
                }
            },
            {
                "content": "Decision Trees split dataset features recursively based on Gini impurity or information gain criteria.",
                "score": 0.82,
                "metadata": {
                    "document_name": "ml_overview.txt",
                    "chunk_id": "chunk-dt-2",
                    "page": 2,
                    "section": "Supervised Methods"
                }
            }
        ]

        self.clarification_agent = ClarificationAgent()
        self.query_understanding_agent = QueryUnderstandingAgent(clarification_agent=self.clarification_agent)
        self.retrieval_agent = RetrievalAgent(embedder=self.mock_embedder, vectorstore=self.mock_vectorstore)
        self.response_generation_agent = ResponseGenerationAgent()

        self.orchestrator = MultiAgentOrchestrator(
            retrieval_agent=self.retrieval_agent,
            query_understanding_agent=self.query_understanding_agent,
            response_generation_agent=self.response_generation_agent,
            clarification_agent=self.clarification_agent
        )

    def test_ambiguous_query_triggers_clarification_and_resolves(self):
        """
        Flow:
        1. Ambiguous query 'explain tree' triggers clarification.
        2. Clarification request has status PENDING.
        3. User provides 'Decision Trees (Machine Learning / Classification)'.
        4. Orchestrator resolves clarification, refines query to 'Explain Decision Trees?', and runs M2 pipeline.
        5. Grounded response is generated with HIGH confidence and source citations.
        """
        # Step 1 & 2: Initial ambiguous query
        initial_result = self.orchestrator.run("explain tree")
        self.assertEqual(initial_result.status, "clarification_requested")
        self.assertTrue(initial_result.response.requires_clarification)
        self.assertIsNotNone(initial_result.clarification_request)
        
        clarification_id = initial_result.clarification_request.clarification_id
        self.assertEqual(initial_result.clarification_request.status, ClarificationStatus.PENDING)

        # Step 3 & 4: Resolve clarification
        resolved_result = self.orchestrator.resolve_clarification(
            clarification_id=clarification_id,
            user_response="Decision Trees (Machine Learning / Classification)"
        )

        self.assertEqual(resolved_result.status, "clarified_success")
        self.assertFalse(resolved_result.response.requires_clarification)
        self.assertTrue(resolved_result.response.is_clarified_resolution)
        self.assertIn("Decision Trees", resolved_result.response.refined_query)
        self.assertTrue(len(resolved_result.response.sources) >= 1)
        self.assertIn("Decision Trees", resolved_result.response.answer)
        self.assertEqual(resolved_result.response.confidence_level, ConfidenceLevel.HIGH)

    def test_unambiguous_query_proceeds_directly_to_answer(self):
        """Unambiguous query 'What is Supervised Learning?' should proceed straight to answer without clarification."""
        result = self.orchestrator.run("What is supervised machine learning?")
        self.assertEqual(result.status, "success")
        self.assertFalse(result.response.requires_clarification)
        self.assertIsNone(result.clarification_request)
        self.assertTrue(len(result.response.sources) >= 1)

    def test_missing_context_query_resolves_grounded_answer(self):
        """
        Flow:
        1. 'How does it work?' triggers MISSING_CONTEXT clarification.
        2. User responds 'TCP 3-way handshake'.
        3. Grounded resolution executes for refined query 'How does TCP 3-way handshake work?'.
        """
        init_res = self.orchestrator.run("How does it work?")
        self.assertEqual(init_res.status, "clarification_requested")
        
        clarif_id = init_res.clarification_request.clarification_id
        res_resolved = self.orchestrator.resolve_clarification(
            clarification_id=clarif_id,
            user_response="TCP 3-way handshake"
        )
        self.assertEqual(res_resolved.status, "clarified_success")
        self.assertIn("TCP 3-way handshake", res_resolved.response.refined_query)

    def test_multi_part_query_resolution_with_sub_question_focus(self):
        """
        Flow:
        1. Multi-part query triggers MULTI_PART_UNRESOLVED.
        2. User chooses to focus on sub-question 1.
        3. Refined query focuses on sub-question and resolves answer.
        """
        multi_q = "What is TCP, and also how to, and explain tree"
        init_res = self.orchestrator.run(multi_q)
        self.assertEqual(init_res.status, "clarification_requested")
        self.assertTrue(init_res.clarification_request.is_multi_part)

        clarif_id = init_res.clarification_request.clarification_id
        resolved_res = self.orchestrator.resolve_clarification(
            clarification_id=clarif_id,
            user_response="Focus only on: 'What is TCP'"
        )
        self.assertEqual(resolved_res.status, "clarified_success")
        self.assertEqual(resolved_res.response.refined_query, "What is TCP")

    def test_orchestrator_error_handling_on_invalid_clarification_id(self):
        """Orchestrator should return status='error' with graceful message when given invalid ID."""
        err_res = self.orchestrator.resolve_clarification(
            clarification_id="invalid-id",
            user_response="Decision Trees"
        )
        self.assertEqual(err_res.status, "error")
        self.assertIsNotNone(err_res.error)
        self.assertIn("Could not resolve clarification", err_res.response.answer)

    def test_orchestrator_error_handling_on_empty_response(self):
        """Orchestrator should return status='error' when given empty clarification response."""
        init_res = self.orchestrator.run("explain tree")
        clarif_id = init_res.clarification_request.clarification_id

        err_res = self.orchestrator.resolve_clarification(
            clarification_id=clarif_id,
            user_response=""
        )
        self.assertEqual(err_res.status, "error")
        self.assertIn("Could not resolve clarification", err_res.response.answer)

if __name__ == "__main__":
    unittest.main()
