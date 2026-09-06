import unittest
from unittest.mock import MagicMock
from app.agents.orchestrator import Orchestrator

class TestOrchestrator(unittest.TestCase):
    def test_orchestrator_empty_query(self):
        retrieval_mock = MagicMock()
        orchestrator = Orchestrator(retrieval_pipeline=retrieval_mock)
        
        res = orchestrator.run("")
        self.assertIn("Please provide more specific details", res["answer"])
        self.assertEqual(res["sources"], [])

    def test_orchestrator_query_flow(self):
        retrieval_mock = MagicMock()
        retrieval_mock.retrieve.return_value = [
            {"content": "Machine learning is a subset of AI.", "metadata": {"source": "ml.txt"}, "score": 0.89}
        ]
        orchestrator = Orchestrator(retrieval_pipeline=retrieval_mock)
        
        res = orchestrator.run("What is machine learning?", top_k=1)
        self.assertEqual(len(res["sources"]), 1)
        self.assertIn("Machine learning is a subset of AI.", res["answer"])
        self.assertEqual(len(orchestrator.memory_agent.get_history()), 1)

if __name__ == "__main__":
    unittest.main()
