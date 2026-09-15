import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.models import QueryType, RoutingTarget, QueryAnalysisResult
from app.agents.query_understanding import QueryUnderstandingAgent

class TestQueryUnderstandingAgent(unittest.TestCase):
    def setUp(self):
        self.agent = QueryUnderstandingAgent()

    # --- 1. MACHINE LEARNING DOMAIN TESTS (Part 10 Requirements) ---
    def test_ml_factual_supervised_learning(self):
        res = self.agent.analyze("What is supervised learning?")
        self.assertEqual(res.query_type, QueryType.FACTUAL)
        self.assertEqual(res.route_to, RoutingTarget.RETRIEVAL)
        self.assertFalse(res.requires_clarification)
        self.assertGreaterEqual(res.classification_confidence, 0.7)

    def test_ml_factual_regression(self):
        res = self.agent.analyze("What is regression?")
        self.assertEqual(res.query_type, QueryType.FACTUAL)
        self.assertEqual(res.route_to, RoutingTarget.RETRIEVAL)
        self.assertFalse(res.requires_clarification)

    def test_ml_procedural_gradient_descent(self):
        res = self.agent.analyze("How does gradient descent work?")
        self.assertEqual(res.query_type, QueryType.PROCEDURAL)
        self.assertEqual(res.route_to, RoutingTarget.RETRIEVAL)
        self.assertFalse(res.requires_clarification)

    def test_ml_procedural_train_model(self):
        res = self.agent.analyze("How can a model be trained?")
        self.assertEqual(res.query_type, QueryType.PROCEDURAL)
        self.assertEqual(res.route_to, RoutingTarget.RETRIEVAL)
        self.assertFalse(res.requires_clarification)

    def test_ml_comparative_classification_regression(self):
        res = self.agent.analyze("Compare classification and regression.")
        self.assertEqual(res.query_type, QueryType.COMPARATIVE)
        self.assertEqual(res.route_to, RoutingTarget.RETRIEVAL)
        self.assertFalse(res.requires_clarification)

    def test_ml_comparative_supervised_unsupervised(self):
        res = self.agent.analyze("What is the difference between supervised and unsupervised learning?")
        self.assertEqual(res.query_type, QueryType.COMPARATIVE)
        self.assertEqual(res.route_to, RoutingTarget.RETRIEVAL)
        self.assertFalse(res.requires_clarification)

    def test_ml_ambiguous_tell_me_about_it(self):
        res = self.agent.analyze("Tell me about it.")
        self.assertEqual(res.query_type, QueryType.AMBIGUOUS)
        self.assertEqual(res.route_to, RoutingTarget.CLARIFICATION)
        self.assertTrue(res.requires_clarification)

    def test_ml_ambiguous_explain_this(self):
        res = self.agent.analyze("Explain this.")
        self.assertEqual(res.query_type, QueryType.AMBIGUOUS)
        self.assertEqual(res.route_to, RoutingTarget.CLARIFICATION)
        self.assertTrue(res.requires_clarification)

    # --- 2. COMPUTER NETWORKS DOMAIN TESTS (Part 10 Requirements) ---
    def test_networks_factual_tcp(self):
        res = self.agent.analyze("What is TCP?")
        self.assertEqual(res.query_type, QueryType.FACTUAL)
        self.assertEqual(res.route_to, RoutingTarget.RETRIEVAL)
        self.assertFalse(res.requires_clarification)

    def test_networks_factual_http(self):
        res = self.agent.analyze("What is HTTP?")
        self.assertEqual(res.query_type, QueryType.FACTUAL)
        self.assertEqual(res.route_to, RoutingTarget.RETRIEVAL)
        self.assertFalse(res.requires_clarification)

    def test_networks_procedural_tcp_connection(self):
        res = self.agent.analyze("How does TCP establish a connection?")
        self.assertEqual(res.query_type, QueryType.PROCEDURAL)
        self.assertEqual(res.route_to, RoutingTarget.RETRIEVAL)
        self.assertFalse(res.requires_clarification)

    def test_networks_procedural_configure_network(self):
        res = self.agent.analyze("How do I configure a network connection?")
        self.assertEqual(res.query_type, QueryType.PROCEDURAL)
        self.assertEqual(res.route_to, RoutingTarget.RETRIEVAL)
        self.assertFalse(res.requires_clarification)

    def test_networks_comparative_tcp_udp(self):
        res = self.agent.analyze("Compare TCP and UDP.")
        self.assertEqual(res.query_type, QueryType.COMPARATIVE)
        self.assertEqual(res.route_to, RoutingTarget.RETRIEVAL)
        self.assertFalse(res.requires_clarification)

    def test_networks_comparative_http_https(self):
        res = self.agent.analyze("What is the difference between HTTP and HTTPS?")
        self.assertEqual(res.query_type, QueryType.COMPARATIVE)
        self.assertEqual(res.route_to, RoutingTarget.RETRIEVAL)
        self.assertFalse(res.requires_clarification)

    # --- 3. EDGE CASES & SPECIAL INPUTS ---
    def test_edge_case_empty_query(self):
        res = self.agent.analyze("")
        self.assertEqual(res.query_type, QueryType.AMBIGUOUS)
        self.assertEqual(res.route_to, RoutingTarget.CLARIFICATION)
        self.assertTrue(res.requires_clarification)
        self.assertEqual(res.classification_confidence, 1.0)

    def test_edge_case_whitespace_only(self):
        res = self.agent.analyze("   \t  \n  ")
        self.assertEqual(res.query_type, QueryType.AMBIGUOUS)
        self.assertEqual(res.route_to, RoutingTarget.CLARIFICATION)
        self.assertTrue(res.requires_clarification)

    def test_edge_case_single_domain_acronym(self):
        res = self.agent.analyze("TCP")
        self.assertEqual(res.query_type, QueryType.FACTUAL)
        self.assertEqual(res.route_to, RoutingTarget.RETRIEVAL)
        self.assertFalse(res.requires_clarification)

    def test_edge_case_single_vague_word(self):
        res = self.agent.analyze("explain")
        self.assertEqual(res.query_type, QueryType.AMBIGUOUS)
        self.assertEqual(res.route_to, RoutingTarget.CLARIFICATION)
        self.assertTrue(res.requires_clarification)

    def test_edge_case_unknown_topic_remains_factual(self):
        # Query classification MUST be independent of KB presence
        res = self.agent.analyze("What is quantum computing?")
        self.assertEqual(res.query_type, QueryType.FACTUAL)
        self.assertEqual(res.route_to, RoutingTarget.RETRIEVAL)
        self.assertFalse(res.requires_clarification)

    def test_edge_case_mixed_case(self):
        res = self.agent.analyze("wHaT iS rEgReSsIoN?")
        self.assertEqual(res.query_type, QueryType.FACTUAL)
        self.assertEqual(res.route_to, RoutingTarget.RETRIEVAL)
        self.assertFalse(res.requires_clarification)

    # --- 4. STRUCTURED OUTPUT & ALIASES ---
    def test_structured_output_fields(self):
        res = self.agent.analyze("What is overfitting?")
        self.assertEqual(res.confidence, res.classification_confidence) # Alias verification
        d = res.to_dict()
        self.assertIn("classification_confidence", d)
        self.assertIn("route_to", d)
        self.assertIn("query_type", d)
        self.assertIn("requires_clarification", d)
        self.assertEqual(d["route_to"], "retrieval")

if __name__ == "__main__":
    unittest.main()
