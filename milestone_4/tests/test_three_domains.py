"""Unit and integration tests for Three Knowledge Domains and Domain Isolation (Phase 7 & Phase 9).

Validates:
1. Machine Learning / AI domain ingestion and retrieval
2. Computer Networks / Cloud domain ingestion and retrieval
3. Cybersecurity domain ingestion and retrieval
4. Strict domain metadata preservation on all chunks
5. Domain isolation: queries explicitly targeted to one domain do not cross-contaminate
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
from app.ingestion.benchmark_dataset import BENCHMARK_DOCUMENTS, load_canonical_benchmark_documents
from app.agents.query_understanding import QueryUnderstandingAgent
from app.agents.retrieval import RetrievalAgent
from app.agents.response_generation import ResponseGenerationAgent
from app.agents.clarification import ClarificationAgent
from app.agents.memory import ConversationMemoryAgent
from app.agents.orchestrator import MultiAgentOrchestrator


class TestThreeDomainsIsolation(unittest.TestCase):
    """Verifies domain isolation and deterministic metadata across all 3 domains."""

    @classmethod
    def setUpClass(cls):
        cls.embedder = Embedder()
        cls.store = ChromaStore(collection_name="test_three_domains_store")
        cls.store.clear()
        cls.pipeline = IngestionPipeline(cls.embedder, cls.store)

        # Ingest benchmark documents deterministically
        load_canonical_benchmark_documents(cls.pipeline)

        cls.clarification_agent = ClarificationAgent()
        cls.query_agent = QueryUnderstandingAgent(clarification_agent=cls.clarification_agent)
        cls.retrieval_agent = RetrievalAgent(cls.embedder, cls.store, confidence_threshold=0.35)
        cls.response_agent = ResponseGenerationAgent()
        cls.memory_agent = ConversationMemoryAgent()

        cls.orchestrator = MultiAgentOrchestrator(
            retrieval_agent=cls.retrieval_agent,
            query_understanding_agent=cls.query_agent,
            response_generation_agent=cls.response_agent,
            clarification_agent=cls.clarification_agent,
            memory_agent=cls.memory_agent
        )

    def test_benchmark_documents_definition(self):
        """Ensure benchmark document registry defines at least 3 distinct domains."""
        domains = set(d["domain"] for d in BENCHMARK_DOCUMENTS)
        self.assertIn("machine_learning", domains)
        self.assertIn("computer_networks", domains)
        self.assertIn("cybersecurity", domains)
        self.assertEqual(len(BENCHMARK_DOCUMENTS), 9)

    def test_domain_1_machine_learning_retrieval_isolation(self):
        """Domain 1: Machine Learning queries return ML documents and preserve domain metadata."""
        query = "What is Supervised Learning and how do decision trees classify data?"
        res = self.orchestrator.run(query, active_domain="machine_learning")
        self.assertEqual(res.status, "success")
        self.assertTrue(res.response.has_sufficient_evidence)

        # Verify all retrieved chunks have domain == 'machine_learning'
        retrieval = res.retrieval_result
        self.assertIsNotNone(retrieval)
        self.assertGreater(len(retrieval.chunks), 0)
        for c in retrieval.chunks:
            chunk_domain = c.metadata.get("domain")
            self.assertEqual(chunk_domain, "machine_learning", f"Chunk {c.chunk_id} leaked from {chunk_domain}")

        # Verify sources belong to ML
        for src in res.response.sources:
            self.assertTrue(
                any(src.endswith(ext) for ext in ["ml_overview.txt", "ml_algorithms.csv", "deep_learning_and_transformers.txt"]),
                f"Unexpected source {src} in machine_learning query"
            )

    def test_domain_2_computer_networks_retrieval_isolation(self):
        """Domain 2: Computer Networks queries return network documents and preserve domain metadata."""
        query = "Explain TCP vs UDP and how the 3-way handshake operates."
        res = self.orchestrator.run(query, active_domain="computer_networks")
        self.assertEqual(res.status, "success")
        self.assertTrue(res.response.has_sufficient_evidence)

        retrieval = res.retrieval_result
        self.assertIsNotNone(retrieval)
        self.assertGreater(len(retrieval.chunks), 0)
        for c in retrieval.chunks:
            chunk_domain = c.metadata.get("domain")
            self.assertEqual(chunk_domain, "computer_networks", f"Chunk {c.chunk_id} leaked from {chunk_domain}")

        for src in res.response.sources:
            self.assertTrue(
                any(src.endswith(ext) for ext in ["computer_networks.txt", "network_protocols.csv", "cloud_distributed_systems.txt"]),
                f"Unexpected source {src} in computer_networks query"
            )

    def test_domain_3_cybersecurity_retrieval_isolation(self):
        """Domain 3: Cybersecurity queries return security documents and preserve domain metadata."""
        query = "What is Zero Trust Security Architecture and AES encryption?"
        res = self.orchestrator.run(query, active_domain="cybersecurity")
        self.assertEqual(res.status, "success")
        self.assertTrue(res.response.has_sufficient_evidence)

        retrieval = res.retrieval_result
        self.assertIsNotNone(retrieval)
        self.assertGreater(len(retrieval.chunks), 0)
        for c in retrieval.chunks:
            chunk_domain = c.metadata.get("domain")
            self.assertEqual(chunk_domain, "cybersecurity", f"Chunk {c.chunk_id} leaked from {chunk_domain}")

        for src in res.response.sources:
            self.assertTrue(
                any(src.endswith(ext) for ext in ["cybersecurity_fundamentals.txt", "cryptography_and_zero_trust.csv", "incident_response_and_threats.txt"]),
                f"Unexpected source {src} in cybersecurity query"
            )


if __name__ == "__main__":
    unittest.main()
