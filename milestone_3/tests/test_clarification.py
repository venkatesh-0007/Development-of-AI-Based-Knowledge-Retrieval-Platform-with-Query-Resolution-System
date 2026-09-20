"""Unit tests for Milestone 3.1 Clarification Agent.

Covers:
- Clear queries (no false-positive clarification)
- Ambiguous terminology and polysemy
- Incomplete queries & dangling operators
- Missing context (pronouns without antecedents)
- Related multi-part queries (should NOT require clarification)
- Genuinely ambiguous/incomplete multi-part queries (require clarification)
- Clarification state lifecycle & error handling (invalid ID, repeated resolution, empty response)
- Query refinement behavior
"""
import sys
from pathlib import Path

# Add milestone_3 root to sys.path
milestone_3_root = str(Path(__file__).resolve().parent.parent)
if milestone_3_root not in sys.path:
    sys.path.insert(0, milestone_3_root)

import unittest
from app.agents.clarification import ClarificationAgent
from app.agents.query_understanding import QueryUnderstandingAgent
from app.agents.models import (
    ClarificationType,
    ClarificationStatus,
    QueryType,
    RoutingTarget,
    QueryAnalysisResult
)

class TestClarificationAgent(unittest.TestCase):
    """Comprehensive test suite for M3.1 Clarification Agent."""

    def setUp(self):
        self.agent = ClarificationAgent()
        self.query_understanding = QueryUnderstandingAgent(clarification_agent=self.agent)

    # -------------------------------------------------------------
    # 1. Clear Queries (Must NOT Trigger Clarification)
    # -------------------------------------------------------------
    def test_clear_queries_do_not_trigger_clarification(self):
        """Verify that specific, unambiguous queries proceed without clarification."""
        clear_queries = [
            "What is the Transmission Control Protocol (TCP)?",
            "Compare TCP vs UDP in terms of reliability and speed",
            "How does DNS resolution work step-by-step?",
            "What is supervised machine learning?",
            "Explain gradient descent algorithm for linear regression",
            "How does TCP 3-way handshake work?",
            "Explain Decision Trees in machine learning",
            "What is Port 80 used for in HTTP?",
            "How does Spanning Tree Protocol prevent network loops?",
            "What is K-Means clustering algorithm?",
            "Explain BGP routing protocol",
            "What is the OSI transport layer function?",
            "Explain Support Vector Machines and their kernel trick",
            "What is TCP and what port does it use?"
        ]
        for q in clear_queries:
            req = self.agent.analyze_query(q)
            self.assertIsNone(req, f"Clear query '{q}' should NOT trigger clarification")

    def test_clear_domain_acronyms_do_not_trigger_clarification(self):
        """Single-word well-known domain acronyms should be treated as factual queries, not ambiguous."""
        domain_acronyms = ["tcp", "udp", "svm", "dns", "http", "https", "dhcp", "icmp", "ftp", "ssh", "pca"]
        for acronym in domain_acronyms:
            req = self.agent.analyze_query(acronym)
            self.assertIsNone(req, f"Domain acronym '{acronym}' should NOT trigger clarification")

    def test_pronoun_with_antecedent_does_not_trigger_clarification(self):
        """Queries containing pronouns WITH an explicit antecedent should not trigger clarification."""
        queries_with_antecedents = [
            "Explain TCP and its 3-way handshake",
            "What is SVM and how does it classify data?",
            "Define DNS and explain its hierarchy",
            "What is UDP and what are its advantages over TCP?"
        ]
        for q in queries_with_antecedents:
            req = self.agent.analyze_query(q)
            self.assertIsNone(req, f"Query with antecedent '{q}' should NOT trigger clarification")

    # -------------------------------------------------------------
    # 2. Ambiguous & Polysemous Terminology
    # -------------------------------------------------------------
    def test_detect_unqualified_polysemous_terms(self):
        """Verify unqualified polysemous domain terms trigger MULTIPLE_INTERPRETATIONS."""
        test_catalog = [
            ("explain tree", ClarificationType.MULTIPLE_INTERPRETATIONS),
            ("how does clustering work?", ClarificationType.MULTIPLE_INTERPRETATIONS),
            ("what is the handshake?", ClarificationType.MULTIPLE_INTERPRETATIONS),
            ("tell me about routing", ClarificationType.MULTIPLE_INTERPRETATIONS),
            ("explain the model", ClarificationType.MULTIPLE_INTERPRETATIONS),
            ("what is the layer?", ClarificationType.MULTIPLE_INTERPRETATIONS),
            ("which port should i use?", ClarificationType.MULTIPLE_INTERPRETATIONS),
            ("pipeline overview", ClarificationType.MULTIPLE_INTERPRETATIONS)
        ]
        for query, expected_type in test_catalog:
            req = self.agent.analyze_query(query)
            self.assertIsNotNone(req, f"Query '{query}' should require clarification")
            self.assertEqual(req.clarification_type, expected_type)
            self.assertTrue(len(req.suggested_options) >= 2)
            self.assertTrue(len(req.follow_up_question) > 5)

    def test_detect_generic_vague_terminology(self):
        """Verify generic references lacking concrete entities trigger AMBIGUOUS_TERM."""
        generic_queries = [
            "explain the algorithm",
            "what is the protocol",
            "how does the system run",
            "what is the best method",
            "tell me about the device"
        ]
        for q in generic_queries:
            req = self.agent.analyze_query(q)
            self.assertIsNotNone(req, f"Generic query '{q}' should require clarification")
            self.assertIn(req.clarification_type, [ClarificationType.AMBIGUOUS_TERM, ClarificationType.MULTIPLE_INTERPRETATIONS])

    # -------------------------------------------------------------
    # 3. Incomplete Queries & Dangling Operators
    # -------------------------------------------------------------
    def test_detect_incomplete_requests(self):
        """Verify truncated query formulations and dangling comparisons trigger INCOMPLETE_REQUEST."""
        incomplete_queries = [
            "how to",
            "compare SVM",
            "compare TCP and",
            "what is the",
            "difference between",
            "how to configure with",
            "vs",
            "compare"
        ]
        for q in incomplete_queries:
            req = self.agent.analyze_query(q)
            self.assertIsNotNone(req, f"Incomplete query '{q}' should require clarification")
            self.assertIn(req.clarification_type, [ClarificationType.INCOMPLETE_REQUEST, ClarificationType.INSUFFICIENT_INFORMATION])

    def test_targeted_options_for_incomplete_comparison(self):
        """Verify incomplete comparison generates relevant entity options."""
        req = self.agent.analyze_query("compare SVM")
        self.assertIsNotNone(req)
        self.assertIn("SVM", req.follow_up_question)
        self.assertTrue(any("Logistic Regression" in opt or "Random Forest" in opt for opt in req.suggested_options))

    # -------------------------------------------------------------
    # 4. Missing Context (Ungrounded Pronouns)
    # -------------------------------------------------------------
    def test_detect_missing_context_pronouns(self):
        """Verify queries with vague pronouns and NO antecedent trigger MISSING_CONTEXT."""
        pronoun_queries = [
            "how does it work?",
            "what are its advantages?",
            "tell me about that",
            "how to use it",
            "explain this"
        ]
        for q in pronoun_queries:
            req = self.agent.analyze_query(q)
            self.assertIsNotNone(req, f"Pronoun query '{q}' should trigger missing context clarification")
            self.assertEqual(req.clarification_type, ClarificationType.MISSING_CONTEXT)

    # -------------------------------------------------------------
    # 5. Multi-Part Queries: Related vs Genuinely Ambiguous
    # -------------------------------------------------------------
    def test_related_multi_part_queries_do_not_require_clarification(self):
        """Coherent, related multi-part queries should NOT trigger clarification."""
        related_multi_part = [
            "What is TCP and what port does it use?",
            "What is Support Vector Machine and what are its kernel functions?",
            "Compare TCP vs UDP and explain their key differences",
            "What is DNS and how does it resolve domain names to IP addresses?"
        ]
        for q in related_multi_part:
            req = self.agent.analyze_query(q)
            self.assertIsNone(req, f"Related multi-part query '{q}' should NOT require clarification")

    def test_ambiguous_or_incomplete_multi_part_triggers_clarification(self):
        """Multi-part queries containing an incomplete or ambiguous sub-part MUST trigger clarification."""
        ambiguous_multi_parts = [
            "What is TCP, and also how to, and explain tree",
            "How does DNS work, and what is the algorithm, and compare SVM"
        ]
        for q in ambiguous_multi_parts:
            req = self.agent.analyze_query(q)
            self.assertIsNotNone(req, f"Ambiguous multi-part '{q}' MUST require clarification")
            self.assertEqual(req.clarification_type, ClarificationType.MULTI_PART_UNRESOLVED)
            self.assertTrue(req.is_multi_part)

    def test_disjoint_multi_domain_queries_trigger_clarification(self):
        """Multi-part query across 3+ completely unrelated domains should trigger multi-part clarification."""
        disjoint_query = "How to configure BGP routing, what is the kernel trick in SVM, and what are HTTP status codes?"
        req = self.agent.analyze_query(disjoint_query)
        self.assertIsNotNone(req)
        self.assertEqual(req.clarification_type, ClarificationType.MULTI_PART_UNRESOLVED)
        self.assertTrue(len(req.sub_questions) >= 3)

    # -------------------------------------------------------------
    # 6. Clarification State Management & Lifecycle
    # -------------------------------------------------------------
    def test_clarification_lifecycle_pending_to_resolved(self):
        """Verify proper state transition from PENDING to RESOLVED with metadata recording."""
        req = self.agent.analyze_query("explain tree")
        self.agent.store_pending(req)
        self.assertEqual(req.status, ClarificationStatus.PENDING)

        refined, resolved_req = self.agent.resolve_clarification(
            clarification_id=req.clarification_id,
            user_response="Decision Trees"
        )
        self.assertEqual(resolved_req.status, ClarificationStatus.RESOLVED)
        self.assertIsNotNone(resolved_req.resolved_at)
        self.assertEqual(resolved_req.user_response, "Decision Trees")
        self.assertIn("Decision Trees", refined)

    def test_invalid_clarification_id_raises_error(self):
        """Attempting to resolve a non-existent clarification ID must raise ValueError."""
        with self.assertRaises(ValueError):
            self.agent.resolve_clarification("non-existent-id", "Decision Trees")

        with self.assertRaises(ValueError):
            self.agent.resolve_clarification("", "Decision Trees")

    def test_repeated_resolution_attempt_raises_error(self):
        """Attempting to resolve an already-RESOLVED clarification must raise ValueError."""
        req = self.agent.analyze_query("clustering")
        self.agent.store_pending(req)

        # First resolution succeeds
        self.agent.resolve_clarification(req.clarification_id, "K-Means Clustering")

        # Second resolution attempt fails
        with self.assertRaises(ValueError):
            self.agent.resolve_clarification(req.clarification_id, "Hierarchical Clustering")

    def test_empty_clarification_response_raises_error(self):
        """Attempting to resolve with an empty or whitespace response must raise ValueError."""
        req = self.agent.analyze_query("how does it work?")
        self.agent.store_pending(req)

        for empty_resp in ["", "   ", "\t\n"]:
            with self.assertRaises(ValueError):
                self.agent.resolve_clarification(req.clarification_id, empty_resp)

    def test_cancel_clarification(self):
        """Verify cancellation changes status to CANCELLED and blocks resolution."""
        req = self.agent.analyze_query("compare SVM")
        self.agent.store_pending(req)

        cancelled = self.agent.cancel_clarification(req.clarification_id)
        self.assertEqual(cancelled.status, ClarificationStatus.CANCELLED)

        with self.assertRaises(ValueError):
            self.agent.resolve_clarification(req.clarification_id, "Logistic Regression")

    def test_meaningless_clarification_response_raises_error(self):
        """Attempting to resolve with an uninformative response like 'I don't know' must raise ValueError."""
        req = self.agent.analyze_query("how does it work?")
        self.agent.store_pending(req)

        for vague_resp in ["I don't know", "dont know", "no idea", "not sure", "idk", "none", "N/A"]:
            with self.assertRaises(ValueError):
                self.agent.resolve_clarification(req.clarification_id, vague_resp)

    # -------------------------------------------------------------
    # 7. Intent-Aware Query Refinement Accuracy
    # -------------------------------------------------------------
    def test_query_refinement_how_does_it_work_direct(self):
        """'How does it work?' + 'TCP' -> 'How does TCP work?'"""
        orig = "How does it work?"
        req = self.agent.analyze_query(orig)
        refined = self.agent.refine_query(orig, "TCP", req)
        self.assertEqual(refined, "How does TCP work?")

    def test_query_refinement_how_does_it_work_conversational(self):
        """'How does it work?' + 'I am asking about TCP protocol' -> 'How does the TCP protocol work?'"""
        orig = "How does it work?"
        req = self.agent.analyze_query(orig)
        refined = self.agent.refine_query(orig, "I am asking about TCP protocol", req)
        self.assertEqual(refined, "How does the TCP protocol work?")

    def test_query_refinement_what_is_it(self):
        """'What is it?' + 'TCP' -> 'What is TCP?'"""
        orig = "What is it?"
        req = self.agent.analyze_query(orig)
        refined = self.agent.refine_query(orig, "TCP", req)
        self.assertEqual(refined, "What is TCP?")

    def test_query_refinement_what_are_its_advantages(self):
        """'What are its advantages?' + 'TCP' -> 'What are the advantages of TCP?'"""
        orig = "What are its advantages?"
        req = self.agent.analyze_query(orig)
        refined = self.agent.refine_query(orig, "TCP", req)
        self.assertEqual(refined, "What are the advantages of TCP?")

    def test_query_refinement_explain_it(self):
        """'Explain it.' + 'TCP three-way handshake' -> 'Explain TCP three-way handshake'"""
        orig = "Explain it."
        req = self.agent.analyze_query(orig)
        refined = self.agent.refine_query(orig, "TCP three-way handshake", req)
        self.assertEqual(refined, "Explain TCP three-way handshake")

    def test_query_refinement_compare_it_with_udp(self):
        """'Compare it with UDP.' + 'TCP' -> 'Compare TCP with UDP'"""
        orig = "Compare it with UDP."
        req = self.agent.analyze_query(orig)
        refined = self.agent.refine_query(orig, "TCP", req)
        self.assertEqual(refined, "Compare TCP with UDP")

    def test_query_refinement_polysemous_term(self):
        """Refines polysemous query by replacing ambiguous keyword."""
        orig = "Explain tree"
        req = self.agent.analyze_query(orig)
        refined = self.agent.refine_query(
            original_query=orig,
            user_response="Decision Trees (Machine Learning / Classification)",
            clarification_request=req
        )
        self.assertEqual(refined, "Explain Decision Trees")

    def test_query_refinement_option_sanitization(self):
        """Strips option markers and parenthetical annotations from response."""
        orig = "Explain tree"
        req = self.agent.analyze_query(orig)
        refined = self.agent.refine_query(
            original_query=orig,
            user_response="📌 1. Spanning Tree Protocol (Networking / STP)",
            clarification_request=req
        )
        self.assertEqual(refined, "Explain Spanning Tree Protocol")

if __name__ == "__main__":
    unittest.main()

