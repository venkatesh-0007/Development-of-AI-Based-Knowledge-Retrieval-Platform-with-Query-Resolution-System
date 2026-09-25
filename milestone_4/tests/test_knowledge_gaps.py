"""Comprehensive unit tests for Knowledge Gap Detection (Phase 5 & Phase 9).

Validates all 6 gap trigger conditions:
1. No relevant documents retrieved.
2. Retrieved document count is zero.
3. Top similarity score is below the configured threshold.
4. Confidence is LOW or NONE.
5. The same topic repeatedly receives low-confidence results.
6. The same unanswered query/theme appears repeatedly.

Validates tracked attributes:
- normalized query, topic/theme, occurrences (frequency), average confidence,
  average retrieval score, domains involved, first seen, last seen,
  unanswered count, low-confidence count.
"""
import os
import sys
import unittest
import tempfile
from pathlib import Path
from datetime import datetime, timezone

m4_root = str(Path(__file__).resolve().parent.parent)
if m4_root not in sys.path:
    sys.path.insert(0, m4_root)

from app.analytics.models import (
    QueryLogEntry,
    ResolutionStatus,
    GapSeverity,
    KnowledgeGapReport
)
from app.analytics.storage import AnalyticsStorage
from app.analytics.gap_detector import KnowledgeGapDetector
from app.confidence.calculator import ConfidenceCalculator, ConfidenceLevel


class TestKnowledgeGaps(unittest.TestCase):
    """Verifies all six conditions and field tracking of Knowledge Gap Detection."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "gap_test.db")
        self.storage = AnalyticsStorage(db_path=self.db_path)
        self.calc = ConfidenceCalculator(high=0.70, medium=0.45, low=0.20)
        self.detector = KnowledgeGapDetector(
            storage=self.storage,
            confidence_threshold=0.45,
            confidence_calculator=self.calc
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_condition_1_and_2_zero_retrieved_chunks(self):
        """Condition 1 & 2: Retrieved document count is 0."""
        entry = QueryLogEntry(
            query_id="q_zero_chunks",
            timestamp=datetime.now(timezone.utc).isoformat(),
            query_text="What is quantum annealing hardware?",
            effective_query="What is quantum annealing hardware?",
            domain="machine_learning",
            query_type="factual",
            route_target="retrieval",
            top_score=0.0,
            avg_score=0.0,
            confidence_score=0.0,
            confidence_level="NONE",
            has_sufficient_evidence=False,
            requires_clarification=False,
            status=ResolutionStatus.UNANSWERED,
            retrieved_chunk_ids=[]
        )
        self.storage.log_query(entry)
        self.assertTrue(self.detector.is_candidate_gap_query(entry))
        gaps = self.detector.detect_gaps(min_cluster_size=1)
        self.assertEqual(len(gaps), 1)
        gap = gaps[0]
        self.assertEqual(gap.unanswered_count, 1)
        self.assertEqual(gap.frequency, 1)
        self.assertEqual(gap.average_retrieval_score, 0.0)
        self.assertIn("machine_learning", gap.domains_involved)

    def test_condition_3_top_similarity_score_below_threshold(self):
        """Condition 3: Top similarity score is below configured threshold (e.g. 0.45)."""
        entry = QueryLogEntry(
            query_id="q_low_score",
            timestamp=datetime.now(timezone.utc).isoformat(),
            query_text="How to configure Kubernetes CNI with eBPF?",
            effective_query="How to configure Kubernetes CNI with eBPF?",
            domain="computer_networks",
            query_type="procedural",
            route_target="retrieval",
            top_score=0.28,
            avg_score=0.22,
            confidence_score=0.26,
            confidence_level="LOW",
            has_sufficient_evidence=False,
            requires_clarification=False,
            status=ResolutionStatus.LOW_CONFIDENCE,
            retrieved_chunk_ids=["chk_irrelevant_1"]
        )
        self.storage.log_query(entry)
        self.assertTrue(self.detector.is_candidate_gap_query(entry))
        gaps = self.detector.detect_gaps()
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0].low_confidence_count, 1)
        self.assertAlmostEqual(gaps[0].average_retrieval_score, 0.28, places=2)

    def test_condition_4_confidence_low_or_none(self):
        """Condition 4: Confidence is LOW or NONE."""
        for level in ["LOW", "NONE"]:
            entry = QueryLogEntry(
                query_id=f"q_{level}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                query_text=f"Explain {level} concept in depth",
                effective_query=f"Explain {level} concept in depth",
                domain="cybersecurity",
                query_type="factual",
                route_target="retrieval",
                top_score=0.15,
                avg_score=0.10,
                confidence_score=0.13,
                confidence_level=level,
                has_sufficient_evidence=False,
                requires_clarification=False,
                status=ResolutionStatus.UNANSWERED
            )
            self.assertTrue(self.detector.is_candidate_gap_query(entry))

    def test_condition_5_repeated_low_confidence_topic(self):
        """Condition 5: The same topic repeatedly receives low-confidence results."""
        queries = [
            "Explain post-quantum lattice cryptography",
            "What is lattice-based post-quantum cryptography?",
            "How does lattice cryptography defend against quantum attacks?"
        ]
        for idx, q in enumerate(queries):
            e = QueryLogEntry(
                query_id=f"rep_low_{idx}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                query_text=q,
                effective_query=q,
                domain="cybersecurity",
                query_type="factual",
                route_target="retrieval",
                top_score=0.35,
                avg_score=0.25,
                confidence_score=0.32,
                confidence_level="LOW",
                has_sufficient_evidence=False,
                requires_clarification=False,
                status=ResolutionStatus.LOW_CONFIDENCE,
                retrieved_chunk_ids=["chk_general_crypto"]
            )
            self.storage.log_query(e)

        gaps = self.detector.detect_gaps(min_cluster_size=2)
        self.assertEqual(len(gaps), 1)
        gap = gaps[0]
        self.assertEqual(gap.frequency, 3)
        self.assertEqual(gap.low_confidence_count, 3)
        self.assertGreaterEqual(gap.average_confidence, 0.30)
        self.assertIn(gap.severity, [GapSeverity.CRITICAL, GapSeverity.HIGH])

    def test_condition_6_repeated_unanswered_theme(self):
        """Condition 6: The same unanswered query/theme appears repeatedly."""
        queries = [
            "What is Neuromorphic computing spiking network?",
            "Explain neuromorphic computing architectures",
            "How do spiking neurons compute in neuromorphic hardware?"
        ]
        for idx, q in enumerate(queries):
            e = QueryLogEntry(
                query_id=f"rep_un_{idx}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                query_text=q,
                effective_query=q,
                domain="machine_learning",
                query_type="factual",
                route_target="retrieval",
                top_score=0.0,
                avg_score=0.0,
                confidence_score=0.0,
                confidence_level="NONE",
                has_sufficient_evidence=False,
                requires_clarification=False,
                status=ResolutionStatus.UNANSWERED,
                retrieved_chunk_ids=[]
            )
            self.storage.log_query(e)

        gaps = self.detector.detect_gaps(min_cluster_size=2)
        self.assertEqual(len(gaps), 1)
        gap = gaps[0]
        self.assertEqual(gap.frequency, 3)
        self.assertEqual(gap.unanswered_count, 3)
        self.assertEqual(gap.severity, GapSeverity.CRITICAL)
        self.assertTrue(len(gap.suggested_actions) >= 2)
        self.assertTrue(bool(gap.normalized_query))

    def test_gap_persistence_across_reloads(self):
        """Analytics and gap reports survive storage re-instantiation (persistence test)."""
        e = QueryLogEntry(
            query_id="persist_q",
            timestamp=datetime.now(timezone.utc).isoformat(),
            query_text="What is zero trust network architecture?",
            effective_query="What is zero trust network architecture?",
            domain="cybersecurity",
            query_type="factual",
            route_target="retrieval",
            top_score=0.1,
            avg_score=0.05,
            confidence_score=0.08,
            confidence_level="NONE",
            has_sufficient_evidence=False,
            requires_clarification=False,
            status=ResolutionStatus.UNANSWERED
        )
        self.storage.log_query(e)
        self.detector.detect_gaps()

        # Re-open storage from the same SQLite file
        reloaded_storage = AnalyticsStorage(db_path=self.db_path)
        persisted_gaps = reloaded_storage.get_gap_reports()
        self.assertEqual(len(persisted_gaps), 1)
        self.assertEqual(persisted_gaps[0].unanswered_count, 1)
        self.assertIn("cybersecurity", persisted_gaps[0].domains_involved)


if __name__ == "__main__":
    unittest.main()
