"""Unit tests for Knowledge Gap Detection Engine (M4.1)."""
import os
import sys
import unittest
import tempfile
from pathlib import Path
from datetime import datetime, timezone

# Add milestone_4 root to sys.path
m4_root = str(Path(__file__).resolve().parent.parent)
if m4_root not in sys.path:
    sys.path.insert(0, m4_root)

from app.analytics.models import (
    QueryLogEntry,
    ResolutionStatus,
    GapSeverity
)
from app.analytics.storage import AnalyticsStorage
from app.analytics.gap_detector import KnowledgeGapDetector

class TestKnowledgeGapDetection(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "gap_test.db")
        self.storage = AnalyticsStorage(db_path=self.db_path)
        self.detector = KnowledgeGapDetector(storage=self.storage, confidence_threshold=0.45)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_no_gaps_when_queries_resolved(self):
        e = QueryLogEntry(
            query_id="q_good",
            timestamp=datetime.now(timezone.utc).isoformat(),
            query_text="What is Supervised Learning?",
            effective_query="What is Supervised Learning?",
            domain="machine_learning",
            query_type="factual",
            route_target="retrieval",
            top_score=0.9,
            avg_score=0.85,
            confidence_score=0.885,
            confidence_level="HIGH",
            has_sufficient_evidence=True,
            requires_clarification=False,
            status=ResolutionStatus.RESOLVED
        )
        self.storage.log_query(e)
        gaps = self.detector.detect_gaps()
        self.assertEqual(len(gaps), 0)

    def test_detect_unanswered_topic_cluster_gap(self):
        q_texts = [
            "What is Quantum Key Distribution (QKD)?",
            "Explain Quantum Key Distribution protocols",
            "How does QKD enhance quantum cryptography?"
        ]
        for i, q in enumerate(q_texts):
            entry = QueryLogEntry(
                query_id=f"gap_q_{i}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                query_text=q,
                effective_query=q,
                domain="cybersecurity",
                query_type="factual",
                route_target="retrieval",
                top_score=0.15,
                avg_score=0.10,
                confidence_score=0.135,
                confidence_level="LOW",
                has_sufficient_evidence=False,
                requires_clarification=False,
                status=ResolutionStatus.UNANSWERED
            )
            self.storage.log_query(entry)

        gaps = self.detector.detect_gaps()
        self.assertGreaterEqual(len(gaps), 1)

        primary_gap = gaps[0]
        self.assertIn(primary_gap.severity, [GapSeverity.CRITICAL, GapSeverity.HIGH])
        self.assertEqual(primary_gap.domain, "cybersecurity")
        self.assertGreaterEqual(primary_gap.frequency, 2)
        self.assertGreaterEqual(primary_gap.unanswered_count, 2)
        self.assertTrue(len(primary_gap.suggested_actions) > 0)
        self.assertTrue(any("Quantum" in s or "Key" in s or "Distribution" in s for s in primary_gap.sample_queries))

    def test_frequent_query_themes(self):
        topics = ["How does TCP work?", "TCP vs UDP", "TCP handshake", "Security policies", "Security monitoring"]
        for idx, t in enumerate(topics):
            entry = QueryLogEntry(
                query_id=f"theme_{idx}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                query_text=t,
                effective_query=t,
                domain="computer_networks" if "tcp" in t.lower() else "cybersecurity",
                query_type="factual",
                route_target="retrieval",
                top_score=0.8,
                avg_score=0.7,
                confidence_score=0.77,
                confidence_level="HIGH",
                has_sufficient_evidence=True,
                requires_clarification=False,
                status=ResolutionStatus.RESOLVED
            )
            self.storage.log_query(entry)

        themes = self.detector.get_frequent_query_themes(top_n=3)
        self.assertTrue(len(themes) > 0)
        keywords = [t["keyword"] for t in themes]
        self.assertTrue("tcp" in keywords or "security" in keywords)

if __name__ == "__main__":
    unittest.main()
