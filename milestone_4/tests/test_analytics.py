"""Unit and integration tests for Query Analytics Storage and Telemetry Tracker (M4.1)."""
import os
import sys
import unittest
import tempfile
import json
from pathlib import Path
from datetime import datetime, timezone

# Add milestone_4 root to sys.path
m4_root = str(Path(__file__).resolve().parent.parent)
if m4_root not in sys.path:
    sys.path.insert(0, m4_root)

from app.analytics.models import (
    QueryLogEntry,
    ResolutionStatus,
    AnalyticsFilter,
    AnalyticsSummary,
    GapSeverity
)
from app.analytics.storage import AnalyticsStorage
from app.analytics.tracker import AnalyticsTracker

class TestAnalyticsModule(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_analytics.db")
        self.storage = AnalyticsStorage(db_path=self.db_path)
        self.tracker = AnalyticsTracker(storage=self.storage, low_confidence_threshold=0.45)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_log_query_and_retrieval(self):
        entry = QueryLogEntry(
            query_id="q1",
            timestamp=datetime.now(timezone.utc).isoformat(),
            query_text="What is TCP?",
            effective_query="What is TCP?",
            domain="computer_networks",
            query_type="factual",
            route_target="retrieval",
            top_score=0.92,
            avg_score=0.88,
            confidence_score=0.908,
            confidence_level="HIGH",
            has_sufficient_evidence=True,
            requires_clarification=False,
            status=ResolutionStatus.RESOLVED,
            retrieved_chunk_ids=["chk_1", "chk_2"],
            sources=["computer_networks.txt"],
            response_text="TCP is a reliable transport layer protocol.",
            latency_ms=12.4,
            session_id="sess_1"
        )
        self.storage.log_query(entry)

        retrieved = self.storage.get_query_by_id("q1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.query_id, "q1")
        self.assertEqual(retrieved.query_text, "What is TCP?")
        self.assertEqual(retrieved.domain, "computer_networks")
        self.assertEqual(retrieved.status, ResolutionStatus.RESOLVED)
        self.assertEqual(len(retrieved.retrieved_chunk_ids), 2)
        self.assertEqual(retrieved.sources, ["computer_networks.txt"])

    def test_analytics_filtering(self):
        domains = ["machine_learning", "computer_networks", "cybersecurity"]
        for idx, d in enumerate(domains):
            e = QueryLogEntry(
                query_id=f"q_{idx}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                query_text=f"Sample query in {d}",
                effective_query=f"Sample query in {d}",
                domain=d,
                query_type="factual" if idx != 1 else "procedural",
                route_target="retrieval",
                top_score=0.75,
                avg_score=0.70,
                confidence_score=0.735,
                confidence_level="HIGH",
                has_sufficient_evidence=True,
                requires_clarification=False,
                status=ResolutionStatus.RESOLVED if idx != 2 else ResolutionStatus.LOW_CONFIDENCE,
                retrieved_chunk_ids=["c1"],
                sources=["doc.txt"]
            )
            self.storage.log_query(e)

        flt_ml = AnalyticsFilter(domain="machine_learning")
        res_ml = self.storage.get_queries(filter_criteria=flt_ml)
        self.assertEqual(len(res_ml), 1)
        self.assertEqual(res_ml[0].domain, "machine_learning")

        flt_low = AnalyticsFilter(status="LOW_CONFIDENCE")
        res_low = self.storage.get_queries(filter_criteria=flt_low)
        self.assertEqual(len(res_low), 1)
        self.assertEqual(res_low[0].query_id, "q_2")

        flt_proc = AnalyticsFilter(query_type="procedural")
        res_proc = self.storage.get_queries(filter_criteria=flt_proc)
        self.assertEqual(len(res_proc), 1)
        self.assertEqual(res_proc[0].query_id, "q_1")

    def test_summary_and_export(self):
        for i in range(5):
            e = QueryLogEntry(
                query_id=f"item_{i}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                query_text=f"Query {i}",
                effective_query=f"Query {i}",
                domain="cybersecurity",
                query_type="factual",
                route_target="retrieval",
                top_score=0.8,
                avg_score=0.7,
                confidence_score=0.77,
                confidence_level="HIGH",
                has_sufficient_evidence=True,
                requires_clarification=False,
                status=ResolutionStatus.RESOLVED,
                latency_ms=10.0
            )
            self.storage.log_query(e)

        summary = self.storage.get_summary()
        self.assertEqual(summary.total_queries, 5)
        self.assertEqual(summary.resolution_rate, 100.0)
        self.assertEqual(summary.avg_latency_ms, 10.0)

        csv_str = self.storage.export_csv()
        self.assertIn("query_id,timestamp,query_text", csv_str)
        self.assertIn("item_0", csv_str)

        json_str = self.storage.export_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["total_queries"], 5)
        self.assertEqual(len(parsed["queries"]), 5)

if __name__ == "__main__":
    unittest.main()
