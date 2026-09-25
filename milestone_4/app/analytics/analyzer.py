"""Analytics Analyzer for Milestone 4.

Provides deep operational metrics, KPI aggregations, domain breakdowns,
query auditing, and export functionality for analytics telemetry.
"""
from typing import List, Dict, Any, Optional
from .storage import AnalyticsStorage
from .models import AnalyticsFilter, AnalyticsSummary, QueryLogEntry
from ..confidence.calculator import ConfidenceCalculator, default_confidence_calculator


class AnalyticsAnalyzer:
    """Computes operational KPIs, domain breakdowns, and query audit analytics."""

    def __init__(
        self,
        storage: AnalyticsStorage,
        confidence_calculator: Optional[ConfidenceCalculator] = None
    ):
        self.storage = storage
        self.confidence_calculator = confidence_calculator or default_confidence_calculator

    def get_summary(self) -> AnalyticsSummary:
        """Fetch pre-aggregated operational telemetry summary."""
        return self.storage.get_summary()

    def get_kpis(self) -> Dict[str, Any]:
        """Return primary KPI cards for dashboard and reporting."""
        summary = self.storage.get_summary()
        answered = summary.total_queries - summary.unanswered_count
        return {
            "total_queries": summary.total_queries,
            "answered_queries": answered,
            "unanswered_queries": summary.unanswered_count,
            "low_confidence_queries": summary.low_confidence_count,
            "clarification_requests": summary.clarification_count,
            "average_confidence": summary.avg_confidence,
            "average_latency_ms": summary.avg_latency_ms,
            "resolution_rate": summary.resolution_rate,
            "active_gaps": summary.active_gaps_count
        }

    def get_domain_performance(self) -> Dict[str, Dict[str, Any]]:
        """Compute per-domain metrics: query count, average confidence, latency, unanswered count."""
        queries = self.storage.get_queries(limit=1000)
        domain_stats: Dict[str, Dict[str, Any]] = {}

        for q in queries:
            d = q.domain or "general"
            if d not in domain_stats:
                domain_stats[d] = {
                    "total_queries": 0,
                    "unanswered_count": 0,
                    "low_confidence_count": 0,
                    "total_confidence": 0.0,
                    "total_latency": 0.0,
                    "top_score_sum": 0.0
                }
            st = domain_stats[d]
            st["total_queries"] += 1
            st["total_confidence"] += q.confidence_score
            st["total_latency"] += q.latency_ms
            st["top_score_sum"] += q.top_score

            if q.status.value == "UNANSWERED":
                st["unanswered_count"] += 1
            elif q.status.value == "LOW_CONFIDENCE":
                st["low_confidence_count"] += 1

        # Finalize averages
        result: Dict[str, Dict[str, Any]] = {}
        for d, st in domain_stats.items():
            tot = st["total_queries"]
            result[d] = {
                "total_queries": tot,
                "unanswered_count": st["unanswered_count"],
                "low_confidence_count": st["low_confidence_count"],
                "average_confidence": round(st["total_confidence"] / tot, 4) if tot else 0.0,
                "average_latency_ms": round(st["total_latency"] / tot, 2) if tot else 0.0,
                "average_top_retrieval_score": round(st["top_score_sum"] / tot, 4) if tot else 0.0,
                "resolution_rate": round(((tot - st["unanswered_count"]) / tot) * 100, 2) if tot else 0.0
            }

        return result

    def audit_queries(
        self,
        filter_criteria: Optional[AnalyticsFilter] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Return structured query audit logs with formatted telemetry."""
        entries = self.storage.get_queries(filter_criteria=filter_criteria, limit=limit)
        return [
            {
                "query_id": e.query_id,
                "timestamp": e.timestamp,
                "query": e.query_text,
                "effective_query": e.effective_query,
                "domain": e.domain,
                "query_type": e.query_type,
                "confidence_score": e.confidence_score,
                "confidence_level": e.confidence_level,
                "resolution_status": e.status.value if hasattr(e.status, "value") else str(e.status),
                "top_score": e.top_score,
                "latency_ms": e.latency_ms,
                "sources_count": len(e.sources),
                "sources": e.sources,
                "input_modality": e.input_modality
            }
            for e in entries
        ]

    def export_csv(self, filter_criteria: Optional[AnalyticsFilter] = None) -> str:
        """Export query telemetry as CSV."""
        return self.storage.export_queries_csv(filter_criteria=filter_criteria)
