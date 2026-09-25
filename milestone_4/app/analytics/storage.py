"""Persistent storage for Query Analytics and Knowledge Gap Reports using SQLite."""
import os
import json
import sqlite3
import csv
import io
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from .models import (
    QueryLogEntry,
    KnowledgeGapReport,
    AnalyticsFilter,
    AnalyticsSummary,
    ResolutionStatus,
    GapSeverity
)

class AnalyticsStorage:
    """Manages independent, persistent storage for query resolution telemetry and knowledge gap reports."""

    def __init__(self, db_path: str = "data/analytics/analytics.db"):
        self.db_path = db_path
        Path(os.path.dirname(db_path)).mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_logs (
                    query_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    effective_query TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    query_type TEXT NOT NULL,
                    route_target TEXT NOT NULL,
                    top_score REAL NOT NULL,
                    avg_score REAL NOT NULL,
                    confidence_score REAL NOT NULL,
                    confidence_level TEXT NOT NULL,
                    has_sufficient_evidence INTEGER NOT NULL,
                    requires_clarification INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    retrieved_chunk_ids TEXT NOT NULL,
                    sources TEXT NOT NULL,
                    clarification_id TEXT,
                    response_text TEXT,
                    latency_ms REAL NOT NULL,
                    session_id TEXT,
                    input_modality TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_logs_timestamp ON query_logs(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_logs_domain ON query_logs(domain)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_logs_status ON query_logs(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_logs_confidence ON query_logs(confidence_level)
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_gaps (
                    gap_id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    frequency INTEGER NOT NULL,
                    average_confidence REAL NOT NULL,
                    unanswered_count INTEGER NOT NULL,
                    sample_queries TEXT NOT NULL,
                    suggested_actions TEXT NOT NULL,
                    first_detected TEXT NOT NULL,
                    last_detected TEXT NOT NULL
                )
            """)
            conn.commit()

    def log_query(self, entry: QueryLogEntry):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO query_logs (
                    query_id, timestamp, query_text, effective_query, domain,
                    query_type, route_target, top_score, avg_score, confidence_score,
                    confidence_level, has_sufficient_evidence, requires_clarification,
                    status, retrieved_chunk_ids, sources, clarification_id,
                    response_text, latency_ms, session_id, input_modality
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.query_id,
                entry.timestamp,
                entry.query_text,
                entry.effective_query,
                entry.domain,
                entry.query_type,
                entry.route_target,
                entry.top_score,
                entry.avg_score,
                entry.confidence_score,
                entry.confidence_level,
                1 if entry.has_sufficient_evidence else 0,
                1 if entry.requires_clarification else 0,
                entry.status.value if isinstance(entry.status, ResolutionStatus) else str(entry.status),
                json.dumps(entry.retrieved_chunk_ids),
                json.dumps(entry.sources),
                entry.clarification_id,
                entry.response_text,
                entry.latency_ms,
                entry.session_id,
                entry.input_modality
            ))
            conn.commit()

    def _row_to_query_entry(self, row: sqlite3.Row) -> QueryLogEntry:
        status_val = row["status"]
        try:
            status_enum = ResolutionStatus(status_val)
        except ValueError:
            status_enum = ResolutionStatus.RESOLVED

        return QueryLogEntry(
            query_id=row["query_id"],
            timestamp=row["timestamp"],
            query_text=row["query_text"],
            effective_query=row["effective_query"],
            domain=row["domain"],
            query_type=row["query_type"],
            route_target=row["route_target"],
            top_score=row["top_score"],
            avg_score=row["avg_score"],
            confidence_score=row["confidence_score"],
            confidence_level=row["confidence_level"],
            has_sufficient_evidence=bool(row["has_sufficient_evidence"]),
            requires_clarification=bool(row["requires_clarification"]),
            status=status_enum,
            retrieved_chunk_ids=json.loads(row["retrieved_chunk_ids"]),
            sources=json.loads(row["sources"]),
            clarification_id=row["clarification_id"],
            response_text=row["response_text"] or "",
            latency_ms=row["latency_ms"],
            session_id=row["session_id"],
            input_modality=row["input_modality"]
        )

    def get_queries(
        self,
        filter_criteria: Optional[AnalyticsFilter] = None,
        limit: int = 200,
        offset: int = 0
    ) -> List[QueryLogEntry]:
        query = "SELECT * FROM query_logs WHERE 1=1"
        params: List[Any] = []

        if filter_criteria:
            if filter_criteria.domain and filter_criteria.domain != "ALL":
                query += " AND domain = ?"
                params.append(filter_criteria.domain)
            if filter_criteria.query_type and filter_criteria.query_type != "ALL":
                query += " AND query_type = ?"
                params.append(filter_criteria.query_type)
            if filter_criteria.confidence_level and filter_criteria.confidence_level != "ALL":
                query += " AND confidence_level = ?"
                params.append(filter_criteria.confidence_level)
            if filter_criteria.status and filter_criteria.status != "ALL":
                query += " AND status = ?"
                params.append(filter_criteria.status)
            if filter_criteria.search_query:
                query += " AND (query_text LIKE ? OR response_text LIKE ?)"
                pattern = f"%{filter_criteria.search_query}%"
                params.extend([pattern, pattern])
            if filter_criteria.date_from:
                query += " AND timestamp >= ?"
                params.append(filter_criteria.date_from)
            if filter_criteria.date_to:
                query += " AND timestamp <= ?"
                params.append(filter_criteria.date_to)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [self._row_to_query_entry(r) for r in cursor.fetchall()]

    def get_query_by_id(self, query_id: str) -> Optional[QueryLogEntry]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM query_logs WHERE query_id = ?", (query_id,))
            row = cursor.fetchone()
            return self._row_to_query_entry(row) if row else None

    def get_summary(self) -> AnalyticsSummary:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM query_logs")
            total = cursor.fetchone()[0]

            if total == 0:
                return AnalyticsSummary(
                    total_queries=0,
                    resolution_rate=0.0,
                    unanswered_count=0,
                    low_confidence_count=0,
                    clarification_count=0,
                    avg_confidence=0.0,
                    avg_latency_ms=0.0
                )

            cursor.execute("SELECT AVG(confidence_score), AVG(latency_ms) FROM query_logs")
            avg_conf, avg_lat = cursor.fetchone()

            cursor.execute("SELECT status, COUNT(*) FROM query_logs GROUP BY status")
            status_map = {r[0]: r[1] for r in cursor.fetchall()}

            cursor.execute("SELECT domain, COUNT(*) FROM query_logs GROUP BY domain")
            domain_map = {r[0]: r[1] for r in cursor.fetchall()}

            cursor.execute("SELECT query_type, COUNT(*) FROM query_logs GROUP BY query_type")
            type_map = {r[0]: r[1] for r in cursor.fetchall()}

            cursor.execute("SELECT confidence_level, COUNT(*) FROM query_logs GROUP BY confidence_level")
            conf_map = {r[0]: r[1] for r in cursor.fetchall()}

            cursor.execute("SELECT substr(timestamp, 1, 10), COUNT(*) FROM query_logs GROUP BY substr(timestamp, 1, 10)")
            daily_map = {r[0]: r[1] for r in cursor.fetchall()}

            cursor.execute("SELECT COUNT(*) FROM knowledge_gaps")
            active_gaps = cursor.fetchone()[0]

            resolved = status_map.get("RESOLVED", 0) + status_map.get("CLARIFICATION_RESOLVED", 0)
            res_rate = round((resolved / total) * 100, 2)
            unanswered = status_map.get("UNANSWERED", 0)
            low_conf = status_map.get("LOW_CONFIDENCE", 0)
            clarifications = status_map.get("CLARIFICATION_REQUESTED", 0) + status_map.get("CLARIFICATION_RESOLVED", 0)

            return AnalyticsSummary(
                total_queries=total,
                resolution_rate=res_rate,
                unanswered_count=unanswered,
                low_confidence_count=low_conf,
                clarification_count=clarifications,
                avg_confidence=round(float(avg_conf or 0.0), 3),
                avg_latency_ms=round(float(avg_lat or 0.0), 2),
                domain_breakdown=domain_map,
                query_type_breakdown=type_map,
                status_breakdown=status_map,
                confidence_distribution=conf_map,
                daily_query_volume=daily_map,
                active_gaps_count=active_gaps
            )

    def save_gap_reports(self, gaps: List[KnowledgeGapReport]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM knowledge_gaps")
            for gap in gaps:
                cursor.execute("""
                    INSERT INTO knowledge_gaps (
                        gap_id, topic, domain, severity, frequency,
                        average_confidence, unanswered_count, sample_queries,
                        suggested_actions, first_detected, last_detected
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    gap.gap_id,
                    gap.topic,
                    gap.domain,
                    gap.severity.value if isinstance(gap.severity, GapSeverity) else str(gap.severity),
                    gap.frequency,
                    gap.average_confidence,
                    gap.unanswered_count,
                    json.dumps(gap.sample_queries),
                    json.dumps(gap.suggested_actions),
                    gap.first_detected,
                    gap.last_detected
                ))
            conn.commit()

    def get_gap_reports(self) -> List[KnowledgeGapReport]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM knowledge_gaps ORDER BY frequency DESC, average_confidence ASC")
            gaps = []
            for r in cursor.fetchall():
                sev_val = r["severity"]
                try:
                    sev_enum = GapSeverity(sev_val)
                except ValueError:
                    sev_enum = GapSeverity.MEDIUM

                gaps.append(KnowledgeGapReport(
                    gap_id=r["gap_id"],
                    topic=r["topic"],
                    domain=r["domain"],
                    severity=sev_enum,
                    frequency=r["frequency"],
                    average_confidence=r["average_confidence"],
                    unanswered_count=r["unanswered_count"],
                    sample_queries=json.loads(r["sample_queries"]),
                    suggested_actions=json.loads(r["suggested_actions"]),
                    first_detected=r["first_detected"],
                    last_detected=r["last_detected"]
                ))
            return gaps

    def export_csv(self) -> str:
        queries = self.get_queries(limit=5000)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "query_id", "timestamp", "query_text", "domain", "query_type",
            "confidence_score", "confidence_level", "status", "latency_ms", "sources"
        ])
        for q in queries:
            writer.writerow([
                q.query_id, q.timestamp, q.query_text, q.domain, q.query_type,
                q.confidence_score, q.confidence_level, q.status.value, q.latency_ms,
                ";".join(q.sources)
            ])
        return output.getvalue()

    def export_json(self) -> str:
        queries = self.get_queries(limit=5000)
        gaps = self.get_gap_reports()
        data = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "total_queries": len(queries),
            "summary": self.get_summary().__dict__,
            "queries": [q.to_dict() for q in queries],
            "gaps": [g.to_dict() for g in gaps]
        }
        return json.dumps(data, indent=2)

    def clear(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM query_logs")
            cursor.execute("DELETE FROM knowledge_gaps")
            conn.commit()
