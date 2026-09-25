"""Knowledge Gap Detection Engine.

Analyzes query resolution telemetry to detect knowledge base gaps,
unanswered query clusters, low-confidence topic themes, and frequently asked patterns.
"""
import re
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from datetime import datetime, timezone

from .models import (
    QueryLogEntry,
    KnowledgeGapReport,
    ResolutionStatus,
    GapSeverity
)
from .storage import AnalyticsStorage

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "up", "about", "into", "through", "what",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "can", "could", "should", "would", "how", "why", "which",
    "who", "whom", "this", "that", "these", "those", "explain", "describe", "tell",
    "me", "please", "compare", "difference", "between"
}

class KnowledgeGapDetector:
    """Detects knowledge gaps and common query themes from query telemetry."""

    def __init__(self, storage: AnalyticsStorage, confidence_threshold: float = 0.50):
        self.storage = storage
        self.confidence_threshold = confidence_threshold

    def extract_keywords(self, text: str) -> List[str]:
        tokens = re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())
        keywords = [t for t in tokens if t not in STOPWORDS and not t.isdigit()]
        return keywords

    def detect_gaps(self, min_cluster_size: int = 1) -> List[KnowledgeGapReport]:
        """
        Scans all queries for unanswered or low-confidence items,
        clusters them by semantic topic theme, and produces prioritized gap reports.
        """
        all_queries = self.storage.get_queries(limit=1000)
        
        # 1. Identify candidate gap queries
        gap_candidates: List[QueryLogEntry] = []
        for q in all_queries:
            is_unanswered = (q.status == ResolutionStatus.UNANSWERED) or (not q.has_sufficient_evidence)
            is_low_conf = (q.status == ResolutionStatus.LOW_CONFIDENCE) or (q.confidence_score < self.confidence_threshold)
            if (is_unanswered or is_low_conf) and q.status != ResolutionStatus.CLARIFICATION_REQUESTED:
                gap_candidates.append(q)

        if not gap_candidates:
            return []

        # 2. Cluster candidates by semantic keyword overlaps
        clusters: List[Dict[str, Any]] = []

        for q in gap_candidates:
            q_keywords = set(self.extract_keywords(q.query_text))
            if not q_keywords:
                q_keywords = {q.query_text.lower().strip()}

            best_cluster = None
            best_overlap = 0.0

            for cl in clusters:
                # Same domain or general
                if cl["domain"] == q.domain or cl["domain"] == "general" or q.domain == "general":
                    intersection = len(q_keywords.intersection(cl["keywords"]))
                    union = len(q_keywords.union(cl["keywords"]))
                    overlap = intersection / union if union > 0 else 0.0
                    if overlap > 0.25 and overlap > best_overlap:
                        best_overlap = overlap
                        best_cluster = cl

            if best_cluster is not None:
                best_cluster["queries"].append(q)
                best_cluster["keywords"].update(q_keywords)
                if best_cluster["domain"] == "general" and q.domain != "general":
                    best_cluster["domain"] = q.domain
            else:
                clusters.append({
                    "domain": q.domain,
                    "keywords": set(q_keywords),
                    "queries": [q]
                })

        # 3. Transform clusters into KnowledgeGapReport items
        gap_reports: List[KnowledgeGapReport] = []

        for cl in clusters:
            queries: List[QueryLogEntry] = cl["queries"]
            if len(queries) < min_cluster_size:
                continue

            # Determine topic title from top keywords
            freq_words = defaultdict(int)
            for q in queries:
                for kw in self.extract_keywords(q.query_text):
                    freq_words[kw] += 1
            
            top_words = sorted(freq_words.items(), key=lambda x: x[1], reverse=True)[:3]
            topic_title = " ".join(w[0].capitalize() for w in top_words) if top_words else "Unknown Concept"

            unanswered_count = sum(1 for q in queries if q.status == ResolutionStatus.UNANSWERED or not q.has_sufficient_evidence)
            avg_conf = sum(q.confidence_score for q in queries) / len(queries)
            freq = len(queries)

            # Compute severity
            severity_score = freq * (1.0 - avg_conf) + 1.5 * unanswered_count
            if severity_score >= 3.5 or unanswered_count >= 2:
                severity = GapSeverity.CRITICAL
            elif severity_score >= 2.0 or freq >= 2:
                severity = GapSeverity.HIGH
            elif severity_score >= 1.0:
                severity = GapSeverity.MEDIUM
            else:
                severity = GapSeverity.LOW

            # Sample queries
            samples = list(dict.fromkeys(q.query_text for q in queries))[:4]
            timestamps = [q.timestamp for q in queries if q.timestamp]
            first_seen = min(timestamps) if timestamps else datetime.now(timezone.utc).isoformat()
            last_seen = max(timestamps) if timestamps else datetime.now(timezone.utc).isoformat()

            # Suggested remedial actions
            actions = [
                f"Ingest reference documentation, FAQs, or technical guides covering '{topic_title}'.",
                f"Validate retrieval thresholds and verify whether related documents in domain '{cl['domain']}' exist in the knowledge base.",
                f"Review {len(samples)} user queries receiving low confidence or unanswered status."
            ]

            gap_id = "gap_" + hashlib.md5(f"{topic_title}_{cl['domain']}".encode("utf-8")).hexdigest()[:8]
            
            report = KnowledgeGapReport(
                gap_id=gap_id,
                topic=topic_title,
                domain=cl["domain"],
                severity=severity,
                frequency=freq,
                average_confidence=round(avg_conf, 3),
                unanswered_count=unanswered_count,
                sample_queries=samples,
                suggested_actions=actions,
                first_detected=first_seen,
                last_detected=last_seen
            )
            gap_reports.append(report)

        gap_reports.sort(key=lambda x: (x.severity == GapSeverity.CRITICAL, x.severity == GapSeverity.HIGH, x.frequency), reverse=True)
        
        # Save to persistent analytics storage
        self.storage.save_gap_reports(gap_reports)
        return gap_reports

    def get_frequent_query_themes(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Identifies frequently asked query topics across all resolved and active queries."""
        all_queries = self.storage.get_queries(limit=1000)
        word_counts = defaultdict(int)
        domain_counts = defaultdict(int)

        for q in all_queries:
            domain_counts[q.domain] += 1
            keywords = self.extract_keywords(q.query_text)
            for kw in keywords:
                word_counts[kw] += 1

        top_themes = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [
            {"keyword": kw, "frequency": count, "relative_share": round(count / len(all_queries), 2) if all_queries else 0.0}
            for kw, count in top_themes
        ]
