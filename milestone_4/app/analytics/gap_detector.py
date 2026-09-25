"""Knowledge Gap Detection Engine for Milestone 4 (M4.1).

Implementation Methodology:
- Uses deterministic lexical and thematic token Jaccard clustering over query keywords
  (avoids ungrounded claims of semantic embedding clustering when using keyword/Jaccard similarity).
- Identifies knowledge base gaps across 6 explicit trigger conditions:
  1. No relevant documents retrieved.
  2. Retrieved document count is zero.
  3. Top similarity score is below the configured threshold.
  4. Confidence level is LOW or NONE.
  5. The same topic repeatedly receives low-confidence results.
  6. The same unanswered query or theme appears repeatedly.
- Tracks: normalized query, topic theme, frequency, average confidence, average retrieval score,
  domains involved, first seen, last seen, unanswered count, and low-confidence count.
"""
import re
import hashlib
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict
from datetime import datetime, timezone

from .models import (
    QueryLogEntry,
    KnowledgeGapReport,
    ResolutionStatus,
    GapSeverity
)
from .storage import AnalyticsStorage
from ..confidence.calculator import ConfidenceCalculator, default_confidence_calculator

STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "then", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "up", "about", "into", "through", "what",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "can", "could", "should", "would", "how", "why", "which",
    "who", "whom", "this", "that", "these", "those", "explain", "describe", "tell",
    "me", "please", "compare", "difference", "between"
}


class KnowledgeGapDetector:
    """Detects knowledge gaps and common query themes from query telemetry using lexical/thematic clustering."""

    def __init__(
        self,
        storage: AnalyticsStorage,
        confidence_threshold: float = 0.45,
        confidence_calculator: Optional[ConfidenceCalculator] = None
    ):
        self.storage = storage
        self.confidence_calculator = confidence_calculator or default_confidence_calculator
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self.confidence_calculator.thresholds.medium
        )

    @staticmethod
    def _stem(word: str) -> str:
        for suffix in ["ing", "ion", "ions", "ed", "es", "s"]:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                word = word[:-len(suffix)]
                break
        if word.endswith("e") and len(word) >= 4:
            word = word[:-1]
        return word

    def extract_keywords(self, text: str) -> List[str]:
        """Extract sanitized alphanumeric keyword tokens excluding stopwords."""
        # Replace hyphens with spaces to preserve compound terms as separate tokens
        clean_text = text.replace("-", " ").replace("_", " ")
        tokens = re.findall(r"\b[a-zA-Z0-9]{2,}\b", clean_text.lower())
        return [t for t in tokens if t not in STOPWORDS and not t.isdigit()]

    def extract_stemmed_keywords(self, text: str) -> Set[str]:
        """Extract stemmed keyword set for robust thematic clustering."""
        raw_kws = self.extract_keywords(text)
        return {self._stem(w) for w in raw_kws}

    def normalize_query(self, query_text: str) -> str:
        """Normalize query by stripping punctuation, extra whitespace, and lowercasing."""
        cleaned = re.sub(r"[^\w\s]", " ", query_text.lower())
        return " ".join(cleaned.split())

    def is_candidate_gap_query(self, q: QueryLogEntry) -> bool:
        """
        Evaluates whether an individual query exhibits knowledge gap symptoms:
        - Condition 1 & 2: No relevant chunks retrieved / chunk count is 0
        - Condition 3: Top similarity score is below threshold
        - Condition 4: Confidence is LOW or NONE, or status is UNANSWERED / LOW_CONFIDENCE
        """
        # Clarification requests are ambiguity inquiries, not immediate knowledge base gaps
        if q.status == ResolutionStatus.CLARIFICATION_REQUESTED:
            return False

        # If fully resolved with evidence and high confidence, exclude from gaps
        if q.status in (ResolutionStatus.RESOLVED, ResolutionStatus.CLARIFICATION_RESOLVED):
            if q.confidence_score >= self.confidence_threshold and q.has_sufficient_evidence:
                return False

        # Condition 1 & 2: Zero retrieved chunks
        zero_chunks = len(q.retrieved_chunk_ids) == 0

        # Condition 3: Top score below threshold
        score_below_threshold = q.top_score < self.confidence_threshold

        # Condition 4: Confidence LOW/NONE or status UNANSWERED/LOW_CONFIDENCE
        conf_low_or_none = (
            q.confidence_level in ("LOW", "NONE")
            or q.confidence_score < self.confidence_threshold
            or not q.has_sufficient_evidence
            or q.status in (ResolutionStatus.UNANSWERED, ResolutionStatus.LOW_CONFIDENCE)
        )

        return zero_chunks or score_below_threshold or conf_low_or_none

    def detect_gaps(self, min_cluster_size: int = 1) -> List[KnowledgeGapReport]:
        """
        Scans queries for unanswered or low-confidence patterns,
        clusters them by lexical thematic Jaccard similarity, and produces gap reports.
        """
        all_queries = self.storage.get_queries(limit=1000)

        # 1. Identify candidate gap queries meeting conditions 1-4
        gap_candidates: List[QueryLogEntry] = [
            q for q in all_queries if self.is_candidate_gap_query(q)
        ]

        if not gap_candidates:
            return []

        # 2. Cluster candidates using lexical/thematic Jaccard overlap & term co-occurrence
        clusters: List[Dict[str, Any]] = []

        for q in gap_candidates:
            q_keywords = set(self.extract_keywords(q.query_text))
            q_stemmed = self.extract_stemmed_keywords(q.query_text)
            if not q_keywords:
                q_keywords = {self.normalize_query(q.query_text)}
                q_stemmed = {self.normalize_query(q.query_text)}

            best_cluster = None
            best_overlap = 0.0

            for cl in clusters:
                # Allow overlap within same domain or if general
                domain_match = (
                    cl["domain"] == q.domain
                    or cl["domain"] == "general"
                    or q.domain == "general"
                )
                if domain_match:
                    intersection = len(q_stemmed.intersection(cl["stemmed_keywords"]))
                    union = len(q_stemmed.union(cl["stemmed_keywords"]))
                    overlap = intersection / union if union > 0 else 0.0

                    # Match if Jaccard overlap >= 0.20 or at least 2 key stemmed terms overlap
                    if (overlap >= 0.20 or intersection >= 2) and overlap >= best_overlap:
                        best_overlap = overlap
                        best_cluster = cl

            if best_cluster is not None:
                best_cluster["queries"].append(q)
                best_cluster["keywords"].update(q_keywords)
                best_cluster["stemmed_keywords"].update(q_stemmed)
                best_cluster["domains"].add(q.domain)
                if best_cluster["domain"] == "general" and q.domain != "general":
                    best_cluster["domain"] = q.domain
            else:
                clusters.append({
                    "domain": q.domain,
                    "domains": {q.domain},
                    "keywords": set(q_keywords),
                    "stemmed_keywords": set(q_stemmed),
                    "queries": [q]
                })

        # 3. Transform clusters into KnowledgeGapReport items (Conditions 5 & 6)
        gap_reports: List[KnowledgeGapReport] = []

        for cl in clusters:
            queries: List[QueryLogEntry] = cl["queries"]
            if len(queries) < min_cluster_size:
                continue

            # Determine topic title from top keywords
            freq_words: Dict[str, int] = defaultdict(int)
            for q in queries:
                for kw in self.extract_keywords(q.query_text):
                    freq_words[kw] += 1

            top_words = sorted(freq_words.items(), key=lambda x: x[1], reverse=True)[:3]
            topic_title = " ".join(w[0].capitalize() for w in top_words) if top_words else "Unknown Concept"

            unanswered_count = sum(
                1 for q in queries
                if q.status == ResolutionStatus.UNANSWERED
                or not q.has_sufficient_evidence
                or q.confidence_level == "NONE"
            )
            low_conf_count = sum(
                1 for q in queries
                if q.status == ResolutionStatus.LOW_CONFIDENCE
                or q.confidence_level == "LOW"
            )

            freq = len(queries)
            avg_conf = sum(q.confidence_score for q in queries) / freq if freq else 0.0
            avg_top_score = sum(q.top_score for q in queries) / freq if freq else 0.0

            # Compute severity score
            severity_score = freq * (1.0 - avg_conf) + 1.5 * unanswered_count
            if severity_score >= 3.5 or unanswered_count >= 2:
                severity = GapSeverity.CRITICAL
            elif severity_score >= 2.0 or freq >= 2:
                severity = GapSeverity.HIGH
            elif severity_score >= 1.0:
                severity = GapSeverity.MEDIUM
            else:
                severity = GapSeverity.LOW

            samples = list(dict.fromkeys(q.query_text for q in queries))[:5]
            timestamps = [q.timestamp for q in queries if q.timestamp]
            first_seen = min(timestamps) if timestamps else datetime.now(timezone.utc).isoformat()
            last_seen = max(timestamps) if timestamps else datetime.now(timezone.utc).isoformat()

            norm_q = self.normalize_query(queries[0].query_text) if queries else ""

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
                average_confidence=round(avg_conf, 4),
                unanswered_count=unanswered_count,
                low_confidence_count=low_conf_count,
                average_retrieval_score=round(avg_top_score, 4),
                domains_involved=sorted(list(cl["domains"])),
                normalized_query=norm_q,
                sample_queries=samples,
                suggested_actions=actions,
                first_detected=first_seen,
                last_detected=last_seen
            )
            gap_reports.append(report)

        gap_reports.sort(
            key=lambda x: (x.severity == GapSeverity.CRITICAL, x.severity == GapSeverity.HIGH, x.frequency),
            reverse=True
        )

        # Save to persistent analytics storage
        self.storage.save_gap_reports(gap_reports)
        return gap_reports

    def get_frequent_query_themes(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Identifies frequently asked query topics across all resolved and active queries."""
        all_queries = self.storage.get_queries(limit=1000)
        word_counts: Dict[str, int] = defaultdict(int)

        for q in all_queries:
            keywords = self.extract_keywords(q.query_text)
            for kw in keywords:
                word_counts[kw] += 1

        top_themes = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        tot = len(all_queries)
        return [
            {
                "keyword": kw,
                "frequency": count,
                "relative_share": round(count / tot, 2) if tot else 0.0
            }
            for kw, count in top_themes
        ]
