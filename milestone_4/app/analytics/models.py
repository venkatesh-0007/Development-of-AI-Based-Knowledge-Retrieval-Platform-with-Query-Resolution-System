"""Data models for Query Analytics and Knowledge Gap Detection."""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime

class ResolutionStatus(str, Enum):
    RESOLVED = "RESOLVED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    UNANSWERED = "UNANSWERED"
    CLARIFICATION_REQUESTED = "CLARIFICATION_REQUESTED"
    CLARIFICATION_RESOLVED = "CLARIFICATION_RESOLVED"
    ERROR = "ERROR"

class GapSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

@dataclass
class QueryLogEntry:
    """Individual query activity log entry with deep resolution telemetry."""
    query_id: str
    timestamp: str
    query_text: str
    effective_query: str
    domain: str
    query_type: str
    route_target: str
    top_score: float
    avg_score: float
    confidence_score: float
    confidence_level: str
    has_sufficient_evidence: bool
    requires_clarification: bool
    status: ResolutionStatus
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    clarification_id: Optional[str] = None
    response_text: str = ""
    latency_ms: float = 0.0
    session_id: Optional[str] = None
    input_modality: str = "text"  # 'text' or 'voice'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "timestamp": self.timestamp,
            "query_text": self.query_text,
            "effective_query": self.effective_query,
            "domain": self.domain,
            "query_type": self.query_type,
            "route_target": self.route_target,
            "top_score": self.top_score,
            "avg_score": self.avg_score,
            "confidence_score": self.confidence_score,
            "confidence_level": self.confidence_level,
            "has_sufficient_evidence": self.has_sufficient_evidence,
            "requires_clarification": self.requires_clarification,
            "status": self.status.value if isinstance(self.status, ResolutionStatus) else str(self.status),
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "sources": self.sources,
            "clarification_id": self.clarification_id,
            "response_text": self.response_text,
            "latency_ms": self.latency_ms,
            "session_id": self.session_id,
            "input_modality": self.input_modality
        }

@dataclass
class KnowledgeGapReport:
    """Identified knowledge base gap derived from recurring unanswered or low-confidence queries."""
    gap_id: str
    topic: str
    domain: str
    severity: GapSeverity
    frequency: int
    average_confidence: float
    unanswered_count: int
    sample_queries: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)
    first_detected: str = ""
    last_detected: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gap_id": self.gap_id,
            "topic": self.topic,
            "domain": self.domain,
            "severity": self.severity.value if isinstance(self.severity, GapSeverity) else str(self.severity),
            "frequency": self.frequency,
            "average_confidence": self.average_confidence,
            "unanswered_count": self.unanswered_count,
            "sample_queries": self.sample_queries,
            "suggested_actions": self.suggested_actions,
            "first_detected": self.first_detected,
            "last_detected": self.last_detected
        }

@dataclass
class AnalyticsFilter:
    """Filter criteria for analytics telemetry."""
    domain: Optional[str] = None
    query_type: Optional[str] = None
    confidence_level: Optional[str] = None
    status: Optional[str] = None
    search_query: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None

@dataclass
class AnalyticsSummary:
    """Aggregated operational telemetry and KPI metrics."""
    total_queries: int
    resolution_rate: float
    unanswered_count: int
    low_confidence_count: int
    clarification_count: int
    avg_confidence: float
    avg_latency_ms: float
    domain_breakdown: Dict[str, int] = field(default_factory=dict)
    query_type_breakdown: Dict[str, int] = field(default_factory=dict)
    status_breakdown: Dict[str, int] = field(default_factory=dict)
    confidence_distribution: Dict[str, int] = field(default_factory=dict)
    daily_query_volume: Dict[str, int] = field(default_factory=dict)
    active_gaps_count: int = 0
