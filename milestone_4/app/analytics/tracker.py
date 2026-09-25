"""Telemetry tracker for query logging and status classification."""
import uuid
from datetime import datetime, timezone
from typing import Optional, List

from .models import QueryLogEntry, ResolutionStatus
from .storage import AnalyticsStorage

class AnalyticsTracker:
    """Interceps multi-agent orchestration events and records detailed resolution telemetry."""

    def __init__(self, storage: AnalyticsStorage, low_confidence_threshold: float = 0.45):
        self.storage = storage
        self.low_confidence_threshold = low_confidence_threshold

    def record_orchestration(
        self,
        orchestration_result,
        input_modality: str = "text"
    ) -> QueryLogEntry:
        """Constructs a QueryLogEntry from an OrchestrationResult and persists it."""
        res = orchestration_result
        qa = getattr(res.response, "query_analysis", None)
        retrieval = res.retrieval_result
        
        # Determine domain
        domain = "general"
        if qa and getattr(qa, "domain", None):
            domain = qa.domain
        elif retrieval and retrieval.chunks:
            # Infer from top chunk metadata
            first_meta = getattr(retrieval.chunks[0], "metadata", {}) or {}
            domain = first_meta.get("domain", "general")

        # Determine query type and route
        query_type = qa.query_type.value if qa and hasattr(qa.query_type, "value") else (str(qa.query_type) if qa else "factual")
        route_target = qa.route_to.value if qa and hasattr(qa.route_to, "value") else "retrieval"

        # Scores
        top_score = 0.0
        avg_score = 0.0
        retrieved_ids = []
        sources = res.response.sources if res.response else []
        if retrieval and retrieval.chunks:
            top_score = retrieval.top_score
            avg_score = retrieval.average_score
            retrieved_ids = [c.chunk_id for c in retrieval.chunks]

        conf_score = getattr(res.response, "confidence_score", 0.0)
        conf_lvl = getattr(res.response, "confidence_level", None)
        conf_level_str = conf_lvl.value if hasattr(conf_lvl, "value") else str(conf_lvl or "NONE")
        has_evidence = getattr(res.response, "has_sufficient_evidence", False)
        req_clarification = getattr(res.response, "requires_clarification", False)

        # Classify status
        if res.status == "clarification_requested":
            status = ResolutionStatus.CLARIFICATION_REQUESTED
        elif res.status == "clarified_success":
            status = ResolutionStatus.CLARIFICATION_RESOLVED
        elif res.status == "error":
            status = ResolutionStatus.ERROR
        elif not has_evidence or (retrieval and len(retrieval.chunks) == 0) or conf_score == 0.0:
            status = ResolutionStatus.UNANSWERED
        elif conf_score < self.low_confidence_threshold:
            status = ResolutionStatus.LOW_CONFIDENCE
        else:
            status = ResolutionStatus.RESOLVED

        effective_q = res.query
        if qa and getattr(qa, "normalized_query", None):
            effective_q = qa.normalized_query
        if getattr(res.response, "refined_query", None):
            effective_q = res.response.refined_query

        clarification_id = None
        if res.clarification_request:
            clarification_id = res.clarification_request.clarification_id

        entry = QueryLogEntry(
            query_id="qry_" + uuid.uuid4().hex[:12],
            timestamp=datetime.now(timezone.utc).isoformat(),
            query_text=res.query,
            effective_query=effective_q,
            domain=domain,
            query_type=query_type,
            route_target=route_target,
            top_score=round(top_score, 4),
            avg_score=round(avg_score, 4),
            confidence_score=round(conf_score, 4),
            confidence_level=conf_level_str,
            has_sufficient_evidence=has_evidence,
            requires_clarification=req_clarification,
            status=status,
            retrieved_chunk_ids=retrieved_ids,
            sources=sources,
            clarification_id=clarification_id,
            response_text=res.response.answer if res.response else "",
            latency_ms=res.execution_time_ms,
            session_id=res.session_id,
            input_modality=input_modality
        )

        self.storage.log_query(entry)
        return entry
