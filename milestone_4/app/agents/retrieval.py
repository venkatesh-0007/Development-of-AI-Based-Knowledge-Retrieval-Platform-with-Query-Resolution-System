"""Optimized Hybrid Retrieval Agent for Milestone 4 (M4.3).

Features:
- Hybrid dense-lexical ranking: Combines semantic vector similarity with term-level overlap
- Dynamic Top-K selection based on query intent complexity
- Domain-aware relevance boosting & cross-domain suppression
- Calibrated similarity thresholding
- Chunk deduplication and rank ordering
"""
import re
import time
from typing import Optional, List, Dict, Any, Set

from .models import (
    QueryType,
    QueryAnalysisResult,
    RetrievalChunk,
    RetrievalResult
)

STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "what", "how", "why", "which", "who", "explain", "describe", "tell",
    "does", "do", "did", "can", "could", "would", "should"
}

class RetrievalAgent:
    """Retrieves relevant document chunks with hybrid semantic-lexical scoring and dynamic Top-K."""

    DEFAULT_THRESHOLD = 0.35

    def __init__(
        self,
        embedder,
        vectorstore,
        confidence_threshold: float = DEFAULT_THRESHOLD,
        default_top_k: int = 5
    ):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.confidence_threshold = confidence_threshold
        self.default_top_k = default_top_k

    def _determine_optimal_top_k(self, query_type: QueryType, requested_top_k: Optional[int]) -> int:
        if requested_top_k is not None and requested_top_k > 0:
            return requested_top_k
        if query_type == QueryType.COMPARATIVE:
            return 6
        elif query_type in (QueryType.PROCEDURAL, QueryType.MULTI_PART):
            return 5
        return 4

    def _extract_query_keywords(self, text: str) -> Set[str]:
        tokens = re.findall(r"[a-zA-Z0-9]{2,}", text.lower())
        return {t for t in tokens if t not in STOPWORDS and not t.isdigit()}

    def _compute_lexical_overlap(self, query_keywords: Set[str], chunk_content: str) -> float:
        if not query_keywords:
            return 0.0
        c_tokens = set(re.findall(r"[a-zA-Z0-9]{2,}", chunk_content.lower()))
        intersection = query_keywords.intersection(c_tokens)
        return len(intersection) / len(query_keywords)

    def retrieve(
        self,
        query_input: Any,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> RetrievalResult:
        start_time = time.perf_counter()

        if isinstance(query_input, QueryAnalysisResult):
            query_text = query_input.cleaned_query or query_input.query
            query_type = query_input.query_type
            domain = query_input.domain
        else:
            query_text = str(query_input)
            query_type = QueryType.FACTUAL
            domain = "general"

        effective_top_k = self._determine_optimal_top_k(query_type, top_k)
        effective_threshold = threshold if threshold is not None else self.confidence_threshold
        query_keywords = self._extract_query_keywords(query_text)

        # Generate query vector
        query_vectors = self.embedder.encode([query_text])
        if not query_vectors:
            return RetrievalResult(
                query=query_text,
                chunks=[],
                top_score=0.0,
                average_score=0.0,
                total_candidates=0,
                filter_threshold=effective_threshold,
                execution_time_ms=round((time.perf_counter() - start_time) * 1000, 2)
            )

        query_emb = query_vectors[0]
        # Query broader pool for hybrid re-ranking
        search_k = min(max(effective_top_k * 4, 20), 50)
        raw_results = self.vectorstore.search(query_emb, top_k=search_k)

        scored_candidates: List[Dict[str, Any]] = []
        seen_fingerprints = set()

        for raw in raw_results:
            content = raw.get("document", "")
            meta = raw.get("metadata", {}) or {}
            vector_score = float(raw.get("score", 0.0))

            # Deduplication
            fp = content[:100].strip().lower()
            if fp in seen_fingerprints:
                continue
            seen_fingerprints.add(fp)

            # Lexical overlap score
            lex_score = self._compute_lexical_overlap(query_keywords, content)

            # Hybrid score blending (55% vector similarity + 45% lexical overlap)
            hybrid_score = 0.55 * vector_score + 0.45 * lex_score

            # Domain affinity adjustment
            chunk_domain = meta.get("domain", "general")
            if domain != "general" and chunk_domain == domain:
                hybrid_score = min(1.0, hybrid_score + 0.05)

            if hybrid_score >= effective_threshold:
                scored_candidates.append({
                    "content": content,
                    "metadata": meta,
                    "score": round(hybrid_score, 4)
                })

        # Rank order
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = scored_candidates[:effective_top_k]

        chunks: List[RetrievalChunk] = []
        for idx, item in enumerate(top_candidates):
            meta = item["metadata"]
            chunks.append(RetrievalChunk(
                chunk_id=str(meta.get("chunk_id", f"chk_{idx+1}")),
                document_name=str(meta.get("document_name", meta.get("source", "Unknown Document"))),
                content=item["content"],
                score=item["score"],
                rank=idx + 1,
                page=meta.get("page"),
                section=meta.get("section"),
                metadata=meta
            ))

        top_score = chunks[0].score if chunks else 0.0
        avg_score = round(sum(c.score for c in chunks) / len(chunks), 4) if chunks else 0.0
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return RetrievalResult(
            query=query_text,
            chunks=chunks,
            top_score=top_score,
            average_score=avg_score,
            total_candidates=len(raw_results),
            filter_threshold=effective_threshold,
            execution_time_ms=round(elapsed_ms, 2)
        )
