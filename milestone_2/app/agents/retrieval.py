"""Retrieval Agent for Milestone 2 (M2.2).

Performs semantic vector search, dynamic Top-K ranking, relevance filtering,
and metadata preservation (document_name, page, section, chunk_id, score).
"""
import os
from typing import Optional, List, Dict, Any, Union
from .models import (
    QueryAnalysisResult,
    QueryType,
    RetrievalChunk,
    RetrievalResult
)

class RetrievalAgent:
    """Retrieves and filters relevant knowledge chunks from the vector database."""

    DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("RETRIEVAL_THRESHOLD", "0.45"))

    def __init__(self, embedder: Any, vectorstore: Any, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
        """
        Initialize the Retrieval Agent.

        :param embedder: Embedding model wrapper (Sentence Transformers)
        :param vectorstore: ChromaDB vector store instance
        :param confidence_threshold: Minimum similarity score (0.0 to 1.0) to retain a chunk
        """
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.confidence_threshold = confidence_threshold

    def retrieve(
        self,
        query_input: Union[str, QueryAnalysisResult],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> RetrievalResult:
        """
        Execute semantic retrieval and filtering.

        :param query_input: Raw query string or QueryAnalysisResult from QueryUnderstandingAgent
        :param top_k: Optional Top-K limit override
        :param threshold: Optional confidence threshold override
        :return: Structured RetrievalResult
        """
        if isinstance(query_input, QueryAnalysisResult):
            query_text = query_input.reformulated_query or query_input.cleaned_query or query_input.query
            query_type = query_input.query_type
        else:
            query_text = str(query_input).strip()
            query_type = QueryType.FACTUAL

        active_threshold = threshold if threshold is not None else self.confidence_threshold

        # Dynamic Top-K selection based on query type if not explicitly specified
        if top_k is None:
            if query_type == QueryType.FACTUAL:
                effective_k = 3
            elif query_type in (QueryType.PROCEDURAL, QueryType.COMPARATIVE):
                effective_k = 5
            else:
                effective_k = 4
        else:
            effective_k = max(1, top_k)

        if not query_text:
            return RetrievalResult(
                query="",
                query_type=query_type,
                chunks=[],
                top_score=0.0,
                total_retrieved=0,
                filtered_count=0,
                has_sufficient_evidence=False,
                confidence_threshold=active_threshold
            )

        try:
            # Generate query embedding
            query_embedding = self.embedder.encode([query_text])[0]

            # Search vector store (query up to 2x effective_k to allow threshold filtering)
            raw_results = self.vectorstore.search(query_embedding, top_k=effective_k * 2)
        except Exception:
            raw_results = []

        total_retrieved = len(raw_results)
        valid_chunks: List[RetrievalChunk] = []

        for item in raw_results:
            score = float(item.get("score", 0.0))
            if score >= active_threshold:
                metadata = item.get("metadata", {}) or {}
                chunk_id = metadata.get("chunk_id") or f"{metadata.get('file_hash', 'chunk')}-{metadata.get('chunk_index', len(valid_chunks))}"
                doc_name = metadata.get("document_name") or metadata.get("source", "Unknown Document")
                page = metadata.get("page")
                page_val = int(page) if (page is not None and str(page) not in ("0", "")) else None
                section = metadata.get("section")
                
                chunk = RetrievalChunk(
                    chunk_id=chunk_id,
                    content=item.get("content", ""),
                    metadata=metadata,
                    similarity_score=round(score, 4),
                    rank=len(valid_chunks) + 1,
                    document_name=doc_name,
                    page=page_val,
                    section=section
                )
                valid_chunks.append(chunk)

                if len(valid_chunks) >= effective_k:
                    break

        filtered_count = total_retrieved - len(valid_chunks)
        top_score = valid_chunks[0].similarity_score if valid_chunks else 0.0
        has_sufficient_evidence = len(valid_chunks) > 0 and top_score >= active_threshold

        return RetrievalResult(
            query=query_text,
            query_type=query_type,
            chunks=valid_chunks,
            top_score=top_score,
            total_retrieved=total_retrieved,
            filtered_count=filtered_count,
            has_sufficient_evidence=has_sufficient_evidence,
            confidence_threshold=active_threshold
        )
