"""Response Generation Agent for Milestone 4 (M4.3 Optimization, Transparency, & Speech).

Synthesizes grounded responses strictly from retrieved knowledge chunks with query-type tailored
formatting (factual, procedural, comparative, clarified), empirical confidence scoring,
source attribution, full Response Transparency Panel payload, and spoken text rendering.
"""
import re
from typing import List, Dict, Any, Optional

from .models import (
    QueryAnalysisResult,
    QueryType,
    RetrievalResult,
    RetrievalChunk,
    SourceAttribution,
    AgentResponse,
    ConfidenceLevel,
    TransparencyScoreBreakdown,
    TransparencyPanelPayload,
    VoicePayload
)
from .voice import VoiceModule
from ..confidence.calculator import ConfidenceCalculator, default_confidence_calculator

class ResponseGenerationAgent:
    """Generates grounded responses strictly supported by retrieved context with transparency payload."""

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        confidence_calculator: Optional[ConfidenceCalculator] = None
    ):
        self.llm_client = llm_client
        self.confidence_calculator = confidence_calculator or default_confidence_calculator
        self.voice_module = VoiceModule()

    def generate(
        self,
        query_analysis: QueryAnalysisResult,
        retrieval_result: RetrievalResult,
        is_clarified_resolution: bool = False,
        refined_query: Optional[str] = None
    ) -> AgentResponse:
        chunks = retrieval_result.chunks

        # Case 1: Insufficient Evidence or Zero Chunks
        conf_res = self.confidence_calculator.calculate(
            retrieval_result.top_score,
            retrieval_result.average_score
        )
        min_evidence_threshold = max(0.30, self.confidence_calculator.thresholds.low)
        if not chunks or retrieval_result.top_score < min_evidence_threshold:
            answer_text = (
                "No sufficiently relevant information was found in the knowledge base to answer your question. "
                "The query may refer to concepts outside the currently indexed documentation."
            )
            score_breakdown = TransparencyScoreBreakdown(
                top_chunk_score=retrieval_result.top_score,
                avg_top_k_score=retrieval_result.average_score,
                combined_score=conf_res.combined_score,
                confidence_level=conf_res.confidence_level
            )
            transparency = TransparencyPanelPayload(
                score_breakdown=score_breakdown,
                ranked_chunks=chunks,
                source_attributions=[],
                query_type=query_analysis.query_type,
                routing_target=query_analysis.route_to,
                anaphora_rewritten_query=query_analysis.cleaned_query if query_analysis.cleaned_query != query_analysis.query else None,
                clarification_occurred=is_clarified_resolution
            )
            spoken = self.voice_module.clean_for_speech(answer_text)

            return AgentResponse(
                answer=answer_text,
                confidence_score=score_breakdown.combined_score,
                confidence_level=score_breakdown.confidence_level,
                sources=[],
                query_analysis=query_analysis,
                has_sufficient_evidence=False,
                requires_clarification=False,
                transparency=transparency,
                spoken_text=spoken,
                is_clarified_resolution=is_clarified_resolution,
                refined_query=refined_query
            )

        # Case 2: Grounded Synthesis
        # 1. Source attributions
        sources: List[SourceAttribution] = []
        source_names: List[str] = []
        for c in chunks:
            doc = c.document_name
            if doc not in source_names:
                source_names.append(doc)
            sources.append(SourceAttribution(
                document_name=doc,
                chunk_id=c.chunk_id,
                relevance_score=c.score,
                page=c.page,
                section=c.section,
                snippet=c.content[:140].strip() + ("..." if len(c.content) > 140 else "")
            ))

        # 2. Confidence Calculation (70% Top Score + 30% Avg Score)
        score_breakdown = TransparencyScoreBreakdown(
            top_chunk_score=retrieval_result.top_score,
            avg_top_k_score=retrieval_result.average_score
        )

        # 3. Format grounded response based on query type
        if self.llm_client and hasattr(self.llm_client, "generate"):
            answer = self._generate_llm(query_analysis, chunks)
        else:
            answer = self._generate_deterministic(query_analysis, chunks)

        # Append source citation list
        citations_summary = "\n\n**Sources:**\n" + "\n".join(
            f"- `{c.document_name}` (Chunk {c.rank}, Similarity: {c.score:.2f})" for c in chunks[:3]
        )
        final_answer = answer + citations_summary

        transparency = TransparencyPanelPayload(
            score_breakdown=score_breakdown,
            ranked_chunks=chunks,
            source_attributions=sources,
            query_type=query_analysis.query_type,
            routing_target=query_analysis.route_to,
            anaphora_rewritten_query=query_analysis.cleaned_query if query_analysis.cleaned_query != query_analysis.query else None,
            clarification_occurred=is_clarified_resolution
        )

        spoken = self.voice_module.clean_for_speech(answer)

        return AgentResponse(
            answer=final_answer,
            confidence_score=score_breakdown.combined_score,
            confidence_level=score_breakdown.confidence_level,
            sources=source_names,
            query_analysis=query_analysis,
            has_sufficient_evidence=True,
            requires_clarification=False,
            transparency=transparency,
            spoken_text=spoken,
            is_clarified_resolution=is_clarified_resolution,
            refined_query=refined_query
        )

    def _generate_deterministic(
        self,
        query_analysis: QueryAnalysisResult,
        chunks: List[RetrievalChunk]
    ) -> str:
        q_type = query_analysis.query_type

        if q_type == QueryType.FACTUAL:
            return self._format_factual(chunks)
        elif q_type == QueryType.PROCEDURAL:
            return self._format_procedural(chunks)
        elif q_type == QueryType.COMPARATIVE:
            return self._format_comparative(chunks)
        elif q_type == QueryType.MULTI_PART:
            return self._format_multi_part(chunks, query_analysis)
        else:
            return "\n\n".join(c.content for c in chunks[:2])

    def _format_factual(self, chunks: List[RetrievalChunk]) -> str:
        lead = chunks[0].content.strip()

        extra_bullets = []
        if len(chunks) > 1:
            for extra in chunks[1:3]:
                for l in extra.content.splitlines():
                    clean_l = l.strip()
                    if clean_l and not clean_l.startswith("#") and len(clean_l) > 25 and clean_l not in lead:
                        extra_bullets.append(clean_l)
                        break

        if extra_bullets:
            lead += "\n\n**Key Aspects & Context:**\n" + "\n".join(f"- {b}" for b in extra_bullets[:3])
        return lead

    def _format_procedural(self, chunks: List[RetrievalChunk]) -> str:
        combined = "\n".join(c.content for c in chunks)
        lines = [l.strip() for l in combined.splitlines() if l.strip()]
        steps = []
        for l in lines:
            if re.match(r"^(?:Step\s*\d+|Phase\s*\d+|\d+\.)\s*", l, re.IGNORECASE):
                steps.append(l)
            elif any(k in l.lower() for k in ["first", "then", "next", "finally", "handshake", "mitigation"]):
                steps.append(l)

        if steps:
            formatted = "**Step-by-Step Procedure & Implementation:**\n\n"
            for i, st in enumerate(steps[:6], 1):
                clean_st = re.sub(r"^(?:Step\s*\d+|Phase\s*\d+|\d+\.)\s*[:\-\.]?\s*", "", st, flags=re.IGNORECASE)
                formatted += f"{i}. {clean_st}\n"
            return formatted.strip()

        return "**Procedural Summary:**\n\n" + "\n\n".join(c.content for c in chunks[:2])

    def _format_comparative(self, chunks: List[RetrievalChunk]) -> str:
        combined = "\n\n".join(c.content for c in chunks)
        output = "**Comparative Analysis & Trade-offs:**\n\n"
        sections = [s.strip() for s in combined.split("\n\n") if s.strip()]
        contrasts = []
        for s in sections:
            if any(term in s.lower() for term in ["vs", "difference", "comparison", "contrast", "table", "protocol", "model", "symmetric"]):
                contrasts.append(s)
        if contrasts:
            output += "\n\n".join(contrasts[:3])
        else:
            output += combined
        return output

    def _format_multi_part(self, chunks: List[RetrievalChunk], qa: QueryAnalysisResult) -> str:
        output = "**Multi-Part Analysis:**\n\n"
        sub_qs = qa.sub_questions
        for i, sq in enumerate(sub_qs, 1):
            matching_chunk = chunks[min(i-1, len(chunks)-1)]
            output += f"**Part {i}: {sq}**\n{matching_chunk.content[:200]}...\n\n"
        return output.strip()

    def _generate_llm(self, query_analysis: QueryAnalysisResult, chunks: List[RetrievalChunk]) -> str:
        context_str = "\n\n---\n\n".join(
            f"[Source: {c.document_name} | Chunk: {c.chunk_id}]\n{c.content}" for c in chunks
        )
        prompt = (
            f"You are an AI knowledge assistant. Answer the user query using strictly the context provided below.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Query: {query_analysis.query}\n"
            f"Query Type: {query_analysis.query_type.value}\n\n"
            f"Grounded Response:"
        )
        try:
            return self.llm_client.generate(prompt)
        except Exception:
            return self._generate_deterministic(query_analysis, chunks)
