"""Response Generation Agent for Milestone 2 (M2.3).

Synthesizes grounded responses from retrieved knowledge chunks with query-type tailored
formatting (factual, procedural, comparative), application-level confidence scoring,
and source attribution including document name, page, and snippet.
"""
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from .models import (
    QueryAnalysisResult,
    QueryType,
    RetrievalResult,
    RetrievalChunk,
    SourceAttribution,
    AgentResponse,
    ConfidenceLevel
)

class ResponseGenerationAgent:
    """Generates grounded responses strictly supported by retrieved context."""

    def __init__(self, llm_client: Optional[Any] = None, prompt_path: Optional[str] = None):
        """
        Initialize Response Generation Agent.
        :param llm_client: Optional LLM client (e.g. OpenAI/Gemini wrapper)
        :param prompt_path: Optional custom path to response generation prompt file
        """
        self.llm_client = llm_client
        self.prompt_template = self._load_prompt(prompt_path)

    def _load_prompt(self, prompt_path: Optional[str]) -> str:
        if prompt_path and Path(prompt_path).exists():
            return Path(prompt_path).read_text(encoding="utf-8")
        
        default_file = Path(__file__).resolve().parent.parent / "prompts" / "response_generation.txt"
        if default_file.exists():
            return default_file.read_text(encoding="utf-8")

        return (
            "You are a grounded knowledge base AI assistant.\n"
            "Answer the question strictly using ONLY the provided context.\n"
            "If insufficient, state: 'No sufficiently relevant information was found in the knowledge base.'\n\n"
            "Context:\n{context}\n\nUser Question: {query}\nQuery Type: {query_type}\n\nGrounded Answer:"
        )

    def generate(
        self,
        query_analysis: QueryAnalysisResult,
        retrieval_result: RetrievalResult
    ) -> AgentResponse:
        """
        Generate grounded response, calculate confidence, and build source attribution.
        """
        # Handle empty or low-confidence evidence cases
        if not retrieval_result.has_sufficient_evidence or not retrieval_result.chunks:
            return AgentResponse(
                answer="No sufficiently relevant information was found in the knowledge base to answer your question.",
                confidence_score=retrieval_result.top_score,
                confidence_level=ConfidenceLevel.NONE,
                sources=[],
                query_analysis=query_analysis,
                has_sufficient_evidence=False,
                requires_clarification=False
            )

        # Build source attributions
        sources: List[SourceAttribution] = []
        for chunk in retrieval_result.chunks:
            doc_name = chunk.document_name or chunk.metadata.get("source", "Unknown Document")
            page_val = chunk.page
            section_val = chunk.section
            snippet = chunk.content[:150].strip() + ("..." if len(chunk.content) > 150 else "")
            
            sources.append(SourceAttribution(
                document_name=doc_name,
                chunk_id=chunk.chunk_id,
                relevance_score=chunk.similarity_score,
                snippet=snippet,
                page=page_val,
                section=section_val
            ))

        # Compute application-level confidence score & level
        confidence_score, confidence_level = self._compute_confidence(retrieval_result)

        # Generate answer text
        if self.llm_client is not None and hasattr(self.llm_client, "generate"):
            answer = self._generate_with_llm(query_analysis, retrieval_result.chunks)
        else:
            answer = self._generate_deterministic(query_analysis, retrieval_result.chunks)

        return AgentResponse(
            answer=answer,
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            sources=sources,
            query_analysis=query_analysis,
            has_sufficient_evidence=True,
            requires_clarification=False
        )

    def _compute_confidence(self, retrieval_result: RetrievalResult) -> tuple:
        """Calculate application-level retrieval-based confidence score and categorization."""
        if not retrieval_result.chunks:
            return 0.0, ConfidenceLevel.NONE

        top_score = retrieval_result.top_score
        avg_score = sum(c.similarity_score for c in retrieval_result.chunks) / len(retrieval_result.chunks)
        
        # Weighted metric (70% top chunk, 30% average across Top-K)
        final_score = round((0.70 * top_score) + (0.30 * avg_score), 4)

        if final_score >= 0.75:
            level = ConfidenceLevel.HIGH
        elif final_score >= 0.60:
            level = ConfidenceLevel.MEDIUM
        elif final_score >= 0.45:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.NONE

        return final_score, level

    def _generate_deterministic(
        self,
        query_analysis: QueryAnalysisResult,
        chunks: List[RetrievalChunk]
    ) -> str:
        """Synthesizes structured grounded text without external API calls."""
        query_type = query_analysis.query_type
        
        if query_type == QueryType.FACTUAL:
            return self._format_factual(chunks)
        elif query_type == QueryType.PROCEDURAL:
            return self._format_procedural(chunks)
        elif query_type == QueryType.COMPARATIVE:
            return self._format_comparative(chunks)
        else:
            return "\n\n".join(c.content for c in chunks)

    def _format_factual(self, chunks: List[RetrievalChunk]) -> str:
        primary = chunks[0].content
        lines = [line.strip() for line in primary.splitlines() if line.strip()]
        lead_summary = "\n\n".join(lines[:4])
        
        additional_points = []
        if len(chunks) > 1:
            for extra in chunks[1:3]:
                extra_lines = [l.strip() for l in extra.content.splitlines() if l.strip() and not l.startswith("#")]
                if extra_lines:
                    additional_points.append(extra_lines[0])

        if additional_points:
            lead_summary += "\n\n**Additional Context:**\n" + "\n".join(f"- {p}" for p in additional_points)

        return lead_summary

    def _format_procedural(self, chunks: List[RetrievalChunk]) -> str:
        combined_text = "\n".join(c.content for c in chunks)
        raw_lines = [l.strip() for l in combined_text.splitlines() if l.strip()]
        steps = []
        for line in raw_lines:
            if re.match(r"^\d+\.\s+", line):
                steps.append(line)
            elif line.startswith("- ") or line.startswith("* "):
                steps.append(line[2:])
            elif any(k in line.lower() for k in ["step", "first", "next", "then", "finally", "update rule", "procedure"]):
                steps.append(line)

        if steps:
            formatted = "**Procedure & Implementation Steps:**\n\n"
            for i, step in enumerate(steps[:8], 1):
                clean_step = re.sub(r"^\d+\.\s*", "", step)
                formatted += f"{i}. {clean_step}\n"
            return formatted.strip()

        return "**Workflow & Procedure Details:**\n\n" + "\n\n".join(c.content for c in chunks[:2])

    def _format_comparative(self, chunks: List[RetrievalChunk]) -> str:
        combined_text = "\n\n".join(c.content for c in chunks)
        output = "**Comparative Analysis:**\n\n"
        sections = [s.strip() for s in combined_text.split("\n\n") if s.strip()]
        
        comparison_points = []
        for sec in sections:
            if any(term in sec.lower() for term in ["vs", "difference", "comparison", "contrast", "table", "protocol", "model"]):
                comparison_points.append(sec)

        if comparison_points:
            output += "\n\n".join(comparison_points[:3])
        else:
            output += combined_text

        return output

    def _generate_with_llm(
        self,
        query_analysis: QueryAnalysisResult,
        chunks: List[RetrievalChunk]
    ) -> str:
        """Invokes external LLM client with strict grounding prompt."""
        context_str = "\n\n---\n\n".join(
            f"[Source: {c.document_name} | Page: {c.page or 'N/A'} | Relevance: {c.similarity_score:.2f}]\n{c.content}"
            for c in chunks
        )
        prompt = self.prompt_template.format(
            context=context_str,
            query=query_analysis.query,
            query_type=query_analysis.query_type.value
        )
        try:
            return self.llm_client.generate(prompt)
        except Exception:
            return self._generate_deterministic(query_analysis, chunks)
