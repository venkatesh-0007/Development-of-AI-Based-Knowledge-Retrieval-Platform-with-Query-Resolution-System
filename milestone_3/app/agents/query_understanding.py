"""Query Understanding Agent for Milestone 3 (M3.1 integration).

Classifies queries into factual, procedural, comparative, ambiguous, and multi-part.
Integrates directly with ClarificationAgent to detect underspecified or polysemous queries.
"""
import os
import re
import json
import logging
from pathlib import Path
from typing import List, Optional, Any, Dict, Tuple

from .models import QueryType, RoutingTarget, QueryAnalysisResult, ClarificationType
from .clarification import ClarificationAgent

logger = logging.getLogger(__name__)

class QueryUnderstandingAgent:
    """Analyzes and classifies queries for the multi-agent RAG system."""

    COMPARATIVE_PATTERNS = [
        r"\b(?:vs|versus)\b",
        r"\b(?:compare|comparing|comparison)\b",
        r"\b(?:difference|differences)\s+between\b",
        r"\b(?:distinguish|differentiate)\s+between\b",
        r"\b(?:pros\s+and\s+cons|advantages\s+and\s+disadvantages)\b",
        r"\b(?:better|faster|more\s+accurate|superior)\s+than\b",
        r"\b(?:how\s+is\s+.+?\s+different\s+from\s+.+?)\b",
        r"\b(?:how\s+does\s+.+?\s+differ\s+from\s+.+?)\b",
    ]

    PROCEDURAL_PATTERNS = [
        r"\bhow\s+(?:to|do\s+i|can\s+i|should\s+i|would\s+one|can\s+a)\b",
        r"\bstep[\s-]by[\s-]step\b",
        r"\b(?:steps?|procedure|process|workflow|guidelines?|tutorial|algorithm)\s+(?:to|for|of|in)\b",
        r"\b(?:instructions?\s+for|guide\s+to|guidance\s+on)\b",
        r"\bexplain\s+(?:the\s+steps|the\s+process|the\s+procedure|the\s+workflow|how)\b",
        r"\bhow\s+does\s+(?:.+?)\s+(?:work|establish|function|operate)\b",
        r"\b(?:implement|implementing|setup|configure|deploy|calculate|compute|training\s+process)\b",
        r"\b(?:dns\s+resolution\s+process|handshake\s+process|routing\s+process)\b",
    ]

    FACTUAL_PATTERNS = [
        r"\bwhat\s+(?:is|are|was|were|does|do|mean)\b",
        r"\b(?:define|definition\s+of|meaning\s+of)\b",
        r"\bwhich\s+(?:port|layer|protocol|algorithm|model|parameter|technique|method)\b",
        r"\b(?:who|when|where)\s+(?:is|are|was|were|does|did)\b",
        r"\b(?:list|name|identify|state)\s+(?:the|all|types\s+of)\b",
        r"\bwhat\s+are\s+the\s+types\s+of\b",
        r"\bexplain\s+(?:the\s+concept\s+of|the\s+term|what)\b",
    ]

    KNOWN_DOMAIN_ENTITIES = {
        "svm", "pca", "tcp", "udp", "dns", "http", "https", "dhcp", "ftp",
        "ssh", "smtp", "icmp", "osi", "sgd", "k-means", "knn", "bert", "rag"
    }

    STOPWORDS = {
        "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of",
        "with", "by", "from", "up", "about", "into", "over", "after", "is",
        "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "can", "could", "should", "would", "will", "shall",
        "what", "how", "why", "which", "who", "when", "where", "please", "tell",
        "me", "explain", "describe", "compare", "steps", "difference", "between"
    }

    def __init__(
        self,
        clarification_agent: Optional[ClarificationAgent] = None,
        llm_client: Optional[Any] = None,
        prompt_path: Optional[str] = None
    ):
        self.clarification_agent = clarification_agent or ClarificationAgent()
        self.llm_client = llm_client
        self.prompt_template = self._load_prompt(prompt_path)

    def _load_prompt(self, prompt_path: Optional[str]) -> str:
        if prompt_path and Path(prompt_path).exists():
            return Path(prompt_path).read_text(encoding="utf-8")
        return "Classify the following query into factual, procedural, comparative, or ambiguous:\nUser Query: {query}"

    def analyze(self, query: str) -> QueryAnalysisResult:
        """
        Analyze incoming query and check with ClarificationAgent for ambiguity,
        incomplete requirements, or multi-part structures.
        """
        if not query or not query.strip():
            clarification_req = self.clarification_agent.analyze_query(query)
            return self._build_result(
                query=query or "",
                cleaned="",
                qtype=QueryType.AMBIGUOUS,
                conf=1.0,
                reason="The input query is empty or whitespace only.",
                entities=[],
                clarification_req=clarification_req
            )

        cleaned = self._clean_query(query)

        # 1. Evaluate with ClarificationAgent (Ambiguity, Incompleteness, Multi-Part)
        clarification_req = self.clarification_agent.analyze_query(cleaned)
        if clarification_req is not None:
            qtype = QueryType.MULTI_PART if clarification_req.is_multi_part else QueryType.AMBIGUOUS
            return self._build_result(
                query=query,
                cleaned=cleaned,
                qtype=qtype,
                conf=0.95,
                reason=clarification_req.reason,
                entities=self._extract_entities(cleaned),
                clarification_req=clarification_req,
                sub_questions=clarification_req.sub_questions
            )

        # 2. Rule-Based Classification for unambiguous queries
        return self._classify_with_rules(query, cleaned)

    def _classify_with_rules(self, raw_query: str, cleaned: str) -> QueryAnalysisResult:
        entities = self._extract_entities(cleaned)

        # Comparative check
        for pattern in self.COMPARATIVE_PATTERNS:
            if re.search(pattern, cleaned, flags=re.IGNORECASE):
                return self._build_result(
                    query=raw_query,
                    cleaned=cleaned,
                    qtype=QueryType.COMPARATIVE,
                    conf=0.95,
                    reason="Contains explicit comparative terminology or multi-entity comparison.",
                    entities=entities
                )

        # Procedural check
        for pattern in self.PROCEDURAL_PATTERNS:
            if re.search(pattern, cleaned, flags=re.IGNORECASE):
                return self._build_result(
                    query=raw_query,
                    cleaned=cleaned,
                    qtype=QueryType.PROCEDURAL,
                    conf=0.92,
                    reason="Query requests steps, workflows, instructions, or process execution details.",
                    entities=entities
                )

        # Factual check
        for pattern in self.FACTUAL_PATTERNS:
            if re.search(pattern, cleaned, flags=re.IGNORECASE):
                return self._build_result(
                    query=raw_query,
                    cleaned=cleaned,
                    qtype=QueryType.FACTUAL,
                    conf=0.90,
                    reason="Query seeks factual definition, property, parameter, or identity.",
                    entities=entities
                )

        # Default fallback: Informational Factual
        return self._build_result(
            query=raw_query,
            cleaned=cleaned,
            qtype=QueryType.FACTUAL,
            conf=0.75,
            reason="Classified as factual based on general informational topic statement.",
            entities=entities
        )

    def _build_result(
        self,
        query: str,
        cleaned: str,
        qtype: QueryType,
        conf: float,
        reason: str,
        entities: List[str],
        clarification_req: Optional[Any] = None,
        sub_questions: Optional[List[str]] = None,
        reformulated: Optional[str] = None
    ) -> QueryAnalysisResult:
        if qtype in (QueryType.AMBIGUOUS, QueryType.MULTI_PART) or clarification_req is not None:
            route = RoutingTarget.CLARIFICATION
            requires_clarification = True
        else:
            route = RoutingTarget.RETRIEVAL
            requires_clarification = False

        if reformulated is None:
            reformulated = self._reformulate_query(cleaned) if cleaned else ""

        return QueryAnalysisResult(
            query=query,
            cleaned_query=cleaned,
            query_type=qtype,
            classification_confidence=conf,
            route_to=route,
            reasoning=reason,
            entities=entities,
            requires_clarification=requires_clarification,
            clarification_request=clarification_req,
            reformulated_query=reformulated,
            sub_questions=sub_questions or []
        )

    def _clean_query(self, query: str) -> str:
        text = query.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _extract_entities(self, text: str) -> List[str]:
        cleaned = re.sub(r"[^\w\s-]", " ", text)
        tokens = cleaned.split()
        entities = [t for t in tokens if t.lower() not in self.STOPWORDS and len(t) > 1]
        seen = set()
        deduped = []
        for e in entities:
            if e.lower() not in seen:
                seen.add(e.lower())
                deduped.append(e)
        return deduped

    def _reformulate_query(self, text: str) -> str:
        cleaned = re.sub(r"^(?:please|could you|can you|kindly|i want to know|tell me)\s+", "", text, flags=re.IGNORECASE)
        cleaned = cleaned.strip(" ?.")
        return cleaned
