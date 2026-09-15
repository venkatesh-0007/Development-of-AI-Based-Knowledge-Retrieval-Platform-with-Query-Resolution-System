"""Query Understanding Agent for Milestone 2 (M2.1).

Classifies incoming queries into exactly four categories:
- factual
- procedural
- comparative
- ambiguous

Implements hybrid LLM classification with a deterministic rule-based fallback engine,
heuristic confidence scoring, entity extraction, and deterministic routing.
"""
import os
import re
import json
import logging
from pathlib import Path
from typing import List, Optional, Any, Dict, Tuple
from .models import QueryType, RoutingTarget, QueryAnalysisResult

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

    AMBIGUOUS_PATTERNS = [
        r"^(?:help|hello|hi|hey|test|info|details|explain|tell\s+me)\b[!?.]*$",
        r"^(?:tell\s+me\s+about\s+it|explain\s+this|what\s+about\s+that|how\s+does\s+it\s+work|give\s+me\s+more\s+information)\b[!?.]*$",
        r"\b(?:that\s+thing|this\s+thing|that\s+one|this\s+one|stuff|something)\b",
        r"^(?:why|how|what|which|who|where|when)[!?.]*$",
        r"^(?:how\s+to\s+do\s+it|how\s+to\s+fix\s+it|tell\s+me\s+about\s+it|what\s+about\s+it)[!?.]*$",
        r"^(?:which\s+is\s+better|what\s+is\s+best|compare\s+them)[!?.]*$",
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

    def __init__(self, llm_client: Optional[Any] = None, prompt_path: Optional[str] = None):
        """
        Initialize Query Understanding Agent.
        :param llm_client: Optional LLM client implementing `generate(prompt: str) -> str`
        :param prompt_path: Optional custom path to query classification prompt file
        """
        self.llm_client = llm_client
        self.prompt_template = self._load_prompt(prompt_path)

    def _load_prompt(self, prompt_path: Optional[str]) -> str:
        if prompt_path and Path(prompt_path).exists():
            return Path(prompt_path).read_text(encoding="utf-8")
        
        # Default prompt path relative to file
        default_file = Path(__file__).resolve().parent.parent / "prompts" / "query_classification.txt"
        if default_file.exists():
            return default_file.read_text(encoding="utf-8")

        return "Classify the following query into factual, procedural, comparative, or ambiguous:\nUser Query: {query}"

    def analyze(self, query: str) -> QueryAnalysisResult:
        """
        Analyze incoming query and produce structured QueryAnalysisResult.
        Pipeline: Input Validation -> Obvious Ambiguity -> Optional LLM -> Rule Fallback -> Deterministic Route.
        """
        # 1. Input validation & cleaning
        if not query or not query.strip():
            return self._build_result(
                query=query or "",
                cleaned="",
                qtype=QueryType.AMBIGUOUS,
                conf=1.0,
                reason="The input query is empty or contains only whitespace.",
                entities=[]
            )

        cleaned = self._clean_query(query)
        words = cleaned.lower().split()

        # 2. Obvious Ambiguity Detection
        is_amb, amb_reason, amb_conf = self._check_ambiguity(cleaned, words)
        if is_amb:
            return self._build_result(
                query=query,
                cleaned=cleaned,
                qtype=QueryType.AMBIGUOUS,
                conf=amb_conf,
                reason=amb_reason,
                entities=self._extract_entities(cleaned)
            )

        # 3. Optional LLM Classification
        if self.llm_client is not None and hasattr(self.llm_client, "generate"):
            try:
                llm_result = self._classify_with_llm(query, cleaned)
                if llm_result is not None:
                    return llm_result
            except Exception as e:
                logger.warning(f"LLM classification failed, falling back to rule engine: {e}")

        # 4. Deterministic Rule-Based Classification (Fallback / Default)
        return self._classify_with_rules(query, cleaned)

    def _classify_with_rules(self, raw_query: str, cleaned: str) -> QueryAnalysisResult:
        """Deterministic regex and pattern matching engine."""
        entities = self._extract_entities(cleaned)

        # Comparative check
        is_comp, comp_reason, comp_conf = self._check_comparative(cleaned)
        if is_comp:
            return self._build_result(
                query=raw_query,
                cleaned=cleaned,
                qtype=QueryType.COMPARATIVE,
                conf=comp_conf,
                reason=comp_reason,
                entities=entities
            )

        # Procedural check
        is_proc, proc_reason, proc_conf = self._check_procedural(cleaned)
        if is_proc:
            return self._build_result(
                query=raw_query,
                cleaned=cleaned,
                qtype=QueryType.PROCEDURAL,
                conf=proc_conf,
                reason=proc_reason,
                entities=entities
            )

        # Factual check
        is_fact, fact_reason, fact_conf = self._check_factual(cleaned)
        if is_fact:
            return self._build_result(
                query=raw_query,
                cleaned=cleaned,
                qtype=QueryType.FACTUAL,
                conf=fact_conf,
                reason=fact_reason,
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

    def _classify_with_llm(self, raw_query: str, cleaned: str) -> Optional[QueryAnalysisResult]:
        """Calls external LLM and validates structured JSON schema."""
        prompt = self.prompt_template.format(query=cleaned)
        raw_output = self.llm_client.generate(prompt)
        
        # Strip potential markdown backticks
        cleaned_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_output.strip())
        data = json.loads(cleaned_json)

        qtype_raw = str(data.get("query_type", "")).lower().strip()
        type_mapping = {
            "factual": QueryType.FACTUAL,
            "procedural": QueryType.PROCEDURAL,
            "comparative": QueryType.COMPARATIVE,
            "ambiguous": QueryType.AMBIGUOUS
        }
        if qtype_raw not in type_mapping:
            return None

        qtype = type_mapping[qtype_raw]
        conf = float(data.get("classification_confidence", data.get("confidence", 0.85)))
        conf = max(0.0, min(1.0, conf)) # Clamp between 0.0 and 1.0

        reason = str(data.get("reasoning", f"LLM classified as {qtype.value}."))
        entities = list(data.get("entities", self._extract_entities(cleaned)))
        reformulated = str(data.get("reformulated_query", cleaned))

        return self._build_result(
            query=raw_query,
            cleaned=cleaned,
            qtype=qtype,
            conf=conf,
            reason=reason,
            entities=entities,
            reformulated=reformulated
        )

    def _build_result(
        self,
        query: str,
        cleaned: str,
        qtype: QueryType,
        conf: float,
        reason: str,
        entities: List[str],
        reformulated: Optional[str] = None
    ) -> QueryAnalysisResult:
        """Applies deterministic routing and builds QueryAnalysisResult."""
        # Deterministic routing logic
        if qtype == QueryType.AMBIGUOUS:
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
            reformulated_query=reformulated
        )

    def _clean_query(self, query: str) -> str:
        text = query.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _check_ambiguity(self, text: str, words: List[str]) -> Tuple[bool, str, float]:
        lower = text.lower()

        for pattern in self.AMBIGUOUS_PATTERNS:
            if re.search(pattern, lower):
                return True, "Query matches ambiguous pattern lacking specific topic entities.", 0.92

        if len(words) == 1:
            clean_word = re.sub(r"[^\w-]", "", words[0])
            if clean_word not in self.KNOWN_DOMAIN_ENTITIES:
                return True, f"Single-word query '{words[0]}' is too underspecified for reliable retrieval.", 0.88

        # Vague pronoun without concrete entity
        vague_refs = re.findall(r"\b(it|this|that|them|those|these|thing|stuff)\b", lower)
        concrete_entities = self._extract_entities(text)
        if vague_refs and len(concrete_entities) == 0:
            return True, "Query contains vague pronouns or demonstratives without identifying a concrete subject.", 0.85

        return False, "", 0.0

    def _check_comparative(self, text: str) -> Tuple[bool, str, float]:
        lower = text.lower()
        for pattern in self.COMPARATIVE_PATTERNS:
            if re.search(pattern, lower):
                return True, "Contains explicit comparative terminology or multi-entity comparison.", 0.95
        return False, "", 0.0

    def _check_procedural(self, text: str) -> Tuple[bool, str, float]:
        lower = text.lower()
        for pattern in self.PROCEDURAL_PATTERNS:
            if re.search(pattern, lower):
                return True, "Query requests steps, workflows, instructions, or process execution details.", 0.92
        return False, "", 0.0

    def _check_factual(self, text: str) -> Tuple[bool, str, float]:
        lower = text.lower()
        for pattern in self.FACTUAL_PATTERNS:
            if re.search(pattern, lower):
                return True, "Query seeks factual definition, property, parameter, or identity.", 0.90
        return False, "", 0.0

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
