"""Clarification Agent for Milestone 3.1 (M3.1).

Intelligently identifies ambiguous, incomplete, or multi-part queries.
Generates targeted follow-up questions asking strictly for the missing information.
Safely manages clarification state lifecycle (PENDING -> RESOLVED / CANCELLED).
Refines and disambiguates queries for downstream retrieval and response generation.
"""
import re
import time
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple, Set

from .models import (
    QueryType,
    RoutingTarget,
    ConfidenceLevel,
    ClarificationType,
    ClarificationStatus,
    ClarificationRequest,
    QueryAnalysisResult,
    AgentResponse
)

logger = logging.getLogger(__name__)

class ClarificationAgent:
    """Production-grade Clarification Agent implementing M3.1 specifications."""

    # Domain Disambiguation Catalog for polysemous concepts
    DOMAIN_DISAMBIGUATION_MAP: Dict[str, Dict[str, Any]] = {
        "tree": {
            "reason": "The term 'tree' is polysemous and can refer to Decision Trees in Machine Learning, Spanning Tree Protocol in Networking, or Tree Data Structures.",
            "follow_up": "Which type of tree are you referring to?",
            "options": [
                "Decision Trees (Machine Learning / Classification)",
                "Spanning Tree Protocol (Networking / STP)",
                "Binary Search Tree / B-Tree (Data Structures)"
            ],
            "qualifiers": {
                "decision", "spanning", "stp", "bst", "binary", "random forest", "cart",
                "id3", "c4.5", "b-tree", "avl", "red-black", "merkle", "syntax", "parse"
            }
        },
        "clustering": {
            "reason": "The term 'clustering' can refer to unsupervised Machine Learning algorithms, server high-availability clusters, or database clustering.",
            "follow_up": "Which aspect of clustering would you like to explore?",
            "options": [
                "K-Means Clustering (Machine Learning)",
                "Hierarchical Clustering (Machine Learning)",
                "High-Availability Server / Database Clustering"
            ],
            "qualifiers": {
                "k-means", "kmeans", "hierarchical", "dbscan", "density", "spectral",
                "gaussian", "gmm", "server", "failover", "database", "node", "ha"
            }
        },
        "handshake": {
            "reason": "The term 'handshake' could refer to the TCP 3-Way Handshake, TLS/SSL security handshake, or WebSocket connection handshake.",
            "follow_up": "Which protocol handshake are you inquiring about?",
            "options": [
                "TCP 3-Way Handshake (SYN, SYN-ACK, ACK)",
                "TLS / SSL Cryptographic Security Handshake",
                "WebSocket Connection Handshake"
            ],
            "qualifiers": {
                "tcp", "tls", "ssl", "syn", "ack", "websocket", "3-way", "three-way",
                "crypto", "certificate", "https", "ssh"
            }
        },
        "routing": {
            "reason": "The term 'routing' can refer to IP packet network routing (OSPF, BGP, RIP) or application web URL routing.",
            "follow_up": "Are you asking about network packet routing or application URL routing?",
            "options": [
                "Network Packet Routing (IP, OSPF, BGP, Routers)",
                "Application / Web Framework URL Routing"
            ],
            "qualifiers": {
                "ospf", "bgp", "rip", "packet", "ip", "router", "static", "dynamic",
                "url", "react", "next", "express", "gateway", "table", "cidr"
            }
        },
        "model": {
            "reason": "The term 'model' is broad without specifying a machine learning model, statistical model, or OSI network layer model.",
            "follow_up": "Could you clarify what kind of model you are interested in?",
            "options": [
                "Supervised Machine Learning Models (SVM, Linear Regression)",
                "Deep Learning / Neural Network Architectures",
                "OSI 7-Layer Reference Model"
            ],
            "qualifiers": {
                "svm", "linear", "logistic", "regression", "neural", "deep", "cnn",
                "rnn", "transformer", "bert", "osi", "tcp/ip", "markov", "language", "llm"
            }
        },
        "layer": {
            "reason": "The term 'layer' can refer to OSI/TCP-IP networking layers or neural network layers (Dense, Conv, Pooling).",
            "follow_up": "Which layer concept are you referring to?",
            "options": [
                "OSI 7-Layer Reference Model (Transport, Network, Application)",
                "TCP/IP 4-Layer Model",
                "Neural Network Hidden Layers (Dense, Convolutional)"
            ],
            "qualifiers": {
                "osi", "transport", "network", "datalink", "physical", "session",
                "presentation", "application", "hidden", "dense", "conv", "pooling", "activation"
            }
        },
        "port": {
            "reason": "The term 'port' could refer to TCP/UDP networking transport ports or hardware physical interfaces.",
            "follow_up": "Which type of port are you asking about?",
            "options": [
                "Standard Network Transport Ports (e.g., Port 80 HTTP, 443 HTTPS, 53 DNS)",
                "Physical Hardware / Switch Interfaces"
            ],
            "qualifiers": {
                "tcp", "udp", "80", "443", "53", "22", "21", "25", "http", "https",
                "dns", "ssh", "ftp", "switch", "hardware", "ethernet", "interface", "number"
            }
        },
        "pipeline": {
            "reason": "The term 'pipeline' can refer to ML Feature/Training Pipelines or CI/CD DevOps Software Pipelines.",
            "follow_up": "Which pipeline workflow do you want to explore?",
            "options": [
                "Machine Learning Data Ingestion & Training Pipeline",
                "CI/CD Software Deployment & Build Pipeline"
            ],
            "qualifiers": {
                "ml", "data", "training", "feature", "sklearn", "ci/cd", "devops",
                "jenkins", "github actions", "etl", "preprocessing"
            }
        }
    }

    # Recognized domain entities that are unambiguous singletons
    KNOWN_DOMAIN_ENTITIES: Set[str] = {
        "svm", "pca", "tcp", "udp", "dns", "http", "https", "dhcp", "ftp",
        "ssh", "smtp", "icmp", "osi", "sgd", "k-means", "knn", "bert", "rag",
        "ip", "bgp", "ospf", "ssl", "tls", "cidr", "arp", "vlan", "nat"
    }

    # Vague non-domain singleton words
    VAGUE_WORDS: Set[str] = {
        "thing", "things", "stuff", "process", "device", "code", "system",
        "details", "info", "help", "item", "concept", "technology", "method",
        "tool", "mechanism", "protocol", "algorithm", "entity"
    }

    # Incomplete syntax patterns (dangling prepositions, incomplete comparative operators)
    INCOMPLETE_PATTERNS = [
        r"^(?:how\s+to|how\s+do\s+i|steps\s+to|what\s+is\s+the|explain|tell\s+me)\s*$",
        r"^(?:compare|diff|versus|vs)\s*$",
        r"^(?:compare|diff|versus)\s+[a-zA-Z0-9_-]+\s*$",
        r"^(?:compare\s+[a-zA-Z0-9_-]+\s+(?:vs|to|with|and))\s*$",
        r"^[a-zA-Z0-9_-]+\s+(?:vs|versus)\s*$",
        r"\b(?:difference\s+between|distinguish\s+between|differences\s+of)\s*$",
        r"\b(?:steps\s+for|procedure\s+for|how\s+to\s+configure\s+with|process\s+of)\s*$",
        r"\b(?:better|faster|superior)\s+than\s*$",
    ]

    # Deictic pronouns and ungrounded demonstratives
    PRONOUN_PATTERNS = [
        r"\b(?:how\s+does\s+(?:it|this|that|they)\s+work|how\s+to\s+use\s+(?:it|this|that)|how\s+do\s+i\s+run\s+(?:it|this|that))\b",
        r"\b(?:what\s+are\s+(?:its|their)\s+advantages|what\s+are\s+(?:its|their)\s+features|what\s+is\s+(?:its|their)\s+purpose)\b",
        r"^(?:tell\s+me\s+about\s+(?:it|this|that|them)|explain\s+(?:this|that|it)|what\s+about\s+(?:that|this|it)|give\s+me\s+more\s+information)\b",
        r"\b(?:that\s+thing|this\s+thing|that\s+one|this\s+one)\b",
    ]

    def __init__(self):
        # In-memory storage for active clarification requests (clarification_id -> ClarificationRequest)
        self._pending_clarifications: Dict[str, ClarificationRequest] = {}

    def analyze_query(self, raw_query: str) -> Optional[ClarificationRequest]:
        """
        Analyze a raw query to determine if clarification is required before retrieval.
        Returns a ClarificationRequest if ambiguous/incomplete/multi-part, else None.
        """
        query = raw_query.strip()
        if not query:
            return self._build_request(
                original_query=raw_query,
                clarification_type=ClarificationType.INSUFFICIENT_INFORMATION,
                reason="The query is empty or contains only whitespace.",
                follow_up="Please provide a specific question or topic you would like to look up.",
                options=[
                    "What is Supervised vs Unsupervised Learning?",
                    "How does TCP 3-way handshake work?",
                    "Compare TCP vs UDP"
                ]
            )

        lower_query = query.lower()
        query_words = set(re.findall(r"\b[a-z0-9_-]+\b", lower_query))

        # 1. Multi-Part Query Analysis (Intelligently check if sub-questions are clear vs genuinely ambiguous)
        multi_part_req = self.analyze_multi_part(raw_query)
        if multi_part_req is not None:
            return multi_part_req

        # 2. Check for Incomplete Request (truncated inputs / dangling operators)
        for pattern in self.INCOMPLETE_PATTERNS:
            if re.search(pattern, query, flags=re.IGNORECASE):
                # Formulate targeted follow-up question
                if re.search(r"\bcompare\s+([a-zA-Z0-9_-]+)\s*$", query, flags=re.IGNORECASE):
                    entity = re.search(r"\bcompare\s+([a-zA-Z0-9_-]+)\s*$", query, flags=re.IGNORECASE).group(1)
                    return self._build_request(
                        original_query=raw_query,
                        clarification_type=ClarificationType.INCOMPLETE_REQUEST,
                        reason=f"You requested a comparison for '{entity}', but did not specify what to compare it against.",
                        follow_up=f"Which algorithm or protocol would you like to compare '{entity}' against?",
                        options=self._generate_comparison_options(entity)
                    )
                return self._build_request(
                    original_query=raw_query,
                    clarification_type=ClarificationType.INCOMPLETE_REQUEST,
                    reason=f"The query '{query}' appears incomplete or cut off.",
                    follow_up="Could you complete your question with the specific topic, action, or target entity?",
                    options=["Provide complete concept name", "Specify target of comparison"]
                )

        # 3. Check for Missing Context (ungrounded pronouns without antecedent in the query)
        has_pronoun_pattern = any(re.search(p, query, flags=re.IGNORECASE) for p in self.PRONOUN_PATTERNS)
        if has_pronoun_pattern:
            # Check if there is already a concrete domain noun antecedent in the query
            has_antecedent = any(entity in lower_query for entity in self.KNOWN_DOMAIN_ENTITIES)
            if not has_antecedent:
                return self._build_request(
                    original_query=raw_query,
                    clarification_type=ClarificationType.MISSING_CONTEXT,
                    reason="The query references an unspecified entity using pronouns ('it', 'this', 'that') without an antecedent.",
                    follow_up="Which specific technology, algorithm, or networking protocol are you asking about?",
                    options=[
                        "Support Vector Machines (SVM)",
                        "Transmission Control Protocol (TCP)",
                        "Domain Name System (DNS)"
                    ]
                )

        # 4. Check for Multiple Interpretations (Polysemous Domain Terms)
        for term, meta in self.DOMAIN_DISAMBIGUATION_MAP.items():
            if term in query_words:
                # Check if query is ALREADY qualified by domain qualifiers or compound expressions
                qualifiers = meta.get("qualifiers", set())
                is_qualified = any(q in lower_query for q in qualifiers) or any(
                    f"{q} {term}" in lower_query or f"{term} {q}" in lower_query for q in qualifiers
                )
                # If NOT qualified and query is short or ambiguous
                if not is_qualified:
                    return self._build_request(
                        original_query=raw_query,
                        clarification_type=ClarificationType.MULTIPLE_INTERPRETATIONS,
                        reason=meta["reason"],
                        follow_up=meta["follow_up"],
                        options=meta["options"]
                    )

        # 5. Check for Generic / Unclear Terminology lacking concrete subject
        for term in self.VAGUE_WORDS:
            if re.search(rf"\b(?:the\s+{term}|best\s+{term}|any\s+{term})\b", lower_query):
                has_concrete_entity = any(e in lower_query for e in self.KNOWN_DOMAIN_ENTITIES)
                if not has_concrete_entity and len(query_words) <= 5:
                    return self._build_request(
                        original_query=raw_query,
                        clarification_type=ClarificationType.AMBIGUOUS_TERM,
                        reason=f"The query refers generically to 'the {term}' without naming a specific technology or domain subject.",
                        follow_up=f"Could you specify which exact algorithm, protocol, or concept you would like to explore?",
                        options=[
                            "Support Vector Machines (Machine Learning)",
                            "TCP / UDP Transport Protocols (Networking)",
                            "Gradient Descent Optimization (Machine Learning)"
                        ]
                    )

        # 6. Check for single vague non-domain words
        words = re.findall(r"\b\w+\b", lower_query)
        if len(words) == 1:
            clean_word = words[0]
            if clean_word not in self.KNOWN_DOMAIN_ENTITIES and clean_word in self.VAGUE_WORDS:
                return self._build_request(
                    original_query=raw_query,
                    clarification_type=ClarificationType.INSUFFICIENT_INFORMATION,
                    reason=f"Single-word query '{clean_word}' is too brief and vague for accurate knowledge retrieval.",
                    follow_up=f"Would you like a definition, step-by-step procedure, or comparison regarding '{clean_word}'?",
                    options=[
                        f"What is {clean_word}? (Definition & Overview)",
                        f"How does {clean_word} work? (Step-by-step procedure)",
                        f"Compare {clean_word} with related technologies"
                    ]
                )

        return None

    def analyze_multi_part(self, raw_query: str) -> Optional[ClarificationRequest]:
        """
        Deconstruct multi-part queries into individual sub-requirements.
        Distinguishes related sub-questions that can be answered together from genuinely ambiguous ones.
        """
        query = raw_query.strip()
        parts: List[str] = []

        # Split on question marks, semicolons, comma-conjunctions, and explicit multi-part connectors
        temp_splits = re.split(r"\?\s*|\s*;\s*|,\s*(?:and\s+also|and\s+what\s+is|and\s+how\s+to|and\s+how|and\s+compare|and\s+)?", query, flags=re.IGNORECASE)
        for segment in temp_splits:
            segment = segment.strip(" ?. ")
            if not segment:
                continue
            # Further split on distinct conjunction clauses
            sub_segments = re.split(
                r"\b(?:and\s+also|and\s+additionally|as\s+well\s+as|furthermore|moreover)\b",
                segment,
                flags=re.IGNORECASE
            )
            for s in sub_segments:
                s_clean = s.strip(" ?. ")
                if len(s_clean) > 2:
                    parts.append(s_clean)

        if len(parts) <= 1:
            return None

        # Analyze each sub-part individually
        ambiguous_sub_parts: List[str] = []
        for p in parts:
            p_clean = p.strip().lower()
            # Check if any sub-part is incomplete (e.g. "how to", "compare", "tell me")
            if any(re.search(pat, p_clean) for pat in self.INCOMPLETE_PATTERNS) or len(p_clean.split()) <= 2 and p_clean in self.VAGUE_WORDS:
                ambiguous_sub_parts.append(p)
            # Check if any sub-part has unqualified polysemy
            elif any(t in p_clean.split() for t in self.DOMAIN_DISAMBIGUATION_MAP):
                # check if qualified
                p_words = set(re.findall(r"\b[a-z0-9_-]+\b", p_clean))
                for t, meta in self.DOMAIN_DISAMBIGUATION_MAP.items():
                    if t in p_words and not any(q in p_clean for q in meta.get("qualifiers", set())):
                        ambiguous_sub_parts.append(p)

        # Case A: If one or more sub-parts are genuinely ambiguous or incomplete
        if ambiguous_sub_parts:
            return self._build_request(
                original_query=raw_query,
                clarification_type=ClarificationType.MULTI_PART_UNRESOLVED,
                reason=f"Your multi-part query contains incomplete or ambiguous sub-question(s): '{', '.join(ambiguous_sub_parts)}'.",
                follow_up="Would you like to clarify the incomplete part, or focus on a specific sub-question first?",
                options=[
                    f"Focus only on: '{parts[0]}'",
                    "Provide clarification for the incomplete sub-question",
                    "Resolve all sub-questions sequentially"
                ],
                sub_questions=parts,
                is_multi_part=True
            )

        # Case B: If there are 3+ completely disjoint sub-questions across unrelated topics
        if len(parts) >= 3:
            # Check if sub-parts share common domain context or entities
            entity_sets = [set(re.findall(r"\b[a-z0-9_-]+\b", p.lower())) for p in parts]
            common_overlap = set.intersection(*entity_sets) if entity_sets else set()
            
            # If sub-parts have no conceptual overlap and span 3+ diverse questions
            if not common_overlap and len(parts) >= 3:
                return self._build_request(
                    original_query=raw_query,
                    clarification_type=ClarificationType.MULTI_PART_UNRESOLVED,
                    reason=f"Your query contains {len(parts)} independent sub-questions that span multiple distinct topics.",
                    follow_up="Would you like to resolve all sub-questions sequentially, or focus on a specific sub-question first?",
                    options=[
                        "Resolve all sub-questions comprehensively",
                        f"Focus first on: '{parts[0]}'",
                        f"Focus first on: '{parts[1]}'"
                    ],
                    sub_questions=parts,
                    is_multi_part=True
                )

        # Case C: Related multi-part queries that are clear (e.g. "What is TCP and what port does it use?") -> NO CLARIFICATION
        return None

    def handle(self, query_analysis: QueryAnalysisResult) -> AgentResponse:
        """
        Generate structured clarification AgentResponse and register pending state.
        """
        req = query_analysis.clarification_request
        if req is None:
            req = self.analyze_query(query_analysis.query) or self._build_request(
                original_query=query_analysis.query,
                clarification_type=ClarificationType.AMBIGUOUS_TERM,
                reason=query_analysis.reasoning or "Query requires clarification.",
                follow_up="Could you provide more specific details about your question?",
                options=[]
            )

        # Store in pending clarification register
        self.store_pending(req)

        # Format user-facing clarification message
        options_text = ""
        if req.suggested_options:
            options_text = "\n\n**Suggested Options:**\n" + "\n".join(f"- {opt}" for opt in req.suggested_options)

        full_message = f"**Clarification Needed:** {req.reason}\n\n👉 **{req.follow_up_question}**{options_text}"

        return AgentResponse(
            answer=full_message,
            confidence_score=0.0,
            confidence_level=ConfidenceLevel.NONE,
            sources=[],
            query_analysis=query_analysis,
            has_sufficient_evidence=False,
            requires_clarification=True,
            clarification_request=req
        )

    def store_pending(self, request: ClarificationRequest) -> None:
        """Maintain context while waiting for user's clarification response."""
        self._pending_clarifications[request.clarification_id] = request

    def get_pending(self, clarification_id: str) -> Optional[ClarificationRequest]:
        """Retrieve pending clarification request by ID."""
        return self._pending_clarifications.get(clarification_id)

    def resolve_clarification(
        self,
        clarification_id: Optional[str],
        user_response: Optional[str]
    ) -> Tuple[str, ClarificationRequest]:
        """
        Safely transition clarification from PENDING -> RESOLVED and return (refined_query, clarification_request).

        :raises ValueError: if clarification_id is invalid/missing, already resolved, or response is empty.
        """
        if not clarification_id or clarification_id not in self._pending_clarifications:
            raise ValueError(f"Clarification ID '{clarification_id}' is invalid or does not exist.")

        req = self._pending_clarifications[clarification_id]

        if req.status == ClarificationStatus.RESOLVED:
            raise ValueError(f"Clarification '{clarification_id}' has already been resolved and cannot be re-resolved.")

        if req.status == ClarificationStatus.CANCELLED:
            raise ValueError(f"Clarification '{clarification_id}' has been cancelled.")

        if not user_response or not user_response.strip():
            raise ValueError("Clarification response cannot be empty or whitespace only.")

        clean_resp = user_response.strip()
        refined = self.refine_query(req.original_query, clean_resp, req)

        # Update state lifecycle
        req.status = ClarificationStatus.RESOLVED
        req.resolved_at = time.time()
        req.user_response = clean_resp
        req.refined_query = refined

        return refined, req

    def cancel_clarification(self, clarification_id: str) -> Optional[ClarificationRequest]:
        """Cancel a pending clarification request."""
        if clarification_id in self._pending_clarifications:
            req = self._pending_clarifications[clarification_id]
            req.status = ClarificationStatus.CANCELLED
            return req
        return None

    def refine_query(
        self,
        original_query: str,
        user_response: str,
        clarification_request: Optional[ClarificationRequest] = None
    ) -> str:
        """
        Combine user's original query and clarification response into a refined, standalone query.
        Preserves original intent without brittle string substitution.
        """
        clean_orig = original_query.strip()
        clean_resp = user_response.strip()

        # Sanitize selected option prefixes & notes e.g. "📌 1. Decision Trees (Machine Learning)" -> "Decision Trees"
        clean_resp_text = re.sub(r"^[📌\s0-9]+[\.\)]\s*", "", clean_resp)
        clean_resp_text = re.sub(r"\s*\([^)]*\)", "", clean_resp_text).strip()

        # Case 1: Focus selection on multi-part query e.g. "Focus first on: 'What is TCP'"
        focus_match = re.search(r"focus\s+(?:first\s+|only\s+)?on:\s*['\"]?([^'\"]+)['\"]?", clean_resp, flags=re.IGNORECASE)
        if focus_match:
            return focus_match.group(1).strip()

        # Case 2: Polysemous/ambiguous domain keyword replacement e.g. orig: "Explain tree", resp: "Decision Trees"
        if clarification_request and clarification_request.clarification_type == ClarificationType.MULTIPLE_INTERPRETATIONS:
            # Find the polysemous term in the original query
            for term in self.DOMAIN_DISAMBIGUATION_MAP:
                if re.search(rf"\b{term}\b", clean_orig, flags=re.IGNORECASE):
                    # Replace the bare polysemous term with the specific disambiguated term
                    refined = re.sub(rf"\b{term}\b", clean_resp_text, clean_orig, flags=re.IGNORECASE)
                    return self._clean_refined(refined)

            prefix_match = re.match(r"^(explain|what\s+is|what\s+are|how\s+does|how\s+do|how\s+to|tell\s+me\s+about)\b", clean_orig, flags=re.IGNORECASE)
            if prefix_match:
                prefix = prefix_match.group(0).capitalize()
                if "how does" in prefix.lower():
                    return f"{prefix} {clean_resp_text} work?"
                return f"{prefix} {clean_resp_text}"
            return f"Explain {clean_resp_text}"

        # Case 3: Incomplete comparison e.g. orig: "Compare SVM", resp: "Logistic Regression"
        if re.search(r"\b(?:compare|vs|versus)\b", clean_orig, flags=re.IGNORECASE) and not re.search(r"\b(?:vs|versus|with|to)\b", clean_orig, flags=re.IGNORECASE):
            if clean_resp_text.lower().startswith(("vs", "versus", "to", "with", "and")):
                return self._clean_refined(f"{clean_orig} {clean_resp_text}")
            else:
                return self._clean_refined(f"{clean_orig} vs {clean_resp_text}")

        # Case 4: Missing context with pronouns e.g. "how does it work", resp: "TCP 3-way handshake"
        if any(re.search(p, clean_orig, flags=re.IGNORECASE) for p in self.PRONOUN_PATTERNS):
            if re.search(r"\b(?:how\s+does\s+it\s+work|how\s+it\s+works)\b", clean_orig, flags=re.IGNORECASE):
                return f"How does {clean_resp_text} work?"
            elif "advantages" in clean_orig.lower():
                return f"What are the advantages of {clean_resp_text}?"
            elif "explain" in clean_orig.lower():
                return f"Explain {clean_resp_text}"
            else:
                # Replace pronouns cleanly
                replaced = re.sub(r"\b(it|this|that|them)\b", clean_resp_text, clean_orig, flags=re.IGNORECASE)
                return self._clean_refined(replaced)

        # Case 5: Incomplete query e.g. "how to", resp: "configure TCP socket"
        if re.match(r"^(?:how\s+to|how\s+do\s+i|steps\s+to|explain)\b", clean_orig, flags=re.IGNORECASE) and len(clean_orig.split()) <= 3:
            return self._clean_refined(f"{clean_orig} {clean_resp_text}")

        # Case 6: Response already is a standalone question
        if re.match(r"^(?:what|how|why|which|explain|compare|define)\b", clean_resp_text, flags=re.IGNORECASE):
            return clean_resp_text

        # Case 7: General fusion
        return self._clean_refined(f"{clean_orig} ({clean_resp_text})")

    def _generate_comparison_options(self, entity: str) -> List[str]:
        """Dynamically generate domain-relevant comparison options for an entity."""
        lower_e = entity.lower()
        if lower_e in {"svm", "support vector machine", "decision tree", "random forest", "knn", "logistic regression"}:
            pool = ["Logistic Regression", "Random Forest", "K-Nearest Neighbors (KNN)", "Decision Trees"]
            return [opt for opt in pool if lower_e not in opt.lower()][:3]
        elif lower_e in {"tcp", "udp", "http", "https", "dns", "dhcp", "bgp", "ospf"}:
            pool = ["UDP (User Datagram Protocol)", "HTTP vs HTTPS", "SCTP", "QUIC"]
            return [opt for opt in pool if lower_e not in opt.lower()][:3]
        return ["Alternative Machine Learning Algorithm", "Alternative Network Protocol"]

    def _clean_refined(self, text: str) -> str:
        """Sanitizes refined query text (removes extra spaces, fixes punctuation)."""
        cleaned = re.sub(r"\s+", " ", text).strip()
        cleaned = re.sub(r"\s+([?.!,])", r"\1", cleaned)
        if cleaned and not cleaned.endswith(("?", ".")):
            # If query starts with question word, append '?'
            if re.match(r"^(?:what|how|why|which|who|where|when|can|do|does|is|are)\b", cleaned, flags=re.IGNORECASE):
                cleaned += "?"
        return cleaned

    def _build_request(
        self,
        original_query: str,
        clarification_type: ClarificationType,
        reason: str,
        follow_up: str,
        options: List[str],
        sub_questions: Optional[List[str]] = None,
        is_multi_part: bool = False
    ) -> ClarificationRequest:
        return ClarificationRequest(
            original_query=original_query,
            clarification_type=clarification_type,
            reason=reason,
            follow_up_question=follow_up,
            suggested_options=options,
            sub_questions=sub_questions or [],
            is_multi_part=is_multi_part,
            status=ClarificationStatus.PENDING
        )
