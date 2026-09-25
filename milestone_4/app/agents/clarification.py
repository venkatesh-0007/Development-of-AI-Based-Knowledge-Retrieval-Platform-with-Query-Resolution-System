"""Clarification Agent for Milestone 4 (M4.1/M4.3).

Intelligently identifies ambiguous, incomplete, or multi-part queries across multiple knowledge domains.
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
    """Production-grade Clarification Agent implementing targeted disambiguation."""

    # Domain Disambiguation Catalog for polysemous concepts across 3 domains
    DOMAIN_DISAMBIGUATION_MAP: Dict[str, Dict[str, Any]] = {
        "tree": {
            "reason": "The term 'tree' is polysemous and can refer to Decision Trees in Machine Learning, Spanning Tree Protocol in Networking, or Tree Data Structures.",
            "follow_up": "Which type of tree are you referring to?",
            "options": [
                "Decision Trees (Machine Learning / Classification)",
                "Spanning Tree Protocol (Networking / STP)",
                "Binary Search Tree / Merkle Tree (Data Structures / Security)"
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
                "High-Availability Server / Cloud Distributed Clustering"
            ],
            "qualifiers": {
                "k-means", "kmeans", "hierarchical", "dbscan", "density", "spectral",
                "gaussian", "gmm", "server", "failover", "database", "node", "ha"
            }
        },
        "handshake": {
            "reason": "The term 'handshake' could refer to the TCP 3-Way Handshake, TLS/SSL cryptographic security handshake, or WebSocket connection handshake.",
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
        "token": {
            "reason": "The term 'token' can refer to LLM/NLP linguistic tokens, OAuth/JWT security authentication tokens, or Token Ring network protocols.",
            "follow_up": "Which domain token are you referring to?",
            "options": [
                "LLM / NLP Subword Tokenization (Deep Learning)",
                "JWT / OAuth Security Bearer Token (Cybersecurity)",
                "Token Ring Network Topology (Networking)"
            ],
            "qualifiers": {
                "nlp", "llm", "jwt", "oauth", "bearer", "subword", "byte-pair", "ring",
                "vocabulary", "session", "auth", "access"
            }
        },
        "signature": {
            "reason": "The term 'signature' could mean Cryptographic Digital Signatures (RSA/ECDSA), Antivirus/IDS Threat Signatures, or Function Type Signatures.",
            "follow_up": "Which type of signature are you inquiring about?",
            "options": [
                "Cryptographic Digital Signatures (RSA, ECDSA)",
                "Malware / Intrusion Detection Attack Signatures (Snort, YARA)",
                "Programming Language Function / Method Signatures"
            ],
            "qualifiers": {
                "digital", "crypto", "rsa", "ecdsa", "hash", "malware", "virus", "yara",
                "snort", "ids", "method", "function", "type"
            }
        },
        "injection": {
            "reason": "The term 'injection' can refer to SQL/Command Injection in Cybersecurity, Prompt Injection in LLMs, or Dependency Injection in Software Engineering.",
            "follow_up": "Which form of injection do you want to explore?",
            "options": [
                "SQL Injection / Command Injection (Web Vulnerability)",
                "Prompt Injection (LLM / AI Security)",
                "Dependency Injection (Software Architecture)"
            ],
            "qualifiers": {
                "sql", "sqli", "command", "prompt", "jailbreak", "llm", "dependency",
                "framework", "container", "ioc"
            }
        }
    }

    # Entity hints for incomplete comparison requests
    COMPARISON_PAIRS: Dict[str, List[str]] = {
        "tcp": ["UDP (User Datagram Protocol)", "SCTP", "QUIC"],
        "udp": ["TCP (Transmission Control Protocol)", "RTP"],
        "supervised": ["Unsupervised Learning", "Reinforcement Learning", "Semi-Supervised Learning"],
        "unsupervised": ["Supervised Learning", "Self-Supervised Learning"],
        "aes": ["RSA", "DES / 3DES", "ChaCha20"],
        "rsa": ["AES", "Elliptic Curve Cryptography (ECC)", "Diffie-Hellman"],
        "cnn": ["Recurrent Neural Networks (RNN)", "Vision Transformers (ViT)"],
        "rnn": ["LSTM", "Transformers", "CNN"],
        "bert": ["GPT", "T5", "RoBERTa"]
    }

    def __init__(self):
        self._clarification_registry: Dict[str, ClarificationRequest] = {}

    def detect_clarification_need(
        self,
        query: str,
        active_domain: Optional[str] = None
    ) -> Tuple[bool, Optional[ClarificationRequest]]:
        """
        Determines whether a query requires clarification based on ambiguity or incompleteness.
        Avoids false positives when queries contain sufficient domain context.
        """
        clean = query.strip()
        lower = clean.lower()

        # Check 1: Extremely short / dangling query without context
        tokens = re.findall(r"\w+", lower)
        if len(tokens) <= 1:
            term = tokens[0] if tokens else ""
            if term in self.DOMAIN_DISAMBIGUATION_MAP:
                mapping = self.DOMAIN_DISAMBIGUATION_MAP[term]
                req = ClarificationRequest(
                    original_query=query,
                    clarification_type=ClarificationType.AMBIGUOUS_TERM,
                    reason=mapping["reason"],
                    follow_up_question=mapping["follow_up"],
                    suggested_options=list(mapping["options"])
                )
                self._clarification_registry[req.clarification_id] = req
                return True, req
            elif len(term) >= 2:
                req = ClarificationRequest(
                    original_query=query,
                    clarification_type=ClarificationType.INCOMPLETE_REQUEST,
                    reason=f"The query consists only of the isolated keyword '{term}'.",
                    follow_up_question=f"Could you specify what you would like to know about '{term}'? (e.g. definition, architecture, comparison, or implementation)",
                    suggested_options=[
                        f"Definition & Overview of {term}",
                        f"How {term} works (Step-by-step)",
                        f"Key advantages and use cases of {term}"
                    ]
                )
                self._clarification_registry[req.clarification_id] = req
                return True, req

        # Check 2: Polysemous terms without domain qualifiers
        for term, data in self.DOMAIN_DISAMBIGUATION_MAP.items():
            pattern = rf"\b{re.escape(term)}\b"
            if re.search(pattern, lower):
                # If qualified by domain words, do NOT trigger clarification
                has_qualifier = any(q in lower for q in data["qualifiers"])
                if not has_qualifier and len(tokens) <= 4:
                    req = ClarificationRequest(
                        original_query=query,
                        clarification_type=ClarificationType.AMBIGUOUS_TERM,
                        reason=data["reason"],
                        follow_up_question=data["follow_up"],
                        suggested_options=list(data["options"])
                    )
                    self._clarification_registry[req.clarification_id] = req
                    return True, req

        # Check 3: Dangling pronouns without antecedent in query
        dangling_pronouns = ["how does it work", "what is it", "explain it", "what are its advantages", "compare it"]
        for dp in dangling_pronouns:
            if lower.startswith(dp) or lower == dp + "?":
                # Check if query mentions an antecedent noun
                after_pronoun = lower.replace(dp, "").strip(" ?.")
                if not after_pronoun or len(after_pronoun.split()) <= 1:
                    req = ClarificationRequest(
                        original_query=query,
                        clarification_type=ClarificationType.MISSING_CONTEXT,
                        reason="The query uses a reference pronoun ('it' / 'its') without specifying the target entity.",
                        follow_up_question="Which specific protocol, model, or technology are you asking about?",
                        suggested_options=[
                            "TCP (Transmission Control Protocol)",
                            "Supervised Machine Learning",
                            "Zero Trust Security Architecture"
                        ]
                    )
                    self._clarification_registry[req.clarification_id] = req
                    return True, req

        # Check 4: Incomplete comparison requests (e.g. "Compare TCP" or "Difference between AES")
        comp_match = re.search(r"\b(compare|difference between|versus|vs)\s+([a-zA-Z0-9_\-\.]+)\s*$", lower)
        if comp_match:
            entity = comp_match.group(2).lower()
            if entity in self.COMPARISON_PAIRS:
                options = [f"Compare {entity.upper()} with {pair}" for pair in self.COMPARISON_PAIRS[entity]]
                req = ClarificationRequest(
                    original_query=query,
                    clarification_type=ClarificationType.INCOMPLETE_REQUEST,
                    reason=f"The query asks to compare '{entity.upper()}' without specifying the second entity.",
                    follow_up_question=f"Which entity would you like to compare {entity.upper()} against?",
                    suggested_options=options
                )
                self._clarification_registry[req.clarification_id] = req
                return True, req

        return False, None

    def handle(self, query_analysis: QueryAnalysisResult) -> AgentResponse:
        """Constructs an AgentResponse prompting the user for clarification."""
        req = query_analysis.clarification_request
        if not req:
            req = ClarificationRequest(
                original_query=query_analysis.query,
                clarification_type=ClarificationType.AMBIGUOUS_TERM,
                reason="The query contains ambiguous terminology that requires disambiguation.",
                follow_up_question="Could you clarify your request?"
            )

        spoken_msg = req.follow_up_question
        return AgentResponse(
            answer=f"⚠️ **Clarification Needed**: {req.follow_up_question}\n\n*Reason*: {req.reason}",
            confidence_score=0.0,
            confidence_level=ConfidenceLevel.NONE,
            sources=[],
            query_analysis=query_analysis,
            has_sufficient_evidence=False,
            requires_clarification=True,
            clarification_request=req,
            spoken_text=spoken_msg
        )

    def resolve_clarification(
        self,
        clarification_id: str,
        user_response: str
    ) -> Tuple[str, ClarificationRequest]:
        """
        Transition state: PENDING -> RESOLVED.
        Synthesizes a refined, standalone query fusing original intent and user answer.
        """
        clean_resp = user_response.strip()
        if not clean_resp:
            raise ValueError("Clarification response cannot be empty or whitespace only.")

        req = self._clarification_registry.get(clarification_id)
        if not req:
            raise ValueError(f"Clarification ID '{clarification_id}' is invalid or does not exist.")

        if req.status == ClarificationStatus.RESOLVED:
            raise ValueError(f"Clarification '{clarification_id}' has already been resolved.")

        # Strip option prefixes (e.g., "1.", "Decision Trees (ML) -> Decision Trees")
        sanitized_resp = re.sub(r"^\d+[\.\)]\s*", "", clean_resp).strip()
        sanitized_resp = re.sub(r"\s*\([^)]*\)", "", sanitized_resp).strip()

        orig = req.original_query.strip()
        orig_lower = orig.lower()

        # Query fusion logic
        if re.search(r"\b(how does it work|how it works)\b", orig_lower):
            refined = f"How does {sanitized_resp} work?"
        elif re.search(r"\b(what is it|what is this)\b", orig_lower):
            refined = f"What is {sanitized_resp}?"
        elif re.search(r"\b(what are its advantages|advantages of it)\b", orig_lower):
            refined = f"What are the advantages of {sanitized_resp}?"
        elif re.search(r"\b(explain it|describe it)\b", orig_lower):
            refined = f"Explain {sanitized_resp}"
        elif re.search(r"\b(compare it with|compare with)\b", orig_lower):
            refined = re.sub(r"\bcompare it with\b", f"Compare {sanitized_resp} with", orig, flags=re.IGNORECASE)
        elif orig_lower.startswith("compare ") and len(orig.split()) <= 2:
            refined = f"{orig} with {sanitized_resp}"
        else:
            # Polysemous substitution or appendage
            replaced = False
            for term in self.DOMAIN_DISAMBIGUATION_MAP:
                if re.search(rf"\b{term}\b", orig_lower):
                    refined = re.sub(rf"\b{term}\b", sanitized_resp, orig, flags=re.IGNORECASE)
                    replaced = True
                    break
            if not replaced:
                refined = f"{orig} ({sanitized_resp})"

        # Clean punctuation
        refined = re.sub(r"\s+", " ", refined).strip()

        req.status = ClarificationStatus.RESOLVED
        req.resolved_at = time.time()
        req.user_response = user_response
        req.refined_query = refined

        return refined, req

    def get_request(self, clarification_id: str) -> Optional[ClarificationRequest]:
        return self._clarification_registry.get(clarification_id)
