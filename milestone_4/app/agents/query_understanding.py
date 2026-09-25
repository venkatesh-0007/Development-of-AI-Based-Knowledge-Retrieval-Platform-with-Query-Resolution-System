"""Query Understanding Agent with multi-domain intent classification and ambiguity detection."""
import re
from typing import List, Optional

from .models import (
    QueryType,
    RoutingTarget,
    QueryAnalysisResult
)
from .clarification import ClarificationAgent

class QueryUnderstandingAgent:
    """Classifies user queries by intent, domain, and entity focus while managing ambiguity detection."""

    DOMAIN_KEYWORDS = {
        "machine_learning": [
            "supervised", "unsupervised", "reinforcement", "neural", "deep learning",
            "transformer", "attention", "bert", "gpt", "svm", "support vector",
            "decision tree", "random forest", "k-means", "pca", "gradient boosting",
            "overfitting", "loss function", "backpropagation", "regularization",
            "precision", "recall", "f1-score", "auc-roc", "llm", "perceptron"
        ],
        "computer_networks": [
            "network", "networking", "osi", "tcp", "udp", "ip", "ipv4", "ipv6",
            "dns", "http", "https", "ssh", "bgp", "icmp", "handshake", "router",
            "packet", "transport layer", "application layer", "cloud", "iaas", "paas",
            "saas", "microservices", "docker", "kubernetes", "cap theorem", "distributed"
        ],
        "cybersecurity": [
            "security", "cyber", "cybersecurity", "cia triad", "confidentiality",
            "integrity", "availability", "defense in depth", "zero trust", "least privilege",
            "assume breach", "aes", "rsa", "sha-256", "diffie-hellman", "cryptography",
            "encryption", "digital signature", "mfa", "phishing", "sqli", "sql injection",
            "xss", "cross-site scripting", "ransomware", "incident response", "nist",
            "mitre", "siem", "soc", "ioc"
        ]
    }

    def __init__(self, clarification_agent: Optional[ClarificationAgent] = None):
        self.clarification_agent = clarification_agent or ClarificationAgent()

    def detect_domain(self, text: str) -> str:
        lower = text.lower()
        domain_scores = {d: 0 for d in self.DOMAIN_KEYWORDS}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            for kw in keywords:
                if re.search(rf"\b{re.escape(kw)}\b", lower):
                    domain_scores[domain] += 1
        
        best_domain, max_score = max(domain_scores.items(), key=lambda x: x[1])
        return best_domain if max_score > 0 else "general"

    def analyze(self, query: str, active_domain: Optional[str] = None) -> QueryAnalysisResult:
        cleaned = re.sub(r"\s+", " ", query).strip()
        lower = cleaned.lower()

        # Step 1: Detect Domain
        detected_from_query = self.detect_domain(cleaned)
        if detected_from_query != "general":
            detected_domain = detected_from_query
        elif active_domain and active_domain != "general":
            detected_domain = active_domain
        else:
            detected_domain = "general"

        # Step 2: Ambiguity / Clarification check
        needs_clarification, clar_req = self.clarification_agent.detect_clarification_need(
            query=cleaned,
            active_domain=detected_domain
        )

        if needs_clarification and clar_req:
            return QueryAnalysisResult(
                query=query,
                cleaned_query=cleaned,
                query_type=QueryType.AMBIGUOUS,
                domain=detected_domain,
                classification_confidence=0.95,
                route_to=RoutingTarget.CLARIFICATION,
                reasoning=f"Query is ambiguous ({clar_req.clarification_type.value}): {clar_req.reason}",
                entities=self._extract_entities(cleaned),
                requires_clarification=True,
                clarification_request=clar_req
            )

        # Step 3: Intent Classification
        sub_questions = self._extract_sub_questions(cleaned)
        is_multi_part = len(sub_questions) > 1

        if is_multi_part:
            query_type = QueryType.MULTI_PART
            reasoning = f"Multi-part query containing {len(sub_questions)} distinct inquiries."
            confidence = 0.90
        elif re.search(r"\b(compare|difference between|versus|vs|pros and cons|advantages of .+ over)\b", lower):
            query_type = QueryType.COMPARATIVE
            reasoning = "Comparative query evaluating differences, trade-offs, or contrasts."
            confidence = 0.92
        elif re.search(r"\b(how to|steps to|how does .+ work|process of|lifecycle|procedure|handshake)\b", lower):
            query_type = QueryType.PROCEDURAL
            reasoning = "Procedural query requesting sequential steps, mechanisms, or workflows."
            confidence = 0.91
        else:
            query_type = QueryType.FACTUAL
            reasoning = "Factual query seeking direct conceptual information or attributes."
            confidence = 0.88

        entities = self._extract_entities(cleaned)

        return QueryAnalysisResult(
            query=query,
            cleaned_query=cleaned,
            query_type=query_type,
            domain=detected_domain,
            classification_confidence=confidence,
            route_to=RoutingTarget.RETRIEVAL,
            reasoning=reasoning,
            entities=entities,
            requires_clarification=False,
            sub_questions=sub_questions
        )

    def _extract_entities(self, text: str) -> List[str]:
        # Extract potential acronyms, technical terms, and noun phrases
        STOP_WORDS = {
            "what", "how", "why", "which", "where", "when", "who", "whom",
            "explain", "compare", "describe", "tell", "does", "do", "did",
            "can", "could", "is", "are", "was", "were", "please", "step", "steps"
        }
        candidates = re.findall(r"\b[A-Z0-9]{2,}\b|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        lower = text.lower()
        known_techs = [
            "tcp", "udp", "ip", "dns", "http", "https", "ssh", "bgp", "osi",
            "supervised learning", "unsupervised learning", "reinforcement learning",
            "decision tree", "random forest", "svm", "support vector machine", "neural network", "transformer",
            "bert", "gpt", "cia triad", "zero trust", "aes", "rsa", "sha-256", "diffie-hellman",
            "phishing", "sql injection", "ransomware", "microservices", "kubernetes", "docker"
        ]
        found = [c for c in candidates if c.lower() not in STOP_WORDS]
        for tech in known_techs:
            if re.search(rf"\b{re.escape(tech)}\b", lower):
                found.append(tech.upper())
        return list(dict.fromkeys(found))

    def _extract_sub_questions(self, text: str) -> List[str]:
        parts = [
            p.strip()
            for p in re.split(r"\?+|\band\s+(?:how|what|why|which|where|when|can|do|does|also)\b|\bmoreover\b|;\s*", text, flags=re.IGNORECASE)
            if len(p.strip()) > 3
        ]
        return parts if len(parts) > 1 else [text]
