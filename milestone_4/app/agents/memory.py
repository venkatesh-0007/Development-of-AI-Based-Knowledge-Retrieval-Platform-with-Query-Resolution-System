"""Conversation Memory Agent with domain-aware context tracking and anaphora resolution."""
import re
import uuid
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Set

from .models import (
    ConversationTurn,
    ConversationSession
)

logger = logging.getLogger(__name__)

class ConversationMemoryAgent:
    """Manages multi-turn conversation sessions, anaphoric pronoun resolution, and cross-domain context isolation."""

    ANAPHORA_PATTERNS = [
        (r"\b(?:how\s+does\s+it\s+work|how\s+it\s+works)\b", "How does {entity} work?"),
        (r"\b(?:what\s+are\s+its\s+advantages|advantages\s+of\s+it)\b", "What are the advantages of {entity}?"),
        (r"\b(?:what\s+is\s+its\s+purpose|what\s+does\s+it\s+do)\b", "What is the purpose of {entity}?"),
        (r"\b(?:explain\s+it|describe\s+it|tell\s+me\s+more\s+about\s+it)\b", "Explain {entity}"),
        (r"\b(?:what\s+are\s+its\s+(?:main\s+)?splitting\s+criteria)\b", "What are the main splitting criteria of {entity}?"),
        (r"\b(?:how\s+does\s+it\s+ensure\s+reliable\s+delivery)\b", "How does {entity} ensure reliable delivery?"),
        (r"\b(?:how\s+does\s+it\s+prevent\s+overfitting)\b", "How does {entity} prevent overfitting?"),
        (r"\b(?:compare\s+it\s+with\s+([a-zA-Z0-9_\-\.]+))\b", "Compare {entity} with \\1"),
        (r"\b(?:compare\s+this\s+with\s+([a-zA-Z0-9_\-\.]+))\b", "Compare {entity} with \\1")
    ]

    def __init__(self, max_history_turns: int = 6):
        self.max_history_turns = max_history_turns
        self._sessions: Dict[str, ConversationSession] = {}

    def get_or_create_session(self, session_id: Optional[str] = None) -> ConversationSession:
        if not session_id:
            session_id = "session_" + uuid.uuid4().hex[:8]
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationSession(session_id=session_id)
        return self._sessions[session_id]

    def record_turn(self, session: ConversationSession, turn: ConversationTurn):
        session.add_turn(turn, max_history=self.max_history_turns)

    def clear_session(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]

    def _extract_primary_entity(self, turn: ConversationTurn) -> Optional[str]:
        if turn.key_entities:
            return turn.key_entities[0]
        # Fallback noun phrase heuristics
        match = re.search(r"\b(?:what is|explain|define|about)\s+([A-Za-z0-9\s\-_]+?)(?:\?|$)", turn.query, re.IGNORECASE)
        if match:
            entity = match.group(1).strip()
            if len(entity.split()) <= 4:
                return entity
        return None

    def contextualize_query(self, query: str, session: ConversationSession) -> str:
        """
        Rewrites anaphoric queries into standalone queries using prior session turns.
        Detects context switching and prevents cross-domain context bleeding.
        """
        clean = query.strip()
        lower = clean.lower()

        last_turn = session.get_last_turn()
        if not last_turn:
            return clean

        # Context switch check: if the query introduces a clear new entity or domain
        explicit_domains = {
            "tcp": "computer_networks", "udp": "computer_networks", "dns": "computer_networks", "osi": "computer_networks",
            "decision tree": "machine_learning", "svm": "machine_learning", "neural network": "machine_learning",
            "zero trust": "cybersecurity", "aes": "cybersecurity", "rsa": "cybersecurity", "cia triad": "cybersecurity"
        }
        for term, dom in explicit_domains.items():
            if re.search(rf"\b{re.escape(term)}\b", lower):
                # User is explicitly querying a new entity -> treat as new context
                return clean

        # Anaphora check
        entity = self._extract_primary_entity(last_turn)
        if not entity:
            return clean

        for pattern, replacement in self.ANAPHORA_PATTERNS:
            if re.search(pattern, lower):
                formatted = re.sub(pattern, replacement.format(entity=entity), clean, flags=re.IGNORECASE)
                return formatted

        # Generic pronoun replacement if dangling
        if re.search(r"\b(it|its|this|that|they|their)\b", lower):
            # If query is short (< 8 words)
            if len(clean.split()) <= 8:
                rewritten = re.sub(r"\bits\b", f"{entity}'s", clean, flags=re.IGNORECASE)
                rewritten = re.sub(r"\b(it|this|that)\b", entity, rewritten, flags=re.IGNORECASE)
                return rewritten

        return clean
