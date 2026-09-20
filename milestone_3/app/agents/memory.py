"""Conversation Memory Agent for Milestone 3.2 (M3.2).

Maintains short-term conversational context across multi-turn interactions.
Performs anaphora / pronoun resolution, contextual query rewriting,
topic continuation vs context switching detection, and sliding window pruning.
"""
import re
import uuid
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Set

from .models import (
    ConversationTurn,
    ConversationSession,
    AgentResponse,
    QueryType
)

logger = logging.getLogger(__name__)

class ConversationMemoryAgent:
    """Manages multi-turn conversation context, anaphora resolution, and session memory."""

    ANAPHORA_PATTERNS = [
        r"\b(?:how\s+does\s+(?:it|this|that|they)\s+work|how\s+to\s+use\s+(?:it|this|that)|how\s+do\s+i\s+run\s+(?:it|this|that))\b",
        r"\b(?:what\s+are\s+(?:its|their)\s+(?:advantages|disadvantages|features|properties|parameters|types|steps|applications))\b",
        r"\b(?:what\s+is\s+(?:its|their)\s+(?:purpose|function|use\s+case|role|definition|port))\b",
        r"^(?:tell\s+me\s+about\s+(?:it|this|that|them)|explain\s+(?:this|that|it)|what\s+about\s+(?:that|this|it)|give\s+me\s+more\s+information)\b",
        r"\b(?:compare\s+(?:it|that|this)\s+(?:with|to|vs|versus)\s+([a-zA-Z0-9_-]+))\b",
        r"\b(?:how\s+is\s+(?:it|this|that)\s+different\s+from\s+([a-zA-Z0-9_-]+))\b",
        r"\b(?:how\s+to\s+configure\s+(?:it|this|that))\b",
        r"\b(?:steps\s+for\s+(?:it|this|that))\b",
        r"\b(?:the\s+protocol|the\s+algorithm|the\s+model|the\s+technique|the\s+method)\b",
        r"\b(it|this|that|its|their|they|them)\b"
    ]

    STOPWORDS: Set[str] = {
        "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of",
        "with", "by", "from", "up", "about", "into", "over", "after", "is",
        "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "can", "could", "should", "would", "will", "shall",
        "what", "how", "why", "which", "who", "when", "where", "please", "tell",
        "me", "explain", "describe", "compare", "steps", "difference", "between"
    }

    def __init__(self, default_window_size: int = 5, max_history_turns: Optional[int] = None):
        """
        Initialize Conversation Memory Agent.
        :param default_window_size: Maximum recent turns passed to downstream agents.
        :param max_history_turns: Alias for default_window_size.
        """
        self.default_window_size = max_history_turns or default_window_size
        self.max_history_turns = self.default_window_size
        self._sessions: Dict[str, ConversationSession] = {}

    def get_or_create_session(self, session_id: Optional[str] = None) -> ConversationSession:
        """Retrieve existing session or instantiate a new ConversationSession."""
        if not session_id:
            session_id = str(uuid.uuid4())[:8]
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationSession(
                session_id=session_id,
                max_turns=self.default_window_size
            )
        return self._sessions[session_id]

    def contextualize_query(
        self,
        query: str,
        session: ConversationSession,
        return_details: bool = False
    ) -> Any:
        """
        Analyze current user query in the context of session history.
        Resolves anaphoric pronouns and implicit references into a clear standalone query.

        :param query: Raw user query
        :param session: Active conversation session
        :param return_details: If True, returns (contextualized_query, is_follow_up, resolved_referent); otherwise string
        :return: contextualized_query (str) or (str, bool, Optional[str])
        """
        clean_query = query.strip()
        if not clean_query:
            return (clean_query, False, None) if return_details else clean_query

        # If no conversation history exists, return query directly
        if not session.turns:
            return (clean_query, False, None) if return_details else clean_query

        last_turn = session.get_last_turn()
        last_entity = self._extract_primary_entity_from_turn(last_turn)

        # 1. Detect if query is a standalone new topic (Context Switch)
        is_context_switch, new_topic = self._check_context_switch(clean_query, session)
        if is_context_switch:
            logger.info(f"Context switch detected to '{new_topic}' (previous: '{last_entity}')")
            return (clean_query, False, None) if return_details else clean_query

        # 2. Check for explicit anaphora / follow-up patterns
        is_anaphoric, pattern_type = self._detect_anaphora(clean_query)
        if not is_anaphoric or not last_entity:
            return (clean_query, False, None) if return_details else clean_query

        # 3. Perform Contextual Rewriting
        rewritten = self._rewrite_with_entity(clean_query, last_entity)
        if return_details:
            return (rewritten, True, last_entity)
        return rewritten

    def record_turn(
        self,
        session: ConversationSession,
        user_query_or_turn: Any,
        contextualized_query: Optional[str] = None,
        response: Optional[Any] = None,
        entities: Optional[List[str]] = None,
        referenced_docs: Optional[List[str]] = None,
        intent_type: QueryType = QueryType.FACTUAL
    ) -> ConversationTurn:
        """Record a completed conversation turn into session memory."""
        if isinstance(user_query_or_turn, ConversationTurn):
            turn = user_query_or_turn
            session.add_turn(turn)
            return turn

        user_query = user_query_or_turn
        effective_q = contextualized_query or user_query
        
        extracted_entities = entities
        if extracted_entities is None and response and hasattr(response, "query_analysis") and response.query_analysis:
            extracted_entities = response.query_analysis.entities
        if not extracted_entities:
            extracted_entities = self._extract_entities_from_text(user_query)

        docs = referenced_docs
        if docs is None and response and hasattr(response, "sources") and response.sources:
            docs = [s.document_name for s in response.sources]
        elif docs is None:
            docs = []

        turn = ConversationTurn(
            turn_id=str(uuid.uuid4())[:8],
            user_query=user_query,
            contextualized_query=effective_q,
            response=response,
            entities=extracted_entities or [],
            referenced_docs=docs,
            intent_type=intent_type,
            timestamp=time.time()
        )
        session.add_turn(turn)
        return turn

    def get_context_summary(self, session: ConversationSession, k: Optional[int] = None) -> Dict[str, Any]:
        """
        Extract recent conversational context summary with token/window budgeting.
        """
        window = k or self.default_window_size
        recent_turns = session.get_recent_turns(window)
        
        turn_summaries = []
        for t in recent_turns:
            turn_summaries.append({
                "turn_id": t.turn_id,
                "user_query": t.user_query,
                "contextualized_query": t.contextualized_query,
                "entities": t.entities,
                "referenced_docs": t.referenced_docs,
                "intent_type": t.intent_type.value
            })

        return {
            "session_id": session.session_id,
            "total_turns": len(session.turns),
            "recent_turn_count": len(recent_turns),
            "primary_topic": session.primary_topic,
            "entity_history": session.entity_history,
            "turns": turn_summaries
        }

    def clear_session(self, session_id: str) -> bool:
        """Clear conversation memory for a specific session."""
        if session_id in self._sessions:
            self._sessions[session_id].clear()
            return True
        return False

    def _detect_anaphora(self, query: str) -> Tuple[bool, str]:
        """Check if query contains unresolved pronouns or deictic expressions without antecedent."""
        lower_q = query.lower()
        
        # Check specific phrase templates
        if re.search(r"\b(?:what\s+are\s+(?:its|their))\b", lower_q):
            return True, "its_attributes"
        if re.search(r"\b(?:how\s+does\s+(?:it|this|that))\b", lower_q):
            return True, "how_it_does"
        if re.search(r"\b(?:how\s+to\s+(?:use|configure|run|implement)\s+(?:it|this|that))\b", lower_q):
            return True, "how_to_use"
        if re.search(r"\b(?:compare\s+(?:it|that|this)\s+(?:with|to|vs|versus))\b", lower_q):
            return True, "compare_it"
        if re.search(r"\b(?:what\s+about\s+(?:it|this|that)|tell\s+me\s+more\s+about\s+(?:it|this|that)|explain\s+(?:it|this|that))\b", lower_q):
            return True, "tell_me_more"
        if re.search(r"\b(?:the\s+protocol|the\s+algorithm|the\s+model)\b", lower_q) and len(lower_q.split()) <= 4:
            return True, "generic_noun"

        # Check for any pronoun
        if re.search(r"\b(it|this|that|its|their|they|them)\b", lower_q):
            return True, "pronoun"

        return False, "none"

    def _check_context_switch(self, query: str, session: ConversationSession) -> Tuple[bool, Optional[str]]:
        """
        Determines if the query introduces a brand new, fully-specified concept/topic,
        indicating a context switch rather than a follow-up.
        """
        entities = self._extract_entities_from_text(query)
        if not entities:
            return False, None

        last_turn = session.get_last_turn()
        if not last_turn:
            return False, None

        last_entities = {e.lower() for e in last_turn.entities}
        current_entities = {e.lower() for e in entities}

        # If current query explicitly names a concrete domain entity that differs from last turn
        # and has a complete grammatical question structure (e.g. "What is Decision Tree?"), it's a context switch.
        if current_entities and not current_entities.intersection(last_entities):
            # Check if query is complete and doesn't contain anaphoric pronouns
            if not any(re.search(rf"\b{p}\b", query.lower()) for p in ["it", "its", "this", "that", "they", "them"]):
                return True, entities[0]

        return False, None

    def _rewrite_with_entity(self, query: str, entity: str) -> str:
        """Rewrites an anaphoric query into an explicit, disambiguated query containing the entity."""
        lower_q = query.lower()

        # 1. Attributes / Advantages e.g. "What are its key advantages?" -> "What are the key advantages of Entity?"
        attr_match = re.search(r"\bwhat\s+are\s+(?:its|their)\s+([a-zA-Z0-9_\s]+?)\??$", query, flags=re.IGNORECASE)
        if attr_match:
            rest = attr_match.group(1).strip()
            return f"What are the {rest} of {entity}?"

        purpose_match = re.search(r"\bwhat\s+is\s+(?:its|their)\s+([a-zA-Z0-9_\s]+?)\??$", query, flags=re.IGNORECASE)
        if purpose_match:
            rest = purpose_match.group(1).strip()
            return f"What is the {rest} of {entity}?"

        # 2. How it works / does something e.g. "How does it establish a connection?" -> "How does Entity establish a connection?"
        how_does_match = re.search(r"\bhow\s+does\s+(?:it|this|that)\s+([a-zA-Z0-9_\s]+?)\??$", query, flags=re.IGNORECASE)
        if how_does_match:
            rest = how_does_match.group(1).strip()
            return f"How does {entity} {rest}?"

        if re.search(r"\bhow\s+to\s+use\s+it\b", lower_q) or re.search(r"\bhow\s+do\s+i\s+use\s+it\b", lower_q):
            return f"How to use {entity}?"
        if re.search(r"\bhow\s+to\s+configure\s+it\b", lower_q) or re.search(r"\bhow\s+do\s+i\s+configure\s+it\b", lower_q):
            return f"How to configure {entity}?"

        # 3. Comparisons e.g. "Compare it to UDP"
        comp_match = re.search(r"\b(?:compare\s+(?:it|this|that)\s+(?:with|to|vs|versus)\s+([a-zA-Z0-9_-]+))\b", query, flags=re.IGNORECASE)
        if comp_match:
            target = comp_match.group(1)
            return f"Compare {entity} vs {target}"

        diff_match = re.search(r"\b(?:how\s+is\s+(?:it|this|that)\s+different\s+from\s+([a-zA-Z0-9_-]+))\b", query, flags=re.IGNORECASE)
        if diff_match:
            target = diff_match.group(1)
            return f"How is {entity} different from {target}?"

        # 4. Explain this / Tell me about it
        if re.search(r"^(?:explain\s+(?:this|that|it)|tell\s+me\s+about\s+(?:it|this|that))\b", lower_q):
            return f"Explain {entity}"

        # 5. Generic noun substitution e.g. "How does the protocol establish connection?"
        for generic in ["the protocol", "the algorithm", "the model", "the method"]:
            if generic in lower_q:
                replaced = re.sub(rf"\b{generic}\b", entity, query, flags=re.IGNORECASE)
                return self._clean_query_text(replaced)

        # 6. Direct pronoun replacement
        replaced = re.sub(r"\b(its|their)\b", f"{entity}'s", query, flags=re.IGNORECASE)
        replaced = re.sub(r"\b(it|this|that)\b", entity, replaced, flags=re.IGNORECASE)
        return self._clean_query_text(replaced)


    def _extract_primary_entity_from_turn(self, turn: Optional[ConversationTurn]) -> Optional[str]:
        """Extract the most relevant entity subject from a prior turn."""
        if not turn:
            return None

        # 1. Check if user_query has a clear interrogative subject pattern
        q = turn.user_query.strip()
        subj_match = re.search(r"^(?:what\s+is\s+(?:the\s+)?|what\s+are\s+(?:the\s+)?|explain\s+|tell\s+me\s+about\s+|describe\s+)(.+?)(?:\s+in\s+.*|\?|\.|$)", q, flags=re.IGNORECASE)
        if subj_match:
            candidate = subj_match.group(1).strip()
            if candidate and candidate.lower() not in self.STOPWORDS and len(candidate) > 2:
                return candidate

        # 2. If turn has entities, check if the first entity is a multi-word or valid entity
        if turn.entities:
            # If all entities can form a compound sequence from query, prefer the compound
            for e in turn.entities:
                if len(e.split()) > 1:
                    return e
            return turn.entities[0]

        # 3. Fallback: extract entities from text
        extracted = self._extract_entities_from_text(turn.user_query)
        if extracted:
            return extracted[0]
        return None

    def _extract_entities_from_text(self, text: str) -> List[str]:
        """Extract candidate noun entities from text."""
        # Check for parentheses entities first e.g. (TCP), (SVM)
        paren_match = re.findall(r"\(([^)]+)\)", text)
        
        cleaned = re.sub(r"[^\w\s-]", " ", text)
        tokens = cleaned.split()
        entities = [t for t in tokens if t.lower() not in self.STOPWORDS and len(t) > 1]
        
        seen = set()
        deduped = []
        for p in paren_match:
            if p.lower() not in seen and len(p) > 1:
                seen.add(p.lower())
                deduped.append(p)

        for e in entities:
            if e.lower() not in seen:
                seen.add(e.lower())
                deduped.append(e)
        return deduped

    def _clean_query_text(self, text: str) -> str:
        """Sanitizes spacing and capitalization."""
        cleaned = re.sub(r"\s+", " ", text).strip()
        cleaned = re.sub(r"\s+([?.!,])", r"\1", cleaned)
        if cleaned and not cleaned.endswith(("?", ".")):
            if re.match(r"^(?:what|how|why|which|who|where|when|can|do|does|is|are)\b", cleaned, flags=re.IGNORECASE):
                cleaned += "?"
        return cleaned
