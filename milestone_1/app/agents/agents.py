"""M1 agent roles. Kept intentionally lightweight and composable."""
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class QueryUnderstandingAgent:
    def run(self, query: str) -> Dict[str, Any]:
        return {"query": query.strip(), "intent": "information_retrieval"}

@dataclass
class RetrievalAgent:
    retrieval_pipeline: Any
    def run(self, query: str, top_k: int = 5):
        return self.retrieval_pipeline.retrieve(query, top_k)

@dataclass
class ResponseGenerationAgent:
    llm_client: Any = None
    def run(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        if not contexts:
            return "I could not find relevant information in the knowledge base."
        if self.llm_client is None:
            return "\n\n".join(
                f"[{c['metadata'].get('source', 'source')}]\n{c['content']}"
                for c in contexts
            )
        return self.llm_client.generate(query, contexts)

@dataclass
class ClarificationAgent:
    def run(self, query: str) -> str:
        return "Please provide more specific details about what you want to retrieve."

@dataclass
class ConversationMemoryAgent:
    history: List[Dict[str, str]] = field(default_factory=list)
    def add(self, query: str, response: str):
        self.history.append({"query": query, "response": response})
    def get_history(self):
        return self.history
