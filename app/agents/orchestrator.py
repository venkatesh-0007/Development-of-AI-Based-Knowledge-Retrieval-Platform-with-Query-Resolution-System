"""Simple multi-agent orchestration for Milestone 1."""
from .agents import (
    QueryUnderstandingAgent, RetrievalAgent,
    ResponseGenerationAgent, ClarificationAgent,
    ConversationMemoryAgent
)

class Orchestrator:
    def __init__(self, retrieval_pipeline, llm_client=None):
        self.query_agent = QueryUnderstandingAgent()
        self.retrieval_agent = RetrievalAgent(retrieval_pipeline)
        self.response_agent = ResponseGenerationAgent(llm_client)
        self.clarification_agent = ClarificationAgent()
        self.memory_agent = ConversationMemoryAgent()

    def run(self, query, top_k=5):
        if not query or not query.strip():
            return {"answer": self.clarification_agent.run(query), "sources": []}

        understood = self.query_agent.run(query)
        results = self.retrieval_agent.run(understood["query"], top_k)
        answer = self.response_agent.run(query, results)
        self.memory_agent.add(query, answer)
        return {"answer": answer, "sources": results}
