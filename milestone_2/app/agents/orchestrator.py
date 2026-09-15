"""Multi-Agent Orchestrator for Milestone 2 (M2.4).

Coordinates the sequential query pipeline:
User Query -> Query Understanding -> (Clarification) -> Retrieval -> Response Generation -> Final Response.
"""
import time
import logging
from typing import Optional, Any
from .models import (
    RoutingTarget,
    QueryAnalysisResult,
    RetrievalResult,
    AgentResponse,
    OrchestrationResult,
    ConfidenceLevel
)
from .query_understanding import QueryUnderstandingAgent
from .retrieval import RetrievalAgent
from .response_generation import ResponseGenerationAgent
from .clarification import ClarificationAgent

logger = logging.getLogger(__name__)

class MultiAgentOrchestrator:
    """Coordinates multi-agent query understanding, retrieval, and grounded response generation."""

    def __init__(
        self,
        retrieval_agent: RetrievalAgent,
        query_understanding_agent: Optional[QueryUnderstandingAgent] = None,
        response_generation_agent: Optional[ResponseGenerationAgent] = None,
        clarification_agent: Optional[ClarificationAgent] = None
    ):
        self.query_understanding_agent = query_understanding_agent or QueryUnderstandingAgent()
        self.retrieval_agent = retrieval_agent
        self.response_generation_agent = response_generation_agent or ResponseGenerationAgent()
        self.clarification_agent = clarification_agent or ClarificationAgent()

    def run(
        self,
        query: str,
        top_k: Optional[int] = None,
        confidence_threshold: Optional[float] = None
    ) -> OrchestrationResult:
        """
        Execute the end-to-end multi-agent resolution workflow.

        :param query: Raw user query
        :param top_k: Optional Top-K retrieval override
        :param confidence_threshold: Optional similarity score threshold override
        :return: Structured OrchestrationResult
        """
        start_time = time.perf_counter()

        try:
            # Step 1: Query Understanding & Intent Classification
            query_analysis = self.query_understanding_agent.analyze(query)

            # Step 2: Route Decision
            if query_analysis.route_to == RoutingTarget.CLARIFICATION:
                clarification_response = self.clarification_agent.handle(query_analysis)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return OrchestrationResult(
                    query=query,
                    response=clarification_response,
                    retrieval_result=None,
                    execution_time_ms=round(elapsed_ms, 2),
                    status="clarification_requested"
                )

            # Step 3: Retrieval Agent (Vector Search & Relevance Filtering)
            retrieval_result = self.retrieval_agent.retrieve(
                query_input=query_analysis,
                top_k=top_k,
                threshold=confidence_threshold
            )

            # Step 4: Response Generation Agent (Grounded Synthesis & Attribution)
            agent_response = self.response_generation_agent.generate(
                query_analysis=query_analysis,
                retrieval_result=retrieval_result
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return OrchestrationResult(
                query=query,
                response=agent_response,
                retrieval_result=retrieval_result,
                execution_time_ms=round(elapsed_ms, 2),
                status="success"
            )

        except Exception as exc:
            logger.error(f"Error during multi-agent orchestration: {exc}", exc_info=True)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            fallback_qa = QueryUnderstandingAgent().analyze(query)
            fallback_response = AgentResponse(
                answer="An unexpected error occurred while processing your query. Please check your system configuration or try again.",
                confidence_score=0.0,
                confidence_level=ConfidenceLevel.NONE,
                sources=[],
                query_analysis=fallback_qa,
                has_sufficient_evidence=False,
                requires_clarification=False
            )
            return OrchestrationResult(
                query=query,
                response=fallback_response,
                retrieval_result=None,
                execution_time_ms=round(elapsed_ms, 2),
                status="error",
                error=str(exc)
            )
